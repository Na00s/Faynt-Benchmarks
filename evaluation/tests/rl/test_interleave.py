"""Lever 1 (5 Sep 2026, ``/rl-dolphin-speedup``): the mix arms' rollouts interleaved frame by frame.

A ``RolloutWorker`` now exposes its rollout as a generator that yields once per frame right after the
frame's actions were sent; ``WorkerDolphinEnv`` can split a step into ``begin_step`` / ``end_step`` so its
Dolphins emulate across the yield while the other arm's policy runs.  Everything is exact per environment:
the trajectories, generator states and training metrics are bit-identical to the sequential rollouts.
"""

from __future__ import annotations

import pytest
import torch

from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.config import load_config
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.opponents import CpuOpponent
from melee_rl.train import RunOptions, Trainer
from melee_rl.trajectory import Trajectory
from tests.rl.test_mix import SMOKE_TINY, _comparable, _mixed, _same_state

B = 3
C = 16
T = 8


def _worker(adapter: MeleePolicyAdapter, seed: int, env_seed: int) -> RolloutWorker:
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=env_seed, game_frames=21))
    config = ActorConfig(rollout_length=T, context_frames=C, seed=seed)
    return RolloutWorker(adapter, env, CpuOpponent(), config)


def _assert_same_trajectory(a: Trajectory, b: Trajectory) -> None:
    for name in ("controller_exec", "controller_prev_sample", "sampled_labels"):
        assert torch.equal(getattr(a, name).buttons, getattr(b, name).buttons), name
        assert torch.equal(getattr(a, name).main_stick, getattr(b, name).main_stick), name
    for name in ("rewards", "env_rewards", "is_resetting", "padding", "frame_index"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for component in a.sampled_logits:
        assert torch.equal(a.sampled_logits[component], b.sampled_logits[component]), component


def test_rollout_steps_generator_is_the_rollout(tiny_adapter: MeleePolicyAdapter) -> None:
    """Driving the generator by hand is ``rollout()``; the trajectory waits in ``take_trajectory``."""

    reference = _worker(tiny_adapter, seed=7, env_seed=5)
    expected = [reference.rollout() for _ in range(2)]
    worker = _worker(tiny_adapter, seed=7, env_seed=5)
    with pytest.raises(RuntimeError, match="no finished rollout"):
        worker.take_trajectory()
    got = []
    for _ in range(2):
        with worker.inference_mode():
            frames = sum(1 for _ in worker.rollout_steps())
        assert frames == T  # one yield per frame
        got.append(worker.take_trajectory())
    for a, b in zip(got, expected, strict=True):
        _assert_same_trajectory(a, b)
    assert worker.generator.get_state().equal(reference.generator.get_state())
    assert worker.frames_seen == reference.frames_seen and worker.rollouts_done == reference.rollouts_done


def test_two_workers_interleaved_match_sequential_rollouts(tiny_adapter: MeleePolicyAdapter) -> None:
    """Two arms stepped frame by frame in alternation produce what each would produce alone."""

    sequential = [_worker(tiny_adapter, seed=7, env_seed=5), _worker(tiny_adapter, seed=8, env_seed=6)]
    expected = [[worker.rollout() for worker in sequential] for _ in range(3)]
    workers = [_worker(tiny_adapter, seed=7, env_seed=5), _worker(tiny_adapter, seed=8, env_seed=6)]
    from melee_rl.train import interleaved_rollouts

    got = [interleaved_rollouts(workers) for _ in range(3)]
    for round_got, round_expected in zip(got, expected, strict=True):
        for a, b in zip(round_got, round_expected, strict=True):
            _assert_same_trajectory(a, b)
    for worker, reference in zip(workers, sequential, strict=True):
        assert worker.generator.get_state().equal(reference.generator.get_state())
        assert worker.frames_seen == reference.frames_seen


def test_interleaved_mix_training_is_bit_identical(tmp_path: object) -> None:
    """``[actor] interleave_arms = true`` changes the schedule, not one number of the training run."""

    from dataclasses import replace

    base = load_config(SMOKE_TINY)
    plain = _mixed(base, ("self*1", "cpu9*2"))
    interleaved = replace(plain, actor=replace(plain.actor, interleave_arms=True))
    assert interleaved.actor.interleave_arms and not plain.actor.interleave_arms
    first = Trainer(plain, RunOptions(wandb_mode="disabled"))
    a = first.run(2)
    first.close()
    second = Trainer(interleaved, RunOptions(wandb_mode="disabled"))
    assert second.interleaved
    b = second.run(2)
    second.close()
    for left, right in zip(a.history, b.history, strict=True):
        assert _comparable(left) == _comparable(right)
    assert _same_state(first.policy.get_state(), second.policy.get_state())


def test_interleave_flag_validation() -> None:
    assert ActorConfig().interleave_arms is False
    base = load_config(SMOKE_TINY)
    from dataclasses import replace

    from melee_rl.config import finalize_config, resolve_model_config

    pipelined = replace(
        base,
        actor=replace(base.actor, interleave_arms=True, batch_steps=2, delay_frames=2),
        delay=replace(base.delay, allow_mismatch=True),
    )
    with pytest.raises(ValueError, match="interleave_arms"):
        finalize_config(pipelined, resolve_model_config(pipelined))
