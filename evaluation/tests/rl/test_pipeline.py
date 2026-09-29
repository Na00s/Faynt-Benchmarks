"""The golden guard (P8-PLAN.md 2, 3.6): pipelined trajectories are **bit-identical** to lockstep.

The pipeline reorders computation, not data -- with FIFO action consumption the sample made at
state ``s_u`` executes into ``s_{u + D + 1}`` for any runahead, exactly like the lockstep delay
queue, and the generator draws keep their per-frame order.  So for every legal ``(b, k, D)`` the
two workers, run from identical seeds on twin dummy environments, must produce equal trajectories
field by field, equal generator states and equal ``frames_seen`` -- an exact comparison, not a
statistical one.  Games restart inside the windows (``game_frames = 21``), so mid-rollout resets
flow through the queues too.
"""

from __future__ import annotations

import pytest
import torch

from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.env.async_env import AsyncEnvAdapter
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.frames import tree_leaves
from melee_rl.opponents import CpuOpponent, SelfOpponent
from melee_rl.pipeline import PipelinedRolloutWorker
from melee_rl.trajectory import Trajectory

B = 3
T = 8
C = 16


def _env(*, two_port: bool = False, game_frames: int = 21) -> DummyMeleeEnv:
    players = ("policy", "policy") if two_port else ("policy", "cpu")
    return DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=5, players=players, game_frames=game_frames))


def _config(*, delay: int, batch_steps: int = 1, mode: str = "ring") -> ActorConfig:
    return ActorConfig(
        rollout_length=T,
        context_frames=C,
        delay_frames=delay,
        context_mode=mode,
        batch_steps=batch_steps,
        seed=7,
    )


def _workers(
    adapter: MeleePolicyAdapter,
    *,
    delay: int,
    batch_steps: int,
    chunk_frames: int,
    two_port: bool = False,
    mode: str = "ring",
    game_frames: int = 21,
) -> tuple[RolloutWorker, PipelinedRolloutWorker]:
    ports = (0, 1) if two_port else (0,)
    opponent = None if two_port else CpuOpponent()
    lockstep = RolloutWorker(
        adapter,
        _env(two_port=two_port, game_frames=game_frames),
        opponent,
        _config(delay=delay, mode=mode),
        ports=ports,
    )
    pipelined = PipelinedRolloutWorker(
        adapter,
        AsyncEnvAdapter(_env(two_port=two_port, game_frames=game_frames), chunk_frames),
        None if two_port else CpuOpponent(),
        _config(delay=delay, batch_steps=batch_steps, mode=mode),
        ports=ports,
    )
    return lockstep, pipelined


def _assert_trajectories_identical(a: Trajectory, b: Trajectory) -> None:
    for (path, leaf), (other_path, other) in zip(tree_leaves(a.frames), tree_leaves(b.frames), strict=True):
        assert path == other_path
        assert torch.equal(leaf, other), path
    for name in ("controller_exec", "controller_prev_sample", "sampled_labels"):
        ours, theirs = getattr(a, name), getattr(b, name)
        assert torch.equal(ours.buttons, theirs.buttons), name
        assert torch.equal(ours.main_stick, theirs.main_stick), name
    for name in ("buttons", "main_stick"):
        assert torch.equal(a.sampled_logits[name], b.sampled_logits[name]), name
    for name in ("rewards", "env_rewards", "is_resetting", "padding", "frame_index"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert (a.context_frames, a.rollout_length, a.delay_frames, a.context_mode) == (
        b.context_frames,
        b.rollout_length,
        b.delay_frames,
        b.context_mode,
    )
    if a.initial_cache is None or b.initial_cache is None:
        assert a.initial_cache is None and b.initial_cache is None
    else:
        for name in ("keys", "values", "valid_length", "write_position", "next_position"):
            assert torch.equal(getattr(a.initial_cache, name), getattr(b.initial_cache, name)), name


@pytest.mark.parametrize(
    "batch_steps,chunk_frames,delay",
    [(1, 1, 0), (1, 1, 5), (2, 2, 5), (4, 3, 5), (2, 4, 5)],
    ids=["b1k1d0", "b1k1d5", "b2k2d5", "b4k3d5", "b2k4d5"],
)
def test_pipelined_rollouts_are_bit_identical_to_lockstep(
    tiny_adapter: MeleePolicyAdapter, batch_steps: int, chunk_frames: int, delay: int
) -> None:
    lockstep, pipelined = _workers(
        tiny_adapter, delay=delay, batch_steps=batch_steps, chunk_frames=chunk_frames
    )
    assert pipelined.batch_steps == batch_steps
    assert pipelined.runahead == delay - (batch_steps - 1)
    for _ in range(3):
        a, b = lockstep.rollout(), pipelined.rollout()
        a.validate()
        b.validate()
        _assert_trajectories_identical(a, b)
        # The boundary invariants: exec deque 1 + r, delay queue b - 1, env in_flight 1 + r.
        exec_len, queue_len, in_flight = pipelined._boundary_invariants()
        assert exec_len == 1 + pipelined.runahead
        assert queue_len == batch_steps - 1
        assert in_flight == 1 + pipelined.runahead
    assert torch.equal(lockstep.generator.get_state(), pipelined.generator.get_state())
    assert lockstep.frames_seen == pipelined.frames_seen == 3 * T * B
    assert lockstep.rollouts_done == pipelined.rollouts_done == 3
    assert set(pipelined.timings) == {"rollout_s", "reprime_s", "policy_s", "env_s", "opponent_s"}


@pytest.mark.parametrize(
    "batch_steps,chunk_frames,delay",
    [(1, 1, 0), (1, 1, 5), (2, 2, 5)],
    ids=["b1k1d0", "b1k1d5", "b2k2d5"],
)
def test_prefix_mode_pipeline_is_bit_identical_to_lockstep(
    tiny_adapter: MeleePolicyAdapter, batch_steps: int, chunk_frames: int, delay: int
) -> None:
    """Prefix mode through the pipeline: every flush lands inside the rollout, so the snapshot taken at the
    boundary equals the lockstep worker's bit for bit -- with ``b > 1`` as well."""

    lockstep, pipelined = _workers(
        tiny_adapter, delay=delay, batch_steps=batch_steps, chunk_frames=chunk_frames, mode="prefix"
    )
    for rollout in range(3):
        a, b = lockstep.rollout(), pipelined.rollout()
        a.validate()
        b.validate()
        assert a.initial_cache is not None and a.context_mode == "prefix"
        assert bool((a.initial_cache.valid_length > 0).any()) == (rollout > 0)
        _assert_trajectories_identical(a, b)
    assert pipelined.timings["reprime_s"] == 0.0


@pytest.mark.parametrize("batch_steps,chunk_frames", [(4, 3), (2, 4)], ids=["b4k3", "b2k4"])
def test_two_port_pipelined_parity(
    tiny_adapter: MeleePolicyAdapter, batch_steps: int, chunk_frames: int
) -> None:
    lockstep, pipelined = _workers(
        tiny_adapter, delay=5, batch_steps=batch_steps, chunk_frames=chunk_frames, two_port=True
    )
    assert lockstep.batch_size == pipelined.batch_size == 2 * B
    for _ in range(3):
        a, b = lockstep.rollout(), pipelined.rollout()
        _assert_trajectories_identical(a, b)
    assert torch.equal(lockstep.generator.get_state(), pipelined.generator.get_state())


def test_reprime_and_fast_step_parity(tiny_adapter: MeleePolicyAdapter) -> None:
    lockstep, pipelined = _workers(tiny_adapter, delay=5, batch_steps=4, chunk_frames=2, mode="reprime")
    for _ in range(2):
        _assert_trajectories_identical(lockstep.rollout(), pipelined.rollout())
    assert pipelined.timings["reprime_s"] > 0.0
    tiny_adapter.fast_step = True
    try:
        fast_lock, fast_pipe = _workers(tiny_adapter, delay=5, batch_steps=2, chunk_frames=2)
        for _ in range(2):
            _assert_trajectories_identical(fast_lock.rollout(), fast_pipe.rollout())
    finally:
        tiny_adapter.fast_step = False


def test_reset_env_mid_sequence_keeps_parity(tiny_adapter: MeleePolicyAdapter) -> None:
    lockstep, pipelined = _workers(tiny_adapter, delay=5, batch_steps=2, chunk_frames=3)
    _assert_trajectories_identical(lockstep.rollout(), pipelined.rollout())
    lockstep.reset_env()
    pipelined.reset_env()
    exec_len, queue_len, in_flight = pipelined._boundary_invariants()
    assert exec_len == 1 + pipelined.runahead and queue_len == 1 and in_flight == 1 + pipelined.runahead
    a, b = lockstep.rollout(), pipelined.rollout()
    _assert_trajectories_identical(a, b)
    assert bool(a.is_resetting[:, C].all())  # the first post-reset rollout starts at a game start
    _assert_trajectories_identical(lockstep.rollout(), pipelined.rollout())
    assert torch.equal(lockstep.generator.get_state(), pipelined.generator.get_state())


def test_pipeline_construction_errors(tiny_adapter: MeleePolicyAdapter) -> None:
    good = _config(delay=5, batch_steps=2)
    with pytest.raises(TypeError, match="AsyncEnvProtocol"):
        PipelinedRolloutWorker(tiny_adapter, _env(), CpuOpponent(), good)
    with pytest.raises(ValueError, match="slack"):
        PipelinedRolloutWorker(tiny_adapter, AsyncEnvAdapter(_env(), chunk_frames=6), CpuOpponent(), good)
    # (b - 1) + (k - 1) = 5 <= D = 5 uses the full slack and is legal.
    PipelinedRolloutWorker(tiny_adapter, AsyncEnvAdapter(_env(), chunk_frames=5), CpuOpponent(), good)
    with pytest.raises(ValueError, match="batch_steps 3 must divide"):
        _config(delay=5, batch_steps=3)
    with pytest.raises(ValueError, match="port-controlling"):
        PipelinedRolloutWorker(
            tiny_adapter,
            AsyncEnvAdapter(_env(two_port=True), chunk_frames=2),
            SelfOpponent(tiny_adapter, B),
            good,
        )
    # b = k = 1 at D = 0 is legal (the degenerate pipeline); b = 2 at D = 0 is not.
    PipelinedRolloutWorker(tiny_adapter, AsyncEnvAdapter(_env()), CpuOpponent(), _config(delay=0))
    with pytest.raises(ValueError, match="slack"):
        PipelinedRolloutWorker(
            tiny_adapter, AsyncEnvAdapter(_env()), CpuOpponent(), _config(delay=0, batch_steps=2)
        )
