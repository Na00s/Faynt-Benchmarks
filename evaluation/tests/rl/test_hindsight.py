"""The hindsight teacher: ``[teacher] delay_frames`` (11 Sep 2026, ``/delay-finetune``).

A student that decides at window row ``k`` under a delay ``D`` executes at ``k + D + 1``.  A teacher trained
at its own delay ``D_T`` (``teacher.delay_frames``; the run's ``D`` when unset) reads row ``j = k + D - D_T``
with its own pairing -- input ``(frame j, controller_exec[j + D_T])``, teacher-forced on
``controller_exec[j + D_T + 1] == sampled_labels[k]`` -- so its logits are about the very controller the
student's decision produces.  ``D_T = 0`` is the hindsight teacher of DIDA (Liotet et al., ICML 2022): the
frozen delay-0 expert read at the frame where the decision lands; ``D_T = D`` is today's path, bit for bit;
``0 < D_T < D`` the curriculum fallback.  A decision whose teacher row lies in another game (a reset in
``(k, j]``) drops out of the teacher terms.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from controller_codec import ControllerLabels
from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import load_checkpoint, sha256_file, trained_delay
from melee_rl.config import CONFIG_DIR, load_config
from melee_rl.distributions import kl
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.frames import labels_index, neutral_frame, tree_index, tree_leaves
from melee_rl.learner import LearnerConfig, PPOConfig, PPOLearner
from melee_rl.opponents import CpuOpponent
from melee_rl.protocol import PolicyProtocol
from melee_rl.train import RunOptions, Trainer, main
from melee_rl.trajectory import Trajectory, Window
from melee_rl.value import ValueConfig, ValueNet
from model import ModelConfig

RolloutFactory = Callable[..., list[Trajectory]]
SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
B = 2
C = 3
T = 8
HINDSIGHT_METRICS = ("hindsight/top1", "hindsight/teacher_entropy", "hindsight/masked_fraction")


def _hand_built(delay: int, *, resets: Sequence[int] = (0,), context: int = C, length: int = T) -> Trajectory:
    """Window-row-coded labels: the sample made at row ``w`` is ``100 + w`` / ``w % 85``.

    ``controller_exec[w]`` is the sample of ``w - 1 - D`` (neutral before the stream began; the queue is not
    cleared at in-game restarts), ``controller_prev_sample[w]`` the sample of ``w - 1`` masked at the reset
    rows, and the bootstrap row's sample is the neutral placeholder -- the actor's invariants.
    """

    window = context + length + 1
    rows = torch.arange(window)
    is_resetting = torch.zeros((B, window), dtype=torch.bool)
    for row in resets:
        is_resetting[:, row] = True
    reset_rows = is_resetting[0]
    sampled_b = torch.where(rows < context + length, 100 + rows, 0)
    sampled_m = torch.where(rows < context + length, rows % 85, 0)
    exec_rows = rows - 1 - delay
    exec_b = torch.where(exec_rows >= 0, 100 + exec_rows, 0)
    exec_m = torch.where(exec_rows >= 0, exec_rows % 85, 0)
    prev_rows = rows - 1
    prev_b = torch.where((prev_rows >= 0) & ~reset_rows, 100 + prev_rows, 0)
    prev_m = torch.where((prev_rows >= 0) & ~reset_rows, prev_rows % 85, 0)

    def batched(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(0).expand(B, window).clone()

    generator = torch.Generator().manual_seed(0)
    return Trajectory(
        frames=neutral_frame((B, window)),
        controller_exec=ControllerLabels(batched(exec_b), batched(exec_m)),
        controller_prev_sample=ControllerLabels(batched(prev_b), batched(prev_m)),
        sampled_labels=ControllerLabels(batched(sampled_b), batched(sampled_m)),
        sampled_logits={
            "buttons": torch.randn((B, length, 728), generator=generator),
            "main_stick": torch.randn((B, length, 85), generator=generator),
        },
        rewards=torch.zeros((B, length)),
        env_rewards=torch.zeros((B, length)),
        is_resetting=is_resetting,
        padding=torch.ones((B, window), dtype=torch.bool),
        frame_index=torch.arange(window).unsqueeze(0).expand(B, window).clone() - 123,
        context_frames=context,
        rollout_length=length,
        delay_frames=delay,
        context_mode="reprime",
    )


def _same_window(left: Window, right: Window) -> bool:
    if left.length != right.length or left.targets is None or right.targets is None:
        return False
    for name in ("buttons", "main_stick"):
        if not torch.equal(getattr(left.controller_prev, name), getattr(right.controller_prev, name)):
            return False
        if not torch.equal(getattr(left.targets, name), getattr(right.targets, name)):
            return False
    if not torch.equal(left.reset_mask, right.reset_mask) or not torch.equal(
        left.padding_mask, right.padding_mask
    ):
        return False
    for (path_a, leaf_a), (path_b, leaf_b) in zip(
        tree_leaves(left.frames), tree_leaves(right.frames), strict=True
    ):
        if path_a != path_b or not torch.equal(leaf_a, leaf_b):
            return False
    return True


def _learner(
    adapter: MeleePolicyAdapter,
    config: ModelConfig,
    value_config: ValueConfig,
    teacher: PolicyProtocol | None,
    *,
    teacher_delay: int | None = None,
    lr: float = 1.0e-3,
    epochs: int = 2,
    microbatch: int | None = None,
    pg: float = 1.0,
    kl_weight: float = 0.1,
    reverse: float = 0.0,
    beta: float = 0.0,
    guard: float = 1.0,
    teacher_prefix: bool = False,
    fast_forward: bool = False,
    compile: bool = False,
    seed: int = 0,
) -> PPOLearner:
    torch.manual_seed(seed)
    value_net = ValueNet(config, value_config).to(adapter.device)
    learner_config = LearnerConfig(
        learning_rate=lr,
        policy_gradient_weight=pg,
        kl_teacher_weight=kl_weight,
        reverse_kl_teacher_weight=reverse,
        microbatch_envs=microbatch,
        teacher_prefix=teacher_prefix,
        fast_forward=fast_forward,
        compile=compile,
        compile_backend="aot_eager",
        ppo=PPOConfig(num_epochs=epochs, beta=beta, max_mean_actor_kl=guard),
    )
    return PPOLearner(adapter, teacher, value_net, learner_config, teacher_delay_frames=teacher_delay)


def _fresh(adapter: MeleePolicyAdapter, config: ModelConfig) -> MeleePolicyAdapter:
    """A new adapter with ``adapter``'s weights (``deepcopy`` cannot copy Ali's codec)."""

    policy = MeleePolicyAdapter(config)
    policy.load_state_dict(adapter.state_dict(), strict=True)
    return policy


def _perturbed(
    adapter: MeleePolicyAdapter, config: ModelConfig, *, scale: float = 0.05, seed: int = 11
) -> MeleePolicyAdapter:
    """A frozen teacher with different weights: the adapter's plus Gaussian noise."""

    teacher = _fresh(adapter, config)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=generator))
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def _params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


# ---------------------------------------------------------------------------
# the trajectory: the teacher's window, its decision rows, the mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 2])
def test_teacher_window_at_the_run_delay_is_the_policy_window(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory, delay: int
) -> None:
    """``D_T = D`` is today's teacher input bit for bit: the same rows, the same pairing, nothing masked --
    on a hand-built trajectory with a reset inside the rollout, and on real rollouts (the first with a padded
    history)."""

    hand = _hand_built(delay, resets=(0, C + 2))
    hand.validate()
    assert hand.teacher_rows(delay) == hand.policy_rows
    assert hand.teacher_decision_rows(delay) == hand.decision_rows
    assert _same_window(hand.teacher_window(delay), hand.policy_window())
    keep = hand.teacher_keep(delay)
    assert keep.dtype == torch.bool and keep.shape == (B, T - delay) and bool(keep.all())
    for trajectory in rollout_trajectories(tiny_adapter, 3, delay=delay):
        assert _same_window(trajectory.teacher_window(delay), trajectory.policy_window())
        assert bool(trajectory.teacher_keep(delay).all())


@pytest.mark.parametrize("teacher_delay", [0, 1])
def test_hindsight_window_reads_where_the_decision_lands(teacher_delay: int) -> None:
    """At ``D_T < D`` the teacher reads row ``j = k + D - D_T`` with its own pairing, and its teacher-forcing
    target at every decision row is the student's own sampled label (the controller that executes at
    ``k + D + 1``)."""

    delay = 2
    trajectory = _hand_built(delay)
    shift = delay - teacher_delay
    rows = trajectory.teacher_rows(teacher_delay)
    decisions = trajectory.teacher_decision_rows(teacher_delay)
    assert rows == slice(0, C + T - teacher_delay)
    assert decisions == slice(C + shift, C + T - teacher_delay)
    assert decisions.stop - decisions.start == T - delay  # one teacher row per trained decision
    window = trajectory.teacher_window(teacher_delay)
    assert window.length == C + T - teacher_delay and window.targets is not None
    for j in range(window.length):
        # The input controller is the one a D_T-delayed policy at row j has just decided: executed on the
        # transition into row j + D_T, i.e. the sample of row j + D_T - 1 - D; neutral at the reset row.
        source = j + teacher_delay - 1 - delay
        expected_prev = 0 if j == 0 or source < 0 else 100 + source
        assert int(window.controller_prev.buttons[0, j]) == expected_prev, j
        landing = j + teacher_delay - delay  # the sample that executes at j + D_T + 1
        assert int(window.targets.buttons[0, j]) == (100 + landing if landing >= 0 else 0), j
    student = trajectory.actor_labels()
    teacher_targets = labels_index(window.targets, (slice(None), decisions))
    assert torch.equal(teacher_targets.buttons, student.buttons)
    assert torch.equal(teacher_targets.main_stick, student.main_stick)
    assert torch.equal(window.reset_mask, trajectory.is_resetting[:, rows])
    assert bool(window.padding_mask.all()) and window.frames["p1"]["percent"].shape == (B, window.length)
    for bad in (delay + 1, -1):
        with pytest.raises(ValueError, match="teacher"):
            trajectory.teacher_window(bad)
        with pytest.raises(ValueError, match="teacher"):
            trajectory.teacher_keep(bad)


def test_teacher_keep_masks_decisions_whose_teacher_row_is_in_another_game() -> None:
    """A reset in ``(k, k + D - D_T]`` puts the teacher's frame in the next game: the decision stays in the
    PPO loss but leaves the teacher terms.  At ``D_T = D`` nothing is ever masked."""

    delay = 3
    reset_row = C + 3  # rollout step 3 starts a new game
    trajectory = _hand_built(delay, resets=(0, reset_row))
    trajectory.validate()
    expectations = {
        0: [False, False, False, True, True],
        1: [True, False, False, True, True],
        3: [True, True, True, True, True],
    }
    for teacher_delay, expected in expectations.items():
        keep = trajectory.teacher_keep(teacher_delay)
        assert keep.dtype == torch.bool and keep.shape == (B, T - delay)
        assert keep[0].tolist() == expected and keep[1].tolist() == expected, teacher_delay
    # A reset at the decision row itself is the student's own new game: nothing to mask.
    assert bool(_hand_built(delay, resets=(0, C)).teacher_keep(0).all())


def test_teacher_keep_on_real_rollouts_matches_a_loop(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory
) -> None:
    delay = 2
    masked_somewhere = False
    for trajectory in rollout_trajectories(tiny_adapter, 3, delay=delay, game_frames=9):
        keep = trajectory.teacher_keep(0)
        context = trajectory.context_frames
        for env in range(trajectory.batch_size):
            for t in range(T - delay):
                k = context + t
                resets = trajectory.is_resetting[env, k + 1 : k + delay + 1]
                assert bool(keep[env, t]) == (not bool(resets.any())), (env, t)
        masked_somewhere = masked_somewhere or not bool(keep.all())
    assert masked_somewhere  # nine-frame games: some decisions land in the next game


# ---------------------------------------------------------------------------
# the learner: today's path unchanged, the hindsight path aligned, masked and reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 2])
def test_an_explicit_teacher_delay_equal_to_the_run_delay_changes_nothing(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
    delay: int,
) -> None:
    rollouts = rollout_trajectories(tiny_adapter, 2, delay=delay, synthetic_rewards=0.1)
    results = []
    for teacher_delay in (None, delay):
        policy = _fresh(tiny_adapter, tiny_config)
        learner = _learner(
            policy, tiny_config, tiny_value_config, policy.clone_frozen(), teacher_delay=teacher_delay
        )
        assert learner.teacher_delay_frames == teacher_delay
        results.append((learner.step(rollouts), _params(policy)))
    (implicit, implicit_params), (explicit, explicit_params) = results
    assert implicit.keys() == explicit.keys()
    for key, value in implicit.items():
        if not key.startswith("timing/"):
            assert value == explicit[key], key
    assert all(torch.equal(a, b) for a, b in zip(implicit_params, explicit_params, strict=True))
    # The teacher is the policy: before the update every decision agrees with it, nothing is masked and the
    # teacher's entropy is the policy's.
    assert implicit["pre/top1"] == 1.0 and implicit["hindsight/masked_fraction"] == 0.0
    assert implicit["hindsight/teacher_entropy"] == pytest.approx(implicit["pre/entropy"], abs=1.0e-6)
    assert 0.0 <= implicit["hindsight/top1"] <= 1.0


def test_hindsight_teacher_terms_are_aligned_masked_and_reported(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """``D_T = 0`` at ``D = 2``: the teacher's decision logits are a plain window forward over the undelayed
    pairing ``(frame, controller_exec) -> controller_exec[+1]`` read ``D`` rows after each decision; the
    teacher terms average over the kept decisions only; the new metrics are finite and consistent."""

    delay = 2
    rollouts = rollout_trajectories(tiny_adapter, 3, delay=delay, game_frames=9, synthetic_rewards=0.1)
    teacher = _perturbed(tiny_adapter, tiny_config)
    policy = _fresh(tiny_adapter, tiny_config)
    learner = _learner(policy, tiny_config, tiny_value_config, teacher, teacher_delay=0, lr=0.0, epochs=1)
    metrics = learner.step(rollouts)
    for key in (*HINDSIGHT_METRICS, "pre/top1", "loss/teacher_kl", "loss/reverse_teacher_kl"):
        assert math.isfinite(metrics[key]), key
    assert 0.0 <= metrics["hindsight/top1"] <= 1.0 and 0.0 <= metrics["pre/top1"] <= 1.0
    assert metrics["hindsight/teacher_entropy"] > 0.0 and metrics["loss/teacher_kl"] > 0.0
    keeps = [trajectory.teacher_keep(0) for trajectory in rollouts]
    masked = 1.0 - sum(float(keep.sum()) for keep in keeps) / sum(keep.numel() for keep in keeps)
    assert 0.0 < masked < 1.0 and metrics["hindsight/masked_fraction"] == pytest.approx(masked, abs=1.0e-6)
    # The alignment, against an independent construction on a rollout with a full history.
    sub = rollouts[2]
    context, length = sub.context_frames, sub.rollout_length
    rows = slice(0, context + length)
    resets = sub.is_resetting[:, rows]
    prev = labels_index(sub.controller_exec, (slice(None), rows))
    prev = ControllerLabels(torch.where(resets, 0, prev.buttons), torch.where(resets, 0, prev.main_stick))
    targets = labels_index(sub.controller_exec, (slice(None), slice(1, context + length + 1)))
    assert bool(sub.padding.all())
    with torch.no_grad():
        output = teacher.unroll_window(
            tree_index(sub.frames, (slice(None), rows)), prev, targets, resets, sub.padding[:, rows]
        )
        got = learner._teacher_logits(sub)
    for name, value in output.logits.items():
        assert torch.equal(got[name], value[:, context + delay :]), name
    # The reported KLs are masked means over every trajectory of the step (lr 0: the post pass sees the same
    # policy the pre pass saw).
    forward_sum = reverse_sum = kept = 0.0
    with torch.no_grad():
        for trajectory in rollouts:
            teacher_logits = learner._teacher_logits(trajectory)
            policy_logits = learner._policy_logits(trajectory)
            keep = trajectory.teacher_keep(0).float()
            forward_sum += float((kl(policy_logits, teacher_logits) * keep).sum())
            reverse_sum += float((kl(teacher_logits, policy_logits) * keep).sum())
            kept += float(keep.sum())
    assert metrics["loss/teacher_kl"] == pytest.approx(forward_sum / kept, rel=1.0e-5)
    assert metrics["loss/reverse_teacher_kl"] == pytest.approx(reverse_sum / kept, rel=1.0e-5)
    assert metrics["pre/teacher_kl"] == pytest.approx(metrics["loss/teacher_kl"], rel=1.0e-6)


def test_distillation_moves_the_student_toward_the_hindsight_teacher(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """Stage 1's loss -- PG weight 0, beta 0, KL(T || pi) alone, the guard out of the way -- is supervised
    learning on the rollouts: on a fixed batch the reverse KL to a different teacher falls step after step."""

    delay = 2
    rollouts = rollout_trajectories(tiny_adapter, 2, delay=delay, synthetic_rewards=0.1)
    teacher = _perturbed(tiny_adapter, tiny_config, scale=0.1)
    policy = _fresh(tiny_adapter, tiny_config)
    learner = _learner(
        policy,
        tiny_config,
        tiny_value_config,
        teacher,
        teacher_delay=0,
        lr=3.0e-3,
        epochs=2,
        pg=0.0,
        kl_weight=0.0,
        reverse=1.0,
        beta=0.0,
        guard=1.0e3,
    )
    history = [learner.step(rollouts) for _ in range(10)]
    first, last = history[0], history[-1]
    assert all(row["ppo/reverted"] == 0.0 for row in history)
    assert first["pre/reverse_teacher_kl"] > 0.0
    assert last["pre/reverse_teacher_kl"] < 0.6 * first["pre/reverse_teacher_kl"]
    assert last["hindsight/teacher_entropy"] == first["hindsight/teacher_entropy"]  # the teacher never moves
    assert last["weights/drift_total"] > 0.0


def test_the_production_combination_at_delay_18(
    tiny_adapter: MeleePolicyAdapter, tiny_config: ModelConfig, tiny_value_config: ValueConfig
) -> None:
    """Stage 1's learner shape at the real delay: prefix rollouts with packed frames at D = 18 (T = 24,
    C = 24, a 48-frame context), the lean compiled learner forward on the student, the hindsight teacher eager
    over the whole window, env microbatches -- finite, exact on the student's stored logits, and it trains."""

    import torch._dynamo

    delay, length, context = 18, 24, 24
    config = replace(tiny_config, context_length=context + length)
    policy = MeleePolicyAdapter(config)
    policy.load_state_dict(tiny_adapter.state_dict(), strict=True)
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=3, seed=5, game_frames=21))
    actor = ActorConfig(
        rollout_length=length,
        context_frames=context,
        delay_frames=delay,
        context_mode="prefix",
        packed_frames=True,
        seed=7,
    )
    worker = RolloutWorker(policy, env, CpuOpponent(), actor)
    rollouts = [worker.rollout() for _ in range(3)]
    for trajectory in rollouts:
        trajectory.validate()
        assert trajectory.delay_frames == delay and trajectory.initial_cache is not None
        assert trajectory.decision_rows == slice(context, context + length - delay)
        assert trajectory.teacher_rows(0) == slice(0, context + length)
        assert trajectory.teacher_decision_rows(0) == slice(context + delay, context + length)
    teacher = _perturbed(policy, config)
    torch._dynamo.reset()
    learner = _learner(
        policy,
        config,
        tiny_value_config,
        teacher,
        teacher_delay=0,
        lr=1.0e-3,
        epochs=2,
        microbatch=1,
        pg=0.0,
        kl_weight=0.0,
        reverse=1.0,
        beta=0.0,
        guard=1.0e3,
        fast_forward=True,
        compile=True,
    )
    before = _params(policy)
    metrics = learner.step(rollouts[:2])
    assert math.isfinite(metrics["loss/total"]) and metrics["ppo/reverted"] == 0.0
    assert metrics["loss/actor_kl_pre"] < 1.0e-4  # the compiled prefix path reproduces the actor
    assert 0.0 <= metrics["hindsight/masked_fraction"] < 1.0 and metrics["hindsight/teacher_entropy"] > 0.0
    assert 0.0 <= metrics["pre/top1"] <= 1.0
    assert not all(torch.equal(a, b) for a, b in zip(before, _params(policy), strict=True))
    second = learner.step(rollouts[1:])
    assert math.isfinite(second["loss/total"]) and second["ppo/reverted"] == 0.0


# ---------------------------------------------------------------------------
# the trainer: Stage 1's shape end to end on the toy, the teacher record, resume
# ---------------------------------------------------------------------------


def test_trainer_distils_at_delay_four_and_records_the_teacher_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 on the toy: a delay-0 checkpoint forked to D = 4 with the hindsight teacher (``source =
    "policy"`` at ``delay_frames = 0``), two interleaved self arms in prefix mode with packed frames and the
    lean learner forward, PG 0 / reverse KL 1, the guard off; the in-loop anchor plays the base at the run's
    delay (``@d4``); the checkpoint records both delays; a resume keeps them and a changed teacher delay is
    refused."""

    from melee_rl import evaluate as evaluate_module
    from melee_rl.opponents import OtherOpponent

    delays: list[int] = []

    class Recording(OtherOpponent):
        def __init__(self, path: str | Path, batch_size: int, **kwargs: Any) -> None:
            delays.append(int(kwargs.get("delay_frames", 0)))
            super().__init__(path, batch_size, **kwargs)

    monkeypatch.setattr(evaluate_module, "OtherOpponent", Recording)
    base = main(
        [
            "--config",
            str(SMOKE_TINY),
            "--checkpoint-dir",
            str(tmp_path / "base"),
            "--wandb-mode",
            "disabled",
            "--steps",
            "1",
        ]
    )
    assert base.checkpoint is not None and trained_delay(base.checkpoint) == 0
    overrides = [
        "policy.init=checkpoint",
        f"policy.checkpoint={base.checkpoint}",
        "policy.init_value=true",
        "teacher.source=policy",
        "teacher.delay_frames=0",
        "actor.delay_frames=4",
        "delay.allow_mismatch=true",
        "actor.context_mode=prefix",
        "actor.interleave_arms=true",
        "actor.packed_frames=true",
        "opponent.type=mix",
        'opponent.mix=["self*2@a", "self*1@b"]',
        'env.dummy.players=["policy", "policy"]',
        "learner.policy_gradient_weight=0.0",
        "learner.kl_teacher_weight=0.0",
        "learner.reverse_kl_teacher_weight=1.0",
        "learner.fast_forward=true",
        "learner.ppo.beta=0.0",
        "learner.ppo.max_mean_actor_kl=1000.0",
        "learner.ppo.num_batches=1",
        "eval.enabled=true",
        "eval.interval_steps=1",
        "eval.frames=48",
        "eval.num_envs=2",
        "eval.rollout_length=8",
        f'eval.opponents=["cpu9", "other:{base.checkpoint}@d4"]',
        "eval.metric_opponent=other",
    ]
    config = load_config(SMOKE_TINY, overrides=overrides)
    assert config.teacher.delay_frames == 0 and config.actor.delay_frames == 4
    run_dir = tmp_path / "distil"
    options = RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled")
    trainer = Trainer(config, options)
    assert trainer.learner.teacher_delay_frames == 0 and len(trainer.workers) == 2
    init = run_dir / "init.pt"
    assert trainer.teacher_record == {"source": str(init), "sha256": sha256_file(init), "delay_frames": 0}
    result = trainer.run(2)
    trainer.close()
    row = [item for item in result.history if "loss/total" in item][-1]  # the eval rows come after each step
    for key in (*HINDSIGHT_METRICS, "pre/top1"):
        assert key in row and math.isfinite(row[key]), key
    assert row["loss/actor_kl_pre"] < 1.0e-6 and row["ppo/reverted"] == 0.0
    assert delays and set(delays) == {4}  # the anchor plays the base at the run's delay, as the spec asks
    assert result.evals and "other" in result.evals[-1].stats
    latest = run_dir / "latest.pt"
    payload = load_checkpoint(latest)
    assert trained_delay(latest) == 4
    assert payload["teacher"]["delay_frames"] == 0 and payload["rl_config"]["teacher"]["delay_frames"] == 0
    resumed = Trainer(config, options)
    assert resumed.resumed and resumed.step == 2 and resumed.learner.teacher_delay_frames == 0
    resumed.close()
    changed = load_config(SMOKE_TINY, overrides=[*overrides, "teacher.delay_frames=2"])
    with pytest.raises(ValueError, match="teacher"):
        Trainer(changed, options)
