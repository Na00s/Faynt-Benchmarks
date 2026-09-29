"""The adaptive levers of 15 Sep 2026 (session 69; ``STRENGTH-LEVERS.md`` §2.2, §2.3, §2.5): the EMA
anchor (``[teacher] source = "ema"``), the target-entropy controller (``[learner] entropy_target``), the
KL-targeted learning rate (``[learner] actor_kl_target``) and the weight-0 reference KL
(``[teacher] reference_path``), plus their checkpoint state and the trainer's resume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import (
    checkpoint_paths,
    load_checkpoint,
    restore_learner,
    save_rl_checkpoint,
)
from melee_rl.config import (
    CONFIG_DIR,
    TEACHER_SOURCES,
    TeacherConfig,
    finalize_config,
    load_config,
    resolve_model_config,
)
from melee_rl.learner import LearnerConfig, PPOConfig, PPOLearner
from melee_rl.protocol import PolicyProtocol
from melee_rl.train import RunOptions, Trainer, build_reference, build_teacher
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueConfig, ValueNet
from model import ModelConfig

RolloutFactory = Callable[..., list[Trajectory]]
LEARN_TINY = CONFIG_DIR / "learn_tiny.toml"


def _params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _all_equal(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def _learner(
    adapter: MeleePolicyAdapter,
    policy_config: ModelConfig,
    value_config: ValueConfig,
    *,
    teacher: PolicyProtocol | None = None,
    reference: PolicyProtocol | None = None,
    ema_tau: float | None = None,
    lr: float = 1.0e-3,
    max_kl: float = 1.0,
    value_burnin: int = 0,
    optimizer_burnin: int = 0,
    seed: int = 0,
    fast_forward: bool = False,
    **learner_fields: object,
) -> PPOLearner:
    torch.manual_seed(seed)
    value_net = ValueNet(policy_config, value_config).to(adapter.device)
    config = LearnerConfig(
        learning_rate=lr,
        kl_teacher_weight=0.1 if teacher is not None else 0.0,
        value_burnin_steps=value_burnin,
        optimizer_burnin_steps=optimizer_burnin,
        fast_forward=fast_forward,
        ppo=PPOConfig(num_epochs=2, max_mean_actor_kl=max_kl),
        **learner_fields,  # type: ignore[arg-type]
    )
    return PPOLearner(adapter, teacher, value_net, config, teacher_ema_tau=ema_tau, reference=reference)


# -- config -------------------------------------------------------------------------------------


def test_teacher_and_learner_config_validation() -> None:
    assert "ema" in TEACHER_SOURCES
    ema = TeacherConfig(source="ema", ema_tau=0.01)
    assert ema.ema_tau == 0.01 and ema.reference_path is None
    with pytest.raises(ValueError, match="ema_tau"):
        TeacherConfig(source="ema")
    with pytest.raises(ValueError, match="ema_tau"):
        TeacherConfig(source="ema", ema_tau=0.0)
    with pytest.raises(ValueError, match="ema_tau"):
        TeacherConfig(source="ema", ema_tau=1.5)
    with pytest.raises(ValueError, match="ema_tau"):
        TeacherConfig(source="policy", ema_tau=0.1)
    with pytest.raises(ValueError, match="reference_path"):
        TeacherConfig(source="none", reference_path="ref.pt")
    assert TeacherConfig(source="checkpoint", path="t.pt", reference_path="ref.pt").reference_path == "ref.pt"

    assert LearnerConfig().entropy_target is None and LearnerConfig().actor_kl_target is None
    controller = LearnerConfig(entropy_target=0.45, entropy_gain=1.0e-3, entropy_weight_max=0.02)
    assert controller.entropy_target == 0.45
    for bad in ({"entropy_target": 0.0}, {"entropy_target": -1.0}, {"entropy_gain": -1.0}):
        with pytest.raises(ValueError, match="entropy"):
            LearnerConfig(**bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entropy_weight_max"):
        LearnerConfig(entropy_target=0.4, entropy_weight=0.05, entropy_weight_max=0.02)
    targeted = LearnerConfig(learning_rate=1.0e-4, actor_kl_target=8.0e-5)
    assert targeted.effective_lr_min == pytest.approx(2.5e-5)
    assert targeted.effective_lr_max == pytest.approx(4.0e-4)
    explicit = LearnerConfig(learning_rate=1.0e-4, actor_kl_target=8.0e-5, lr_min=3.0e-5, lr_max=5.0e-4)
    assert explicit.effective_lr_min == 3.0e-5 and explicit.effective_lr_max == 5.0e-4
    with pytest.raises(ValueError, match="actor_kl_target"):
        LearnerConfig(actor_kl_target=0.0)
    with pytest.raises(ValueError, match="lr_adapt_factor"):
        LearnerConfig(actor_kl_target=1.0e-4, lr_adapt_factor=1.0)
    with pytest.raises(ValueError, match="lr_min"):
        LearnerConfig(learning_rate=1.0e-4, actor_kl_target=1.0e-4, lr_min=2.0e-4)
    with pytest.raises(ValueError, match="lr_max"):
        LearnerConfig(learning_rate=1.0e-4, actor_kl_target=1.0e-4, lr_max=5.0e-5)


def test_finalize_config_rules_for_the_ema_teacher() -> None:
    config = load_config(LEARN_TINY)
    assert config.teacher.source == "none" and config.policy.init == "random"
    model_config = resolve_model_config(config)
    # A random-init policy may be leashed to its own EMA (the anchor tracks the policy, it is not noise).
    ema = replace(
        config,
        teacher=TeacherConfig(source="ema", ema_tau=0.5),
        learner=replace(config.learner, kl_teacher_weight=0.1),
    )
    assert finalize_config(ema, model_config).learner.kl_teacher_weight == 0.1
    # ... but not to a frozen random init.
    frozen = replace(ema, teacher=TeacherConfig(source="policy"))
    with pytest.raises(ValueError, match="random-init teacher"):
        finalize_config(frozen, model_config)
    # The EMA teacher plays at the run's delay: a hindsight EMA has no meaning.
    hindsight = replace(
        ema,
        teacher=TeacherConfig(source="ema", ema_tau=0.5, delay_frames=0),
        actor=replace(config.actor, delay_frames=2),
        delay=replace(config.delay, allow_mismatch=True),
    )
    with pytest.raises(ValueError, match=r"ema.*delay_frames"):
        finalize_config(hindsight, model_config)


# -- the learner ---------------------------------------------------------------------------------


def test_ema_teacher_tracks_accepted_steps_only(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    teacher = tiny_adapter.clone_frozen()
    learner = _learner(
        tiny_adapter, tiny_config, tiny_value_config, teacher=teacher, ema_tau=0.25, value_burnin=1
    )
    assert learner.teacher_ema_tau == 0.25
    init = _params(teacher)
    # The value burn-in does not move the policy, and the anchor does not move either.
    burnin = learner.step(trajectories)
    assert burnin["ppo/stage"] == 0.0 and _all_equal(_params(teacher), init)
    # One accepted PPO step: the anchor moves a quarter of the way, ``torch.lerp`` bit for bit.
    metrics = learner.step(trajectories)
    assert metrics["ppo/reverted"] == 0.0 and metrics["loss/actor_kl_mean"] > 0.0
    policy_after = _params(tiny_adapter)
    assert not _all_equal(policy_after, init)
    expected = [torch.lerp(a, b, 0.25) for a, b in zip(init, policy_after, strict=True)]
    assert _all_equal(_params(teacher), expected)
    assert metrics["teacher/ema_tau"] == 0.25
    # A reverted step leaves the anchor where it was.
    before = _params(teacher)
    learner.config = replace(learner.config, ppo=replace(learner.config.ppo, max_mean_actor_kl=1.0e-12))
    reverted = learner.step(trajectories)
    assert reverted["ppo/reverted"] == 1.0 and _all_equal(_params(teacher), before)
    assert _all_equal(_params(tiny_adapter), policy_after)


def test_ema_tau_one_makes_the_anchor_the_policy_and_the_teacher_kl_vanish(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    teacher = tiny_adapter.clone_frozen()
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, teacher=teacher, ema_tau=1.0)
    first = learner.step(trajectories)
    assert first["pre/teacher_kl"] < 1.0e-6 and first["loss/teacher_kl"] > 0.0
    assert _all_equal(_params(teacher), _params(tiny_adapter))
    # The next step starts at the anchor again: the pre-update teacher KL is back at the parity floor.
    second = learner.step(trajectories)
    assert second["pre/teacher_kl"] < 1.0e-6
    with pytest.raises(ValueError, match="teacher_ema_tau"):
        _learner(tiny_adapter, tiny_config, tiny_value_config, teacher=None, ema_tau=0.5)
    with pytest.raises(ValueError, match="teacher_ema_tau"):
        _learner(tiny_adapter, tiny_config, tiny_value_config, teacher=teacher, ema_tau=0.0)


def test_entropy_controller_moves_the_weight_toward_its_target(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    # Below the target the weight rises by gain * (target - H) per accepted step and clamps at the maximum.
    learner = _learner(
        tiny_adapter,
        tiny_config,
        tiny_value_config,
        entropy_target=100.0,
        entropy_gain=1.0e-3,
        entropy_weight_max=0.15,
    )
    assert learner.entropy_weight == 0.0
    first = learner.step(trajectories)
    assert first["opt/entropy_weight"] == 0.0  # the weight the step was taken with
    expected = min(1.0e-3 * (100.0 - first["loss/entropy"]), 0.15)
    assert learner.entropy_weight == pytest.approx(expected)
    second = learner.step(trajectories)
    assert second["opt/entropy_weight"] == pytest.approx(expected)
    for _ in range(3):
        learner.step(trajectories)
    assert learner.entropy_weight == 0.15
    # Above the target the weight falls, never below zero.
    fixed = _learner(
        MeleePolicyAdapter(tiny_config), tiny_config, tiny_value_config, entropy_weight=0.05, seed=3
    )
    controlled = _learner(
        MeleePolicyAdapter(tiny_config),
        tiny_config,
        tiny_value_config,
        entropy_weight=0.05,
        entropy_target=1.0e-3,
        entropy_gain=1.0,
        entropy_weight_max=0.05,
        seed=3,
    )
    controlled.policy.set_state(fixed.policy.get_state())
    controlled.value_net.load_state_dict(fixed.value_net.state_dict())
    same = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1, seed=11)
    a, b = fixed.step(same), controlled.step(same)
    assert a["loss/total"] == b["loss/total"] and b["opt/entropy_weight"] == 0.05  # the weight is applied
    assert controlled.entropy_weight == 0.0  # 0.05 - 1.0 * (H - 0.001) clamps at zero


def test_kl_targeted_learning_rate_adapts_between_its_bounds(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    # Every real step's actor KL is far above 2 x 1e-12: the rate halves per step down to lr_min.
    learner = _learner(
        tiny_adapter,
        tiny_config,
        tiny_value_config,
        lr=1.0e-3,
        optimizer_burnin=1,
        actor_kl_target=1.0e-12,
        lr_adapt_factor=2.0,
        lr_min=2.0e-4,
        lr_max=4.0e-3,
    )
    burnin = learner.step(trajectories)
    assert burnin["opt/lr_policy"] == 0.0 and learner.policy_lr == 1.0e-3  # burn-in: no adaptation
    first = learner.step(trajectories)
    assert first["opt/lr_policy"] == 1.0e-3 and first["loss/actor_kl_mean"] > 2.0e-12
    assert learner.policy_lr == 5.0e-4 and learner.stage() == ("ppo", 2, 5.0e-4)
    second = learner.step(trajectories)
    assert second["opt/lr_policy"] == 5.0e-4 and learner.policy_lr == 2.5e-4
    learner.step(trajectories)
    learner.step(trajectories)
    assert learner.policy_lr == 2.0e-4  # clamped
    # A reverted step counts as too large a step: the rate comes down as well.
    learner.config = replace(learner.config, ppo=replace(learner.config.ppo, max_mean_actor_kl=1.0e-12))
    learner.policy_lr = 8.0e-4
    reverted = learner.step(trajectories)
    assert reverted["ppo/reverted"] == 1.0 and learner.policy_lr == 4.0e-4
    # Far below target / 2 the rate grows up to lr_max.
    growing = _learner(
        MeleePolicyAdapter(tiny_config),
        tiny_config,
        tiny_value_config,
        lr=1.0e-3,
        actor_kl_target=1.0e9,
        lr_adapt_factor=2.0,
        lr_max=3.0e-3,
    )
    growing.step(trajectories)
    assert growing.policy_lr == 2.0e-3
    growing.step(trajectories)
    assert growing.policy_lr == 3.0e-3


def test_reference_kl_is_logged_at_weight_zero(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    plain = _learner(MeleePolicyAdapter(tiny_config), tiny_config, tiny_value_config, seed=5)
    without = plain.step(rollout_trajectories(plain.policy, 2, synthetic_rewards=0.1))
    assert without["loss/reference_kl"] == 0.0 and without["loss/reverse_reference_kl"] == 0.0
    reference = tiny_adapter.clone_frozen()
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, reference=reference, seed=5)
    assert learner.reference is reference
    before = _params(tiny_adapter)
    # The reference has no weight in the loss: a twin without one takes the same step (same weights, same
    # value net, same data).
    twin = _learner(MeleePolicyAdapter(tiny_config), tiny_config, tiny_value_config, seed=5)
    twin.policy.set_state(tiny_adapter.get_state())
    twin.value_net.load_state_dict(learner.value_net.state_dict())
    first = learner.step(trajectories)
    assert first["pre/reference_kl"] < 1.0e-6  # the reference is the init: the parity floor
    assert first["loss/reference_kl"] > 0.0 and first["loss/reverse_reference_kl"] > 0.0
    assert _all_equal(_params(reference), before)  # never updated
    twin_metrics = twin.step(trajectories)
    assert twin_metrics["loss/total"] == first["loss/total"]
    assert twin_metrics["loss/reference_kl"] == 0.0
    assert _all_equal(_params(twin.policy), _params(tiny_adapter))  # type: ignore[arg-type]


def test_set_teacher_rebinds_the_lean_forward(
    tiny_adapter: MeleePolicyAdapter, tiny_config: ModelConfig, tiny_value_config: ValueConfig
) -> None:
    teacher = tiny_adapter.clone_frozen()
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, teacher=teacher, fast_forward=True)
    assert id(teacher) in learner._fast
    replacement = tiny_adapter.clone_frozen()
    learner.set_teacher(replacement)
    assert learner.teacher is replacement and id(replacement) in learner._fast
    assert id(teacher) not in learner._fast


# -- state ---------------------------------------------------------------------------------------


def test_learner_state_carries_the_ema_anchor_the_entropy_weight_and_the_rate(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
    tmp_path: Path,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 3, synthetic_rewards=0.1)
    teacher = tiny_adapter.clone_frozen()
    learner = _learner(
        tiny_adapter,
        tiny_config,
        tiny_value_config,
        teacher=teacher,
        ema_tau=0.5,
        entropy_target=100.0,
        entropy_gain=1.0e-3,
        entropy_weight_max=0.5,
        actor_kl_target=1.0e-12,
        lr_adapt_factor=2.0,
    )
    learner.step(trajectories[:1])
    learner.step(trajectories[1:2])
    state = learner.state_dict()
    assert state["entropy_weight"] == learner.entropy_weight > 0.0
    assert state["policy_lr"] == learner.policy_lr == 2.5e-4
    anchor = teacher.get_state()
    assert state["teacher"].keys() == anchor.keys()
    assert all(torch.equal(state["teacher"][key], anchor[key]) for key in anchor)

    latest, _ = checkpoint_paths(tmp_path, 2)
    payload = save_rl_checkpoint(
        latest, learner, tiny_config, step=2, teacher={"source": "init.pt", "sha256": "x", "ema_tau": 0.5}
    )
    assert payload["entropy_weight"] == state["entropy_weight"] and payload["policy_lr"] == 2.5e-4
    assert set(payload["teacher_state_dict"]) == set(state["teacher"])
    loaded = load_checkpoint(latest)  # weights_only-safe

    fresh_policy = MeleePolicyAdapter(tiny_config)
    fresh_teacher = fresh_policy.clone_frozen()
    resumed = _learner(
        fresh_policy,
        tiny_config,
        tiny_value_config,
        teacher=fresh_teacher,
        ema_tau=0.5,
        entropy_target=100.0,
        entropy_gain=1.0e-3,
        entropy_weight_max=0.5,
        actor_kl_target=1.0e-12,
        lr_adapt_factor=2.0,
        seed=9,
    )
    assert restore_learner(loaded, resumed) == 2
    assert resumed.entropy_weight == learner.entropy_weight and resumed.policy_lr == 2.5e-4
    assert _all_equal(_params(fresh_teacher), _params(teacher))
    assert _all_equal(_params(fresh_policy), _params(tiny_adapter))
    original = learner.step(trajectories[2:])
    continued = resumed.step(trajectories[2:])
    for key in ("loss/total", "loss/teacher_kl", "opt/entropy_weight", "opt/lr_policy"):
        assert original[key] == continued[key], key
    assert _all_equal(_params(fresh_teacher), _params(teacher))

    # A payload from before these levers restores with the config's own values.
    old = {k: v for k, v in loaded.items() if k not in ("entropy_weight", "policy_lr", "teacher_state_dict")}
    legacy = _learner(MeleePolicyAdapter(tiny_config), tiny_config, tiny_value_config, lr=7.0e-4)
    assert restore_learner(old, legacy) == 2
    assert legacy.entropy_weight == 0.0 and legacy.policy_lr == 7.0e-4


# -- the trainer ---------------------------------------------------------------------------------


def _ema_config(ema_tau: float = 0.5, **teacher_fields: object):  # type: ignore[no-untyped-def]
    config = load_config(LEARN_TINY)
    config = replace(
        config,
        teacher=TeacherConfig(source="ema", ema_tau=ema_tau, **teacher_fields),  # type: ignore[arg-type]
        learner=replace(config.learner, kl_teacher_weight=0.01),
        runtime=replace(config.runtime, save_interval_steps=1, snapshot_interval_steps=2),
    )
    return finalize_config(config, resolve_model_config(config))


def test_trainer_builds_saves_and_resumes_the_ema_anchor(tmp_path: Path) -> None:
    config = _ema_config()
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    first = Trainer(config, options)
    assert first.teacher is not None and first.learner.teacher_ema_tau == 0.5
    assert first.teacher_record == {"random_init": True, "seed": config.policy.seed, "ema_tau": 0.5}
    init = first.teacher.get_state()
    first.run(2)
    assert first.step == 2 and first.history[-1]["ppo/reverted"] == 0.0
    teacher_state, policy_state = first.teacher.get_state(), first.policy.get_state()
    moved = [k for k in init if not torch.equal(init[k], teacher_state[k])]
    assert moved, "the learnable signal moved the policy, so the EMA anchor must have moved too"
    payload = load_checkpoint(checkpoint_paths(tmp_path / "run", 2)[0])
    assert payload["teacher"]["ema_tau"] == 0.5 and payload["teacher_state_dict"] is not None
    first.close()

    second = Trainer(config, options)  # resumes: the anchor is the saved EMA, not the init
    assert second.resumed and second.step == 2 and second.teacher is not None
    assert second.learner.teacher is second.teacher
    resumed_state = second.teacher.get_state()
    assert all(torch.equal(resumed_state[k], teacher_state[k]) for k in teacher_state)
    assert all(torch.equal(second.policy.get_state()[k], policy_state[k]) for k in policy_state)
    second.run(1)
    assert second.step == 3
    second.close()

    # Another tau is another teacher: refused like another file.
    with pytest.raises(ValueError, match="teacher"):
        Trainer(_ema_config(ema_tau=0.25), options)


def test_reference_path_loads_a_frozen_file_whose_kl_is_logged(tmp_path: Path) -> None:
    config = _ema_config()
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    trainer = Trainer(config, options)
    trainer.run(2)
    trainer.close()
    snapshot = checkpoint_paths(tmp_path / "run", 2)[1]
    assert snapshot.exists()

    with_reference = _ema_config(reference_path=str(snapshot))
    assert with_reference.teacher.reference_path == str(snapshot)
    model_config = resolve_model_config(with_reference)
    reference = build_reference(with_reference, model_config, torch.device("cpu"))
    assert reference is not None and not any(p.requires_grad for p in reference.parameters())  # type: ignore[attr-defined]
    assert build_reference(config, model_config, torch.device("cpu")) is None
    teacher, record = build_teacher(with_reference, model_config, reference, torch.device("cpu"))
    assert teacher is not None and record is not None and record["ema_tau"] == 0.5
    assert "reference" not in record  # the reference is not part of the teacher's identity

    second = Trainer(with_reference, RunOptions(checkpoint_dir=tmp_path / "run2", wandb_mode="disabled"))
    assert second.learner.reference is not None
    second.run(1)
    row = second.history[-1]
    assert row["loss/reference_kl"] > 0.0 and row["loss/reverse_reference_kl"] > 0.0
    assert row["pre/reference_kl"] >= 0.0
    second.close()
