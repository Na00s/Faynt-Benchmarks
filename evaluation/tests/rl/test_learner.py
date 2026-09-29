"""PPO learner (PLAN.md §3.6, §6 P2): step, guard revert, microbatching, burn-in ladder, behaviour modes."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.learner import LearnerConfig, PPOConfig, PPOLearner, TrainablePolicy
from melee_rl.reward import RewardConfig
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueConfig, ValueNet
from model import ModelConfig

RolloutFactory = Callable[..., list[Trajectory]]

PLAN_METRICS = {
    "loss/total",
    "loss/ppo_objective",
    "loss/teacher_kl",
    "loss/reverse_teacher_kl",
    "loss/actor_kl_mean",
    "loss/actor_kl_max",
    "loss/actor_kl_pre",
    "loss/entropy",
    "ppo/log_rho_mean",
    "ppo/log_rho_abs_max",
    "ppo/clip_fraction",
    "ppo/reverted",
    "ppo/epochs",
    "value/loss",
    "value/uev",
    "value/return_mean",
    "value/value_mean",
    "opt/lr_policy",
    "opt/lr_value",
    "timing/learner_s",
}


def _learner(
    adapter: MeleePolicyAdapter,
    policy_config: ModelConfig,
    value_config: ValueConfig,
    *,
    teacher: bool = True,
    lr: float = 1.0e-3,
    epochs: int = 2,
    microbatch: int | None = None,
    max_kl: float = 1.0,
    value_burnin: int = 0,
    optimizer_burnin: int = 0,
    behaviour: str = "auto",
    kl_teacher_weight: float = 0.1,
    reward: RewardConfig | None = None,
    seed: int = 0,
    teacher_prefix: bool = False,
    prefix_on_host: bool = False,
    matmul_precision: str = "highest",
    fast_forward: bool = False,
    fast_forward_attention: str = "sdpa",
    compile: bool = False,
    compile_backend: str = "inductor",
    loss_weighting: str = "trajectory",
    value_lr: float | None = None,
) -> PPOLearner:
    torch.manual_seed(seed)
    value_net = ValueNet(policy_config, value_config).to(adapter.device)
    frozen = adapter.clone_frozen() if teacher else None
    config = LearnerConfig(
        learning_rate=lr,
        value_learning_rate=value_lr,
        loss_weighting=loss_weighting,
        kl_teacher_weight=kl_teacher_weight,
        microbatch_envs=microbatch,
        value_burnin_steps=value_burnin,
        optimizer_burnin_steps=optimizer_burnin,
        reward=reward,
        teacher_prefix=teacher_prefix,
        prefix_on_host=prefix_on_host,
        matmul_precision=matmul_precision,
        fast_forward=fast_forward,
        fast_forward_attention=fast_forward_attention,
        compile=compile,
        compile_backend=compile_backend,
        ppo=PPOConfig(num_epochs=epochs, max_mean_actor_kl=max_kl, behaviour_logits=behaviour),
    )
    return PPOLearner(adapter, frozen, value_net, config)


def _params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _all_equal(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def _optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return copy.deepcopy(optimizer.state_dict())


def _assert_optimizer_state_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    # The learning rate is a per-step setting (burn-in ladder), not reverted state.
    def without_lr(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in group.items() if key != "lr"} for group in groups]

    assert without_lr(left["param_groups"]) == without_lr(right["param_groups"])
    assert left["state"].keys() == right["state"].keys()
    for key, entries in left["state"].items():
        other = right["state"][key]
        assert entries.keys() == other.keys()
        for name, value in entries.items():
            assert torch.equal(value, other[name]), (key, name)


def as_trainable(adapter: MeleePolicyAdapter) -> TrainablePolicy:
    """Static proof that the adapter satisfies ``TrainablePolicy`` (checked by mypy)."""

    return adapter


@pytest.mark.parametrize("delay", [0, 2])
def test_ppo_step_is_finite_moves_params_and_starts_at_the_teacher(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
    delay: int,
) -> None:
    assert as_trainable(tiny_adapter) is tiny_adapter
    trajectories = rollout_trajectories(tiny_adapter, 2, delay=delay, synthetic_rewards=0.1)
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config)
    assert learner.stage() == ("ppo", 2, 1.0e-3) and learner.discount == pytest.approx(0.99712, abs=1.0e-5)
    before = _params(tiny_adapter)
    value_before = _params(learner.value_net)
    metrics = learner.step(trajectories)
    assert set(metrics) >= PLAN_METRICS
    assert all(torch.isfinite(torch.tensor(list(metrics.values()))).tolist()), metrics
    # At init the teacher is the policy: teacher KL vanishes before the update and the learner's
    # re-forward matches the actor exactly (reprime mode), so the first epoch sees actor KL = 0.
    assert metrics["pre/teacher_kl"] < 1.0e-6 and metrics["loss/actor_kl_pre"] < 1.0e-6
    assert metrics["ppo_step/0/actor_kl_mean"] < 1.0e-6 and metrics["ppo_step/0/clip_fraction"] == 0.0
    assert metrics["loss/entropy"] > 0.0 and metrics["pre/entropy"] > 0.0
    assert 0.0 <= metrics["ppo/clip_fraction"] <= 1.0
    assert metrics["ppo/epochs"] == 2.0 and metrics["ppo/reverted"] == 0.0 and metrics["learner/step"] == 1.0
    assert metrics["opt/grad_norm_policy"] > 0.0 and metrics["opt/grad_norm_value"] >= 0.0
    assert metrics["opt/lr_policy"] == 1.0e-3 and metrics["opt/lr_value"] == 1.0e-3
    assert not _all_equal(before, _params(tiny_adapter))
    assert not _all_equal(value_before, _params(learner.value_net))
    # After the update the policy has moved away from the teacher and from the actor.
    assert metrics["loss/teacher_kl"] > 0.0 and metrics["loss/actor_kl_mean"] > 0.0
    assert metrics["loss/actor_kl_max"] >= metrics["loss/actor_kl_mean"]
    # The last training epoch's gradients stay on the parameters for diagnostics.
    assert any(parameter.grad is not None for parameter in tiny_adapter.parameters())
    second = learner.step(rollout_trajectories(tiny_adapter, 2, delay=delay, seed=9, synthetic_rewards=0.1))
    assert second["learner/step"] == 2.0 and second["pre/teacher_kl"] > 0.0
    assert torch.isfinite(torch.tensor([second["loss/total"], second["value/loss"]])).all()


def test_guard_revert_restores_parameters_and_adam_state_bit_exactly(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    # One optimizer burn-in step populates the Adam moments without moving the parameters: the
    # post-update actor KL stays at the actor/learner parity floor (cached vs parallel path in fp32,
    # < 1e-6 here, cf. test_ppo_step_*), below the deliberately tiny 1e-6 guard ...
    learner = _learner(
        tiny_adapter, tiny_config, tiny_value_config, lr=1.0e-2, max_kl=1.0e-6, optimizer_burnin=1
    )
    first = learner.step(trajectories)
    assert first["ppo/epochs"] == 1.0 and first["opt/lr_policy"] == 0.0 and first["ppo/reverted"] == 0.0
    assert first["loss/actor_kl_mean"] < 1.0e-6
    assert len(learner.policy_optimizer.state) > 0
    params = _params(tiny_adapter)
    optimizer_state = _optimizer_state(learner.policy_optimizer)
    # ... then a real step at lr 1e-2 moves the policy beyond the (tiny) guard and is reverted.
    for _ in range(2):
        metrics = learner.step(trajectories)
        assert metrics["ppo/epochs"] == 2.0 and metrics["opt/lr_policy"] == 1.0e-2
        assert metrics["loss/actor_kl_mean"] > 1.0e-6 and metrics["ppo/reverted"] == 1.0
        assert _all_equal(params, _params(tiny_adapter))
        _assert_optimizer_state_equal(optimizer_state, _optimizer_state(learner.policy_optimizer))
    # The value net is not part of the revert: it kept training.
    assert learner.steps_done == 3


def test_microbatched_gradients_equal_full_batch_gradients(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    full_policy = MeleePolicyAdapter.from_policy(tiny_adapter)
    micro_policy = MeleePolicyAdapter.from_policy(tiny_adapter)
    full = _learner(full_policy, tiny_config, tiny_value_config, epochs=1, microbatch=None)
    micro = _learner(micro_policy, tiny_config, tiny_value_config, epochs=1, microbatch=1)
    assert _all_equal(_params(full.value_net), _params(micro.value_net))
    full_metrics = full.step(trajectories)
    micro_metrics = micro.step(trajectories)
    keys = ("loss/total", "loss/entropy", "pre/total", "value/loss", "value/return_mean", "ppo/clip_fraction")
    for key in keys:  # fp32 summation order differs between the two paths (host-dependent, ~1e-7 relative)
        assert math.isclose(full_metrics[key], micro_metrics[key], rel_tol=1.0e-6, abs_tol=1.0e-6), key
    compared = 0
    pairs = list(zip(full_policy.parameters(), micro_policy.parameters(), strict=True))
    pairs += list(zip(full.value_net.parameters(), micro.value_net.parameters(), strict=True))
    for left, right in pairs:
        if left.grad is None:  # a parameter the forward does not use (e.g. an unused embedding)
            assert right.grad is None
            continue
        assert right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=0.0, atol=1.0e-6)
        compared += 1
    assert compared > 10
    assert micro._chunks(3) == [slice(0, 1), slice(1, 2), slice(2, 3)] and full._chunks(3) == [slice(0, 3)]


def test_burnin_ladder_value_then_optimizer_then_ppo(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 1, synthetic_rewards=0.1)
    learner = _learner(
        tiny_adapter, tiny_config, tiny_value_config, lr=1.0e-2, value_burnin=1, optimizer_burnin=1
    )
    stages = [learner.stage(index)[0] for index in range(4)]
    assert stages == ["value_burnin", "optimizer_burnin", "ppo", "ppo"]
    policy_before, value_before = _params(tiny_adapter), _params(learner.value_net)
    # Value burn-in: no policy epoch, the value net trains, the policy optimizer has no state.
    metrics = learner.step(trajectories)
    assert metrics["ppo/epochs"] == 0.0 and metrics["ppo/stage"] == 0.0 and metrics["opt/lr_policy"] == 0.0
    assert metrics["opt/grad_norm_policy"] == 0.0 and metrics["loss/actor_kl_mean"] < 1.0e-6
    assert _all_equal(policy_before, _params(tiny_adapter))
    assert not _all_equal(value_before, _params(learner.value_net))
    assert len(learner.policy_optimizer.state) == 0
    # Optimizer burn-in: one epoch at lr 0 warms the Adam moments and leaves the parameters alone.
    metrics = learner.step(trajectories)
    assert metrics["ppo/epochs"] == 1.0 and metrics["ppo/stage"] == 1.0 and metrics["opt/lr_policy"] == 0.0
    assert _all_equal(policy_before, _params(tiny_adapter))
    moments = [state["exp_avg"] for state in learner.policy_optimizer.state.values()]
    assert moments and any(float(moment.abs().sum()) > 0.0 for moment in moments)
    # Full PPO afterwards.
    metrics = learner.step(trajectories)
    assert metrics["ppo/epochs"] == 2.0 and metrics["ppo/stage"] == 2.0 and metrics["opt/lr_policy"] == 1.0e-2
    assert not _all_equal(policy_before, _params(tiny_adapter))


def test_behaviour_logits_modes_on_ring_rollouts(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    # Rollout 3 of a ring-mode worker mismatches the learner re-forward (tests/rl/test_actor.py).
    ring = rollout_trajectories(tiny_adapter, 4, context_mode="ring", game_frames=10_000)[-1:]
    auto = _learner(tiny_adapter, tiny_config, tiny_value_config, epochs=1, lr=0.0)
    assert auto._behaviour_mode(ring) == "learner"
    metrics = auto.step(ring)
    # Learner-side behaviour logits: the first epoch's actor KL is 0 by construction, the mismatch
    # between the stored actor logits and the learner's pre-pass is reported separately.
    assert metrics["ppo_step/0/actor_kl_mean"] < 1.0e-6 and metrics["ppo_step/0/clip_fraction"] == 0.0
    assert metrics["loss/actor_kl_pre"] > 0.0
    actor = _learner(tiny_adapter, tiny_config, tiny_value_config, epochs=1, lr=0.0, behaviour="actor")
    assert actor._behaviour_mode(ring) == "actor"
    actor_metrics = actor.step(ring)
    assert actor_metrics["loss/actor_kl_pre"] == actor_metrics["ppo_step/0/actor_kl_mean"] > 0.0
    assert abs(actor_metrics["loss/actor_kl_pre"] - metrics["loss/actor_kl_pre"]) < 1.0e-6
    reprime = rollout_trajectories(tiny_adapter, 1)
    assert auto._behaviour_mode(reprime) == "actor"


def test_prefix_rollouts_are_exact_for_the_learner(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """Prefix mode (INTERFACE.md patch #1): the learner forwards the decision rows against the stored cache
    and reproduces the actor, so the stored logits are the exact behaviour distribution and the first
    epoch's actor KL -- ``loss/actor_kl_pre`` -- reads 0: for rollout 0 (an empty ring, as reprime mode)
    and for rollout 3, the one that mismatches in ring mode, with and without env microbatches.  The
    teacher still reads the ``C + T`` window, so its KL at rollout 3 is a context mismatch, not zero."""

    rollouts = rollout_trajectories(tiny_adapter, 4, context_mode="prefix", game_frames=10_000)
    assert all(trajectory.initial_cache is not None for trajectory in rollouts)
    for index, microbatch in ((0, None), (3, None), (3, 1)):
        learner = _learner(
            tiny_adapter, tiny_config, tiny_value_config, epochs=1, lr=0.0, microbatch=microbatch
        )
        batch = rollouts[index : index + 1]
        assert learner._behaviour_mode(batch) == "actor"
        metrics = learner.step(batch)
        assert metrics["loss/actor_kl_pre"] < 1.0e-6, (index, microbatch)
        assert metrics["ppo_step/0/clip_fraction"] == 0.0 and metrics["ppo/reverted"] == 0.0
        assert math.isfinite(metrics["loss/total"])
        if index == 0:
            assert metrics["pre/teacher_kl"] < 1.0e-6  # empty ring: the window and prefix paths agree
    # The teacher's context mismatch at rollout 3 (its C + T window vs the policy's ring) vanishes when the
    # frozen teacher attends to the same stored prefix (`learner.teacher_prefix`): the smoke of 3 Sep 2026
    # read teacher_kl 4.95e-3 at unchanged weights without it.
    window_teacher = _learner(tiny_adapter, tiny_config, tiny_value_config, epochs=1, lr=0.0)
    prefix_teacher = _learner(
        tiny_adapter, tiny_config, tiny_value_config, epochs=1, lr=0.0, teacher_prefix=True
    )
    mismatch = window_teacher.step(rollouts[-1:])["pre/teacher_kl"]
    matched = prefix_teacher.step(rollouts[-1:])["pre/teacher_kl"]
    assert matched < 1.0e-6 and matched < mismatch
    # Training through the prefix path moves the parameters and stays finite.
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, epochs=2, lr=1.0e-3)
    before = _params(tiny_adapter)
    metrics = learner.step(rollouts[-2:])
    assert math.isfinite(metrics["loss/total"]) and metrics["weights/drift_step"] > 0.0
    assert not _all_equal(before, _params(tiny_adapter))


def _same_weights(
    adapter: MeleePolicyAdapter, config: ModelConfig, device: str = "cpu"
) -> MeleePolicyAdapter:
    """A fresh adapter with ``adapter``'s weights on ``device`` (``deepcopy`` cannot copy Ali's codec)."""

    policy = MeleePolicyAdapter(config)
    policy.load_state_dict(adapter.state_dict(), strict=True)
    return policy.to(device)


def test_prefix_on_host_changes_nothing_but_where_the_caches_live(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """``learner.prefix_on_host`` (the 75m run, 4 Sep 2026): the stored prefixes stay in host memory across
    the PPO epochs and each microbatch's slice is moved to the learner device on demand -- 16 x 168 fp32
    75m prefixes are 10.8 GiB per update, which cannot sit beside the activations on a 24 GB L4.  Same
    trajectories, same seed: bit-identical metrics and parameters either way."""

    rollouts = rollout_trajectories(tiny_adapter, 3, context_mode="prefix", game_frames=10_000)
    results = []
    for on_host in (False, True):
        policy = _same_weights(tiny_adapter, tiny_config)
        learner = _learner(
            policy,
            tiny_config,
            tiny_value_config,
            epochs=2,
            lr=1.0e-3,
            microbatch=2,
            teacher_prefix=True,
            prefix_on_host=on_host,
        )
        assert learner.config.prefix_on_host is on_host
        results.append((learner.step(rollouts), _params(policy)))
    (plain, plain_params), (hosted, hosted_params) = results
    assert plain.keys() == hosted.keys()
    for key, value in plain.items():
        if not key.startswith("timing/"):
            assert value == hosted[key], key
    assert hosted["weights/drift_step"] > 0.0 and _all_equal(plain_params, hosted_params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_prefix_on_host_keeps_the_caches_off_the_cuda_learner(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """On a real second device the prepared batches keep their prefixes on the CPU while everything else
    is on the learner's device, the exactness of the prefix path survives the on-demand move, and the
    metrics match the all-on-device learner."""

    rollouts = rollout_trajectories(tiny_adapter, 2, context_mode="prefix", game_frames=10_000)
    hosted = _learner(
        _same_weights(tiny_adapter, tiny_config, "cuda"),
        tiny_config,
        tiny_value_config,
        epochs=1,
        lr=0.0,
        microbatch=2,
        teacher_prefix=True,
        prefix_on_host=True,
    )
    assert hosted.device.type == "cuda"
    for batch in [hosted._prepare(trajectory, "actor") for trajectory in rollouts]:
        cache = batch.trajectory.initial_cache
        assert (
            cache is not None and cache.keys.device.type == "cpu" and cache.valid_length.device.type == "cpu"
        )
        assert batch.trajectory.rewards.device.type == "cuda" and batch.log_old.device.type == "cuda"
    metrics = hosted.step(rollouts)
    # The actor ran on the CPU: only float differences between the devices remain.
    assert metrics["loss/actor_kl_pre"] < 1.0e-4 and metrics["pre/teacher_kl"] < 1.0e-4
    plain = _learner(
        _same_weights(tiny_adapter, tiny_config, "cuda"),
        tiny_config,
        tiny_value_config,
        epochs=1,
        lr=0.0,
        microbatch=2,
        teacher_prefix=True,
    )
    reference = plain.step(rollouts)
    for key in ("loss/total", "loss/ppo_objective", "loss/entropy", "value/loss", "ppo/clip_fraction"):
        assert metrics[key] == pytest.approx(reference[key], rel=1.0e-5, abs=1.0e-7), key


def test_no_teacher_reward_recompute_and_validation(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 1)
    with pytest.raises(ValueError, match="teacher KL weights"):
        _learner(tiny_adapter, tiny_config, tiny_value_config, teacher=False, kl_teacher_weight=0.1)
    no_teacher = _learner(
        tiny_adapter, tiny_config, tiny_value_config, teacher=False, kl_teacher_weight=0.0, epochs=1
    )
    metrics = no_teacher.step(trajectories)
    assert metrics["loss/teacher_kl"] == 0.0 and metrics["loss/reverse_teacher_kl"] == 0.0
    # Recomputing the rewards with the worker's config changes nothing.
    same = _learner(tiny_adapter, tiny_config, tiny_value_config, epochs=0, reward=RewardConfig())
    plain = _learner(tiny_adapter, tiny_config, tiny_value_config, epochs=0)
    assert same.step(trajectories)["value/return_mean"] == plain.step(trajectories)["value/return_mean"]
    with pytest.raises(ValueError, match="at least one"):
        plain.step([])
    with pytest.raises(ValueError, match="epsilon"):
        PPOConfig(epsilon=0.0)
    with pytest.raises(ValueError, match="behaviour_logits"):
        PPOConfig(behaviour_logits="cache")
    with pytest.raises(ValueError, match="num_epochs"):
        PPOConfig(num_epochs=-1)
    with pytest.raises(ValueError, match="microbatch_envs"):
        LearnerConfig(microbatch_envs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LearnerConfig(learning_rate=-1.0)
    with pytest.raises(ValueError, match="kl_teacher_weight"):
        LearnerConfig(kl_teacher_weight=-0.1)
    assert LearnerConfig(value_learning_rate=None).effective_value_learning_rate == 1.0e-4
    assert LearnerConfig(value_learning_rate=0.5).effective_value_learning_rate == 0.5
    state = plain.state_dict()
    # 15 Sep 2026: the live entropy bonus and learning rate ride along (an EMA anchor adds "teacher").
    assert set(state) == {
        "policy", "value", "policy_optimizer", "value_optimizer", "step", "entropy_weight", "policy_lr"
    }
    assert state["step"] == 1 and state["entropy_weight"] == 0.0 and state["policy_lr"] == 1.0e-3


def _relative_distance(
    current: Sequence[torch.Tensor], reference: Sequence[torch.Tensor], scale: float
) -> float:
    total = sum(
        float(((now.detach().float() - before.detach().float()) ** 2).sum())
        for now, before in zip(current, reference, strict=True)
    )
    return math.sqrt(total) / scale


def test_weight_drift_metrics_say_whether_the_policy_actually_moved(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """``weights/drift_*``: what actually stuck, as opposed to what was attempted (session 13).

    Two runs produced a flat policy for opposite reasons -- updates far too small to matter (actor KL
    3.5e-5 against a 1e-2 guard) and updates so large that the guard reverted every one of them (520 of
    520).  ``loss/actor_kl_mean`` reports the *attempted* step in both cases; only the cumulative drift
    distinguishes "moving slowly" from "not moving at all", which otherwise costs a checkpoint download
    and an offline diff to establish.
    """

    trajectories = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=0.1)
    # A reverted step restores the parameters bit-exactly, so both drifts must be exactly zero.
    reverting = _learner(tiny_adapter, tiny_config, tiny_value_config, lr=1.0e-2, max_kl=1.0e-6)
    reverted = reverting.step(trajectories)
    assert reverted["ppo/reverted"] == 1.0
    assert reverted["weights/drift_step"] == 0.0 and reverted["weights/drift_total"] == 0.0

    # An accepted step moves the policy, and both numbers match a directly computed distance.
    moving = _learner(tiny_adapter, tiny_config, tiny_value_config, lr=1.0e-2, max_kl=1.0)
    reference = [parameter.detach().clone() for parameter in moving.policy_parameters]
    scale = math.sqrt(sum(float((p.detach().float() ** 2).sum()) for p in reference))
    first = moving.step(trajectories)
    assert first["ppo/reverted"] == 0.0
    expected = _relative_distance(moving.policy_parameters, reference, scale)
    assert first["weights/drift_step"] == pytest.approx(expected, rel=1.0e-5)
    assert first["weights/drift_total"] == pytest.approx(expected, rel=1.0e-5)
    assert first["weights/drift_total"] > 0.0

    # The total is measured from the learner's construction, the step from the previous step only.
    after = [parameter.detach().clone() for parameter in moving.policy_parameters]
    second = moving.step(trajectories)
    assert second["weights/drift_step"] == pytest.approx(
        _relative_distance(moving.policy_parameters, after, scale), rel=1.0e-5
    )
    assert second["weights/drift_total"] == pytest.approx(
        _relative_distance(moving.policy_parameters, reference, scale), rel=1.0e-5
    )
    # Resuming re-bases the reference: drift is "since this launch", not "since a random init".
    moving.load_state_dict(moving.state_dict())
    rebased = moving.step(trajectories)
    assert rebased["weights/drift_total"] == pytest.approx(rebased["weights/drift_step"], rel=1.0e-5)


def test_matmul_precision_is_scoped_to_the_policy_passes_and_restored(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """5 Sep 2026 (/rl-speedup): ``learner.matmul_precision`` sets ``torch.set_float32_matmul_precision``
    around the policy's passes only and puts it back; on the CPU it changes nothing: the run is identical."""

    from melee_rl.learner import MATMUL_PRECISIONS, matmul_precision

    assert MATMUL_PRECISIONS == ("highest", "high", "medium")
    with pytest.raises(ValueError, match="matmul_precision"):
        LearnerConfig(matmul_precision="fast")
    assert torch.get_float32_matmul_precision() == "highest"
    with matmul_precision("high"):
        assert torch.get_float32_matmul_precision() == "high"
        with matmul_precision("highest"):
            assert torch.get_float32_matmul_precision() == "high"  # "highest" means: leave it as it is
    assert torch.get_float32_matmul_precision() == "highest"
    rollouts = rollout_trajectories(tiny_adapter, 2, synthetic_rewards=1.0)
    results = []
    for setting in ("highest", "high"):
        policy = _same_weights(tiny_adapter, tiny_config)
        learner = _learner(
            policy, tiny_config, tiny_value_config, epochs=1, lr=1e-3, matmul_precision=setting
        )
        assert learner.config.matmul_precision == setting
        results.append((learner.step(rollouts), _params(policy)))
        assert torch.get_float32_matmul_precision() == "highest"
    (plain_metrics, plain_params), (fast_metrics, fast_params) = results
    for key in plain_metrics:
        if not key.startswith("timing/"):
            assert plain_metrics[key] == fast_metrics[key], key
    assert all(torch.equal(left, right) for left, right in zip(plain_params, fast_params, strict=True))


@pytest.mark.parametrize("attention", ["sdpa", "bmm"])
def test_fast_forward_matches_the_eager_learner(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
    attention: str,
) -> None:
    """5 Sep 2026 (/rl-learner-speedup): ``learner.fast_forward`` routes the policy's and the teacher's
    prefix passes through the lean forward.  Same prefix rollouts (an empty ring, a partial one, a full one;
    resets in the windows), same seed, ``prefix_on_host`` staging on: every metric agrees with the eager
    learner within fp32 re-association, the exactness monitors stay at the noise floor, the gradients agree
    per parameter, and a training step through the lean path moves the parameters."""

    rollouts = rollout_trajectories(tiny_adapter, 4, context_mode="prefix")
    results = []
    for fast in (False, True):
        policy = _same_weights(tiny_adapter, tiny_config)
        learner = _learner(
            policy,
            tiny_config,
            tiny_value_config,
            epochs=1,
            lr=0.0,
            microbatch=2,
            teacher_prefix=True,
            prefix_on_host=True,
            fast_forward=fast,
            fast_forward_attention=attention,
        )
        assert learner.config.fast_forward is fast and bool(learner._fast) is fast
        results.append((learner.step(rollouts), [p.grad for p in policy.parameters()]))
    (plain, plain_grads), (lean, lean_grads) = results
    assert plain.keys() == lean.keys()
    for key, value in plain.items():
        if key.startswith("timing/"):
            continue
        assert math.isclose(value, lean[key], rel_tol=1.0e-4, abs_tol=1.0e-6), (key, value, lean[key])
    assert lean["loss/actor_kl_pre"] < 1.0e-6 and lean["pre/teacher_kl"] < 1.0e-6
    compared = 0
    for a, b in zip(plain_grads, lean_grads, strict=True):
        if a is None:
            assert b is None
            continue
        assert b is not None
        torch.testing.assert_close(b, a, rtol=1.0e-4, atol=1.0e-5)
        compared += 1
    assert compared > 20
    policy = _same_weights(tiny_adapter, tiny_config)
    learner = _learner(
        policy, tiny_config, tiny_value_config, epochs=2, lr=1.0e-3, microbatch=2, fast_forward=True
    )
    before = _params(policy)
    metrics = learner.step(rollouts[-2:])
    assert math.isfinite(metrics["loss/total"]) and metrics["weights/drift_step"] > 0.0
    assert not _all_equal(before, _params(policy))
    with pytest.raises(ValueError, match="fast_forward_attention"):
        LearnerConfig(fast_forward_attention="flash")


def test_row_weighting_makes_a_mix_step_the_step_of_its_concatenation(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """17 Sep 2026: the learner weighs every trajectory equally. Each one's row-mean loss is scaled by
    1 / trajectories, so a mix whose arms hold different row counts trains each ARM at an equal share: the
    league's five arms at 20 % each, against its planned rows 48 / 19 / 19 / 10 / 5 %, with a Falco row
    counting 10x a self-play row. ``loss_weighting = "row"`` weighs every decision row of the step the same,
    so trajectories of 4 and 2 rows give the gradient of their concatenation. The default ``"trajectory"``
    keeps today's arithmetic (and the metrics, which were always row-weighted)."""

    wide = rollout_trajectories(tiny_adapter, 1, num_envs=4, synthetic_rewards=1.0)[0]
    narrow = rollout_trajectories(tiny_adapter, 1, num_envs=2, env_seed=9, synthetic_rewards=1.0)[0]
    joined = Trajectory.batch([wide, narrow])

    def grad_norm(trajectories: Sequence[Trajectory], weighting: str) -> float:
        policy = _same_weights(tiny_adapter, tiny_config)
        learner = _learner(
            policy,
            tiny_config,
            tiny_value_config,
            epochs=1,
            microbatch=2,
            loss_weighting=weighting,
            value_lr=0.0,  # a frozen critic: the second trajectory's advantages match the joined step's
        )
        return learner.step(trajectories)["ppo_step/0/grad_norm"]

    reference = grad_norm([joined], "trajectory")
    assert math.isclose(grad_norm([wide, narrow], "row"), reference, rel_tol=1.0e-5)
    assert math.isclose(grad_norm([joined], "row"), reference, rel_tol=1.0e-6)  # one trajectory: the same
    per_arm = grad_norm([wide, narrow], "trajectory")
    assert not math.isclose(per_arm, reference, rel_tol=1.0e-2)  # the 2-row arm counts double per row
    assert LearnerConfig().loss_weighting == "trajectory"
    with pytest.raises(ValueError, match="loss_weighting"):
        LearnerConfig(loss_weighting="arm")


def test_compile_keeps_one_graph_per_mode_over_arms_of_different_sizes(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """17 Sep 2026: a mix step's trajectories have different row counts (the league: 60 / 24 / 24 / 12 /
    6), all cut into equal chunks. Through the value burn-in, the optimizer burn-in and PPO, the compiled lean
    pass stays at three graphs: the teacher's pass and the policy's with and without gradients. Before the
    contiguous inputs, each row count added its own set, and the league smoke died at step 7 on dynamo's
    recompile limit."""

    import torch._dynamo
    from torch._dynamo.utils import counters

    arms = [
        rollout_trajectories(tiny_adapter, 1, num_envs=envs, context_mode="prefix", env_seed=seed)[0]
        for envs, seed in ((6, 5), (4, 6), (2, 7))
    ]
    torch._dynamo.reset()
    counters.clear()
    policy = _same_weights(tiny_adapter, tiny_config)
    learner = _learner(
        policy,
        tiny_config,
        tiny_value_config,
        epochs=1,
        microbatch=2,
        value_burnin=1,
        optimizer_burnin=1,
        teacher_prefix=True,
        fast_forward=True,
        compile=True,
        compile_backend="aot_eager",
    )
    stages = []
    for _ in range(3):
        metrics = learner.step(arms)
        stages.append(int(metrics["ppo/stage"]))
        assert math.isfinite(metrics["loss/total"])
    assert stages == [0, 1, 2]
    assert counters["stats"]["unique_graphs"] <= 3, dict(counters["stats"])
    torch._dynamo.reset()


def test_compile_runs_the_lean_pass_as_one_graph_per_mode(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
) -> None:
    """5 Sep 2026 (/rl-learner-speedup, lever 2): ``learner.compile`` traces the lean pass with
    ``torch.compile(fullgraph=True)``; on the CPU the ``aot_eager`` backend proves the graph closes (no graph
    breaks), the backward compiles, and the metrics of two learner steps match the eager lean learner within
    the decompositions' rounding.  The config refuses ``compile`` without ``fast_forward`` and with
    checkpointing on."""

    import torch._dynamo
    from torch._dynamo.utils import counters

    with pytest.raises(ValueError, match="compile"):
        LearnerConfig(compile=True)
    with pytest.raises(ValueError, match="compile_backend"):
        LearnerConfig(fast_forward=True, compile=True, compile_backend="cudagraphs")
    checkpointed = _same_weights(tiny_adapter, replace(tiny_config, gradient_checkpointing=True))
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        _learner(
            checkpointed,
            tiny_config,
            tiny_value_config,
            fast_forward=True,
            compile=True,
            compile_backend="aot_eager",
        )
    rollouts = rollout_trajectories(tiny_adapter, 3, context_mode="prefix")
    results = []
    for compiled in (False, True):
        torch._dynamo.reset()
        counters.clear()
        policy = _same_weights(tiny_adapter, tiny_config)
        learner = _learner(
            policy,
            tiny_config,
            tiny_value_config,
            epochs=2,
            lr=1.0e-3,
            microbatch=3,
            teacher_prefix=True,
            fast_forward=True,
            compile=compiled,
            compile_backend="aot_eager",
        )
        metrics = [learner.step(rollouts[:2]), learner.step(rollouts[1:])]
        if compiled:
            assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
            # The policy's train and post passes and the teacher's pass: three graphs through AOTAutograd.
            assert counters["stats"]["unique_graphs"] >= 3 and counters["aot_autograd"]["ok"] >= 3
        results.append((metrics, _params(policy)))
    (eager, eager_params), (compiled_metrics, compiled_params) = results
    for step_eager, step_compiled in zip(eager, compiled_metrics, strict=True):
        for key, value in step_eager.items():
            if key.startswith("timing/"):
                continue
            assert math.isclose(value, step_compiled[key], rel_tol=1.0e-4, abs_tol=1.0e-6), (key, value)
    for left, right in zip(eager_params, compiled_params, strict=True):
        torch.testing.assert_close(
            left, right, rtol=1.0e-4, atol=1.0e-5 * max(float(right.abs().max()), 1e-3)
        )
