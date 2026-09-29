"""Trajectory slices (PLAN.md §2.3 / INTERFACE.md §9) on hand-built trajectories for D = 0 and D = 2."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from controller_codec import ControllerLabels
from melee_rl.frames import labels_index, neutral_frame, tree_leaves
from melee_rl.trajectory import Trajectory, Window
from model import KVCache

B = 2
C = 3
T = 5
W = C + T + 1


def _fake_cache(batch: int, *, layers: int = 2, capacity: int = 6, seed: int = 1) -> KVCache:
    """A ring-shaped ``[L, B, C, K, Dh]`` cache with per-row metadata ``1, 2, ...`` (no model needed)."""

    generator = torch.Generator().manual_seed(seed)
    shape = (layers, batch, capacity, 1, 4)
    rows = torch.arange(1, batch + 1)
    return KVCache(
        keys=torch.randn(shape, generator=generator),
        values=torch.randn(shape, generator=generator),
        valid_length=rows.clone(),
        write_position=rows.clone(),
        next_position=rows.clone(),
    )


def _hand_built(delay: int, *, env_reward: float = 0.0) -> Trajectory:
    """Row-coded labels: the sample at row ``w`` is ``100 + w`` (buttons) / ``w % 85`` (main stick).

    ``controller_exec[w]`` is the sample executed on the transition into ``w``, i.e. the sample of
    row ``w - 1 - D`` (neutral before the stream started), ``controller_prev_sample[w]`` the sample
    of ``w - 1``; row ``C + T`` of ``sampled_labels`` is the neutral placeholder.
    """

    rows = torch.arange(W)
    sampled_b = torch.where(rows < C + T, 100 + rows, 0)
    sampled_m = torch.where(rows < C + T, rows % 85, 0)
    exec_rows = rows - 1 - delay
    exec_b = torch.where(exec_rows >= 0, 100 + exec_rows, 0)
    exec_m = torch.where(exec_rows >= 0, exec_rows % 85, 0)
    prev_rows = rows - 1
    prev_b = torch.where(prev_rows >= 0, 100 + prev_rows, 0)
    prev_m = torch.where(prev_rows >= 0, prev_rows % 85, 0)

    def batched(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(0).expand(B, W).clone()

    frames = neutral_frame((B, W))
    frames["p1"]["percent"][:, C + 1 :] = 50  # p1 takes 50 percent on the first rollout transition
    generator = torch.Generator().manual_seed(0)
    is_resetting = torch.zeros((B, W), dtype=torch.bool)
    is_resetting[:, 0] = True
    return Trajectory(
        frames=frames,
        controller_exec=ControllerLabels(batched(exec_b), batched(exec_m)),
        controller_prev_sample=ControllerLabels(batched(prev_b), batched(prev_m)),
        sampled_labels=ControllerLabels(batched(sampled_b), batched(sampled_m)),
        sampled_logits={
            "buttons": torch.randn((B, T, 728), generator=generator),
            "main_stick": torch.randn((B, T, 85), generator=generator),
        },
        rewards=torch.full((B, T), 0.5 + env_reward),
        env_rewards=torch.full((B, T), env_reward),
        is_resetting=is_resetting,
        padding=torch.ones((B, W), dtype=torch.bool),
        frame_index=torch.arange(W).unsqueeze(0).expand(B, W).clone() - 123,
        context_frames=C,
        rollout_length=T,
        delay_frames=delay,
        context_mode="reprime",
    )


@pytest.mark.parametrize("delay", [0, 2])
def test_slices_follow_the_contract(delay: int) -> None:
    trajectory = _hand_built(delay)
    trajectory.validate()
    assert trajectory.batch_size == B and trajectory.window_length == W
    assert trajectory.policy_rows == slice(0, C + T - delay)
    assert trajectory.decision_rows == slice(C, C + T - delay)
    assert trajectory.actor_rows == slice(0, T - delay)
    assert trajectory.advantage_rows == slice(delay, T)
    assert trajectory.value_rows == slice(0, W)

    window = trajectory.policy_window()
    assert isinstance(window, Window) and window.length == C + T - delay
    assert window.targets is not None
    assert window.targets.buttons[0].tolist() == [100 + w for w in range(C + T - delay)]
    assert window.controller_prev.buttons[0].tolist() == [0, *(100 + w for w in range(C + T - delay - 1))]
    assert window.reset_mask[0].tolist() == [True] + [False] * (C + T - delay - 1)
    assert bool(window.padding_mask.all())
    assert window.frames["p1"]["percent"].shape == (B, C + T - delay)

    # The decisions entering the loss and their behaviour logits line up row by row.
    labels = trajectory.actor_labels()
    assert labels.buttons[0].tolist() == [100 + C + t for t in range(T - delay)]
    logits = trajectory.actor_logits()
    assert logits["buttons"].shape == (B, T - delay, 728) and logits["main_stick"].shape == (B, T - delay, 85)
    torch.testing.assert_close(logits["buttons"], trajectory.sampled_logits["buttons"][:, : T - delay])

    # sampled_labels[w] == controller_exec[w + 1 + D] for every decision row of the window.
    decisions = labels_index(trajectory.sampled_labels, (slice(None), slice(C, C + T - delay)))
    executed = labels_index(trajectory.controller_exec, (slice(None), slice(C + 1 + delay, C + T + 1)))
    assert torch.equal(decisions.buttons, executed.buttons)
    assert torch.equal(decisions.main_stick, executed.main_stick)

    # The delayed actions are the last D samples, not executed inside this window.
    delayed = trajectory.delayed_actions
    assert delayed.buttons.shape == (B, delay)
    assert delayed.buttons[0].tolist() == [100 + w for w in range(C + T - delay, C + T)]
    # The bootstrap row carries no sample (neutral placeholder).
    assert trajectory.sampled_labels.buttons[:, C + T].tolist() == [0] * B

    value = trajectory.value_window()
    assert value.length == W and value.targets is None
    assert torch.equal(value.controller_prev.buttons, trajectory.controller_exec.buttons)
    rollout = trajectory.rollout_frames()
    assert rollout["p0"]["percent"].shape == (B, T + 1)


def test_recompute_rewards_batch_index_to_and_memory_report() -> None:
    trajectory = _hand_built(0, env_reward=0.25)
    recomputed = trajectory.recompute_rewards()
    # +50 percent on p1 at the first transition -> +0.5, plus the env reward 0.25 everywhere.
    expected = torch.full((B, T), 0.25)
    expected[:, 0] += 0.5
    torch.testing.assert_close(recomputed.rewards, expected)
    assert torch.equal(recomputed.env_rewards, trajectory.env_rewards)

    doubled = Trajectory.batch([trajectory, trajectory])
    doubled.validate()
    assert doubled.batch_size == 2 * B
    assert doubled.sampled_logits["buttons"].shape == (2 * B, T, 728)
    assert doubled.frames["items"]["exists"].shape == (2 * B, W, 15)
    single = doubled.index_envs([1])
    single.validate()
    assert single.batch_size == 1 and torch.equal(single.rewards[0], trajectory.rewards[1])
    moved = trajectory.to("cpu")
    assert moved.device.type == "cpu" and moved.context_frames == C

    report = trajectory.memory_report()
    assert report["total"] == sum(value for key, value in report.items() if key != "total")
    assert report["logits"] == B * T * (728 + 85) * 4
    assert report["labels"] == 3 * 2 * B * W * 8
    frame_bytes = sum(leaf.numel() * leaf.element_size() for _, leaf in tree_leaves(trajectory.frames))
    assert report["frames"] == frame_bytes

    with pytest.raises(ValueError, match="must share"):
        Trajectory.batch([trajectory, _hand_built(2)])
    with pytest.raises(ValueError, match="at least one"):
        Trajectory.batch([])


@pytest.mark.parametrize("delay", [0, 2])
def test_prefix_mode_carries_the_actor_cache(delay: int) -> None:
    """``context_mode = "prefix"``: the snapshot rides along every per-env operation, and the policy window
    is the decision rows alone (the history lives in the cache)."""

    good = _hand_built(delay)
    with pytest.raises(ValueError, match="requires initial_cache"):
        replace(good, context_mode="prefix").validate()
    with pytest.raises(ValueError, match="does not carry"):
        replace(good, initial_cache=_fake_cache(B)).validate()
    prefix = replace(good, context_mode="prefix", initial_cache=_fake_cache(B))
    prefix.validate()
    with pytest.raises(ValueError, match="initial_cache K/V"):
        replace(prefix, initial_cache=_fake_cache(B + 1)).validate()
    cache = prefix.initial_cache
    assert cache is not None

    window = prefix.prefix_window()
    assert window.length == T - delay and window.targets is not None
    assert window.targets.buttons[0].tolist() == [100 + w for w in range(C, C + T - delay)]
    assert window.controller_prev.buttons[0].tolist() == [100 + w - 1 for w in range(C, C + T - delay)]
    assert torch.equal(window.reset_mask, good.is_resetting[:, C : C + T - delay])
    assert torch.equal(window.padding_mask, good.padding[:, C : C + T - delay])
    assert window.frames["p1"]["percent"].shape == (B, T - delay)

    doubled = Trajectory.batch([prefix, prefix])
    doubled.validate()
    assert doubled.initial_cache is not None and doubled.initial_cache.batch_size == 2 * B
    single = doubled.index_envs([1])
    single.validate()
    assert single.initial_cache is not None
    torch.testing.assert_close(single.initial_cache.keys[:, 0], cache.keys[:, 1], rtol=0.0, atol=0.0)
    assert single.initial_cache.next_position.tolist() == [int(cache.next_position[1])]
    moved = prefix.to("cpu")
    assert moved.initial_cache is not None and moved.initial_cache.keys.device.type == "cpu"
    kept = prefix.to("cpu", cache=False)  # the learner's prefix_on_host: the snapshot stays where it is
    assert kept.initial_cache is cache and kept.rewards.device.type == "cpu"
    assert prefix.recompute_rewards().initial_cache is cache

    report = prefix.memory_report()
    assert report["cache"] == cache.storage_report()["bytes"] > 0
    assert report["total"] == sum(value for key, value in report.items() if key != "total")
    assert good.memory_report()["cache"] == 0
    with pytest.raises(ValueError, match="all carry"):
        Trajectory.batch([prefix, replace(prefix, initial_cache=None)])


def test_validate_rejects_bad_shapes_and_delays() -> None:
    good = _hand_built(0)

    with pytest.raises(ValueError, match="rewards must be"):
        replace(good, rewards=good.rewards[:, :-1]).validate()
    with pytest.raises(ValueError, match="sampled_logits"):
        replace(good, sampled_logits={"buttons": good.sampled_logits["buttons"]}).validate()
    with pytest.raises(ValueError, match="delay_frames"):
        replace(good, delay_frames=T).validate()
    with pytest.raises(ValueError, match="context_mode"):
        replace(good, context_mode="cache").validate()
    with pytest.raises(ValueError, match="padded rows cannot be resetting"):
        replace(good, padding=torch.zeros_like(good.padding)).validate()
    with pytest.raises(ValueError, match=r"controller_exec\.buttons must be in"):
        bad = ControllerLabels(good.controller_exec.buttons + 700, good.controller_exec.main_stick)
        replace(good, controller_exec=bad).validate()
