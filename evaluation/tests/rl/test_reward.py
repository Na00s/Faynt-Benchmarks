"""Reward hand sequences (PLAN.md §6 P1, P6a): deaths, damage, Nana, zero-sum, reset zeroing, bounds, and the
slippi-ai ledge / stalling / approach terms (``slippi_ai/reward.py:24-92, 142-161``)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from melee_rl import reward as reward_lib
from melee_rl.env.protocol import swap_perspective
from melee_rl.frames import random_valid_frames
from melee_rl.reward import RewardConfig


def _player(
    action: list[int],
    percent: list[int],
    *,
    x: list[float] | None = None,
    y: list[float] | None = None,
    invulnerable: list[bool] | None = None,
    nana_exists: list[bool] | None = None,
    nana_action: list[int] | None = None,
    nana_percent: list[int] | None = None,
) -> dict[str, Any]:
    length = len(action)
    player: dict[str, Any] = {
        "action": torch.tensor([action]),
        "percent": torch.tensor([percent]),
        "x": torch.tensor([x if x is not None else [0.0] * length], dtype=torch.float32),
        "y": torch.tensor([y if y is not None else [0.0] * length], dtype=torch.float32),
        "invulnerable": torch.tensor([invulnerable if invulnerable is not None else [False] * length]),
    }
    if nana_exists is not None:
        player["nana"] = {
            "exists": torch.tensor([nana_exists]),
            "action": torch.tensor([nana_action if nana_action is not None else [14] * length]),
            "percent": torch.tensor([nana_percent if nana_percent is not None else [0] * length]),
        }
    return player


STANDING = 14
DEAD = 3  # DEAD_UP, <= 0xA
HALO = 12  # ON_HALO_DESCENT, > 0xA
LEDGE = 0xFC  # EDGE_CATCHING
FD = 25  # Final Destination (libmelee id), edge x 88.4735488892
BF = 24  # Battlefield, edge x 71.3078536987
UNKNOWN_STAGE = 32  # the dummy env's id: no edge entry -> 100


def test_deaths_are_rising_edges_and_damages_are_positive_increments() -> None:
    action = torch.tensor([STANDING, STANDING, DEAD, DEAD, DEAD, HALO, STANDING, 0, 0])
    assert reward_lib.is_dying(action).tolist() == [False, False, True, True, True, False, False, True, True]
    assert reward_lib.deaths(action).tolist() == [False, True, False, False, False, False, True, False]
    percent = torch.tensor([0, 12, 12, 30, 0, 5])
    damage = reward_lib.damages(percent)
    assert damage.tolist() == [12.0, 0.0, 18.0, 0.0, 5.0] and damage.dtype == torch.float32
    # Batched [B, L] works along the last dimension.
    batched = reward_lib.deaths(torch.stack([action, action]))
    assert batched.shape == (2, 8) and torch.equal(batched[0], batched[1])


def test_reward_hand_sequence_is_zero_sum_and_float32() -> None:
    # p0: takes 12 at t1, 20 at t2, dies at t3 (percent stays), respawns at t6 (percent back to 0).
    p0 = _player(
        [STANDING, STANDING, STANDING, DEAD, DEAD, DEAD, HALO, STANDING],
        [0, 12, 32, 32, 32, 32, 0, 0],
    )
    # p1: takes 30 at t4 and 5 at t7.
    p1 = _player([STANDING] * 8, [0, 0, 0, 0, 30, 30, 30, 35])
    frames = {"p0": p0, "p1": p1}
    rewards = reward_lib.compute_rewards(frames, RewardConfig())
    assert rewards.dtype == torch.float32 and rewards.shape == (1, 7)
    expected = [-0.12, -0.20, -1.0, 0.30, 0.0, 0.0, 0.05]
    assert rewards[0].tolist() == pytest.approx(expected, abs=1e-6)
    swapped = reward_lib.compute_rewards(swap_perspective(frames))
    torch.testing.assert_close(swapped, -rewards, rtol=0.0, atol=0.0)
    # damage_ratio scales only the damage terms.
    halved = reward_lib.compute_rewards(frames, RewardConfig(damage_ratio=0.005))
    assert halved[0].tolist() == pytest.approx([-0.06, -0.10, -1.0, 0.15, 0.0, 0.0, 0.025], abs=1e-6)
    assert reward_lib.ko_diff(frames)[0].tolist() == [0, 0, -1, 0, 0, 0, 0]


def test_nana_term_weighted_masked_by_exists_and_optional() -> None:
    p0 = _player(
        [STANDING] * 5,
        [0, 0, 0, 0, 0],
        nana_exists=[True, True, True, False, False],
        nana_action=[STANDING, DEAD, DEAD, STANDING, DEAD],
        nana_percent=[0, 10, 10, 10, 10],
    )
    p1 = _player([STANDING] * 5, [0] * 5, nana_exists=[False] * 5)
    frames = {"p0": p0, "p1": p1}
    rewards = reward_lib.compute_rewards(frames)
    # t0->t1: nana dies (-0.5) and takes 10 (-0.05); t2->t3 and t3->t4: nana no longer exists -> 0.
    assert rewards[0].tolist() == pytest.approx([-0.55, 0.0, 0.0, 0.0], abs=1e-6)
    assert reward_lib.compute_rewards(frames, RewardConfig(nana_ratio=0.0))[0].tolist() == [0.0] * 4
    assert reward_lib.compute_rewards(frames, RewardConfig(nana_ratio=1.0))[0].tolist() == pytest.approx(
        [-1.1, 0.0, 0.0, 0.0], abs=1e-6
    )
    without_nana = {"p0": _player([STANDING] * 5, [0] * 5), "p1": _player([STANDING] * 5, [0] * 5)}
    assert reward_lib.compute_rewards(without_nana)[0].tolist() == [0.0] * 4
    stats = reward_lib.player_stats(p0)
    assert float(stats["deaths"]) == 0.0
    assert float(stats["nana_deaths"]) == pytest.approx(1 / 4 * reward_lib.FRAMES_PER_MINUTE)
    assert float(stats["nana_damages"]) == pytest.approx(10 / 4 * reward_lib.FRAMES_PER_MINUTE)


def test_zero_at_resets_and_bounds() -> None:
    rewards = torch.tensor([[0.1, -1.0, 0.3, 0.0]])
    is_resetting = torch.tensor([[True, False, True, False, False]])
    zeroed = reward_lib.zero_at_resets(rewards, is_resetting)
    assert zeroed[0].tolist() == pytest.approx([0.1, 0.0, 0.3, 0.0])
    with pytest.raises(ValueError, match="is_resetting must have shape"):
        reward_lib.zero_at_resets(rewards, is_resetting[:, :4])
    # p0 dies, its Nana dies and p1 deals 100 damage in one transition: 2.5 > 2 violates the sanity bound.
    p0 = _player(
        [STANDING, DEAD],
        [0, 100],
        nana_exists=[True, True],
        nana_action=[STANDING, DEAD],
        nana_percent=[0, 0],
    )
    p1 = _player([STANDING, STANDING], [0, 0])
    frames = {"p0": p0, "p1": p1}
    with pytest.raises(ValueError, match=r"rewards must lie in \(-2, 2\)"):
        reward_lib.compute_rewards(frames)
    unchecked = reward_lib.compute_rewards(frames, check_bounds=False)
    assert unchecked[0].tolist() == pytest.approx([-2.5], abs=1e-6)
    # A death plus a full-stock damage stays inside: -1 - 1.0 = -2 is excluded, -1.99 is fine.
    bounded = {"p0": _player([STANDING, DEAD], [0, 99]), "p1": _player([STANDING, STANDING], [0, 0])}
    assert reward_lib.compute_rewards(bounded)[0].tolist() == pytest.approx([-1.99], abs=1e-6)


# ---------------------------------------------------------------------------
# P6a: ledge grabs, stalling, approach (slippi-ai semantics)
# ---------------------------------------------------------------------------


def test_bad_ledge_grabs_rising_edge_with_centre_side_and_invulnerable_exemptions() -> None:
    # p0 hangs on the left ledge (x = -70): grabs at t0->t1 and t3->t4 (rising edges of EDGE_CATCHING).
    p0 = _player([STANDING, LEDGE, LEDGE, STANDING, LEDGE, LEDGE], [0] * 6, x=[-70.0] * 6)
    # The opponent is on the centre side (x = 10) for the first grab and offstage left (x = -90) for the
    # second.
    p1 = _player([STANDING] * 6, [0] * 6, x=[10.0, 10.0, 10.0, -90.0, -90.0, -90.0])
    assert reward_lib.grabbed_ledge(p0["action"])[0].tolist() == [True, False, False, True, False]
    bad = reward_lib.bad_ledge_grabs(p0, p1)
    assert bad.dtype == torch.bool and bad.shape == (1, 5)
    assert bad[0].tolist() == [True, False, False, False, False]  # the second grab: opponent offstage
    frames = {"p0": p0, "p1": p1}
    config = RewardConfig(ledge_grab_penalty=0.02)
    rewards = reward_lib.compute_rewards(frames, config)
    assert rewards[0].tolist() == pytest.approx([-0.02, 0.0, 0.0, 0.0, 0.0], abs=1e-7)
    # Zero-sum: the opponent's perspective gains what p0 loses; weight 0 switches the term off.
    assert reward_lib.compute_rewards(swap_perspective(frames), config)[0].tolist() == pytest.approx(
        [0.02, 0.0, 0.0, 0.0, 0.0], abs=1e-7
    )
    assert reward_lib.compute_rewards(frames, RewardConfig())[0].tolist() == [0.0] * 5
    # An invulnerable opponent (e.g. just respawned) exempts the grab; the source frame decides.
    centre = [10.0] * 6
    exempt = _player(
        [STANDING] * 6, [0] * 6, x=centre, invulnerable=[True, False, False, False, False, False]
    )
    assert reward_lib.bad_ledge_grabs(p0, exempt)[0].tolist() == [False, False, False, True, False]
    later = _player([STANDING] * 6, [0] * 6, x=centre, invulnerable=[False, True, True, True, True, True])
    assert reward_lib.bad_ledge_grabs(p0, later)[0].tolist() == [True, False, False, False, False]
    # On the right ledge (x > 0) the centre is to the left: an opponent at x = -10 is on the centre side.
    right = _player([STANDING, LEDGE], [0, 0], x=[70.0, 70.0])
    inside = _player([STANDING] * 2, [0, 0], x=[-10.0] * 2)
    outside = _player([STANDING] * 2, [0, 0], x=[90.0] * 2)
    assert reward_lib.bad_ledge_grabs(right, inside)[0].tolist() == [True]
    assert reward_lib.bad_ledge_grabs(right, outside)[0].tolist() == [False]


def test_stalling_offstage_destination_frame_stage_map_and_per_frame_stage() -> None:
    p0 = _player(
        [STANDING] * 6,
        [0] * 6,
        x=[95.0, 95.0, 0.0, 0.0, 95.0, 0.0],
        y=[0.0, -30.0, 100.0, 50.0, 0.0, 0.0],
    )
    p1 = _player([STANDING] * 6, [0] * 6)
    fd = torch.full((1, 6), FD)
    amount = reward_lib.amount_offstage(p0, fd)
    dx = 95.0 - 88.4735488892
    assert amount.shape == (1, 6) and amount.dtype == torch.float32
    assert amount[0].tolist() == pytest.approx([dx, math.hypot(dx, 30.0), 40.0, 0.0, dx, 0.0], rel=1e-5)
    stalling = reward_lib.is_stalling_offstage(p0, fd, 20.0)
    assert stalling[0].tolist() == [False, True, True, False, False, False]
    # The penalty is charged on the destination frame of a transition, stalling_penalty / 60 per frame.
    frames = {"p0": p0, "p1": p1, "stage": fd}
    config = RewardConfig(stalling_penalty=0.1, stalling_threshold=20.0)
    rewards = reward_lib.compute_rewards(frames, config)
    assert rewards[0].tolist() == pytest.approx([-0.1 / 60, -0.1 / 60, 0.0, 0.0, 0.0], abs=1e-7)
    assert reward_lib.compute_rewards(swap_perspective(frames), config)[0].tolist() == pytest.approx(
        [0.1 / 60, 0.1 / 60, 0.0, 0.0, 0.0], abs=1e-7
    )
    # A higher threshold turns the 30.7 / 40 amounts off.
    assert (
        reward_lib.compute_rewards(frames, RewardConfig(stalling_penalty=0.1, stalling_threshold=50.0))[
            0
        ].tolist()
        == [0.0] * 5
    )
    # Stage map: Battlefield's edge is at 71.3, so x = 95 is 23.7 offstage even at y = 0; unknown ids -> 100.
    bf = torch.full((1, 6), BF)
    on_bf = reward_lib.is_stalling_offstage(p0, bf, 20.0)
    assert on_bf[0].tolist() == [True, True, True, False, True, False]
    unknown = torch.full((1, 6), UNKNOWN_STAGE)
    far = reward_lib.amount_offstage(p0, unknown)
    assert far[0].tolist() == pytest.approx([0.0, 30.0, 40.0, 0.0, 0.0, 0.0])
    edges = reward_lib.stage_edge_x(torch.tensor([BF, FD, 26, 8, 18, 6, UNKNOWN_STAGE, 0]))
    assert edges.tolist() == pytest.approx(
        [71.3078536987, 88.4735488892, 80.1791534424, 66.2554016113, 90.657852, 58.907848, 100.0, 100.0]
    )
    assert set(reward_lib.EDGE_X) == {24, 25, 26, 8, 18, 6} and reward_lib.UNKNOWN_STAGE_EDGE_X == 100.0
    assert reward_lib.MAX_STALLING_Y == 60.0 and reward_lib.EDGE_CATCHING == 0xFC
    # The stage is read per frame (slippi-ai reads stage[0] per chunk): frame 0 on Battlefield, then FD.
    mixed = torch.tensor([[BF, FD, FD, FD, FD, FD]])
    per_frame = reward_lib.is_stalling_offstage(p0, mixed, 20.0)
    assert per_frame[0].tolist() == [True, True, True, False, False, False]
    # A stalling term without a stage leaf is an error; with weight 0 the stage is not needed.
    with pytest.raises(ValueError, match="stage"):
        reward_lib.compute_rewards({"p0": p0, "p1": p1}, config)
    assert reward_lib.compute_rewards({"p0": p0, "p1": p1}, RewardConfig())[0].tolist() == [0.0] * 5


def test_approaching_factor_unit_direction_respawn_edge_and_zero_distance() -> None:
    # p0 moves (0,0) -> (3,4) towards the opponent at (30,40): 5.0 along the unit direction (0.6, 0.8);
    # stays; dies at t2 and teleports at t2->t3 (masked); stays; moves 10 away at t4->t5.
    p0 = _player(
        [STANDING, STANDING, DEAD, STANDING, STANDING, STANDING],
        [0] * 6,
        x=[0.0, 3.0, 3.0, 100.0, 100.0, 90.0],
        y=[0.0, 4.0, 4.0, 0.0, 0.0, 0.0],
    )
    p1 = _player(
        [STANDING] * 6,
        [0] * 6,
        x=[30.0, 30.0, 30.0, 130.0, 130.0, 130.0],
        y=[40.0, 40.0, 40.0, 0.0, 0.0, 0.0],
    )
    factor = reward_lib.approaching_factor(p0, p1)
    assert factor.shape == (1, 5) and factor.dtype == torch.float32
    assert factor[0].tolist() == pytest.approx([5.0, 0.0, 0.0, 0.0, -10.0], rel=1e-5, abs=1e-6)
    # The opponent's own approach at t2->t3: (100, -40) . unit(-27, -36) = -60 + 32 = -28.
    assert reward_lib.approaching_factor(p1, p0)[0].tolist() == pytest.approx(
        [0.0, 0.0, -28.0, 0.0, 0.0], rel=1e-5, abs=1e-6
    )
    frames = {"p0": p0, "p1": p1}
    config = RewardConfig(approaching_factor=1e-3)
    rewards = reward_lib.compute_rewards(frames, config)
    # p0: +0.005, the death at t1->t2 (-1), +0.028 from the opponent moving away, 0, -0.01.
    assert rewards[0].tolist() == pytest.approx([0.005, -1.0, 0.028, 0.0, -0.01], abs=1e-6)
    torch.testing.assert_close(reward_lib.compute_rewards(swap_perspective(frames), config), -rewards)
    # Coincident positions: the direction is the zero vector (epsilon in the normalisation), no NaN.
    same = _player([STANDING] * 2, [0, 0], x=[5.0, 6.0], y=[5.0, 5.0])
    still = _player([STANDING] * 2, [0, 0], x=[5.0, 5.0], y=[5.0, 5.0])
    assert reward_lib.approaching_factor(same, still)[0].tolist() == [0.0]
    assert reward_lib.normalize(torch.zeros(2)).tolist() == [0.0, 0.0]
    assert reward_lib.normalize(torch.tensor([3.0, 4.0])).tolist() == pytest.approx([0.6, 0.8], rel=1e-5)


def test_antisymmetry_under_swap_stats_names_and_config_validation() -> None:
    frames = random_valid_frames((2, 9), torch.Generator().manual_seed(11))
    config = RewardConfig(
        damage_ratio=0.01,
        nana_ratio=0.5,
        ledge_grab_penalty=0.02,
        approaching_factor=1e-3,
        stalling_penalty=0.1,
        stalling_threshold=50.0,
    )
    rewards = reward_lib.compute_rewards(frames, config, check_bounds=False)
    swapped = reward_lib.compute_rewards(swap_perspective(frames), config, check_bounds=False)
    assert rewards.shape == (2, 8) and torch.isfinite(rewards).all()
    assert torch.equal(swapped, -rewards)  # every term enters for both players: exactly zero-sum
    stats = reward_lib.player_stats(frames["p0"], frames["p1"], frames["stage"], stalling_threshold=50.0)
    assert set(stats) == {
        "deaths",
        "damages",
        "ledge_grabs",
        "stalling",
        "approaching_factor",
        "nana_deaths",
        "nana_damages",
    }
    bad = reward_lib.bad_ledge_grabs(frames["p0"], frames["p1"]).to(torch.float32)
    assert float(stats["ledge_grabs"]) == pytest.approx(float(bad.mean()) * reward_lib.FRAMES_PER_MINUTE)
    stalling = reward_lib.is_stalling_offstage(frames["p0"], frames["stage"], 50.0).to(torch.float32)
    assert float(stats["stalling"]) == pytest.approx(float(stalling.mean()))  # a fraction of all frames
    assert float(stats["approaching_factor"]) == pytest.approx(
        float(reward_lib.approaching_factor(frames["p0"], frames["p1"]).mean())
    )
    # Without an opponent / stage the P1 subset is returned (the dummy-env callers).
    assert set(reward_lib.player_stats(frames["p0"])) == {"deaths", "damages", "nana_deaths", "nana_damages"}
    assert "ledge_grabs" in reward_lib.player_stats(frames["p0"], frames["p1"])
    assert "stalling" not in reward_lib.player_stats(frames["p0"], frames["p1"])
    assert "stalling" in reward_lib.player_stats(frames["p0"], stage=frames["stage"])
    # The launch values of slippi-ai's RL scripts are accepted; defaults keep every P6 term at 0.
    assert RewardConfig().ledge_grab_penalty == 0.0 and RewardConfig().stalling_penalty == 0.0
    assert RewardConfig().approaching_factor == 0.0 and RewardConfig().stalling_threshold == 20.0
    assert reward_lib.DEFAULT_STALLING_THRESHOLD == 20.0
    launch = RewardConfig(
        ledge_grab_penalty=0.02, stalling_penalty=0.1, stalling_threshold=50.0, approaching_factor=1e-3
    )
    assert launch.stalling_threshold == 50.0
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(ledge_grab_penalty=-0.02)
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(stalling_penalty=-0.1)
    with pytest.raises(ValueError, match="stalling_threshold"):
        RewardConfig(stalling_threshold=0.0)
    with pytest.raises(ValueError, match="finite"):
        RewardConfig(approaching_factor=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        RewardConfig(damage_ratio=float("inf"))


def test_reward_config_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(damage_ratio=-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(nana_ratio=-0.5)
    assert RewardConfig().damage_ratio == 0.01 and RewardConfig().nana_ratio == 0.5
