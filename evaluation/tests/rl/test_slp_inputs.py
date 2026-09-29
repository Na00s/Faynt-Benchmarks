"""P26: the controller stream of a replay, and the fidelity instruments built on it.

A win rate hides an input-path defect; the P25 campaign only saw one because the *four-stock rate*
was published next to it.  These are the finer instruments (``melee_rl.slp_inputs``):

* **waveshine chains.**  Fox's shine is the most input-timing-sensitive thing in the game -- a
  jump-cancelled shine has a 3-frame window -- so the length of an uninterrupted chain reads input
  fidelity directly, without depending on who won.  Every offset the pre-frame parser claims is
  pinned here by a replay written byte by byte, and the chain rule (onsets, not held frames; a gap
  ends a chain) has its own tests.
* **uncontested deaths.**  A death with no damage taken in the preceding window is a self-destruct
  or a failed recovery, not a punish -- the direct measurement session 52 could only infer.

The last test reads real replays if any are lying about, and only asserts self-consistency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from melee_rl import slp, slp_inputs
from tests.rl._slp_fixtures import game_end, game_start, post_frame, pre_frame, write_replay


def _replay(tmp_path: Path, events: list[bytes], name: str = "match.slp") -> Path:
    return write_replay(tmp_path / name, [game_start(seed=1), *events, game_end()])


# ---------------------------------------------------------------------------
# the pre-frame stream
# ---------------------------------------------------------------------------


def test_read_frames_pins_the_pre_frame_offsets(tmp_path: Path) -> None:
    """Buttons, sticks and trigger come off libmelee's own offsets (console.py:1145-1212)."""

    path = _replay(
        tmp_path,
        [
            pre_frame(frame=-123, port=1, buttons=slp_inputs.BUTTON_B, main_stick=(0.0, -0.5), trigger=0.25),
            post_frame(frame=-123, port=1, action=0x0E),
            pre_frame(frame=-122, port=1, buttons=slp_inputs.BUTTON_X | slp_inputs.BUTTON_A),
            post_frame(frame=-122, port=1, action=0x18),
        ],
    )
    frames = slp_inputs.read_frames(path)
    port = frames.ports[1]
    assert port.frames == (-123, -122)
    assert port.buttons == (slp_inputs.BUTTON_B, slp_inputs.BUTTON_X | slp_inputs.BUTTON_A)
    assert port.actions == (0x0E, 0x18)
    assert port.main_stick[0] == pytest.approx((0.0, -0.5))
    assert port.trigger[0] == pytest.approx(0.25)


def test_read_frames_ignores_follower_rows(tmp_path: Path) -> None:
    """Nana shares her leader's port and would otherwise double every frame."""

    path = _replay(
        tmp_path,
        [
            pre_frame(frame=-123, port=1, buttons=slp_inputs.BUTTON_B),
            pre_frame(frame=-123, port=1, buttons=0, follower=True),
            post_frame(frame=-123, port=1, action=0x0E),
            post_frame(frame=-123, port=1, action=0x0E, follower=True),
        ],
    )
    assert slp_inputs.read_frames(path).ports[1].frames == (-123,)


def test_read_frames_pairs_a_post_frame_that_has_no_pre_frame(tmp_path: Path) -> None:
    """A CPU port has post frames only; its actions still have to be readable."""

    path = _replay(tmp_path, [post_frame(frame=-123, port=2, action=0x0E)])
    port = slp_inputs.read_frames(path).ports[2]
    assert port.frames == (-123,)
    assert port.actions == (0x0E,)
    assert port.buttons == (0,)


# ---------------------------------------------------------------------------
# waveshine chains
# ---------------------------------------------------------------------------


def _shine_run(port: int, onsets: list[int], *, hold: int = 3) -> list[bytes]:
    """Post frames where the action enters the grounded shine on each frame in ``onsets``."""

    events: list[bytes] = []
    shine_frames = {frame + offset: offset for frame in onsets for offset in range(hold)}
    for frame in range(min(onsets) - 2, max(onsets) + hold + 2):
        offset = shine_frames.get(frame)
        if offset is None:
            action = 0x0E  # standing
        else:
            action = slp_inputs.SHINE_GROUND_START if offset == 0 else slp_inputs.SHINE_GROUND
        events.append(post_frame(frame=frame, port=port, action=action))
    return events


def test_shine_chains_counts_onsets_not_held_frames(tmp_path: Path) -> None:
    """A shine held for three frames is one shine, not three."""

    path = _replay(tmp_path, _shine_run(1, [0], hold=8))
    chains = slp_inputs.shine_chains(slp_inputs.read_frames(path).ports[1])
    assert [chain.length for chain in chains] == [1]


def test_shine_chains_links_shines_inside_the_gap(tmp_path: Path) -> None:
    """A waveshine cycle is ~12 frames; four of them in a row are one chain of four."""

    path = _replay(tmp_path, _shine_run(1, [0, 12, 24, 36]))
    chains = slp_inputs.shine_chains(slp_inputs.read_frames(path).ports[1], gap_frames=25)
    assert [chain.length for chain in chains] == [4]
    assert chains[0].start_frame == 0
    assert chains[0].span == 36


def test_shine_chains_breaks_on_a_gap_longer_than_the_threshold(tmp_path: Path) -> None:
    path = _replay(tmp_path, _shine_run(1, [0, 12, 100, 112, 124]))
    chains = slp_inputs.shine_chains(slp_inputs.read_frames(path).ports[1], gap_frames=25)
    assert [chain.length for chain in chains] == [2, 3]


def test_shine_chains_counts_the_air_shine_too(tmp_path: Path) -> None:
    """A waveshine alternates grounded and aerial shines depending on the jump cancel."""

    events = [
        post_frame(frame=0, port=1, action=slp_inputs.SHINE_GROUND_START),
        post_frame(frame=1, port=1, action=0x0E),
        post_frame(frame=10, port=1, action=slp_inputs.SHINE_AIR),
        post_frame(frame=11, port=1, action=0x0E),
    ]
    chains = slp_inputs.shine_chains(slp_inputs.read_frames(_replay(tmp_path, events)).ports[1])
    assert [chain.length for chain in chains] == [2]


def test_shine_chains_rejects_a_non_positive_gap(tmp_path: Path) -> None:
    path = _replay(tmp_path, _shine_run(1, [0]))
    with pytest.raises(ValueError, match="gap_frames"):
        slp_inputs.shine_chains(slp_inputs.read_frames(path).ports[1], gap_frames=0)


# ---------------------------------------------------------------------------
# uncontested deaths
# ---------------------------------------------------------------------------


def _life(port: int, frames: int, *, damage_at: dict[int, float], die_at: int) -> list[bytes]:
    events: list[bytes] = []
    percent = 0.0
    for frame in range(frames):
        percent = damage_at.get(frame, percent)
        action = 0x00 if frame == die_at else 0x0E
        events.append(post_frame(frame=frame, port=port, action=action, percent=percent))
    return events


def test_uncontested_death_is_a_death_with_no_recent_damage(tmp_path: Path) -> None:
    path = _replay(tmp_path, _life(1, 400, damage_at={10: 40.0}, die_at=300))
    port = slp_inputs.read_frames(path).ports[1]
    assert slp_inputs.deaths(port) == 1
    assert slp_inputs.uncontested_deaths(port, window_frames=120) == 1


def test_a_death_shortly_after_taking_damage_is_contested(tmp_path: Path) -> None:
    path = _replay(tmp_path, _life(1, 400, damage_at={250: 40.0}, die_at=300))
    port = slp_inputs.read_frames(path).ports[1]
    assert slp_inputs.deaths(port) == 1
    assert slp_inputs.uncontested_deaths(port, window_frames=120) == 0


def test_uncontested_deaths_rejects_a_negative_window(tmp_path: Path) -> None:
    path = _replay(tmp_path, _life(1, 20, damage_at={}, die_at=10))
    with pytest.raises(ValueError, match="window_frames"):
        slp_inputs.uncontested_deaths(slp_inputs.read_frames(path).ports[1], window_frames=-1)


# ---------------------------------------------------------------------------
# how busy the controller is
# ---------------------------------------------------------------------------


def test_button_change_rate_is_the_share_of_frames_the_buttons_moved(tmp_path: Path) -> None:
    events = []
    for index, buttons in enumerate([0, 0, slp_inputs.BUTTON_B, slp_inputs.BUTTON_B, 0]):
        events.append(pre_frame(frame=index, port=1, buttons=buttons))
        events.append(post_frame(frame=index, port=1))
    port = slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]
    assert slp_inputs.button_change_rate(port) == pytest.approx(2 / 4)


def test_button_change_rate_of_a_port_with_no_inputs_is_zero(tmp_path: Path) -> None:
    path = _replay(tmp_path, [post_frame(frame=0, port=2), post_frame(frame=1, port=2)])
    assert slp_inputs.button_change_rate(slp_inputs.read_frames(path).ports[2]) == 0.0


# ---------------------------------------------------------------------------
# the whole replay
# ---------------------------------------------------------------------------


def test_replay_metrics_reports_every_port(tmp_path: Path) -> None:
    events = [*_shine_run(1, [0, 12, 24]), post_frame(frame=0, port=2, action=0x0E)]
    metrics = slp_inputs.replay_metrics(_replay(tmp_path, events))
    assert set(metrics.ports) == {1, 2}
    assert metrics.ports[1].longest_shine_chain == 3
    assert metrics.ports[1].shines == 3
    assert metrics.ports[1].mean_shine_chain == pytest.approx(3.0)
    assert metrics.ports[2].shines == 0
    assert metrics.ports[2].longest_shine_chain == 0


def _real_replays() -> list[Path]:
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3]):
        candidate = parent / "data" / "videos"
        if candidate.is_dir():
            return sorted(candidate.rglob("*.slp"))[:3]
    return []


@pytest.mark.skipif(not _real_replays(), reason="no recorded replays present")
def test_real_replays_are_self_consistent() -> None:
    for path in _real_replays():
        metrics = slp_inputs.replay_metrics(path)
        summary = slp.summarize_replay(path)
        assert set(metrics.ports) == {player.port for player in summary.start.players}
        for side in metrics.ports.values():
            assert side.frames == summary.frames
            assert 0.0 <= side.button_change_rate <= 1.0
            assert side.longest_shine_chain <= side.shines
            assert side.deaths >= side.uncontested_deaths


# ---------------------------------------------------------------------------
# the round trip: how long an agent's answer takes to reach the game
# ---------------------------------------------------------------------------


def _shine_then_jump(port: int, *, jump_after: int) -> list[bytes]:
    """A grounded shine whose ``action_frame`` climbs 1, 2, 3, ... and a Y press ``jump_after`` frames
    after the first frame that satisfies SmashBot's ``action_frame >= 3`` rule."""

    events: list[bytes] = []
    trigger = slp_inputs.SHINE_JUMP_CANCEL_FRAME  # frame 3 is the first with action_frame >= 3
    for frame in range(0, trigger + jump_after + 3):
        action_frame = frame if 1 <= frame <= 12 else 0
        action = slp_inputs.SHINE_GROUND if 1 <= frame <= 12 else 0x0E
        pressed = slp_inputs.BUTTON_Y if frame == trigger + jump_after else 0
        events.append(pre_frame(frame=frame, port=port, buttons=pressed))
        events.append(post_frame(frame=frame, port=port, action=action, action_frame=action_frame))
    return events


def test_read_frames_reads_the_action_frame(tmp_path: Path) -> None:
    path = _replay(tmp_path, [post_frame(frame=0, port=1, action=0x169, action_frame=7)])
    assert slp_inputs.read_frames(path).ports[1].action_frame == (7,)


@pytest.mark.parametrize("latency", [1, 2, 3])
def test_jump_cancel_latency_is_the_frames_from_the_trigger_to_the_press(
    tmp_path: Path, latency: int
) -> None:
    """SmashBot presses Y on the first shine gamestate with ``action_frame >= 3``
    (``Chains/waveshine.py:107-110``), so this measures observation -> decision -> emulator."""

    path = _replay(tmp_path, _shine_then_jump(1, jump_after=latency))
    assert slp_inputs.jump_cancel_latencies(slp_inputs.read_frames(path).ports[1]) == (latency,)


def test_jump_cancel_latency_ignores_a_shine_that_was_never_jumped(tmp_path: Path) -> None:
    events = [
        post_frame(frame=frame, port=1, action=slp_inputs.SHINE_GROUND, action_frame=frame)
        for frame in range(1, 10)
    ]
    assert slp_inputs.jump_cancel_latencies(slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]) == ()


def test_jump_cancel_latency_ignores_a_y_press_before_the_rule_fires(tmp_path: Path) -> None:
    """A Y on shine frame 1 is a different chain; counting it would report a latency of 0."""

    events: list[bytes] = []
    for frame in range(1, 8):
        events.append(pre_frame(frame=frame, port=1, buttons=slp_inputs.BUTTON_Y if frame == 1 else 0))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.SHINE_GROUND, action_frame=frame))
    assert slp_inputs.jump_cancel_latencies(slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]) == ()


# ---------------------------------------------------------------------------
# airdodges: was it a wavedash or a glide off the stage
# ---------------------------------------------------------------------------


def test_airdodges_are_the_rising_edges_of_the_shoulder(tmp_path: Path) -> None:
    """L held for five frames is one airdodge, and the stick is read on the frame it was pressed."""

    events: list[bytes] = []
    for frame in range(6):
        pressed = slp_inputs.BUTTON_L if 1 <= frame <= 5 else 0
        stick = (-0.95, -0.30) if frame == 1 else (0.0, 0.0)
        action = slp_inputs.AIRDODGE if frame >= 2 else 0x0E
        events.append(pre_frame(frame=frame, port=1, buttons=pressed, main_stick=stick))
        events.append(post_frame(frame=frame, port=1, action=action))
    dodges = slp_inputs.airdodges(slp_inputs.read_frames(_replay(tmp_path, events)).ports[1])
    assert [(d.frame, d.x, d.y) for d in dodges] == [(1, pytest.approx(-0.95), pytest.approx(-0.30))]


def test_a_shield_press_is_not_an_airdodge(tmp_path: Path) -> None:
    """L on the ground is the shield button; a third of SmashBot's L presses never leave the stage."""

    events = [
        pre_frame(frame=0, port=1, buttons=slp_inputs.BUTTON_L, main_stick=(0.0, 0.0)),
        post_frame(frame=0, port=1, action=0xB3),  # SHIELD_REFLECT
        pre_frame(frame=1, port=1, buttons=slp_inputs.BUTTON_L),
        post_frame(frame=1, port=1, action=0xB3),
    ]
    assert slp_inputs.airdodges(slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]) == ()


def test_a_flat_airdodge_is_one_with_no_usable_downward_component(tmp_path: Path) -> None:
    """SmashBot always asks for ``y = 0.35`` (-0.30 in replay units); anything inside Melee's
    0.2875 deadzone is a horizontal airdodge, which off a ledge is a self-destruct."""

    def dodge(frame: int, y: float) -> list[bytes]:
        return [
            pre_frame(frame=frame, port=1, buttons=slp_inputs.BUTTON_L, main_stick=(-0.95, y)),
            post_frame(frame=frame, port=1, action=slp_inputs.AIRDODGE),
            pre_frame(frame=frame + 1, port=1, buttons=0),
            post_frame(frame=frame + 1, port=1, action=slp_inputs.AIRDODGE),
        ]

    events = dodge(0, -0.30) + dodge(2, 0.0) + dodge(4, -0.1) + dodge(6, -0.7)
    port = slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]
    assert len(slp_inputs.airdodges(port)) == 4
    assert slp_inputs.flat_airdodges(port) == 2
    assert slp_inputs.flat_airdodge_rate(port) == pytest.approx(0.5)


def test_flat_airdodge_rate_of_a_port_that_never_airdodged_is_zero(tmp_path: Path) -> None:
    path = _replay(tmp_path, [post_frame(frame=0, port=2)])
    assert slp_inputs.flat_airdodge_rate(slp_inputs.read_frames(path).ports[2]) == 0.0


def test_port_metrics_carries_the_two_new_instruments(tmp_path: Path) -> None:
    events = _shine_then_jump(1, jump_after=1)
    events += [
        pre_frame(frame=100, port=1, buttons=slp_inputs.BUTTON_L, main_stick=(-0.95, 0.0)),
        post_frame(frame=100, port=1, action=slp_inputs.AIRDODGE),
    ]
    side = slp_inputs.replay_metrics(_replay(tmp_path, events)).ports[1]
    assert side.jump_cancel_latencies == (1,)
    assert side.airdodges == 1
    assert side.flat_airdodges == 1


def test_jump_cancel_latency_only_counts_the_two_states_smashbots_rule_names(tmp_path: Path) -> None:
    """``DOWN_B_STUN`` and the aerial shine are outside the rule; counting them reads 2.37 frames of
    round trip on a delay-0 arm that actually runs at 1.00."""

    events: list[bytes] = []
    for frame in range(1, 8):
        events.append(pre_frame(frame=frame, port=1, buttons=slp_inputs.BUTTON_Y if frame == 7 else 0))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.SHINE_STUN, action_frame=frame))
    port = slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]
    assert slp_inputs.jump_cancel_latencies(port) == ()
    assert slp_inputs.SHINE_STUN not in slp_inputs.SHINE_GROUNDED


def test_a_wavedash_that_lands_never_shows_the_airdodge_state(tmp_path: Path) -> None:
    """Jumpsquat -> shoulder -> ``LANDING_SPECIAL`` on the spot is a wavedash, and it is invisible to a
    detector that only looks for ``Action.AIRDODGE`` -- which is 96 % of them once the angle survives."""

    events: list[bytes] = []
    for frame in range(4):
        events.append(pre_frame(frame=frame, port=1, buttons=0))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.JUMPSQUAT))
    for frame in range(4, 9):
        pressed = slp_inputs.BUTTON_L if frame == 4 else 0
        events.append(pre_frame(frame=frame, port=1, buttons=pressed, main_stick=(-0.95, -0.2875)))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.LANDING_SPECIAL))
    port = slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]
    dodges = slp_inputs.airdodges(port)
    assert [(d.frame, d.landed, d.flat) for d in dodges] == [(4, True, False)]
    assert slp_inputs.airdodge_landing_rate(port) == pytest.approx(1.0)


def test_a_glide_is_an_airdodge_that_did_not_land(tmp_path: Path) -> None:
    events: list[bytes] = []
    for frame in range(3):
        events.append(pre_frame(frame=frame, port=1, buttons=0))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.JUMPSQUAT))
    for frame in range(3, 20):
        pressed = slp_inputs.BUTTON_L if frame == 3 else 0
        events.append(pre_frame(frame=frame, port=1, buttons=pressed, main_stick=(-0.95, 0.0)))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.AIRDODGE))
    port = slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]
    assert [(d.landed, d.flat) for d in slp_inputs.airdodges(port)] == [(False, True)]
    assert slp_inputs.airdodge_landing_rate(port) == 0.0


def test_a_shine_landing_is_not_counted_as_a_wavedash(tmp_path: Path) -> None:
    """``LANDING_SPECIAL`` also ends an aerial shine; without the jumpsquat check every one counts."""

    events: list[bytes] = []
    for frame in range(3):
        events.append(pre_frame(frame=frame, port=1, buttons=0))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.SHINE_AIR))
    for frame in range(3, 8):
        pressed = slp_inputs.BUTTON_L if frame == 3 else 0
        events.append(pre_frame(frame=frame, port=1, buttons=pressed))
        events.append(post_frame(frame=frame, port=1, action=slp_inputs.LANDING_SPECIAL))
    assert slp_inputs.airdodges(slp_inputs.read_frames(_replay(tmp_path, events)).ports[1]) == ()
