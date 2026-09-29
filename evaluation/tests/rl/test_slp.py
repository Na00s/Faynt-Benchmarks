"""P11: reading a Slippi ``.slp`` replay (``melee_rl.slp``).

Every offset the parser claims is pinned here by synthetic replays this module writes byte by byte:
the game-start block (version, stage, RNG seed, per-port character / type / stock start / costume /
CPU level), the post-frame stream (stocks, percent, dying action states) and ``GAME_END``.  A replay
Dolphin is still writing carries ``raw`` length 0, which has its own test -- reading the seed of a
live match depends on it.

The last test reads whatever real replays sit under ``data/videos`` (git-ignored, so it skips when
they are absent).  Those files are what showed on 28 Aug 2026 that our matches ran in the no-stock
endless mode; the test only asserts self-consistency, so it keeps passing once real matches land
there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from melee_rl import slp
from tests.rl._slp_fixtures import game_end, game_start, post_frame, write_replay


def _data_videos() -> Path | None:
    """Where session 24 downloaded the recorded clips (git-ignored, so it may not be here at all)."""

    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3]):
        candidate = parent / "data" / "videos"
        if candidate.is_dir():
            return candidate
    return None


DATA_VIDEOS = _data_videos()


# ---------------------------------------------------------------------------
# the game-start block
# ---------------------------------------------------------------------------


def test_read_game_start_reads_the_seed_and_the_rule_set(tmp_path: Path) -> None:
    path = write_replay(tmp_path / "game.slp", [two_fox_start()])

    start = slp.read_game_start(path)

    assert start.seed == 0xD831FED0
    assert start.stage == 32
    assert start.slp_version == (3, 19, 0)
    assert [player.port for player in start.players] == [1, 2]  # empty ports are dropped
    policy, cpu = start.players
    assert (policy.character, policy.player_type, policy.stocks, policy.costume) == (0x02, 0, 4, 1)
    assert policy.cpu_level == 0  # a bot port's cpu_level byte is meaningless: libmelee zeroes it too
    assert (cpu.character, cpu.player_type, cpu.stocks, cpu.costume) == (0x02, 1, 4, 0)
    assert cpu.cpu_level == 9
    assert start.stocks == 4


def test_read_game_start_on_a_replay_dolphin_is_still_writing(tmp_path: Path) -> None:
    path = write_replay(
        tmp_path / "live.slp",
        [two_fox_start(seed=0x0000BEEF), post_frame(frame=-123, port=1)],
        declare_length=False,
        metadata=False,
    )

    assert slp.read_game_start(path).seed == 0x0000BEEF


def test_read_game_start_rejects_a_file_that_is_not_a_replay(tmp_path: Path) -> None:
    path = tmp_path / "not.slp"
    path.write_bytes(b"nothing to see here")

    with pytest.raises(ValueError, match="raw"):
        slp.read_game_start(path)


# ---------------------------------------------------------------------------
# the whole replay
# ---------------------------------------------------------------------------


def two_fox_start(seed: int = 0xD831FED0) -> bytes:
    """Our own recording setup: Fox on both ports, port 2 a level-9 CPU, four stocks each."""

    return game_start(seed=seed)


def _match_events() -> list[bytes]:
    """Port 1 takes 12 % and dies once; port 2 takes 30 % over two hits and dies twice, losing."""

    events = [two_fox_start()]
    script = [
        # frame, p1 (action, percent, stocks), p2 (action, percent, stocks)
        (-123, (0x0E, 0.0, 4), (0x0E, 0.0, 4)),
        (0, (0x0E, 0.0, 4), (0x0E, 0.0, 4)),
        (1, (0x0E, 12.0, 4), (0x0E, 20.0, 4)),
        (2, (0x0E, 12.0, 4), (0x00, 20.0, 4)),  # port 2 dying (action <= 0x0A)
        (3, (0x0E, 12.0, 4), (0x01, 0.0, 3)),  # still dying: one death, not two
        (4, (0x00, 12.0, 4), (0x0E, 0.0, 3)),  # port 1 dying
        (5, (0x0E, 0.0, 3), (0x0E, 30.0, 3)),
        (6, (0x0E, 0.0, 3), (0x04, 30.0, 3)),  # port 2 dying again
        (7, (0x0E, 0.0, 3), (0x0E, 0.0, 2)),
    ]
    for frame, (action1, percent1, stocks1), (action2, percent2, stocks2) in script:
        events.append(post_frame(frame=frame, port=1, action=action1, percent=percent1, stocks=stocks1))
        events.append(post_frame(frame=frame, port=2, action=action2, percent=percent2, stocks=stocks2))
    return events


def test_summarize_replay_counts_deaths_damage_and_stocks(tmp_path: Path) -> None:
    path = write_replay(tmp_path / "match.slp", [*_match_events(), game_end(method=slp.END_GAME)])

    summary = slp.summarize_replay(path)

    assert summary.frames == 9
    assert (summary.first_frame, summary.last_frame) == (-123, 7)
    first, second = summary.players
    assert (first.port, second.port) == (1, 2)
    assert (first.deaths, second.deaths) == (1, 2)
    assert (first.stocks_start, first.stocks_end) == (4, 3)
    assert (second.stocks_start, second.stocks_end) == (4, 2)
    assert first.damage_taken == pytest.approx(12.0)
    assert second.damage_taken == pytest.approx(50.0)  # 20 before the first death, 30 after it
    assert first.damage_dealt == pytest.approx(second.damage_taken)
    assert summary.finished and summary.end_method == slp.END_GAME
    assert summary.end_method_name == "GAME!"
    assert summary.winner == 1
    assert summary.seed == 0xD831FED0


def test_summarize_replay_reports_a_match_that_never_ended(tmp_path: Path) -> None:
    path = write_replay(tmp_path / "endless.slp", _match_events())

    summary = slp.summarize_replay(path)

    assert not summary.finished
    assert summary.end_method is None
    assert summary.end_method_name == "unfinished"
    assert summary.winner is None
    assert summary.players[1].deaths == 2  # the frame stream is read either way


def test_summarize_replay_ignores_follower_rows(tmp_path: Path) -> None:
    """Nana writes her own post-frame rows (``is_follower``); counting them would double a death."""

    events = [
        two_fox_start(),
        post_frame(frame=0, port=1, action=0x0E, percent=0.0, stocks=4),
        post_frame(frame=0, port=1, action=0x00, percent=99.0, stocks=1, follower=True),
        post_frame(frame=1, port=1, action=0x0E, percent=10.0, stocks=4),
    ]
    path = write_replay(tmp_path / "nana.slp", events)

    summary = slp.summarize_replay(path)

    played, silent = summary.players
    assert played.port == 1
    assert played.deaths == 0  # Nana's dying row is not port 1 dying
    assert played.stocks_end == 4  # nor is her stock count port 1's
    assert played.damage_taken == pytest.approx(10.0)
    assert (silent.port, silent.stocks_end, silent.deaths) == (2, 4, 0)  # a port with no rows at all


def test_summarize_replay_survives_a_truncated_final_event(tmp_path: Path) -> None:
    """A killed Dolphin leaves a half-written event; the frames before it must still be readable."""

    path = write_replay(tmp_path / "cut.slp", _match_events(), declare_length=False, metadata=False)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 100])  # frame 7 loses both of its rows

    summary = slp.summarize_replay(path)

    assert summary.frames == 8
    assert not summary.finished


def test_format_summary_is_one_line_per_replay(tmp_path: Path) -> None:
    path = write_replay(tmp_path / "match.slp", [*_match_events(), game_end(method=slp.END_GAME)])

    line = slp.format_summary(slp.summarize_replay(path))

    assert "match.slp" in line
    assert "seed 0xd831fed0" in line
    assert "GAME!" in line
    assert "4->3" in line and "4->2" in line


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(DATA_VIDEOS is None, reason="no downloaded replays under data/videos")
def test_recorded_replays_parse_and_are_self_consistent() -> None:
    assert DATA_VIDEOS is not None
    replays = sorted(DATA_VIDEOS.rglob("*.slp"))
    if not replays:
        pytest.skip("data/videos holds no .slp")

    for path in replays:
        summary = slp.summarize_replay(path)
        assert summary.frames > 0
        assert summary.first_frame == slp.INITIAL_FRAME
        assert len(summary.players) == 2
        for player in summary.players:
            assert 0 <= player.stocks_end <= player.stocks_start
            assert player.deaths >= 0
            assert player.damage_taken >= 0.0
        assert summary.end_method in (None, slp.END_TIME, slp.END_GAME, slp.END_NO_CONTEST)
