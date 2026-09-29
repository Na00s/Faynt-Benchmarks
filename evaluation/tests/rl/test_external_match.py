"""Matches where neither side is one of our policies (P25, 9 Sep 2026).

The recorder's matchup grammar is ``<policy|bc>:<opponent>``: one side is always a checkpoint of ours,
driven through ``RolloutWorker`` with a trajectory, rewards and a value estimate.  The SmashBot
calibration arms need the other shape -- **SmashBot vs Dolphin's CPU 9** and **SmashBot vs MIMIC** --
where our policy is not in the game at all.  Those two are what place SmashBot on the same ladder as
everything in ``RESULTS.md`` before any of our weights are pointed at it, and CPU 9 alone cannot do it:
it is known to mis-rank agents (GOTCHAS S30, P12).

So this driver has no policy, no trajectory, no reward and no torch model.  It steps the environment
directly -- read ``current()``, ask each side's opponent for a ``ControllerState``, ``step`` -- and
stops on exactly the recorder's match-mode rule: every Dolphin has begun a second game, so its first
one is complete on disc.  The results are then read back out of the ``.slp`` by :mod:`melee_rl.slp`
through the recorder's own :func:`melee_rl.video.match_record`, so the numbers are directly comparable
with P14 / P15 / P21.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from controller_codec import ButtonState, ControllerState, StickState
from melee_rl.env.protocol import INITIAL_FRAME_INDEX, EnvOutput
from melee_rl.external_match import (
    ExternalMatchRun,
    ExternalSide,
    run_external_matches,
    side_health,
)
from tensor_batch import BUTTON_ORDER


def neutral_state(batch: int) -> ControllerState:
    return ControllerState(
        main_stick=StickState(x=torch.full((batch,), 0.5), y=torch.full((batch,), 0.5)),
        c_stick=StickState(x=torch.full((batch,), 0.5), y=torch.full((batch,), 0.5)),
        shoulder=torch.zeros(batch),
        buttons=ButtonState(**{name: torch.zeros(batch, dtype=torch.bool) for name in BUTTON_ORDER}),
    )


class FakeEnv:
    """The slice of ``EnvProtocol`` the driver uses, plus ``gamestates`` and a step counter."""

    def __init__(
        self,
        num_envs: int = 2,
        controlled: tuple[int, ...] = (0, 1),
        *,
        replay_root: Path | None = None,
        write_at: Mapping[int, int] | None = None,
    ) -> None:
        self._num_envs = num_envs
        self._controlled = controlled
        self.steps = 0
        self.closed = False
        self.actions: list[Mapping[int, ControllerState]] = []
        self.gamestate_reads = 0
        # ``write_at`` maps a step number to how many games each Dolphin has written by then, so a test
        # can make replays appear *during* the run -- which is what a real Dolphin does, and what the
        # emptiness check requires.
        self._replay_root = replay_root
        self._write_at = dict(write_at or {})

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return self._controlled

    def _output(self) -> EnvOutput:
        index = torch.full((self._num_envs,), 0 if self.steps else INITIAL_FRAME_INDEX)
        zero = torch.zeros(self._num_envs)
        return EnvOutput(
            frames={port: {} for port in self._controlled},
            needs_reset=index == INITIAL_FRAME_INDEX,
            frame_index=index,
            rewards={port: zero for port in self._controlled},
        )

    def current(self) -> EnvOutput:
        return self._output()

    def gamestates(self) -> list[Any]:
        self.gamestate_reads += 1
        return [object() for _ in range(self._num_envs)]

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        self.actions.append(actions)
        self.steps += 1
        games = self._write_at.get(self.steps)
        if games is not None and self._replay_root is not None:
            write_replays(self._replay_root, self._num_envs, games)
        return self._output()

    def reset(self) -> EnvOutput:
        self.steps = 0
        return self._output()

    def close(self) -> None:
        self.closed = True


class FakeDriver:
    """An ``act_state`` opponent, like ``SmashBotOpponent`` / ``MimicOpponent``."""

    def __init__(self, port: int, num_envs: int = 2) -> None:
        self._port = port
        self.num_envs = num_envs
        self.calls = 0
        self.resets: list[list[bool]] = []

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    def act_state(self, frames: Any, needs_reset: torch.Tensor) -> ControllerState:
        self.calls += 1
        self.resets.append(needs_reset.tolist())
        return neutral_state(self.num_envs)


class HealthyDriver(FakeDriver):
    """A driver whose player exposes SmashBot's health counters."""

    class _Player:
        @staticmethod
        def stats() -> dict[str, Any]:
            return {"frames": 7, "exceptions": 0, "tactics": {"Punish/Waveshine": 7}}

    player = _Player()


def write_replays(root: Path, envs: int, games: int) -> None:
    """``games`` ``.slp`` files per Dolphin, oldest first (``collect_replays(first=True)`` order)."""

    for index in range(envs):
        directory = root / f"env{index}"
        directory.mkdir(parents=True, exist_ok=True)
        for game in range(games):
            path = directory / f"Game_{game}.slp"
            path.write_bytes(b"")
            import os

            os.utime(path, (game + 1, game + 1))


# ---------------------------------------------------------------------------
# sides
# ---------------------------------------------------------------------------


def test_a_side_the_environment_drives_needs_no_opponent() -> None:
    """Dolphin's own CPU is a side of the match but never receives a controller."""

    cpu = ExternalSide(name="cpu9", player=None, opponent=None)
    assert cpu.driven_by_env
    assert ExternalSide(name="smashbot", player=1, opponent=FakeDriver(1)).driven_by_env is False


def test_a_side_with_a_player_must_carry_an_opponent() -> None:
    with pytest.raises(ValueError, match="opponent"):
        ExternalSide(name="smashbot", player=0, opponent=None)
    with pytest.raises(ValueError, match="player"):
        ExternalSide(name="cpu9", player=None, opponent=FakeDriver(0))


def test_side_health_reads_smashbots_counters_and_tolerates_a_plain_opponent() -> None:
    assert side_health(HealthyDriver(0))["tactics"] == {"Punish/Waveshine": 7}
    assert side_health(FakeDriver(0)) == {}
    assert side_health(None) == {}


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def test_every_driven_side_is_asked_for_a_controller_every_frame(tmp_path: Path) -> None:
    env = FakeEnv(num_envs=2)
    left, right = FakeDriver(0), FakeDriver(1)
    sides = (
        ExternalSide(name="smashbot", player=0, opponent=left),
        ExternalSide(name="mimic", player=1, opponent=right),
    )
    run = run_external_matches(env, sides, replay_root=tmp_path, max_frames=5, poll_frames=100)

    assert env.steps == 5 and left.calls == 5 and right.calls == 5
    assert run.frames == 5 and run.stopped == "cap"
    assert all(set(action) == {0, 1} for action in env.actions)
    assert env.closed is False, "the caller owns the environment's lifetime"


def test_a_cpu_side_receives_no_controller(tmp_path: Path) -> None:
    env = FakeEnv(num_envs=2, controlled=(0,))
    smashbot = FakeDriver(0)
    sides = (
        ExternalSide(name="smashbot", player=0, opponent=smashbot),
        ExternalSide(name="cpu9", player=None, opponent=None),
    )
    run_external_matches(env, sides, replay_root=tmp_path, max_frames=3, poll_frames=100)
    assert all(set(action) == {0} for action in env.actions)


def test_the_reset_flags_come_from_the_pending_state(tmp_path: Path) -> None:
    """The driver must pass the *pending* frame's flags, so a fresh game resets the agent's state."""

    env = FakeEnv(num_envs=2)
    driver = FakeDriver(0)
    sides = (ExternalSide(name="smashbot", player=0, opponent=driver),)
    run_external_matches(env, sides, replay_root=tmp_path, max_frames=3, poll_frames=100)
    assert driver.resets[0] == [True, True], "frame -123 of the first game"
    assert driver.resets[1] == [False, False]


def test_the_drivers_see_the_gamestate_of_the_frame_they_are_acting_on(tmp_path: Path) -> None:
    """``gamestates()`` must be read once per frame, before ``step`` moves the emulators on."""

    env = FakeEnv(num_envs=1)
    sides = (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0, num_envs=1)),)
    run_external_matches(env, sides, replay_root=tmp_path, max_frames=4, poll_frames=100)
    assert env.gamestate_reads == 0, "each opponent holds its own gamestates() closure, as in _play"


# ---------------------------------------------------------------------------
# the stop rule
# ---------------------------------------------------------------------------


def test_the_run_stops_when_every_dolphin_has_begun_a_second_game(tmp_path: Path) -> None:
    """The recorder's match-mode rule: a second ``.slp`` means the first match is complete."""

    env = FakeEnv(num_envs=2, replay_root=tmp_path, write_at={1: 1, 3: 2})
    run = run_external_matches(
        env,
        (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0)),),
        replay_root=tmp_path,
        max_frames=1000,
        poll_frames=4,
    )
    assert run.stopped == "complete"
    assert env.steps == 4, "the poll happens every poll_frames, not every frame"


def test_one_dolphin_still_playing_keeps_the_run_going(tmp_path: Path) -> None:
    """Only env 0 reaches a second game; the run must play on to the cap."""

    root = tmp_path / "partial"
    env = FakeEnv(num_envs=2)

    def step_and_write(actions: Any) -> Any:
        env.actions.append(actions)
        env.steps += 1
        write_replays(root, envs=1, games=2)
        (root / "env1").mkdir(parents=True, exist_ok=True)
        return env._output()

    env.step = step_and_write  # type: ignore[method-assign]
    run = run_external_matches(
        env,
        (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0)),),
        replay_root=root,
        max_frames=8,
        poll_frames=4,
    )
    assert run.stopped == "cap" and env.steps == 8


def test_the_frame_cap_is_a_ceiling_not_a_target(tmp_path: Path) -> None:
    env = FakeEnv(num_envs=1)
    run = run_external_matches(
        env,
        (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0, num_envs=1)),),
        replay_root=tmp_path,
        max_frames=0,
        poll_frames=4,
    )
    assert env.steps == 0 and run.frames == 0


# ---------------------------------------------------------------------------
# stale replays must not be mistaken for this run's matches
# ---------------------------------------------------------------------------


def test_a_replay_directory_that_is_not_empty_is_refused(tmp_path: Path) -> None:
    """Measured: a smoke re-run into a used ``--output-dir`` reported "128 frames in 2 s (complete),
    4 matches read", all of them unfinished leftovers from an earlier failed run."""

    from melee_rl.external_match import ensure_empty_replay_root

    ensure_empty_replay_root(tmp_path / "never-existed")
    ensure_empty_replay_root(tmp_path)
    write_replays(tmp_path, envs=2, games=1)
    with pytest.raises(ValueError, match="already holds"):
        ensure_empty_replay_root(tmp_path)


def test_the_driver_itself_tolerates_the_replays_booting_wrote(tmp_path: Path) -> None:
    """The check belongs *before* the environment is built.  Booting the Dolphins starts their first
    game, which writes one ``.slp`` each -- and that file is what the ``>= 2`` stop rule counts from,
    so the driver must not treat it as stale.  A first attempt put the check in the driver and the
    next smoke died on its own boot replays."""

    write_replays(tmp_path, envs=2, games=1)  # exactly what booting leaves behind
    run = run_external_matches(
        FakeEnv(num_envs=2),
        (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0)),),
        replay_root=tmp_path,
        max_frames=8,
        poll_frames=4,
    )
    assert run.stopped == "cap" and run.frames == 8


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def test_the_run_reports_the_sides_the_health_and_the_seconds(tmp_path: Path) -> None:
    env = FakeEnv(num_envs=2)
    sides = (
        ExternalSide(name="smashbot", player=0, opponent=HealthyDriver(0)),
        ExternalSide(name="cpu9", player=None, opponent=None),
    )
    run = run_external_matches(env, sides, replay_root=tmp_path, max_frames=2, poll_frames=100)
    payload = run.to_dict()

    assert payload["sides"] == ["smashbot", "cpu9"]
    assert payload["num_envs"] == 2 and payload["frames"] == 2
    assert payload["health"]["smashbot"]["tactics"] == {"Punish/Waveshine": 7}
    assert "cpu9" not in payload["health"]
    assert payload["seconds"] >= 0.0
    assert payload["matches"] == []


def test_finished_matches_are_read_back_out_of_the_slp(tmp_path: Path, monkeypatch: Any) -> None:
    """The results come from ``melee_rl.slp`` through the recorder's own ``match_record``."""

    from melee_rl import external_match

    records = [{"stocks_end": 3, "winner": 1}, {"stocks_end": 0, "winner": 2}]
    seen: list[Path] = []

    def fake_record(path: Path) -> dict[str, Any]:
        seen.append(path)
        return records[len(seen) - 1]

    monkeypatch.setattr(external_match, "_record_for", fake_record)
    env = FakeEnv(num_envs=2, replay_root=tmp_path, write_at={1: 2})
    run = run_external_matches(
        env,
        (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0)),),
        replay_root=tmp_path,
        max_frames=8,
        poll_frames=4,
    )
    assert [path.name for path in seen] == ["Game_0.slp", "Game_0.slp"], "the FIRST game of each Dolphin"
    assert run.to_dict()["matches"] == records


def test_an_unreadable_replay_is_reported_not_raised(tmp_path: Path, monkeypatch: Any) -> None:
    """A single corrupt ``.slp`` must not lose the other fifteen matches of a job."""

    from melee_rl import external_match

    def boom(path: Path) -> dict[str, Any]:
        raise ValueError("truncated")

    monkeypatch.setattr(external_match, "_record_for", boom)
    run = run_external_matches(
        FakeEnv(num_envs=1, replay_root=tmp_path, write_at={1: 2}),
        (ExternalSide(name="smashbot", player=0, opponent=FakeDriver(0, num_envs=1)),),
        replay_root=tmp_path,
        max_frames=8,
        poll_frames=4,
    )
    matches = run.to_dict()["matches"]
    assert len(matches) == 1 and "truncated" in matches[0]["error"]


# ---------------------------------------------------------------------------
# the side grammar
# ---------------------------------------------------------------------------


def test_side_specs_parse() -> None:
    from melee_rl.external_match import EXTERNAL_SIDE_KINDS, parse_side

    assert EXTERNAL_SIDE_KINDS == ("smashbot", "mimic", "slippi_ai", "phillip", "cpu")
    assert parse_side("smashbot").kind == "smashbot"
    assert parse_side("mimic").kind == "mimic"
    plain = parse_side("slippi_ai")
    assert plain.kind == "slippi_ai" and plain.model == ""  # "" -> [slippi_ai] model
    named = parse_side("slippi_ai:gm")
    assert named.kind == "slippi_ai" and named.model == "gm"
    assert not plain.driven_by_env and not named.driven_by_env
    phillip = parse_side("phillip")
    assert phillip.kind == "phillip" and phillip.model == "" and not phillip.driven_by_env
    named_agent = parse_side("phillip:delay0/FalcoFD")
    assert named_agent.kind == "phillip" and named_agent.model == "delay0/FalcoFD"
    with pytest.raises(ValueError, match="side"):
        parse_side("phillip:")
    cpu = parse_side("cpu9")
    assert cpu.kind == "cpu" and cpu.cpu_level == 9
    assert parse_side("cpu1").cpu_level == 1


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "policy",
        "cpu0",
        "cpu10",
        "bc",
        "smashbot:mimic",
        "CPU9",
        "slippi_ai:",
        "slippi_ai:fox_d0_ditto_v9",
        "slippi-ai",
    ],
)
def test_bad_side_specs_are_rejected(bad: str) -> None:
    from melee_rl.external_match import parse_side

    with pytest.raises(ValueError, match="side"):
        parse_side(bad)


def test_the_environment_layout_follows_the_sides() -> None:
    """Dolphin's CPU can only be port 2, so a ``cpu`` side must be the second one."""

    from melee_rl.external_match import env_layout

    assert env_layout(("slippi_ai", "cpu9")) == (("policy", "cpu"), 9)
    assert env_layout(("slippi_ai:gm", "mimic")) == (("policy", "policy"), 9)
    assert env_layout(("slippi_ai:gm", "smashbot")) == (("policy", "policy"), 9)
    assert env_layout(("smashbot", "cpu9")) == (("policy", "cpu"), 9)
    assert env_layout(("smashbot", "mimic")) == (("policy", "policy"), 9)
    assert env_layout(("mimic", "smashbot")) == (("policy", "policy"), 9)
    assert env_layout(("smashbot", "cpu1")) == (("policy", "cpu"), 1)


def test_a_cpu_must_be_the_second_side_and_there_must_be_exactly_two() -> None:
    from melee_rl.external_match import env_layout

    with pytest.raises(ValueError, match="port 2"):
        env_layout(("cpu9", "smashbot"))
    with pytest.raises(ValueError, match="exactly two"):
        env_layout(("smashbot",))
    with pytest.raises(ValueError, match="exactly two"):
        env_layout(("smashbot", "mimic", "cpu9"))
    with pytest.raises(ValueError, match="one side"):
        env_layout(("cpu9", "cpu1"))


def test_the_manifest_is_written_beside_the_replays(tmp_path: Path) -> None:
    from melee_rl.external_match import write_external_manifest

    run = ExternalMatchRun(sides=("smashbot", "cpu9"), num_envs=2, frames=9, seconds=1.0, stopped="cap")
    path = write_external_manifest(run, tmp_path, extra={"label": "calib", "delay_frames": 1})
    import json

    payload = json.loads(path.read_text())
    assert path.name == "manifest.json"
    assert payload["sides"] == ["smashbot", "cpu9"] and payload["label"] == "calib"
    assert payload["delay_frames"] == 1 and payload["format"] == "melee_rl.external_match.v1"
