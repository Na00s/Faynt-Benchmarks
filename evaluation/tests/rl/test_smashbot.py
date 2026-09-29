"""SmashBot as a live opponent (P25, 9 Sep 2026).

`altf4/SmashBot <https://github.com/altf4/SmashBot>`_ (GPL-3.0, pinned at ``67c4120``) is an expert
system, not a network: ``ESAgent.act(gamestate)`` reads a raw libmelee ``GameState`` and *presses a
libmelee controller*.  That is the same boundary :mod:`melee_rl.mimic` already crosses, so the shape is
the same -- :class:`~melee_rl.mimic.ControllerRecorder` stands in for the controller,
``DolphinEnv.gamestates()`` supplies the observation, and the opponent returns a ``ControllerState``
through ``act_state`` rather than labels through ``act``.

Three things differ from MIMIC, and they are what these tests pin.

* **The recorder must persist across frames.**  libmelee's ``Controller.current`` is durable -- a button
  pressed on frame N stays pressed until it is released, and ``flush()`` only copies current to prev.
  MIMIC builds a *fresh* recorder every frame because its decoder writes every field every frame;
  SmashBot's chains do not, so a fresh recorder would silently drop held inputs.
* **The RNG is the ``random`` module.**  Seven SmashBot modules draw from the global stream
  (``Tactics/recover.py`` picks its recovery angle that way), so it gets a private stream in the same
  shape as MIMIC's ``_RngStream``, and the caller's ``random`` is never disturbed.
* **``FrameData`` must be shared.**  ``melee.framedata.FrameData()`` measures 800 ms and 54 MB per
  instance here; one per Dolphin would be 19 s and 1.3 GB at 24 envs and 76 s and 5.2 GB at 96.  One
  instance is built and injected into every agent.

The real SmashBot checkout is not on the path here, so the tests drive
:class:`melee_rl.smashbot.SmashBotApi` with a fake ``ESAgent``.  What is pinned is *our* half of the
boundary: the constructor arguments we pass, the controller semantics we present, the per-environment
state we keep, the error policy, and the health counters the calibration gate reads.
"""

from __future__ import annotations

import random
import sys
from typing import Any, ClassVar

import pytest
import torch

from melee_rl.env.dolphin import ControllerRow, neutral_row
from melee_rl.mimic import ControllerRecorder
from melee_rl.smashbot import (
    DEFAULT_DIFFICULTY,
    IN_GAME_MENUS,
    SmashBotApi,
    SmashBotConfig,
    SmashBotPlayer,
    load_smashbot_api,
)

melee = pytest.importorskip("melee", reason="the player reads libmelee enums for its menu guard")


# ---------------------------------------------------------------------------
# fakes: the one upstream symbol (esagent.ESAgent) and a libmelee gamestate
# ---------------------------------------------------------------------------


class FakePlayerState:
    """The fields the invulnerability tracker and the menu guard read off a libmelee ``PlayerState``."""

    def __init__(self) -> None:
        self.position = type("P", (), {"x": 0.0, "y": 0.0})()
        self.action: Any = None
        self.action_frame: int = 0
        self.invulnerability_left: int = 0
        self.invulnerable: bool = False


class FakeGamestate:
    """Only what the player itself reads: ``menu_state`` and ``players``.  ``tag`` identifies it."""

    def __init__(self, tag: int, *, menu_state: Any = None, ports: tuple[int, ...] = (1, 2)) -> None:
        self.tag = tag
        self.menu_state = melee.Menu.IN_GAME if menu_state is None else menu_state
        self.players: dict[int, FakePlayerState] = {port: FakePlayerState() for port in ports}
        self.stage = melee.Stage.FINAL_DESTINATION
        self.frame = 0


class FakeESAgent:
    """Records how it was built and drives the controller the way SmashBot's chains do."""

    instances: ClassVar[list[FakeESAgent]] = []

    def __init__(
        self, dolphin: Any, smashbot_port: int, opponent_port: int, controller: Any, difficulty: int = 4
    ) -> None:
        self.dolphin = dolphin
        self.smashbot_port = smashbot_port
        self.opponent_port = opponent_port
        self.controller = controller
        self.difficulty = difficulty
        self.framedata = dolphin.framedata
        self.strategy = FakeStrategy()
        self.seen: list[int] = []
        self.script: Any = None
        FakeESAgent.instances.append(self)

    def act(self, gamestate: Any) -> None:
        self.seen.append(gamestate.tag)
        if self.script is not None:
            self.script(self, gamestate)


class FakeTactic:
    def __init__(self) -> None:
        self.chain = None


class FakeStrategy:
    """``str(strategy)`` upstream is ``"Bait<class ...><class ...>"``; we read the objects instead."""

    def __init__(self) -> None:
        self.tactic: Any = FakeTactic()


def press_a(agent: FakeESAgent, gamestate: Any) -> None:
    agent.controller.press_button(melee.Button.BUTTON_A)


def tilt_right(agent: FakeESAgent, gamestate: Any) -> None:
    agent.controller.tilt_analog(melee.Button.BUTTON_MAIN, 1.0, 0.5)


def do_nothing(agent: FakeESAgent, gamestate: Any) -> None:
    return None


def boom(agent: FakeESAgent, gamestate: Any) -> None:
    raise RuntimeError("a tactic blew up")


@pytest.fixture(autouse=True)
def _fresh_agents() -> None:
    FakeESAgent.instances.clear()


def make_player(num_envs: int = 2, **kwargs: Any) -> SmashBotPlayer:
    config = SmashBotConfig(source_dir="/nowhere", **kwargs)
    api = SmashBotApi(agent_factory=FakeESAgent)
    return SmashBotPlayer(config, num_envs, api=api, framedata=object())


def step(player: SmashBotPlayer, states: list[Any], resets: list[bool] | None = None) -> list[ControllerRow]:
    flags = torch.tensor([False] * len(states) if resets is None else resets, dtype=torch.bool)
    return player.rows(states, flags)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_smashbot_api_names_the_missing_file(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match=r"esagent\.py"):
        load_smashbot_api(tmp_path)


def test_compat_defaults_on_and_reaches_the_loader(monkeypatch: Any, tmp_path: Any) -> None:
    """The shim is on by default; the flag exists so calibration can measure what the drift costs."""

    import melee_rl.smashbot as smashbot_mod

    assert SmashBotConfig(source_dir="/x").compat is True
    calls: list[bool] = []

    def fake_install() -> tuple[str, ...]:
        calls.append(True)
        return ()

    monkeypatch.setattr(smashbot_mod.smashbot_compat, "install", fake_install)
    (tmp_path / "esagent.py").write_text("class ESAgent:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    load_smashbot_api(tmp_path)
    assert calls == [True]
    load_smashbot_api(tmp_path, compat=False)
    assert calls == [True], "compat=False must not install the shim"


def test_the_health_panel_reports_the_knobs_the_gate_depends_on() -> None:
    stats = make_player(num_envs=1, delay_frames=2, compat=False).stats()
    assert stats["delay_frames"] == 2 and stats["compat"] is False
    assert stats["revision"] == "67c412092e4ef1576e1357a79461df5faedad3c2"


def test_config_rejects_a_bad_player_or_delay() -> None:
    with pytest.raises(ValueError, match="player must be 0 or 1"):
        SmashBotConfig(source_dir="/x", player=2)
    with pytest.raises(ValueError, match="delay_frames must be >= 0"):
        SmashBotConfig(source_dir="/x", delay_frames=-1)
    with pytest.raises(ValueError, match="source_dir"):
        SmashBotConfig(source_dir="")


# ---------------------------------------------------------------------------
# construction: one agent per environment, one shared FrameData
# ---------------------------------------------------------------------------


def test_one_agent_per_environment_on_the_right_libmelee_ports() -> None:
    """Environment player 1 is Dolphin port 2, and its opponent is port 1 (``dolphin.PORTS``)."""

    make_player(num_envs=3, player=1)
    assert len(FakeESAgent.instances) == 3
    for agent in FakeESAgent.instances:
        assert (agent.smashbot_port, agent.opponent_port) == (2, 1)


def test_player_zero_takes_port_one() -> None:
    make_player(num_envs=1, player=0)
    agent = FakeESAgent.instances[0]
    assert (agent.smashbot_port, agent.opponent_port) == (1, 2)


def test_the_strongest_difficulty_is_passed_explicitly() -> None:
    """``Strategies/bait.py:46-50`` re-derives the difficulty every frame from the constructor value:
    ``-1`` (upstream's argparse default) self-handicaps to stocks remaining, ``5`` is a no-attack debug
    dummy, ``4`` is the strongest real setting.  The constructor's own ``self.difficulty = 4`` is
    overwritten on the first frame, so passing 4 is load-bearing."""

    make_player(num_envs=1)
    assert FakeESAgent.instances[0].difficulty == 4
    assert DEFAULT_DIFFICULTY == 4


def test_every_agent_shares_one_framedata() -> None:
    """800 ms and 54 MB per instance: one per Dolphin would be 5.2 GB at 96 envs."""

    player = make_player(num_envs=4)
    shared = {agent.framedata for agent in FakeESAgent.instances}
    assert len(shared) == 1
    assert shared.pop() is player.framedata


def test_each_agent_gets_its_own_controller_recorder() -> None:
    make_player(num_envs=3)
    recorders = {id(agent.controller) for agent in FakeESAgent.instances}
    assert len(recorders) == 3


# ---------------------------------------------------------------------------
# the controller: persistent, and libmelee's own reset semantics
# ---------------------------------------------------------------------------


def test_a_press_persists_until_it_is_released() -> None:
    """The MIMIC pattern -- a fresh recorder per frame -- would drop this, and SmashBot holds buttons."""

    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    agent.script = press_a
    first = step(player, [FakeGamestate(1)])[0]
    assert first.buttons[0] is True

    agent.script = do_nothing  # a frame that touches nothing at all
    second = step(player, [FakeGamestate(2)])[0]
    assert second.buttons[0] is True, "the A press did not survive a frame that left it alone"

    agent.script = lambda a, g: a.controller.release_button(melee.Button.BUTTON_A)
    third = step(player, [FakeGamestate(3)])[0]
    assert third.buttons[0] is False


def test_a_stick_tilt_persists_too() -> None:
    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    agent.script = tilt_right
    assert step(player, [FakeGamestate(1)])[0].main_x == pytest.approx(1.0)
    agent.script = do_nothing
    assert step(player, [FakeGamestate(2)])[0].main_x == pytest.approx(1.0)


def test_release_all_and_empty_input_match_libmelee() -> None:
    """``release_all`` is buttons off, both sticks 0.5, both shoulders 0; ``empty_input`` calls it."""

    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]

    def press_everything(a: FakeESAgent, g: Any) -> None:
        a.controller.press_button(melee.Button.BUTTON_A)
        a.controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.0, 1.0)
        a.controller.tilt_analog(melee.Button.BUTTON_C, 1.0, 0.0)
        a.controller.press_shoulder(melee.Button.BUTTON_L, 1.0)

    agent.script = press_everything
    assert step(player, [FakeGamestate(1)])[0] != neutral_row()

    agent.script = lambda a, g: a.controller.release_all()
    assert step(player, [FakeGamestate(2)])[0] == neutral_row()

    agent.script = press_everything
    step(player, [FakeGamestate(3)])
    agent.script = lambda a, g: a.controller.empty_input()
    assert step(player, [FakeGamestate(4)])[0] == neutral_row()


def test_the_controller_exposes_prev_as_a_real_libmelee_controller_state() -> None:
    """SmashBot reads ``controller.prev`` (28 sites, e.g. ``Chains/dashdance.py:25``): libmelee's
    snapshot of what was sent at the *previous* flush.  It is a real ``ControllerState`` so the button
    dict is keyed by ``melee.Button`` exactly as upstream expects."""

    from melee.controller import ControllerState as LibmeleeControllerState

    player = make_player(num_envs=1)
    recorder = FakeESAgent.instances[0].controller
    assert isinstance(recorder.prev, LibmeleeControllerState)
    assert recorder.prev.button[melee.Button.BUTTON_A] is False
    assert recorder.prev.main_stick == (0.5, 0.5) and recorder.prev.c_stick == (0.5, 0.5)
    assert recorder.prev.l_shoulder == 0
    assert player.num_envs == 1


def test_prev_is_the_previous_frame_not_this_one() -> None:
    """libmelee's ``flush()`` does ``prev = copy(current)`` once per frame, where ``DolphinEnv.step``
    flushes -- so inside frame N's ``act`` the agent sees what it sent on frame N-1."""

    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    seen: list[bool] = []

    def press_a_and_watch(a: FakeESAgent, g: Any) -> None:
        seen.append(bool(a.controller.prev.button[melee.Button.BUTTON_A]))
        a.controller.press_button(melee.Button.BUTTON_A)

    agent.script = press_a_and_watch
    for index in range(3):
        step(player, [FakeGamestate(index)])
    assert seen == [False, True, True], "frame 1 had no previous frame; frames 2 and 3 saw the A press"


def test_prev_carries_the_sticks_and_the_analog_shoulder() -> None:
    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]

    def set_everything(a: FakeESAgent, g: Any) -> None:
        a.controller.tilt_analog(melee.Button.BUTTON_MAIN, 1.0, 0.25)
        a.controller.tilt_analog(melee.Button.BUTTON_C, 0.0, 0.75)
        a.controller.press_shoulder(melee.Button.BUTTON_L, 0.6)

    agent.script = set_everything
    step(player, [FakeGamestate(1)])
    prev = agent.controller.prev
    assert prev.main_stick == pytest.approx((1.0, 0.25))
    assert prev.c_stick == pytest.approx((0.0, 0.75))
    assert prev.l_shoulder == pytest.approx(0.6)


def test_prev_advances_on_a_frame_the_agent_did_not_act_on() -> None:
    """A real controller is flushed every frame, menu or not; ``prev`` must not skip."""

    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    agent.script = press_a
    step(player, [FakeGamestate(1)])
    agent.script = lambda a, g: a.controller.release_all()
    step(player, [FakeGamestate(2)])
    assert bool(agent.controller.prev.button[melee.Button.BUTTON_A]) is False
    step(player, [FakeGamestate(3, menu_state=melee.Menu.CHARACTER_SELECT)])
    assert bool(agent.controller.prev.button[melee.Button.BUTTON_A]) is False


def test_a_reset_gives_a_fresh_prev() -> None:
    player = make_player(num_envs=1)
    FakeESAgent.instances[0].script = press_a
    step(player, [FakeGamestate(1)])
    step(player, [FakeGamestate(2)])
    assert bool(FakeESAgent.instances[0].controller.prev.button[melee.Button.BUTTON_A]) is True
    step(player, [FakeGamestate(3)], resets=[True])
    assert bool(FakeESAgent.instances[-1].controller.prev.button[melee.Button.BUTTON_A]) is False


def test_the_libmelee_lookup_is_cached(monkeypatch: Any) -> None:
    """``commit`` needs nine libmelee lookups per Dolphin per frame, and ``import_melee`` re-runs
    ``importlib.metadata.version`` on every call -- doing it live measured 73 ms per Dolphin-frame
    against a 53-65 ms frame-step.  Cache or the opponent becomes the bottleneck."""

    from melee_rl.env import libmelee_frames

    smashbot_mod = sys.modules["melee_rl.smashbot"]
    monkeypatch.setattr(smashbot_mod, "_CONTROLLER_BITS", None)
    calls: list[int] = []
    real = libmelee_frames.import_melee

    def counted() -> Any:
        calls.append(1)
        return real()

    monkeypatch.setattr(libmelee_frames, "import_melee", counted)
    player = make_player(num_envs=3)
    for index in range(20):
        step(player, [FakeGamestate(index)] * 3)
    assert len(calls) == 1, f"libmelee was re-imported {len(calls)} times across 60 Dolphin-frames"


def test_the_recorder_accepts_connect_and_flush() -> None:
    """Upstream chains call neither, but ``ESAgent`` construction and cleanup paths may."""

    recorder = ControllerRecorder()
    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    agent.script = lambda a, g: (a.controller.connect(), a.controller.flush())
    step(player, [FakeGamestate(1)])
    assert agent.controller.flushes == 1
    assert recorder.flushes == 0


# ---------------------------------------------------------------------------
# per-environment isolation and resets
# ---------------------------------------------------------------------------


def test_environments_do_not_leak_into_each_other() -> None:
    player = make_player(num_envs=2)
    first, second = FakeESAgent.instances
    first.script = press_a
    second.script = do_nothing
    rows = step(player, [FakeGamestate(1), FakeGamestate(2)])
    assert rows[0].buttons[0] is True
    assert rows[1] == neutral_row()
    assert first.seen == [1] and second.seen == [2]


def test_a_reset_rebuilds_the_agent_and_neutralises_the_controller() -> None:
    """A hard environment reset must forget the strategy, the lockout counters and the held input."""

    player = make_player(num_envs=2)
    FakeESAgent.instances[0].script = press_a
    assert step(player, [FakeGamestate(1), FakeGamestate(2)])[0].buttons[0] is True

    rows = step(player, [FakeGamestate(3), FakeGamestate(4)], resets=[True, False])
    assert len(FakeESAgent.instances) == 3, "the reset environment got a fresh agent"
    assert rows[0] == neutral_row(), "the fresh agent has no script, and its controller is neutral"
    assert FakeESAgent.instances[-1].seen == [3]


def test_reset_all_rebuilds_every_agent() -> None:
    player = make_player(num_envs=2)
    player.reset_all()
    assert len(FakeESAgent.instances) == 4


# ---------------------------------------------------------------------------
# invulnerability: the countdown 0.47.3 stopped computing
# ---------------------------------------------------------------------------


def invuln_gamestate(tag: int, action_1: Any = None, frame: int = 200, action_frame: int = 1) -> Any:
    """A gamestate whose port 1 is in ``action_1``; port 2 is always standing."""

    state = FakeGamestate(tag)
    state.frame = frame
    for port in (1, 2):
        player = state.players[port]
        player.action = melee.Action.STANDING
        player.action_frame = action_frame
        player.invulnerability_left = 0
    if action_1 is not None:
        state.players[1].action = action_1
    return state


def test_respawn_invulnerability_is_restored() -> None:
    """0.41.1 computed ``invulnerability_left`` in the parser (``console.py:818-838``): 120 frames on
    respawn, 36 on a ledge grab, counting down per port.  0.47.3 never assigns it -- it only has the
    dataclass default 0 -- so SmashBot believes nobody is ever invulnerable.  It reads the field in
    eight places in ``Tactics/edgeguard.py`` alone, plus its approach gate (``Strategies/bait.py:66``).
    """

    player = make_player(num_envs=1)
    step(player, [invuln_gamestate(1, melee.Action.ON_HALO_WAIT, frame=200)])
    assert FakeESAgent.instances[0].seen == [1]
    later = invuln_gamestate(2, frame=230)
    step(player, [later])
    # 120 granted at frame 200, read 30 frames later
    assert later.players[1].invulnerability_left == 90


def test_ledge_grab_invulnerability_is_36_frames() -> None:
    player = make_player(num_envs=1)
    step(player, [invuln_gamestate(1, melee.Action.EDGE_CATCHING, frame=300, action_frame=1)])
    later = invuln_gamestate(2, frame=310)
    step(player, [later])
    assert later.players[1].invulnerability_left == 26


def test_the_first_descent_grants_nothing() -> None:
    """0.41.1 explicitly skips ``ON_HALO_DESCENT`` before frame 150 -- the opening drop-in."""

    player = make_player(num_envs=1)
    early = invuln_gamestate(1, melee.Action.ON_HALO_DESCENT, frame=100)
    step(player, [early])
    assert early.players[1].invulnerability_left == 0
    late = invuln_gamestate(2, melee.Action.ON_HALO_DESCENT, frame=400)
    step(player, [late])
    assert late.players[1].invulnerability_left == 120


def test_the_countdown_never_goes_negative() -> None:
    player = make_player(num_envs=1)
    step(player, [invuln_gamestate(1, melee.Action.ON_HALO_WAIT, frame=200)])
    later = invuln_gamestate(2, frame=1000)
    step(player, [later])
    assert later.players[1].invulnerability_left == 0


def test_both_ports_are_tracked_independently() -> None:
    """SmashBot reads its *opponent's* invulnerability as well as its own."""

    player = make_player(num_envs=1)
    first = invuln_gamestate(1, frame=200)
    first.players[2].action = melee.Action.ON_HALO_WAIT
    step(player, [first])
    later = invuln_gamestate(2, frame=210)
    step(player, [later])
    assert later.players[2].invulnerability_left == 110
    assert later.players[1].invulnerability_left == 0


def test_a_reset_clears_the_countdown() -> None:
    player = make_player(num_envs=1)
    step(player, [invuln_gamestate(1, melee.Action.ON_HALO_WAIT, frame=200)])
    fresh = invuln_gamestate(2, frame=210)
    step(player, [fresh], resets=[True])
    assert fresh.players[1].invulnerability_left == 0


def test_compat_off_leaves_the_countdown_dead() -> None:
    """The A/B must measure the whole drift, this restoration included."""

    player = make_player(num_envs=1, compat=False)
    step(player, [invuln_gamestate(1, melee.Action.ON_HALO_WAIT, frame=200)])
    later = invuln_gamestate(2, frame=210)
    step(player, [later])
    assert later.players[1].invulnerability_left == 0


def test_our_own_policys_boolean_is_never_touched() -> None:
    """``melee_rl`` feeds the model ``player.invulnerable`` (``frames.py:55``), which 0.47.3 *does*
    populate from the raw byte.  Only SmashBot's countdown is restored, so the model's input is
    untouched -- and 0.41.1's extra ``invulnerable = True`` OR is deliberately not reproduced."""

    player = make_player(num_envs=1)
    state = invuln_gamestate(1, melee.Action.ON_HALO_WAIT, frame=200)
    for entry in state.players.values():
        entry.invulnerable = False
    step(player, [state])
    assert all(entry.invulnerable is False for entry in state.players.values())


# ---------------------------------------------------------------------------
# frames the agent must not see
# ---------------------------------------------------------------------------


def test_a_menu_frame_is_neutral_and_never_reaches_the_agent() -> None:
    """Upstream only calls ``act`` while ``menu_state is IN_GAME``; our environment owns the menus."""

    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    agent.script = press_a
    rows = step(player, [FakeGamestate(1, menu_state=melee.Menu.CHARACTER_SELECT)])
    assert rows[0] == neutral_row()
    assert agent.seen == []


def test_sudden_death_still_counts_as_in_game() -> None:
    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]
    step(player, [FakeGamestate(1, menu_state=melee.Menu.SUDDEN_DEATH)])
    assert agent.seen == [1]
    assert melee.Menu.SUDDEN_DEATH in IN_GAME_MENUS


def test_a_missing_gamestate_is_neutral() -> None:
    """``gamestates()`` returns ``None`` for an environment that is not live."""

    player = make_player(num_envs=2)
    FakeESAgent.instances[0].script = press_a
    rows = step(player, [None, FakeGamestate(2)])
    assert rows[0] == neutral_row()
    assert FakeESAgent.instances[0].seen == []


def test_a_frame_without_smashbots_port_is_neutral() -> None:
    player = make_player(num_envs=1, player=1)
    agent = FakeESAgent.instances[0]
    agent.script = press_a
    rows = step(player, [FakeGamestate(1, ports=(1,))])
    assert rows[0] == neutral_row()
    assert agent.seen == []


# ---------------------------------------------------------------------------
# the error policy: loud by default, counted when asked
# ---------------------------------------------------------------------------


def test_strict_lets_the_exception_out() -> None:
    """Bring-up default: a silently passive SmashBot is the failure mode we are guarding against."""

    player = make_player(num_envs=1, strict=True)
    FakeESAgent.instances[0].script = boom
    with pytest.raises(RuntimeError, match="a tactic blew up"):
        step(player, [FakeGamestate(1)])


def test_lenient_counts_the_exception_and_holds_a_neutral_input() -> None:
    """Upstream ``smashbot.py`` behaviour -- but counted and reported, never silent."""

    player = make_player(num_envs=2, strict=False)
    FakeESAgent.instances[0].script = boom
    FakeESAgent.instances[1].script = press_a
    rows = step(player, [FakeGamestate(1), FakeGamestate(2)])
    assert rows[0] == neutral_row()
    assert rows[1].buttons[0] is True
    stats = player.stats()
    assert stats["exceptions"] == 1
    assert "a tactic blew up" in stats["first_error"]
    assert stats["exception_envs"] == [0]


def test_a_lenient_exception_still_clears_the_held_input() -> None:
    """Upstream calls ``controller.empty_input()`` in its handler; a half-written frame must not stick."""

    player = make_player(num_envs=1, strict=False)
    agent = FakeESAgent.instances[0]
    agent.script = press_a
    step(player, [FakeGamestate(1)])

    def press_then_fail(a: FakeESAgent, g: Any) -> None:
        a.controller.press_button(melee.Button.BUTTON_B)
        raise RuntimeError("half way")

    agent.script = press_then_fail
    assert step(player, [FakeGamestate(2)])[0] == neutral_row()


# ---------------------------------------------------------------------------
# health counters -- what the calibration gate reads
# ---------------------------------------------------------------------------


def test_health_counters_track_frames_neutral_frames_and_tactics() -> None:
    player = make_player(num_envs=1)
    agent = FakeESAgent.instances[0]

    agent.script = do_nothing
    step(player, [FakeGamestate(1)])
    agent.strategy.tactic = FakeTactic()
    agent.strategy.tactic.chain = FakeTactic()
    agent.script = press_a
    step(player, [FakeGamestate(2)])

    stats = player.stats()
    assert stats["frames"] == 2
    assert stats["neutral_frames"] == 1
    assert stats["exceptions"] == 0
    assert stats["tactics"]["FakeTactic/-"] == 1
    assert stats["tactics"]["FakeTactic/FakeTactic"] == 1


def test_menu_and_dead_frames_are_not_counted_as_play() -> None:
    player = make_player(num_envs=1)
    step(player, [FakeGamestate(1, menu_state=melee.Menu.CHARACTER_SELECT)])
    step(player, [None])
    assert player.stats()["frames"] == 0


# ---------------------------------------------------------------------------
# the RNG: SmashBot draws from the global ``random`` module
# ---------------------------------------------------------------------------


def draw(agent: FakeESAgent, gamestate: Any) -> None:
    if random.randint(0, 1):
        agent.controller.press_button(melee.Button.BUTTON_A)
    else:
        agent.controller.release_button(melee.Button.BUTTON_A)


def test_the_callers_random_stream_is_never_disturbed() -> None:
    """Seven SmashBot modules call ``random.*``; our own draws must be unaffected."""

    player = make_player(num_envs=1)
    FakeESAgent.instances[0].script = draw
    random.seed(1234)
    expected = [random.random() for _ in range(5)]

    random.seed(1234)
    observed = []
    for index in range(5):
        step(player, [FakeGamestate(index)])
        observed.append(random.random())
    assert observed == expected


def test_two_players_with_the_same_seed_draw_the_same_sequence() -> None:
    def run(seed: int) -> list[bool]:
        player = make_player(num_envs=1, seed=seed)
        FakeESAgent.instances[-1].script = draw
        return [step(player, [FakeGamestate(i)])[0].buttons[0] for i in range(24)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_reset_all_restarts_the_random_stream() -> None:
    player = make_player(num_envs=1, seed=3)
    FakeESAgent.instances[-1].script = draw
    first = [step(player, [FakeGamestate(i)])[0].buttons[0] for i in range(24)]
    player.reset_all()
    FakeESAgent.instances[-1].script = draw
    second = [step(player, [FakeGamestate(i)])[0].buttons[0] for i in range(24)]
    assert first == second


# ---------------------------------------------------------------------------
# the opponent the rollout worker sees
# ---------------------------------------------------------------------------


def make_opponent(num_envs: int = 2, **kwargs: Any) -> Any:
    from melee_rl.opponents import SmashBotOpponent

    player = make_player(num_envs, **kwargs)
    states = [FakeGamestate(index) for index in range(num_envs)]
    return SmashBotOpponent(
        player, lambda: states, port=player.config.player, delay_frames=player.config.delay_frames
    )


def test_opponent_port_is_the_environment_player_index_not_the_libmelee_port() -> None:
    """``actor.py`` checks ``opponent.port`` against ``EnvProtocol.controlled_ports``, which is
    ``(0,)`` or ``(0, 1)``; ``ESAgent`` and ``gamestate.players`` use libmelee ports 1 and 2."""

    opponent = make_opponent(player=1)
    assert opponent.port == 1
    assert opponent.player.config.libmelee_port == 2
    assert FakeESAgent.instances[0].smashbot_port == 2
    assert opponent.controls_port is True


def test_opponent_act_state_carries_the_recorded_rows() -> None:
    opponent = make_opponent(2)
    FakeESAgent.instances[0].script = press_a
    state = opponent.act_state(None, torch.zeros(2, dtype=torch.bool))
    assert state.buttons.A.tolist() == [True, False]
    assert state.main_stick.x.tolist() == pytest.approx([0.5, 0.5])


def test_opponent_refuses_the_label_path() -> None:
    """SmashBot sets raw stick floats; Ali's 85-value ``main_stick`` vocabulary would quantise them."""

    opponent = make_opponent(1)
    with pytest.raises(NotImplementedError, match="act_state"):
        opponent.act(None, torch.zeros(1, dtype=torch.bool))


def test_opponent_refresh_is_a_no_op() -> None:
    """SmashBot is never trained; ``refresh`` exists only to satisfy the ``Opponent`` protocol."""

    opponent = make_opponent(1)
    assert opponent.refresh(None) is None


def test_opponent_reset_state_rebuilds_every_agent() -> None:
    opponent = make_opponent(2)
    before = len(FakeESAgent.instances)
    opponent.reset_state()
    assert len(FakeESAgent.instances) == before + 2


def test_opponent_satisfies_the_opponent_protocol() -> None:
    from melee_rl.opponents import Opponent

    assert isinstance(make_opponent(1), Opponent)


# ---------------------------------------------------------------------------
# the input-delay sweep
# ---------------------------------------------------------------------------


def test_zero_delay_sends_the_frame_it_decided_on() -> None:
    """Our environment's default: decide on frame N, land on frame N (``use_exi_inputs``, no queue)."""

    opponent = make_opponent(1, delay_frames=0)
    FakeESAgent.instances[0].script = press_a
    assert opponent.act_state(None, torch.zeros(1, dtype=torch.bool)).buttons.A.tolist() == [True]


def test_a_delay_of_k_holds_the_input_for_k_frames() -> None:
    """The sweep's instrument: SmashBot sees frame N, the game sees its input at frame N + k."""

    opponent = make_opponent(1, delay_frames=2)
    FakeESAgent.instances[0].script = press_a
    flags = torch.zeros(1, dtype=torch.bool)
    pressed = [bool(opponent.act_state(None, flags).buttons.A[0]) for _ in range(4)]
    assert pressed == [False, False, True, True], "two neutral frames precede the first real input"


def test_a_delay_queue_is_refilled_on_reset() -> None:
    """A new game must not inherit the previous game's in-flight inputs."""

    opponent = make_opponent(1, delay_frames=1)
    FakeESAgent.instances[0].script = press_a
    opponent.act_state(None, torch.zeros(1, dtype=torch.bool))
    assert bool(opponent.act_state(None, torch.zeros(1, dtype=torch.bool)).buttons.A[0]) is True

    opponent.reset_state()
    FakeESAgent.instances[-1].script = press_a
    assert bool(opponent.act_state(None, torch.zeros(1, dtype=torch.bool)).buttons.A[0]) is False


def test_the_delay_is_per_environment() -> None:
    """Two Dolphins in one pool must not shift each other's inputs."""

    opponent = make_opponent(2, delay_frames=1)
    FakeESAgent.instances[0].script = press_a
    flags = torch.zeros(2, dtype=torch.bool)
    assert opponent.act_state(None, flags).buttons.A.tolist() == [False, False]
    assert opponent.act_state(None, flags).buttons.A.tolist() == [True, False]


def test_a_per_environment_reset_flag_drops_that_environments_queued_inputs() -> None:
    """Frame -123 of a new game must not receive the last game's in-flight buttons."""

    opponent = make_opponent(2, delay_frames=1)
    for agent in FakeESAgent.instances:
        agent.script = press_a
    opponent.act_state(None, torch.zeros(2, dtype=torch.bool))  # both presses now in flight
    reset_first = torch.tensor([True, False], dtype=torch.bool)
    landed = opponent.act_state(None, reset_first).buttons.A.tolist()
    assert landed == [False, True], "env 0 reset, so its queued press was dropped; env 1 kept its own"
