"""``DolphinEnv`` (PLAN.md §6 P5) on a fake libmelee backend: boot, stepping, faults, resets, keepalive.

No Dolphin is launched: ``FakeBackend`` scripts what ``Console.step`` returns per launch (menu frames,
in-game frames, ``None`` timeouts, exceptions) and records every controller command and menu-helper
call.  Real libmelee 0.47.3 dataclasses and enums are used (``pytest.importorskip``), so the converter
runs for real; everything stays offline and under a second.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

melee = pytest.importorskip("melee")  # the ``dolphin`` extra (melee==0.47.3)

from controller_codec import ButtonState, ControllerState, CustomV1Codec, StickState  # noqa: E402
from melee_rl.env import dolphin as dolphin_lib  # noqa: E402
from melee_rl.env.dolphin import (  # noqa: E402
    DEFAULT_DOLPHIN_PATH,
    DEFAULT_ISO_PATH,
    PORTS,
    BuildInfo,
    ConnectFailed,
    ControllerRow,
    DolphinEnv,
    DolphinEnvConfig,
    DolphinLaunchError,
    check_dolphin_exe,
    check_iso_identity,
    controller_rows,
    neutral_row,
    pick_slippi_port,
    wait_for_pipe_reader,
)
from melee_rl.env.protocol import INITIAL_FRAME_INDEX, EnvProtocol  # noqa: E402
from melee_rl.frames import neutral_labels, tree_leaves  # noqa: E402
from tensor_batch import BUTTON_ORDER  # noqa: E402

CODEC = CustomV1Codec()
FD = melee.Stage.FINAL_DESTINATION
BF, PS = melee.Stage.BATTLEFIELD, melee.Stage.POKEMON_STADIUM
FOX, FALCO = melee.Character.FOX, melee.Character.FALCO
IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)


# ---------------------------------------------------------------------------
# gamestate builders
# ---------------------------------------------------------------------------


def _player(character: Any = FOX, x: float = 0.0, cpu_level: int = 0) -> Any:
    player = melee.PlayerState()
    player.character = character
    player.cpu_level = cpu_level
    player.position = melee.Position(np.float32(x), np.float32(0.0))
    player.action = melee.Action.STANDING
    player.jumps_left = 2
    player.controller_state = melee.ControllerState()
    return player


def _menu(menu: Any = melee.Menu.CHARACTER_SELECT, *, cpu_ready: bool = True) -> Any:
    """A menu gamestate; at the character select, port 2 reads as a level-9 CPU iff ``cpu_ready``."""

    state = melee.GameState()
    state.menu_state = menu
    state.players = {port: melee.PlayerState() for port in (1, 2, 3, 4)}
    if menu is melee.Menu.CHARACTER_SELECT:
        state.players[1].controller_status = melee.ControllerStatus.CONTROLLER_HUMAN
        opponent = state.players[2]
        if cpu_ready:
            opponent.controller_status = melee.ControllerStatus.CONTROLLER_CPU
            opponent.cpu_level = 9
        else:
            opponent.controller_status = melee.ControllerStatus.CONTROLLER_HUMAN
            opponent.cpu_level = 0
        opponent.is_holding_cpu_slider = False
    return state


def _ingame(
    frame: int,
    *,
    characters: tuple[Any, Any] = (FOX, FOX),
    stage: Any = FD,
    projectiles: Sequence[Any] = (),
    cpu_levels: tuple[int, int] = (0, 9),
) -> Any:
    state = melee.GameState()
    state.frame = frame
    state.stage = stage
    state.menu_state = melee.Menu.IN_GAME
    state.players = {
        1: _player(characters[0], -20.0 + frame, cpu_levels[0]),
        2: _player(characters[1], 20.0 - frame, cpu_levels[1]),
    }
    state.projectiles = list(projectiles)
    return state


def _projectile(spawn_id: int) -> Any:
    return melee.Projectile(
        position=melee.Position(np.float32(spawn_id), np.float32(0.0)),
        owner=1,
        type=melee.ProjectileType.FOX_LASER,
        subtype=0,
        spawn_id=np.uint32(spawn_id),
    )


def _game(count: int, start: int = INITIAL_FRAME_INDEX, **kwargs: Any) -> list[Any]:
    return [_ingame(start + i, **kwargs) for i in range(count)]


def _boot(menu_frames: int = 3, game_frames: int = 60, *, unready: int = 0, **kwargs: Any) -> list[Any]:
    """Main menu, ``menu_frames`` character-select frames (the first ``unready`` with port 2 still human),
    stage select, then ``game_frames`` in-game frames."""

    css = [_menu(cpu_ready=False)] * unready + [_menu()] * (menu_frames - unready)
    return [_menu(melee.Menu.MAIN_MENU), *css, _menu(melee.Menu.STAGE_SELECT), *_game(game_frames, **kwargs)]


# ---------------------------------------------------------------------------
# the fake backend
# ---------------------------------------------------------------------------


@dataclass
class Script:
    """What one launch of one fake console returns from ``step`` (GameState / None / exception)."""

    events: list[Any]
    connect_ok: bool = True


class FakeController:
    def __init__(self, console: FakeConsole, port: int) -> None:
        self.console = console
        self.port = port
        self.log: list[tuple[Any, ...]] = []
        self.connected = False
        self.disconnected = 0
        self.main = (0.5, 0.5)
        self.c = (0.5, 0.5)
        self.shoulder = 0.0
        self.buttons = dict.fromkeys(BUTTON_ORDER, False)

    def connect(self) -> bool:
        self.connected = True
        self.console.controllers.append(self)
        return True

    def disconnect(self) -> None:
        self.disconnected += 1

    def press_button(self, button: Any) -> None:
        self.buttons[button.value] = True
        self.log.append(("press", button.value))

    def release_button(self, button: Any) -> None:
        self.buttons[button.value] = False
        self.log.append(("release", button.value))

    def tilt_analog(self, button: Any, x: float, y: float) -> None:
        if button is melee.Button.BUTTON_MAIN:
            self.main = (x, y)
        else:
            self.c = (x, y)
        self.log.append(("tilt", button.value, x, y))

    def press_shoulder(self, button: Any, amount: float) -> None:
        self.shoulder = amount
        self.log.append(("shoulder", button.value, amount))

    def flush(self) -> None:
        self.log.append(("flush",))


class FakeConsole:
    def __init__(
        self,
        events: list[Any],
        *,
        connect_ok: bool,
        index: int,
        slippi_port: int,
        poll_timeout: float,
        backend: FakeBackend | None = None,
    ) -> None:
        self.events = list(events)
        self.connect_ok = connect_ok
        self.index = index
        self.slippi_port = slippi_port
        self.poll_timeout = poll_timeout
        self.backend = backend
        self.controllers: list[FakeController] = []
        self.run_calls: list[dict[str, Any]] = []
        self.connect_calls = 0
        self.stopped = 0
        self.steps = 0
        self.step_threads: set[str] = set()
        self._process: Any = None

    def run(self, iso_path: str | None = None, platform: str | None = None) -> None:
        self.run_calls.append({"iso_path": iso_path, "platform": platform})

    def connect(self) -> bool:
        self.connect_calls += 1
        if self.backend is not None:
            self.backend.timeline.append(("connect", self.index))
            self.backend.connect_threads.append(threading.current_thread().name)
        return self.connect_ok

    def get_dolphin_pipes_path(self, port: int) -> None:
        return None

    def stop(self) -> None:
        self.stopped += 1

    def step(self) -> Any:
        for controller in self.controllers:
            controller.flush()
        self.steps += 1
        self.step_threads.add(threading.current_thread().name)
        if not self.events:
            raise AssertionError(f"fake console {self.index}: script exhausted after {self.steps} steps")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        if event is None:
            time.sleep(self.poll_timeout)  # what polling_mode does: wait polling_timeout, then give up
            return None
        if event.menu_state in IN_GAME:
            for controller in self.controllers:  # what Dolphin would report for the inputs it just read
                player = event.players.get(controller.port)
                if player is None:
                    continue
                state = player.controller_state
                state.main_stick = controller.main
                state.c_stick = controller.c
                state.l_shoulder = controller.shoulder
                state.r_shoulder = controller.shoulder
                for name, pressed in controller.buttons.items():
                    state.processed_button[melee.Button(name)] = pressed
        return event


class FakeMenuHelper:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, dict[str, Any]]] = []

    def menu_helper_simple(self, gamestate: Any, controller: Any, **kwargs: Any) -> None:
        self.calls.append((gamestate.menu_state, controller.port, dict(kwargs)))


@dataclass
class FakeBackend:
    """One ``Script`` per launch per env; records consoles, ports, helpers and start calls."""

    scripts: dict[int, list[Script]]
    launches: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    consoles: list[FakeConsole] = field(default_factory=list)
    ports: list[tuple[int, int]] = field(default_factory=list)
    helpers: list[FakeMenuHelper] = field(default_factory=list)
    started: list[FakeConsole] = field(default_factory=list)
    prepared: DolphinEnvConfig | None = None
    timeline: list[tuple[str, int]] = field(default_factory=list)  # ("run" | "connect", env index) in order
    console_threads: list[str] = field(default_factory=list)  # the thread that created each console
    connect_threads: list[str] = field(default_factory=list)  # the thread of each console.connect()

    def prepare(self, config: DolphinEnvConfig) -> BuildInfo:
        self.prepared = config
        return BuildInfo(
            mainline=True, version="fake 4.0.0-mainline ExiAI", exi_ai=True, exe=config.dolphin_path
        )

    def pick_port(self, candidate: int) -> int:
        return candidate

    def make_console(self, config: DolphinEnvConfig, index: int, slippi_port: int) -> FakeConsole:
        launch = self.launches[index]
        self.launches[index] += 1
        scripts = self.scripts[index]
        if launch >= len(scripts):
            raise AssertionError(
                f"env {index}: unexpected launch #{launch + 1} (only {len(scripts)} scripted)"
            )
        script = scripts[launch]
        console = FakeConsole(
            script.events,
            connect_ok=script.connect_ok,
            index=index,
            slippi_port=slippi_port,
            poll_timeout=config.console_timeout_s,
            backend=self,
        )
        self.consoles.append(console)
        self.ports.append((index, slippi_port))
        self.console_threads.append(threading.current_thread().name)
        return console

    def make_controller(self, console: FakeConsole, port: int) -> FakeController:
        return FakeController(console, port)

    def make_menu_helper(self) -> FakeMenuHelper:
        helper = FakeMenuHelper()
        self.helpers.append(helper)
        return helper

    def start(self, console: FakeConsole, config: DolphinEnvConfig) -> None:
        console.run(iso_path=config.iso_path, platform="headless")
        self.started.append(console)
        self.timeline.append(("run", console.index))


def _env(scripts: dict[int, list[Script]], **overrides: Any) -> tuple[DolphinEnv, FakeBackend]:
    backend = FakeBackend(scripts)
    settings: dict[str, Any] = {"num_envs": len(scripts), "launch_stagger_s": 0.0, "check_iso": False}
    settings.update(overrides)
    env = DolphinEnv(DolphinEnvConfig(**settings), cpu_level=9, backend=backend)
    return env, backend


def _control(
    batch: int,
    *,
    main: tuple[float, float] = (0.5, 0.5),
    c: tuple[float, float] = (0.5, 0.5),
    shoulder: float = 0.0,
    pressed: Sequence[str] = (),
) -> ControllerState:
    def full(value: float) -> torch.Tensor:
        return torch.full((batch,), value, dtype=torch.float32)

    return ControllerState(
        main_stick=StickState(x=full(main[0]), y=full(main[1])),
        c_stick=StickState(x=full(c[0]), y=full(c[1])),
        shoulder=full(shoulder),
        buttons=ButtonState(**{name: torch.full((batch,), name in pressed) for name in BUTTON_ORDER}),
    )


def _controllers(console: FakeConsole) -> dict[int, FakeController]:
    return {controller.port: controller for controller in console.controllers}


def _frames(env: DolphinEnv) -> list[int]:
    return env.current().frame_index.tolist()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_boot_through_menus_protocol_conformance_and_menu_calls() -> None:
    scripts = {
        0: [Script([None, None, *_boot(menu_frames=40, unready=10)])],
        1: [Script(_boot(menu_frames=5))],
    }
    env, backend = _env(scripts, slippi_port=51441, console_timeout_s=0.001)
    try:
        assert isinstance(env, EnvProtocol)
        assert env.num_envs == 2 and env.controlled_ports == (0,)
        assert env.build == BuildInfo(
            mainline=True, version="fake 4.0.0-mainline ExiAI", exi_ai=True, exe=DEFAULT_DOLPHIN_PATH
        )
        assert backend.prepared is env.config
        output = env.current()
        output.validate(2)
        assert tuple(output.frames) == (0,) and tuple(output.rewards) == (0,)
        assert output.needs_reset.tolist() == [True, True]
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX] * 2
        assert output.rewards[0].tolist() == [0.0, 0.0] and output.rewards[0].dtype == torch.float32
        tree = output.frames[0]
        assert tree["p0"]["character"].tolist() == [1, 1] and tree["p1"]["character"].tolist() == [1, 1]
        assert tree["stage"].tolist() == [25, 25]
        assert tree["p0"]["x"].tolist() == [-143.0, -143.0]  # -20 + frame (-123): the scripted position
        # Launch sequence per env: console -> controllers on ports 1, 2 -> run -> connect -> controllers.
        assert backend.ports == [(0, 51441), (1, 51442)]
        assert [console.index for console in backend.consoles] == [0, 1]
        for console in backend.consoles:
            assert console.run_calls == [{"iso_path": DEFAULT_ISO_PATH, "platform": "headless"}]
            assert console.connect_calls == 1 and console.stopped == 0
            assert sorted(_controllers(console)) == list(PORTS)
            assert all(controller.connected for controller in console.controllers)
        assert backend.started == backend.consoles
        # Boot polls through the two ``None`` (no frame yet) before the first menu frame.
        assert backend.consoles[0].steps == 2 + 1 + 40 + 1 + 1
        # Menu helper: one stateful helper per port, called once per menu frame with slippi-ai's kwargs.
        helper_p1, helper_p2 = backend.helpers[0], backend.helpers[1]
        assert len(helper_p1.calls) == 42 and len(helper_p2.calls) == 42
        menus = [call[0] for call in helper_p1.calls]
        assert menus[0] is melee.Menu.MAIN_MENU and menus[-1] is melee.Menu.STAGE_SELECT
        assert {call[1] for call in helper_p1.calls} == {1} and {call[1] for call in helper_p2.calls} == {2}
        first = helper_p1.calls[0][2]
        assert first["character_selected"] is FOX and first["stage_selected"] is FD
        assert first["costume"] == 0 and first["cpu_level"] == 0 and first["connect_code"] is None
        assert first["swag"] is False and first["autostart"] is False
        assert helper_p2.calls[0][2]["costume"] == 1 and helper_p2.calls[0][2]["cpu_level"] == 9
        autostart = [call[2]["autostart"] for call in helper_p1.calls]
        assert (
            autostart == [False] * 31 + [True] * 11
        )  # autostart after 30 menu frames, first controller only
        assert not any(call[2]["autostart"] for call in helper_p2.calls)
        env_1_helper = backend.helpers[2]
        assert len(env_1_helper.calls) == 7 and not any(call[2]["autostart"] for call in env_1_helper.calls)
        stats = env.stats()
        assert [row["launches"] for row in stats] == [1, 1] and [row["relaunches"] for row in stats] == [0, 0]
        assert [row["slippi_port"] for row in stats] == [51441, 51442]
        assert all(row["launch_seconds"] >= 0.0 for row in stats)
        assert [row["cpu_levels"] for row in stats] == [
            {1: 0, 2: 9},
            {1: 0, 2: 9},
        ]  # from the game-start event
    finally:
        env.close()
    assert all(console.stopped == 1 for console in backend.consoles)


def test_autostart_waits_for_the_cpu_port_and_the_game_start_verifies_the_level() -> None:
    # Port 1 may press START only once port 2 reads as a CPU at the configured level with the slider released
    # (libmelee sets the toggle and slider after the coin is down; an early START starts a human-type port 2).
    events = [
        _menu(melee.Menu.MAIN_MENU),
        *[_menu(cpu_ready=False)] * 40,
        *[_menu(cpu_ready=True)] * 5,
        _menu(melee.Menu.STAGE_SELECT),
        *_game(5),
    ]
    env, backend = _env({0: [Script(events)]})
    try:
        helper_p1, helper_p2 = backend.helpers[0], backend.helpers[1]
        assert [call[2]["autostart"] for call in helper_p1.calls] == [False] * 41 + [True] * 6
        assert not any(call[2]["autostart"] for call in helper_p2.calls)
        assert env.stats()[0]["cpu_levels"] == {1: 0, 2: 9}
    finally:
        env.close()
    # Self-play has no cpu port: the 30-frame rule alone decides.
    env, backend = _env({0: [Script(_boot(menu_frames=40, unready=40))]}, players=("policy", "policy"))
    try:
        assert [call[2]["autostart"] for call in backend.helpers[0].calls] == [False] * 31 + [True] * 11
    finally:
        env.close()
    # A game that starts with port 2 at another level is a fault: relaunch, loud in the stats.
    scripts = {0: [Script(_boot(game_frames=2, cpu_levels=(0, 0))), Script(_boot(menu_frames=1))]}
    env, backend = _env(scripts)
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX] and len(backend.consoles) == 2
        assert "WrongCpuLevel" in env.stats()[0]["faults"][0] and "expected 9" in env.stats()[0]["faults"][0]
        assert env.stats()[0]["cpu_levels"] == {1: 0, 2: 9}
    finally:
        env.close()
    # Self-play ignores the game-start cpu levels of port 2.
    env, backend = _env({0: [Script(_boot(cpu_levels=(0, 0)))]}, players=("policy", "policy"))
    try:
        assert env.stats()[0]["cpu_levels"] == {1: 0, 2: 0} and env.stats()[0]["faults"] == []
    finally:
        env.close()


def test_step_sends_the_controller_in_slippi_ai_order_and_the_frame_echoes_it() -> None:
    env, backend = _env({0: [Script(_boot())]})
    try:
        console = backend.consoles[0]
        controllers = _controllers(console)
        for controller in controllers.values():
            controller.log.clear()
        control = _control(1, main=(0.25, 0.75), c=(1.0, 0.0), shoulder=0.35, pressed=("A", "Z"))
        output = env.step({0: control})
        output.validate(1)
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX + 1] and not bool(output.needs_reset.any())
        expected: list[tuple[Any, ...]] = [
            ("press" if name in ("A", "Z") else "release", name) for name in BUTTON_ORDER
        ]
        expected += [
            ("tilt", "MAIN", 0.25, 0.75),
            ("tilt", "C", 1.0, 0.0),
            ("shoulder", "L", pytest.approx(0.35)),
            ("flush",),
        ]
        assert controllers[1].log == expected
        assert controllers[2].log == [("flush",)]  # the cpu port is flushed by Console.step, never driven
        echoed = output.frames[0]["p0"]["controller"]
        assert echoed["main_stick"]["x"].tolist() == [0.25] and echoed["c_stick"]["y"].tolist() == [0.0]
        assert echoed["shoulder"].tolist() == [pytest.approx(0.35)]
        assert {name: bool(echoed["buttons"][name][0]) for name in BUTTON_ORDER} == {
            name: name in ("A", "Z") for name in BUTTON_ORDER
        }
        sent = CODEC.encode(control)
        got = CODEC.encode(echoed)
        assert torch.equal(sent.buttons, got.buttons) and torch.equal(sent.main_stick, got.main_stick)
        # Consecutive steps advance one frame each; the pending state is the last output.
        second = env.step({0: _control(1)})
        assert second.frame_index.tolist() == [INITIAL_FRAME_INDEX + 2]
        assert env.current() is second
        assert env.stats()[0]["frames"] == 3  # -123, -122, -121
        with pytest.raises(ValueError, match="needs controllers for ports"):
            env.step({})
        with pytest.raises(ValueError, match="needs controllers for ports"):
            env.step({0: _control(1), 1: _control(1)})
        with pytest.raises(ValueError, match="batch"):
            env.step({0: _control(2)})
    finally:
        env.close()


def test_game_restart_inside_step_is_a_reset_row_and_resets_item_slots() -> None:
    first_game = _game(5)
    first_game[3].projectiles = [_projectile(7), _projectile(8)]
    first_game[4].projectiles = [_projectile(7), _projectile(8)]
    restart = _ingame(INITIAL_FRAME_INDEX, projectiles=[_projectile(9)])
    events = [*_boot(game_frames=0), *first_game, restart, *_game(3, start=INITIAL_FRAME_INDEX + 1)]
    env, backend = _env({0: [Script(events)]})
    try:
        neutral = _control(1)
        outputs = [env.step({0: neutral}) for _ in range(4)]
        assert [int(output.frame_index[0]) for output in outputs] == [-122, -121, -120, -119]
        assert outputs[-1].frames[0]["items"]["exists"][0].tolist()[:3] == [True, True, False]
        assert outputs[-1].frames[0]["items"]["x"][0].tolist()[:2] == [7.0, 8.0]
        reset = env.step({0: neutral})
        reset.validate(1)
        assert reset.frame_index.tolist() == [INITIAL_FRAME_INDEX] and reset.needs_reset.tolist() == [True]
        # The item slots were reset at the game start: spawn id 9 lands in slot 0, nothing else exists.
        assert reset.frames[0]["items"]["exists"][0].tolist()[:3] == [True, False, False]
        assert reset.frames[0]["items"]["x"][0].tolist()[0] == 9.0
        after = env.step({0: neutral})
        assert after.frame_index.tolist() == [INITIAL_FRAME_INDEX + 1] and not bool(after.needs_reset.any())
        assert env.stats()[0]["relaunches"] == 0 and len(backend.consoles) == 1
    finally:
        env.close()


def test_in_game_timeout_and_disconnect_relaunch_the_env_and_retries_are_bounded() -> None:
    disconnect = melee.slippstream.EnetDisconnected()
    scripts = {
        0: [Script([*_boot(game_frames=3), None]), Script(_boot(menu_frames=1))],
        1: [Script([*_boot(game_frames=4), disconnect]), Script(_boot(menu_frames=1))],
    }
    env, backend = _env(scripts, slippi_port=51500, console_timeout_s=0.001)
    try:
        neutral = _control(2)
        for _ in range(2):
            env.step({0: neutral})
        output = env.step({0: neutral})  # env 0's console times out in game -> relaunch -> a -123 row
        output.validate(2)
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX, INITIAL_FRAME_INDEX + 3]
        assert output.needs_reset.tolist() == [True, False]
        assert len(backend.consoles) == 3 and backend.consoles[0].stopped == 1
        assert backend.ports == [(0, 51500), (1, 51501), (0, 51500)]
        assert all(controller.disconnected == 1 for controller in backend.consoles[0].controllers)
        output = env.step({0: neutral})  # env 1 disconnects -> relaunch
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX + 1, INITIAL_FRAME_INDEX]
        assert output.needs_reset.tolist() == [False, True]
        stats = env.stats()
        assert [row["relaunches"] for row in stats] == [1, 1]
        assert "ConsoleTimeout" in stats[0]["faults"][0] and "EnetDisconnected" in stats[1]["faults"][0]
        assert len(backend.consoles) == 4
    finally:
        env.close()
    # Bounded: launch_retries counts the additional attempts; failing connects exhaust them.
    failing = {
        0: [Script([*_boot(game_frames=2), None]), Script([], connect_ok=False), Script([], connect_ok=False)]
    }
    env, backend = _env(failing, launch_retries=1, console_timeout_s=0.001)
    try:
        env.step({0: _control(1)})
        with pytest.raises(DolphinLaunchError, match="2 launch attempts"):
            env.step({0: _control(1)})
        assert len(backend.consoles) == 3 and all(console.stopped >= 1 for console in backend.consoles[1:])
    finally:
        env.close()
    # A connect failure at construction is retried on the next launch attempt (the port is re-picked).
    env, backend = _env({0: [Script([], connect_ok=False), Script(_boot())]}, launch_retries=2)
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX] and len(backend.consoles) == 2
        assert backend.consoles[0].stopped == 1 and backend.consoles[0].connect_calls == 1
        assert env.stats()[0]["launches"] == 2 and env.stats()[0]["relaunches"] == 0
    finally:
        env.close()
    with pytest.raises(DolphinLaunchError, match="1 launch attempt"):
        _env({0: [Script([], connect_ok=False)]}, launch_retries=0)


def test_menu_stall_wrong_character_and_conversion_errors_relaunch() -> None:
    stalled = Script([_menu()] * 12)
    env, backend = _env({0: [stalled, Script(_boot())]}, menu_frame_cap=10)
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX]
        assert len(backend.consoles) == 2 and backend.consoles[0].stopped == 1
        assert "MenuStall" in env.stats()[0]["faults"][0]
    finally:
        env.close()
    wrong = Script(_boot(characters=(FALCO, FOX)))
    env, backend = _env({0: [wrong, Script(_boot())]})
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX] and len(backend.consoles) == 2
        assert "WrongCharacter" in env.stats()[0]["faults"][0] and "FALCO" in env.stats()[0]["faults"][0]
        assert env.current().frames[0]["p0"]["character"].tolist() == [1]
    finally:
        env.close()
    unknown = _ingame(INITIAL_FRAME_INDEX + 2, characters=(melee.Character.UNKNOWN_CHARACTER, FOX))
    broken = Script([*_boot(game_frames=2), unknown])
    env, backend = _env({0: [broken, Script(_boot(menu_frames=1))]})
    try:
        env.step({0: _control(1)})
        output = env.step({0: _control(1)})  # the converter rejects the frame -> relaunch -> reset row
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX] and output.needs_reset.tolist() == [True]
        assert "FrameConversionError" in env.stats()[0]["faults"][0]
    finally:
        env.close()
    # A menu phase that never produces a frame also stalls (boot_timeout_s bounds the wall time).
    silent = Script([None] * 200)
    env, backend = _env({0: [silent, Script(_boot())]}, boot_timeout_s=0.05, console_timeout_s=0.001)
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX] and len(backend.consoles) == 2
        assert "ConnectFailed" in env.stats()[0]["faults"][0]
    finally:
        env.close()


def test_reset_and_close_are_idempotent_and_stop_every_console() -> None:
    scripts = {
        0: [Script(_boot()), Script(_boot(menu_frames=1))],
        1: [Script(_boot()), Script(_boot(menu_frames=1))],
    }
    env, backend = _env(scripts)
    neutral = _control(2)
    env.step({0: neutral})
    env.step({0: neutral})
    assert _frames(env) == [INITIAL_FRAME_INDEX + 2] * 2
    output = env.reset()
    output.validate(2)
    assert output.needs_reset.tolist() == [True, True] and env.current() is output
    assert len(backend.consoles) == 4 and [console.stopped for console in backend.consoles] == [1, 1, 0, 0]
    assert [row["launches"] for row in env.stats()] == [2, 2] and [
        row["relaunches"] for row in env.stats()
    ] == [0, 0]
    after = env.step({0: neutral})
    assert after.frame_index.tolist() == [INITIAL_FRAME_INDEX + 1] * 2
    assert not env.closed
    env.close()
    assert env.closed and [console.stopped for console in backend.consoles] == [1, 1, 1, 1]
    env.close()
    assert [console.stopped for console in backend.consoles] == [1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="closed"):
        env.step({0: neutral})
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_keepalive_ticks_while_idle_and_parks_a_reset_frame() -> None:
    env, backend = _env({0: [Script(_boot(game_frames=400))]})
    try:
        assert env.keepalive_active is False
    finally:
        env.close()
    env, backend = _env({0: [Script(_boot(game_frames=400))]}, keepalive_seconds=0.02)
    try:
        assert env.keepalive_active is True
        console = backend.consoles[0]
        before = console.steps
        deadline = time.monotonic() + 5.0
        while env.stats()[0]["ticks"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        ticks_before = env.stats()[0]["ticks"]
        current = env.current()
        ticks_after = env.stats()[0]["ticks"]
        assert ticks_before >= 3 and console.steps >= before + ticks_before
        # current() follows the ticks (read between two tick counts: the thread may tick in between).
        assert (
            INITIAL_FRAME_INDEX + ticks_before
            <= int(current.frame_index[0])
            <= INITIAL_FRAME_INDEX + ticks_after
        )
        controller = _controllers(console)[1]
        assert ("tilt", "MAIN", 0.5, 0.5) in controller.log  # ticks hold the neutral controller
        output = env.step({0: _control(1, pressed=("B",))})
        assert int(output.frame_index[0]) > INITIAL_FRAME_INDEX + ticks_before
        assert ("press", "B") in controller.log
    finally:
        env.close()
    assert env.keepalive_active is False
    # A tick that reaches a game start parks the -123 frame: ticking stops, the next step delivers it
    # without sending the action, and the step after continues the new game.
    events = [*_boot(game_frames=3), *_game(1), *_game(5, start=INITIAL_FRAME_INDEX + 1)]
    env, backend = _env({0: [Script(events)]}, keepalive_seconds=0.02)
    try:
        console = backend.consoles[0]
        deadline = time.monotonic() + 5.0
        while not env.stats()[0]["parked"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert env.stats()[0]["parked"] is True
        steps_when_parked = console.steps
        time.sleep(0.1)
        assert console.steps == steps_when_parked  # no more ticks while a frame is parked
        current = env.current()
        assert current.frame_index.tolist() == [INITIAL_FRAME_INDEX] and current.needs_reset.tolist() == [
            True
        ]
        controller = _controllers(console)[1]
        controller.log.clear()
        output = env.step({0: _control(1, pressed=("A",))})
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX] and output.needs_reset.tolist() == [True]
        assert ("press", "A") not in controller.log and console.steps == steps_when_parked
        assert env.stats()[0]["parked"] is False
        output = env.step({0: _control(1, pressed=("A",))})
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX + 1] and ("press", "A") in controller.log
    finally:
        env.close()


def test_self_play_two_ports_share_leaves_and_drive_both_controllers() -> None:
    env, backend = _env({0: [Script(_boot())], 1: [Script(_boot())]}, players=("policy", "policy"))
    try:
        assert env.controlled_ports == (0, 1)
        output = env.current()
        output.validate(2)
        assert tuple(output.frames) == (0, 1) and tuple(output.rewards) == (0, 1)
        for (path, leaf), (other_path, other) in zip(
            tree_leaves(output.frames[1]["p0"]), tree_leaves(output.frames[0]["p1"]), strict=True
        ):
            assert path == other_path and leaf is other
        assert output.frames[1]["stage"] is output.frames[0]["stage"]
        for helper in backend.helpers:
            assert all(call[2]["cpu_level"] == 0 for call in helper.calls)
        left = _control(2, main=(0.0, 0.5))
        right = _control(2, main=(1.0, 0.5), pressed=("Y",))
        stepped = env.step({0: left, 1: right})
        stepped.validate(2)
        for console in backend.consoles:
            controllers = _controllers(console)
            assert ("tilt", "MAIN", 0.0, 0.5) in controllers[1].log and ("press", "Y") not in controllers[
                1
            ].log
            assert ("tilt", "MAIN", 1.0, 0.5) in controllers[2].log and ("press", "Y") in controllers[2].log
        assert stepped.frames[1]["p0"]["controller"]["main_stick"]["x"].tolist() == [1.0, 1.0]
        assert stepped.frames[0]["p1"]["controller"]["buttons"]["Y"].tolist() == [True, True]
        assert stepped.frames[0]["p0"]["controller"]["main_stick"]["x"].tolist() == [0.0, 0.0]
        with pytest.raises(ValueError, match="needs controllers for ports"):
            env.step({0: left})
    finally:
        env.close()
    with pytest.raises(DolphinLaunchError):
        _env({0: [Script([], connect_ok=False)]}, players=("policy", "policy"), launch_retries=0)


def test_config_validation_and_defaults() -> None:
    config = DolphinEnvConfig()
    assert config.num_envs == 1 and config.players == ("policy", "cpu") and config.controlled_ports == (0,)
    assert config.dolphin_path == DEFAULT_DOLPHIN_PATH and config.iso_path == DEFAULT_ISO_PATH
    assert config.characters == (1, 1) and config.costumes == (0, 1) and config.stage == 25
    assert config.stages == () and config.stage_for(3) == 25 and config.stage_ids == (25,)
    assert config.slippi_port == 51441 and config.console_timeout_s == 30.0 and config.boot_timeout_s == 120.0
    assert config.connect_timeout_s == 60.0 and config.menu_frame_cap == 3000 and config.launch_retries == 2
    assert config.launch_stagger_s == 0.0 and config.keepalive_seconds == 0.0
    assert config.infinite_time and config.instant_match_restart and config.online_delay == 0
    assert not config.save_replays and config.replay_dir is None and config.check_iso
    assert DolphinEnvConfig(players=("policy", "policy")).controlled_ports == (0, 1)
    with pytest.raises(ValueError, match="num_envs"):
        DolphinEnvConfig(num_envs=0)
    with pytest.raises(ValueError, match="players"):
        DolphinEnvConfig(players=("cpu", "cpu"))
    with pytest.raises(ValueError, match="players"):
        DolphinEnvConfig(players=("policy", "human"))
    with pytest.raises(ValueError, match="players"):
        DolphinEnvConfig(players=("cpu", "policy"))
    with pytest.raises(ValueError, match="characters"):
        DolphinEnvConfig(characters=(1, 33))
    with pytest.raises(ValueError, match="characters"):
        DolphinEnvConfig(characters=(-1, 1))
    with pytest.raises(ValueError, match="costumes"):
        DolphinEnvConfig(costumes=(0, -1))
    with pytest.raises(ValueError, match="stage"):
        DolphinEnvConfig(stage=64)
    with pytest.raises(ValueError, match="slippi_port"):
        DolphinEnvConfig(slippi_port=80)
    with pytest.raises(ValueError, match="slippi_port"):
        DolphinEnvConfig(slippi_port=65535, num_envs=2)
    for name in ("console_timeout_s", "boot_timeout_s", "connect_timeout_s"):
        zero: dict[str, Any] = {name: 0.0}
        with pytest.raises(ValueError, match=name):
            DolphinEnvConfig(**zero)
    with pytest.raises(ValueError, match="menu_frame_cap"):
        DolphinEnvConfig(menu_frame_cap=0)
    with pytest.raises(ValueError, match="launch_retries"):
        DolphinEnvConfig(launch_retries=-1)
    with pytest.raises(ValueError, match="launch_stagger_s"):
        DolphinEnvConfig(launch_stagger_s=-1.0)
    with pytest.raises(ValueError, match="keepalive_seconds"):
        DolphinEnvConfig(keepalive_seconds=-0.5)
    with pytest.raises(ValueError, match="online_delay"):
        DolphinEnvConfig(online_delay=-1)
    with pytest.raises(ValueError, match="iso_path"):
        DolphinEnvConfig(iso_path="")
    with pytest.raises(ValueError, match="dolphin_path"):
        DolphinEnvConfig(dolphin_path="")


def test_helpers_iso_identity_exe_check_pipe_probe_rows_and_ports(tmp_path: Path) -> None:
    iso = tmp_path / "game.iso"
    iso.write_bytes(b"GALE01" + bytes([0, 2]) + b"\0" * 24)
    identity = check_iso_identity(iso)
    assert identity == {"game_id": "GALE01", "revision": 2, "bytes": 32}
    wrong = tmp_path / "other.iso"
    wrong.write_bytes(b"GALE01" + bytes([0, 1]) + b"\0" * 24)
    with pytest.raises(ValueError, match="GALE01 rev 2"):
        check_iso_identity(wrong)
    wrong.write_bytes(b"GALJ01" + bytes([0, 2]))
    with pytest.raises(ValueError, match="GALE01 rev 2"):
        check_iso_identity(wrong)
    with pytest.raises(FileNotFoundError):
        check_iso_identity(tmp_path / "missing.iso")
    # The executable check: missing, not executable, fine.
    exe = tmp_path / "dolphin-emu"
    with pytest.raises(FileNotFoundError, match="dolphin"):
        check_dolphin_exe(exe)
    exe.write_text("#!/bin/sh\nexit 0\n")
    with pytest.raises(PermissionError, match="executable"):
        check_dolphin_exe(exe)
    exe.chmod(0o755)
    assert check_dolphin_exe(exe) == exe
    # The FIFO probe: a reader makes it return, no reader times out, a dead process is reported.
    fifo = tmp_path / "slippibot1"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        assert wait_for_pipe_reader(fifo, time.monotonic() + 1.0) is True
    finally:
        os.close(reader)
    started = time.monotonic()
    with pytest.raises(ConnectFailed, match="no reader"):
        wait_for_pipe_reader(fifo, time.monotonic() + 0.15)
    assert 0.1 <= time.monotonic() - started < 2.0
    dead = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    dead.wait()
    with pytest.raises(ConnectFailed, match="exit code 3"):
        wait_for_pipe_reader(fifo, time.monotonic() + 5.0, process=dead)
    assert wait_for_pipe_reader(tmp_path / "not-a-fifo", time.monotonic() + 1.0) is False  # nothing to probe
    # Controller rows: one plain-Python row per env, buttons in BUTTON_ORDER.
    control = _control(2, main=(0.1, 0.9), c=(0.3, 0.7), shoulder=1.0, pressed=("X", "D_UP"))
    rows = controller_rows(control, 2)
    assert len(rows) == 2 and rows[0] == rows[1]
    assert isinstance(rows[0], ControllerRow)
    assert (rows[0].main_x, rows[0].main_y) == (pytest.approx(0.1), pytest.approx(0.9))
    assert (rows[0].c_x, rows[0].c_y) == (pytest.approx(0.3), pytest.approx(0.7))
    assert rows[0].shoulder == 1.0
    assert rows[0].buttons == tuple(name in ("X", "D_UP") for name in BUTTON_ORDER)
    assert neutral_row() == ControllerRow(
        main_x=0.5, main_y=0.5, c_x=0.5, c_y=0.5, shoulder=0.0, buttons=(False,) * 8
    )
    assert controller_rows(CODEC.decode(neutral_labels((3,))), 3) == [neutral_row()] * 3
    with pytest.raises(ValueError, match="batch"):
        controller_rows(control, 3)
    # Port picking skips occupied UDP ports.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        assert pick_slippi_port(port) == port + 1
        with pytest.raises(RuntimeError, match="UDP port"):
            pick_slippi_port(port, span=1)
    assert pick_slippi_port(port) == port
    # Build info and module constants.
    assert BuildInfo(mainline=True, version="v", exi_ai=True, exe="x").platform == "headless"
    assert BuildInfo(mainline=False, version="v", exi_ai=True, exe="x").platform is None
    assert dolphin_lib.PORTS == (1, 2) and dolphin_lib.MENU_AUTOSTART_FRAMES == 30
    assert dolphin_lib.ISO_GAME_ID == b"GALE01" and dolphin_lib.ISO_REVISION == 2


# ---------------------------------------------------------------------------
# P6a: threaded stepping and the two-phase boot
# ---------------------------------------------------------------------------


def _three(game_frames: int = 40) -> dict[int, list[Script]]:
    return {i: [Script(_boot(game_frames=game_frames))] for i in range(3)}


def test_threaded_step_matches_sequential_outputs_and_per_console_call_order() -> None:
    sequential, seq_backend = _env(_three(), step_threads=1)
    threaded, thr_backend = _env(_three(), step_threads=0)  # 0 -> one thread per Dolphin
    try:
        assert sequential.step_threads == 1 and threaded.step_threads == 3
        assert not sequential.pool_active and threaded.pool_active
        for step in range(5):
            control = _control(3, main=(0.1 * step, 0.5), pressed=("A",) if step % 2 else ())
            a, b = sequential.step({0: control}), threaded.step({0: control})
            a.validate(3)
            b.validate(3)
            assert a.frame_index.tolist() == b.frame_index.tolist() == [INITIAL_FRAME_INDEX + step + 1] * 3
            assert a.needs_reset.tolist() == b.needs_reset.tolist() == [False] * 3
            for (path, leaf), (_, other) in zip(
                tree_leaves(a.frames[0]), tree_leaves(b.frames[0]), strict=True
            ):
                assert torch.equal(leaf, other), path
        for s_console, t_console in zip(seq_backend.consoles, thr_backend.consoles, strict=True):
            assert s_console.steps == t_console.steps
            for port in PORTS:
                assert _controllers(s_console)[port].log == _controllers(t_console)[port].log
        # Sequential: consoles are touched only by the calling thread; threaded: by the pool.
        assert all(name == "MainThread" for console in seq_backend.consoles for name in console.step_threads)
        pool_names = {name for console in thr_backend.consoles for name in console.step_threads}
        assert any(name.startswith("melee-rl-dolphin") for name in pool_names)
        assert [row["step_threads"] for row in threaded.stats()] == [3, 3, 3]
        assert [row["boot"] for row in threaded.stats()] == ["two_phase"] * 3
        assert [row["step_threads"] for row in sequential.stats()] == [1, 1, 1]
        assert [row["frames"] for row in threaded.stats()] == [6, 6, 6]
    finally:
        sequential.close()
        threaded.close()
    assert not threaded.pool_active and not sequential.pool_active


def test_two_phase_boot_starts_every_console_before_connecting_and_settles_in_the_pool() -> None:
    env, backend = _env(_three())  # boot = "two_phase" is the default
    try:
        runs = [position for position, (kind, _) in enumerate(backend.timeline) if kind == "run"]
        connects = [position for position, (kind, _) in enumerate(backend.timeline) if kind == "connect"]
        assert len(runs) == 3 and len(connects) == 3 and max(runs) < min(connects)
        assert [index for kind, index in backend.timeline if kind == "run"] == [0, 1, 2]
        assert [index for kind, index in backend.timeline if kind == "connect"] == [0, 1, 2]
        assert _frames(env) == [INITIAL_FRAME_INDEX] * 3 and env.current().needs_reset.tolist() == [True] * 3
        # Process creation and connect (libmelee forks its enet worker there) stay on the main thread; the
        # menu settling of the three Dolphins runs in the pool.
        assert backend.console_threads == ["MainThread"] * 3 and backend.connect_threads == ["MainThread"] * 3
        settle_names = {name for console in backend.consoles for name in console.step_threads}
        assert any(name.startswith("melee-rl-dolphin") for name in settle_names)
        assert [row["launches"] for row in env.stats()] == [1, 1, 1]
        assert [row["relaunches"] for row in env.stats()] == [0, 0, 0]
        assert all(row["launch_seconds"] >= 0.0 for row in env.stats())
        # Stepping afterwards works as before.
        output = env.step({0: _control(3)})
        assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX + 1] * 3
    finally:
        env.close()
    env, backend = _env(_three(), boot="sequential")
    try:
        assert backend.timeline == [
            ("run", 0),
            ("connect", 0),
            ("run", 1),
            ("connect", 1),
            ("run", 2),
            ("connect", 2),
        ]
        assert all(name == "MainThread" for console in backend.consoles for name in console.step_threads)
        assert env.boot_mode == "sequential" and [row["boot"] for row in env.stats()] == ["sequential"] * 3
    finally:
        env.close()
    # A two-phase boot whose connect fails falls back to the sequential retries (attempts stay bounded).
    env, backend = _env(
        {0: [Script([], connect_ok=False), Script(_boot())], 1: [Script(_boot())]}, launch_retries=1
    )
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX] * 2 and len(backend.consoles) == 3
        assert [row["launches"] for row in env.stats()] == [2, 1]
        assert "ConnectFailed" in env.stats()[0]["faults"][0]
    finally:
        env.close()
    with pytest.raises(DolphinLaunchError, match="2 launch attempts"):
        _env({0: [Script([], connect_ok=False), Script([], connect_ok=False)]}, launch_retries=1)


def test_one_env_faulting_inside_a_threaded_step_relaunches_alone_on_the_main_thread() -> None:
    scripts = {
        0: [Script(_boot(game_frames=10))],
        1: [Script([*_boot(game_frames=3), None]), Script(_boot(menu_frames=1, game_frames=10))],
        2: [Script(_boot(game_frames=10))],
    }
    env, backend = _env(scripts, console_timeout_s=0.001)
    try:
        assert env.step_threads == 3
        neutral = _control(3)
        for _ in range(2):
            env.step({0: neutral})
        output = env.step({0: neutral})  # env 1's console times out in game -> relaunch -> a -123 row
        output.validate(3)
        start = INITIAL_FRAME_INDEX
        assert output.frame_index.tolist() == [start + 3, start, start + 3]
        assert output.needs_reset.tolist() == [False, True, False]
        assert [row["relaunches"] for row in env.stats()] == [0, 1, 0]
        assert "ConsoleTimeout" in env.stats()[1]["faults"][0]
        assert env.stats()[0]["faults"] == [] and env.stats()[2]["faults"] == []
        assert len(backend.consoles) == 4 and backend.console_threads[-1] == "MainThread"
        assert backend.consoles[1].stopped == 1 and backend.consoles[0].stopped == 0
        # The other environments' frames are untouched (the scripted p0.x is -20 + frame).
        assert output.frames[0]["p0"]["x"].tolist() == [-20.0 + start + 3, -20.0 + start, -20.0 + start + 3]
        after = env.step({0: neutral})
        assert after.frame_index.tolist() == [start + 4, start + 1, start + 4]
        assert after.needs_reset.tolist() == [False] * 3
    finally:
        env.close()


def test_step_threads_and_boot_knobs() -> None:
    config = DolphinEnvConfig()
    assert config.step_threads == 0 and config.boot == "two_phase" and config.resolved_step_threads == 1
    assert DolphinEnvConfig(num_envs=8).resolved_step_threads == 8
    assert DolphinEnvConfig(num_envs=8, step_threads=3).resolved_step_threads == 3
    assert DolphinEnvConfig(num_envs=8, step_threads=1).resolved_step_threads == 1
    assert dolphin_lib.BOOT_MODES == ("sequential", "two_phase")
    with pytest.raises(ValueError, match="step_threads"):
        DolphinEnvConfig(step_threads=-1)
    with pytest.raises(ValueError, match="boot"):
        DolphinEnvConfig(boot="warp")
    env, _ = _env({0: [Script(_boot())], 1: [Script(_boot())]}, step_threads=1)
    try:
        assert env.step_threads == 1 and env.boot_mode == "two_phase" and not env.pool_active
        assert env.stats()[0]["step_threads"] == 1 and env.stats()[1]["boot"] == "two_phase"
    finally:
        env.close()
    env, _ = _env({0: [Script(_boot())], 1: [Script(_boot())]}, step_threads=2, boot="sequential")
    try:
        assert env.step_threads == 2 and env.boot_mode == "sequential" and env.pool_active
        env.step({0: _control(2)})
    finally:
        env.close()
    assert not env.pool_active
    env.close()  # idempotent with the pool gone
    assert not env.pool_active


def test_chunk_frames_knob_and_stats_step_timings() -> None:
    """P8: ``chunk_frames`` validates >= 1; stats rows carry the env_s micro-decomposition."""

    assert DolphinEnvConfig().chunk_frames == 1
    assert DolphinEnvConfig(chunk_frames=4, num_envs=1).chunk_frames == 4
    with pytest.raises(ValueError, match="chunk_frames"):
        DolphinEnvConfig(chunk_frames=0)
    env, _ = _env({0: [Script(_boot())], 1: [Script(_boot())]})
    try:
        before = env.stats()
        assert [row["step_seconds"] for row in before] == [0.0, 0.0]
        assert [row["convert_seconds"] for row in before] == [0.0, 0.0]
        env.step({0: _control(2)})
        env.step({0: _control(2)})
        stats = env.stats()
        for row in stats:
            assert isinstance(row["step_seconds"], float) and row["step_seconds"] > 0.0
            assert isinstance(row["convert_seconds"], float) and row["convert_seconds"] > 0.0
            assert row["step_seconds"] + row["convert_seconds"] < 60.0
    finally:
        env.close()


# ---------------------------------------------------------------------------
# P9: per-environment replay directories and the graceful stop
# ---------------------------------------------------------------------------


class FakeProcess:
    """What ``Console._process`` is: a ``subprocess.Popen`` we may terminate, wait for and kill."""

    def __init__(self, *, exits_on_terminate: bool = True) -> None:
        self.exits_on_terminate = exits_on_terminate
        self.terminated = 0
        self.killed = 0
        self.waits: list[float | None] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        if self.exits_on_terminate:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("dolphin-emu", timeout or 0.0)
        return self.returncode

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9


def test_instance_replay_dir_is_per_environment_and_survives_worker_slicing(tmp_path: Path) -> None:
    """Slippi names replays ``Game_<UTC>.slp``, so every Dolphin needs its own directory (P9)."""

    off = DolphinEnvConfig(num_envs=2, save_replays=False, replay_dir=str(tmp_path))
    assert dolphin_lib.instance_replay_dir(off, 0) is None
    assert dolphin_lib.instance_replay_dir(DolphinEnvConfig(num_envs=2, save_replays=True), 0) is None
    on = DolphinEnvConfig(num_envs=4, save_replays=True, replay_dir=str(tmp_path / "clips"))
    first = dolphin_lib.instance_replay_dir(on, 0)
    third = dolphin_lib.instance_replay_dir(on, 2)
    assert first == str(tmp_path / "clips" / "env0") and third == str(tmp_path / "clips" / "env2")
    assert Path(first).is_dir() and Path(third).is_dir()  # Dolphin does not create it itself
    assert dolphin_lib.instance_replay_dir(on, 0) == first  # idempotent
    assert DolphinEnvConfig().replay_monthly_folders is False
    # Worker processes slice the environments; the directory follows the GLOBAL index, never the local one.
    from melee_rl.env.dolphin_mp import worker_configs

    sliced = worker_configs(replace(on, worker_processes=2))
    assert [config.env_index_offset for config in sliced] == [0, 2]
    names = [dolphin_lib.instance_replay_dir(config, j) for config in sliced for j in range(2)]
    assert names == [str(tmp_path / "clips" / f"env{i}") for i in range(4)]
    assert len(set(names)) == 4
    with pytest.raises(ValueError, match="env_index_offset"):
        DolphinEnvConfig(env_index_offset=-1)


def test_libmelee_backend_passes_the_per_env_replay_dir(tmp_path: Path) -> None:
    """``LibmeleeBackend.make_console`` is where the per-instance directory reaches libmelee."""

    calls: list[dict[str, Any]] = []

    def console(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    stub = SimpleNamespace(Console=console)
    backend = dolphin_lib.LibmeleeBackend(stub)
    backend.info = BuildInfo(mainline=True, version="fake ExiAI", exi_ai=True, exe="/opt/dolphin-emu")
    config = DolphinEnvConfig(num_envs=2, save_replays=True, replay_dir=str(tmp_path / "clips"))
    backend.make_console(config, 1, 51442)
    assert calls[-1]["save_replays"] is True
    assert calls[-1]["replay_dir"] == str(tmp_path / "clips" / "env1")
    assert calls[-1]["gfx_backend"] == "Null"  # the ExiAI build accepts nothing else
    # Dolphin's own default is Slippi's <dir>/<YYYY-MM>/ layout, which libmelee leaves unset: a clip
    # must be exactly where we put its Dolphin, so the key is written explicitly (probe, 26 Aug 2026).
    assert calls[-1]["replay_monthly_folders"] is False
    backend.make_console(replace(config, save_replays=False), 1, 51442)
    assert calls[-1]["save_replays"] is False and calls[-1]["replay_dir"] is None


def test_terminate_console_and_stop_timeout(tmp_path: Path) -> None:
    """``Console.stop`` kills, which truncates a ``.slp``; ``stop_timeout_s`` sends SIGTERM first (P9)."""

    assert DolphinEnvConfig().stop_timeout_s == 0.0
    with pytest.raises(ValueError, match="stop_timeout_s"):
        DolphinEnvConfig(stop_timeout_s=-1.0)
    clean = FakeProcess()
    assert dolphin_lib.terminate_console(SimpleNamespace(_process=clean), 5.0) is True
    assert clean.terminated == 1 and clean.waits == [5.0] and clean.killed == 0
    assert dolphin_lib.terminate_console(SimpleNamespace(_process=clean), 5.0) is True  # already exited
    assert clean.terminated == 1
    stuck = FakeProcess(exits_on_terminate=False)
    assert dolphin_lib.terminate_console(SimpleNamespace(_process=stuck), 0.01) is False
    assert stuck.terminated == 1 and stuck.waits == [0.01]
    # With blocking input the emulator is parked waiting for a controller frame, so it cannot see the
    # signal until it runs one: the pump is called between polls and its errors are ignored.
    pumped = FakeProcess(exits_on_terminate=False)
    calls: list[int] = []

    def pump() -> None:
        calls.append(len(calls))
        if len(calls) == 1:
            raise OSError("the pipes go away while Dolphin shuts down")
        pumped.returncode = 0

    assert dolphin_lib.terminate_console(SimpleNamespace(_process=pumped), 5.0, pump) is True
    assert pumped.terminated == 1 and pumped.waits == [] and len(calls) == 2
    assert dolphin_lib.terminate_console(SimpleNamespace(_process=None), 5.0) is False
    assert dolphin_lib.terminate_console(SimpleNamespace(), 5.0) is False
    # The environment: stop_timeout_s = 0 keeps today's kill; a positive value terminates first.
    for timeout, terminated in ((0.0, 0), (5.0, 1)):
        env, backend = _env({0: [Script(_boot())]}, stop_timeout_s=timeout)
        process = FakeProcess()
        backend.consoles[0]._process = process
        env.close()
        assert process.terminated == terminated
        assert backend.consoles[0].stopped == 1  # Console.stop still runs (it removes the temp home)


# ---------------------------------------------------------------------------
# P11: the seed knobs
# ---------------------------------------------------------------------------


def test_seed_knobs_are_off_by_default_and_indexed_by_the_global_environment() -> None:
    default = DolphinEnvConfig()
    assert default.rtc_values == () and default.menu_idle_frames == ()
    assert default.rtc_value(0) is None and default.menu_idle(0) == 0

    # A worker slice indexes the full tuple through env_index_offset, exactly as replay dirs do.
    seeded = DolphinEnvConfig(
        num_envs=2, rtc_values=(11, 22, 33, 44), menu_idle_frames=(0, 5, 10, 15), env_index_offset=2
    )
    assert (seeded.rtc_value(0), seeded.rtc_value(1)) == (33, 44)
    assert (seeded.menu_idle(0), seeded.menu_idle(1)) == (10, 15)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_envs": 2, "rtc_values": (1,)}, "rtc_values is indexed"),
        ({"num_envs": 1, "env_index_offset": 1, "menu_idle_frames": (5,)}, "menu_idle_frames is indexed"),
        ({"rtc_values": (-1,)}, "rtc_values entries must be >= 0"),
        ({"menu_idle_frames": (9000,)}, "menu_idle_frames must stay under menu_frame_cap"),
    ],
)
def test_seed_knobs_reject_a_configuration_that_cannot_be_indexed(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DolphinEnvConfig(**overrides)


def test_menu_idle_frames_hold_the_autostart_so_a_start_can_be_seeded() -> None:
    # Melee's RNG advances once per menu frame, so the frame START is pressed on decides the seed the
    # match starts from.  Default: slippi-ai's 30-frame rule.
    env, backend = _env({0: [Script(_boot(menu_frames=80))]}, players=("policy", "policy"))
    try:
        assert [call[2]["autostart"] for call in backend.helpers[0].calls] == [False] * 31 + [True] * 51
    finally:
        env.close()
    # Seeded: START waits for the configured hold instead.
    env, backend = _env(
        {0: [Script(_boot(menu_frames=80))]}, players=("policy", "policy"), menu_idle_frames=(60,)
    )
    try:
        assert [call[2]["autostart"] for call in backend.helpers[0].calls] == [False] * 61 + [True] * 21
    finally:
        env.close()


def test_write_custom_rtc_adds_the_determinism_keys_without_losing_libmelees(tmp_path: Path) -> None:
    config_dir = tmp_path / "User" / "Config"
    config_dir.mkdir(parents=True)
    (config_dir / "Dolphin.ini").write_text(
        "[Core]\ngfxbackend = Null\nemulationspeed = 0.0\n\n[Slippi]\nsavereplays = True\n"
    )

    path = dolphin_lib.write_custom_rtc(config_dir, 1_700_000_000)

    text = path.read_text()
    assert "CustomRTCEnable = True" in text
    assert "CustomRTCValue = 1700000000" in text
    assert "gfxbackend = Null" in text and "savereplays = True" in text  # libmelee's own keys survive


def test_write_custom_rtc_creates_the_file_when_libmelee_has_not_written_one(tmp_path: Path) -> None:
    path = dolphin_lib.write_custom_rtc(tmp_path / "Config", 42)

    assert path == tmp_path / "Config" / "Dolphin.ini"
    assert "CustomRTCValue = 42" in path.read_text()


def test_console_config_dir_follows_libmelees_temporary_home(tmp_path: Path) -> None:
    console = SimpleNamespace(dolphin_home_path=str(tmp_path / "User"))

    assert dolphin_lib.console_config_dir(console) == tmp_path / "User" / "Config"


def test_stages_cycle_over_the_global_environment_index_and_the_game_start_checks_them() -> None:
    """``stages`` (user decision 2 Sep 2026: self-play on every legal stage, not only FD).  Non-empty,
    environment ``i`` plays ``stages[(env_index_offset + i) % len]`` -- balanced by construction: 96
    training Dolphins over six stages play 16 each, 30 eval Dolphins 5 each -- and worker slices keep
    the global numbering.  A game that starts on another stage is a relaunchable fault, like a wrong
    character."""

    config = DolphinEnvConfig(num_envs=4, stages=(25, 24, 18))
    assert [config.stage_for(i) for i in range(4)] == [25, 24, 18, 25]
    assert config.stage_ids == (18, 24, 25)
    assert DolphinEnvConfig(num_envs=2, env_index_offset=2, stages=(25, 24, 18)).stage_for(0) == 18
    plain = DolphinEnvConfig(num_envs=2)
    assert plain.stage_for(1) == 25 and plain.stage_ids == (25,)
    with pytest.raises(ValueError, match="stages"):
        DolphinEnvConfig(stages=(25, 64))
    with pytest.raises(ValueError, match="stages"):
        DolphinEnvConfig(stages=(-1,))
    from melee_rl.env.dolphin_mp import worker_configs

    sliced = worker_configs(DolphinEnvConfig(num_envs=6, worker_processes=3, stages=(25, 24)))
    assert [[c.stage_for(j) for j in range(2)] for c in sliced] == [[25, 24], [25, 24], [25, 24]]
    sliced = worker_configs(DolphinEnvConfig(num_envs=6, worker_processes=2, stages=(25, 24, 18)))
    assert [[c.stage_for(j) for j in range(3)] for c in sliced] == [[25, 24, 18], [25, 24, 18]]
    sliced = worker_configs(DolphinEnvConfig(num_envs=4, worker_processes=2, stages=(25, 24, 18)))
    assert [[c.stage_for(j) for j in range(2)] for c in sliced] == [[25, 24], [18, 25]]
    # Each Dolphin's menu helpers select that Dolphin's stage; the frame tree carries what the game reports.
    env, backend = _env({0: [Script(_boot(stage=FD))], 1: [Script(_boot(stage=BF))]}, stages=(25, 24))
    try:
        output = env.current()  # the Dolphins boot at construction; reset() would relaunch them all
        assert output.frames[0]["stage"].tolist() == [25, 24]
        assert [helper.calls[0][2]["stage_selected"] for helper in backend.helpers] == [FD, FD, BF, BF]
        assert [row["stage"] for row in env.stats()] == [25, 24]
    finally:
        env.close()
    # The game started on Battlefield where Final Destination was configured: fault, relaunch, then fine.
    env, backend = _env({0: [Script(_boot(stage=BF)), Script(_boot(stage=FD))]}, stages=(25,))
    try:
        assert _frames(env) == [INITIAL_FRAME_INDEX] and len(backend.consoles) == 2
        fault = env.stats()[0]["faults"][0]
        assert "WrongStage" in fault and "BATTLEFIELD" in fault and "FINAL_DESTINATION" in fault
        assert env.stats()[0]["stage"] == 25 and env.current().frames[0]["stage"].tolist() == [25]
    finally:
        env.close()


def test_opponent_characters_are_per_environment_and_checked_at_each_game_start() -> None:
    """``opponent_characters`` (17 Sep 2026, the league run): player 1's character per environment, indexed by
    the global environment index like ``rtc_values`` -- a mix arm of twelve characters needs a Dolphin per
    matchup, and a Dolphin cannot change character without a relaunch. Empty (the default) keeps
    ``characters`` on every Dolphin; port 0 is always ``characters[0]``."""

    from melee_rl.env.dolphin_mp import worker_configs

    config = DolphinEnvConfig(num_envs=4, opponent_characters=(22, 18, 9, 10))
    assert [config.characters_for(i) for i in range(4)] == [(1, 22), (1, 18), (1, 9), (1, 10)]
    assert config.character_ids == (1, 9, 10, 18, 22)
    shifted = DolphinEnvConfig(num_envs=2, env_index_offset=2, opponent_characters=(22, 18, 9, 10))
    assert [shifted.characters_for(i) for i in range(2)] == [(1, 9), (1, 10)]
    plain = DolphinEnvConfig(num_envs=2, characters=(1, 22))
    assert plain.opponent_characters == () and plain.characters_for(1) == (1, 22)
    assert plain.character_ids == (1, 22)
    with pytest.raises(ValueError, match="opponent_characters"):
        DolphinEnvConfig(
            num_envs=4, opponent_characters=(22, 18)
        )  # needs env_index_offset + num_envs entries
    with pytest.raises(ValueError, match="opponent_characters"):
        DolphinEnvConfig(num_envs=1, opponent_characters=(33,))
    with pytest.raises(ValueError, match="opponent_characters"):
        DolphinEnvConfig(num_envs=1, opponent_characters=(-1,))
    sliced = worker_configs(
        DolphinEnvConfig(num_envs=4, worker_processes=2, opponent_characters=(22, 18, 9, 10))
    )
    assert [[c.characters_for(j) for j in range(2)] for c in sliced] == [
        [(1, 22), (1, 18)],
        [(1, 9), (1, 10)],
    ]
    # Each Dolphin's menu helpers select that Dolphin's pair; the game start is checked against it.
    scripts = {
        0: [Script(_boot(characters=(FOX, FALCO)))],
        1: [Script(_boot(characters=(FOX, melee.Character.MARTH)))],
    }
    env, backend = _env(scripts, opponent_characters=(22, 18))
    try:
        output = env.current()
        assert output.frames[0]["p1"]["character"].tolist() == [22, 18]
        selected = [helper.calls[0][2]["character_selected"] for helper in backend.helpers]
        assert selected == [FOX, FALCO, FOX, melee.Character.MARTH]
        assert [row["characters"] for row in env.stats()] == [(1, 22), (1, 18)]
    finally:
        env.close()
    # Dolphin 1 started as Falco where Marth was configured: a relaunchable fault on that Dolphin only.
    scripts = {
        0: [Script(_boot(characters=(FOX, FALCO)))],
        1: [Script(_boot(characters=(FOX, FALCO))), Script(_boot(characters=(FOX, melee.Character.MARTH)))],
    }
    env, backend = _env(scripts, opponent_characters=(22, 18))
    try:
        assert env.current().frames[0]["p1"]["character"].tolist() == [22, 18]
        stats = env.stats()
        assert stats[0]["faults"] == [] and len(stats[1]["faults"]) == 1
        assert "WrongCharacter" in stats[1]["faults"][0] and "MARTH" in stats[1]["faults"][0]
    finally:
        env.close()
