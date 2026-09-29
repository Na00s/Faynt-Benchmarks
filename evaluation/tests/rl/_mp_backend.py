"""Picklable fake-Dolphin factories for the worker-process tests (P7-PLAN.md 2.3).

A worker process cannot receive live objects, so ``tests/rl/test_dolphin_mp.py`` passes one of the
module-level factory functions below (pickled by reference through ``multiprocessing``) and every
worker builds its scripted backend inside the child.  The fakes mirror
``tests/rl/test_dolphin_env.py``'s, reduced to what the worker path needs (no thread bookkeeping);
the gamestates are real libmelee dataclasses, so ``FrameConverter`` runs for real inside the worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import melee
import numpy as np

from melee_rl.env.dolphin import BuildInfo, DolphinEnvConfig
from melee_rl.env.protocol import INITIAL_FRAME_INDEX
from tensor_batch import BUTTON_ORDER

IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)
FOX = melee.Character.FOX


# ---------------------------------------------------------------------------
# gamestate builders (real libmelee dataclasses)
# ---------------------------------------------------------------------------


def _player(x: float, cpu_level: int) -> Any:
    player = melee.PlayerState()
    player.character = FOX
    player.cpu_level = cpu_level
    player.position = melee.Position(np.float32(x), np.float32(0.0))
    player.action = melee.Action.STANDING
    player.jumps_left = 2
    player.controller_state = melee.ControllerState()
    return player


def _menu(menu: Any = melee.Menu.CHARACTER_SELECT) -> Any:
    state = melee.GameState()
    state.menu_state = menu
    state.players = {port: melee.PlayerState() for port in (1, 2, 3, 4)}
    if menu is melee.Menu.CHARACTER_SELECT:
        state.players[1].controller_status = melee.ControllerStatus.CONTROLLER_HUMAN
        opponent = state.players[2]
        opponent.controller_status = melee.ControllerStatus.CONTROLLER_CPU
        opponent.cpu_level = 9
        opponent.is_holding_cpu_slider = False
    return state


def _projectile(spawn_id: int) -> Any:
    return melee.Projectile(
        position=melee.Position(np.float32(spawn_id), np.float32(0.0)),
        owner=1,
        type=melee.ProjectileType.FOX_LASER,
        subtype=0,
        spawn_id=np.uint32(spawn_id),
    )


def _ingame(frame: int, *, spawn_id: int | None = None) -> Any:
    state = melee.GameState()
    state.frame = frame
    state.stage = melee.Stage.FINAL_DESTINATION
    state.menu_state = melee.Menu.IN_GAME
    state.players = {1: _player(-20.0 + frame, 0), 2: _player(20.0 - frame, 9)}
    state.projectiles = [] if spawn_id is None else [_projectile(spawn_id)]
    return state


def _game(count: int, start: int = INITIAL_FRAME_INDEX, *, spawn_id: int | None = None) -> list[Any]:
    return [_ingame(start + i, spawn_id=spawn_id) for i in range(count)]


def _boot(menu_frames: int = 3, game_frames: int = 400, *, spawn_id: int | None = None) -> list[Any]:
    return [
        _menu(melee.Menu.MAIN_MENU),
        *[_menu()] * menu_frames,
        _menu(melee.Menu.STAGE_SELECT),
        *_game(game_frames, spawn_id=spawn_id),
    ]


# ---------------------------------------------------------------------------
# the fakes (constructed inside the worker by a factory below)
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
        self.main = (0.5, 0.5)
        self.c = (0.5, 0.5)
        self.shoulder = 0.0
        self.buttons = dict.fromkeys(BUTTON_ORDER, False)

    def connect(self) -> bool:
        self.console.controllers.append(self)
        return True

    def disconnect(self) -> None:
        pass

    def press_button(self, button: Any) -> None:
        self.buttons[button.value] = True

    def release_button(self, button: Any) -> None:
        self.buttons[button.value] = False

    def tilt_analog(self, button: Any, x: float, y: float) -> None:
        if button is melee.Button.BUTTON_MAIN:
            self.main = (x, y)
        else:
            self.c = (x, y)

    def press_shoulder(self, button: Any, amount: float) -> None:
        self.shoulder = amount

    def flush(self) -> None:
        pass


class FakeConsole:
    def __init__(self, events: list[Any], *, connect_ok: bool, index: int, poll_timeout: float) -> None:
        self.events = list(events)
        self.connect_ok = connect_ok
        self.index = index
        self.poll_timeout = poll_timeout
        self.controllers: list[FakeController] = []
        self.stopped = 0
        self.steps = 0
        self._process = None

    def run(self, iso_path: str | None = None, platform: str | None = None) -> None:
        pass

    def connect(self) -> bool:
        return self.connect_ok

    def get_dolphin_pipes_path(self, port: int) -> None:
        return None

    def stop(self) -> None:
        self.stopped += 1

    def step(self) -> Any:
        for controller in self.controllers:
            controller.flush()
        self.steps += 1
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
    def menu_helper_simple(self, gamestate: Any, controller: Any, **kwargs: Any) -> None:
        pass


@dataclass
class FakeBackendMP:
    """One ``Script`` per launch per (worker-local) env index; records the consoles it made."""

    scripts: dict[int, list[Script]]
    launches: dict[int, int] = field(default_factory=dict)
    consoles: list[FakeConsole] = field(default_factory=list)

    def prepare(self, config: DolphinEnvConfig) -> BuildInfo:
        return BuildInfo(
            mainline=True, version="fake 4.0.0-mainline ExiAI", exi_ai=True, exe=config.dolphin_path
        )

    def pick_port(self, candidate: int) -> int:
        return candidate

    def make_console(self, config: DolphinEnvConfig, index: int, slippi_port: int) -> FakeConsole:
        launch = self.launches.get(index, 0)
        self.launches[index] = launch + 1
        scripts = self.scripts[index]
        if launch >= len(scripts):
            raise AssertionError(f"env {index}: unexpected launch #{launch + 1} ({len(scripts)} scripted)")
        script = scripts[launch]
        console = FakeConsole(
            script.events, connect_ok=script.connect_ok, index=index, poll_timeout=config.console_timeout_s
        )
        self.consoles.append(console)
        return console

    def make_controller(self, console: FakeConsole, port: int) -> FakeController:
        return FakeController(console, port)

    def make_menu_helper(self) -> FakeMenuHelper:
        return FakeMenuHelper()

    def start(self, console: FakeConsole, config: DolphinEnvConfig) -> None:
        console.run(iso_path=config.iso_path, platform="headless")


# ---------------------------------------------------------------------------
# factories (module-level: picklable by reference; called inside the worker)
# ---------------------------------------------------------------------------


def plain_backend(config: DolphinEnvConfig, worker_index: int) -> FakeBackendMP:
    """Every env: two scripted launches (boot + one relaunch for ``reset()``), 400 game frames each."""

    return FakeBackendMP({i: [Script(_boot()), Script(_boot(menu_frames=1))] for i in range(config.num_envs)})


def numbered_backend(config: DolphinEnvConfig, worker_index: int) -> FakeBackendMP:
    """Like :func:`plain_backend`, but every game frame of worker-local env ``i`` carries one
    projectile at x = ``10 + 10 * worker_index + i`` -- rows are distinguishable across envs and
    workers (the parity and slicing tests read it back from ``items.x``)."""

    return FakeBackendMP(
        {i: [Script(_boot(spawn_id=10 + 10 * worker_index + i))] for i in range(config.num_envs)}
    )


def flaky_backend(config: DolphinEnvConfig, worker_index: int) -> FakeBackendMP:
    """Worker 1's local env 0 times out in game after 3 frames (one relaunch); everything else is plain."""

    scripts = {i: [Script(_boot())] for i in range(config.num_envs)}
    if worker_index == 1:
        scripts[0] = [Script([*_boot(game_frames=3), None]), Script(_boot(menu_frames=1))]
    return FakeBackendMP(scripts)


def failing_backend(config: DolphinEnvConfig, worker_index: int) -> FakeBackendMP:
    """Every launch of every env fails to connect (boot failure paths)."""

    return FakeBackendMP({i: [Script([], connect_ok=False)] for i in range(config.num_envs)})
