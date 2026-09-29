"""vladfi1's Phillip as a live opponent (P32, 18 Sep 2026).

`vladfi1/phillip <https://github.com/vladfi1/phillip>`_ (GPL-3.0, pinned at :data:`PHILLIP_REVISION`)
is the 2017-2019 deep-RL Melee agent that preceded slippi-ai: a small actor-critic (an MLP, a GRU in the
later agents) over a hand-built embedding of both players' memory fields -- position, action state,
percent, speeds, hitlag and hitstun, jumps, shield -- that picks **one of 30-54 discrete controller
actions every** ``act_every`` **frames** (2, 3 or 4) and holds it through an action chain in between.
Delay, where an agent has it, is a queue of whole decisions inside the agent.  Trained against
Dolphin's CPU and by self-play with a damage-and-stocks reward, mostly on Final Destination.  The
repository ships fourteen agents with weights (:data:`IN_REPO_AGENTS`), seven of them at delay 0
(:data:`DELAY0_AGENTS`); the delay-18/21 agents of the README's Google Drive zip are not in git.

**Why a sidecar.**  Upstream pins ``tensorflow==2.13`` (``setup.py``, ``python_requires='<3.12'``), so it
cannot import in the recorder's Python 3.12 environment.  Its ``Agent`` therefore runs in its own
interpreter -- ``/opt/phillip/venv/bin/python``, 3.11 with ``tensorflow-cpu`` -- as a subprocess
(:mod:`melee_rl.phillip_sidecar`) that this module drives over a line protocol: the recorder converts
each libmelee gamestate into the fields of upstream's ``GameMemory`` (:func:`game_message`), the sidecar
calls ``Agent.act(state, pad)`` exactly as ``phillip/cpu.py`` does, and the controller upstream pressed
comes back and is applied on Phillip's port through the usual ``act_state`` seam.  Exactness is by
construction (their code, their weights, their action chains and banned-move rules). The separate
interpreter is a compatibility boundary. Process separation does not establish license clearance;
users retain the applicable upstream obligations described in ``THIRD_PARTY_NOTICES.md``.

**The three places the conversion cannot be a plain read**, each pinned by ``tests/rl/test_phillip.py``:

* ``character`` is upstream's ``state.Character`` -- the **character-select id** (Fox 10), read at
  ``803F0E08`` -- while libmelee's ``PlayerState.character`` is the internal id (Fox 1); the map is
  libmelee's own ``enums.from_internal``.
* ``jumps_used`` is the game's counter at ``+0x19C8``; libmelee reports ``jumps_left`` (ground jump
  included, 2 for Fox on the ground), so ``jumps_used = max_jumps - jumps_left`` with the per-character
  maximum from libmelee's ``characterdata.csv``.
* ``charging_smash`` is the smash-charge byte at ``+0x2174`` masked with ``0x2`` -- up while charging
  **and** while swinging -- which the Slippi stream does not export.  It is derived from the action
  state instead: the forward / up / down smash animations (``0x3A-0x40``).  The one field that is an
  approximation (its first frames before the charge window may differ); a single scalar among ~800.

Upstream acts from its 121st in-game frame (``cpu.py:256``), Slippi's frame -3 (:data:`ACT_FROM_FRAME`),
plays player index 1 unless ``swap`` (``agent.py:38``), and its README plays with ``--epsilon 0`` -- the
sampling without the uniform exploration mix -- which is the default here (``PhillipConfig.epsilon``).
The sticks reach libmelee with the two-decimal rounding of upstream's pipe writer and should go through
``[env.dolphin] legacy_analog_ports`` on Phillip's port, so the game reads what upstream's Dolphin read.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import select
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import torch

from melee_rl.env.dolphin import PORTS, ControllerRow, neutral_row
from tensor_batch import BUTTON_ORDER

LOG: Final[logging.Logger] = logging.getLogger(__name__)

PHILLIP_REPO_URL: Final[str] = "https://github.com/vladfi1/phillip.git"
PHILLIP_REVISION: Final[str] = "114aea4c446b31359c5637b271db52b200440caa"
"""``Remove git-lfs tracked agents, stop using git-lfs`` (6 Jul 2026), the repository's HEAD."""
PHILLIP_LICENSE: Final[str] = "GPL-3.0"
DEFAULT_SOURCE_DIR: Final[str] = "/opt/phillip/source"
"""Default location for the separately supplied :data:`PHILLIP_REVISION` source tree."""
DEFAULT_PYTHON: Final[str] = "/opt/phillip/venv/bin/python"
"""The sidecar interpreter: Python 3.11 with :data:`TENSORFLOW_PIN` (upstream's pin; 3.12 cannot run it)."""
TENSORFLOW_PIN: Final[str] = "tensorflow-cpu==2.13.1"
"""Upstream's ``tensorflow==2.13`` (``fcf75d3``: "Actually need tf 2.13, not 2.14"), CPU wheel, last patch."""
SIDECAR_SCRIPT: Final[Path] = Path(__file__).with_name("phillip_sidecar.py")
PROTOCOL_VERSION: Final[str] = "melee_rl.phillip_sidecar.v1"
AGENTS_SUBDIR: Final[str] = "agents"
DEFAULT_AGENT: Final[str] = "delay0/FoxFD"
DELAY0_AGENTS: Final[tuple[str, ...]] = (
    "delay0/FoxFD",
    "delay0/FalcoFD",
    "FoxFD0",
    "MarthFD0",
    "PeachFD",
    "SheikFD",
    "FalconFalconBF",
)
"""The seven agents in git that play at delay 0: the fair fight for a delay-0 model of ours."""
IN_REPO_AGENTS: Final[tuple[str, ...]] = (
    *DELAY0_AGENTS,
    "FoxFD1",
    "MarthFD1",
    "PeachFD1",
    "SheikFD1",
    "PeachFD2",
    "SheikFD2",
    "delay12/MarthFD",
)
"""Every agent whose weights are in the repository at :data:`PHILLIP_REVISION` (the ``*FD1`` play at
3 frames of delay, ``*FD2`` at 6, ``delay12/MarthFD`` at 12; the ``delay18/*`` directories hold params
only)."""
PHILLIP_CHARACTERS: Final[Mapping[str, int]] = MappingProxyType(
    {
        # upstream ``menu_manager.characters`` names -> libmelee ``Character`` (internal) ids
        "fox": 0x01,
        "falco": 0x16,
        "falcon": 0x02,
        "roy": 0x1A,
        "marth": 0x12,
        "zelda": 0x13,
        "sheik": 0x07,
        "mewtwo": 0x10,
        "luigi": 0x11,
        "puff": 0x0F,
        "kirby": 0x04,
        "peach": 0x09,
        "ganon": 0x19,
        "samus": 0x0D,
        "bowser": 0x05,
        "yoshi": 0x0E,
        "dk": 0x03,
    }
)
STAGE_IDS: Final[Mapping[str, int]] = MappingProxyType({"final_destination": 25, "battlefield": 24})
"""Upstream's ``movie.stages`` names -> ``[env.dolphin] stage`` ids (upstream's own ``state.Stage``)."""
MAX_ACTION_STATE: Final[int] = 0x17E
"""``embed.py:296``: the one-hot's size is ``1 + maxAction``; anything above embeds to zeros."""
SMASH_ATTACK_ACTIONS: Final[frozenset[int]] = frozenset(range(0x3A, 0x41))
"""libmelee ``FSMASH_HIGH`` (0x3A) ... ``DOWNSMASH`` (0x40): the animations upstream's smash-charge flag
covers (``state_manager.py:184-185``: ``2 = charging, 3 = attacking`` masked with ``0x2``)."""
ACT_FROM_FRAME: Final[int] = -3
"""Upstream skips its first 120 in-game frames (``cpu.py:256``); Slippi's first in-game frame is -123."""
UNSELECTED_CHARACTER: Final[int] = 25
"""Upstream's ``state.Character.Unselected``: what a character without a character-select id embeds as."""
GAME_MENU: Final[int] = 2
"""Upstream's ``state.Menu.Game``."""
_MISSING_PORT_MENU: Final[tuple[str, ...]] = ("IN_GAME", "SUDDEN_DEATH")


class SidecarError(RuntimeError):
    """The sidecar answered with an error, timed out, or died."""


# ---------------------------------------------------------------------------
# configuration and the agent's own parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhillipConfig:
    """Which Phillip agent plays, on which port, through which interpreter.

    ``agent`` is a directory under ``<source_dir>/agents`` (``delay0/FoxFD``); its ``params`` file says
    which character it plays, so ``[env.dolphin] characters`` must start that character on ``player``
    (checked before any Dolphin boots).  ``player`` is the *environment* player index (0 = Dolphin port 1,
    1 = port 2); upstream's agent plays index 1 by default and index 0 with ``swap``, which the sidecar
    sets from this.  ``epsilon`` is upstream's uniform exploration mix; its README plays with 0.
    ``strict`` lets an exception inside ``Agent.act`` stop the job; lenient counts it and holds a neutral
    input for that frame.  A restore that initialises or pads variables is refused either way.
    ``save_cpu`` gives every agent's TensorFlow session one thread (upstream's ``--save_cpu``): the
    network is tiny and the recorder's cores belong to the Dolphins.
    """

    source_dir: str = DEFAULT_SOURCE_DIR
    python: str = DEFAULT_PYTHON
    agent: str = DEFAULT_AGENT
    player: int = 1
    epsilon: float = 0.0
    seed: int = 0
    strict: bool = True
    save_cpu: bool = True
    act_from_frame: int = ACT_FROM_FRAME
    call_timeout_s: float = 60.0
    load_timeout_s: float = 900.0

    def __post_init__(self) -> None:
        if not self.source_dir:
            raise ValueError("[phillip] source_dir must name the Phillip checkout")
        if not self.python:
            raise ValueError("[phillip] python must name the sidecar interpreter")
        agent = Path(self.agent)
        if not self.agent or agent.is_absolute() or ".." in agent.parts:
            raise ValueError(f"[phillip] agent must be a directory under agents/, got {self.agent!r}")
        if self.player not in (0, 1):
            raise ValueError(f"[phillip] player must be 0 or 1, got {self.player}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"[phillip] epsilon must be in [0, 1], got {self.epsilon}")
        if self.call_timeout_s <= 0 or self.load_timeout_s <= 0:
            raise ValueError("[phillip] call_timeout_s and load_timeout_s must be > 0")

    @property
    def libmelee_port(self) -> int:
        return PORTS[self.player]

    @property
    def libmelee_opponent_port(self) -> int:
        return PORTS[1 - self.player]

    @property
    def agent_dir(self) -> Path:
        return Path(self.source_dir) / AGENTS_SUBDIR / self.agent


@dataclass(frozen=True)
class PhillipAgentInfo:
    """What an agent's ``params`` file says about how it plays (read here; the sidecar reads it again)."""

    name: str
    char: str
    act_every: int
    delay: int
    memory: int
    action_type: str
    stage: str
    recurrent: bool
    predict: bool

    @property
    def delay_frames(self) -> int:
        """Upstream's delay is in decisions (``agent.py:43``); a decision is ``act_every`` frames."""

        return self.delay * self.act_every

    @property
    def libmelee_character(self) -> int:
        return PHILLIP_CHARACTERS[self.char]

    @property
    def stage_id(self) -> int:
        return STAGE_IDS[self.stage]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "delay_frames": self.delay_frames,
            "libmelee_character": self.libmelee_character,
            "stage_id": self.stage_id,
        }


def read_agent_info(agent_dir: str | Path, name: str) -> PhillipAgentInfo:
    """Read ``<agent_dir>/params`` the way upstream's ``util.load_params(path, 'agent')`` does.

    The older agents keep their settings in an ``agent`` sub-table, which upstream folds into the top
    level; the defaults are the ``Option`` defaults of ``RLConfig`` (``act_every`` 3, ``delay`` 0,
    ``memory`` 0), ``RL`` (``action_type`` ``"diagonal"``) and ``CPU`` (``stage`` ``final_destination``).
    """

    path = Path(agent_dir) / "params"
    if not path.is_file():
        raise FileNotFoundError(f"Phillip agent {name!r} has no params file at {path}")
    params = json.loads(path.read_text())
    if isinstance(params.get("agent"), dict):
        params = {**params, **params["agent"]}
    char = params.get("char")
    if not isinstance(char, str) or char not in PHILLIP_CHARACTERS:
        raise ValueError(
            f"Phillip agent {name!r} plays {char!r}, which is not in the character table "
            f"{sorted(PHILLIP_CHARACTERS)}"
        )
    stage = str(params.get("stage", "final_destination"))
    if stage not in STAGE_IDS:
        raise ValueError(f"Phillip agent {name!r} names stage {stage!r}; known: {sorted(STAGE_IDS)}")
    return PhillipAgentInfo(
        name=name,
        char=char,
        act_every=int(params.get("act_every", 3)),
        delay=int(params.get("delay", 0)),
        memory=int(params.get("memory", 0)),
        action_type=str(params.get("action_type", "diagonal")),
        stage=stage,
        recurrent=bool(params.get("recurrent", False)),
        predict=bool(params.get("predict", 0)),
    )


def agent_info(config: PhillipConfig) -> PhillipAgentInfo:
    return read_agent_info(config.agent_dir, config.agent)


def check_phillip_character(config: PhillipConfig, characters: Sequence[int]) -> PhillipAgentInfo:
    """The agent's own character must be what ``[env.dolphin] characters`` starts on its player.

    A Phillip agent is one character (its ``params`` say which); a Dolphin that starts another one on its
    port would make the agent embed a body it never controlled.  Checked before any Dolphin boots.
    """

    info = agent_info(config)
    playing = int(characters[config.player])
    if playing != info.libmelee_character:
        raise ValueError(
            f"[phillip] {config.agent} plays {info.char} (libmelee character {info.libmelee_character}), but "
            f"[env.dolphin] characters starts player {config.player} as {playing}; pass "
            f"env.dolphin.characters=[{characters[0] if config.player else info.libmelee_character},"
            f"{info.libmelee_character if config.player else characters[1]}]"
        )
    return info


# ---------------------------------------------------------------------------
# the field conversion: libmelee GameState -> upstream's GameMemory fields
# ---------------------------------------------------------------------------


def load_max_jumps(melee: Any) -> dict[Any, int]:
    """``Character -> jumps_left on the ground`` from libmelee's ``characterdata.csv`` (its ``Jumps`` column
    counts midair jumps; libmelee's ``jumps_left`` includes the ground jump)."""

    table = Path(melee.__file__).parent / "characterdata.csv"
    jumps: dict[Any, int] = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                character = melee.Character(int(row["CharacterIndex"]))
            except ValueError:
                continue
            jumps[character] = int(row["Jumps"]) + 1
    return jumps


def player_message(state: Any, *, melee: Any, max_jumps: Mapping[Any, int]) -> dict[str, Any]:
    """One libmelee ``PlayerState`` as the ``PlayerMemory`` fields upstream embeds (``embed.py:323-346``)."""

    action = getattr(state.action, "value", -1)
    action_state = min(max(int(action), 0), MAX_ACTION_STATE)
    external = melee.enums.from_internal(state.character)
    character = int(external) if 0 <= int(external) < UNSELECTED_CHARACTER else UNSELECTED_CHARACTER
    jumps_used = max(0, int(max_jumps.get(state.character, 2)) - int(state.jumps_left))
    return {
        "percent": max(0, int(state.percent)),  # upstream reads the displayed short (+0x60)
        "stock": max(0, int(state.stock)),
        "facing": 1.0 if state.facing else -1.0,
        "x": float(state.position.x),
        "y": float(state.position.y),
        "z": 0.0,
        "action_state": action_state,
        "action_counter": 0,  # not embedded (``embed.py:329``)
        "action_frame": float(state.action_frame),
        "character": character,
        "invulnerable": bool(state.invulnerable),
        "hitlag_frames_left": float(state.hitlag_left),
        "hitstun_frames_left": float(state.hitstun_frames_left),
        "jumps_used": jumps_used,
        "charging_smash": action_state in SMASH_ATTACK_ACTIONS,
        "in_air": not bool(state.on_ground),
        "speed_air_x_self": float(state.speed_air_x_self),
        "speed_ground_x_self": float(state.speed_ground_x_self),
        "speed_y_self": float(state.speed_y_self),
        "speed_x_attack": float(state.speed_x_attack),
        "speed_y_attack": float(state.speed_y_attack),
        "shield_size": float(state.shield_strength),
        "cursor_x": 0.0,
        "cursor_y": 0.0,
    }


def game_message(gamestate: Any, *, melee: Any, max_jumps: Mapping[Any, int]) -> dict[str, Any]:
    """The whole frame: players in Dolphin port order (upstream's ``players[0]`` is port 1)."""

    return {
        "frame": int(gamestate.frame),
        "menu": GAME_MENU,
        "stage": 0,  # not embedded (``embed.py:383``) and never read by ``Agent.act``
        "players": [
            player_message(gamestate.players[port], melee=melee, max_jumps=max_jumps) for port in PORTS
        ],
    }


def row_from_controller(payload: Mapping[str, Any]) -> ControllerRow:
    """The sidecar's controller -> a :class:`ControllerRow` (``START`` dropped, ``D_UP`` never pressed)."""

    buttons = dict(payload["buttons"])
    unknown = set(buttons) - set(BUTTON_ORDER) - {"START"}
    if unknown:
        raise ValueError(f"unknown buttons from the sidecar: {sorted(unknown)}")
    main = payload["main"]
    c = payload["c"]
    return ControllerRow(
        main_x=float(main[0]),
        main_y=float(main[1]),
        c_x=float(c[0]),
        c_y=float(c[1]),
        shoulder=0.0,  # upstream presses L digitally; it never sets an analog shoulder
        buttons=tuple(bool(buttons.get(name, False)) for name in BUTTON_ORDER),
    )


# ---------------------------------------------------------------------------
# the sidecar process and its line protocol
# ---------------------------------------------------------------------------


class _LineReader:
    """Newline-delimited reads off a pipe fd with a deadline (``select`` + ``os.read``)."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = bytearray()

    def read_line(self, timeout_s: float) -> bytes | None:
        """One line without its newline; ``None`` on timeout; ``b""`` at end of file."""

        deadline = time.monotonic() + timeout_s
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self._fd, 1 << 16)
            if not chunk:
                return b""
            self._buffer += chunk


class PhillipSidecar:
    """The subprocess and the protocol: ``call(op, **payload) -> response`` or :class:`SidecarError`.

    ``source_dir`` goes first on the child's ``PYTHONPATH`` so ``import phillip`` finds the checkout
    (or, in the tests, the stub package).  The GPU is hidden from the child: the sidecar is CPU-only and
    the L4 belongs to our own policy.  Stderr is drained by a thread into a tail that every error
    carries, because that is where upstream's own diagnostics go.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        source_dir: str,
        call_timeout_s: float = 60.0,
        stderr_lines: int = 400,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if call_timeout_s <= 0:
            raise ValueError("call_timeout_s must be > 0")
        self.command = tuple(str(part) for part in command)
        self.source_dir = source_dir
        self.call_timeout_s = float(call_timeout_s)
        self._env = env
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: _LineReader | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._process is not None:
            return
        environment = dict(os.environ if self._env is None else self._env)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = self.source_dir + (os.pathsep + existing if existing else "")
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment.setdefault("PYTHONSAFEPATH", "1")  # 3.11+: the script's own directory stays off sys.path
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        environment["CUDA_VISIBLE_DEVICES"] = ""
        self._process = subprocess.Popen(
            list(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        assert self._process.stdout is not None
        self._reader = _LineReader(self._process.stdout.fileno())
        threading.Thread(target=self._drain_stderr, name="phillip-sidecar-stderr", daemon=True).start()

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for raw in process.stderr:
            self._stderr.append(raw.decode("utf-8", "replace").rstrip("\n"))

    def stderr_tail(self, lines: int = 30) -> str:
        return "\n".join(list(self._stderr)[-lines:])

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def call(self, op: str, *, timeout_s: float | None = None, **payload: Any) -> dict[str, Any]:
        self.start()
        assert self._process is not None and self._reader is not None
        timeout = self.call_timeout_s if timeout_s is None else float(timeout_s)
        request = json.dumps({"op": op, **payload}) + "\n"
        with self._lock:
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(request.encode())
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                raise SidecarError(
                    f"phillip sidecar: {op} could not be sent ({error}); exit code {self.returncode}; "
                    f"stderr tail:\n{self.stderr_tail()}"
                ) from error
            line = self._reader.read_line(timeout)
        if line is None:
            self.kill()
            raise SidecarError(
                f"phillip sidecar: {op} timed out after {timeout:g} s (the process was killed); "
                f"stderr tail:\n{self.stderr_tail()}"
            )
        if line == b"":
            code = self._process.wait(timeout=5.0)
            raise SidecarError(
                f"phillip sidecar: exited with code {code} while answering {op}; stderr tail:\n"
                f"{self.stderr_tail()}"
            )
        try:
            response = json.loads(line)
        except ValueError as error:
            raise SidecarError(f"phillip sidecar: unreadable reply to {op}: {line[:200]!r}") from error
        if not isinstance(response, dict) or not response.get("ok"):
            error_text = response.get("error", "?") if isinstance(response, dict) else "?"
            trace = response.get("traceback", "") if isinstance(response, dict) else ""
            raise SidecarError(f"phillip sidecar: {op} failed: {error_text}\n{trace}")
        return response

    def kill(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.kill()
        with contextlib.suppress(
            subprocess.TimeoutExpired
        ):  # pragma: no cover - a killed process that will not die
            process.wait(timeout=5.0)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            with contextlib.suppress(SidecarError):
                self.call("quit", timeout_s=5.0)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.kill()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def __enter__(self) -> PhillipSidecar:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------


class PhillipPlayer:
    """One Phillip agent on one port across ``num_envs`` environments, through one sidecar.

    Construction reads the agent's ``params`` (so a wrong name fails before anything starts), starts the
    sidecar, checks the protocol version and loads one ``Agent`` per environment -- refusing a restore
    that initialised or padded variables, whatever ``strict`` says, because that agent would be playing
    random weights.  :meth:`rows` is one frame for every environment: the frames upstream would act on
    (in game, both ports present, at or after :attr:`PhillipConfig.act_from_frame`) go to the sidecar in
    one request; every other environment holds a neutral input, as upstream's ``cpu.py`` sends in menus.
    """

    def __init__(
        self,
        config: PhillipConfig,
        num_envs: int,
        *,
        melee: Any | None = None,
        sidecar: PhillipSidecar | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.config = config
        self.num_envs = num_envs
        self.info = agent_info(config)
        if melee is None:
            from melee_rl.env.libmelee_frames import import_melee

            melee = import_melee()
        self._melee = melee
        self._max_jumps = load_max_jumps(melee)
        self._in_game = frozenset(getattr(melee.Menu, name) for name in _MISSING_PORT_MENU)
        self.sidecar = (
            PhillipSidecar(
                [config.python, str(SIDECAR_SCRIPT)],
                source_dir=config.source_dir,
                call_timeout_s=config.call_timeout_s,
            )
            if sidecar is None
            else sidecar
        )
        self._rows: list[ControllerRow] = [neutral_row() for _ in range(num_envs)]
        self.frames_seen = 0
        self.live_frames = 0
        self.decisions = 0
        self.neutral_frames = 0
        self.exceptions = 0
        self.exception_envs: set[int] = set()
        self.first_error = ""
        self.sidecar_seconds = 0.0
        say = log or (lambda message: LOG.info("%s", message))
        try:
            self.hello = self.sidecar.call("hello")
            if self.hello.get("protocol") != PROTOCOL_VERSION:
                raise SidecarError(
                    f"phillip sidecar speaks {self.hello.get('protocol')!r}, this recorder "
                    f"{PROTOCOL_VERSION!r}"
                )
            self.load = self.sidecar.call(
                "load",
                timeout_s=config.load_timeout_s,
                agent=config.agent,
                envs=num_envs,
                epsilon=config.epsilon,
                seed=config.seed,
                swap=int(config.player == 0),
                save_cpu=config.save_cpu,
                source_dir=config.source_dir,
            )
            restore = self.load["restore"]
            if restore["missing"] or restore["padded"]:
                raise SidecarError(
                    f"phillip {config.agent} did not restore cleanly: {len(restore['missing'])} variable(s) "
                    f"initialised {restore['missing'][:4]}, {len(restore['padded'])} padded "
                    f"{restore['padded'][:2]}; an agent loaded that way is not the released one"
                )
        except BaseException:
            self.sidecar.close()
            raise
        say(
            f"phillip {config.agent}: {self.info.char}, act_every {self.info.act_every}, delay "
            f"{self.info.delay_frames} frames, {num_envs} agents loaded in {self.load['seconds']:.1f} s "
            f"(python {self.hello.get('python')}, tensorflow {self.hello.get('tensorflow')})"
        )

    # -- one frame ---------------------------------------------------------

    def _actionable(self, gamestate: Any) -> bool:
        if gamestate is None or getattr(gamestate, "menu_state", None) not in self._in_game:
            return False
        players = getattr(gamestate, "players", {})
        if any(port not in players for port in PORTS):
            return False
        return int(gamestate.frame) >= self.config.act_from_frame

    def rows(self, gamestates: Sequence[Any], needs_reset: torch.Tensor) -> list[ControllerRow]:
        if len(gamestates) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} gamestates, got {len(gamestates)}")
        if int(needs_reset.shape[0]) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} reset flags, got {int(needs_reset.shape[0])}")
        flags = needs_reset.to(dtype=torch.bool, device="cpu").tolist()
        resets = [index for index, flag in enumerate(flags) if flag]
        if resets:
            self._timed("reset", envs=resets)
            for index in resets:
                self._rows[index] = neutral_row()
        frames: list[dict[str, Any]] = []
        for index, gamestate in enumerate(gamestates):
            self.frames_seen += 1
            if self._actionable(gamestate):
                frames.append(
                    {
                        "env": index,
                        "state": game_message(gamestate, melee=self._melee, max_jumps=self._max_jumps),
                    }
                )
            else:
                self._rows[index] = neutral_row()
        if frames:
            self.live_frames += len(frames)
            response = self._timed("act", frames=frames)
            self.decisions += int(response.get("decisions", 0))
            answered: set[int] = set()
            for controller in response["controllers"]:
                index = int(controller["env"])
                answered.add(index)
                if "error" in controller:
                    self._record_exception(
                        index, str(controller["error"]), str(controller.get("traceback", ""))
                    )
                    if self.config.strict:
                        raise SidecarError(
                            f"phillip env {index}: {controller['error']}\n{controller.get('traceback', '')}"
                        )
                    self._rows[index] = neutral_row()
                    self.neutral_frames += 1
                    continue
                row = row_from_controller(controller)
                self._rows[index] = row
                if row == neutral_row():
                    self.neutral_frames += 1
            asked = {frame["env"] for frame in frames}
            if answered != asked:
                raise SidecarError(
                    f"phillip sidecar answered for envs {sorted(answered)}, asked {sorted(asked)}"
                )
        return list(self._rows)

    def _timed(self, op: str, **payload: Any) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return self.sidecar.call(op, **payload)
        finally:
            self.sidecar_seconds += time.perf_counter() - started

    def _record_exception(self, index: int, error: str, trace: str) -> None:
        self.exceptions += 1
        self.exception_envs.add(index)
        if not self.first_error:
            self.first_error = f"env {index}: {error}\n{trace}"
        LOG.warning("phillip env %d raised %s", index, error)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, index: int) -> None:
        self._timed("reset", envs=[index])
        self._rows[index] = neutral_row()

    def reset_all(self) -> None:
        self._timed("reset", envs=list(range(self.num_envs)))
        self._rows = [neutral_row() for _ in range(self.num_envs)]

    def close(self) -> None:
        self.sidecar.close()

    # -- health ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What the calibration gate reads: the restore report, exceptions (must be 0), the neutral share,
        the decision count, and the sidecar's own per-environment counters (a live query)."""

        sidecar: dict[str, Any] = {}
        if self.sidecar.returncode is None:
            try:
                reply = self.sidecar.call("stats")
                sidecar = {"envs": reply.get("envs", []), "loaded": reply.get("loaded", {})}
            except SidecarError as error:  # pragma: no cover - a dead sidecar still gets a health panel
                sidecar = {"error": str(error)}
        return {
            "agent": self.config.agent,
            "char": self.info.char,
            "act_every": self.info.act_every,
            "delay_frames": self.info.delay_frames,
            "memory": self.info.memory,
            "action_type": self.info.action_type,
            "stage": self.info.stage,
            "epsilon": self.config.epsilon,
            "player": self.config.player,
            "libmelee_port": self.config.libmelee_port,
            "revision": PHILLIP_REVISION,
            "frames": self.frames_seen,
            "live_frames": self.live_frames,
            "decisions": self.decisions,
            "neutral_frames": self.neutral_frames,
            "neutral_share": self.neutral_frames / self.live_frames if self.live_frames else 0.0,
            "exceptions": self.exceptions,
            "exception_envs": sorted(self.exception_envs),
            "first_error": self.first_error,
            "sidecar_seconds": self.sidecar_seconds,
            "restore": {key: self.load["restore"][key] for key in ("missing", "padded")},
            "versions": {key: self.hello.get(key) for key in ("python", "tensorflow", "numpy")},
            "sidecar": sidecar,
        }


# ---------------------------------------------------------------------------
# the preflight
# ---------------------------------------------------------------------------


class _SyntheticPlayer:
    """A libmelee-shaped player at rest, for timing the sidecar without a Dolphin."""

    def __init__(self, melee: Any, x: float) -> None:
        self.character = melee.Character.FOX
        self.position = type("P", (), {"x": x, "y": 0.0})()
        self.percent = 0.0
        self.stock = 4
        self.facing = x < 0
        self.action = melee.Action.STANDING
        self.action_frame = 1
        self.invulnerable = False
        self.hitlag_left = 0
        self.hitstun_frames_left = 0
        self.jumps_left = 2
        self.on_ground = True
        self.speed_air_x_self = 0.0
        self.speed_ground_x_self = 0.0
        self.speed_y_self = 0.0
        self.speed_x_attack = 0.0
        self.speed_y_attack = 0.0
        self.shield_strength = 60.0


class _SyntheticGamestate:
    def __init__(self, melee: Any, frame: int) -> None:
        self.frame = frame
        self.menu_state = melee.Menu.IN_GAME
        self.players = {PORTS[0]: _SyntheticPlayer(melee, -20.0), PORTS[1]: _SyntheticPlayer(melee, 20.0)}


def preflight(
    config: PhillipConfig, *, frames: int = 60, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Start a sidecar, load one agent, play ``frames`` synthetic frames, close: the check that runs before
    any Dolphin boots (a bad interpreter, a missing checkout or a restore that initialises variables is a
    ten-second error here and a forty-minute one after a pool of emulators has come up)."""

    from melee_rl.env.libmelee_frames import import_melee

    melee = import_melee()
    started = time.perf_counter()
    player = PhillipPlayer(config, 1, melee=melee, log=log)
    try:
        flags = torch.zeros(1, dtype=torch.bool)
        for frame in range(frames):
            player.rows([_SyntheticGamestate(melee, frame)], flags)
        stats = player.stats()
    finally:
        player.close()
    decisions = max(1, stats["decisions"])
    return {
        "agent": config.agent,
        "hello": player.hello,
        "info": player.info.as_dict(),
        "config": player.load["config"],
        "restore": player.load["restore"],
        "seconds": time.perf_counter() - started,
        "load_seconds": player.load["seconds"],
        "frames": frames,
        "decisions": stats["decisions"],
        "decision_ms": 1000.0 * stats["sidecar_seconds"] / decisions,
        "frame_ms": 1000.0 * stats["sidecar_seconds"] / max(1, frames),
        "exceptions": stats["exceptions"],
        "first_error": stats["first_error"],
    }


def format_preflight(report: Mapping[str, Any]) -> str:
    restore = report["restore"]
    return (
        f"phillip {report['agent']}: {report['info']['char']} act_every {report['info']['act_every']} "
        f"delay {report['info']['delay_frames']} frames; python {report['hello'].get('python')}, tensorflow "
        f"{report['hello'].get('tensorflow')}; load {report['load_seconds']:.1f} s, "
        f"{len(restore['missing'])} missing / {len(restore['padded'])} padded variables; "
        f"{report['decisions']} decisions in {report['frames']} frames, "
        f"{report['decision_ms']:.2f} ms per decision, {report['frame_ms']:.2f} ms per frame; "
        f"{report['exceptions']} exceptions"
    )


__all__ = [
    "ACT_FROM_FRAME",
    "DEFAULT_AGENT",
    "DEFAULT_PYTHON",
    "DEFAULT_SOURCE_DIR",
    "DELAY0_AGENTS",
    "GAME_MENU",
    "IN_REPO_AGENTS",
    "MAX_ACTION_STATE",
    "PHILLIP_CHARACTERS",
    "PHILLIP_LICENSE",
    "PHILLIP_REPO_URL",
    "PHILLIP_REVISION",
    "PROTOCOL_VERSION",
    "SIDECAR_SCRIPT",
    "SMASH_ATTACK_ACTIONS",
    "STAGE_IDS",
    "TENSORFLOW_PIN",
    "UNSELECTED_CHARACTER",
    "PhillipAgentInfo",
    "PhillipConfig",
    "PhillipPlayer",
    "PhillipSidecar",
    "SidecarError",
    "agent_info",
    "check_phillip_character",
    "format_preflight",
    "game_message",
    "load_max_jumps",
    "player_message",
    "preflight",
    "read_agent_info",
    "row_from_controller",
]
