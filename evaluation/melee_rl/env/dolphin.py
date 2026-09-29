"""``DolphinEnv``: headless Dolphin through libmelee 0.47.3, behind ``EnvProtocol`` (PLAN.md §6 P5).

Semantics follow slippi-ai's ``Dolphin`` / ``Environment`` / ``SafeEnvironment``
(``slippi_ai/dolphin.py:73-250``, ``envs.py:30-190``, ``controller_lib.py:22-33``) and MIMIC's actor
(``rlvr/online/dolphin_actor.py:479-607``: one Slippi UDP port per Dolphin, a keepalive between
rollouts); no code is copied.  ``ENV.md`` §6 has the verified libmelee facts this module relies on.

* One ``melee.Console`` per environment (``B`` of them, stepped in lock-step): ExiAI Dolphin with
  ``gfx_backend="Null"``, no audio, EXI inputs + fast-forward gecko codes, ``blocking_input`` (Dolphin
  waits for our inputs, so an idle environment is a paused game), ``polling_mode`` with
  ``console_timeout_s``, a temporary home directory, Slippi port ``slippi_port + i`` (re-picked if in
  use), ``infinite_time`` + ``instant_match_restart`` (an endless match: the only game starts are
  launches, relaunches and :meth:`DolphinEnv.reset`).  Mainline builds get ``--platform headless``
  and ``emulation_speed = 0`` (unlimited); the build is detected with ``get_dolphin_version``.
* Launch: console -> controllers on ports 1 and 2 (created before ``run``) -> ``run`` -> ``connect``
  -> controller pipes (probed without blocking, the process watched) -> the first frame (polled until
  ``boot_timeout_s``) -> menus with one stateful ``MenuHelper`` per controller
  (``menu_helper_simple``: character, costume, stage, ``cpu_level`` for the cpu port, ``autostart``
  for the first controller after 30 menu frames *and* once the cpu port reads as a CPU at the configured
  level with its slider released -- libmelee applies the CPU toggle / slider only after the coin is down,
  so an earlier START starts a human-type idle port 2; slippi-ai's loop otherwise) -> the first in-game
  frame, always ``-123`` (``needs_reset``); at every game start the converter's item slots reset, the
  characters are checked and, with a cpu opponent, the game-start ``cpu_level`` of port 2 must be the
  configured one (``WrongCpuLevel``).
* :meth:`DolphinEnv.step`: per environment ``send_controller`` for every controlled port (the
  slippi-ai press / release / tilt / shoulder sequence, no flush -- ``Console.step`` flushes), one
  ``console.step()``, convert, and ``EnvOutput`` with ``needs_reset == (frame == -123)``.  In game a
  ``None`` from ``step`` (the polling timeout) is a fault, never re-polled (re-polling would flush the
  controllers twice and misalign every later frame by one).
* Faults (``ConnectFailed`` / ``ConsoleTimeout`` / ``MenuStall`` / ``WrongCharacter``,
  ``TimeoutError``, ``EnetDisconnected``, ``FrameConversionError``, ``OSError`` on the pipes) stop the
  instance and relaunch it -- at most ``launch_retries`` extra attempts per launch, then
  :class:`DolphinLaunchError`; the first frame of the relaunched game is a legal reset row, the
  action sent in that step is dropped (slippi-ai's ``SafeEnvironment`` retries the same way).
* Keepalive (``keepalive_seconds > 0``; off by default -- libmelee 0.47.3 services enet in its own
  worker process, so an idle Dolphin keeps its connection): a thread steps every idle environment
  with the neutral controller every ``keepalive_seconds``; those frames are invisible to the rollout
  worker (they fall between two recorded frames), except that a tick reaching a game start parks the
  ``-123`` frame, stops ticking that environment and the next :meth:`step` delivers it unchanged
  (the action is dropped, as on any reset row).  :meth:`current` always reflects the last frame.
* :meth:`close` (idempotent; also registered with ``atexit``) stops the thread and every Dolphin
  (``Console.stop`` kills the process and removes the temporary home).  :meth:`reset` relaunches
  everything -- seconds per environment, so ``[runtime] reset_every_n_steps`` is for the dummy env.
* Throughput (P6a).  ``boot = "two_phase"`` (default) launches in three phases instead of one Dolphin
  after another: every console is started (``start``: process creation, main thread), then every
  console is connected on the main thread (``connect``: ``console.connect()`` -- libmelee forks its
  enet worker process there, ``slippstream.py:146-190``, so it never runs on a worker thread -- the
  controller pipes, the first frame; the Dolphins boot concurrently, so the waits overlap), then the
  menus are driven in a thread pool (``settle``: no forks).  A recoverable error in any phase sends
  that environment to the sequential retries (``launch_retries`` bounds the total attempts as before).
  ``step_threads`` (0 = ``num_envs``, 1 = sequential) is the size of a ``ThreadPoolExecutor``
  (``melee-rl-dolphin-*``) that fans ``send + advance + convert`` out over the live Dolphins in every
  :meth:`DolphinEnv.step`: ``Console.step`` blocks in ``mp.Pipe.poll / recv_bytes`` and the controller
  flush in a FIFO write, both releasing the GIL, every console owns its worker process and pipe, and
  the tasks write disjoint rows of the step's ``FrameBatch``.  Faults found inside the fan-out are
  relaunched afterwards on the main thread (sequentially); the keepalive ticks stay sequential; the
  pool is created before the boot and shut down in :meth:`close`.  The environment lock is held by
  the calling thread for the whole step -- worker threads touch only their own instance.
"""

from __future__ import annotations

import atexit
import configparser
import errno
import logging
import os
import socket
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import torch

from controller_codec import ControllerState
from melee_rl.env.libmelee_frames import FrameBatch, FrameConversionError, FrameConverter, import_melee
from melee_rl.env.protocol import INITIAL_FRAME_INDEX, EnvOutput, swap_perspective
from melee_rl.env.timing import bin_index, new_env_timing
from model import SlippiEncoder
from tensor_batch import BUTTON_ORDER

LOG = logging.getLogger("melee_rl.env.dolphin")

DEFAULT_DOLPHIN_PATH: Final[str] = "/opt/dolphin/mainline-exiai-14.1-noleak/squashfs-root/usr/bin/dolphin-emu"
"""The mainline ExiAI executable of the Modal Dolphin image (``modal_app.DOLPHIN_BUILDS``, ENV.md §6.1)."""
DEFAULT_ISO_PATH: Final[str] = "/iso/melee.iso"
"""The Modal Volume ``melee-iso`` mounted at ``/iso`` (ENV.md §6.4)."""
DEFAULT_SLIPPI_PORT: Final[int] = 51441
PORTS: Final[tuple[int, int]] = (1, 2)
"""Dolphin controller ports of environment players 0 and 1."""
PLAYER_KINDS: Final[tuple[str, ...]] = ("policy", "cpu")
MENU_AUTOSTART_FRAMES: Final[int] = 30
"""slippi-ai grants ``autostart`` to the first menuing controller after this many menu frames."""
ISO_GAME_ID: Final[bytes] = b"GALE01"
ISO_REVISION: Final[int] = 2
PIPE_POLL_S: Final[float] = 0.05
PORT_SPAN: Final[int] = 64
BOOT_MODES: Final[tuple[str, ...]] = ("sequential", "two_phase")
"""``DolphinEnvConfig.boot``: launch the Dolphins one after another (P5), or start them all, then connect and
settle them (P6a: the boots overlap)."""
MP_START_METHODS: Final[tuple[str, ...]] = ("spawn", "forkserver")
"""``DolphinEnvConfig.mp_start_method`` (P7a): how ``melee_rl.env.dolphin_mp`` starts its worker
processes.  Never plain ``fork`` -- the trainer process holds CUDA and thread pools."""


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class DolphinFault(RuntimeError):
    """A recoverable per-environment fault: the instance is stopped and relaunched."""


class ConnectFailed(DolphinFault):
    """Dolphin did not come up: no Slippi connection, no controller pipe reader, no first frame."""


class ConsoleTimeout(DolphinFault):
    """``Console.step`` returned ``None`` in game (the polling timeout elapsed)."""


class MenuStall(DolphinFault):
    """The menus did not lead into a game within ``menu_frame_cap`` frames / ``boot_timeout_s``."""


class WrongCharacter(DolphinFault):
    """A game started with a character other than the configured one on a port."""


class WrongCpuLevel(DolphinFault):
    """A game started with the cpu port at another level than configured (or as a human-type port)."""


class WrongStage(DolphinFault):
    """A game started on a stage other than the one configured for that environment."""


class DolphinLaunchError(RuntimeError):
    """Every launch attempt of one environment failed (``launch_retries`` exhausted)."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DolphinEnvConfig:
    """``[env.dolphin]`` of an RL run file (TOML-able types only).

    ``players`` are the two ports in order (port 0 is the trained player, always ``"policy"``); the
    second is ``"cpu"`` (Dolphin's level-``cpu_level`` CPU, no controller sent) or ``"policy"``
    (self-play: the environment expects a controller for port 1 too).  ``characters`` / ``costumes`` /
    ``stage`` are libmelee internal ids (Fox = 1, Final Destination = 25).  Environment ``i`` uses Slippi
    port ``slippi_port + i`` (the next free one if that is taken).  Timeouts are seconds:
    ``console_timeout_s`` per frame in game, ``boot_timeout_s`` from launch to the first in-game frame
    (also the wall bound of any menu phase), ``connect_timeout_s`` for the controller pipes.
    ``launch_retries`` extra attempts per launch; ``launch_stagger_s`` between sequential launches;
    ``keepalive_seconds`` 0 disables the idle ticks; ``check_iso`` verifies the ``GALE01`` rev 2 header.
    ``step_threads`` (P6a) is the size of the thread pool that fans ``send + advance + convert`` out over the
    Dolphins in every :meth:`DolphinEnv.step` (0 = one thread per environment, 1 = sequential on the calling
    thread); ``boot`` is ``"two_phase"`` (start every Dolphin, then connect and settle them -- the boots
    overlap) or ``"sequential"`` (the P5 behaviour).

    ``worker_processes`` (P7a) moves the Dolphins into OS worker processes: 0 (the default) keeps
    today's in-process :class:`DolphinEnv`; ``N >= 1`` runs ``num_envs / N`` Dolphins per process
    behind :class:`melee_rl.env.dolphin_mp.WorkerDolphinEnv` (frame conversion and libmelee's event
    parsing escape the GIL).  ``mp_start_method`` is ``"spawn"`` (default) or ``"forkserver"``; worker
    mode requires ``keepalive_seconds == 0`` (it is 0 in every real config -- an idle Dolphin keeps
    its connection).

    ``save_replays`` / ``replay_dir`` (P9) hand Slippi a directory to write ``.slp`` replays into;
    every instance gets ``<replay_dir>/env<global index>`` (:func:`instance_replay_dir`) because Slippi
    names files ``Game_<UTC timestamp>.slp`` and N Dolphins starting in the same second would collide.
    ``env_index_offset`` is the global index of this configuration's environment 0 (set by
    ``dolphin_mp.worker_configs``; it only names the replay directories).  ``replay_monthly_folders``
    is Slippi's ``<replay_dir>/<YYYY-MM>/`` layout, which libmelee leaves at Dolphin's default (on) --
    we turn it off so a clip's file is exactly where we put the instance.  ``stop_timeout_s`` > 0
    SIGTERMs Dolphin and waits that long before ``Console.stop`` kills it -- a killed Dolphin leaves the
    ``.slp`` without its trailing metadata block, so anything we mean to keep needs the clean shutdown.

    ``rtc_values`` / ``menu_idle_frames`` (P11) are the two seed knobs, both **off by default** so
    training keeps a random start: each is indexed by the *global* environment index (like
    :func:`instance_replay_dir`), empty meaning "leave it to Dolphin".  ``rtc_values[i]`` pins that
    Dolphin's emulated console clock (:func:`write_custom_rtc`); ``menu_idle_frames[i]`` holds its
    START press back that many character-select frames.  Melee's RNG advances every frame from boot,
    so both change the seed a match starts from -- which of them actually *controls* it is what the
    P11 probe measures, and the replay's game-start block (``melee_rl.slp``) is what verifies it.

    ``chunk_frames`` (P8) is ``k``, the frames per environment message of the delay pipeline:
    worker mode sends one ``step_chunk`` pipe message per worker per ``k`` pushed actions, and the
    in-process env is wrapped in ``AsyncEnvAdapter(env, k)`` by the trainer.  1 (the default) is
    today's frame-per-message stepping; ``k > 1`` needs the pipeline slack ``(batch_steps - 1) +
    (chunk_frames - 1) <= delay_frames`` (enforced by ``finalize_config``).

    ``shared_memory`` (5 Sep 2026, lever 0 / IPC) moves the worker mode's per-frame exchange out of the
    pickled pipes: the parent writes the controllers into a shared block, each worker converts its frames
    straight into a shared frame slot, and the pipe carries tokens (:mod:`melee_rl.env.shm`); the leaves
    are bit-identical to the pickled path's.  Lockstep ``step`` only; the chunked wire keeps pickling.

    ``legacy_analog_ports`` (P26, 9 Sep 2026) lists the libmelee ports whose ``Controller`` is built
    with ``fix_analog_inputs=False``.  Our pin (``melee==0.47.3``, vladfi1's fork) turns that remap
    **on** by default (``controller.py:89``): every ``tilt_analog`` value is rounded onto Melee's
    internal [-80, 80] grid before it reaches the pipe, so ``0.35`` is sent as ``0.4059``.  altf4's
    0.41.1 -- the version SmashBot was written against -- has no such step and writes ``0.35``.  It
    matters for a third-party agent that aims its sticks in Melee units: SmashBot's "near perfect
    wavedash angle" (``Chains/wavedash.py:67``) loses its downward component under the remap whenever
    the horizontal component is near-extreme, turning a wavedash into a horizontal airdodge.  Empty
    (the default) is today's behaviour on every port; our own policy and MIMIC were measured with the
    remap on and stay that way.

    ``stages`` (2 Sep 2026) lists libmelee stage ids cycled over the *global* environment index:
    environment ``i`` plays ``stages[(env_index_offset + i) % len(stages)]`` (:meth:`stage_for`), so
    a run's Dolphins cover the list evenly -- 96 over the six legal stages is 16 each, 30 evaluation
    Dolphins is 5 each -- and worker slices keep the global assignment.  Empty (the default) plays
    ``stage`` everywhere.  A game that starts on another stage is a :class:`WrongStage` fault
    (relaunched, like a wrong character).

    ``opponent_characters`` (17 Sep 2026, the league run) lists player 1's character per environment, indexed
    by the *global* environment index like ``rtc_values`` (entry ``env_index_offset + i``): a mix arm whose
    opponent plays twelve characters needs one Dolphin per matchup, and a Dolphin cannot change character
    without a relaunch.  Empty (the default) plays ``characters`` on every Dolphin; port 0 is always
    ``characters[0]`` (:meth:`characters_for`).  A game that starts with another character is a
    :class:`WrongCharacter` fault on that Dolphin alone.
    """

    num_envs: int = 1
    players: tuple[str, str] = ("policy", "cpu")
    dolphin_path: str = DEFAULT_DOLPHIN_PATH
    iso_path: str = DEFAULT_ISO_PATH
    characters: tuple[int, int] = (1, 1)
    opponent_characters: tuple[int, ...] = ()
    costumes: tuple[int, int] = (0, 1)
    stage: int = 25
    stages: tuple[int, ...] = ()
    slippi_port: int = DEFAULT_SLIPPI_PORT
    console_timeout_s: float = 30.0
    boot_timeout_s: float = 120.0
    connect_timeout_s: float = 60.0
    menu_frame_cap: int = 3000
    launch_retries: int = 2
    launch_stagger_s: float = 0.0
    keepalive_seconds: float = 0.0
    infinite_time: bool = True
    instant_match_restart: bool = True
    online_delay: int = 0
    legacy_analog_ports: tuple[int, ...] = ()
    save_replays: bool = False
    replay_dir: str | None = None
    rtc_values: tuple[int, ...] = ()
    menu_idle_frames: tuple[int, ...] = ()
    check_iso: bool = True
    step_threads: int = 0
    boot: str = "two_phase"
    worker_processes: int = 0
    mp_start_method: str = "spawn"
    chunk_frames: int = 1
    shared_memory: bool = False
    stop_timeout_s: float = 0.0
    env_index_offset: int = 0
    replay_monthly_folders: bool = False

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if len(self.players) != 2 or any(kind not in PLAYER_KINDS for kind in self.players):
            raise ValueError(f"players must be two of {PLAYER_KINDS}, got {self.players!r}")
        if self.players[0] != "policy":
            raise ValueError(
                f"players[0] must be 'policy' (the trained player on Dolphin port 1), got {self.players!r}"
            )
        for name, values in (("rtc_values", self.rtc_values), ("menu_idle_frames", self.menu_idle_frames)):
            needed = self.env_index_offset + self.num_envs
            if values and len(values) < needed:
                raise ValueError(
                    f"{name} is indexed by the global environment index: it needs {needed} entries "
                    f"for num_envs {self.num_envs} at env_index_offset {self.env_index_offset}, "
                    f"got {len(values)}"
                )
            if any(value < 0 for value in values):
                raise ValueError(f"{name} entries must be >= 0, got {values!r}")
        if any(value >= self.menu_frame_cap for value in self.menu_idle_frames):
            raise ValueError(
                f"menu_idle_frames must stay under menu_frame_cap = {self.menu_frame_cap} "
                f"(a longer hold stalls the boot), got {self.menu_idle_frames!r}"
            )
        if not self.dolphin_path:
            raise ValueError("dolphin_path must name the Dolphin executable")
        if not self.iso_path:
            raise ValueError("iso_path must name the Melee ISO")
        if len(self.characters) != 2 or any(
            not 0 <= character < SlippiEncoder.CHARACTER_SIZE for character in self.characters
        ):
            raise ValueError(f"characters must be two libmelee ids in [0, {SlippiEncoder.CHARACTER_SIZE})")
        if self.opponent_characters:
            needed = self.env_index_offset + self.num_envs
            if len(self.opponent_characters) < needed:
                raise ValueError(
                    "opponent_characters is indexed by the global environment index: it needs "
                    f"{needed} entries for num_envs {self.num_envs} at env_index_offset "
                    f"{self.env_index_offset}, got {len(self.opponent_characters)}"
                )
            if any(
                not 0 <= character < SlippiEncoder.CHARACTER_SIZE for character in self.opponent_characters
            ):
                raise ValueError(
                    f"opponent_characters must be libmelee ids in [0, {SlippiEncoder.CHARACTER_SIZE}), "
                    f"got {self.opponent_characters!r}"
                )
        if len(self.costumes) != 2 or any(costume < 0 for costume in self.costumes):
            raise ValueError("costumes must be two non-negative indices")
        if not 0 <= self.stage < SlippiEncoder.STAGE_SIZE:
            raise ValueError(f"stage must be a libmelee stage id in [0, {SlippiEncoder.STAGE_SIZE})")
        if any(not 0 <= stage < SlippiEncoder.STAGE_SIZE for stage in self.stages):
            raise ValueError(
                f"stages must be libmelee stage ids in [0, {SlippiEncoder.STAGE_SIZE}), got {self.stages!r}"
            )
        if not self.slippi_port >= 1024 or self.slippi_port + self.num_envs - 1 > 65535:
            raise ValueError(
                f"slippi_port {self.slippi_port} + num_envs {self.num_envs} must fit in [1024, 65535]"
            )
        for name in ("console_timeout_s", "boot_timeout_s", "connect_timeout_s"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if self.menu_frame_cap < 1:
            raise ValueError("menu_frame_cap must be >= 1")
        if self.launch_retries < 0:
            raise ValueError("launch_retries must be >= 0")
        if self.launch_stagger_s < 0.0:
            raise ValueError("launch_stagger_s must be >= 0")
        if self.keepalive_seconds < 0.0:
            raise ValueError("keepalive_seconds must be >= 0 (0 disables the idle ticks)")
        if self.online_delay < 0:
            raise ValueError("online_delay must be >= 0")
        if len(set(self.legacy_analog_ports)) != len(self.legacy_analog_ports) or any(
            port not in PORTS for port in self.legacy_analog_ports
        ):
            raise ValueError(f"legacy_analog_ports must be distinct libmelee ports from {PORTS}")
        if self.step_threads < 0:
            raise ValueError("step_threads must be >= 0 (0 = one thread per environment, 1 = sequential)")
        if self.boot not in BOOT_MODES:
            raise ValueError(f"boot must be one of {BOOT_MODES}, got {self.boot!r}")
        if self.worker_processes < 0:
            raise ValueError("worker_processes must be >= 0 (0 = in-process stepping)")
        if self.worker_processes > 0 and self.num_envs % self.worker_processes:
            raise ValueError(f"worker_processes {self.worker_processes} must divide num_envs {self.num_envs}")
        if self.worker_processes > 0 and self.keepalive_seconds != 0.0:
            raise ValueError("worker mode requires keepalive_seconds = 0 (no idle ticks across processes)")
        if self.shared_memory and self.worker_processes < 1:
            raise ValueError(
                "shared_memory is the worker mode's frame exchange: it needs worker_processes >= 1"
            )
        if self.mp_start_method not in MP_START_METHODS:
            raise ValueError(
                f"mp_start_method must be one of {MP_START_METHODS}, got {self.mp_start_method!r}"
            )
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1 (frames per environment message, P8)")
        if self.stop_timeout_s < 0.0:
            raise ValueError("stop_timeout_s must be >= 0 (0 = kill Dolphin, as Console.stop does)")
        if self.env_index_offset < 0:
            raise ValueError("env_index_offset must be >= 0 (the global index of this slice's env 0)")

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return tuple(port for port, kind in enumerate(self.players) if kind == "policy")

    @property
    def resolved_step_threads(self) -> int:
        """The pool size :meth:`DolphinEnv.step` uses: ``num_envs`` for 0, else ``step_threads``."""

        return self.num_envs if self.step_threads == 0 else self.step_threads

    @property
    def envs_per_worker(self) -> int:
        """The Dolphins per worker process (``num_envs`` itself in the in-process mode)."""

        return self.num_envs if self.worker_processes == 0 else self.num_envs // self.worker_processes

    def rtc_value(self, index: int) -> int | None:
        """The custom console clock of environment ``index``, or ``None`` while the knob is off."""

        if not self.rtc_values:
            return None
        return self.rtc_values[self.env_index_offset + index]

    def menu_idle(self, index: int) -> int:
        """Extra character-select frames environment ``index`` holds before pressing START (0 = off)."""

        if not self.menu_idle_frames:
            return 0
        return self.menu_idle_frames[self.env_index_offset + index]

    def characters_for(self, index: int) -> tuple[int, int]:
        """The libmelee characters environment ``index`` plays: ``characters``, with player 1's taken from
        ``opponent_characters`` at the global index ``env_index_offset + index`` when that table is set."""

        if not self.opponent_characters:
            return self.characters
        return (self.characters[0], self.opponent_characters[self.env_index_offset + index])

    @property
    def character_ids(self) -> tuple[int, ...]:
        """Every character id this configuration's environments select, ascending."""

        chosen = {self.characters[0]}
        chosen.update(self.characters_for(index)[1] for index in range(self.num_envs))
        return tuple(sorted(chosen))

    def stage_for(self, index: int) -> int:
        """The libmelee stage id environment ``index`` plays: ``stage``, or ``stages`` cycled by the global
        index ``env_index_offset + index``."""

        if not self.stages:
            return self.stage
        return self.stages[(self.env_index_offset + index) % len(self.stages)]

    @property
    def stage_ids(self) -> tuple[int, ...]:
        """Every stage id this configuration selects, ascending (``stage`` alone while ``stages`` is
        empty)."""

        return tuple(sorted(set(self.stages))) if self.stages else (self.stage,)


@dataclass(frozen=True)
class BuildInfo:
    """What ``get_dolphin_version`` said about the executable (mainline vs Ishiiruka, ExiAI)."""

    mainline: bool
    version: str
    exi_ai: bool
    exe: str

    @property
    def platform(self) -> str | None:
        """``Console.run(platform=...)``: ``"headless"`` on mainline, nothing on Ishiiruka."""

        return "headless" if self.mainline else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def check_dolphin_exe(path: str | Path) -> Path:
    """``path`` as an existing, executable file."""

    exe = Path(path)
    if not exe.is_file():
        raise FileNotFoundError(f"dolphin executable {exe} does not exist")
    if not os.access(exe, os.X_OK):
        raise PermissionError(f"dolphin executable {exe} is not executable")
    return exe


def check_iso_identity(path: str | Path) -> dict[str, Any]:
    """The first bytes of the disc image must say Melee NTSC 1.02 (``GALE01``, revision 2).

    Returns ``{"game_id", "revision", "bytes"}``; never prints or returns anything else of the file.
    """

    iso = Path(path)
    if not iso.is_file():
        raise FileNotFoundError(f"Melee ISO {iso} does not exist")
    with iso.open("rb") as stream:
        header = stream.read(8)
    game_id = header[:6]
    revision = header[7] if len(header) > 7 else None
    if game_id != ISO_GAME_ID or revision != ISO_REVISION:
        raise ValueError(
            f"{iso} is not Melee NTSC 1.02 (GALE01 rev 2): header {game_id.decode('ascii', 'replace')!r} "
            f"revision {revision!r}"
        )
    return {"game_id": game_id.decode("ascii"), "revision": revision, "bytes": iso.stat().st_size}


def pick_slippi_port(candidate: int, span: int = PORT_SPAN) -> int:
    """The first UDP port in ``[candidate, candidate + span)`` that can be bound on the loopback."""

    for port in range(candidate, min(candidate + span, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"no free UDP port in [{candidate}, {candidate + span}) for Dolphin's Slippi server")


def wait_for_pipe_reader(
    path: str | Path,
    deadline: float,
    process: Any | None = None,
    *,
    poll_s: float = PIPE_POLL_S,
) -> bool:
    """Wait until Dolphin has opened the read end of the controller FIFO ``path``.

    A non-blocking write-open succeeds exactly when a reader exists (``ENXIO`` otherwise); it is
    closed again immediately (Dolphin's pipe reader ignores an EOF).  Returns ``False`` when ``path``
    is not a FIFO (nothing to probe); raises :class:`ConnectFailed` when ``process`` exited or the
    monotonic ``deadline`` passed.  Without this, ``Controller.connect`` blocks forever on a Dolphin
    that died before opening its pipes.
    """

    pipe = Path(path)
    try:
        mode = pipe.stat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISFIFO(mode):
        return False
    while True:
        try:
            descriptor = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as error:
            if error.errno != errno.ENXIO:
                raise ConnectFailed(f"cannot open the controller pipe {pipe}: {error}") from error
        else:
            os.close(descriptor)
            return True
        if process is not None and process.poll() is not None:
            raise ConnectFailed(
                f"dolphin exited with exit code {process.returncode} before opening the pipe {pipe}"
            )
        if time.monotonic() >= deadline:
            raise ConnectFailed(f"no reader on the controller pipe {pipe} within the connect timeout")
        time.sleep(poll_s)


class _DolphinIni(configparser.ConfigParser):
    """A parser that keeps Dolphin's CamelCase option names (the same reason as ``render._Ini``)."""

    def optionxform(self, optionstr: str) -> str:
        return optionstr


def console_config_dir(console: Any) -> Path:
    """``<user directory>/Config`` of a libmelee ``Console`` (its temporary home, ``tmp_home_directory``)."""

    home = getattr(console, "dolphin_home_path", None)
    if home:
        return Path(home) / "Config"
    return Path(console._get_dolphin_config_path())  # no public accessor; only for a non-temporary home


def write_custom_rtc(config_dir: str | Path, value: int) -> Path:
    """Pin the emulated console clock: ``[Core] CustomRTCEnable`` / ``CustomRTCValue`` in ``Dolphin.ini``.

    libmelee writes that file while the ``Console`` is being constructed and reads any existing one
    first (``melee/console.py:714-717``), so keys added afterwards survive into the launch.  A fixed
    clock is Dolphin's own determinism switch; whether Melee's RNG seed follows it is what the P11
    probe measures, and until it says so this stays off everywhere (``rtc_values`` defaults to ``()``).
    """

    directory = Path(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "Dolphin.ini"
    parser = _DolphinIni()
    if path.is_file():
        parser.read(path)
    if not parser.has_section("Core"):
        parser.add_section("Core")
    parser.set("Core", "CustomRTCEnable", "True")
    parser.set("Core", "CustomRTCValue", str(int(value)))
    with path.open("w") as handle:
        parser.write(handle)
    return path


def instance_replay_dir(config: DolphinEnvConfig, index: int) -> str | None:
    """``<replay_dir>/env<global index>`` for environment ``index``, created; ``None`` with replays off.

    Slippi names replays ``Game_<UTC timestamp>.slp`` and ``replay_dir`` reaches every ``Console``
    verbatim, so N Dolphins starting in the same second would collide in one directory (ENV.md §6.5);
    each instance gets its own, mirroring the ``slippi_port + index`` rule.  The global index is
    ``env_index_offset + index``, so worker slices (``dolphin_mp.worker_configs``) stay distinct.
    """

    if not config.save_replays or config.replay_dir is None:
        return None
    path = Path(config.replay_dir) / f"env{config.env_index_offset + index}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def terminate_console(console: Any, timeout_s: float, pump: Callable[[], Any] | None = None) -> bool:
    """SIGTERM Dolphin and wait up to ``timeout_s``; True when it exited on its own.

    ``Console.stop`` kills the process ("Sadly dolphin doesn't respect terminate",
    ``melee/console.py:677-681``), and a killed Dolphin leaves its ``.slp`` without the trailing
    metadata block.  With ``blocking_input`` the emulator is parked waiting for our controller input,
    so it cannot even see the signal until it runs another frame: ``pump`` (one send + step) is called
    between polls to let it reach its shutdown path.  Best effort -- any failure is reported as False
    and the caller kills.
    """

    process = getattr(console, "_process", None)
    if process is None:
        return False
    try:
        if process.poll() is not None:
            return True
        process.terminate()
        if pump is None:
            process.wait(timeout=timeout_s)
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return True
            try:
                pump()
            except Exception:  # the pipes go away while Dolphin shuts down; that is the point
                time.sleep(PIPE_POLL_S)
        return process.poll() is not None
    except Exception as error:  # a stuck or already-reaped process must not break the shutdown
        LOG.warning("console terminate failed: %s", error)
        return False


@dataclass(frozen=True)
class ControllerRow:
    """One environment's controller as plain Python values (``buttons`` in ``BUTTON_ORDER``)."""

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    shoulder: float
    buttons: tuple[bool, ...]


def neutral_row() -> ControllerRow:
    return ControllerRow(
        main_x=0.5, main_y=0.5, c_x=0.5, c_y=0.5, shoulder=0.0, buttons=(False,) * len(BUTTON_ORDER)
    )


def controller_rows(control: ControllerState, batch: int) -> list[ControllerRow]:
    """Split a ``[B]`` ``ControllerState`` into per-environment rows (one ``tolist`` per leaf)."""

    if tuple(control.shoulder.shape) != (batch,):
        raise ValueError(f"controller must have batch shape ({batch},), got {tuple(control.shoulder.shape)}")
    main_x = control.main_stick.x.tolist()
    main_y = control.main_stick.y.tolist()
    c_x = control.c_stick.x.tolist()
    c_y = control.c_stick.y.tolist()
    shoulder = control.shoulder.tolist()
    buttons = [getattr(control.buttons, name).tolist() for name in BUTTON_ORDER]
    return [
        ControllerRow(
            main_x=float(main_x[i]),
            main_y=float(main_y[i]),
            c_x=float(c_x[i]),
            c_y=float(c_y[i]),
            shoulder=float(shoulder[i]),
            buttons=tuple(bool(column[i]) for column in buttons),
        )
        for i in range(batch)
    ]


def send_controller(melee: Any, controller: Any, row: ControllerRow) -> None:
    """Queue ``row`` on a libmelee ``Controller`` (slippi-ai ``send_controller`` order; no flush)."""

    for name, pressed in zip(BUTTON_ORDER, row.buttons, strict=True):
        button = melee.Button(name)
        if pressed:
            controller.press_button(button)
        else:
            controller.release_button(button)
    controller.tilt_analog(melee.Button.BUTTON_MAIN, row.main_x, row.main_y)
    controller.tilt_analog(melee.Button.BUTTON_C, row.c_x, row.c_y)
    controller.press_shoulder(melee.Button.BUTTON_L, row.shoulder)


def describe_error(error: BaseException | None) -> str:
    return "unknown" if error is None else f"{type(error).__name__}: {error}"


def build_env_output(
    batch: FrameBatch, frame_index: Sequence[int], controlled_ports: tuple[int, ...]
) -> EnvOutput:
    """``EnvOutput`` over ``batch``'s arrays: torch views, per-port perspectives, zero rewards.

    The perspective trees share their leaves (``swap_perspective``) and every port shares one zero
    reward tensor, exactly as :meth:`DolphinEnv.step` always built its outputs; the worker mode
    (``melee_rl.env.dolphin_mp``) assembles the same output from concatenated worker payloads.
    """

    if len(frame_index) != batch.batch:
        raise ValueError(f"frame_index has {len(frame_index)} rows for a batch of {batch.batch}")
    tree = batch.tree()
    index = torch.tensor(list(frame_index), dtype=torch.long)
    frames = {port: (tree if port == 0 else swap_perspective(tree)) for port in controlled_ports}
    zero = torch.zeros(batch.batch, dtype=torch.float32)
    return EnvOutput(
        frames=frames,
        needs_reset=index == INITIAL_FRAME_INDEX,
        frame_index=index,
        rewards={port: zero for port in controlled_ports},
    )


# ---------------------------------------------------------------------------
# backends: what touches libmelee's Console / Controller / MenuHelper
# ---------------------------------------------------------------------------


class DolphinBackend(Protocol):
    """The process-facing side of the environment (faked in the tests)."""

    def prepare(self, config: DolphinEnvConfig) -> BuildInfo:
        """Validate the executable / ISO / ids and detect the build (once, before any launch)."""
        ...

    def pick_port(self, candidate: int) -> int:
        """The Slippi UDP port to use for a launch that would like ``candidate``."""
        ...

    def make_console(self, config: DolphinEnvConfig, index: int, slippi_port: int) -> Any: ...

    def make_controller(self, console: Any, port: int) -> Any: ...

    def make_menu_helper(self) -> Any: ...

    def start(self, console: Any, config: DolphinEnvConfig) -> None:
        """``console.run(...)`` with the ISO and the build's platform."""
        ...


class LibmeleeBackend:
    """The real thing: ``melee.Console`` (the headless ExiAI recipe of ENV.md §6.3), ``Controller``,
    ``MenuHelper``."""

    def __init__(self, melee: Any) -> None:
        self._melee = melee
        self.info: BuildInfo | None = None
        self.legacy_analog_ports: frozenset[int] = frozenset()

    def configure_analog(self, config: DolphinEnvConfig) -> None:
        """Remember which ports opt out of the remap; ``make_controller`` gets no config of its own."""

        self.legacy_analog_ports = frozenset(config.legacy_analog_ports)

    def prepare(self, config: DolphinEnvConfig) -> BuildInfo:
        melee = self._melee
        self.configure_analog(config)
        exe = check_dolphin_exe(config.dolphin_path)
        if config.check_iso:
            check_iso_identity(config.iso_path)
        version = melee.console.get_dolphin_version(str(exe))
        exi_ai = version.build is melee.console.DolphinBuild.EXI_AI
        if not exi_ai:
            raise ValueError(
                f"{exe} reports {version.version!r}: not an ExiAI build; DolphinEnv needs EXI inputs and "
                "fast-forward (ENV.md §6.1)"
            )
        for character in config.character_ids:
            try:
                melee.Character(character)
            except ValueError as error:
                raise ValueError(f"characters: {character} is not a libmelee Character id") from error
        for stage in config.stage_ids:
            try:
                melee.Stage(stage)
            except ValueError as error:
                raise ValueError(f"stage: {stage} is not a libmelee Stage id") from error
        self.info = BuildInfo(
            mainline=bool(version.mainline), version=str(version.version), exi_ai=True, exe=str(exe)
        )
        return self.info

    def pick_port(self, candidate: int) -> int:
        return pick_slippi_port(candidate)

    def _require_info(self) -> BuildInfo:
        if self.info is None:
            raise RuntimeError("LibmeleeBackend.prepare() must run before launching")
        return self.info

    def make_console(self, config: DolphinEnvConfig, index: int, slippi_port: int) -> Any:
        info = self._require_info()
        console = self._melee.Console(
            path=info.exe,
            tmp_home_directory=True,
            copy_home_directory=False,
            slippi_port=slippi_port,
            online_delay=config.online_delay,
            blocking_input=True,
            polling_mode=True,
            polling_timeout=config.console_timeout_s,
            setup_gecko_codes=True,
            fullscreen=False,
            gfx_backend="Null",
            disable_audio=True,
            emulation_speed=0.0 if info.mainline else 1.0,
            save_replays=config.save_replays,
            replay_dir=instance_replay_dir(config, index),
            replay_monthly_folders=config.replay_monthly_folders,
            infinite_time=config.infinite_time,
            instant_match_restart=config.instant_match_restart,
            use_exi_inputs=True,
            enable_ffw=True,
            logger=None,
        )
        rtc = config.rtc_value(index)
        if rtc is not None:
            write_custom_rtc(console_config_dir(console), rtc)
        return console

    def make_controller(self, console: Any, port: int) -> Any:
        return self._melee.Controller(
            console,
            port,
            self._melee.ControllerType.STANDARD,
            fix_analog_inputs=port not in self.legacy_analog_ports,
        )

    def make_menu_helper(self) -> Any:
        return self._melee.MenuHelper()

    def start(self, console: Any, config: DolphinEnvConfig) -> None:
        console.run(iso_path=config.iso_path, platform=self._require_info().platform)


# ---------------------------------------------------------------------------
# one Dolphin
# ---------------------------------------------------------------------------


class _DolphinInstance:
    """One environment: its console, controllers, menu helpers, converter and counters."""

    def __init__(
        self,
        index: int,
        config: DolphinEnvConfig,
        backend: DolphinBackend,
        melee: Any,
        cpu_level: int,
    ) -> None:
        self.index = index
        self.config = config
        self.cpu_level = cpu_level
        self._backend = backend
        self._melee = melee
        self.converter = FrameConverter(melee, ports=PORTS)
        self.console: Any | None = None
        self.controllers: dict[int, Any] = {}
        self.helpers: dict[int, Any] = {}
        self.slippi_port: int | None = None
        self.gamestate: Any | None = None
        self.parked: Any | None = None
        self.launches = 0
        self.relaunches = 0
        self.faults: list[str] = []
        self.frames = 0
        self.ticks = 0
        self.menu_frames = 0
        self.launch_seconds = 0.0
        self.step_seconds = 0.0
        self.convert_seconds = 0.0
        self.cpu_levels: dict[int, int] = {}
        self._started = 0.0
        self._characters = [melee.Character(character) for character in config.characters_for(index)]
        self._stage = melee.Stage(config.stage_for(index))
        self._in_game = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)

    # -- lifecycle ------------------------------------------------------------------------

    @property
    def live(self) -> bool:
        return self.console is not None

    def start(self) -> None:
        """Launch phase 1: stop, pick the Slippi port, make the console / controllers / helpers and spawn
        Dolphin (``console.run``).  Process creation: main thread only."""

        self.stop()
        self._started = time.monotonic()
        config = self.config
        port = self._backend.pick_port(config.slippi_port + self.index)
        self.slippi_port = port
        console = self._backend.make_console(config, self.index, port)
        self.console = console
        self.launches += 1
        self.controllers = {port_id: self._backend.make_controller(console, port_id) for port_id in PORTS}
        self.helpers = {port_id: self._backend.make_menu_helper() for port_id in PORTS}
        self._backend.start(console, config)

    def connect(self) -> Any:
        """Launch phase 2: ``console.connect()`` (libmelee forks its enet worker here: main thread only), the
        controller pipes and the first frame (polled until ``boot_timeout_s``); returns that frame."""

        console = self.console
        if console is None:
            raise ConnectFailed(f"env {self.index}: start() must run before connect()")
        config = self.config
        if not console.connect():
            raise ConnectFailed(
                f"env {self.index}: could not connect to Dolphin's Slippi port {self.slippi_port}"
            )
        deadline = self._started + config.connect_timeout_s
        process = getattr(console, "_process", None)
        for port_id, controller in self.controllers.items():
            pipe = console.get_dolphin_pipes_path(port_id)
            if pipe is not None:
                wait_for_pipe_reader(pipe, deadline, process)
            if not controller.connect():
                raise ConnectFailed(f"env {self.index}: the controller on port {port_id} did not connect")
        boot_deadline = self._started + config.boot_timeout_s
        gamestate = console.step()
        while gamestate is None:
            if time.monotonic() >= boot_deadline:
                raise ConnectFailed(
                    f"env {self.index}: no frame from Dolphin within boot_timeout_s = "
                    f"{config.boot_timeout_s:g} s"
                )
            gamestate = console.step()
        return gamestate

    def settle(self, gamestate: Any) -> Any:
        """Launch phase 3: drive the menus from ``gamestate`` to the first in-game frame (``-123``) and accept
        it; no forks, so any thread may run it.  Returns the in-game gamestate."""

        gamestate = self._settle(gamestate, self._started + self.config.boot_timeout_s)
        self.launch_seconds = time.monotonic() - self._started
        self._accept(gamestate)
        return gamestate

    def launch(self) -> Any:
        """One launch attempt up to the first in-game frame (``-123``); raises on any fault."""

        self.start()
        return self.settle(self.connect())

    def stop(self) -> None:
        """Disconnect the controllers and kill Dolphin (idempotent)."""

        console = self.console
        if console is None:
            return
        timeout = self.config.stop_timeout_s
        # The graceful stop runs BEFORE the controllers are disconnected: with blocking input the
        # emulator is parked on the input pipe, and a disconnected controller can no longer feed it.
        if timeout > 0.0 and not terminate_console(console, timeout, self._pump):
            LOG.warning(
                "env %d: Dolphin did not exit within stop_timeout_s = %g s; killing it (a replay written "
                "by this instance may lack its metadata block)",
                self.index,
                timeout,
            )
        for controller in self.controllers.values():
            disconnect = getattr(controller, "disconnect", None)
            if disconnect is not None:
                try:
                    disconnect()
                except Exception as error:
                    LOG.warning("env %d: controller disconnect failed: %s", self.index, error)
        try:
            console.stop()
        except Exception as error:
            LOG.warning("env %d: console stop failed: %s", self.index, error)
        self.console = None
        self.controllers = {}
        self.helpers = {}
        self.parked = None

    # -- frames ---------------------------------------------------------------------------

    def _pump(self) -> None:
        """One neutral frame: what an emulator parked on ``blocking_input`` needs to see a signal."""

        console = self.console
        if console is None:
            return
        self.send(dict.fromkeys(self.config.controlled_ports, neutral_row()))
        console.step()

    def send(self, rows: Mapping[int, ControllerRow]) -> None:
        """Queue one row per environment port (0 -> Dolphin port 1, 1 -> port 2)."""

        for env_port, row in rows.items():
            send_controller(self._melee, self.controllers[PORTS[env_port]], row)

    def advance(self) -> Any:
        """One game frame: exactly one poll in game (``None`` is a fault), the menu loop otherwise."""

        if self.console is None:
            raise ConnectFailed(f"env {self.index}: Dolphin is not running")
        gamestate = self.console.step()
        if gamestate is None:
            frame = None if self.gamestate is None else int(self.gamestate.frame)
            raise ConsoleTimeout(
                f"env {self.index}: no frame within console_timeout_s = {self.config.console_timeout_s:g} s "
                f"(last frame {frame})"
            )
        gamestate = self._settle(gamestate, None)
        self._accept(gamestate)
        return gamestate

    def _settle(self, gamestate: Any, deadline: float | None) -> Any:
        """Drive the menus until the game is on; check the game start; return the in-game gamestate."""

        assert self.console is not None
        config = self.config
        menu_frames = 0
        autostart_after = max(MENU_AUTOSTART_FRAMES, config.menu_idle(self.index))
        while gamestate.menu_state not in self._in_game:
            if deadline is None:
                deadline = time.monotonic() + config.boot_timeout_s
            if menu_frames >= config.menu_frame_cap:
                raise MenuStall(
                    f"env {self.index}: still in {gamestate.menu_state.name} after {menu_frames} menu frames"
                )
            if time.monotonic() >= deadline:
                raise MenuStall(
                    f"env {self.index}: still in {gamestate.menu_state.name} after the menu deadline"
                )
            opponent_ready = self._opponent_ready(gamestate)
            for i, port_id in enumerate(PORTS):
                kind = config.players[i]
                self.helpers[port_id].menu_helper_simple(
                    gamestate,
                    self.controllers[port_id],
                    character_selected=self._characters[i],
                    costume=config.costumes[i],
                    stage_selected=self._stage,
                    cpu_level=self.cpu_level if kind == "cpu" else 0,
                    connect_code=None,
                    autostart=i == 0 and menu_frames > autostart_after and opponent_ready,
                    swag=False,
                )
            gamestate = self.console.step()
            while (
                gamestate is None
            ):  # loading screens may exceed the per-frame timeout; menus are safe to re-poll
                if time.monotonic() >= deadline:
                    raise MenuStall(f"env {self.index}: no frame while menuing before the menu deadline")
                gamestate = self.console.step()
            menu_frames += 1
        self.menu_frames += menu_frames
        if int(gamestate.frame) == INITIAL_FRAME_INDEX:
            self.converter.reset()
            self._check_characters(gamestate)
            self._check_stage(gamestate)
        return gamestate

    def _opponent_ready(self, gamestate: Any) -> bool:
        """At the character select: is the cpu port toggled to CPU at the configured level, slider released?

        ``autostart`` waits for this.  libmelee's ``choose_character`` applies the CPU toggle and slider only
        after the port's character coin is down, and a START pressed earlier starts the match with a
        human-type, idle port 2 (observed 22 Aug 2026: game-start ``cpu_levels {1: 0, 2: 0}``, nobody moved).
        """

        melee = self._melee
        if self.config.players[1] != "cpu" or gamestate.menu_state is not melee.Menu.CHARACTER_SELECT:
            return True
        state = gamestate.players.get(PORTS[1])
        if state is None:
            return False
        is_cpu = getattr(state, "controller_status", None) is melee.ControllerStatus.CONTROLLER_CPU
        level = int(getattr(state, "cpu_level", 0) or 0)
        holding = bool(getattr(state, "is_holding_cpu_slider", False))
        return is_cpu and level == self.cpu_level and not holding

    def _check_characters(self, gamestate: Any) -> None:
        """At a game start: the configured characters are on their ports; remember the CPU levels Dolphin
        reports (``PlayerState.cpu_level`` from the game-start event: 0 for a bot / human port)."""

        levels: dict[int, int] = {}
        for i, port_id in enumerate(PORTS):
            player = gamestate.players.get(port_id)
            if player is None:
                raise WrongCharacter(f"env {self.index}: no player on port {port_id} at the game start")
            levels[port_id] = int(getattr(player, "cpu_level", 0) or 0)
            expected = self._characters[i]
            actual = player.character
            if int(actual.value) != int(expected.value):
                raise WrongCharacter(
                    f"env {self.index}: port {port_id} plays {actual.name} (id {int(actual.value)}), "
                    f"expected "
                    f"{expected.name} (id {int(expected.value)})"
                )
        self.cpu_levels = levels
        if self.config.players[1] == "cpu" and levels.get(PORTS[1]) != self.cpu_level:
            raise WrongCpuLevel(
                f"env {self.index}: port {PORTS[1]} started as cpu_level {levels.get(PORTS[1])}, expected "
                f"{self.cpu_level} (the CPU toggle / slider was not applied at the character select)"
            )

    def _check_stage(self, gamestate: Any) -> None:
        """At a game start: the game is on this environment's stage (the menu helper steers a cursor, and a
        missed target starts the game wherever the cursor landed)."""

        actual = getattr(gamestate, "stage", None)
        if actual is None:
            return
        value = getattr(actual, "value", actual)
        if int(value) != int(self._stage.value):
            name = getattr(actual, "name", str(actual))
            raise WrongStage(
                f"env {self.index}: the game started on {name} (id {int(value)}), expected "
                f"{self._stage.name} (id {int(self._stage.value)})"
            )

    def _accept(self, gamestate: Any) -> None:
        self.gamestate = gamestate
        self.frames += 1

    def record_fault(self, error: BaseException) -> None:
        self.faults.append(describe_error(error))

    def stats(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "slippi_port": self.slippi_port,
            "live": self.live,
            "launches": self.launches,
            "relaunches": self.relaunches,
            "faults": list(self.faults),
            "frames": self.frames,
            "ticks": self.ticks,
            "menu_frames": self.menu_frames,
            "launch_seconds": self.launch_seconds,
            "step_seconds": self.step_seconds,
            "convert_seconds": self.convert_seconds,
            "parked": self.parked is not None,
            "cpu_levels": dict(self.cpu_levels),
            "stage": int(self._stage.value),
            "characters": tuple(int(character.value) for character in self._characters),
            "frame": None if self.gamestate is None else int(self.gamestate.frame),
            "dropped_items": self.converter.dropped_items,
        }


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------


class DolphinEnv:
    """``B`` headless Dolphins behind ``EnvProtocol``; see the module docstring."""

    def __init__(
        self,
        config: DolphinEnvConfig,
        *,
        cpu_level: int = 9,
        backend: DolphinBackend | None = None,
    ) -> None:
        if not 1 <= cpu_level <= 9:
            raise ValueError("cpu_level must be in [1, 9]")
        self.config = config
        self.cpu_level = cpu_level
        melee = import_melee()
        self._melee = melee
        self._backend: DolphinBackend = LibmeleeBackend(melee) if backend is None else backend
        self.build = self._backend.prepare(config)
        enet_disconnected = getattr(getattr(melee, "slippstream", None), "EnetDisconnected", None)
        recoverable: tuple[type[BaseException], ...] = (
            DolphinFault,
            TimeoutError,
            FrameConversionError,
            OSError,
        )
        if isinstance(enet_disconnected, type):
            recoverable = (*recoverable, enet_disconnected)
        self._recoverable = recoverable
        self._instances = [
            _DolphinInstance(i, config, self._backend, melee, cpu_level) for i in range(config.num_envs)
        ]
        self._lock = threading.RLock()
        self._closed = False
        self._pending: EnvOutput | None = None
        self._last_activity = time.monotonic()
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._timing = new_env_timing()  # lever 0 (5 Sep 2026): where a step's wall goes
        self._step_threads = config.resolved_step_threads
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self._step_threads, thread_name_prefix="melee-rl-dolphin")
            if self._step_threads > 1
            else None
        )
        atexit.register(self.close)
        try:
            self._pending = self._boot_all()
        except BaseException:
            self.close()
            raise
        if config.keepalive_seconds > 0.0:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, name="melee-rl-dolphin-keepalive", daemon=True
            )
            self._keepalive_thread.start()

    # -- protocol facts -----------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return self.config.controlled_ports

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def keepalive_active(self) -> bool:
        thread = self._keepalive_thread
        return thread is not None and thread.is_alive()

    @property
    def step_threads(self) -> int:
        """The resolved pool size (``num_envs`` for ``step_threads = 0``); 1 means sequential stepping."""

        return self._step_threads

    @property
    def boot_mode(self) -> str:
        return self.config.boot

    @property
    def pool_active(self) -> bool:
        """True while the thread pool exists (``step_threads > 1`` and not closed)."""

        return self._pool is not None

    def current(self) -> EnvOutput:
        assert self._pending is not None
        return self._pending

    def gamestates(self) -> list[Any]:
        """The raw libmelee ``GameState`` of every environment, or ``None`` where one is not live.

        The frame each entry describes is the one :meth:`current` reports, so a caller must read this
        *before* the next :meth:`step`.  Only an opponent with its own observation needs it (P12:
        MIMIC builds its features from the gamestate, not from our frame tree); everything else works
        from ``EnvOutput``.  In-process only -- with ``worker_processes > 0`` the gamestates live in
        the worker and this returns ``None`` for every environment.
        """

        with self._lock:
            return [instance.gamestate for instance in self._instances]

    def stats(self) -> list[dict[str, Any]]:
        """Per-environment counters (launches, relaunches, faults, frames, ticks, ports, ...) plus the
        environment's ``step_threads`` / ``boot`` settings."""

        with self._lock:
            return [
                {**instance.stats(), "step_threads": self._step_threads, "boot": self.config.boot}
                for instance in self._instances
            ]

    def timing(self) -> dict[str, Any]:
        """The cumulative step-timing counters (:mod:`melee_rl.env.timing`): steps, the step wall, the
        fan-out wall, the sum and the per-step maximum of the per-Dolphin advance times, conversion,
        the deferred handling, the output build, and the advance-time histogram.  A copy."""

        with self._lock:
            return {**self._timing, "advance_hist": list(self._timing["advance_hist"])}

    # -- protocol actions -----------------------------------------------------------------------

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        ports = tuple(sorted(actions))
        if ports != self.controlled_ports:
            raise ValueError(f"step needs controllers for ports {self.controlled_ports}, got {ports}")
        rows = {port: controller_rows(actions[port], self.num_envs) for port in ports}
        self.step_rows(rows)
        pending = self._pending
        assert pending is not None
        return pending

    def step_rows(
        self,
        rows: Mapping[int, Sequence[ControllerRow]],
        *,
        into: FrameBatch | None = None,
        output: bool = True,
    ) -> tuple[FrameBatch, list[int]]:
        """One environment step from plain rows; the step's ``(FrameBatch, frame_index)``.

        The core of :meth:`step` (which decodes the ``ControllerState`` batch into rows first) and the
        per-step work of a ``dolphin_mp`` worker (whose rows arrive as numpy arrays over the pipe).
        Updates the pending state, so :meth:`current` reflects the returned frame -- unless ``output`` is
        false (a worker that ships the leaves and never reads ``current``; the pending state is then
        stale).  ``into`` adopts an existing batch (the shared-memory slot of ``melee_rl.env.shm``); it is
        cleared to the neutral frame first, as a fresh batch would be.
        """

        with self._lock:
            step_started = time.perf_counter()
            self._check_open()
            ports = tuple(sorted(rows))
            if ports != self.controlled_ports:
                raise ValueError(f"step needs controllers for ports {self.controlled_ports}, got {ports}")
            batch_size = self.num_envs
            for port in ports:
                if len(rows[port]) != batch_size:
                    raise ValueError(
                        f"rows[{port}] must have batch length {batch_size}, got {len(rows[port])}"
                    )
            if into is None:
                batch = FrameBatch(batch_size)
            else:
                if into.batch != batch_size:
                    raise ValueError(f"into has {into.batch} rows, the environment {batch_size}")
                batch = into
                batch.clear()
            frame_index = [0] * batch_size
            # Parked / dead instances are handled on this thread below; the live ones are stepped in the pool.
            deferred: list[tuple[int, Any]] = []  # (row, gamestate to convert | None = launch first)
            live_rows: list[int] = []
            for i, instance in enumerate(self._instances):
                if instance.parked is not None:
                    deferred.append((i, instance.parked))
                    instance.parked = None
                elif not instance.live:
                    deferred.append((i, None))
                else:
                    live_rows.append(i)
            tasks = [
                (self._instances[i], {port: rows[port][i] for port in ports}, batch, i) for i in live_rows
            ]
            timing = self._timing
            before = [
                (self._instances[i].step_seconds, self._instances[i].convert_seconds) for i in live_rows
            ]
            tick = time.perf_counter()
            outcomes = self._fan_out(self._advance_and_write, tasks)
            timing["fanout_s"] += time.perf_counter() - tick
            advance_max = 0.0
            for i, outcome, (step_before, convert_before) in zip(live_rows, outcomes, before, strict=True):
                if isinstance(outcome, BaseException):  # a recoverable fault inside the fan-out
                    self._note_fault(self._instances[i], outcome)
                    deferred.append((i, None))
                else:
                    frame_index[i] = int(outcome.frame)
                    instance = self._instances[i]
                    advance = instance.step_seconds - step_before
                    timing["advance_s"] += advance
                    timing["convert_s"] += instance.convert_seconds - convert_before
                    timing["advance_hist"][bin_index(advance)] += 1
                    timing["instance_frames"] += 1
                    advance_max = max(advance_max, advance)
            timing["advance_max_s"] += advance_max
            tick = time.perf_counter()
            for i, gamestate in sorted(deferred, key=lambda item: item[0]):
                instance = self._instances[i]
                if gamestate is None:
                    gamestate = self._launch(instance)  # forks: main thread, one after another
                gamestate = self._write(instance, batch, i, gamestate)
                frame_index[i] = int(gamestate.frame)
            timing["deferred_s"] += time.perf_counter() - tick
            tick = time.perf_counter()
            if output:
                self._pending = self._output(batch, frame_index)
            timing["output_s"] += time.perf_counter() - tick
            timing["steps"] += 1
            timing["wall_s"] += time.perf_counter() - step_started
            self._last_activity = time.monotonic()
            return batch, frame_index

    def reset(self) -> EnvOutput:
        """Hard reset: kill and relaunch every Dolphin; the pending state is the first frame of each game."""

        with self._lock:
            self._check_open()
            for instance in self._instances:
                instance.stop()
            self._pending = self._boot_all()
            self._last_activity = time.monotonic()
            return self._pending

    def close(self) -> None:
        """Stop the keepalive thread, the pool and every Dolphin (idempotent; registered with ``atexit``)."""

        if self._closed:
            return
        self._keepalive_stop.set()
        thread = self._keepalive_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            self._closed = True
            for instance in self._instances:
                instance.stop()
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        self._keepalive_thread = None
        atexit.unregister(self.close)

    # -- launching and faults -------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("DolphinEnv is closed")

    def _fan_out(self, fn: Callable[..., Any], tasks: Sequence[tuple[Any, ...]]) -> list[Any]:
        """``fn(*task)`` for every task -- in the pool when there is one and more than one task, else inline.

        ``fn`` returns recoverable errors instead of raising them; anything else propagates."""

        pool = self._pool
        if pool is None or len(tasks) <= 1:
            return [fn(*task) for task in tasks]
        futures = [pool.submit(fn, *task) for task in tasks]
        return [future.result() for future in futures]

    def _boot_all(self) -> EnvOutput:
        if self.config.boot == "two_phase":
            self._boot_two_phase()
        else:
            stagger = self.config.launch_stagger_s
            for i, instance in enumerate(self._instances):
                if i and stagger > 0.0:
                    time.sleep(stagger)
                self._launch(instance)
        return self._collect()

    def _boot_two_phase(self) -> None:
        """Start every Dolphin, connect them all on this thread, settle them in the pool.

        A recoverable error in any phase records the fault, stops that instance and hands it to the
        sequential retries afterwards (``launch_retries`` bounds the total attempts, the two-phase one
        included).
        """

        stagger = self.config.launch_stagger_s
        fallback: list[tuple[_DolphinInstance, BaseException]] = []
        started: list[_DolphinInstance] = []
        for i, instance in enumerate(self._instances):
            if i and stagger > 0.0:
                time.sleep(stagger)
            try:
                instance.start()
            except self._recoverable as error:
                self._note_launch_failure(instance, error)
                fallback.append((instance, error))
            else:
                started.append(instance)
        first: dict[int, Any] = {}
        connected: list[_DolphinInstance] = []
        for instance in started:
            try:
                first[instance.index] = instance.connect()
            except self._recoverable as error:
                self._note_launch_failure(instance, error)
                fallback.append((instance, error))
            else:
                connected.append(instance)
        outcomes = self._fan_out(
            self._settle_task, [(instance, first[instance.index]) for instance in connected]
        )
        for instance, outcome in zip(connected, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                self._note_launch_failure(instance, outcome)
                fallback.append((instance, outcome))
        for instance, failure in sorted(fallback, key=lambda item: item[0].index):
            self._retry_launch(instance, attempts=self.config.launch_retries, failed=failure)

    def _settle_task(self, instance: _DolphinInstance, gamestate: Any) -> Any:
        """``instance.settle`` for the pool: recoverable errors are returned, not raised."""

        try:
            return instance.settle(gamestate)
        except self._recoverable as error:
            return error

    def _note_launch_failure(self, instance: _DolphinInstance, error: BaseException) -> None:
        instance.record_fault(error)
        LOG.warning("env %d: two-phase boot failed: %s", instance.index, describe_error(error))
        instance.stop()

    def _launch(self, instance: _DolphinInstance) -> Any:
        """Launch ``instance`` with ``launch_retries`` extra attempts; the first in-game gamestate."""

        return self._retry_launch(instance, attempts=self.config.launch_retries + 1, failed=None)

    def _retry_launch(
        self, instance: _DolphinInstance, *, attempts: int, failed: BaseException | None
    ) -> Any:
        """``attempts`` sequential launch attempts of ``instance`` (``failed`` is the error of an attempt that
        already happened, e.g. the two-phase boot); the first in-game gamestate, else
        ``DolphinLaunchError``."""

        last = failed
        done = 0 if failed is None else 1
        total = attempts + done
        for attempt in range(attempts):
            try:
                return instance.launch()
            except self._recoverable as error:
                last = error
                instance.record_fault(error)
                LOG.warning(
                    "env %d: launch attempt %d/%d failed: %s",
                    instance.index,
                    done + attempt + 1,
                    total,
                    describe_error(error),
                )
                instance.stop()
        plural = "s" if total != 1 else ""
        raise DolphinLaunchError(
            f"env {instance.index}: {total} launch attempt{plural} failed; last error: {describe_error(last)}"
        ) from last

    def _advance(self, instance: _DolphinInstance, rows: Mapping[int, ControllerRow]) -> Any:
        """Send and step one instance; a recoverable fault relaunches it (its first frame is returned)."""

        try:
            instance.send(rows)
            return instance.advance()
        except self._recoverable as error:
            self._note_fault(instance, error)
            return self._launch(instance)

    def _advance_and_write(
        self, instance: _DolphinInstance, rows: Mapping[int, ControllerRow], batch: FrameBatch, row: int
    ) -> Any:
        """Send, step and convert one live instance into ``batch[row]`` (a pool task).

        Returns the gamestate, or the recoverable error (the caller relaunches on the main thread).
        Accumulates the per-env ``step_seconds`` (send + console advance) and ``convert_seconds``
        (frame conversion) surfaced in :meth:`DolphinEnv.stats` (P8: the env_s micro-decomposition).
        """

        try:
            tick = time.perf_counter()
            instance.send(rows)
            gamestate = instance.advance()
            instance.step_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            instance.converter.write(batch, row, gamestate)
            instance.convert_seconds += time.perf_counter() - tick
        except self._recoverable as error:
            return error
        return gamestate

    def _write(self, instance: _DolphinInstance, batch: FrameBatch, row: int, gamestate: Any) -> Any:
        """Convert ``gamestate`` into ``batch[row]``; a rejected frame relaunches the instance."""

        try:
            instance.converter.write(batch, row, gamestate)
        except FrameConversionError as error:
            self._note_fault(instance, error)
            gamestate = self._launch(instance)
            instance.converter.write(batch, row, gamestate)
        return gamestate

    def _note_fault(self, instance: _DolphinInstance, error: BaseException) -> None:
        instance.record_fault(error)
        instance.relaunches += 1
        LOG.warning("env %d: %s -> relaunching Dolphin", instance.index, describe_error(error))
        instance.stop()

    # -- outputs ----------------------------------------------------------------------------------

    def collect_rows(self) -> tuple[FrameBatch, list[int]]:
        """The pending state as ``(FrameBatch, frame_index)``, rebuilt from every instance's latest
        gamestate (idempotent -- a parked frame stays parked; the worker mode's boot / reset reply)."""

        with self._lock:
            self._check_open()
            batch = FrameBatch(self.num_envs)
            frame_index: list[int] = []
            for i, instance in enumerate(self._instances):
                gamestate = instance.parked if instance.parked is not None else instance.gamestate
                assert gamestate is not None
                gamestate = self._write(instance, batch, i, gamestate)
                frame_index.append(int(gamestate.frame))
            return batch, frame_index

    def _collect(self) -> EnvOutput:
        """The pending state from every instance's latest gamestate."""

        batch, frame_index = self.collect_rows()
        return self._output(batch, frame_index)

    def _output(self, batch: FrameBatch, frame_index: list[int]) -> EnvOutput:
        return build_env_output(batch, frame_index, self.controlled_ports)

    # -- keepalive -----------------------------------------------------------------------------------

    def _keepalive_loop(self) -> None:
        period = self.config.keepalive_seconds
        while not self._keepalive_stop.wait(timeout=period):
            with self._lock:
                if self._closed:
                    return
                if time.monotonic() - self._last_activity < period:
                    continue
                try:
                    self._tick_all()
                except DolphinLaunchError as error:
                    LOG.error("keepalive: %s (the next step() will retry)", error)

    def _tick_all(self) -> None:
        """Step every idle, live, unparked environment once with the neutral controller."""

        neutral = neutral_row()
        changed = False
        for instance in self._instances:
            if instance.parked is not None or not instance.live:
                continue
            rows = {port: neutral for port in self.controlled_ports}
            gamestate = self._advance(instance, rows)
            instance.ticks += 1
            if int(gamestate.frame) == INITIAL_FRAME_INDEX:
                instance.parked = gamestate
            changed = True
        if changed:
            self._pending = self._collect()


__all__ = [
    "BOOT_MODES",
    "DEFAULT_DOLPHIN_PATH",
    "DEFAULT_ISO_PATH",
    "DEFAULT_SLIPPI_PORT",
    "ISO_GAME_ID",
    "ISO_REVISION",
    "MENU_AUTOSTART_FRAMES",
    "MP_START_METHODS",
    "PLAYER_KINDS",
    "PORTS",
    "BuildInfo",
    "ConnectFailed",
    "ConsoleTimeout",
    "ControllerRow",
    "DolphinBackend",
    "DolphinEnv",
    "DolphinEnvConfig",
    "DolphinFault",
    "DolphinLaunchError",
    "LibmeleeBackend",
    "MenuStall",
    "WrongCharacter",
    "WrongCpuLevel",
    "WrongStage",
    "build_env_output",
    "check_dolphin_exe",
    "check_iso_identity",
    "console_config_dir",
    "controller_rows",
    "describe_error",
    "instance_replay_dir",
    "neutral_row",
    "pick_slippi_port",
    "send_controller",
    "terminate_console",
    "wait_for_pipe_reader",
    "write_custom_rtc",
]
