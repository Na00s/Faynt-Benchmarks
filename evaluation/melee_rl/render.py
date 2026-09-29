"""Stage 2 of P9: render a Slippi ``.slp`` replay to mp4 with a rendering Dolphin and ffmpeg.

Our training and evaluation Dolphins can never produce video: libmelee's exi-inputs / fast-forward
gecko codes "internally disable melee's rendering in the same way that is used to fast-forward a
replay during playback" (``melee-0.47.3.dist-info/METADATA:32``) and libmelee refuses any non-Null
video backend on an ExiAI build (``melee/console.py:538-545``).  So the pipeline is *play first,
render second* (P9-PLAN.md §2): :mod:`melee_rl.video` plays the matchups with the normal headless
environment and keeps every clip's ``.slp``; this module replays each file in a second Dolphin
process started in **playback mode** and muxes the frame dump into a captioned mp4.

The moving parts, all of them pure functions except the two that run subprocesses:

* :func:`render_ini` / :func:`write_user_dir` -- a throwaway Dolphin user directory whose
  ``Config/Dolphin.ini`` turns frame dumping on (``[Movie] DumpFrames`` /``DumpFramesSilent``, the
  keys libmelee's own ``DumpConfig`` writes, ``console.py:313-333``, ``:763-764``, ``:796-797``) and
  whose ``Config/GFX.ini`` sets the dump bitrate and ``InternalResolutionFrameDumps``.  No
  ``DumpPath``: the dump lands at the documented ``<user>/Dump/Frames/framedump0.avi``, inside a
  directory we own.
* :func:`comm_payload` -- the playback comm file (project-slippi/slippi-wiki ``COMM_SPEC.md``):
  ``mode`` / ``replay`` / ``startFrame`` / ``endFrame`` / ``isRealTimeMode`` / ``commandId``.
  ``endFrame`` is the last frame the recording observed, so playback stops where we stopped.
* :func:`render_command` / :func:`render_environment` -- ``dolphin -b -e <iso> -u <user> -i <comm>``
  plus ``--platform <p>`` or an ``xvfb-run`` prefix and a ``DISPLAY``.
* :func:`ffmpeg_command` / :func:`ffprobe_command` / :func:`parse_ffprobe` -- the mux with the
  burned-in caption (``drawtext``, read from a file so nothing has to be escaped into the filter
  graph) and the verification of every output.
* :func:`render_replay` / :func:`render_many` -- one job end to end, errors reported instead of
  raised (a clip that fails to render still has its ``.slp``).

Which video backend actually initialises in a gVisor container without an X server or a GPU is the
one open question of the phase; :data:`RENDER_CANDIDATES` is the ladder the ``video_check`` probe
walks (``modal_app.run_video_check``), and the winner becomes the default here.
"""

from __future__ import annotations

import configparser
import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Final, Protocol

from melee_rl.env.dolphin import DEFAULT_DOLPHIN_PATH, DEFAULT_ISO_PATH
from melee_rl.env.protocol import INITIAL_FRAME_INDEX

COMM_MODE: Final[str] = "normal"
"""``mode`` of the playback comm file: one replay, played once (COMM_SPEC.md)."""
DUMP_SUBDIR: Final[str] = "Dump/Frames"
"""Where Dolphin writes frame dumps inside its user directory."""
DUMP_SUFFIXES: Final[tuple[str, ...]] = (".avi", ".mp4", ".mkv", ".png")
AUDIO_DUMP_SUBDIR: Final[str] = "Dump/Audio"
"""Where Dolphin writes its audio dumps (``DSP.DumpAudio``): ``dspdump.wav`` (the DSP mix: sound effects,
voices) and ``dtkdump.wav`` (the streamed disc audio: Melee's music).  8 Sep 2026."""
AUDIO_DUMP_NAMES: Final[tuple[str, ...]] = ("dspdump.wav", "dtkdump.wav")
DEFAULT_FONT: Final[str] = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
"""``fonts-dejavu-core`` in the video image; the caption falls back to no caption if it is missing."""
XVFB_ARGS: Final[tuple[str, ...]] = ("xvfb-run", "-a", "-s", "-screen 0 1280x1024x24")
KILL_GRACE_S: Final[float] = 10.0
"""How long a killed process group gets to close its pipes before its output is dropped."""
STOP_GRACE_S: Final[float] = 10.0
"""How long a SIGTERMed playback Dolphin gets to finalise its frame dump before the SIGKILL."""
_FILTER_SPECIALS: Final[str] = "\\':,[];%"

EFB_SCALE: Final[Mapping[int, int]] = MappingProxyType({1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 9})
"""Internal-resolution multiplier -> Ishiiruka's ``GFX.ini [Settings] EFBScale`` enum value.

``SCALE_AUTO = 0, SCALE_AUTO_INTEGRAL = 1, SCALE_1X = 2, SCALE_1_5X = 3, SCALE_2X = 4, ...``
(``VideoCommon/VideoConfig.h:42-54`` @ ``project-slippi/Ishiiruka`` v3.5.2, the version of the pinned
playback zip); kevinsung/slp-to-video maps ``{"1x": 2, "2x": 4, "3x": 6, "4x": 7}`` the same way.
**Ishiiruka has no ``InternalResolution`` key at all**, and its ``EFBScale`` default on Linux is
``SCALE_2X`` (``VideoConfig.cpp:141-145``), so a config that does not write this key renders at
1280 x 960 -- four times the pixels of native, which on a software rasteriser is what P9's stage 2 was
actually paying for.
"""

PLAYBACK_MARKERS: Final[frozenset[str]] = frozenset(
    {"PLAYBACK_START_FRAME", "PLAYBACK_END_FRAME", "GAME_END_FRAME", "CURRENT_FRAME", "NO_GAME"}
)
"""The progress markers the playback build prints on stdout under ``--cout``.

``EXI_DeviceSlippi.cpp:1057`` prints ``[CURRENT_FRAME] n`` *every frame*; ``:1300-1303`` prints
``[PLAYBACK_START_FRAME]`` / ``[GAME_END_FRAME]`` / ``[PLAYBACK_END_FRAME]`` once, when the replay
loads; ``SlippiReplayComm.cpp:89`` prints ``[NO_GAME]`` when the queue empties.  All of them are gated
on ``SConfig::m_coutEnabled`` (``:178``), which ``--cout`` sets (``Main.cpp:322-323``, ``:389``).
"""
_MARKER_RE: Final[re.Pattern[str]] = re.compile(r"\[([A-Z_]+)\]\s*(-?\d+)?")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderConfig:
    """``[video.render]``: how a ``.slp`` becomes an mp4.

    ``dolphin_path`` / ``iso_path`` are the rendering Dolphin and the ISO (the same image assets the
    environment uses; a playback-only build can be pointed at here once one is pinned).  ``backend``
    is Dolphin's ``GFXBackend`` (``OGL`` / ``Vulkan`` / ``Software Renderer``); ``platform`` is the
    mainline ``--platform`` flag (``headless`` or nothing) and ``xvfb`` wraps the call in
    ``xvfb-run`` instead, with ``display`` setting ``DISPLAY`` for an X server started elsewhere.
    ``internal_resolution`` 1 is native 640 x 480, ``bitrate_kbps`` the dump bitrate;
    ``emulation_speed`` 0 is "as fast as the machine allows".  ``crf`` / ``caption`` / ``font`` drive
    the ffmpeg pass and ``workers`` renders that many replays at once.

    A playback Dolphin **does not exit when the replay ends** (measured 26 Aug 2026), so the render
    call is bounded and the dump it leaves behind is used: the bound is ``max(min_timeout_s,
    replay seconds x speed_factor)``, capped by ``timeout_s``, and ffmpeg then trims the dump to the
    replay's exact length.  ``speed_factor`` is wall seconds per second of replay -- software GL in a
    container measured about 14x, so 20 leaves headroom.
    """

    dolphin_path: str = DEFAULT_DOLPHIN_PATH
    iso_path: str = DEFAULT_ISO_PATH
    backend: str = "OGL"
    platform: str | None = "headless"
    xvfb: bool = False
    display: str | None = None
    cout: bool = True
    hide_seekbar: bool = True
    stop_grace_s: float = STOP_GRACE_S
    internal_resolution: int = 1
    bitrate_kbps: int = 3000
    dump_format: str = "avi"
    dump_codec: str | None = None
    emulation_speed: float = 0.0
    caption: bool = True
    font: str = DEFAULT_FONT
    font_size: int = 18
    crf: int = 20
    audio: bool = True
    """Dump the game audio while the replay plays and mix it into the clip (8 Sep 2026); False = silent."""
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    timeout_s: float = 1800.0
    speed_factor: float = 20.0
    min_timeout_s: float = 120.0
    workers: int = 1
    keep_work_dir: bool = False

    def __post_init__(self) -> None:
        if not self.dolphin_path:
            raise ValueError("[video.render] dolphin_path must name the rendering Dolphin executable")
        if not self.iso_path:
            raise ValueError("[video.render] iso_path must name the Melee ISO")
        if not self.backend:
            raise ValueError("[video.render] backend must be a Dolphin GFXBackend name")
        if not self.dump_format:
            raise ValueError("[video.render] dump_format must be a Dolphin dump format (avi, mp4, png)")
        if self.internal_resolution not in EFB_SCALE:
            raise ValueError(
                "[video.render] internal_resolution must be one of "
                f"{sorted(EFB_SCALE)} (1 = native 640x480); got {self.internal_resolution}"
            )
        if self.stop_grace_s < 0.0:
            raise ValueError("[video.render] stop_grace_s must be >= 0")
        if self.bitrate_kbps < 1:
            raise ValueError("[video.render] bitrate_kbps must be >= 1")
        if self.emulation_speed < 0.0:
            raise ValueError("[video.render] emulation_speed must be >= 0 (0 = unlimited)")
        if self.font_size < 1:
            raise ValueError("[video.render] font_size must be >= 1")
        if self.crf < 0 or self.crf > 51:
            raise ValueError("[video.render] crf must be in [0, 51]")
        if not self.timeout_s > 0.0:
            raise ValueError("[video.render] timeout_s must be positive")
        if not self.speed_factor > 0.0:
            raise ValueError("[video.render] speed_factor must be positive")
        if not self.min_timeout_s > 0.0:
            raise ValueError("[video.render] min_timeout_s must be positive")
        if self.workers < 1:
            raise ValueError("[video.render] workers must be >= 1")


@dataclass(frozen=True)
class RenderCandidate:
    """One rung of the backend ladder the ``video_check`` probe walks (P9-PLAN.md §6)."""

    name: str
    backend: str
    platform: str | None
    display: str | None = None
    xvfb: bool = False


RENDER_CANDIDATES: Final[tuple[RenderCandidate, ...]] = (
    RenderCandidate("headless-ogl", backend="OGL", platform="headless"),
    RenderCandidate("headless-vulkan", backend="Vulkan", platform="headless"),
    RenderCandidate("headless-software", backend="Software Renderer", platform="headless"),
    RenderCandidate("xvfb-ogl", backend="OGL", platform=None, xvfb=True),
)
"""``--platform headless`` x {OGL, Vulkan, Software Renderer}, then Xvfb x OGL."""


def candidate_config(config: RenderConfig, candidate: RenderCandidate) -> RenderConfig:
    """``config`` with only the video knobs of ``candidate`` applied."""

    return replace(
        config,
        backend=candidate.backend,
        platform=candidate.platform,
        display=candidate.display,
        xvfb=candidate.xvfb,
    )


# ---------------------------------------------------------------------------
# jobs and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderJob:
    """One replay to render: the ``.slp``, where the mp4 goes, its caption and the frames recorded."""

    slp: Path
    target: Path
    caption: str = ""
    frames: int = 0

    @property
    def name(self) -> str:
        return self.target.stem

    @property
    def seconds(self) -> float:
        """The replay's length in seconds (Melee runs at 60 fps)."""

        return max(0.0, self.frames / 60.0)

    @property
    def end_frame(self) -> int | None:
        """The last frame index the recording observed (games start at ``-123``); ``None`` = play all."""

        return None if self.frames < 1 else INITIAL_FRAME_INDEX + self.frames - 1


@dataclass
class RenderResult:
    """What came of one :class:`RenderJob`; ``ok`` iff a verified video file exists."""

    slp: str
    target: str
    ok: bool = False
    video: str | None = None
    returncode: int | None = None
    seconds: float = 0.0
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    expected_frames: int = 0
    codec: str | None = None
    dump: str | None = None
    audio_tracks: int = 0
    """How many audio dumps were mixed into the clip (0 = silent).  8 Sep 2026."""
    killed: bool = False
    error: str | None = None
    startup_s: float | None = None
    render_s: float | None = None
    render_fps: float | None = None
    frames_rendered: int = 0
    slowdown: float | None = None
    finished: bool = False
    stop_reason: str = ""
    commands: list[dict[str, Any]] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        """A JSON-able row for the manifest."""

        return {
            "slp": self.slp,
            "video": self.video,
            "ok": self.ok,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3),
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "expected_frames": self.expected_frames,
            "codec": self.codec,
            "dump": self.dump,
            "killed": self.killed,
            "error": self.error,
            "startup_s": None if self.startup_s is None else round(self.startup_s, 3),
            "render_s": None if self.render_s is None else round(self.render_s, 3),
            "render_fps": None if self.render_fps is None else round(self.render_fps, 3),
            "frames_rendered": self.frames_rendered,
            "slowdown": None if self.slowdown is None else round(self.slowdown, 3),
            "finished": self.finished,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class CommandResult:
    """One subprocess: its argv, exit code, captured output and wall time (never raises)."""

    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def record(self, *, tail: int = 600) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3),
            "stdout": self.stdout[-tail:],
            "stderr": self.stderr[-tail:],
            "error": self.error,
        }


class Runner(Protocol):
    """Runs one command; :func:`run_command` is the real one, tests inject their own."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> CommandResult: ...


def run_command(
    command: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> CommandResult:
    """``subprocess.run`` with captured output and a wall-clock bound; failures become the result."""

    argv = tuple(str(part) for part in command)
    environment = None if env is None else {**os.environ, **env}
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            cwd=None if cwd is None else str(cwd),
            start_new_session=True,  # its own process group: a timeout kills the children too
        )
    except OSError as error:
        return CommandResult(argv, None, "", "", time.perf_counter() - started, repr(error))
    with process:
        try:
            out, err = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            _kill_group(process)
            try:
                out, err = process.communicate(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:  # a grandchild still holds the pipes
                out, err = "", ""
            elapsed = time.perf_counter() - started
            return CommandResult(argv, None, out or "", err or "", elapsed, repr(error))
        return CommandResult(argv, process.returncode, out, err, time.perf_counter() - started)


# ---------------------------------------------------------------------------
# playback progress: stopping the render when the replay ends, not when a budget runs out
# ---------------------------------------------------------------------------


def parse_playback_line(line: str) -> tuple[str, int | None] | None:
    """One ``--cout`` progress marker, or ``None`` for any other output.

    ``"[CURRENT_FRAME] 42"`` -> ``("CURRENT_FRAME", 42)``, ``"[NO_GAME]"`` -> ``("NO_GAME", None)``.
    The marker is *searched* for rather than anchored because Dolphin interleaves its own logging on
    the same stream, exactly as kevinsung/slp-to-video's ``line.includes(...)`` does.
    """

    match = _MARKER_RE.search(line)
    if match is None:
        return None
    name = match.group(1)
    if name not in PLAYBACK_MARKERS:
        return None
    if name == "NO_GAME":
        return name, None
    raw = match.group(2)
    return None if raw is None else (name, int(raw))


@dataclass
class PlaybackProgress:
    """What the frame markers said: where the replay ends, how far it got, and how fast.

    Folding these in is what turns "the render took its whole budget" into a measurement:
    :attr:`startup_s` is the wall clock spent before the first rendered frame (Dolphin start, Melee
    boot, the replay load) and :attr:`fps` the steady-state rate after it.
    """

    start_frame: int | None = None
    end_frame: int | None = None
    game_end_frame: int | None = None
    last_frame: int | None = None
    frames_rendered: int = 0
    startup_s: float | None = None
    render_s: float | None = None
    reason: str = "running"

    @property
    def target_frame(self) -> int | None:
        """The frame the render is done at: the **shorter** of the two bounds Dolphin reported.

        ``[PLAYBACK_END_FRAME]`` is what our comm file asked for and ``[GAME_END_FRAME]`` is what the
        replay actually holds; asking for more frames than the file has would otherwise never complete.
        """

        bounds = [frame for frame in (self.end_frame, self.game_end_frame) if frame is not None]
        return min(bounds) if bounds else None

    @property
    def complete(self) -> bool:
        target = self.target_frame
        return target is not None and self.last_frame is not None and self.last_frame >= target

    @property
    def fps(self) -> float | None:
        """Rendered frames per wall-clock second, over the intervals between markers."""

        if self.render_s is None or self.render_s <= 0.0 or self.frames_rendered < 2:
            return None
        return (self.frames_rendered - 1) / self.render_s

    @property
    def finished(self) -> bool:
        """Did the replay run out, as opposed to the safety timeout firing?"""

        if self.reason in ("end-frame", "no-game"):
            return True
        return self.reason == "exit" and self.frames_rendered > 0

    def observe(self, name: str, value: int | None, elapsed: float) -> str | None:
        """Fold one marker in; returns a stop reason once the render is done, else ``None``."""

        if name == "PLAYBACK_START_FRAME":
            self.start_frame = value
        elif name == "PLAYBACK_END_FRAME":
            self.end_frame = value
        elif name == "GAME_END_FRAME":
            self.game_end_frame = value
        elif name == "NO_GAME":
            # Also printed at boot, before anything has been rendered, so it only ends a live render.
            return "no-game" if self.frames_rendered > 0 else None
        elif name == "CURRENT_FRAME" and value is not None:
            if self.frames_rendered == 0:
                self.startup_s = elapsed
            self.frames_rendered += 1
            self.last_frame = value
            self.render_s = max(0.0, elapsed - (self.startup_s or 0.0))
            if self.complete:
                return "end-frame"
        return None

    def record(self) -> dict[str, Any]:
        """A JSON-able row for the manifest and the bench report."""

        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "game_end_frame": self.game_end_frame,
            "target_frame": self.target_frame,
            "last_frame": self.last_frame,
            "frames_rendered": self.frames_rendered,
            "startup_s": None if self.startup_s is None else round(self.startup_s, 3),
            "render_s": None if self.render_s is None else round(self.render_s, 3),
            "fps": None if self.fps is None else round(self.fps, 3),
            "reason": self.reason,
            "finished": self.finished,
        }


class PlaybackRunner(Protocol):
    """Runs the Dolphin call and reports its progress; :func:`run_playback` is the real one."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
        stop_grace_s: float = ...,
    ) -> tuple[CommandResult, PlaybackProgress]: ...


def run_playback(
    command: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    stop_grace_s: float = STOP_GRACE_S,
) -> tuple[CommandResult, PlaybackProgress]:
    """Run a playback Dolphin, stopping it when the replay ends rather than when a budget runs out.

    A playback Dolphin never exits on its own -- it keeps the emulator alive watching the comm file for
    the next replay -- so the previous code ran it to a fixed timeout and SIGKILLed it, which both
    wasted the whole budget on every clip and left an AVI whose index was never finalised (``ffprobe``
    cannot read one).  This reads the ``--cout`` markers instead: the render ends at
    :attr:`PlaybackProgress.target_frame`, or at ``[NO_GAME]``, and Dolphin is then **SIGTERMed** so it
    can close the dump, with ``timeout_s`` left as a safety net for a build that prints nothing.
    """

    argv = tuple(str(part) for part in command)
    environment = None if env is None else {**os.environ, **env}
    progress = PlaybackProgress()
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line buffered: a marker must be visible the moment Dolphin prints it
            env=environment,
            cwd=None,
            start_new_session=True,  # its own process group, so xvfb-run's child dies with it
        )
    except OSError as error:
        progress.reason = "error"
        return CommandResult(argv, None, "", "", time.perf_counter() - started, repr(error)), progress

    stopped = threading.Event()

    def stop(reason: str) -> None:
        if stopped.is_set():
            return
        stopped.set()
        progress.reason = reason
        _stop_group(process, stop_grace_s)

    watchdog = threading.Timer(timeout_s, stop, ("timeout",))
    watchdog.daemon = True
    watchdog.start()
    errors: list[str] = []
    drain = threading.Thread(target=_drain, args=(process.stderr, errors), daemon=True)
    drain.start()
    out: list[str] = []
    try:
        if process.stdout is not None:
            for line in process.stdout:
                out.append(line)
                marker = parse_playback_line(line)
                if marker is None:
                    continue
                reason = progress.observe(marker[0], marker[1], time.perf_counter() - started)
                if reason is not None:
                    stop(reason)
    finally:
        watchdog.cancel()
        try:
            process.wait(timeout=stop_grace_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=stop_grace_s)
        drain.join(timeout=stop_grace_s)
        with contextlib.suppress(OSError):
            if process.stdout is not None:
                process.stdout.close()
    if not stopped.is_set():
        progress.reason = "exit"
    elapsed = time.perf_counter() - started
    return CommandResult(argv, process.returncode, "".join(out), "".join(errors), elapsed), progress


def _drain(stream: IO[str] | None, into: list[str]) -> None:
    """Consume stderr on its own thread so a chatty Dolphin cannot block on a full pipe."""

    if stream is None:
        return
    with contextlib.suppress(ValueError, OSError):
        for line in stream:
            into.append(line)


def _stop_group(process: subprocess.Popen[str], grace_s: float) -> None:
    """SIGTERM the process group, then SIGKILL whatever survives ``grace_s``.

    The SIGTERM is the point: NunoDasNeves/slp-to-mp4 stops Dolphin the same way (``terminate()``, then
    ``kill()`` after 5 s) and a Dolphin killed outright mid-write leaves an unindexed AVI.  The sweep
    afterwards is equally load-bearing -- ``xvfb-run`` execs Dolphin as a child, and a survivor holding
    the stdout pipe would keep the read loop from ever seeing EOF.
    """

    _signal_group(process, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_s)
    _kill_group(process)


def _signal_group(process: subprocess.Popen[str], number: int) -> bool:
    try:
        os.killpg(os.getpgid(process.pid), number)
    except OSError:
        return False
    return True


def _kill_group(process: subprocess.Popen[str]) -> None:
    """SIGKILL the whole process group.

    ``xvfb-run`` execs Dolphin as a child and a plain ``kill`` leaves the grandchild holding the
    stdout pipe, so the read after the timeout never returns -- one hung Dolphin would hang the job.
    """

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            process.kill()


# ---------------------------------------------------------------------------
# the Dolphin side
# ---------------------------------------------------------------------------


class _Ini(configparser.ConfigParser):
    """A parser that keeps Dolphin's CamelCase option names.

    Dolphin's own reader is case-insensitive (``Common/IniFile.h:18,87`` uses
    ``CaseInsensitiveStringCompare``), so the lowercasing ``configparser`` does by default was never a
    bug -- but an ini that reads like the one Dolphin writes is far easier to compare against the
    reference tools, which all set ``optionxform = str`` for the same reason.
    """

    def optionxform(self, optionstr: str) -> str:
        return optionstr


def render_ini(config: RenderConfig) -> dict[str, str]:
    """``Dolphin.ini`` / ``GFX.ini`` / ``Logger.ini`` of the throwaway render user directory."""

    dolphin = _Ini()
    dolphin["Core"] = {
        "GFXBackend": config.backend,
        "EmulationSpeed": str(config.emulation_speed),
        "SlippiPlaybackDisplayFrameIndex": "False",
    }
    # The window is pinned to native so that nothing -- a dump taken at window size, an auto-sized
    # window -- can inherit the 1280x1024 Xvfb screen.
    dolphin["Display"] = {
        "Fullscreen": "False",
        "RenderToMain": "False",
        "RenderWindowWidth": "640",
        "RenderWindowHeight": "480",
        "RenderWindowAutoSize": "False",
    }
    dolphin["DSP"] = {
        "Backend": "No Audio Output",
        "Volume": "0",
        # The mixer writes the dumps on push, independent of the (muted) backend: the clip gets its sound
        # from <user>/Dump/Audio/{dspdump,dtkdump}.wav (8 Sep 2026).
        "DumpAudio": "True" if config.audio else "False",
        "DumpAudioSilent": "True",
    }
    dolphin["Movie"] = {"DumpFrames": "True", "DumpFramesSilent": "True"}
    dolphin["Input"] = {"BackgroundInput": "True"}
    # Nothing may be drawn over the frame dump: NunoDasNeves/slp-to-mp4 turns the three wx bars off
    # ("doesn't render properly with these enabled") and kevinsung/slp-to-video turns the OSD off.
    dolphin["Interface"] = {
        "ConfirmStop": "False",
        "PauseOnFocusLost": "False",
        "OnScreenDisplayMessages": "False",
        "ShowToolbar": "False",
        "ShowStatusbar": "False",
        "ShowSeekbar": "False",
    }
    gfx = _Ini()
    settings = {
        "DumpFormat": config.dump_format,
        "BitrateKbps": str(config.bitrate_kbps),
        "InternalResolutionFrameDumps": "True",
        "EFBScale": str(EFB_SCALE[config.internal_resolution]),
        "MSAA": "1",
        "SSAA": "False",
    }
    if config.dump_codec:
        settings["DumpCodec"] = config.dump_codec
    gfx["Settings"] = settings
    gfx["Hardware"] = {"VSync": "False"}
    logger = _Ini()
    logger["Options"] = {"WriteToFile": "True", "Verbosity": "2"}
    return {
        "Dolphin.ini": _ini_text(dolphin),
        "GFX.ini": _ini_text(gfx),
        "Logger.ini": _ini_text(logger),
    }


def _ini_text(parser: configparser.ConfigParser) -> str:
    stream = io.StringIO()
    parser.write(stream)
    return stream.getvalue()


def write_user_dir(path: str | Path, config: RenderConfig) -> Path:
    """Create the render user directory (``Config/*.ini`` + an empty ``Dump/Frames``) and return it."""

    user = Path(path)
    (user / "Config").mkdir(parents=True, exist_ok=True)
    (user / DUMP_SUBDIR).mkdir(parents=True, exist_ok=True)
    (user / AUDIO_DUMP_SUBDIR).mkdir(parents=True, exist_ok=True)
    for name, text in render_ini(config).items():
        (user / "Config" / name).write_text(text)
    return user


def dump_files(user_dir: str | Path) -> list[Path]:
    """The frame-dump files Dolphin left in ``<user>/Dump/Frames`` (newest last)."""

    directory = Path(user_dir) / DUMP_SUBDIR
    if not directory.is_dir():
        return []
    found = [
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in DUMP_SUFFIXES and item.stat().st_size > 0
    ]
    return sorted(found, key=lambda item: (item.stat().st_mtime, item.name))


def audio_dump_files(user_dir: str | Path) -> list[Path]:
    """The non-empty audio dumps Dolphin left in ``<user>/Dump/Audio``, DSP first, then DTK (a fixed order
    so the ffmpeg graph is stable)."""

    directory = Path(user_dir) / AUDIO_DUMP_SUBDIR
    if not directory.is_dir():
        return []
    found = []
    for name in AUDIO_DUMP_NAMES:
        item = directory / name
        if item.is_file() and item.stat().st_size > 44:  # more than a bare RIFF header
            found.append(item)
    return found


def comm_payload(
    slp: str | Path,
    *,
    end_frame: int | None = None,
    start_frame: int = INITIAL_FRAME_INDEX,
    command_id: str = "melee-rl",
) -> dict[str, Any]:
    """The playback comm file Dolphin reads through ``-i`` (slippi-wiki ``COMM_SPEC.md``)."""

    payload: dict[str, Any] = {
        "mode": COMM_MODE,
        "replay": str(slp),
        "startFrame": int(start_frame),
        "isRealTimeMode": False,
        "outputOverlayFiles": False,
        "commandId": command_id,
    }
    if end_frame is not None:
        payload["endFrame"] = int(end_frame)
    return payload


def render_command(config: RenderConfig, *, user_dir: str | Path, comm_path: str | Path) -> list[str]:
    """``[xvfb-run ...] <dolphin> -b -e <iso> -u <user> -i <comm> [--cout] [--hide-seekbar] [--platform]``.

    ``--cout`` turns on the per-frame progress markers :func:`run_playback` stops on, and
    ``--hide-seekbar`` keeps the playback seekbar off the screen.  Both are real switches on the pinned
    playback build (``DolphinWX/Main.cpp:320-323`` @ v3.5.2) but stay optional, so a build without them
    is still drivable (on such a build the render falls back to the safety timeout).
    """

    command = list(XVFB_ARGS) if config.xvfb else []
    command += [
        config.dolphin_path,
        "-b",
        "-e",
        config.iso_path,
        "-u",
        str(user_dir),
        "-i",
        str(comm_path),
    ]
    if config.cout:
        command.append("--cout")
    if config.hide_seekbar:
        command.append("--hide-seekbar")
    if config.platform:
        command += ["--platform", config.platform]
    return command


def render_timeout(config: RenderConfig, job: RenderJob) -> float:
    """How long the render call gets: the replay's length scaled by ``speed_factor``, within bounds."""

    return min(config.timeout_s, max(config.min_timeout_s, job.seconds * config.speed_factor))


def render_environment(config: RenderConfig) -> dict[str, str] | None:
    """Extra environment variables for the render call (``DISPLAY`` for an external X server)."""

    return None if config.display is None else {"DISPLAY": config.display}


# ---------------------------------------------------------------------------
# the ffmpeg side
# ---------------------------------------------------------------------------


def escape_filter_value(text: str) -> str:
    """Escape a value for an ffmpeg filter-graph option (``:``, ``,``, ``[``, ``'``, ``\\`` ...)."""

    out = text.replace("\\", "\\\\")
    for char in _FILTER_SPECIALS:
        if char == "\\":
            continue
        out = out.replace(char, "\\" + char)
    return out


def drawtext_filter(config: RenderConfig, caption_file: str | Path) -> str:
    """The ``drawtext`` filter that burns the caption in; the text is read from a file, never quoted."""

    return (
        f"drawtext=fontfile={escape_filter_value(config.font)}:"
        f"textfile={escape_filter_value(str(caption_file))}:reload=0:"
        f"x=(w-text_w)/2:y=h-text_h-12:fontsize={config.font_size}:fontcolor=white:"
        "box=1:boxcolor=black@0.55:boxborderw=8"
    )


def ffmpeg_command(
    config: RenderConfig,
    *,
    source: str | Path,
    target: str | Path,
    caption_file: str | Path | None = None,
    trim_seconds: float | None = None,
    bitrate_kbps: int | None = None,
    audio_files: Sequence[str | Path] | None = None,
) -> list[str]:
    """The frame dump muxed into an mp4: optional caption, optional trim, and the audio dumps (if any)
    mixed into one AAC track -- without them the clip is silent (``-an``).

    Both the frame dump and the audio dumps run on emulated time from the boot, so they line up without
    an offset; ``-t`` trims video and audio alike.  With audio the caption moves into the same
    ``-filter_complex`` graph (ffmpeg refuses ``-vf`` beside a complex graph that feeds the output).
    """

    audio = [str(path) for path in (audio_files or [])]
    command = [config.ffmpeg, "-y", "-loglevel", "error", "-i", str(source)]
    for path in audio:
        command += ["-i", path]
    if trim_seconds is not None:
        command += ["-t", f"{trim_seconds:g}"]
    if audio:
        graph = []
        video_label = "0:v"
        if caption_file is not None:
            graph.append(f"[0:v]{drawtext_filter(config, caption_file)}[v]")
            video_label = "v"
        inputs = "".join(f"[{i + 1}:a]" for i in range(len(audio)))
        if len(audio) > 1:
            graph.append(f"{inputs}amix=inputs={len(audio)}:duration=first:dropout_transition=0[a]")
        else:
            graph.append(f"{inputs}anull[a]")
        command += [
            "-filter_complex",
            ";".join(graph),
            "-map",
            f"[{video_label}]" if video_label == "v" else "0:v",
            "-map",
            "[a]",
        ]
    elif caption_file is not None:
        command += ["-vf", drawtext_filter(config, caption_file)]
    command += ["-c:v", "libx264", "-preset", "veryfast"]
    if bitrate_kbps is not None:
        command += ["-b:v", f"{bitrate_kbps}k"]
    else:
        command += ["-crf", str(config.crf)]
    command += ["-pix_fmt", "yuv420p"]
    if audio:
        command += ["-c:a", "aac", "-b:a", "128k"]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(target)]
    return command


def ffprobe_command(config: RenderConfig, path: str | Path) -> list[str]:
    """``ffprobe`` in JSON mode: duration, resolution, frame count, codec."""

    return [
        config.ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]


def parse_ffprobe(output: str) -> dict[str, Any]:
    """The fields :func:`ffprobe_command` asks for; unparsable output yields all ``None``."""

    empty: dict[str, Any] = {
        "duration_s": None,
        "width": None,
        "height": None,
        "frames": None,
        "codec": None,
    }
    try:
        payload = json.loads(output)
    except (ValueError, TypeError):
        return empty
    if not isinstance(payload, Mapping):
        return empty
    streams = payload.get("streams") or []
    stream = streams[0] if streams and isinstance(streams[0], Mapping) else {}
    fmt = payload.get("format") or {}
    return {
        "duration_s": _number(fmt.get("duration"), float),
        "width": _number(stream.get("width"), int),
        "height": _number(stream.get("height"), int),
        "frames": _number(stream.get("nb_frames"), int),
        "codec": stream.get("codec_name") or None,
    }


def _number(value: Any, kind: type) -> Any:
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_replay(
    config: RenderConfig,
    job: RenderJob,
    *,
    work_dir: str | Path,
    runner: Runner | None = None,
    playback: PlaybackRunner | None = None,
) -> RenderResult:
    """Render one replay; every failure is reported in the result, never raised.

    ``runner`` runs ffmpeg / ffprobe, ``playback`` runs Dolphin and reports how far the replay got.
    """

    call: Runner = run_command if runner is None else runner
    play: PlaybackRunner = run_playback if playback is None else playback
    result = RenderResult(slp=str(job.slp), target=str(job.target))
    started = time.perf_counter()
    try:
        if not Path(job.slp).is_file():
            result.error = f"replay {job.slp} does not exist"
            return result
        job_dir = Path(work_dir) / job.name
        user = write_user_dir(job_dir / "user", config)
        comm = job_dir / "comm.json"
        comm.write_text(json.dumps(comm_payload(job.slp, end_frame=job.end_frame, command_id=job.name)))
        command = render_command(config, user_dir=user, comm_path=comm)
        dolphin, progress = play(
            command,
            timeout_s=render_timeout(config, job),
            env=render_environment(config),
            stop_grace_s=config.stop_grace_s,
        )
        result.commands.append(dolphin.record())
        result.returncode = dolphin.returncode
        result.startup_s = progress.startup_s
        result.render_s = progress.render_s
        result.render_fps = progress.fps
        result.frames_rendered = progress.frames_rendered
        result.finished = progress.finished
        result.stop_reason = progress.reason
        result.killed = progress.reason != "exit"  # we ended it: it never exits on its own
        result.expected_frames = job.frames
        if job.seconds > 0.0:
            result.slowdown = dolphin.seconds / job.seconds
        dumps = dump_files(user)
        if not dumps:
            detail = dolphin.error or dolphin.stderr[-200:]
            result.error = (
                f"dolphin produced no frame dump in {user / DUMP_SUBDIR} "
                f"(exit {dolphin.returncode}): {detail}"
            )
            return result
        dump = dumps[-1]
        result.dump = str(dump)
        audio = audio_dump_files(user) if config.audio else []
        result.audio_tracks = len(audio)
        Path(job.target).parent.mkdir(parents=True, exist_ok=True)
        caption_file = _write_caption(config, job_dir, job)
        mux = call(
            ffmpeg_command(
                config,
                source=dump,
                target=job.target,
                caption_file=caption_file,
                trim_seconds=job.seconds or None,  # the dump runs past the replay: cut it to length
                audio_files=audio,
            ),
            timeout_s=config.timeout_s,
        )
        result.commands.append(mux.record())
        if not mux.ok or not Path(job.target).is_file():
            result.error = f"ffmpeg failed ({mux.returncode}): {mux.error or mux.stderr[-200:]}"
            return result
        probe = call(ffprobe_command(config, job.target), timeout_s=60.0)
        result.commands.append(probe.record())
        info = parse_ffprobe(probe.stdout)
        result.duration_s = info["duration_s"]
        result.width = info["width"]
        result.height = info["height"]
        result.frames = info["frames"]
        result.codec = info["codec"]
        if not probe.ok or info["duration_s"] is None:
            result.error = f"ffprobe could not verify {job.target}: {probe.error or probe.stderr[-200:]}"
            return result
        result.video = str(job.target)
        result.ok = True
        return result
    except OSError as error:  # a full disk or an unwritable directory is a per-clip failure
        result.error = repr(error)
        return result
    finally:
        result.seconds = time.perf_counter() - started
        if not config.keep_work_dir:
            shutil.rmtree(Path(work_dir) / job.name, ignore_errors=True)


def _write_caption(config: RenderConfig, directory: Path, job: RenderJob) -> Path | None:
    """The caption text file ``drawtext`` reads, or ``None`` when captions are off / the font is gone."""

    if not config.caption or not job.caption:
        return None
    if not Path(config.font).is_file():
        return None
    path = directory / "caption.txt"
    path.write_text(job.caption)
    return path


def render_many(
    config: RenderConfig,
    jobs: Iterable[RenderJob],
    *,
    work_dir: str | Path,
    runner: Runner | None = None,
    playback: PlaybackRunner | None = None,
    workers: int | None = None,
) -> list[RenderResult]:
    """:func:`render_replay` over ``jobs`` (in order), up to ``workers`` replays at a time.

    Concurrency is close to free here: eight renders were measured to take the wall clock of one
    (ENV.md 6.5), because each one spends most of its time in a single Dolphin thread.
    """

    items = list(jobs)
    if not items:
        return []
    count = config.workers if workers is None else workers
    if count <= 1 or len(items) == 1:
        return [
            render_replay(config, job, work_dir=work_dir, runner=runner, playback=playback) for job in items
        ]
    with ThreadPoolExecutor(max_workers=min(count, len(items)), thread_name_prefix="melee-rl-render") as pool:
        futures = [
            pool.submit(render_replay, config, job, work_dir=work_dir, runner=runner, playback=playback)
            for job in items
        ]
        return [future.result() for future in futures]


__all__ = [
    "AUDIO_DUMP_SUBDIR",
    "COMM_MODE",
    "DEFAULT_FONT",
    "DUMP_SUBDIR",
    "DUMP_SUFFIXES",
    "EFB_SCALE",
    "PLAYBACK_MARKERS",
    "RENDER_CANDIDATES",
    "STOP_GRACE_S",
    "XVFB_ARGS",
    "CommandResult",
    "PlaybackProgress",
    "PlaybackRunner",
    "RenderCandidate",
    "RenderConfig",
    "RenderJob",
    "RenderResult",
    "Runner",
    "audio_dump_files",
    "candidate_config",
    "comm_payload",
    "drawtext_filter",
    "dump_files",
    "escape_filter_value",
    "ffmpeg_command",
    "ffprobe_command",
    "parse_ffprobe",
    "parse_playback_line",
    "render_command",
    "render_environment",
    "render_ini",
    "render_many",
    "render_replay",
    "render_timeout",
    "run_command",
    "run_playback",
    "write_user_dir",
]
