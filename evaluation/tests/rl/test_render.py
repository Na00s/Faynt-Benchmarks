"""Stage 2 of P9: rendering a Slippi ``.slp`` to mp4 (``melee_rl.render``).

Everything here is offline: the ini / comm / argv builders are pure, the Dolphin and ffmpeg calls go
through an injectable ``runner``, and the one end-to-end test uses a synthetic two-second video that
ffmpeg generates locally (skipped where ffmpeg is missing).
"""

from __future__ import annotations

import configparser
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from melee_rl.render import (
    AUDIO_DUMP_SUBDIR,
    COMM_MODE,
    DUMP_SUBDIR,
    EFB_SCALE,
    RENDER_CANDIDATES,
    CommandResult,
    PlaybackProgress,
    RenderCandidate,
    RenderConfig,
    RenderJob,
    RenderResult,
    audio_dump_files,
    candidate_config,
    comm_payload,
    dump_files,
    escape_filter_value,
    ffmpeg_command,
    ffprobe_command,
    parse_ffprobe,
    parse_playback_line,
    render_command,
    render_environment,
    render_ini,
    render_many,
    render_replay,
    render_timeout,
    run_command,
    run_playback,
    write_user_dir,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg / ffprobe not installed")


class StubRunner:
    """Records commands and replays scripted results; the dolphin call may create the dump file.

    ``playback`` satisfies :class:`melee_rl.render.PlaybackRunner` -- the Dolphin call goes through it
    now that the render is stopped by the ``--cout`` frame markers rather than by a timeout.
    """

    def __init__(
        self,
        *,
        dump: Path | None = None,
        dolphin_rc: int = 0,
        ffmpeg_rc: int = 0,
        progress: PlaybackProgress | None = None,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.environments: list[Mapping[str, str] | None] = []
        self.dump = dump
        self.dolphin_rc = dolphin_rc
        self.ffmpeg_rc = ffmpeg_rc
        self.progress = progress

    def playback(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
        stop_grace_s: float = 10.0,
    ) -> tuple[CommandResult, PlaybackProgress]:
        result = self(command, timeout_s=timeout_s, env=env)
        if self.progress is not None:
            return result, self.progress
        progress = PlaybackProgress(
            start_frame=-123, end_frame=476, game_end_frame=476, last_frame=476, frames_rendered=600
        )
        progress.startup_s, progress.render_s, progress.reason = 12.0, 30.0, "end-frame"
        return result, progress

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> CommandResult:
        argv = tuple(str(part) for part in command)
        self.commands.append(argv)
        self.environments.append(env)
        name = Path(argv[0]).name
        if name == "ffprobe":
            payload = json.dumps(
                {
                    "streams": [{"width": 640, "height": 480, "nb_frames": "120", "codec_name": "h264"}],
                    "format": {"duration": "2.000000"},
                }
            )
            return CommandResult(argv, 0, payload, "", 0.1)
        if name == "ffmpeg":
            if self.ffmpeg_rc == 0:
                Path(argv[-1]).write_bytes(b"mp4" * 100)
            return CommandResult(argv, self.ffmpeg_rc, "", "", 0.1)
        if self.dolphin_rc == 0 and self.dump is not None:
            self.dump.parent.mkdir(parents=True, exist_ok=True)
            self.dump.write_bytes(b"avi" * 100)
        return CommandResult(argv, self.dolphin_rc, "boot\n", "", 1.0)


def _config(**overrides: object) -> RenderConfig:
    settings: dict[str, object] = {"dolphin_path": "/opt/dolphin-emu", "iso_path": "/iso/melee.iso"}
    settings.update(overrides)
    return RenderConfig(**settings)  # type: ignore[arg-type]


def test_render_config_defaults_and_validation() -> None:
    config = RenderConfig()
    assert config.backend == "OGL" and config.platform == "headless" and config.xvfb is False
    assert config.internal_resolution == 1 and config.bitrate_kbps == 3000  # 1x native 640x480 (P9 §3)
    assert config.dump_format == "avi" and config.emulation_speed == 0.0
    assert config.ffmpeg == "ffmpeg" and config.ffprobe == "ffprobe" and config.caption is True
    assert config.font.endswith(".ttf") and config.timeout_s > 0.0 and config.workers == 1
    assert config.speed_factor == 20.0 and config.min_timeout_s == 120.0
    # P9b: the render is stopped by the --cout frame markers, not by the timeout.
    assert config.cout is True and config.hide_seekbar is True and config.stop_grace_s == 10.0
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"internal_resolution": 0}, "internal_resolution"),
        ({"bitrate_kbps": 0}, "bitrate_kbps"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"workers": 0}, "workers"),
        ({"crf": -1}, "crf"),
        ({"emulation_speed": -1.0}, "emulation_speed"),
        ({"dump_format": ""}, "dump_format"),
        ({"backend": ""}, "backend"),
        ({"speed_factor": 0.0}, "speed_factor"),
        ({"min_timeout_s": 0.0}, "min_timeout_s"),
        ({"internal_resolution": 7}, "internal_resolution"),  # no EFBScale for 7x
        ({"stop_grace_s": -1.0}, "stop_grace_s"),
    )
    for bad, match in cases:
        with pytest.raises(ValueError, match=match):
            RenderConfig(**bad)


def test_render_ini_and_user_directory(tmp_path: Path) -> None:
    config = _config(backend="Vulkan", internal_resolution=2, bitrate_kbps=6000, dump_format="mp4")
    files = render_ini(config)
    assert set(files) == {"Dolphin.ini", "GFX.ini", "Logger.ini"}
    dolphin = configparser.ConfigParser()
    dolphin.read_string(files["Dolphin.ini"])
    assert dolphin["Core"]["gfxbackend"] == "Vulkan"
    assert dolphin["Core"]["emulationspeed"] == "0.0"
    assert dolphin["Movie"]["dumpframes"] == "True" and dolphin["Movie"]["dumpframessilent"] == "True"
    assert dolphin["DSP"]["backend"] == "No Audio Output"  # muted output; the sound comes from the dump
    # 8 Sep 2026: the game audio is dumped (DSP sound effects + the DTK music stream) and mixed into the clip.
    assert dolphin["DSP"]["dumpaudio"] == "True" and dolphin["DSP"]["dumpaudiosilent"] == "True"
    silent = configparser.ConfigParser()
    silent.read_string(render_ini(_config(audio=False))["Dolphin.ini"])
    assert silent["DSP"]["dumpaudio"] == "False"
    assert dolphin["Display"]["fullscreen"] == "False"
    gfx = configparser.ConfigParser()
    gfx.read_string(files["GFX.ini"])
    assert gfx["Settings"]["internalresolutionframedumps"] == "True"
    # P9b: Ishiiruka's internal-resolution knob is EFBScale (an enum), and it has no
    # "InternalResolution" key at all -- writing one left the Linux default SCALE_2X in force, so every
    # render rasterised 1280x960 (VideoConfig.cpp:141-145, VideoConfig.h:42-54 @ v3.5.2).
    assert gfx["Settings"]["efbscale"] == str(EFB_SCALE[2]) == "4"
    assert "internalresolution" not in gfx["Settings"]
    assert render_ini(_config())["GFX.ini"].count("EFBScale = 2") == 1  # the default is 1x native
    assert EFB_SCALE[1] == 2 and EFB_SCALE[2] == 4 and EFB_SCALE[3] == 6  # SCALE_1X / SCALE_2X / SCALE_3X
    assert gfx["Settings"]["bitratekbps"] == "6000" and gfx["Settings"]["dumpformat"] == "mp4"
    assert gfx["Settings"]["msaa"] == "1" and gfx["Settings"]["ssaa"] == "False"
    assert "dumppath" not in gfx["Settings"]  # the dump lands in <user>/Dump/Frames, which we own
    # Nothing may overlay the frame dump: no OSD, no toolbar, no seekbar, and a pinned 640x480 window
    # so a non-internal-resolution dump cannot inherit the 1280x1024 Xvfb screen.
    assert dolphin["Interface"]["onscreendisplaymessages"] == "False"
    assert dolphin["Interface"]["showtoolbar"] == "False" and dolphin["Interface"]["showseekbar"] == "False"
    display = dolphin["Display"]
    assert display["renderwindowwidth"] == "640" and display["renderwindowheight"] == "480"
    assert dolphin["Display"]["renderwindowautosize"] == "False"
    user = write_user_dir(tmp_path / "render-user", config)
    assert (user / "Config" / "Dolphin.ini").is_file() and (user / "Config" / "GFX.ini").is_file()
    assert (user / "Config" / "Logger.ini").is_file()
    assert (user / DUMP_SUBDIR).is_dir() and DUMP_SUBDIR == "Dump/Frames"
    assert dump_files(user) == []
    (user / DUMP_SUBDIR / "framedump0.avi").write_bytes(b"x")
    (user / DUMP_SUBDIR / "notes.txt").write_text("ignored")
    assert [path.name for path in dump_files(user)] == ["framedump0.avi"]
    assert (user / AUDIO_DUMP_SUBDIR).is_dir() and AUDIO_DUMP_SUBDIR == "Dump/Audio"
    assert audio_dump_files(user) == []
    (user / AUDIO_DUMP_SUBDIR / "dtkdump.wav").write_bytes(b"RIFF" + b"x" * 100)
    (user / AUDIO_DUMP_SUBDIR / "dspdump.wav").write_bytes(b"RIFF" + b"x" * 100)
    (user / AUDIO_DUMP_SUBDIR / "empty.wav").write_bytes(b"")
    assert [path.name for path in audio_dump_files(user)] == ["dspdump.wav", "dtkdump.wav"]  # a fixed order
    assert write_user_dir(tmp_path / "render-user", config) == user  # idempotent


def test_comm_payload_and_render_command(tmp_path: Path) -> None:
    slp = tmp_path / "Game_20260826T120000.slp"
    payload = comm_payload(slp, end_frame=7200, command_id="clip-1")
    assert payload["mode"] == COMM_MODE == "normal"
    assert payload["replay"] == str(slp) and payload["commandId"] == "clip-1"
    assert payload["isRealTimeMode"] is False and payload["startFrame"] == -123
    assert payload["endFrame"] == 7200 and payload["outputOverlayFiles"] is False
    assert "endFrame" not in comm_payload(slp, end_frame=None)  # Dolphin's INT_MAX default
    config = _config()
    command = render_command(config, user_dir=tmp_path / "user", comm_path=tmp_path / "comm.json")
    assert command == [
        "/opt/dolphin-emu",
        "-b",
        "-e",
        "/iso/melee.iso",
        "-u",
        str(tmp_path / "user"),
        "-i",
        str(tmp_path / "comm.json"),
        "--cout",
        "--hide-seekbar",
        "--platform",
        "headless",
    ]
    # Both flags are real on the pinned playback build (Ishiiruka Main.cpp:320-323 @ v3.5.2) but a build
    # without them must stay drivable, so each is a switch.
    plain = render_command(_config(cout=False, hide_seekbar=False), user_dir=tmp_path, comm_path=tmp_path)
    assert "--cout" not in plain and "--hide-seekbar" not in plain
    assert render_environment(config) is None
    with_display = _config(platform=None, display=":99")
    no_platform = render_command(with_display, user_dir=tmp_path, comm_path=tmp_path / "c.json")
    assert no_platform[no_platform.index("-i") + 1] == str(tmp_path / "c.json")
    assert "--platform" not in no_platform
    assert render_environment(with_display) == {"DISPLAY": ":99"}
    xvfb = render_command(_config(platform=None, xvfb=True), user_dir=tmp_path, comm_path=tmp_path / "c")
    assert xvfb[:4] == ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"]
    assert xvfb[4] == "/opt/dolphin-emu" and "--platform" not in xvfb


def test_candidate_ladder() -> None:
    """The probe walks headless x {OGL, Vulkan, Software Renderer} then Xvfb x OGL (P9 §6)."""

    assert [item.name for item in RENDER_CANDIDATES] == [
        "headless-ogl",
        "headless-vulkan",
        "headless-software",
        "xvfb-ogl",
    ]
    assert [item.backend for item in RENDER_CANDIDATES] == ["OGL", "Vulkan", "Software Renderer", "OGL"]
    assert [item.xvfb for item in RENDER_CANDIDATES] == [False, False, False, True]
    base = _config()
    first = candidate_config(base, RENDER_CANDIDATES[0])
    assert first.backend == "OGL" and first.platform == "headless" and not first.xvfb
    last = candidate_config(base, RENDER_CANDIDATES[-1])
    assert last.platform is None and last.xvfb and last.backend == "OGL"
    assert last.dolphin_path == base.dolphin_path  # only the video knobs move
    custom = candidate_config(base, RenderCandidate("x", backend="OGL", platform=None, display=":7"))
    assert custom.display == ":7"


def test_ffmpeg_commands_and_escaping(tmp_path: Path) -> None:
    config = _config(crf=18)
    source, target = tmp_path / "framedump0.avi", tmp_path / "clip.mp4"
    caption = tmp_path / "caption.txt"
    plain = ffmpeg_command(config, source=source, target=target)
    assert plain[0] == "ffmpeg" and plain[1] == "-y" and plain[-1] == str(target)
    assert "-i" in plain and str(source) in plain and "-an" in plain
    assert "-crf" in plain and plain[plain.index("-crf") + 1] == "18"
    assert plain[plain.index("-pix_fmt") + 1] == "yuv420p"
    assert "-vf" not in plain
    with_caption = ffmpeg_command(config, source=source, target=target, caption_file=caption)
    video_filter = with_caption[with_caption.index("-vf") + 1]
    assert video_filter.startswith("drawtext=") and f"textfile={caption}" in video_filter
    assert "fontfile=" in video_filter and "box=1" in video_filter
    trimmed = ffmpeg_command(config, source=source, target=target, trim_seconds=30.0, bitrate_kbps=1200)
    assert trimmed[trimmed.index("-t") + 1] == "30"
    assert trimmed[trimmed.index("-b:v") + 1] == "1200k" and "-crf" not in trimmed
    # 8 Sep 2026: audio dumps are extra inputs mixed into one AAC track; the caption joins the same graph.
    dsp, dtk = tmp_path / "dspdump.wav", tmp_path / "dtkdump.wav"
    with_audio = ffmpeg_command(
        config, source=source, target=target, audio_files=[dsp, dtk], trim_seconds=30.0
    )
    assert (
        "-an" not in with_audio
        and with_audio.count("-i") == 3
        and str(dsp) in with_audio
        and str(dtk) in with_audio
    )
    graph = with_audio[with_audio.index("-filter_complex") + 1]
    assert "[1:a][2:a]amix=inputs=2" in graph and "[a]" in graph and "-vf" not in with_audio
    assert with_audio[with_audio.index("-c:a") + 1] == "aac" and "-map" in with_audio
    assert with_audio[with_audio.index("-t") + 1] == "30"
    one_track = ffmpeg_command(config, source=source, target=target, audio_files=[dsp], caption_file=caption)
    graph = one_track[one_track.index("-filter_complex") + 1]
    assert (
        "amix" not in graph and "[1:a]" in graph and "drawtext=" in graph and f"textfile={caption}" in graph
    )
    assert ffmpeg_command(config, source=source, target=target, audio_files=[]) == plain
    assert escape_filter_value("a:b") == r"a\:b"
    assert escape_filter_value("c,d[e]") == r"c\,d\[e\]"
    assert escape_filter_value(r"back\slash") == "back\\\\slash"
    assert escape_filter_value("100%") == r"100\%"
    assert escape_filter_value("it's") == r"it\'s"
    assert ffprobe_command(config, target)[-1] == str(target)
    assert "-show_entries" in ffprobe_command(config, target)


def test_render_timeout_scales_with_the_replay(tmp_path: Path) -> None:
    """A playback Dolphin does not exit when the replay ends, so the call is bounded by its length."""

    config = _config(speed_factor=20.0, min_timeout_s=120.0, timeout_s=1800.0)
    slp = tmp_path / "clip.slp"
    short = RenderJob(slp=slp, target=tmp_path / "a.mp4", frames=1200)  # 20 s of replay
    assert short.seconds == 20.0 and render_timeout(config, short) == 400.0
    tiny = RenderJob(slp=slp, target=tmp_path / "b.mp4", frames=60)
    assert render_timeout(config, tiny) == 120.0  # the floor
    long_clip = RenderJob(slp=slp, target=tmp_path / "c.mp4", frames=7200)  # 2 minutes
    assert render_timeout(config, long_clip) == 1800.0  # capped by timeout_s
    assert RenderJob(slp=slp, target=tmp_path / "d.mp4").seconds == 0.0


def test_render_replay_keeps_a_dump_from_an_unfinished_render(tmp_path: Path) -> None:
    """The safety timeout still fires if no marker ever arrives -- the partial dump stays the deliverable."""

    slp = tmp_path / "clip.slp"
    slp.write_bytes(b"slp")
    work = tmp_path / "work"
    dump = work / "clip" / "user" / DUMP_SUBDIR / "framedump0.avi"
    stalled = PlaybackProgress()
    stalled.reason = "timeout"
    runner = StubRunner(dump=dump, progress=stalled)
    job = RenderJob(slp=slp, target=tmp_path / "clip.mp4", caption="c", frames=1200)
    result = render_replay(
        _config(keep_work_dir=True), job, work_dir=work, runner=runner, playback=runner.playback
    )
    assert result.ok and result.killed and not result.finished and result.stop_reason == "timeout"
    assert result.video == str(tmp_path / "clip.mp4") and result.dump == str(dump)
    assert result.render_fps is None and result.frames_rendered == 0
    mux = next(command for command in result.commands if Path(command["command"][0]).name == "ffmpeg")
    argv = mux["command"]
    assert argv[argv.index("-t") + 1] == "20"  # trimmed to the replay's own length
    assert result.record()["killed"] is True and result.record()["finished"] is False


def test_parse_ffprobe() -> None:
    payload = json.dumps(
        {
            "streams": [{"width": 640, "height": 480, "nb_frames": "7200", "codec_name": "h264"}],
            "format": {"duration": "120.0"},
        }
    )
    info = parse_ffprobe(payload)
    assert info == {"duration_s": 120.0, "width": 640, "height": 480, "frames": 7200, "codec": "h264"}
    assert parse_ffprobe(json.dumps({"streams": [], "format": {}})) == {
        "duration_s": None,
        "width": None,
        "height": None,
        "frames": None,
        "codec": None,
    }
    assert parse_ffprobe("not json")["duration_s"] is None


def test_render_replay_happy_path(tmp_path: Path) -> None:
    slp = tmp_path / "clip.slp"
    slp.write_bytes(b"slp")
    target = tmp_path / "out" / "policy_vs_cpu9_1.mp4"
    work = tmp_path / "work"
    runner = StubRunner(dump=work / "policy_vs_cpu9_1" / "user" / DUMP_SUBDIR / "framedump0.avi")
    job = RenderJob(slp=slp, target=target, caption="20m_bc step 7 (P1) vs CPU 9 (P2) - clip 1/2", frames=600)
    result = render_replay(
        _config(keep_work_dir=True), job, work_dir=work, runner=runner, playback=runner.playback
    )
    assert isinstance(result, RenderResult) and result.ok and result.error is None
    assert result.video == str(target) and target.is_file()
    assert result.duration_s == 2.0 and result.width == 640 and result.height == 480
    assert result.seconds >= 0.0 and result.returncode == 0
    # P9b: the render reports its own split, so the rate is measured rather than bracketed.
    assert result.startup_s == 12.0 and result.render_s == 30.0 and result.frames_rendered == 600
    assert result.render_fps is not None and result.render_fps > 0.0
    assert result.finished and result.stop_reason == "end-frame"
    assert result.expected_frames == 600 and result.slowdown is not None
    names = [Path(command[0]).name for command in runner.commands]
    assert names == ["dolphin-emu", "ffmpeg", "ffprobe"]
    comm = json.loads((work / "policy_vs_cpu9_1" / "comm.json").read_text())
    assert comm["replay"] == str(slp) and comm["endFrame"] == 600 - 123 - 1
    caption_file = work / "policy_vs_cpu9_1" / "caption.txt"
    assert caption_file.read_text() == job.caption
    assert runner.environments[0] is None
    # With the default the whole job directory (the frame dump included) is removed after the mux.
    again = render_replay(_config(), job, work_dir=work, runner=StubRunner(dump=runner.dump))
    assert again.ok and not (work / "policy_vs_cpu9_1").exists()


def test_render_replay_mixes_the_audio_dumps_when_dolphin_left_them(tmp_path: Path) -> None:
    """8 Sep 2026: wav dumps in <user>/Dump/Audio reach the ffmpeg mux; without them the clip stays silent."""

    slp = tmp_path / "clip.slp"
    slp.write_bytes(b"slp")
    target = tmp_path / "out" / "policy_vs_mimic_1.mp4"
    work = tmp_path / "work"
    user = work / "policy_vs_mimic_1" / "user"
    (user / AUDIO_DUMP_SUBDIR).mkdir(parents=True)
    (user / AUDIO_DUMP_SUBDIR / "dspdump.wav").write_bytes(b"RIFF" + b"x" * 100)
    (user / AUDIO_DUMP_SUBDIR / "dtkdump.wav").write_bytes(b"RIFF" + b"x" * 100)
    runner = StubRunner(dump=user / DUMP_SUBDIR / "framedump0.avi")
    job = RenderJob(slp=slp, target=target, caption="c", frames=600)
    result = render_replay(
        _config(keep_work_dir=True), job, work_dir=work, runner=runner, playback=runner.playback
    )
    assert result.ok and result.audio_tracks == 2
    ffmpeg = next(command for command in runner.commands if Path(command[0]).name == "ffmpeg")
    assert "-an" not in ffmpeg and "-filter_complex" in ffmpeg and ffmpeg.count("-i") == 3
    silent = render_replay(
        _config(keep_work_dir=True, audio=False), job, work_dir=work, runner=runner, playback=runner.playback
    )
    assert silent.ok and silent.audio_tracks == 0
    ffmpeg = next(command for command in runner.commands[3:] if Path(command[0]).name == "ffmpeg")
    assert "-an" in ffmpeg


def test_render_replay_failures(tmp_path: Path) -> None:
    slp = tmp_path / "clip.slp"
    slp.write_bytes(b"slp")
    job = RenderJob(slp=slp, target=tmp_path / "out.mp4", caption="c", frames=600)
    broken = StubRunner(dolphin_rc=3)
    failed = render_replay(_config(), job, work_dir=tmp_path / "w1", runner=broken, playback=broken.playback)
    assert not failed.ok and failed.returncode == 3 and failed.video is None
    assert failed.error is not None and "dolphin" in failed.error.lower()
    blank = StubRunner(dump=None)
    empty = render_replay(_config(), job, work_dir=tmp_path / "w2", runner=blank, playback=blank.playback)
    assert not empty.ok and empty.error is not None and "dump" in empty.error.lower()
    assert "3" in (empty.error or "") or empty.returncode == 0  # the exit code is in the message
    dump = tmp_path / "w3" / "out" / "user" / DUMP_SUBDIR / "framedump0.avi"
    muxfail = StubRunner(dump=dump, ffmpeg_rc=1)
    bad_ffmpeg = render_replay(
        _config(), job, work_dir=tmp_path / "w3", runner=muxfail, playback=muxfail.playback
    )
    assert not bad_ffmpeg.ok and bad_ffmpeg.error is not None and "ffmpeg" in bad_ffmpeg.error.lower()
    missing = render_replay(_config(), replace(job, slp=tmp_path / "nope.slp"), work_dir=tmp_path / "w4")
    assert not missing.ok and missing.error is not None and "does not exist" in missing.error


def test_render_many_over_several_jobs(tmp_path: Path) -> None:
    jobs = []
    for index in (1, 2):
        slp = tmp_path / f"clip{index}.slp"
        slp.write_bytes(b"slp")
        jobs.append(RenderJob(slp=slp, target=tmp_path / f"clip{index}.mp4", caption=f"c{index}", frames=300))
    work = tmp_path / "work"
    runner = StubRunner(dump=work / "clip1" / "user" / DUMP_SUBDIR / "framedump0.avi")
    results = render_many(_config(), jobs, work_dir=work, runner=runner, playback=runner.playback)
    assert len(results) == 2 and results[0].ok and not results[1].ok  # only clip1's dump was scripted
    assert set(results[0].record()) == {
        "slp",
        "video",
        "ok",
        "returncode",
        "seconds",
        "duration_s",
        "width",
        "height",
        "frames",
        "expected_frames",
        "codec",
        "dump",
        "killed",
        "error",
        "startup_s",
        "render_s",
        "render_fps",
        "slowdown",
        "finished",
        "stop_reason",
        "frames_rendered",
    }
    assert [Path(result.slp).name for result in results] == ["clip1.slp", "clip2.slp"]
    assert render_many(_config(), [], work_dir=work, runner=runner) == []


def test_run_command_captures_output_and_timeouts(tmp_path: Path) -> None:
    ok = run_command(["python3", "-c", "print('hi')"], timeout_s=60.0)
    assert ok.ok and ok.returncode == 0 and ok.stdout.strip() == "hi" and ok.error is None
    bad = run_command(["python3", "-c", "raise SystemExit(4)"], timeout_s=60.0)
    assert not bad.ok and bad.returncode == 4
    slow = run_command(["python3", "-c", "import time; time.sleep(5)"], timeout_s=0.2)
    assert not slow.ok and slow.returncode is None and slow.error is not None
    missing = run_command([str(tmp_path / "nothing")], timeout_s=1.0)
    assert not missing.ok and missing.error is not None


@needs_ffmpeg
def test_ffmpeg_burns_the_caption_into_a_real_video(tmp_path: Path) -> None:
    """The post-processing chain end to end on a synthetic two-second source."""

    source = tmp_path / "src.avi"
    generate = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x480:rate=60:duration=2",
        str(source),
    ]
    assert run_command(generate, timeout_s=120.0).ok
    config = _config()
    caption = tmp_path / "caption.txt"
    caption.write_text("20m_bc step 1234 (P1) vs CPU 9 (P2) - clip 1/2")
    target = tmp_path / "clip.mp4"
    assert run_command(
        ffmpeg_command(config, source=source, target=target, caption_file=caption), timeout_s=180.0
    ).ok
    probe = run_command(ffprobe_command(config, target), timeout_s=60.0)
    assert probe.ok
    info = parse_ffprobe(probe.stdout)
    assert info["width"] == 640 and info["height"] == 480
    assert info["duration_s"] is not None and 1.5 <= float(info["duration_s"]) <= 2.5
    assert target.stat().st_size > 1000


def test_run_command_timeout_kills_grandchildren(tmp_path: Path) -> None:
    """A hung Dolphin must not hang the job (26 Aug 2026: ``xvfb-run`` execs Dolphin as a child, and a
    plain kill leaves the grandchild holding the stdout pipe, so the read after the timeout never
    returns -- the probe sat there until Modal killed the container)."""

    script = tmp_path / "spawn.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n"
    )
    started = time.perf_counter()
    result = run_command(["python3", str(script)], timeout_s=1.0)
    assert time.perf_counter() - started < 20.0  # not 120: the whole group is killed
    assert not result.ok and result.returncode is None and result.error is not None


# ---------------------------------------------------------------------------
# P9b: completion detection from the playback build's --cout frame markers
# ---------------------------------------------------------------------------


def test_parse_playback_line() -> None:
    """The five markers the playback build prints (EXI_DeviceSlippi.cpp:1057, :1300-1303)."""

    assert parse_playback_line("[CURRENT_FRAME] 42") == ("CURRENT_FRAME", 42)
    assert parse_playback_line("[PLAYBACK_START_FRAME] -123") == ("PLAYBACK_START_FRAME", -123)
    assert parse_playback_line("[PLAYBACK_END_FRAME] 7076") == ("PLAYBACK_END_FRAME", 7076)
    assert parse_playback_line("[GAME_END_FRAME] 7076\n") == ("GAME_END_FRAME", 7076)
    assert parse_playback_line("[NO_GAME]") == ("NO_GAME", None)
    assert parse_playback_line("  [CURRENT_FRAME] 7 \r\n") == ("CURRENT_FRAME", 7)
    assert parse_playback_line("noise [CURRENT_FRAME] 8") == ("CURRENT_FRAME", 8)  # interleaved output
    for junk in ("", "\n", "booting", "[FILE_PATH] /runs/x.slp", "[LRAS]", "[CURRENT_FRAME]"):
        assert parse_playback_line(junk) is None


def test_playback_progress_folds_markers_and_measures_the_rate() -> None:
    progress = PlaybackProgress()
    assert progress.target_frame is None and not progress.complete and progress.fps is None
    assert progress.observe("PLAYBACK_END_FRAME", 100, 1.0) is None
    assert progress.observe("GAME_END_FRAME", 60, 1.0) is None
    assert progress.target_frame == 60  # the shorter of the two bounds wins
    assert progress.observe("CURRENT_FRAME", -123, 12.0) is None
    assert progress.startup_s == 12.0 and progress.frames_rendered == 1
    assert progress.observe("CURRENT_FRAME", 59, 22.0) is None
    assert progress.fps is not None and abs(progress.fps - 0.1) < 1e-9  # 1 interval over 10 s
    assert progress.observe("CURRENT_FRAME", 60, 24.0) == "end-frame"
    assert progress.complete and progress.last_frame == 60 and progress.render_s == 12.0
    # [NO_GAME] only ends the render once a game has actually started (it also fires at boot).
    boot = PlaybackProgress()
    assert boot.observe("NO_GAME", None, 0.5) is None
    boot.observe("CURRENT_FRAME", 1, 5.0)
    assert boot.observe("NO_GAME", None, 9.0) == "no-game"
    record = progress.record()
    assert record["reason"] == "running" and record["frames_rendered"] == 3  # reason is set by the runner
    assert record["target_frame"] == 60 and record["startup_s"] == 12.0


def _fake_dolphin(tmp_path: Path, body: str, name: str = "fake_dolphin.py") -> Path:
    script = tmp_path / name
    script.write_text("import sys, time\n" + body)
    return script


def test_run_playback_stops_at_the_end_frame(tmp_path: Path) -> None:
    """A playback Dolphin never exits by itself, so the frame markers are what ends the call."""

    script = _fake_dolphin(
        tmp_path,
        "print('[PLAYBACK_START_FRAME] -123', flush=True)\n"
        "print('[GAME_END_FRAME] 5', flush=True)\n"
        "print('[PLAYBACK_END_FRAME] 5', flush=True)\n"
        "for i in range(-123, 6):\n"
        "    print('[CURRENT_FRAME] %d' % i, flush=True)\n"
        "time.sleep(120)\n",  # like the real one: it sits there forever
    )
    started = time.perf_counter()
    result, progress = run_playback(["python3", str(script)], timeout_s=60.0, stop_grace_s=5.0)
    assert time.perf_counter() - started < 30.0  # not 120: the marker ended it
    assert progress.reason == "end-frame" and progress.finished
    assert progress.last_frame == 5 and progress.frames_rendered == 129
    assert progress.start_frame == -123 and progress.end_frame == 5 and progress.target_frame == 5
    assert progress.startup_s is not None and progress.render_s is not None
    assert result.error is None and "[CURRENT_FRAME] 5" in result.stdout


def test_run_playback_uses_the_shorter_of_the_two_end_frames(tmp_path: Path) -> None:
    """Our comm file may over-ask (``endFrame`` from the recording); the replay's own end wins."""

    script = _fake_dolphin(
        tmp_path,
        "print('[GAME_END_FRAME] 3', flush=True)\n"
        "print('[PLAYBACK_END_FRAME] 999999', flush=True)\n"
        "for i in range(0, 4):\n"
        "    print('[CURRENT_FRAME] %d' % i, flush=True)\n"
        "time.sleep(120)\n",
    )
    result, progress = run_playback(["python3", str(script)], timeout_s=60.0, stop_grace_s=5.0)
    assert progress.target_frame == 3 and progress.last_frame == 3
    assert progress.reason == "end-frame" and progress.finished and result.returncode is not None


def test_run_playback_reports_a_clean_exit_and_a_timeout(tmp_path: Path) -> None:
    exits = _fake_dolphin(
        tmp_path,
        "print('[NO_GAME]', flush=True)\nprint('[CURRENT_FRAME] 1', flush=True)\n",
        name="exits.py",
    )
    _, progress = run_playback(["python3", str(exits)], timeout_s=60.0, stop_grace_s=5.0)
    assert progress.reason == "exit" and progress.finished and progress.frames_rendered == 1
    quiet = _fake_dolphin(tmp_path, "time.sleep(120)\n", name="quiet.py")
    started = time.perf_counter()
    _, stalled = run_playback(["python3", str(quiet)], timeout_s=1.0, stop_grace_s=3.0)
    assert time.perf_counter() - started < 30.0
    assert stalled.reason == "timeout" and not stalled.finished and stalled.frames_rendered == 0
    missing, absent = run_playback([str(tmp_path / "nothing")], timeout_s=5.0)
    assert missing.error is not None and absent.reason == "error"


def test_run_playback_kills_the_whole_process_group(tmp_path: Path) -> None:
    """``xvfb-run`` execs Dolphin as a child; a plain kill leaves the grandchild on the stdout pipe."""

    script = _fake_dolphin(
        tmp_path,
        "import subprocess\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print('[GAME_END_FRAME] 1', flush=True)\n"
        "print('[CURRENT_FRAME] 1', flush=True)\n"
        "time.sleep(120)\n",
        name="grandchild.py",
    )
    started = time.perf_counter()
    _, progress = run_playback(["python3", str(script)], timeout_s=60.0, stop_grace_s=3.0)
    assert time.perf_counter() - started < 30.0 and progress.reason == "end-frame"
