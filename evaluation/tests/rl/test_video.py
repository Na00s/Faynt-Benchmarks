"""Stage 1 of P9: recording clips for a checkpoint (``melee_rl.video``).

Matchup expansion, the sha256-keyed baseline directory, replay collection, render jobs and the whole
:func:`record_matches` chain on the dummy env with a stub renderer (the toy writes no ``.slp``, so the
test seeds the replay directories the recorder is going to look in).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from melee_rl import slp
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import save_bc_checkpoint, sha256_file
from melee_rl.config import CONFIG_DIR, RLConfig, finalize_config, load_config, resolve_model_config
from melee_rl.render import RenderConfig, RenderJob, RenderResult
from melee_rl.video import (
    CLIP_STATS,
    FRAMES_PER_SECOND,
    MATCHUP_KINDS,
    VIDEO_MODES,
    ClipRecord,
    MatchSpec,
    VideoConfig,
    VideoRun,
    baseline_dir,
    caption_for,
    collect_replays,
    expand_matches,
    format_summary,
    games_started,
    match_record,
    parse_matchup,
    record_matches,
    render_jobs,
    wandb_upload,
    write_clip_records,
    write_manifest,
)

SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
STAMP = "20260826-120000"


class StubRenderer:
    """Stands in for ``render.render_many``: records the jobs and reports every clip as rendered."""

    def __init__(self, *, ok: bool = True) -> None:
        self.jobs: list[RenderJob] = []
        self.ok = ok

    def __call__(self, jobs: Sequence[RenderJob]) -> list[RenderResult]:
        self.jobs.extend(jobs)
        results = []
        for job in jobs:
            if self.ok:
                job.target.parent.mkdir(parents=True, exist_ok=True)
                job.target.write_bytes(b"mp4" * 100)
            results.append(
                RenderResult(
                    slp=str(job.slp),
                    target=str(job.target),
                    ok=self.ok,
                    video=str(job.target) if self.ok else None,
                    duration_s=job.frames / FRAMES_PER_SECOND if self.ok else None,
                    width=640,
                    height=480,
                    expected_frames=job.frames,
                    startup_s=11.0,
                    render_s=20.0,
                    render_fps=57.5,
                    frames_rendered=job.frames,
                    slowdown=1.4,
                    finished=self.ok,
                    stop_reason="end-frame" if self.ok else "timeout",
                    error=None if self.ok else "no backend",
                )
            )
        return results


def _video(**overrides: object) -> VideoConfig:
    settings: dict[str, object] = {"clips": 2, "seconds": 1.0, "rollout_length": 8}
    settings.update(overrides)
    return VideoConfig(**settings)  # type: ignore[arg-type]


def _config(video: VideoConfig) -> RLConfig:
    base = load_config(SMOKE_TINY)
    return finalize_config(replace(base, video=video), resolve_model_config(base))


def test_recorder_plays_a_checkpoint_at_the_delay_it_was_trained_at(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: Any,
    tiny_value_config: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """11 Sep 2026 (/delay-finetune): the recorder used to check ``actor.delay_frames`` against the model's
    ``action_offset_frames - 1`` -- always 0 -- and would have recorded a D = 18 checkpoint at 0 through the
    stock run file.  Now the file's trained delay is the default; a run file that names another delay needs
    ``[delay] allow_mismatch`` (then its value wins, 0 included); the manifest says which delay was played;
    the ``bc`` rival plays at its own file's delay unless ``[video] bc_delay_frames`` says otherwise."""

    from melee_rl import video as video_module
    from melee_rl.checkpoint import save_rl_checkpoint
    from melee_rl.learner import LearnerConfig, PPOLearner
    from melee_rl.opponents import OtherOpponent
    from melee_rl.value import ValueNet
    from melee_rl.video import _actor_config

    learner = PPOLearner(
        tiny_adapter, None, ValueNet(tiny_config, tiny_value_config), LearnerConfig(kl_teacher_weight=0.0)
    )
    delayed = tmp_path / "d3.pt"
    save_rl_checkpoint(delayed, learner, tiny_config, step=0, delay_frames=3, context_mode="ring")
    plain = tmp_path / "d0.pt"
    save_bc_checkpoint(plain, tiny_adapter.get_state(), tiny_adapter.config)
    config = _config(_video())
    assert config.actor.delay_frames == 0 and not config.delay.allow_mismatch
    assert _actor_config(config, tiny_adapter, trained_delay=3).delay_frames == 3
    assert _actor_config(config, tiny_adapter, trained_delay=0).delay_frames == 0
    five = replace(config, actor=replace(config.actor, delay_frames=5))
    with pytest.raises(ValueError, match="delay"):
        _actor_config(five, tiny_adapter, trained_delay=3)  # a disagreeing override needs allow_mismatch
    forced = replace(five, delay=replace(five.delay, allow_mismatch=True))
    assert _actor_config(forced, tiny_adapter, trained_delay=3).delay_frames == 5
    zero = replace(config, delay=replace(config.delay, allow_mismatch=True))
    assert _actor_config(zero, tiny_adapter, trained_delay=3).delay_frames == 0  # explicit: a D file at 0
    assert VideoConfig().bc_delay_frames is None
    with pytest.raises(ValueError, match="bc_delay_frames"):
        VideoConfig(bc_delay_frames=-1)

    seen: list[int] = []

    class Recording(OtherOpponent):
        def __init__(self, path: str | Path, batch_size: int, **kwargs: Any) -> None:
            seen.append(int(kwargs.get("delay_frames", 0)))
            super().__init__(path, batch_size, **kwargs)

    monkeypatch.setattr(video_module, "OtherOpponent", Recording)
    output = tmp_path / "videos"
    for bc_delay in (None, 2):
        video = _video(
            matchups=("policy:cpu9", "policy:bc"),
            bc=str(delayed),
            output_dir=str(output),
            bc_delay_frames=bc_delay,
        )
        stamp = f"20260911-08000{0 if bc_delay is None else 1}"
        run_dir = output / f"d3-step1-{stamp}"
        for name in ("policy_vs_cpu9", "policy_vs_bc"):
            for index in range(2):
                directory = run_dir / "replays" / name / f"env{index}"
                directory.mkdir(parents=True)
                (directory / f"Game_{name}_{index}.slp").write_bytes(b"slp")
        run = record_matches(_config(video), checkpoint=delayed, step=1, stamp=stamp, renderer=StubRenderer())
        assert (
            run.delay_frames == 3 and json.loads((run_dir / "manifest.json").read_text())["delay_frames"] == 3
        )
    assert seen == [3, 2]


def test_matchup_parsing_and_expansion() -> None:
    assert MATCHUP_KINDS == ("policy", "bc")
    spec = parse_matchup("policy:cpu9", clips=2, seconds=120.0)
    assert spec == MatchSpec(student="policy", opponent="cpu9", clips=2, seconds=120.0)
    assert spec.name == "policy_vs_cpu9" and not spec.baseline
    assert spec.cpu_level == 9 and spec.players == ("policy", "cpu")
    assert spec.frames == 7200 and FRAMES_PER_SECOND == 60
    versus_bc = parse_matchup("policy:bc", clips=1, seconds=2.0)
    assert versus_bc.cpu_level is None and versus_bc.players == ("policy", "policy")
    assert versus_bc.name == "policy_vs_bc" and not versus_bc.baseline and versus_bc.frames == 120
    baseline = parse_matchup("bc:cpu1", clips=2, seconds=120.0)
    assert baseline.baseline and baseline.name == "bc_vs_cpu1" and baseline.cpu_level == 1
    versus_mimic = parse_matchup("policy:mimic", clips=16, seconds=360.0)
    assert versus_mimic.cpu_level is None and versus_mimic.players == ("policy", "policy")
    assert versus_mimic.name == "policy_vs_mimic" and not versus_mimic.baseline
    assert versus_mimic.mimic and not parse_matchup("policy:bc", clips=1, seconds=1.0).mimic
    versus_smashbot = parse_matchup("policy:smashbot", clips=16, seconds=520.0)
    assert versus_smashbot.cpu_level is None and versus_smashbot.players == ("policy", "policy")
    assert versus_smashbot.name == "policy_vs_smashbot" and not versus_smashbot.baseline
    assert versus_smashbot.smashbot and not versus_mimic.smashbot
    assert not versus_smashbot.mimic
    assert VideoConfig(matchups=("policy:smashbot",)).needs_smashbot
    assert not VideoConfig(matchups=("policy:mimic",)).needs_smashbot
    versus_slippi = parse_matchup("policy:slippi_ai", clips=16, seconds=520.0)
    assert versus_slippi.cpu_level is None and versus_slippi.players == ("policy", "policy")
    assert versus_slippi.name == "policy_vs_slippi_ai" and not versus_slippi.baseline
    assert versus_slippi.slippi_ai and versus_slippi.external
    assert not versus_slippi.mimic and not versus_slippi.smashbot
    assert not versus_smashbot.slippi_ai and not versus_mimic.slippi_ai
    assert VideoConfig(matchups=("policy:slippi_ai",)).needs_slippi_ai
    assert not VideoConfig(matchups=("policy:smashbot",)).needs_slippi_ai
    assert not VideoConfig(matchups=("policy:slippi_ai",)).needs_smashbot
    versus_phillip = parse_matchup("policy:phillip", clips=16, seconds=520.0)
    assert versus_phillip.phillip and versus_phillip.external
    assert versus_phillip.cpu_level is None and versus_phillip.players == ("policy", "policy")
    assert versus_phillip.name == "policy_vs_phillip" and not versus_phillip.baseline
    assert not (versus_phillip.mimic or versus_phillip.smashbot or versus_phillip.slippi_ai)
    assert not versus_slippi.phillip and not versus_smashbot.phillip and not versus_mimic.phillip
    assert VideoConfig(matchups=("policy:phillip",)).needs_phillip
    assert not VideoConfig(matchups=("policy:slippi_ai",)).needs_phillip
    assert not VideoConfig(matchups=("policy:phillip",)).needs_slippi_ai
    for bad in ("", "policy", "policy:", ":cpu9", "policy:cpu0", "policy:cpu10", "bc:bc", "x:cpu9"):
        with pytest.raises(ValueError, match="matchup"):
            parse_matchup(bad, clips=1, seconds=1.0)
    default = VideoConfig()
    assert default.matchups == ("policy:cpu1", "policy:cpu9", "policy:bc", "bc:cpu1", "bc:cpu9")
    assert default.clips == 2 and default.seconds == 120.0 and default.bc is None
    assert default.rollout_length == 128 and default.temperature == 1.0 and default.wandb is False
    assert isinstance(default.render, RenderConfig)
    specs = expand_matches(default)
    assert [item.name for item in specs] == [
        "policy_vs_cpu1",
        "policy_vs_cpu9",
        "policy_vs_bc",
        "bc_vs_cpu1",
        "bc_vs_cpu9",
    ]
    assert sum(item.clips for item in specs) == 10  # 2 clips x 5 matchups (P9 §1)
    assert [item.baseline for item in specs] == [False, False, False, True, True]
    assert default.needs_bc and not VideoConfig(matchups=("policy:cpu9",)).needs_bc
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"clips": 0}, "clips"),
        ({"seconds": 0.0}, "seconds"),
        ({"matchups": ()}, "matchups"),
        ({"matchups": ("policy:cpu9", "policy:cpu9")}, "matchups"),
        ({"rollout_length": 0}, "rollout_length"),
        ({"temperature": -1.0}, "temperature"),
        ({"wandb_seconds": 0.0}, "wandb_seconds"),
        ({"port_offset": -1}, "port_offset"),
    )
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            VideoConfig(**kwargs)


def test_baseline_directory_is_keyed_by_the_bc_checkpoint(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path
) -> None:
    """The four BC clips are re-recorded only when the BC checkpoint changes (user decision)."""

    first = tmp_path / "20m-muon-scaling-c3.pt"
    save_bc_checkpoint(first, tiny_adapter.get_state(), tiny_adapter.config)
    digest = sha256_file(first)
    directory = baseline_dir(tmp_path / "videos", first, subdir="baseline")
    assert directory == tmp_path / "videos" / "baseline" / f"20m-muon-scaling-c3-{digest[:8]}"
    assert baseline_dir(tmp_path / "videos", first, subdir="baseline") == directory  # stable
    other = tmp_path / "other.pt"
    state = tiny_adapter.get_state()
    key = next(iter(state))
    state[key] = state[key] + 1.0
    save_bc_checkpoint(other, state, tiny_adapter.config)
    assert baseline_dir(tmp_path / "videos", other, subdir="baseline") != directory


def test_collect_replays_and_render_jobs(tmp_path: Path) -> None:
    root = tmp_path / "replays" / "policy_vs_cpu9"
    for index in (0, 1):
        (root / f"env{index}").mkdir(parents=True)
    assert collect_replays(root, 2) == [None, None]
    (root / "env0" / "Game_20260826T120000.slp").write_bytes(b"a")
    (root / "env1" / "Game_20260826T120001.slp").write_bytes(b"b")
    (root / "env1" / "notes.txt").write_text("ignored")
    found = collect_replays(root, 2)
    assert [path.name for path in found if path is not None] == [
        "Game_20260826T120000.slp",
        "Game_20260826T120001.slp",
    ]
    assert collect_replays(root, 3)[2] is None  # a Dolphin that never wrote anything
    assert collect_replays(tmp_path / "missing", 1) == [None]
    # Slippi's monthly layout (Dolphin's default) is found too: the search is recursive.
    monthly = root / "env2" / "2026-08"
    monthly.mkdir(parents=True)
    (monthly / "Game_20260826T120002.slp").write_bytes(b"c")
    third = collect_replays(root, 3)[2]
    assert third is not None and third.name == "Game_20260826T120002.slp"
    clips = [
        ClipRecord(
            matchup="policy_vs_cpu9",
            index=1,
            student="policy",
            opponent="cpu9",
            seconds=2.0,
            frames=120,
            caption="cap 1",
            slp=str(tmp_path / "policy_vs_cpu9_1.slp"),
            video=str(tmp_path / "policy_vs_cpu9_1.mp4"),
        ),
        ClipRecord(
            matchup="policy_vs_cpu9",
            index=2,
            student="policy",
            opponent="cpu9",
            seconds=2.0,
            frames=120,
            caption="cap 2",
            slp=None,
            video=str(tmp_path / "policy_vs_cpu9_2.mp4"),
        ),
    ]
    jobs = render_jobs(clips)
    assert len(jobs) == 1 and jobs[0].caption == "cap 1" and jobs[0].frames == 120
    assert jobs[0].slp == tmp_path / "policy_vs_cpu9_1.slp"
    assert jobs[0].target == tmp_path / "policy_vs_cpu9_1.mp4"


def test_caption_for() -> None:
    spec = parse_matchup("policy:cpu9", clips=2, seconds=120.0)
    caption = caption_for(spec, 1, student_label="20m_bc step 1234", opponent_label="CPU 9")
    assert caption == "20m_bc step 1234 (P1) vs CPU 9 (P2) - clip 1/2"


def test_record_matches_on_the_dummy_env(tiny_adapter: MeleePolicyAdapter, tmp_path: Path) -> None:
    """The whole chain: play, per-clip statistics, replays, render jobs, manifest, baseline reuse."""

    student = tmp_path / "latest.pt"
    save_bc_checkpoint(student, tiny_adapter.get_state(), tiny_adapter.config)
    bc = tmp_path / "bc.pt"
    save_bc_checkpoint(bc, tiny_adapter.get_state(), tiny_adapter.config)
    output = tmp_path / "videos"
    video = _video(matchups=("policy:cpu9", "policy:bc", "bc:cpu1"), bc=str(bc), output_dir=str(output))
    config = _config(video)
    run_dir = output / f"latest-step7-{STAMP}"
    baseline = baseline_dir(output, bc, subdir=video.baseline_subdir)
    # The toy environment writes no replays; seed the files the recorder will look for.
    for name, count in (("policy_vs_cpu9", 2), ("policy_vs_bc", 2), ("bc_vs_cpu1", 2)):
        parent = baseline if name.startswith("bc_") else run_dir
        for index in range(count):
            directory = parent / "replays" / name / f"env{index}"
            directory.mkdir(parents=True)
            (directory / f"Game_{name}_{index}.slp").write_bytes(b"slp")
    renderer = StubRenderer()
    run = record_matches(
        config,
        checkpoint=student,
        step=7,
        stamp=STAMP,
        renderer=renderer,
        label="20m_bc",
    )
    assert isinstance(run, VideoRun) and Path(run.directory) == run_dir
    assert run.checkpoint == str(student) and run.checkpoint_sha256 == sha256_file(student)
    assert run.step == 7 and not run.baseline_reused and Path(run.baseline_directory) == baseline
    assert len(run.clips) == 6 and [clip.matchup for clip in run.clips][:2] == ["policy_vs_cpu9"] * 2
    assert [clip.index for clip in run.clips[:2]] == [1, 2]
    for clip in run.clips:
        assert set(clip.stats) == set(CLIP_STATS)
        assert clip.frames >= video.seconds * FRAMES_PER_SECOND
        assert clip.stats["frames"] == float(clip.frames)
        assert clip.slp is not None and Path(clip.slp).is_file()
        assert clip.video is not None and Path(clip.video).is_file()
        assert clip.rendered and clip.error is None
        # P9b: how long the render took and how complete it was travels into the manifest, so the
        # rate is readable after the fact instead of being inferred from the job's wall clock.
        assert clip.render is not None
        assert clip.render["startup_s"] == 11.0 and clip.render["render_fps"] == 57.5
        assert clip.render["slowdown"] == 1.4 and clip.render["stop_reason"] == "end-frame"
        assert clip.render["frames_rendered"] == clip.render["expected_frames"] == clip.frames
        assert Path(clip.slp).parent == (baseline if clip.student == "bc" else run_dir)
        assert Path(clip.slp).stem == f"{clip.matchup}_{clip.index}"
        assert json.dumps(clip.record())  # JSON-able
        # A stats file sits beside every clip, carrying what the manifest carries for it.
        beside = Path(clip.slp).with_suffix(".json")
        assert beside.is_file()
        record = json.loads(beside.read_text())
        assert record["matchup"] == clip.matchup and record["index"] == clip.index
        assert record["video"] == clip.video and record["stats"] == clip.stats
        assert record["caption"] == clip.caption and record["frames"] == clip.frames
        assert record["render"] == clip.render
    assert [Path(job.target).name for job in renderer.jobs] == [
        "policy_vs_cpu9_1.mp4",
        "policy_vs_cpu9_2.mp4",
        "policy_vs_bc_1.mp4",
        "policy_vs_bc_2.mp4",
        "bc_vs_cpu1_1.mp4",
        "bc_vs_cpu1_2.mp4",
    ]
    captions = [job.caption for job in renderer.jobs]
    assert captions[0] == "20m_bc step 7 (P1) vs CPU 9 (P2) - clip 1/2"
    assert captions[2].endswith("(P2) - clip 1/2") and "bc" in captions[2]
    assert captions[4].startswith("bc (P1) vs CPU 1 (P2)")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["checkpoint"] == str(student) and manifest["step"] == 7
    assert len(manifest["clips"]) == 6 and manifest["baseline_reused"] is False
    assert json.loads((baseline / "manifest.json").read_text())["clips"]
    summary = format_summary(run)
    assert "6 clips" in summary and "policy_vs_cpu9" in summary
    # A second run reuses the baseline clips (same BC file) and re-records nothing for them.
    again_renderer = StubRenderer()
    for index in range(2):
        directory = output / f"latest-step8-{STAMP}" / "replays" / "policy_vs_cpu9" / f"env{index}"
        directory.mkdir(parents=True)
        (directory / f"Game_again_{index}.slp").write_bytes(b"slp")
        directory = output / f"latest-step8-{STAMP}" / "replays" / "policy_vs_bc" / f"env{index}"
        directory.mkdir(parents=True)
        (directory / f"Game_again_bc_{index}.slp").write_bytes(b"slp")
    again = record_matches(
        config, checkpoint=student, step=8, stamp=STAMP, renderer=again_renderer, label="20m_bc"
    )
    assert again.baseline_reused and len(again.clips) == 6
    assert [Path(job.target).name for job in again_renderer.jobs] == [
        "policy_vs_cpu9_1.mp4",
        "policy_vs_cpu9_2.mp4",
        "policy_vs_bc_1.mp4",
        "policy_vs_bc_2.mp4",
    ]
    reused = [clip for clip in again.clips if clip.student == "bc"]
    assert len(reused) == 2 and all(clip.reused for clip in reused)
    assert all(Path(clip.video or "").is_file() for clip in reused)


def test_record_matches_reports_missing_pieces(tiny_adapter: MeleePolicyAdapter, tmp_path: Path) -> None:
    student = tmp_path / "latest.pt"
    save_bc_checkpoint(student, tiny_adapter.get_state(), tiny_adapter.config)
    output = tmp_path / "videos"
    # A matchup that needs the BC checkpoint without one configured is an error before anything plays.
    config = _config(_video(matchups=("policy:bc",), bc=None, output_dir=str(output)))
    with pytest.raises(ValueError, match="bc"):
        record_matches(config, checkpoint=student, stamp=STAMP, renderer=StubRenderer())
    assert not output.exists()
    # No replay written (the toy env): the clip is recorded with its statistics and no video.
    config = _config(_video(matchups=("policy:cpu9",), clips=1, output_dir=str(output)))
    renderer = StubRenderer()
    run = record_matches(config, checkpoint=student, stamp=STAMP, renderer=renderer, render=True)
    assert renderer.jobs == [] and len(run.clips) == 1
    clip = run.clips[0]
    # Even a clip with no replay gets its statistics file, in the run directory.
    stats_file = Path(run.directory) / f"{clip.matchup}_{clip.index}.json"
    assert stats_file.is_file() and json.loads(stats_file.read_text())["slp"] is None
    assert clip.slp is None and clip.video is None and not clip.rendered
    assert clip.error is not None and "replay" in clip.error
    assert clip.stats["frames"] > 0.0  # the match itself was played and measured
    # ``render=False`` stops after stage 1 even when replays exist.
    directory = output / f"latest-step0-{STAMP}" / "replays" / "policy_vs_cpu9" / "env0"
    directory.mkdir(parents=True)
    (directory / "Game_x.slp").write_bytes(b"slp")
    stage_one = record_matches(
        config, checkpoint=student, step=0, stamp=STAMP, renderer=renderer, render=False
    )
    assert renderer.jobs == [] and stage_one.clips[0].slp is not None
    assert stage_one.clips[0].video is None and not stage_one.clips[0].rendered


def test_write_clip_records(tmp_path: Path) -> None:
    clips = [
        ClipRecord(
            matchup="policy_vs_cpu1",
            index=index,
            student="policy",
            opponent="cpu1",
            seconds=2.0,
            frames=120,
            caption=f"c{index}",
            stats={name: 1.0 for name in CLIP_STATS},
        )
        for index in (1, 2)
    ]
    written = write_clip_records(clips, tmp_path)
    assert [path.name for path in written] == ["policy_vs_cpu1_1.json", "policy_vs_cpu1_2.json"]
    payload = json.loads(written[1].read_text())
    assert payload["index"] == 2 and payload["caption"] == "c2" and set(payload["stats"]) == set(CLIP_STATS)
    assert write_clip_records([], tmp_path) == []


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    run = VideoRun(
        directory=str(tmp_path),
        checkpoint="/runs/x/latest.pt",
        checkpoint_sha256="ab" * 32,
        step=12,
        baseline_directory=str(tmp_path / "baseline"),
        baseline_reused=True,
        clips=[
            ClipRecord(
                matchup="policy_vs_cpu1",
                index=1,
                student="policy",
                opponent="cpu1",
                seconds=120.0,
                frames=7200,
                caption="c",
                stats={name: 1.0 for name in CLIP_STATS},
            )
        ],
        seconds=42.0,
    )
    path = write_manifest(run, tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    assert payload["checkpoint_sha256"] == "ab" * 32 and payload["baseline_reused"] is True
    assert payload["seconds"] == 42.0 and payload["clips"][0]["matchup"] == "policy_vs_cpu1"
    assert payload["format"] == "melee_rl.video_manifest.v1"


def test_wandb_upload_is_off_by_default_and_skips_without_rendered_clips(tmp_path: Path) -> None:
    """The uploader never runs unless asked, and never imports wandb when there is nothing to send."""

    config = _config(_video(matchups=("policy:cpu9",), output_dir=str(tmp_path)))
    assert not config.video.wandb  # user decision 26 Aug 2026: keep the clips on the Volume
    run = VideoRun(
        directory=str(tmp_path),
        checkpoint="/runs/x/latest.pt",
        checkpoint_sha256="ab" * 32,
        step=1,
        baseline_directory=str(tmp_path),
        baseline_reused=False,
        clips=[
            ClipRecord(
                matchup="policy_vs_cpu9",
                index=1,
                student="policy",
                opponent="cpu9",
                seconds=2.0,
                frames=120,
                caption="c",
                rendered=False,
            )
        ],
    )
    already = "wandb" in sys.modules
    report = wandb_upload(config, run)
    assert report == {"uploaded": 0, "skipped": 1, "error": None}
    assert already or "wandb" not in sys.modules  # nothing above forces the (15 s) import


@pytest.mark.skipif(
    not os.environ.get("MELEE_RL_TEST_WANDB"),
    reason="offline W&B run: set MELEE_RL_TEST_WANDB=1 (imports wandb, about 15 s on this mount)",
)
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_wandb_upload_offline(tmp_path: Path) -> None:
    source = tmp_path / "policy_vs_cpu9_1.mp4"
    generate = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x480:rate=60:duration=3",
        "-pix_fmt",
        "yuv420p",
        str(source),
    ]
    from melee_rl.render import run_command

    assert run_command(generate, timeout_s=180.0).ok
    config = _config(_video(matchups=("policy:cpu9",), output_dir=str(tmp_path), wandb_seconds=1.0))
    config = replace(config, logging=replace(config.logging, dir=str(tmp_path / "wandb")))
    run = VideoRun(
        directory=str(tmp_path),
        checkpoint="/runs/x/latest.pt",
        checkpoint_sha256="ab" * 32,
        step=1,
        baseline_directory=str(tmp_path),
        baseline_reused=False,
        clips=[
            ClipRecord(
                matchup="policy_vs_cpu9",
                index=1,
                student="policy",
                opponent="cpu9",
                seconds=3.0,
                frames=180,
                caption="c",
                video=str(source),
                rendered=True,
                stats={name: 1.0 for name in CLIP_STATS},
            )
        ],
    )
    report = wandb_upload(config, run, mode="offline", work_dir=tmp_path / "trims")
    assert report["uploaded"] == 1 and report["error"] is None and report["run_id"]
    assert (tmp_path / "trims" / "policy_vs_cpu9_1.mp4").is_file()


# ---------------------------------------------------------------------------
# P11: match mode
# ---------------------------------------------------------------------------


def test_video_mode_is_validated_and_defaults_to_clips() -> None:
    assert VIDEO_MODES == ("clip", "match")
    assert _video().mode == "clip"  # P9 behaviour is the default
    assert _video(mode="match").mode == "match"
    with pytest.raises(ValueError, match="mode must be one of"):
        _video(mode="tournament")


def test_games_started_counts_the_replays_of_each_dolphin(tmp_path: Path) -> None:
    root = tmp_path / "replays" / "policy_vs_cpu9"
    for index in (0, 1, 2):
        (root / f"env{index}").mkdir(parents=True)
    assert games_started(root, 3) == [0, 0, 0]
    (root / "env0" / "Game_A.slp").write_bytes(b"a")
    (root / "env0" / "Game_B.slp").write_bytes(b"b")
    (root / "env1" / "Game_A.slp").write_bytes(b"a")
    assert games_started(root, 3) == [2, 1, 0]
    assert games_started(tmp_path / "missing", 2) == [0, 0]


def test_collect_replays_can_take_the_first_game(tmp_path: Path) -> None:
    """Match mode keeps the completed match, not the game the recorder was cut off in."""

    root = tmp_path / "replays" / "policy_vs_cpu9"
    (root / "env0").mkdir(parents=True)
    first = root / "env0" / "Game_20260828T120000.slp"
    first.write_bytes(b"one")
    os.utime(first, (1_700_000_000, 1_700_000_000))
    second = root / "env0" / "Game_20260828T120400.slp"
    second.write_bytes(b"two")
    os.utime(second, (1_700_000_600, 1_700_000_600))

    newest = collect_replays(root, 1)[0]
    oldest = collect_replays(root, 1, first=True)[0]

    assert newest is not None and newest.name == second.name
    assert oldest is not None and oldest.name == first.name


def test_match_record_reads_the_result_out_of_a_replay() -> None:
    summary = slp.ReplaySummary(
        path=Path("policy_vs_cpu9_1.slp"),
        start=slp.GameStart(
            slp_version=(3, 19, 0),
            stage=32,
            seed=0x7FDBDF50,
            players=(
                slp.PlayerStart(port=1, character=2, player_type=0, stocks=4, costume=1, cpu_level=0),
                slp.PlayerStart(port=2, character=2, player_type=1, stocks=4, costume=0, cpu_level=9),
            ),
        ),
        frames=12731,
        first_frame=-123,
        last_frame=12607,
        end_method=slp.END_GAME,
        lras_initiator=-1,
        players=(
            slp.PlayerResult(
                port=1,
                stocks_start=4,
                stocks_end=2,
                percent_end=63.0,
                deaths=2,
                damage_taken=383.0,
                damage_dealt=653.0,
            ),
            slp.PlayerResult(
                port=2,
                stocks_start=4,
                stocks_end=0,
                percent_end=0.0,
                deaths=4,
                damage_taken=653.0,
                damage_dealt=383.0,
            ),
        ),
    )

    record = match_record(summary)

    assert record["seed"] == "0x7fdbdf50" and record["stage"] == 32
    assert record["finished"] and record["ending"] == "GAME!" and record["winner"] == 1
    assert record["seconds"] == pytest.approx(212.18, abs=0.01)
    policy, cpu = record["players"]
    assert (policy["stocks_start"], policy["stocks_end"], policy["deaths"]) == (4, 2, 2)
    assert (cpu["stocks_end"], cpu["deaths"]) == (0, 4)
    # Rates of the kept game alone, not of everything the Dolphin played.
    assert policy["damage_dealt_per_minute"] == pytest.approx(653.0 / (12731 / 3600), abs=0.1)
    assert json.dumps(record)  # JSON-able, like every manifest field


def test_record_matches_in_match_mode_stops_at_the_first_game(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path
) -> None:
    """Every Dolphin already has two replays, so the first rollout ends the matchup."""

    student = tmp_path / "latest.pt"
    save_bc_checkpoint(student, tiny_adapter.get_state(), tiny_adapter.config)
    output = tmp_path / "videos"
    video = _video(matchups=("policy:cpu9",), mode="match", seconds=4.0, output_dir=str(output))
    config = _config(video)
    run_dir = output / f"latest-step1-{STAMP}"
    for index in range(video.clips):
        directory = run_dir / "replays" / "policy_vs_cpu9" / f"env{index}"
        directory.mkdir(parents=True)
        old = directory / "Game_first.slp"
        old.write_bytes(b"the match")
        os.utime(old, (1_700_000_000, 1_700_000_000))
        new = directory / "Game_second.slp"
        new.write_bytes(b"the next game")
        os.utime(new, (1_700_000_600, 1_700_000_600))

    run = record_matches(
        config, checkpoint=student, step=1, stamp=STAMP, renderer=StubRenderer(), label="probe"
    )

    assert len(run.clips) == video.clips
    for clip in run.clips:
        # Stopped after one rollout of 8 frames instead of playing the 240-frame cap.
        assert clip.frames == video.rollout_length
        assert clip.slp is not None and Path(clip.slp).read_bytes() == b"the match"  # the first game
        assert clip.match is None  # the stub replay does not parse; the clip still stands
        assert clip.error is None


def test_the_recorder_preloads_the_checkout_the_release_will_play_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """22 Sep 2026: the preload (import the upstream tree before any Dolphin boots) used the run file's
    ``source_dir`` -- the pin -- and the player then asked for the shared delay-0 Fox's own checkout, which
    the one-tree guard refused: sixteen campaign jobs died at construction.  The preload and the player must
    name the same tree, and the log line the release's own revision."""

    from dataclasses import replace as dataclass_replace

    from melee_rl import video as video_module
    from melee_rl.slippi_ai_agent import SLIPPI_AI_D0_REVISION, SLIPPI_AI_REVISION, SlippiAIConfig

    loaded: list[str] = []
    monkeypatch.setattr(video_module, "load_slippi_ai_api", lambda source_dir: loaded.append(str(source_dir)))
    said: list[str] = []
    config = _config(VideoConfig(matchups=("policy:slippi_ai",)))

    pinned = dataclass_replace(config, slippi_ai=SlippiAIConfig(model="fox_d21_ditto_v4"))
    video_module.preload_slippi_ai(pinned, said.append)
    assert loaded == ["/opt/slippi-ai/source"] and SLIPPI_AI_REVISION[:7] in said[-1]

    shared = dataclass_replace(config, slippi_ai=SlippiAIConfig(model="fox_d0_tx_like_3x512"))
    video_module.preload_slippi_ai(shared, said.append)
    assert loaded[-1] == shared.slippi_ai.resolved_source_dir == "/opt/slippi-ai/source-9eca7479a955"
    assert SLIPPI_AI_D0_REVISION[:7] in said[-1] and "fox_d0_tx_like_3x512" in said[-1]

    custom = SlippiAIConfig(model="fox_d0_tx_like_3x512", source_dir="/x")
    overridden = dataclass_replace(shared, slippi_ai=custom)
    video_module.preload_slippi_ai(overridden, said.append)
    assert loaded[-1] == "/x"
