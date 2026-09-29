"""Research execution helpers extracted without cloud bootstrap side effects."""
from __future__ import annotations
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "rl"

def available_configs() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.toml"))

def resolve_config(config: str | Path) -> Path:
    """A path to a TOML run file, or a name (``smoke_3m`` / ``smoke_3m.toml``) under ``configs/rl``."""

    candidate = Path(config)
    if candidate.is_file():
        return candidate.resolve()
    for option in (CONFIG_DIR / candidate.name, CONFIG_DIR / f"{candidate.name}.toml"):
        if option.is_file():
            return option
    raise FileNotFoundError(
        f"no run file {str(config)!r}: give a path or one of {available_configs()} (in {CONFIG_DIR})"
    )

def split_tags(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())

def run_record_video(
    config: str | Path,
    *,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    render: bool = True,
    force_baseline: bool = False,
    wandb: bool = False,
    wandb_mode: str = "online",
    device: str | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Record every matchup of ``config``'s ``[video]`` table for ``checkpoint`` (both stages); ``overrides``
    are ``table.key=value`` lines applied to the run file first (5 Sep 2026: ``actor.decision_stride=k``)."""

    from dataclasses import replace as dataclass_replace

    from melee_rl.checkpoint import load_checkpoint
    from melee_rl.config import load_config
    from melee_rl.video import format_summary, record_matches, wandb_upload

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    video = run_config.video
    video_overrides: dict[str, Any] = {}
    if clips is not None:
        video_overrides["clips"] = clips
    if seconds is not None:
        video_overrides["seconds"] = seconds
    if matchups is not None:
        video_overrides["matchups"] = split_tags(matchups)
    if video_overrides:
        run_config = dataclass_replace(run_config, video=dataclass_replace(video, **video_overrides))
    source = Path(checkpoint)
    if step is None:
        try:
            step = int(load_checkpoint(source)["step"])
        except Exception:  # a BC export has no step; the caption then omits it
            step = None
    if label is None and source.stem in ("latest", "best"):
        label = source.parent.name
    run = record_matches(
        run_config,
        checkpoint=source,
        output_dir=output_dir,
        bc=bc,
        step=step,
        label=label,
        # The run file's [runtime] device, unless the caller overrides it: every P9/P11 recorder says
        # "cpu" (batch 2 is latency-bound), and P12 says "cuda" because MIMIC has no KV cache.
        device=device or run_config.runtime.device,
        render=render,
        force_baseline=force_baseline,
        log=print,
    )
    summary = format_summary(run)
    print(summary)
    upload = None
    if wandb or run_config.video.wandb:
        upload = wandb_upload(run_config, run, mode=wandb_mode)
        print(f"wandb upload: {upload}")
    return {
        **run.record(),
        "summary": summary,
        "wandb": upload,
        "duration_s": time.perf_counter() - started,
    }

def run_external_match(
    config: str | Path,
    *,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P25: real four-stock matches between two outside agents (``sides`` = ``"smashbot,cpu9"``).

    Neither side is one of our checkpoints, which the recorder's ``<policy|bc>:<opponent>`` grammar
    cannot express -- these are the calibration arms that place SmashBot on the same ladder as every
    model in ``RESULTS.md`` before any of our weights are pointed at it.
    """

    from melee_rl.config import load_config
    from melee_rl.external_match import record_external_matches

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    result = record_external_matches(
        run_config,
        sides=split_tags(sides),
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        log=print,
    )
    return {**result, "duration_s": time.perf_counter() - started}

def run_phillip_check(
    config: str | Path, *, agents: str = "", overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """P32: :func:`melee_rl.phillip.preflight` for every agent in ``agents`` (comma separated; empty = the
    seven delay-0 agents), on the run file's ``[phillip]`` table.  The report per agent is what the gate
    reads:
    a clean restore (no variable initialised or padded), zero exceptions, and the sidecar's cost per
    decision.  Nothing boots a Dolphin."""

    from dataclasses import replace as dataclass_replace

    from melee_rl.config import load_config
    from melee_rl.phillip import DELAY0_AGENTS, format_preflight, preflight

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    names = split_tags(agents) or DELAY0_AGENTS
    reports: dict[str, Any] = {}
    for name in names:
        settings = dataclass_replace(run_config.phillip, agent=name)
        try:
            report = preflight(settings, frames=180, log=print)
        except Exception as error:  # one broken agent must not hide the others' reports
            report = {"agent": name, "error": f"{type(error).__name__}: {error}"}
            print(f"phillip {name}: FAILED {report['error']}")
        else:
            print(format_preflight(report))
        reports[name] = report
    ok = all(
        "error" not in report
        and not report["restore"]["missing"]
        and not report["restore"]["padded"]
        and report["exceptions"] == 0
        for report in reports.values()
    )
    print(f"phillip_check: {'OK' if ok else 'FAILED'} for {len(reports)} agent(s)")
    return {"ok": ok, "agents": reports, "duration_s": time.perf_counter() - started}

def run_slippi_check(
    config: str | Path, *, models: str = "", overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """22 Sep 2026: :func:`melee_rl.slippi_ai_agent.preflight` for every release in ``models`` (comma
    separated; empty = the run file's ``[slippi_ai] model``), on the run file's ``[slippi_ai]`` table.
    The report per release is what the gate reads: the sha256 verified, zero exceptions over the
    synthetic frames, a controller that moved.  Nothing boots a Dolphin; the delay-0 JAX file was the
    reason (its pip layer, its restore and its host copy all had to be proven before a campaign)."""

    from dataclasses import replace as dataclass_replace

    from melee_rl.config import load_config
    from melee_rl.slippi_ai_agent import format_preflight, preflight

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    names = split_tags(models) or (run_config.slippi_ai.model,)
    reports: dict[str, Any] = {}
    for name in names:
        settings = dataclass_replace(run_config.slippi_ai, model=name)
        try:
            report = preflight(settings, frames=180, num_envs=2, log=print)
        except Exception as error:  # one broken release must not hide the others' reports
            report = {"model": name, "error": f"{type(error).__name__}: {error}"}
            print(f"slippi-ai {name}: FAILED {report['error']}")
        else:
            print(format_preflight(report))
        reports[name] = report
    ok = all(
        "error" not in report
        and report["sha256_verified"]
        and report["exceptions"] == 0
        and report["rows_changed"] > 0
        for report in reports.values()
    )
    print(f"slippi_check: {'OK' if ok else 'FAILED'} for {len(reports)} release(s)")
    return {"ok": ok, "models": reports, "duration_s": time.perf_counter() - started}

