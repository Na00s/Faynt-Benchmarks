#!/usr/bin/env python3
"""Run or inspect a frozen expanded final-winner benchmark suite.

Gameplay writes only the runner's immutable selected replay and summary.  The
separate media worker publishes ``game/game.slp`` and ``game/game.mp4`` after
the complete suite has passed artifact validation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from melee_policy.integration.final_benchmark_suite import (  # noqa: E402
    BENCHMARK_VARIANT,
    GLOBAL_LOCK_PATH,
    LOCK_PATH,
    MANIFEST_PATH,
    SCHEDULE,
    SCHEDULE_BY_LABEL,
    SUITE_DIRECTORY,
    V3_LOCK_PATH,
    V4_LOCK_PATH,
    SuiteArtifactError,
    atomic_json,
    command_for_game,
    import_semantically_valid_v1_artifacts,
    load_state,
    require_mapping,
    save_state,
    schedule_manifest,
    utc_now,
    validate_completed_game,
    verify_frozen_inputs,
)

DEFERRED_POSTGAME_STATUS = "blocked-deferred-postgame-validation"
DEFERRED_POSTGAME_BACKLOG = "deferred_postgame_validation_backlog"
DEFERABLE_VALIDATION_ERRORS = (FileNotFoundError, SuiteArtifactError)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify the frozen schedule, checkpoints, opponents, and runner surfaces",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="reconcile and print suite state without launching a game",
    )
    parser.add_argument(
        "--label",
        action="append",
        choices=tuple(SCHEDULE_BY_LABEL),
        help="run or inspect only one exact label; repeat to select several",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=tuple(dict.fromkeys(spec.group for spec in SCHEDULE)),
        help="run or inspect one schedule group; repeat to select several",
    )
    parser.add_argument(
        "--maximum-games",
        type=int,
        help="optional bound on newly launched games for a controlled canary batch",
    )
    parser.add_argument(
        "--defer-postgame-validation-failures",
        action="store_true",
        help=(
            "continue after a saved natural-end game fails postgame validation; "
            "leave the game unresolved in the validation backlog"
        ),
    )
    return parser


def _write_or_verify_manifest() -> dict[str, Any]:
    expected = schedule_manifest()
    if MANIFEST_PATH.is_file():
        try:
            observed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SuiteArtifactError(f"suite manifest is unreadable: {error}") from error
        if observed != expected:
            raise SuiteArtifactError(
                f"the persisted suite manifest differs from the frozen {len(SCHEDULE)}-game schedule"
            )
    else:
        atomic_json(MANIFEST_PATH, expected)
    return expected


def _verify_runner_surfaces() -> dict[str, str]:
    dual = ROOT / "scripts" / "run_frisson_dual_oneoff.py"
    if not dual.is_file():
        raise FileNotFoundError(f"head-to-head runner is missing: {dual}")
    play = importlib.util.find_spec("melee_policy.integration.play")
    if play is None or play.origin is None:
        raise FileNotFoundError("public integration.play launcher is unavailable")
    slippi = importlib.util.find_spec("melee_policy.integration.frisson_slippi_match")
    if slippi is None or slippi.origin is None:
        raise FileNotFoundError(
            "Frisson-versus-Slippi-AI runner is unavailable; finish its tested integration first"
        )
    return {
        "head_to_head": str(dual.relative_to(ROOT)),
        "public_play": str(Path(play.origin).resolve().relative_to(ROOT)),
        "frisson_slippi": str(Path(slippi.origin).resolve().relative_to(ROOT)),
    }


def _selected_specs(arguments: argparse.Namespace) -> list[Any]:
    labels = set(arguments.label or ())
    groups = set(arguments.group or ())
    if labels and groups:
        selected = [
            spec for spec in SCHEDULE if spec.label in labels or spec.group in groups
        ]
    elif labels:
        selected = [spec for spec in SCHEDULE if spec.label in labels]
    elif groups:
        selected = [spec for spec in SCHEDULE if spec.group in groups]
    else:
        selected = list(SCHEDULE)

    def execution_priority(spec: Any) -> tuple[int, int]:
        if BENCHMARK_VARIANT in {"posttraining", "postrl"}:
            if spec.kind != "head-to-head" and spec.frisson_profile == "75m":
                return (0, spec.ordinal)
            if spec.kind == "head-to-head":
                return (1, spec.ordinal)
            return (2, spec.ordinal)
        if spec.kind == "head-to-head":
            return (0, spec.ordinal)
        if spec.frisson_profile == "75m":
            return (1, spec.ordinal)
        return (2, spec.ordinal)

    return sorted(selected, key=execution_priority)


def _game_state(state: dict[str, Any], label: str) -> dict[str, Any]:
    games = state.get("games")
    if not isinstance(games, dict):
        raise SuiteArtifactError("suite state games must be an object")
    record = games.get(label)
    if not isinstance(record, dict):
        raise SuiteArtifactError(f"suite state game {label} must be an object")
    return record


def _capture_file_identity(path: Path) -> tuple[str, int]:
    """Hash one stable file and return its digest and byte length."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    stable_metadata = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_metadata):
        raise SuiteArtifactError(f"file changed while hashing: {path}")
    current = path.stat()
    if any(getattr(current, name) != getattr(after, name) for name in stable_metadata):
        raise SuiteArtifactError(f"file path changed while hashing: {path}")
    return digest.hexdigest(), after.st_size


def _saved_natural_end_evidence(spec: Any) -> dict[str, Any]:
    """Bind one failed summary to a complete saved natural-end scoring replay."""

    try:
        summary_bytes = spec.summary_path.read_bytes()
        summary = json.loads(summary_bytes)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteArtifactError(
            f"{spec.label} summary cannot support deferred validation: {error}"
        ) from error
    summary = require_mapping(summary, f"{spec.label} summary")
    execution = require_mapping(summary.get("execution"), "summary.execution")
    if execution.get("game_end_observed") is not True or execution.get("termination") != "natural-game-end":
        raise SuiteArtifactError(f"{spec.label} cannot defer validation without a recorded natural game end")

    artifacts = require_mapping(summary.get("artifacts"), "summary.artifacts")
    replays = artifacts.get("replays")
    if not isinstance(replays, list):
        raise SuiteArtifactError("summary.artifacts.replays must be an array")
    replay_records = [
        require_mapping(record, f"summary.artifacts.replays[{index}]") for index, record in enumerate(replays)
    ]
    selected = [record for record in replay_records if record.get("tournament_result_replay") is True]
    if len(selected) != 1:
        raise SuiteArtifactError(f"{spec.label} has {len(selected)} selected scoring replays")
    replay_record = selected[0]
    replay_value = replay_record.get("path")
    if not isinstance(replay_value, str) or not replay_value:
        raise SuiteArtifactError("scoring replay path must be a nonempty string")
    replay_path = Path(replay_value).expanduser()
    replay_path = (replay_path if replay_path.is_absolute() else ROOT / replay_path).resolve()
    if not replay_path.is_relative_to(spec.artifact_root.resolve()):
        raise SuiteArtifactError(f"{spec.label} scoring replay escapes its artifact root")
    if not replay_path.is_file():
        raise SuiteArtifactError(f"{spec.label} scoring replay is missing or empty")
    replay_sha256, byte_length = _capture_file_identity(replay_path)
    if byte_length <= 0:
        raise SuiteArtifactError(f"{spec.label} scoring replay is missing or empty")
    if replay_record.get("byte_length") != byte_length:
        raise SuiteArtifactError(f"{spec.label} scoring replay byte length changed")
    if replay_record.get("sha256") != replay_sha256:
        raise SuiteArtifactError(f"{spec.label} scoring replay SHA-256 changed")

    validation = require_mapping(replay_record.get("validation"), "scoring replay validation")
    coverage = require_mapping(validation.get("required_trace_coverage"), "scoring replay trace coverage")
    first = validation.get("first_parsed_frame")
    last = validation.get("last_parsed_frame")
    if (
        validation.get("libmelee_parseable") is not True
        or validation.get("parsed_frames_strictly_consecutive") is not True
        or coverage.get("complete") is not True
        or isinstance(first, bool)
        or not isinstance(first, int)
        or isinstance(last, bool)
        or not isinstance(last, int)
        or last < first
    ):
        raise SuiteArtifactError(f"{spec.label} scoring replay failed completeness checks")

    return {
        "summary": {
            "path": str(spec.summary_path.resolve()),
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "byte_length": len(summary_bytes),
        },
        "replay": {
            "path": str(replay_path),
            "sha256": replay_sha256,
            "byte_length": byte_length,
            "first_parsed_frame": first,
            "last_parsed_frame": last,
        },
        "natural_end": {
            "game_end_observed": True,
            "termination": "natural-game-end",
        },
        "reported_result_unverified": {
            "winner": execution.get("winner"),
            "stocks": execution.get("last_stocks"),
        },
    }


def _backlog(state: dict[str, Any]) -> dict[str, Any]:
    value = state.setdefault(DEFERRED_POSTGAME_BACKLOG, {})
    if not isinstance(value, dict):
        raise SuiteArtifactError(f"suite state {DEFERRED_POSTGAME_BACKLOG} must be an object")
    return value


def _clear_deferred(state: dict[str, Any], spec: Any, record: dict[str, Any]) -> None:
    value = state.get(DEFERRED_POSTGAME_BACKLOG)
    if isinstance(value, dict):
        value.pop(spec.label, None)
    record.pop("deferred_postgame_validation", None)


def _mark_deferred(
    state: dict[str, Any],
    spec: Any,
    record: dict[str, Any],
    *,
    error: BaseException,
    evidence: dict[str, Any],
) -> None:
    now = utc_now()
    previous = record.get("deferred_postgame_validation")
    first_deferred_at = previous.get("first_deferred_at_utc") if isinstance(previous, dict) else None
    reason = f"{type(error).__name__}: {error}"
    deferred = {
        "status": "unresolved",
        "first_deferred_at_utc": first_deferred_at or now,
        "last_confirmed_at_utc": now,
        "validation_error": reason,
        "saved_natural_end_evidence": evidence,
    }
    record["status"] = DEFERRED_POSTGAME_STATUS
    record["result"] = None
    record["blocked_reason"] = reason
    record["deferred_postgame_validation"] = deferred
    _backlog(state)[spec.label] = {
        "ordinal": spec.ordinal,
        "label": spec.label,
        **deferred,
    }


def _reconcile_one(
    state: dict[str, Any],
    spec: Any,
    *,
    defer_postgame_validation_failures: bool = False,
) -> str:
    record = _game_state(state, spec.label)
    if spec.summary_path.is_file():
        try:
            result = validate_completed_game(spec)
        except (FileNotFoundError, SuiteArtifactError) as error:
            may_defer = defer_postgame_validation_failures or record.get("status") == DEFERRED_POSTGAME_STATUS
            if may_defer:
                try:
                    evidence = _saved_natural_end_evidence(spec)
                except DEFERABLE_VALIDATION_ERRORS:
                    pass
                else:
                    _mark_deferred(
                        state,
                        spec,
                        record,
                        error=error,
                        evidence=evidence,
                    )
                    return "deferred"
            _clear_deferred(state, spec, record)
            raise
        record["status"] = "complete"
        record["result"] = result
        record["completed_at_utc"] = record.get("completed_at_utc", utc_now())
        record.pop("blocked_reason", None)
        _clear_deferred(state, spec, record)
        return "complete"
    if os.path.lexists(spec.artifact_root):
        _clear_deferred(state, spec, record)
        record["status"] = "blocked-incomplete-artifact"
        record["result"] = None
        record["blocked_reason"] = (
            "the stable artifact directory exists without a validated summary; "
            "preserve it for diagnosis before authorizing a retry"
        )
        return "blocked"
    if record.get("status") == "running":
        _clear_deferred(state, spec, record)
        record["status"] = "blocked-interrupted-before-artifact"
        record["blocked_reason"] = (
            "the previous launcher ended before publishing its stable artifact directory"
        )
        return "blocked"
    if str(record.get("status", "")).startswith("blocked"):
        return "blocked"
    record["status"] = "pending"
    record["result"] = None
    return "pending"


def _reconcile_all(
    state: dict[str, Any],
    *,
    defer_postgame_validation_failures: bool = False,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for spec in SCHEDULE:
        try:
            status = _reconcile_one(
                state,
                spec,
                defer_postgame_validation_failures=defer_postgame_validation_failures,
            )
        except SuiteArtifactError as error:
            record = _game_state(state, spec.label)
            record["status"] = "blocked-invalid-artifact"
            record["blocked_reason"] = str(error)
            record["result"] = None
            status = "blocked"
        counts[status] += 1
    running_labels = [
        spec.label
        for spec in SCHEDULE
        if _game_state(state, spec.label).get("status") == "running"
    ]
    state["current_label"] = running_labels[0] if len(running_labels) == 1 else None
    if counts["complete"] == len(SCHEDULE):
        state["status"] = "gameplay-complete"
        state["current_label"] = None
        state["gameplay_completed_at_utc"] = state.get("gameplay_completed_at_utc", utc_now())
    elif counts["blocked"]:
        state["status"] = "blocked"
    elif counts["deferred"]:
        state["status"] = "incomplete-deferred-postgame-validation"
    elif counts["complete"]:
        state["status"] = "running"
    else:
        state["status"] = "pending"
    return counts


def _print_status(state: dict[str, Any], *, verbose: bool) -> None:
    games = require_mapping(state.get("games"), "suite state games")
    statuses: Counter[str] = Counter()
    for value in games.values():
        record = require_mapping(value, "suite game record")
        status = str(record.get("status", ""))
        if status == DEFERRED_POSTGAME_STATUS:
            statuses["deferred"] += 1
        elif status.startswith("blocked"):
            statuses["blocked"] += 1
        else:
            statuses[status] += 1
    print(
        "FINAL_BENCHMARK_STATUS "
        f"suite={state['status']} complete={statuses['complete']} "
        f"pending={statuses['pending']} blocked={statuses['blocked']} "
        f"deferred={statuses['deferred']} total={len(SCHEDULE)}",
        flush=True,
    )
    if verbose:
        for spec in SCHEDULE:
            record = _game_state(state, spec.label)
            line = f"{spec.ordinal:03d}/{len(SCHEDULE)} {record['status']} {spec.label}"
            result = record.get("result")
            if isinstance(result, dict):
                line += f" winner={result.get('winner')} stocks={result.get('stocks')}"
            if record.get("blocked_reason"):
                line += f" reason={record['blocked_reason']}"
            print(line, flush=True)


def _launch_game(
    state: dict[str, Any],
    spec: Any,
    *,
    defer_postgame_validation_failures: bool = False,
) -> str:
    record = _game_state(state, spec.label)
    if record.get("status") != "pending":
        raise SuiteArtifactError(f"{spec.label} cannot launch from status {record.get('status')!r}")
    command = command_for_game(spec, sys.executable)
    record["status"] = "running"
    record["attempts_started"] = int(record.get("attempts_started", 0)) + 1
    record["started_at_utc"] = utc_now()
    record["command"] = command
    state["status"] = "running"
    state["current_label"] = spec.label
    save_state(state)
    print(f"GAME {spec.ordinal}/{len(SCHEDULE)} START {spec.label}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    record["launcher_returncode"] = completed.returncode
    record["launcher_finished_at_utc"] = utc_now()
    try:
        result = validate_completed_game(spec)
    except (FileNotFoundError, SuiteArtifactError) as error:
        if defer_postgame_validation_failures:
            try:
                evidence = _saved_natural_end_evidence(spec)
            except DEFERABLE_VALIDATION_ERRORS:
                pass
            else:
                _mark_deferred(
                    state,
                    spec,
                    record,
                    error=error,
                    evidence=evidence,
                )
                state["status"] = "incomplete-deferred-postgame-validation"
                state["current_label"] = None
                save_state(state)
                print(
                    f"GAME {spec.ordinal}/{len(SCHEDULE)} DEFERRED {spec.label} "
                    f"validation={type(error).__name__}: {error}",
                    flush=True,
                )
                return "deferred"
        record["status"] = "blocked-launch-or-artifact-failure"
        record["blocked_reason"] = (
            f"launcher exit={completed.returncode}; validation={type(error).__name__}: {error}"
        )
        state["status"] = "blocked"
        state["current_label"] = None
        save_state(state)
        raise SuiteArtifactError(record["blocked_reason"]) from error
    record["status"] = "complete"
    record["result"] = result
    record["completed_at_utc"] = utc_now()
    record.pop("blocked_reason", None)
    _clear_deferred(state, spec, record)
    state["current_label"] = None
    save_state(state)
    print(
        f"GAME {spec.ordinal}/{len(SCHEDULE)} COMPLETE {spec.label} "
        f"winner={result['winner']} stocks={result['stocks']}",
        flush=True,
    )
    return "complete"


def _run_selected_games(
    state: dict[str, Any],
    selected: list[Any],
    arguments: argparse.Namespace,
) -> int:
    launched = 0
    for spec in selected:
        status = _reconcile_one(
            state,
            spec,
            defer_postgame_validation_failures=(arguments.defer_postgame_validation_failures),
        )
        if status == "complete":
            print(
                f"GAME {spec.ordinal}/{len(SCHEDULE)} SKIP complete {spec.label}",
                flush=True,
            )
            continue
        if status == "deferred":
            print(
                f"GAME {spec.ordinal}/{len(SCHEDULE)} SKIP deferred {spec.label}",
                flush=True,
            )
            continue
        if status == "blocked":
            save_state(state)
            raise SystemExit(f"suite game is blocked: {spec.label}")
        if arguments.maximum_games is not None and launched >= arguments.maximum_games:
            break
        _launch_game(
            state,
            spec,
            defer_postgame_validation_failures=(arguments.defer_postgame_validation_failures),
        )
        launched += 1
    return launched


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.maximum_games is not None and arguments.maximum_games <= 0:
        raise ValueError("--maximum-games must be positive")
    SUITE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    GLOBAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    V3_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    V4_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_paths = tuple(
        dict.fromkeys((GLOBAL_LOCK_PATH, V3_LOCK_PATH, V4_LOCK_PATH, LOCK_PATH))
    )
    with ExitStack() as stack:
        locks = [stack.enter_context(path.open("a+")) for path in lock_paths]
        for path, lock in zip(lock_paths, locks, strict=True):
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"another final benchmark orchestrator owns cross-version lock {path}"
                ) from error

        manifest = _write_or_verify_manifest()
        identities = verify_frozen_inputs(include_opponents=True)
        runners = _verify_runner_surfaces()
        imported = import_semantically_valid_v1_artifacts()
        state = load_state()
        counts = _reconcile_all(
            state,
            defer_postgame_validation_failures=(arguments.defer_postgame_validation_failures),
        )
        state["preflight"] = {
            "at_utc": utc_now(),
            "schedule_sha256": manifest["schedule_sha256"],
            "frozen_inputs": identities,
            "runners": runners,
            "v1_imports": imported,
            "passed": True,
        }
        save_state(state)
        print(
            f"PREFLIGHT PASS schedule={manifest['schedule_sha256']} games={len(SCHEDULE)}",
            flush=True,
        )

        if arguments.preflight_only or arguments.status_only:
            _print_status(state, verbose=arguments.status_only)
            return
        if counts["blocked"]:
            _print_status(state, verbose=True)
            raise SystemExit("suite contains a blocked artifact; gameplay launch refused")
        if counts["deferred"] and not arguments.defer_postgame_validation_failures:
            _print_status(state, verbose=True)
            raise SystemExit(
                "suite contains deferred postgame validation failures; pass "
                "--defer-postgame-validation-failures to launch later pending games"
            )

        selected = _selected_specs(arguments)
        _run_selected_games(state, selected, arguments)

        _reconcile_all(
            state,
            defer_postgame_validation_failures=(arguments.defer_postgame_validation_failures),
        )
        save_state(state)
        _print_status(state, verbose=False)


if __name__ == "__main__":
    main()
