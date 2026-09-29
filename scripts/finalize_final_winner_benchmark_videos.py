#!/usr/bin/env python3
"""Serially publish SLP and labeled MP4 bundles for the completed final suite.

The worker refuses to render until every one of the 309 exact gameplay artifacts
passes the shared frozen-suite validator.  Each render uses isolated Dolphin
window capture, Dolphin-only audio, normal-speed output, and the hardened
audio/video synchronization checks in ``render_replay_video``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from melee_policy.integration.final_benchmark_suite import (  # noqa: E402
    MANIFEST_PATH,
    SCHEDULE,
    SCHEDULE_BY_LABEL,
    SUITE_DIRECTORY,
    SUITE_ID,
    SuiteArtifactError,
    _selected_replay,
    _validate_model_identity,
    _validate_players,
    atomic_json,
    expected_video_players,
    load_state,
    read_json,
    require_mapping,
    resolve_artifact_path,
    schedule_manifest,
    sha256_file,
    utc_now,
    validate_completed_game,
    verify_frozen_inputs,
)
from melee_policy.integration.game_bundle import (  # noqa: E402
    GAME_BUNDLE_SCHEMA_VERSION,
    VideoRenderContext,
    finalize_game_bundle,
)
from melee_policy.integration.match_runtime import (  # noqa: E402
    _load_config,
    _resolve_game_image_path,
)
from melee_policy.integration.replay_video import (  # noqa: E402
    _player_video_label,
    render_replay_video,
)

MEDIA_STATE_SCHEMA = "integration.final_winner_benchmark_media_state.v4"
PENDING_FIDELITY_MEDIA_SCHEMA = "integration.pending_fidelity_media.v1"
DEFERRED_POSTGAME_STATUS = "blocked-deferred-postgame-validation"
DEFERRED_POSTGAME_BACKLOG = "deferred_postgame_validation_backlog"
MEDIA_SAFE_PENDING_GATE_CHECKS = frozenset(
    {
        "controller_audit.both_slots_physical_analog_shoulders_within_tolerance",
        "controller_audit.both_slots_physical_buttons_exact",
        "controller_audit.both_slots_processed_upstream_buttons_exact",
    }
)
REQUIRED_DEFERRED_MEDIA_GATE_CHECKS = frozenset(
    {
        "current_run_replay_covers_every_policy_frame",
        "exactly_one_tournament_result_replay",
        "natural_game_end_observed",
        "no_unexpected_auxiliary_replays",
        "parseable_current_run_replay",
        "replay_identity_audit.exact_expected_characters",
        "replay_identity_audit.exact_expected_ports",
        "replay_identity_audit.exact_expected_stage",
        "replay_identity_audit.human_slots",
        "replay_identity_audit.replay_parsed",
        "replay_identity_audit.selected_result_replay_exactly_one",
        "replay_identity_audit.selected_result_replay_identity_exact",
        "strict_consecutive_policy_frames",
    }
)
DEFERRED_MEDIA_SUMMARY_SCHEMAS = {
    "ancestor": "integration.frisson_vs_frisson.v2",
    "cpu9": "integration.frisson_vs_cpu9.v1",
    "head-to-head": "integration.frisson_vs_frisson.v2",
    "mimic": "integration.frisson_vs_mimic.v1",
    "slippi-ai": "integration.frisson_vs_slippi_ai.v1",
}
DEFERRED_MEDIA_CONTROLLER_AUDIT_SCHEMA = "integration.controller_boundary.v12"
MEDIA_STATE_PATH = SUITE_DIRECTORY / "media-state.json"
VIDEO_EXPORT_POLICY_PATH = SUITE_DIRECTORY / "video-export-policy.json"
VIDEO_EXPORT_POLICY_SCHEMA = "integration.benchmark_video_export_policy.v1"
CAPTURE_LOCK = ROOT / ".e010-cache" / "requested-checkpoints" / "wandb-best" / ".hardened-rerender-batch.lock"
HARDENED_CLASSIFIER = (
    ROOT / ".e010-cache" / "requested-checkpoints" / "wandb-best" / "run_hardened_rerender_batch.py"
)
MAX_ATTEMPTS = 3
RETRY_SETTLE_SECONDS = 2.0
ALLOWED_RETRY_BASES = frozenset(
    {
        "transient-render-cause:capture-delivery-sck-display-gap",
        "transient-render-cause:capture-delivery-sck-terminal-gap",
        "transient-render-cause:capture-delivery-encoded-gap",
    }
)
EARLY_COLD_WHITE_RETRY_SCHEMA = "integration.early_cold_white_media_retry.v1"
EARLY_COLD_WHITE_RETRY_BASIS = "reviewed-transient-render-cause:cold-start-near-white-before-frame-21"
EARLY_COLD_WHITE_RETRY_CLASS = (
    "renderer:" + hashlib.sha256(EARLY_COLD_WHITE_RETRY_BASIS.encode("utf-8")).hexdigest()
)
EARLY_COLD_WHITE_MAX_FRAME = 20
ACTIVE_PROCESS_MARKERS = (
    "Slippi Dolphin",
    "macos-replay-recorder",
    "run_final_winner_benchmark.py",
    "run_frisson_dual_oneoff.py",
    "run_final_winners_head_to_head_5.py",
    "run_final_75m_vs_slippi_ai_character_sweep.py",
    "melee_policy.integration.play",
    "melee_policy.integration.frisson_slippi_match",
)

FailureClassifier = Callable[[str], tuple[str, str] | None]
VideoRenderer = Callable[[Path, Path, VideoRenderContext], Mapping[str, Any]]
MediaStatus = Literal["complete", "pending", "blocked"]


class MediaWorkersBusy(RuntimeError):
    """A competing emulator or capture worker requires a clean wait."""


def _video_export_disabled() -> bool:
    if not VIDEO_EXPORT_POLICY_PATH.exists():
        return False
    if VIDEO_EXPORT_POLICY_PATH.is_symlink() or not VIDEO_EXPORT_POLICY_PATH.is_file():
        raise SuiteArtifactError("video export policy must be a regular file")
    policy = read_json(VIDEO_EXPORT_POLICY_PATH, "video export policy")
    expected_schedule_hash = schedule_manifest()["schedule_sha256"]
    if (
        policy.get("schema_version") != VIDEO_EXPORT_POLICY_SCHEMA
        or policy.get("suite_id") != SUITE_ID
        or policy.get("schedule_sha256") != expected_schedule_hash
        or policy.get("save_slp") is not True
        or type(policy.get("save_video")) is not bool
    ):
        raise SuiteArtifactError("video export policy differs from the frozen benchmark suite")
    return policy["save_video"] is False


def _require_video_export_enabled() -> None:
    if _video_export_disabled():
        raise SuiteArtifactError("video export is disabled by the frozen suite policy")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "integration.toml",
    )
    parser.add_argument("--iso-path", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify all 309 gameplay artifacts and media prerequisites without rendering",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="verify all gameplay and print persisted media state without rendering",
    )
    parser.add_argument(
        "--label",
        action="append",
        choices=tuple(SCHEDULE_BY_LABEL),
        help="render one label after the full-suite gameplay preflight; repeat as needed",
    )
    parser.add_argument(
        "--maximum-videos",
        type=int,
        help="optional bound on newly rendered videos for a controlled media canary",
    )
    parser.add_argument(
        "--allow-exact-deferred-for-media",
        action="store_true",
        help=(
            "render hash-bound natural-end replays whose three recognized controller "
            "fidelity checks remain explicitly unresolved"
        ),
    )
    return parser


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256_file(resolved),
        "byte_length": resolved.stat().st_size,
    }


def _stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SuiteArtifactError(f"{label} is missing or is not a regular file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise SuiteArtifactError(f"{label} changed while being read")
    current = path.stat()
    if any(getattr(current, field) != getattr(after, field) for field in stable_fields):
        raise SuiteArtifactError(f"{label} path changed while being read")
    return payload


def _identity_from_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
    }


def _resolve_receipt_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SuiteArtifactError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise SuiteArtifactError(f"{label} escapes the project")
    return resolved


def _deferred_media_result(
    spec: Any,
    record: Mapping[str, Any],
    backlog_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one saved game while leaving its gameplay gate unresolved."""

    if record.get("status") != DEFERRED_POSTGAME_STATUS or record.get("result") is not None:
        raise SuiteArtifactError(f"{spec.label} is not an unresolved deferred game")
    deferred = require_mapping(
        record.get("deferred_postgame_validation"),
        f"{spec.label} deferred receipt",
    )
    expected_backlog = {"ordinal": spec.ordinal, "label": spec.label, **deferred}
    validation_error = deferred.get("validation_error")
    if (
        deferred.get("status") != "unresolved"
        or backlog_record != expected_backlog
        or record.get("blocked_reason") != validation_error
        or validation_error != f"SuiteArtifactError: {spec.label} failed its postgame fidelity gate"
    ):
        raise SuiteArtifactError(f"{spec.label} deferred receipt or backlog differs")

    evidence = require_mapping(
        deferred.get("saved_natural_end_evidence"),
        f"{spec.label} saved natural-end evidence",
    )
    summary_receipt = require_mapping(evidence.get("summary"), f"{spec.label} deferred summary identity")
    summary_path = spec.summary_path.resolve()
    summary_bytes = _stable_bytes(summary_path, f"{spec.label} deferred summary")
    summary_identity = _identity_from_bytes(summary_path, summary_bytes)
    if (
        _resolve_receipt_path(summary_receipt.get("path"), "deferred summary path") != summary_path
        or summary_receipt.get("sha256") != hashlib.sha256(summary_bytes).hexdigest()
        or summary_receipt.get("byte_length") != len(summary_bytes)
    ):
        raise SuiteArtifactError(f"{spec.label} deferred summary identity changed")
    try:
        summary = json.loads(summary_bytes)
    except json.JSONDecodeError as error:
        raise SuiteArtifactError(f"{spec.label} deferred summary is invalid JSON") from error
    summary = require_mapping(summary, f"{spec.label} deferred summary")
    if summary.get("schema_version") != DEFERRED_MEDIA_SUMMARY_SCHEMAS.get(spec.kind):
        raise SuiteArtifactError(f"{spec.label} deferred summary schema differs")
    gate = require_mapping(summary.get("gate"), f"{spec.label} deferred gate")
    gate_checks = require_mapping(gate.get("checks"), f"{spec.label} deferred gate checks")
    if any(not isinstance(passed, bool) for passed in gate_checks.values()):
        raise SuiteArtifactError(f"{spec.label} deferred gate checks must be booleans")
    false_checks = sorted(name for name, passed in gate_checks.items() if passed is not True)
    if (
        summary.get("result") != "failed"
        or gate.get("decision") != "fail"
        or not false_checks
        or set(false_checks) != MEDIA_SAFE_PENDING_GATE_CHECKS
        or any(gate_checks.get(name) is not True for name in REQUIRED_DEFERRED_MEDIA_GATE_CHECKS)
    ):
        raise SuiteArtifactError(
            f"{spec.label} has failures outside the recognized media-safe fidelity checks"
        )
    controller_audit = require_mapping(
        summary.get("controller_boundary_audit"),
        f"{spec.label} controller boundary audit",
    )
    if controller_audit.get("schema_version") != DEFERRED_MEDIA_CONTROLLER_AUDIT_SCHEMA:
        raise SuiteArtifactError(f"{spec.label} controller audit schema differs")
    controller_gate = require_mapping(controller_audit.get("gate"), f"{spec.label} controller audit gate")
    controller_checks = require_mapping(
        controller_gate.get("checks"), f"{spec.label} controller audit checks"
    )
    prefixed_controller_checks = {
        name.removeprefix("controller_audit."): passed
        for name, passed in gate_checks.items()
        if name.startswith("controller_audit.")
    }
    if controller_gate.get("decision") != "fail" or controller_checks != prefixed_controller_checks:
        raise SuiteArtifactError(f"{spec.label} nested controller gate differs")
    _validate_players(summary, spec)
    _validate_model_identity(summary, spec)

    execution = require_mapping(summary.get("execution"), f"{spec.label} deferred execution")
    natural_end = require_mapping(evidence.get("natural_end"), f"{spec.label} natural-end receipt")
    if (
        execution.get("game_end_observed") is not True
        or execution.get("termination") != "natural-game-end"
        or natural_end != {"game_end_observed": True, "termination": "natural-game-end"}
    ):
        raise SuiteArtifactError(f"{spec.label} lacks an exact saved natural game end")

    artifacts = require_mapping(summary.get("artifacts"), "deferred summary artifacts")
    trace_record = require_mapping(artifacts.get("trace"), f"{spec.label} summary trace identity")
    controller_trace = require_mapping(
        controller_audit.get("trace"), f"{spec.label} controller audit trace identity"
    )
    if trace_record != controller_trace:
        raise SuiteArtifactError(f"{spec.label} trace receipts differ")
    trace_path = _resolve_receipt_path(trace_record.get("path"), f"{spec.label} controller trace path")
    if not trace_path.is_relative_to(spec.artifact_root.resolve()):
        raise SuiteArtifactError(f"{spec.label} controller trace escapes its artifact root")
    trace_bytes = _stable_bytes(trace_path, f"{spec.label} controller trace")
    if (
        trace_record.get("sha256") != hashlib.sha256(trace_bytes).hexdigest()
        or trace_record.get("byte_length") != len(trace_bytes)
        or trace_record.get("rows") != execution.get("processed_policy_frames")
    ):
        raise SuiteArtifactError(f"{spec.label} controller trace identity changed")
    replay_values = artifacts.get("replays")
    if not isinstance(replay_values, list):
        raise SuiteArtifactError(f"{spec.label} replay records must be an array")
    selected = [
        require_mapping(value, f"{spec.label} replay record")
        for value in replay_values
        if isinstance(value, Mapping) and value.get("tournament_result_replay") is True
    ]
    if len(selected) != 1:
        raise SuiteArtifactError(f"{spec.label} must have one selected scoring replay")
    replay_record = selected[0]
    replay_path = _resolve_receipt_path(replay_record.get("path"), f"{spec.label} scoring replay path")
    if not replay_path.is_relative_to(spec.artifact_root.resolve()):
        raise SuiteArtifactError(f"{spec.label} scoring replay escapes its artifact root")
    replay_bytes = _stable_bytes(replay_path, f"{spec.label} scoring replay")
    replay_identity = _identity_from_bytes(replay_path, replay_bytes)
    if replay_record.get("sha256") != hashlib.sha256(replay_bytes).hexdigest() or replay_record.get(
        "byte_length"
    ) != len(replay_bytes):
        raise SuiteArtifactError(f"{spec.label} scoring replay identity changed")
    validation = require_mapping(replay_record.get("validation"), f"{spec.label} replay validation")
    coverage = require_mapping(
        validation.get("required_trace_coverage"),
        f"{spec.label} replay trace coverage",
    )
    first_frame = validation.get("first_parsed_frame")
    last_frame = validation.get("last_parsed_frame")
    processed_frames = execution.get("processed_policy_frames")
    if (
        validation.get("libmelee_parseable") is not True
        or validation.get("parsed_frames_strictly_consecutive") is not True
        or coverage.get("complete") is not True
        or isinstance(first_frame, bool)
        or not isinstance(first_frame, int)
        or isinstance(last_frame, bool)
        or not isinstance(last_frame, int)
        or last_frame < first_frame
        or processed_frames != last_frame - first_frame + 1
        or coverage.get("covered_frame_count") != processed_frames
        or coverage.get("required_frame_count") != processed_frames
    ):
        raise SuiteArtifactError(f"{spec.label} deferred replay completeness differs")
    frozen_replay_path, frozen_replay_record = _selected_replay(summary, spec)
    if (
        frozen_replay_path != replay_path
        or frozen_replay_record.get("sha256") != replay_identity["sha256"]
        or frozen_replay_record.get("byte_length") != replay_identity["byte_length"]
    ):
        raise SuiteArtifactError(f"{spec.label} frozen replay validation differs")
    controller_replay = require_mapping(
        controller_audit.get("replay"), f"{spec.label} controller audit replay"
    )
    expected_controller_replay = {
        "path": replay_record.get("path"),
        "sha256": replay_identity["sha256"],
        "byte_length": replay_identity["byte_length"],
    }
    if controller_replay != expected_controller_replay:
        raise SuiteArtifactError(f"{spec.label} controller audit replay differs")

    replay_receipt = require_mapping(evidence.get("replay"), f"{spec.label} deferred replay identity")
    if (
        _resolve_receipt_path(replay_receipt.get("path"), "deferred replay path") != replay_path
        or replay_receipt.get("sha256") != replay_identity["sha256"]
        or replay_receipt.get("byte_length") != replay_identity["byte_length"]
        or replay_receipt.get("first_parsed_frame") != first_frame
        or replay_receipt.get("last_parsed_frame") != last_frame
    ):
        raise SuiteArtifactError(f"{spec.label} deferred replay receipt changed")

    reported = require_mapping(
        evidence.get("reported_result_unverified"),
        f"{spec.label} reported unverified result",
    )
    if reported != {
        "winner": execution.get("winner"),
        "stocks": execution.get("last_stocks"),
    }:
        raise SuiteArtifactError(f"{spec.label} deferred result receipt differs")
    stocks = require_mapping(reported.get("stocks"), f"{spec.label} deferred stocks")
    stock_values = list(stocks.values())
    positive = [name for name, stocks_left in stocks.items() if stocks_left in {1, 2, 3, 4}]
    if (
        len(stock_values) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in stock_values)
        or sorted(stock_values)[0] != 0
        or len(positive) != 1
        or reported.get("winner") != positive[0]
    ):
        raise SuiteArtifactError(f"{spec.label} deferred stock-out result is inconclusive")

    provenance = {
        "schema_version": PENDING_FIDELITY_MEDIA_SCHEMA,
        "status": "pending-fidelity-audit",
        "gameplay_result_scored": False,
        "gameplay_gate_decision": "fail",
        "failed_gate_checks": false_checks,
        "validation_error": validation_error,
        "summary": summary_identity,
        "replay": {
            **replay_identity,
            "first_parsed_frame": first_frame,
            "last_parsed_frame": last_frame,
        },
        "reported_result_unverified": dict(reported),
    }
    return {
        "replay": replay_identity,
        "pending_fidelity_provenance": provenance,
    }


def _pending_fidelity_provenance(
    gameplay_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = gameplay_result.get("pending_fidelity_provenance")
    if value is None:
        return None
    provenance = require_mapping(value, "pending fidelity media provenance")
    if (
        provenance.get("schema_version") != PENDING_FIDELITY_MEDIA_SCHEMA
        or provenance.get("status") != "pending-fidelity-audit"
        or provenance.get("gameplay_result_scored") is not False
        or provenance.get("gameplay_gate_decision") != "fail"
    ):
        raise SuiteArtifactError("pending fidelity media provenance differs")
    return provenance


def _expected_labels(spec: Any) -> tuple[str, str]:
    players = expected_video_players(spec)
    return (
        _player_video_label(players[0], 1),
        _player_video_label(players[1], 2),
    )


def _validate_render_metadata(render: object, spec: Any) -> dict[str, Any]:
    value = require_mapping(render, "game video render metadata")
    expected_labels = _expected_labels(spec)
    labels = require_mapping(value.get("labels"), "game video labels")
    native = require_mapping(value.get("native"), "game video native metadata")
    recording = require_mapping(native.get("recording"), "game video recording metadata")
    startup_sync = require_mapping(recording.get("startup_sync"), "game video startup synchronization")
    mux = require_mapping(native.get("mux"), "game video mux metadata")
    if (
        value.get("method") != "postgame-slippi-playback-isolated-window"
        or value.get("normal_speed") is not True
        or value.get("capture_emulation_speed") != 0.5
        or value.get("requested_capture_fps") != 120
        or value.get("output_playback_speed") != 1.0
        or value.get("output_frame_rate") != 60.0
        or value.get("desktop_captured") is not False
        or value.get("cursor_captured") is not False
        or value.get("microphone_captured") is not False
        or value.get("audio_scope") != "Dolphin replay DSP and DTK dumps only"
        or value.get("video_alignment") != "Dolphin audio sample barrier with piecewise audio-clock retiming"
        or value.get("audio_presentation_delay_seconds") != 0.050
        or labels.get("burned_in") is not True
        or (labels.get("p1"), labels.get("p2")) != expected_labels
    ):
        raise SuiteArtifactError(f"{spec.label} video provenance or labels differ")
    if (
        recording.get("has_video") is not True
        or recording.get("alignment_method") != "dolphin-audio-sample-barrier"
        or startup_sync.get("passed") is not True
        or mux.get("has_video") is not True
        or mux.get("has_audio") is not True
        or mux.get("labels_burned_in") is not True
        or mux.get("player_labels") != list(expected_labels)
        or mux.get("timing_method") != "dolphin-audio-clock-piecewise"
        or mux.get("audio_fully_preserved") is not True
        or mux.get("audio_presentation_delay_seconds") != 0.050
    ):
        raise SuiteArtifactError(f"{spec.label} final mux contract differs")
    return value


def _scoring_replay(spec: Any, result: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    replay = require_mapping(result.get("replay"), "validated gameplay replay")
    path = resolve_artifact_path(replay.get("path"))
    actual = _file_record(path)
    if actual != replay:
        raise SuiteArtifactError(f"{spec.label} replay changed after gameplay validation")
    return path, actual


def _validate_bundle(spec: Any, gameplay_result: Mapping[str, Any]) -> bool:
    game = spec.artifact_root / "game"
    if not os.path.lexists(game):
        return False
    if game.is_symlink() or not game.is_dir():
        raise SuiteArtifactError(f"{spec.label} game output is not a real directory")
    replay_path, replay_identity = _scoring_replay(spec, gameplay_result)
    manifest_path = game / "manifest.json"
    slp_path = game / "game.slp"
    video_path = game / "game.mp4"
    manifest = read_json(manifest_path, f"{spec.label} game manifest")
    if (
        manifest.get("schema_version") != GAME_BUNDLE_SCHEMA_VERSION
        or manifest.get("save_slp") is not True
        or manifest.get("save_video") is not True
    ):
        raise SuiteArtifactError(f"{spec.label} game bundle is incomplete")
    pending_provenance = _pending_fidelity_provenance(gameplay_result)
    if (pending_provenance is None and "gameplay_provenance" in manifest) or (
        pending_provenance is not None and manifest.get("gameplay_provenance") != pending_provenance
    ):
        raise SuiteArtifactError(f"{spec.label} gameplay provenance differs")
    source = require_mapping(manifest.get("source_replay"), "manifest source replay")
    if (
        resolve_artifact_path(source.get("path")) != replay_path
        or source.get("sha256") != replay_identity["sha256"]
        or source.get("byte_length") != replay_identity["byte_length"]
    ):
        raise SuiteArtifactError(f"{spec.label} bundle source replay differs")
    outputs = require_mapping(manifest.get("outputs"), "manifest outputs")
    slp = require_mapping(outputs.get("slp"), "manifest game.slp")
    video = require_mapping(outputs.get("video"), "manifest game.mp4")
    actual_slp = _file_record(slp_path)
    actual_video = _file_record(video_path)
    if (
        resolve_artifact_path(slp.get("path")) != slp_path.resolve()
        or slp.get("sha256") != actual_slp["sha256"]
        or slp.get("byte_length") != actual_slp["byte_length"]
        or actual_slp["sha256"] != replay_identity["sha256"]
        or actual_slp["byte_length"] != replay_identity["byte_length"]
    ):
        raise SuiteArtifactError(f"{spec.label} game.slp differs from its scoring replay")
    if (
        resolve_artifact_path(video.get("path")) != video_path.resolve()
        or video.get("sha256") != actual_video["sha256"]
        or video.get("byte_length") != actual_video["byte_length"]
        or actual_video["byte_length"] <= 0
    ):
        raise SuiteArtifactError(f"{spec.label} game.mp4 identity differs")
    _validate_render_metadata(video.get("render"), spec)
    return True


def _global_gameplay_preflight(*, allow_exact_deferred_for_media: bool = False) -> dict[str, dict[str, Any]]:
    expected_manifest = schedule_manifest()
    if not MANIFEST_PATH.is_file() or read_json(MANIFEST_PATH, "suite manifest") != expected_manifest:
        raise SuiteArtifactError("frozen suite manifest is missing or differs")
    suite_state = load_state()
    games = require_mapping(suite_state.get("games"), "gameplay suite state games")
    if set(games) != set(SCHEDULE_BY_LABEL):
        raise SuiteArtifactError("gameplay suite state labels differ from the frozen schedule")
    deferred_labels = {
        spec.label
        for spec in SCHEDULE
        if require_mapping(games[spec.label], f"state game {spec.label}").get("status")
        == DEFERRED_POSTGAME_STATUS
    }
    backlog = require_mapping(
        suite_state.get(DEFERRED_POSTGAME_BACKLOG, {}),
        "deferred postgame validation backlog",
    )
    if allow_exact_deferred_for_media:
        unsupported = {
            spec.label: require_mapping(games[spec.label], f"state game {spec.label}").get("status")
            for spec in SCHEDULE
            if require_mapping(games[spec.label], f"state game {spec.label}").get("status")
            not in {"complete", DEFERRED_POSTGAME_STATUS}
        }
        expected_suite_status = (
            "incomplete-deferred-postgame-validation" if deferred_labels else "gameplay-complete"
        )
        if unsupported:
            raise SuiteArtifactError(f"media requires every gameplay record to be terminal: {unsupported}")
        if set(backlog) != deferred_labels:
            raise SuiteArtifactError("deferred postgame backlog labels differ from suite state")
        if suite_state.get("status") != expected_suite_status:
            raise SuiteArtifactError("gameplay suite status differs from its terminal records")
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for spec in SCHEDULE:
        try:
            record = require_mapping(games[spec.label], f"state game {spec.label}")
            if allow_exact_deferred_for_media and spec.label in deferred_labels:
                results[spec.label] = _deferred_media_result(
                    spec,
                    record,
                    require_mapping(backlog[spec.label], f"backlog game {spec.label}"),
                )
            else:
                results[spec.label] = validate_completed_game(spec)
        except (FileNotFoundError, SuiteArtifactError) as error:
            failures.append(f"{spec.ordinal:02d} {spec.label}: {type(error).__name__}: {error}")
    if failures:
        preview = "\n".join(failures[:10])
        suffix = f"\n... {len(failures) - 10} more" if len(failures) > 10 else ""
        raise SuiteArtifactError(
            f"video export requires all {len(SCHEDULE)} exact gameplay artifacts; "
            f"failures:\n{preview}{suffix}"
        )
    if suite_state.get("schedule_sha256") != expected_manifest["schedule_sha256"]:
        raise SuiteArtifactError("gameplay suite state schedule binding differs")
    return results


def _initial_media_state() -> dict[str, Any]:
    manifest = schedule_manifest()
    return {
        "schema_version": MEDIA_STATE_SCHEMA,
        "suite_id": SUITE_ID,
        "schedule_sha256": manifest["schedule_sha256"],
        "status": "pending",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "maximum_attempts_per_game": MAX_ATTEMPTS,
        "games": {
            spec.label: {
                "frisson_profile": spec.frisson_profile,
                "matchup_class": spec.matchup_class,
                "player_1_character": spec.player_1_character,
                "opponent_character": spec.opponent_character,
                "status": "pending",
                "attempts": [],
            }
            for spec in SCHEDULE
        },
    }


def _load_media_state() -> dict[str, Any]:
    if not MEDIA_STATE_PATH.is_file():
        return _initial_media_state()
    state = read_json(MEDIA_STATE_PATH, "media state")
    manifest = schedule_manifest()
    if (
        state.get("schema_version") != MEDIA_STATE_SCHEMA
        or state.get("suite_id") != SUITE_ID
        or state.get("schedule_sha256") != manifest["schedule_sha256"]
        or state.get("maximum_attempts_per_game") != MAX_ATTEMPTS
    ):
        raise SuiteArtifactError("media state is bound to a different suite contract")
    games = require_mapping(state.get("games"), "media state games")
    if set(games) != set(SCHEDULE_BY_LABEL):
        raise SuiteArtifactError("media state labels differ from the frozen schedule")
    return state


def _save_media_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json(MEDIA_STATE_PATH, state)


def _media_record(state: dict[str, Any], label: str) -> dict[str, Any]:
    games = state.get("games")
    if not isinstance(games, dict):
        raise SuiteArtifactError("media state games must be an object")
    record = games.get(label)
    if not isinstance(record, dict):
        raise SuiteArtifactError(f"media state {label} must be an object")
    return record


def _load_classifier() -> FailureClassifier | None:
    if not HARDENED_CLASSIFIER.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_final_suite_hardened_classifier", HARDENED_CLASSIFIER)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "render_failure_classification_text", None)
    return cast(FailureClassifier, function) if callable(function) else None


def _classify_retry(error: BaseException, classifier: FailureClassifier | None) -> tuple[str, str] | None:
    if classifier is None:
        return None
    result = classifier(f"{type(error).__name__}: {error}")
    if result is None or result[1] not in ALLOWED_RETRY_BASES:
        return None
    return result


def _early_cold_white_retry_evidence(
    error_text: str,
    expected_replay_value: object,
) -> dict[str, Any] | None:
    """Recognize the one reviewed cold-start failure without relaxing capture gates."""

    expected_replay = require_mapping(expected_replay_value, "early-white source replay")
    blank_matches = re.findall(
        r"captured gameplay blank class ([A-Z]) persisted ([0-9]+(?:\.[0-9]+)?) seconds",
        error_text,
    )
    if not blank_matches or len(blank_matches) > 2 or len(set(blank_matches)) != 1:
        return None
    blank_class, duration_text = blank_matches[0]
    duration = float(duration_text)
    if blank_class != "W" or not 0.5 < duration <= 0.75:
        return None
    if len(blank_matches) == 2:
        blank_message = f"captured gameplay blank class {blank_class} persisted {duration_text} seconds"
        cocoa_duplicate = (
            'Error Domain=melee-policy.replay-recorder Code=1 "'
            f'{blank_message}" UserInfo={{NSLocalizedDescription={blank_message}}}'
        )
        if cocoa_duplicate not in error_text:
            return None

    replay_markers = re.findall(r"\[FILE_PATH\] ([^\r\n]+)", error_text)
    start_markers = re.findall(r"(?m)^\[PLAYBACK_START_FRAME\] (-?[0-9]+)$", error_text)
    game_end_markers = re.findall(r"(?m)^\[GAME_END_FRAME\] (-?[0-9]+)$", error_text)
    end_markers = re.findall(r"(?m)^\[PLAYBACK_END_FRAME\] (-?[0-9]+)$", error_text)
    current_frames = [int(value) for value in re.findall(r"(?m)^\[CURRENT_FRAME\] (-?[0-9]+)$", error_text)]
    if not (len(replay_markers) == len(start_markers) == len(game_end_markers) == len(end_markers) == 1):
        return None
    if start_markers != ["0"] or not current_frames:
        return None
    last_current_frame = current_frames[-1]
    if not 1 <= last_current_frame <= EARLY_COLD_WHITE_MAX_FRAME:
        return None
    if current_frames != list(range(last_current_frame + 1)):
        return None
    game_end_frame = int(game_end_markers[0])
    playback_end_frame = int(end_markers[0])
    if game_end_frame < 120 or playback_end_frame != game_end_frame + 1:
        return None

    replay_path = resolve_artifact_path(expected_replay.get("path"))
    if Path(replay_markers[0]).expanduser().resolve() != replay_path:
        return None
    if _file_record(replay_path) != expected_replay:
        return None

    evidence: dict[str, Any] = {
        "schema_version": EARLY_COLD_WHITE_RETRY_SCHEMA,
        "failure_class": EARLY_COLD_WHITE_RETRY_CLASS,
        "failure_basis": EARLY_COLD_WHITE_RETRY_BASIS,
        "blank_class": blank_class,
        "blank_duration_seconds": duration,
        "first_current_frame": current_frames[0],
        "last_current_frame": last_current_frame,
        "current_frames_strictly_consecutive": True,
        "playback_start_frame": 0,
        "game_end_frame": game_end_frame,
        "playback_end_frame": playback_end_frame,
        "source_replay": dict(expected_replay),
        "capture_gates_relaxed": False,
        "whole_attempt_discarded_before_retry": True,
        "maximum_reviewed_retries": 1,
    }
    signature_payload = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence["signature_sha256"] = hashlib.sha256(signature_payload).hexdigest()
    return evidence


def _reconcile_reviewed_early_cold_white_retry(
    state: dict[str, Any],
    record: dict[str, Any],
    gameplay_result: Mapping[str, Any],
) -> bool:
    """Make the exact retained first failure eligible for one fresh attempt."""

    if record.get("status") != "blocked-nonretryable":
        return False
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        return False
    attempt = attempts[0]
    if not isinstance(attempt, dict) or attempt.get("number") != 1:
        return False
    if attempt.get("status") != "blocked-nonretryable":
        return False
    error_text = attempt.get("error")
    expected_replay = require_mapping(gameplay_result.get("replay"), "gameplay replay")
    if not isinstance(error_text, str) or attempt.get("source_replay") != expected_replay:
        return False
    if record.get("blocked_reason") != error_text:
        return False
    evidence = _early_cold_white_retry_evidence(error_text, expected_replay)
    if evidence is None:
        return False

    attempt["failure_class"] = EARLY_COLD_WHITE_RETRY_CLASS
    attempt["failure_basis"] = EARLY_COLD_WHITE_RETRY_BASIS
    attempt["reviewed_retry_eligibility"] = evidence
    record["reviewed_retry_eligibility"] = evidence
    record["status"] = "retryable-reviewed-early-cold-white"
    _save_media_state(state)
    return True


def _active_conflicts() -> list[str]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("could not prove that gameplay and media processes are idle")
    conflicts: list[str] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        pid_text, separator, command = stripped.partition(" ")
        if not separator or not pid_text.isdigit() or int(pid_text) == os.getpid():
            continue
        if any(marker in command for marker in ACTIVE_PROCESS_MARKERS):
            conflicts.append(stripped)
    return conflicts


def _guarded_renderer(spec: Any, delegate: VideoRenderer = render_replay_video) -> VideoRenderer:
    def render(
        replay_path: Path,
        output_path: Path,
        context: VideoRenderContext,
    ) -> Mapping[str, Any]:
        before = _file_record(replay_path)
        metadata = dict(
            delegate(
                replay_path,
                output_path,
                replace(context, players=expected_video_players(spec)),
            )
        )
        after = _file_record(replay_path)
        if after != before:
            raise SuiteArtifactError(f"{spec.label} scoring replay changed during render")
        _validate_render_metadata(metadata, spec)
        return metadata

    return render


def _render_one(
    state: dict[str, Any],
    spec: Any,
    gameplay_result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    iso_path: Path,
    classifier: FailureClassifier | None,
) -> MediaStatus:
    _require_video_export_enabled()
    record = _media_record(state, spec.label)
    pending_provenance = _pending_fidelity_provenance(gameplay_result)
    if _validate_bundle(spec, gameplay_result):
        record["status"] = "complete"
        record["completed_at_utc"] = record.get("completed_at_utc", utc_now())
        record["bundle"] = str((spec.artifact_root / "game").relative_to(ROOT))
        if pending_provenance is not None:
            record["gameplay_provenance"] = pending_provenance
        _save_media_state(state)
        return "complete"
    _reconcile_reviewed_early_cold_white_retry(state, record, gameplay_result)
    if str(record.get("status", "")).startswith("blocked"):
        return "blocked"
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or any(not isinstance(value, dict) for value in attempts):
        raise SuiteArtifactError(f"{spec.label} media attempts are malformed")
    if any(attempt.get("status") == "started" for attempt in attempts):
        record["status"] = "blocked-interrupted-render"
        record["blocked_reason"] = "a previous render did not persist a terminal outcome"
        _save_media_state(state)
        return "blocked"
    prior_classes = {
        str(attempt["failure_class"]) for attempt in attempts if isinstance(attempt.get("failure_class"), str)
    }
    while len(attempts) < MAX_ATTEMPTS:
        conflicts = _active_conflicts()
        if conflicts:
            raise MediaWorkersBusy(f"gameplay or media workers are active: {conflicts}")
        attempt = {
            "number": len(attempts) + 1,
            "status": "started",
            "started_at_utc": utc_now(),
            "source_replay": dict(require_mapping(gameplay_result.get("replay"), "replay")),
        }
        if pending_provenance is not None:
            attempt["gameplay_provenance"] = pending_provenance
        attempts.append(attempt)
        record["status"] = "rendering"
        _save_media_state(state)
        summary = read_json(spec.summary_path, f"{spec.label} summary")
        try:
            if (
                pending_provenance is not None
                and _file_record(spec.summary_path) != pending_provenance["summary"]
            ):
                raise SuiteArtifactError(f"{spec.label} deferred source summary changed before rendering")
            finalize_game_bundle(
                summary,
                project_root=ROOT,
                config=config,
                iso_path=iso_path,
                save_slp=True,
                save_video=True,
                video_renderer=_guarded_renderer(spec),
                gameplay_provenance=pending_provenance,
                preserve_source_summary=pending_provenance is not None,
            )
            if pending_provenance is None:
                refreshed_result = validate_completed_game(spec)
            else:
                if _file_record(spec.summary_path) != pending_provenance["summary"]:
                    raise SuiteArtifactError(f"{spec.label} deferred source summary changed during rendering")
                refreshed_result = gameplay_result
            if refreshed_result["replay"] != gameplay_result["replay"]:
                raise SuiteArtifactError(f"{spec.label} replay identity changed across bundle publication")
            if not _validate_bundle(spec, refreshed_result):
                raise SuiteArtifactError(f"{spec.label} bundle publication was incomplete")
        except BaseException as error:
            attempt["finished_at_utc"] = utc_now()
            attempt["error"] = f"{type(error).__name__}: {error}"
            error_text = attempt["error"]
            reviewed_evidence = _early_cold_white_retry_evidence(
                error_text,
                gameplay_result.get("replay"),
            )
            if reviewed_evidence is not None:
                attempt["reviewed_retry_eligibility"] = reviewed_evidence
                record.setdefault("reviewed_retry_eligibility", reviewed_evidence)
                classification = (
                    EARLY_COLD_WHITE_RETRY_CLASS,
                    EARLY_COLD_WHITE_RETRY_BASIS,
                )
            else:
                classification = _classify_retry(error, classifier)
            if classification is None:
                attempt["status"] = "blocked-nonretryable"
                record["status"] = attempt["status"]
                record["blocked_reason"] = attempt["error"]
                _save_media_state(state)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                return "blocked"
            failure_class, basis = classification
            attempt["failure_class"] = failure_class
            attempt["failure_basis"] = basis
            if failure_class in prior_classes:
                attempt["status"] = "blocked-repeated-transient-class"
                record["status"] = attempt["status"]
                record["blocked_reason"] = "the same capture gap class occurred twice"
                _save_media_state(state)
                return "blocked"
            reviewed_origin_numbers = [
                int(prior_attempt["number"])
                for prior_attempt in attempts[:-1]
                if prior_attempt.get("failure_class") == EARLY_COLD_WHITE_RETRY_CLASS
                and isinstance(prior_attempt.get("number"), int)
            ]
            if reviewed_origin_numbers and int(attempt["number"]) > min(reviewed_origin_numbers):
                attempt["status"] = "blocked-reviewed-retry-budget-exhausted"
                record["status"] = attempt["status"]
                record["blocked_reason"] = (
                    "the reviewed early-cold-white failure already received its one fresh whole-attempt retry"
                )
                _save_media_state(state)
                return "blocked"
            prior_classes.add(failure_class)
            if len(attempts) >= MAX_ATTEMPTS:
                attempt["status"] = "blocked-attempt-budget-exhausted"
                record["status"] = attempt["status"]
                record["blocked_reason"] = "bounded render retry budget exhausted"
                _save_media_state(state)
                return "blocked"
            attempt["status"] = "retryable-transient-capture-gap"
            record["status"] = attempt["status"]
            _save_media_state(state)
            time.sleep(RETRY_SETTLE_SECONDS)
            continue
        attempt["status"] = "complete"
        attempt["finished_at_utc"] = utc_now()
        record["status"] = "complete"
        record["completed_at_utc"] = utc_now()
        record["bundle"] = str((spec.artifact_root / "game").relative_to(ROOT))
        if pending_provenance is not None:
            record["gameplay_provenance"] = pending_provenance
        record.pop("blocked_reason", None)
        _save_media_state(state)
        return "complete"
    record["status"] = "blocked-attempt-budget-exhausted"
    record["blocked_reason"] = "bounded render retry budget exhausted"
    _save_media_state(state)
    return "blocked"


def _print_status(state: Mapping[str, Any]) -> None:
    games = require_mapping(state.get("games"), "media state games")
    complete = 0
    blocked = 0
    for value in games.values():
        record = require_mapping(value, "media game record")
        status = str(record.get("status"))
        complete += status == "complete"
        blocked += status.startswith("blocked")
    print(
        f"FINAL_MEDIA_STATUS complete={complete} blocked={blocked} "
        f"pending={len(SCHEDULE) - complete - blocked} total={len(SCHEDULE)}",
        flush=True,
    )


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.maximum_videos is not None and arguments.maximum_videos <= 0:
        raise ValueError("--maximum-videos must be positive")
    if _video_export_disabled():
        print("FINAL_VIDEO_STATUS not-requested save_video=false save_slp=true", flush=True)
        return
    verify_frozen_inputs(include_opponents=True, config_path=arguments.config)
    gameplay_results = _global_gameplay_preflight(
        allow_exact_deferred_for_media=arguments.allow_exact_deferred_for_media
    )
    config, project_root = _load_config(arguments.config)
    if project_root != ROOT:
        raise RuntimeError(f"configured project root differs: {project_root}")
    iso_path = _resolve_game_image_path(config, ROOT, arguments.iso_path)
    classifier = _load_classifier()
    state = _load_media_state()
    if arguments.preflight_only or arguments.status_only:
        print(
            f"MEDIA_PREFLIGHT PASS exact_gameplay_artifacts={len(SCHEDULE)}",
            flush=True,
        )
        _print_status(state)
        return

    CAPTURE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with CAPTURE_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            print("MEDIA_WAIT another hardened media worker owns the capture lock", flush=True)
            raise SystemExit(75) from error
        selected_labels = set(arguments.label or ())
        targets = [spec for spec in SCHEDULE if not selected_labels or spec.label in selected_labels]
        rendered = 0
        blocked = 0
        waiting = False
        for spec in targets:
            if arguments.maximum_videos is not None and rendered >= arguments.maximum_videos:
                break
            print(f"VIDEO {spec.ordinal}/{len(SCHEDULE)} CHECK {spec.label}", flush=True)
            before = _media_record(state, spec.label).get("status")
            try:
                status = _render_one(
                    state,
                    spec,
                    gameplay_results[spec.label],
                    config=config,
                    iso_path=iso_path,
                    classifier=classifier,
                )
            except MediaWorkersBusy as error:
                print(f"MEDIA_WAIT {error}", flush=True)
                waiting = True
                break
            if status == "blocked":
                blocked += 1
                print(
                    f"VIDEO {spec.ordinal}/{len(SCHEDULE)} BLOCKED {spec.label}",
                    flush=True,
                )
                if arguments.maximum_videos is not None:
                    break
                continue
            if before != "complete":
                rendered += 1
            print(
                f"VIDEO {spec.ordinal}/{len(SCHEDULE)} COMPLETE {spec.label}",
                flush=True,
            )
        all_complete = all(_media_record(state, spec.label).get("status") == "complete" for spec in SCHEDULE)
        state["status"] = (
            "complete"
            if all_complete
            else "waiting-for-workers"
            if waiting
            else "blocked"
            if blocked
            else "pending"
        )
        if all_complete:
            state["completed_at_utc"] = state.get("completed_at_utc", utc_now())
        _save_media_state(state)
    _print_status(state)
    if waiting:
        raise SystemExit(75)
    if blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
