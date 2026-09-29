from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from melee_policy.integration import final_benchmark_suite as suite
from melee_policy.integration.game_bundle import GAME_BUNDLE_SCHEMA_VERSION


def _load_finalizer() -> ModuleType:
    path = suite.ROOT / "scripts" / "finalize_final_winner_benchmark_videos.py"
    spec = importlib.util.spec_from_file_location("_test_final_suite_media", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_video_export_policy(
    finalizer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: object,
) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "video-export-policy.json"
    policy: dict[str, object] = {
        "schema_version": "integration.benchmark_video_export_policy.v1",
        "suite_id": "fixture-suite",
        "schedule_sha256": "fixture-schedule",
        "save_video": False,
        "save_slp": True,
    }
    policy.update(overrides)
    path.write_text(json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "VIDEO_EXPORT_POLICY_PATH", path)
    monkeypatch.setattr(finalizer, "SUITE_ID", "fixture-suite")
    monkeypatch.setattr(
        finalizer, "schedule_manifest", lambda: {"schedule_sha256": "fixture-schedule"}
    )
    return path, policy


def test_frozen_policy_disables_video_and_preserves_slp(tmp_path: Path, monkeypatch) -> None:
    finalizer = _load_finalizer()
    path, _ = _install_video_export_policy(finalizer, tmp_path, monkeypatch)
    before = path.read_bytes()
    assert finalizer._video_export_disabled() is True
    assert path.read_bytes() == before


def test_disabled_policy_exits_before_any_export_preflight_or_state_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    finalizer = _load_finalizer()
    path, _ = _install_video_export_policy(finalizer, tmp_path, monkeypatch)
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["finalizer"])
    for name in (
        "verify_frozen_inputs",
        "_global_gameplay_preflight",
        "_load_config",
        "_load_media_state",
    ):
        monkeypatch.setattr(
            finalizer,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(f"called {_name}"),
        )

    finalizer.main()

    assert capsys.readouterr().out.strip() == (
        "FINAL_VIDEO_STATUS not-requested save_video=false save_slp=true"
    )
    assert path.read_bytes() == before


def test_disabled_policy_blocks_direct_single_game_render_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    _install_video_export_policy(finalizer, tmp_path, monkeypatch)
    state = {"games": {"fixture": {"status": "pending", "attempts": []}}}
    before = json.dumps(state, sort_keys=True)
    monkeypatch.setattr(finalizer, "_validate_bundle", lambda *args: pytest.fail("export reached"))

    with pytest.raises(finalizer.SuiteArtifactError, match="video export is disabled"):
        finalizer._render_one(
            state,
            SimpleNamespace(label="fixture"),
            {"replay": {}},
            config={},
            iso_path=tmp_path / "melee.iso",
            classifier=None,
        )
    assert json.dumps(state, sort_keys=True) == before


@pytest.mark.parametrize(
    "override",
    [
        {"suite_id": "wrong-suite"},
        {"schedule_sha256": "wrong-schedule"},
        {"save_slp": False},
        {"save_video": "false"},
    ],
)
def test_video_export_policy_identity_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    finalizer = _load_finalizer()
    _install_video_export_policy(finalizer, tmp_path, monkeypatch, **override)
    with pytest.raises(finalizer.SuiteArtifactError, match="differs from the frozen benchmark suite"):
        finalizer._video_export_disabled()


def test_active_worker_is_transient_and_leaves_untouched_media_record_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    state = {"games": {"first": {"status": "pending", "attempts": []}}}
    before = json.dumps(state, sort_keys=True)
    monkeypatch.setattr(finalizer, "_validate_bundle", lambda *_args: False)
    monkeypatch.setattr(finalizer, "_active_conflicts", lambda: ["123 Dolphin"])
    monkeypatch.setattr(
        finalizer, "_save_media_state", lambda _state: pytest.fail("Unexpected state mutation")
    )
    with pytest.raises(finalizer.MediaWorkersBusy):
        finalizer._render_one(
            state,
            SimpleNamespace(label="first"),
            {"replay": {}},
            config={},
            iso_path=tmp_path / "melee.iso",
            classifier=None,
        )
    assert json.dumps(state, sort_keys=True) == before


@pytest.mark.parametrize("outcome,exit_code", [("blocked", 1), ("busy", 75)])
def test_bounded_batch_stops_at_first_failed_or_busy_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, exit_code: int
) -> None:
    finalizer = _load_finalizer()
    schedule = [
        SimpleNamespace(label=label, ordinal=index)
        for index, label in enumerate(("first", "second", "third"), start=1)
    ]
    state = {"games": {spec.label: {"status": "pending", "attempts": []} for spec in schedule}}
    calls = []
    monkeypatch.setattr(sys, "argv", ["finalizer", "--maximum-videos", "1"])
    monkeypatch.setattr(finalizer, "SCHEDULE", schedule)
    monkeypatch.setattr(finalizer, "verify_frozen_inputs", lambda **kw: None)
    monkeypatch.setattr(
        finalizer, "_global_gameplay_preflight", lambda **kw: {spec.label: {} for spec in schedule}
    )
    monkeypatch.setattr(finalizer, "_load_config", lambda _path: ({}, finalizer.ROOT))
    monkeypatch.setattr(finalizer, "_resolve_game_image_path", lambda *args: tmp_path / "melee.iso")
    monkeypatch.setattr(finalizer, "_load_classifier", lambda: None)
    monkeypatch.setattr(finalizer, "_load_media_state", lambda: state)
    monkeypatch.setattr(finalizer, "_save_media_state", lambda value: None)
    monkeypatch.setattr(finalizer, "CAPTURE_LOCK", tmp_path / "capture.lock")

    def render_one(state, spec, result, **kwargs):
        calls.append(spec.label)
        if outcome == "busy":
            raise finalizer.MediaWorkersBusy("another capture is active")
        state["games"][spec.label]["status"] = "blocked-nonretryable"
        return "blocked"

    monkeypatch.setattr(finalizer, "_render_one", render_one)
    with pytest.raises(SystemExit) as error:
        finalizer.main()
    assert error.value.code == exit_code
    assert calls == ["first"]
    assert state["games"]["second"] == {"status": "pending", "attempts": []}
    assert state["games"]["third"] == {"status": "pending", "attempts": []}
    if outcome == "busy":
        assert state["games"]["first"] == {"status": "pending", "attempts": []}
        assert state["status"] == "waiting-for-workers"


def _render_metadata(finalizer: ModuleType, game: suite.GameSpec) -> dict[str, object]:
    p1, p2 = finalizer._expected_labels(game)
    return {
        "method": "postgame-slippi-playback-isolated-window",
        "normal_speed": True,
        "capture_emulation_speed": 0.5,
        "requested_capture_fps": 120,
        "output_playback_speed": 1.0,
        "output_frame_rate": 60.0,
        "desktop_captured": False,
        "cursor_captured": False,
        "microphone_captured": False,
        "audio_scope": "Dolphin replay DSP and DTK dumps only",
        "video_alignment": "Dolphin audio sample barrier with piecewise audio-clock retiming",
        "audio_presentation_delay_seconds": 0.050,
        "labels": {"burned_in": True, "p1": p1, "p2": p2},
        "native": {
            "recording": {
                "has_video": True,
                "alignment_method": "dolphin-audio-sample-barrier",
                "startup_sync": {"passed": True},
            },
            "mux": {
                "has_video": True,
                "has_audio": True,
                "labels_burned_in": True,
                "player_labels": [p1, p2],
                "timing_method": "dolphin-audio-clock-piecewise",
                "audio_fully_preserved": True,
                "audio_presentation_delay_seconds": 0.050,
            },
        },
    }


def test_video_labels_name_both_exact_frameworks_and_follow_physical_ports() -> None:
    finalizer = _load_finalizer()
    first = suite.SCHEDULE[0]
    swapped = suite.SCHEDULE[1]
    assert finalizer._expected_labels(first) == (
        "P1  Frisson-AI 10M v2 final step-122064  FOX",
        "P2  Frisson-AI 75M final step-86016  FOX",
    )
    assert finalizer._expected_labels(swapped) == (
        "P1  Frisson-AI 75M final step-86016  FOX",
        "P2  Frisson-AI 10M v2 final step-122064  FOX",
    )
    slippi = next(game for game in suite.SCHEDULE if game.kind == "slippi-ai")
    assert "vladfi1/slippi-ai medium-v2 (Master Player)" in finalizer._expected_labels(slippi)[1]
    falco = next(
        game
        for game in suite.SCHEDULE
        if game.kind == "slippi-ai" and game.frisson_profile == "10m" and game.opponent_character == "FALCO"
    )
    assert finalizer._expected_labels(falco) == (
        "P1  Frisson-AI 10M v2 final step-122064  FALCO",
        "P2  vladfi1/slippi-ai medium-v2 (Master Player)  FALCO",
    )


def test_all_extension_cpu_video_labels_use_the_exact_mirror_character() -> None:
    extension_cpu = [spec for spec in suite.SCHEDULE[117:] if spec.kind == "cpu9"]
    assert len(extension_cpu) == 56
    for spec in extension_cpu:
        players = suite.expected_video_players(spec)
        assert players[0]["character"] == spec.player_1_character
        assert players[1]["character"] == spec.opponent_character
        assert players[0]["character"] == players[1]["character"]


def test_render_contract_rejects_desktop_capture_and_accepts_hardened_av() -> None:
    finalizer = _load_finalizer()
    game = suite.SCHEDULE[0]
    metadata = _render_metadata(finalizer, game)
    assert finalizer._validate_render_metadata(metadata, game) == metadata
    metadata["desktop_captured"] = True
    with pytest.raises(suite.SuiteArtifactError, match="provenance"):
        finalizer._validate_render_metadata(metadata, game)


def test_complete_bundle_binds_slp_to_raw_scoring_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "ARTIFACT_OUTPUT", tmp_path / "artifacts")
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    game = suite.SCHEDULE[0]
    raw = game.artifact_root / "replays" / "scoring.slp"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"immutable raw replay")
    replay = finalizer._file_record(raw)
    gameplay_result = {"replay": replay}

    bundle = game.artifact_root / "game"
    bundle.mkdir()
    slp = bundle / "game.slp"
    video = bundle / "game.mp4"
    slp.write_bytes(raw.read_bytes())
    video.write_bytes(b"synthetic mp4")
    manifest = {
        "schema_version": GAME_BUNDLE_SCHEMA_VERSION,
        "save_slp": True,
        "save_video": True,
        "source_replay": replay,
        "outputs": {
            "slp": finalizer._file_record(slp),
            "video": {
                **finalizer._file_record(video),
                "render": _render_metadata(finalizer, game),
            },
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert finalizer._validate_bundle(game, gameplay_result) is True

    slp.write_bytes(b"different replay")
    with pytest.raises(suite.SuiteArtifactError, match=r"game\.slp differs"):
        finalizer._validate_bundle(game, gameplay_result)


def test_media_state_record_mutation_is_persistent() -> None:
    finalizer = _load_finalizer()
    state = finalizer._initial_media_state()
    label = suite.SCHEDULE[0].label
    record = finalizer._media_record(state, label)
    assert record["frisson_profile"] == suite.SCHEDULE[0].frisson_profile
    assert record["player_1_character"] == suite.SCHEDULE[0].player_1_character
    record["status"] = "complete"
    assert state["games"][label]["status"] == "complete"


def _deferred_game(
    finalizer: ModuleType,
    tmp_path: Path,
) -> tuple[SimpleNamespace, dict[str, Any], dict[str, Any]]:
    label = "deferred-controller-fidelity"
    artifact_root = tmp_path / "artifacts" / label
    replay_path = artifact_root / "replays" / "scoring.slp"
    replay_path.parent.mkdir(parents=True)
    replay_bytes = b"complete natural-end replay"
    replay_path.write_bytes(replay_bytes)
    trace_path = artifact_root / "controller_trace.jsonl"
    trace_bytes = b"{}\n" * 101
    trace_path.write_bytes(trace_bytes)
    false_checks = {
        "controller_audit.both_slots_physical_analog_shoulders_within_tolerance",
        "controller_audit.both_slots_physical_buttons_exact",
        "controller_audit.both_slots_processed_upstream_buttons_exact",
    }
    checks = {name: True for name in finalizer.REQUIRED_DEFERRED_MEDIA_GATE_CHECKS}
    checks.update({name: False for name in false_checks})
    trace_record = {
        "path": str(trace_path.relative_to(tmp_path)),
        "sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "byte_length": len(trace_bytes),
        "rows": 101,
    }
    controller_checks = {
        name.removeprefix("controller_audit."): passed
        for name, passed in checks.items()
        if name.startswith("controller_audit.")
    }
    replay_record = {
        "path": str(replay_path.relative_to(tmp_path)),
        "sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "byte_length": len(replay_bytes),
        "tournament_result_replay": True,
        "validation": {
            "libmelee_parseable": True,
            "parsed_frames_strictly_consecutive": True,
            "first_parsed_frame": -23,
            "last_parsed_frame": 77,
            "required_trace_coverage": {
                "complete": True,
                "covered_frame_count": 101,
                "required_frame_count": 101,
            },
        },
    }
    summary = {
        "schema_version": "integration.frisson_vs_mimic.v1",
        "result": "failed",
        "error": "RuntimeError: controller fidelity checks failed",
        "gate": {"decision": "fail", "checks": checks},
        "execution": {
            "game_end_observed": True,
            "termination": "natural-game-end",
            "processed_policy_frames": 101,
            "winner": "frisson-ai",
            "last_stocks": {"frisson-ai": 2, "mimic": 0},
        },
        "controller_boundary_audit": {
            "schema_version": "integration.controller_boundary.v12",
            "gate": {"decision": "fail", "checks": controller_checks},
            "trace": trace_record,
            "replay": {key: replay_record[key] for key in ("path", "sha256", "byte_length")},
        },
        "artifacts": {
            "trace": trace_record,
            "replays": [replay_record],
        },
    }
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_bytes = summary_path.read_bytes()
    error = f"SuiteArtifactError: {label} failed its postgame fidelity gate"
    deferred = {
        "status": "unresolved",
        "first_deferred_at_utc": "start",
        "last_confirmed_at_utc": "finish",
        "validation_error": error,
        "saved_natural_end_evidence": {
            "summary": {
                "path": str(summary_path),
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "byte_length": len(summary_bytes),
            },
            "replay": {
                "path": str(replay_path),
                "sha256": hashlib.sha256(replay_bytes).hexdigest(),
                "byte_length": len(replay_bytes),
                "first_parsed_frame": -23,
                "last_parsed_frame": 77,
            },
            "natural_end": {
                "game_end_observed": True,
                "termination": "natural-game-end",
            },
            "reported_result_unverified": {
                "winner": "frisson-ai",
                "stocks": {"frisson-ai": 2, "mimic": 0},
            },
        },
    }
    spec = SimpleNamespace(
        label=label,
        ordinal=7,
        kind="mimic",
        artifact_root=artifact_root,
        summary_path=summary_path,
    )
    record = {
        "status": finalizer.DEFERRED_POSTGAME_STATUS,
        "result": None,
        "blocked_reason": error,
        "deferred_postgame_validation": deferred,
    }
    backlog = {"ordinal": 7, "label": label, **deferred}
    return spec, record, backlog


def test_deferred_media_receipt_stays_unscored_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    spec, record, backlog = _deferred_game(finalizer, tmp_path)
    summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    replay_record = summary["artifacts"]["replays"][0]
    replay_path = tmp_path / replay_record["path"]
    monkeypatch.setattr(finalizer, "_validate_players", lambda *_args: None)
    monkeypatch.setattr(finalizer, "_validate_model_identity", lambda *_args: None)
    monkeypatch.setattr(
        finalizer,
        "_selected_replay",
        lambda *_args: (replay_path, replay_record),
    )

    result = finalizer._deferred_media_result(spec, record, backlog)

    provenance = result["pending_fidelity_provenance"]
    assert provenance["status"] == "pending-fidelity-audit"
    assert provenance["gameplay_result_scored"] is False
    assert provenance["gameplay_gate_decision"] == "fail"
    assert set(provenance["failed_gate_checks"]) == (finalizer.MEDIA_SAFE_PENDING_GATE_CHECKS)
    assert result["replay"]["sha256"] == provenance["replay"]["sha256"]

    record["deferred_postgame_validation"]["saved_natural_end_evidence"]["summary"]["sha256"] = "0" * 64
    backlog = {
        "ordinal": spec.ordinal,
        "label": spec.label,
        **record["deferred_postgame_validation"],
    }
    with pytest.raises(suite.SuiteArtifactError, match="summary identity changed"):
        finalizer._deferred_media_result(spec, record, backlog)


def test_deferred_media_rejects_any_unrecognized_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    spec, record, backlog = _deferred_game(finalizer, tmp_path)
    monkeypatch.setattr(finalizer, "_validate_players", lambda *_args: None)
    monkeypatch.setattr(finalizer, "_validate_model_identity", lambda *_args: None)
    summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    summary["gate"]["checks"]["replay_identity_audit.exact_expected_ports"] = False
    spec.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_bytes = spec.summary_path.read_bytes()
    receipt = record["deferred_postgame_validation"]["saved_natural_end_evidence"]["summary"]
    receipt["sha256"] = hashlib.sha256(summary_bytes).hexdigest()
    receipt["byte_length"] = len(summary_bytes)
    backlog = {
        "ordinal": spec.ordinal,
        "label": spec.label,
        **record["deferred_postgame_validation"],
    }

    with pytest.raises(suite.SuiteArtifactError, match="outside the recognized"):
        finalizer._deferred_media_result(spec, record, backlog)


def test_global_preflight_requires_opt_in_and_terminal_hash_bound_deferred_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    complete = SimpleNamespace(label="complete", ordinal=1)
    deferred = SimpleNamespace(label="deferred", ordinal=2)
    schedule = (complete, deferred)
    expected_manifest = {"schedule_sha256": "a" * 64}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(expected_manifest), encoding="utf-8")
    deferred_receipt = {"status": "unresolved"}
    state = {
        "schedule_sha256": "a" * 64,
        "status": "incomplete-deferred-postgame-validation",
        "games": {
            "complete": {"status": "complete"},
            "deferred": {
                "status": finalizer.DEFERRED_POSTGAME_STATUS,
                "deferred_postgame_validation": deferred_receipt,
            },
        },
        finalizer.DEFERRED_POSTGAME_BACKLOG: {
            "deferred": {"ordinal": 2, "label": "deferred", **deferred_receipt}
        },
    }

    monkeypatch.setattr(finalizer, "SCHEDULE", schedule)
    monkeypatch.setattr(finalizer, "SCHEDULE_BY_LABEL", {game.label: game for game in schedule})
    monkeypatch.setattr(finalizer, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(finalizer, "schedule_manifest", lambda: expected_manifest)
    monkeypatch.setattr(finalizer, "load_state", lambda: state)

    def validate(game: Any) -> dict[str, Any]:
        if game.label == "deferred":
            raise suite.SuiteArtifactError("gate remains failed")
        return {"label": game.label, "replay": {"sha256": "1" * 64}}

    monkeypatch.setattr(finalizer, "validate_completed_game", validate)
    monkeypatch.setattr(
        finalizer,
        "_deferred_media_result",
        lambda *_args: {
            "replay": {"sha256": "2" * 64},
            "pending_fidelity_provenance": {"status": "pending-fidelity-audit"},
        },
    )

    with pytest.raises(suite.SuiteArtifactError, match="requires all 2"):
        finalizer._global_gameplay_preflight()

    results = finalizer._global_gameplay_preflight(allow_exact_deferred_for_media=True)
    assert set(results) == {"complete", "deferred"}
    assert results["deferred"]["pending_fidelity_provenance"]["status"] == ("pending-fidelity-audit")


def test_pending_fidelity_render_uses_preserved_summary_and_manifest_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    artifact_root = tmp_path / "artifacts" / "deferred"
    artifact_root.mkdir(parents=True)
    summary_path = artifact_root / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    summary_identity = finalizer._file_record(summary_path)
    provenance = {
        "schema_version": finalizer.PENDING_FIDELITY_MEDIA_SCHEMA,
        "status": "pending-fidelity-audit",
        "gameplay_result_scored": False,
        "gameplay_gate_decision": "fail",
        "summary": summary_identity,
    }
    gameplay_result = {
        "replay": {"path": "replay.slp", "sha256": "1" * 64, "byte_length": 1},
        "pending_fidelity_provenance": provenance,
    }
    spec = SimpleNamespace(
        label="deferred",
        artifact_root=artifact_root,
        summary_path=summary_path,
    )
    state = {"games": {"deferred": {"status": "pending", "attempts": []}}}
    validate_calls = 0

    def validate_bundle(*_args: Any) -> bool:
        nonlocal validate_calls
        validate_calls += 1
        return validate_calls > 1

    captured: dict[str, Any] = {}

    def finalize(_summary: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(finalizer, "_validate_bundle", validate_bundle)
    monkeypatch.setattr(finalizer, "_active_conflicts", lambda: [])
    monkeypatch.setattr(finalizer, "_save_media_state", lambda _state: None)
    monkeypatch.setattr(finalizer, "finalize_game_bundle", finalize)
    monkeypatch.setattr(
        finalizer,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(AssertionError("must stay unresolved")),
    )

    status = finalizer._render_one(
        state,
        spec,
        gameplay_result,
        config={},
        iso_path=tmp_path / "melee.iso",
        classifier=None,
    )

    assert status == "complete"
    assert captured["preserve_source_summary"] is True
    assert captured["gameplay_provenance"] == provenance
    assert summary_path.read_bytes() == b"{}\n"
    assert state["games"]["deferred"]["gameplay_provenance"] == provenance


def _cold_white_error(
    replay: Path,
    frames: list[int] | None = None,
    *,
    cocoa_wrapped: bool = True,
) -> str:
    observed_frames = list(range(17)) if frames is None else frames
    trace = "\n".join(f"[CURRENT_FRAME] {frame}" for frame in observed_frames)
    blank_message = "captured gameplay blank class W persisted 0.505008 seconds"
    failure = blank_message
    if cocoa_wrapped:
        failure = (
            f'Error Domain=melee-policy.replay-recorder Code=1 "{blank_message}" '
            f"UserInfo={{NSLocalizedDescription={blank_message}}}"
        )
    return (
        "RuntimeError: Slippi replay video recording failed: recording failed during "
        f"replay playback: {failure}; "
        f"Dolphin log: [FILE_PATH] {replay}\n"
        "[PLAYBACK_START_FRAME] 0\n"
        "[GAME_END_FRAME] 8927\n"
        "[PLAYBACK_END_FRAME] 8928\n"
        f"{trace}"
    )


def _cold_white_replay(
    finalizer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    replay = tmp_path / "artifacts" / "game" / "replays" / "scoring.slp"
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"hash-bound complete replay")
    return replay, finalizer._file_record(replay)


def test_early_cold_white_retry_requires_exact_startup_and_replay_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    replay, replay_record = _cold_white_replay(finalizer, tmp_path, monkeypatch)
    error = _cold_white_error(replay)

    evidence = finalizer._early_cold_white_retry_evidence(error, replay_record)
    assert evidence["blank_class"] == "W"
    assert evidence["first_current_frame"] == 0
    assert evidence["last_current_frame"] == 16
    assert evidence["current_frames_strictly_consecutive"] is True
    assert evidence["source_replay"] == replay_record
    assert evidence["capture_gates_relaxed"] is False
    assert len(evidence["signature_sha256"]) == 64

    assert (
        finalizer._early_cold_white_retry_evidence(
            error.replace("blank class W", "blank class K"), replay_record
        )
        is None
    )
    assert (
        finalizer._early_cold_white_retry_evidence(_cold_white_error(replay, [0, 1, 3]), replay_record)
        is None
    )
    assert (
        finalizer._early_cold_white_retry_evidence(_cold_white_error(replay, list(range(22))), replay_record)
        is None
    )
    assert (
        finalizer._early_cold_white_retry_evidence(
            error.replace(str(replay), str(tmp_path / "other.slp")), replay_record
        )
        is None
    )
    malformed_duplicate = (
        _cold_white_error(replay, cocoa_wrapped=False)
        + " captured gameplay blank class W persisted 0.505008 seconds"
    )
    assert finalizer._early_cold_white_retry_evidence(malformed_duplicate, replay_record) is None

    replay.write_bytes(b"changed")
    assert finalizer._early_cold_white_retry_evidence(error, replay_record) is None


def test_exact_retained_cold_white_receipt_gets_one_preserved_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    replay, replay_record = _cold_white_replay(finalizer, tmp_path, monkeypatch)
    error = _cold_white_error(replay)
    first_attempt = {
        "number": 1,
        "status": "blocked-nonretryable",
        "started_at_utc": "start",
        "finished_at_utc": "finish",
        "source_replay": replay_record,
        "error": error,
    }
    record = {
        "status": "blocked-nonretryable",
        "blocked_reason": error,
        "attempts": [first_attempt],
    }
    state = {"games": {"game": record}}
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(finalizer, "_save_media_state", lambda value: saved.append(value))

    assert (
        finalizer._reconcile_reviewed_early_cold_white_retry(state, record, {"replay": replay_record}) is True
    )
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["error"] == error
    assert record["attempts"][0]["status"] == "blocked-nonretryable"
    assert record["status"] == "retryable-reviewed-early-cold-white"
    assert record["reviewed_retry_eligibility"]["source_replay"] == replay_record
    assert saved == [state]


def test_repeated_early_cold_white_is_blocked_without_a_third_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    replay, replay_record = _cold_white_replay(finalizer, tmp_path, monkeypatch)
    error = _cold_white_error(replay)
    artifact_root = tmp_path / "artifacts" / "game"
    summary_path = artifact_root / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    spec = SimpleNamespace(label="game", artifact_root=artifact_root, summary_path=summary_path)
    record: dict[str, Any] = {"status": "pending", "attempts": []}
    state = {"games": {"game": record}}
    calls = 0

    def fail_render(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError(error)

    monkeypatch.setattr(finalizer, "_validate_bundle", lambda *_args: False)
    monkeypatch.setattr(finalizer, "_active_conflicts", lambda: [])
    monkeypatch.setattr(finalizer, "_save_media_state", lambda _state: None)
    monkeypatch.setattr(finalizer, "finalize_game_bundle", fail_render)
    monkeypatch.setattr(finalizer.time, "sleep", lambda _seconds: None)

    status = finalizer._render_one(
        state,
        spec,
        {"replay": replay_record},
        config={},
        iso_path=tmp_path / "melee.iso",
        classifier=None,
    )

    assert status == "blocked"
    assert calls == 2
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["status"] == "retryable-transient-capture-gap"
    assert record["attempts"][1]["status"] == "blocked-repeated-transient-class"
    assert record["attempts"][0]["failure_class"] == finalizer.EARLY_COLD_WHITE_RETRY_CLASS
    assert record["attempts"][1]["failure_class"] == finalizer.EARLY_COLD_WHITE_RETRY_CLASS


def test_early_cold_white_allows_only_one_retry_across_failure_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    replay, replay_record = _cold_white_replay(finalizer, tmp_path, monkeypatch)
    cold_error = _cold_white_error(replay)
    artifact_root = tmp_path / "artifacts" / "game"
    summary_path = artifact_root / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    spec = SimpleNamespace(label="game", artifact_root=artifact_root, summary_path=summary_path)
    record: dict[str, Any] = {"status": "pending", "attempts": []}
    state = {"games": {"game": record}}
    calls = 0

    def fail_render(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(cold_error)
        raise RuntimeError("a distinct classifier-approved capture delivery gap")

    retry_basis = next(iter(finalizer.ALLOWED_RETRY_BASES))
    monkeypatch.setattr(finalizer, "_validate_bundle", lambda *_args: False)
    monkeypatch.setattr(finalizer, "_active_conflicts", lambda: [])
    monkeypatch.setattr(finalizer, "_save_media_state", lambda _state: None)
    monkeypatch.setattr(finalizer, "finalize_game_bundle", fail_render)
    monkeypatch.setattr(finalizer.time, "sleep", lambda _seconds: None)

    status = finalizer._render_one(
        state,
        spec,
        {"replay": replay_record},
        config={},
        iso_path=tmp_path / "melee.iso",
        classifier=lambda _text: ("distinct-approved-class", retry_basis),
    )

    assert status == "blocked"
    assert calls == 2
    assert len(record["attempts"]) == 2
    assert record["attempts"][1]["status"] == ("blocked-reviewed-retry-budget-exhausted")
    assert record["attempts"][1]["failure_class"] == "distinct-approved-class"
