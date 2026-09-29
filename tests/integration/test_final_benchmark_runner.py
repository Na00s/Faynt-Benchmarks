from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_final_winner_benchmark.py"


def _load_runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "test_run_final_winner_benchmark",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _state_for(*specs: Any) -> dict[str, Any]:
    return {
        "status": "pending",
        "current_label": None,
        "games": {
            spec.label: {
                "status": "pending",
                "attempts_started": 0,
                "result": None,
            }
            for spec in specs
        },
    }


def _saved_natural_end_game(
    tmp_path: Path,
    *,
    label: str = "game-001",
    ordinal: int = 1,
) -> Any:
    artifact_root = tmp_path / "artifacts" / label
    replay_path = artifact_root / "replays" / "game.slp"
    replay_path.parent.mkdir(parents=True)
    replay_bytes = b"saved natural-end scoring replay"
    replay_path.write_bytes(replay_bytes)
    summary_path = artifact_root / "summary.json"
    summary = {
        "execution": {
            "game_end_observed": True,
            "termination": "natural-game-end",
            "winner": "player-2",
            "last_stocks": {"player-1": 0, "player-2": 2},
        },
        "artifacts": {
            "replays": [
                {
                    "path": str(replay_path),
                    "sha256": hashlib.sha256(replay_bytes).hexdigest(),
                    "byte_length": len(replay_bytes),
                    "tournament_result_replay": True,
                    "validation": {
                        "libmelee_parseable": True,
                        "parsed_frames_strictly_consecutive": True,
                        "first_parsed_frame": -123,
                        "last_parsed_frame": 9123,
                        "required_trace_coverage": {"complete": True},
                    },
                }
            ]
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return SimpleNamespace(
        label=label,
        ordinal=ordinal,
        artifact_root=artifact_root,
        summary_path=summary_path,
    )


def test_deferred_postgame_mode_is_an_explicit_opt_in() -> None:
    runner = _load_runner()
    assert runner._parser().parse_args([]).defer_postgame_validation_failures is False
    assert (
        runner._parser()
        .parse_args(["--defer-postgame-validation-failures"])
        .defer_postgame_validation_failures
        is True
    )


def test_default_execution_order_finishes_75m_before_10m() -> None:
    runner = _load_runner()
    selected = runner._selected_specs(SimpleNamespace(label=None, group=None))

    assert len(selected) == 309
    priorities = [
        0 if spec.kind == "head-to-head" else 1 if spec.frisson_profile == "75m" else 2
        for spec in selected
    ]
    assert priorities == sorted(priorities)
    assert sum(spec.kind == "head-to-head" for spec in selected) == 5
    assert sum(
        spec.kind != "head-to-head" and spec.frisson_profile == "75m"
        for spec in selected
    ) == 152
    assert sum(
        spec.kind != "head-to-head" and spec.frisson_profile == "10m"
        for spec in selected
    ) == 152


def test_explicit_mixed_profile_selection_keeps_75m_first() -> None:
    runner = _load_runner()
    seventy_five = next(
        spec
        for spec in runner.SCHEDULE
        if spec.kind != "head-to-head" and spec.frisson_profile == "75m"
    )
    ten = next(
        spec
        for spec in runner.SCHEDULE
        if spec.kind != "head-to-head" and spec.frisson_profile == "10m"
    )
    selected = runner._selected_specs(
        SimpleNamespace(label=[ten.label, seventy_five.label], group=None)
    )

    assert [spec.label for spec in selected] == [seventy_five.label, ten.label]


def test_replay_identity_capture_rejects_a_file_that_changes_while_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    replay_path = tmp_path / "game.slp"
    replay_path.write_bytes(b"initial replay")
    real_fstat = runner.os.fstat
    calls = 0

    def mutate_after_first_fstat(descriptor: int) -> Any:
        nonlocal calls
        snapshot = real_fstat(descriptor)
        calls += 1
        if calls == 1:
            replay_path.write_bytes(b"changed replay with a different byte length")
        return snapshot

    monkeypatch.setattr(runner.os, "fstat", mutate_after_first_fstat)

    with pytest.raises(runner.SuiteArtifactError, match="file changed while hashing"):
        runner._capture_file_identity(replay_path)


def test_opt_in_does_not_bypass_frozen_input_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    launched = False

    def reject_frozen_inputs(*, include_opponents: bool) -> dict[str, Any]:
        assert include_opponents is True
        raise runner.SuiteArtifactError("frozen checkpoint identity changed")

    def unexpected_launch(*args: Any, **kwargs: Any) -> str:
        nonlocal launched
        launched = True
        return "complete"

    monkeypatch.setattr(runner, "SUITE_DIRECTORY", tmp_path / "suite")
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "suite" / "runner.lock")
    monkeypatch.setattr(runner, "GLOBAL_LOCK_PATH", tmp_path / "global.lock")
    monkeypatch.setattr(runner, "V3_LOCK_PATH", tmp_path / "v3" / "runner.lock")
    monkeypatch.setattr(runner, "V4_LOCK_PATH", tmp_path / "v4" / "runner.lock")
    monkeypatch.setattr(
        runner,
        "_write_or_verify_manifest",
        lambda: {"schedule_sha256": "frozen-schedule"},
    )
    monkeypatch.setattr(runner, "verify_frozen_inputs", reject_frozen_inputs)
    monkeypatch.setattr(runner, "_launch_game", unexpected_launch)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [str(SCRIPT), "--defer-postgame-validation-failures"],
    )

    with pytest.raises(runner.SuiteArtifactError, match="frozen checkpoint identity changed"):
        runner.main()

    assert launched is False


def test_v4_runner_refuses_to_overlap_a_held_v3_orchestrator_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    suite_directory = tmp_path / "v4"
    legacy_lock_path = tmp_path / "v3" / ".suite.lock"
    legacy_lock_path.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "SUITE_DIRECTORY", suite_directory)
    monkeypatch.setattr(runner, "GLOBAL_LOCK_PATH", tmp_path / "global.lock")
    monkeypatch.setattr(runner, "V3_LOCK_PATH", legacy_lock_path)
    monkeypatch.setattr(runner, "LOCK_PATH", suite_directory / ".suite.lock")
    monkeypatch.setattr(runner.sys, "argv", [str(SCRIPT), "--status-only"])

    with legacy_lock_path.open("a+") as legacy_lock:
        fcntl.flock(legacy_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="cross-version lock"):
            runner.main()


def test_launch_defers_saved_natural_end_validation_failure_without_scoring_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    spec = _saved_natural_end_game(tmp_path)
    state = _state_for(spec)
    failure = runner.SuiteArtifactError("postgame fidelity gate failed")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE", tmp_path / "src")
    monkeypatch.setattr(runner, "SCHEDULE", (spec,))
    monkeypatch.setattr(runner, "command_for_game", lambda _spec, _python: ["launcher"])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(
        runner,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(runner, "save_state", lambda _state: None)
    monkeypatch.setattr(runner, "utc_now", lambda: "2026-09-02T17:00:00Z")

    result = runner._launch_game(
        state,
        spec,
        defer_postgame_validation_failures=True,
    )

    record = state["games"][spec.label]
    assert result == "deferred"
    assert record["status"] == runner.DEFERRED_POSTGAME_STATUS
    assert record["result"] is None
    assert record["launcher_returncode"] == 1
    deferred = record["deferred_postgame_validation"]
    assert deferred["status"] == "unresolved"
    assert deferred["validation_error"].endswith("postgame fidelity gate failed")
    assert deferred["saved_natural_end_evidence"]["replay"]["last_parsed_frame"] == 9123
    assert state["deferred_postgame_validation_backlog"][spec.label]["ordinal"] == 1
    assert state["status"] == "incomplete-deferred-postgame-validation"
    assert state["current_label"] is None


def test_launch_still_stops_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    spec = _saved_natural_end_game(tmp_path)
    state = _state_for(spec)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE", tmp_path / "src")
    monkeypatch.setattr(runner, "SCHEDULE", (spec,))
    monkeypatch.setattr(runner, "command_for_game", lambda _spec, _python: ["launcher"])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(
        runner,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(runner.SuiteArtifactError("postgame fidelity gate failed")),
    )
    monkeypatch.setattr(runner, "save_state", lambda _state: None)

    with pytest.raises(runner.SuiteArtifactError, match="postgame fidelity gate failed"):
        runner._launch_game(state, spec)

    record = state["games"][spec.label]
    assert record["status"] == "blocked-launch-or-artifact-failure"
    assert record["result"] is None
    assert "deferred_postgame_validation_backlog" not in state


def test_opt_in_still_stops_when_natural_end_or_complete_replay_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    spec = _saved_natural_end_game(tmp_path)
    summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    summary["execution"]["game_end_observed"] = False
    spec.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    state = _state_for(spec)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE", tmp_path / "src")
    monkeypatch.setattr(runner, "SCHEDULE", (spec,))
    monkeypatch.setattr(runner, "command_for_game", lambda _spec, _python: ["launcher"])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(
        runner,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(runner.SuiteArtifactError("postgame fidelity gate failed")),
    )
    monkeypatch.setattr(runner, "save_state", lambda _state: None)

    with pytest.raises(runner.SuiteArtifactError, match="postgame fidelity gate failed"):
        runner._launch_game(
            state,
            spec,
            defer_postgame_validation_failures=True,
        )

    assert state["games"][spec.label]["status"] == ("blocked-launch-or-artifact-failure")


def test_reconciliation_keeps_deferred_games_unscored_and_suite_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    deferred_spec = _saved_natural_end_game(tmp_path)
    pending_spec = SimpleNamespace(
        label="game-002",
        ordinal=2,
        artifact_root=tmp_path / "artifacts" / "game-002",
        summary_path=tmp_path / "artifacts" / "game-002" / "summary.json",
    )
    state = _state_for(deferred_spec, pending_spec)
    state["games"][deferred_spec.label]["status"] = runner.DEFERRED_POSTGAME_STATUS
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SCHEDULE", (deferred_spec, pending_spec))
    monkeypatch.setattr(
        runner,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(runner.SuiteArtifactError("postgame fidelity gate failed")),
    )
    monkeypatch.setattr(runner, "utc_now", lambda: "2026-09-02T17:00:00Z")

    counts = runner._reconcile_all(state)

    assert counts == {"deferred": 1, "pending": 1}
    assert state["status"] == "incomplete-deferred-postgame-validation"
    assert state["games"][deferred_spec.label]["result"] is None
    assert deferred_spec.label in state["deferred_postgame_validation_backlog"]


def test_selected_loop_skips_deferred_game_and_launches_later_pending_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    deferred_spec = SimpleNamespace(label="game-001", ordinal=1)
    pending_spec = SimpleNamespace(label="game-002", ordinal=2)
    state = _state_for(deferred_spec, pending_spec)
    arguments = argparse.Namespace(
        defer_postgame_validation_failures=True,
        maximum_games=None,
    )
    launched: list[str] = []
    statuses = iter(("deferred", "pending"))
    monkeypatch.setattr(runner, "SCHEDULE", (deferred_spec, pending_spec))
    monkeypatch.setattr(runner, "_reconcile_one", lambda *args, **kwargs: next(statuses))

    def record_launch(_state: dict[str, Any], spec: Any, **kwargs: Any) -> str:
        launched.append(spec.label)
        return "complete"

    monkeypatch.setattr(
        runner,
        "_launch_game",
        record_launch,
    )

    count = runner._run_selected_games(
        state,
        [deferred_spec, pending_spec],
        arguments,
    )

    assert count == 1
    assert launched == [pending_spec.label]


def test_restart_round_trip_skips_deferred_game_and_clears_stale_current_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    deferred_spec = _saved_natural_end_game(tmp_path)
    pending_spec = SimpleNamespace(
        label="game-002",
        ordinal=2,
        artifact_root=tmp_path / "artifacts" / "game-002",
        summary_path=tmp_path / "artifacts" / "game-002" / "summary.json",
    )
    state = _state_for(deferred_spec, pending_spec)
    state["status"] = "running"
    state["current_label"] = deferred_spec.label
    state["games"][deferred_spec.label]["status"] = runner.DEFERRED_POSTGAME_STATUS
    failure = runner.SuiteArtifactError("postgame fidelity gate failed")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SCHEDULE", (deferred_spec, pending_spec))
    monkeypatch.setattr(
        runner,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(runner, "utc_now", lambda: "2026-09-02T17:00:00Z")

    counts = runner._reconcile_all(
        state,
        defer_postgame_validation_failures=True,
    )
    restarted = json.loads(json.dumps(state))

    assert counts == {"deferred": 1, "pending": 1}
    assert restarted["current_label"] is None
    assert restarted["games"][deferred_spec.label]["result"] is None

    launched: list[str] = []

    def record_launch(
        resumed_state: dict[str, Any],
        spec: Any,
        **_kwargs: Any,
    ) -> str:
        launched.append(spec.label)
        resumed_state["games"][spec.label]["status"] = "complete"
        return "complete"

    monkeypatch.setattr(runner, "_launch_game", record_launch)
    arguments = argparse.Namespace(
        defer_postgame_validation_failures=True,
        maximum_games=None,
    )

    launched_count = runner._run_selected_games(
        restarted,
        [deferred_spec, pending_spec],
        arguments,
    )

    assert launched_count == 1
    assert launched == [pending_spec.label]
    assert restarted["games"][deferred_spec.label]["result"] is None


def test_repaired_deferred_game_is_promoted_and_removed_from_backlog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    spec = _saved_natural_end_game(tmp_path)
    state = _state_for(spec)
    state["current_label"] = spec.label
    failure = runner.SuiteArtifactError("postgame fidelity gate failed")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SCHEDULE", (spec,))
    monkeypatch.setattr(
        runner,
        "validate_completed_game",
        lambda _spec: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(runner, "utc_now", lambda: "2026-09-02T17:00:00Z")

    assert (
        runner._reconcile_one(
            state,
            spec,
            defer_postgame_validation_failures=True,
        )
        == "deferred"
    )
    assert spec.label in state[runner.DEFERRED_POSTGAME_BACKLOG]

    repaired_result = {
        "winner": "player-2",
        "stocks": {"player-1": 0, "player-2": 2},
        "gate": "pass",
    }
    monkeypatch.setattr(runner, "validate_completed_game", lambda _spec: repaired_result)

    counts = runner._reconcile_all(
        state,
        defer_postgame_validation_failures=True,
    )

    record = state["games"][spec.label]
    assert counts == {"complete": 1}
    assert state["status"] == "gameplay-complete"
    assert state["current_label"] is None
    assert record["status"] == "complete"
    assert record["result"] == repaired_result
    assert "blocked_reason" not in record
    assert "deferred_postgame_validation" not in record
    assert spec.label not in state[runner.DEFERRED_POSTGAME_BACKLOG]


def test_reconciliation_preserves_completed_game_and_only_launches_pending_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    complete_spec = _saved_natural_end_game(tmp_path)
    pending_spec = SimpleNamespace(
        label="game-002",
        ordinal=2,
        artifact_root=tmp_path / "artifacts" / "game-002",
        summary_path=tmp_path / "artifacts" / "game-002" / "summary.json",
    )
    state = _state_for(complete_spec, pending_spec)
    completed_record = state["games"][complete_spec.label]
    completed_record.update(
        {
            "status": "complete",
            "attempts_started": 1,
            "completed_at_utc": "2026-09-02T16:00:00Z",
            "command": ["original-launcher"],
            "result": {"winner": "player-2"},
        }
    )
    validated_result = {
        "winner": "player-2",
        "stocks": {"player-1": 0, "player-2": 2},
        "gate": "pass",
    }
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SCHEDULE", (complete_spec, pending_spec))
    monkeypatch.setattr(runner, "validate_completed_game", lambda _spec: validated_result)
    monkeypatch.setattr(runner, "utc_now", lambda: "2026-09-02T17:00:00Z")

    counts = runner._reconcile_all(
        state,
        defer_postgame_validation_failures=True,
    )

    assert counts == {"complete": 1, "pending": 1}
    assert completed_record["status"] == "complete"
    assert completed_record["attempts_started"] == 1
    assert completed_record["completed_at_utc"] == "2026-09-02T16:00:00Z"
    assert completed_record["command"] == ["original-launcher"]
    assert completed_record["result"] == validated_result

    launched: list[str] = []

    def record_launch(_state: dict[str, Any], spec: Any, **_kwargs: Any) -> str:
        launched.append(spec.label)
        return "complete"

    monkeypatch.setattr(runner, "_launch_game", record_launch)
    arguments = argparse.Namespace(
        defer_postgame_validation_failures=True,
        maximum_games=None,
    )

    assert (
        runner._run_selected_games(
            state,
            [complete_spec, pending_spec],
            arguments,
        )
        == 1
    )
    assert launched == [pending_spec.label]
    assert completed_record["attempts_started"] == 1
