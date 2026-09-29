"""Late D0 starts defer while active games and saved-result recovery continue."""

# Imported pytest fixtures intentionally share names with fixture parameters.
# ruff: noqa: F811
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_cloud_app as app
from test_modal_panel_cloud_queue import dispatcher  # noqa: F401


@pytest.fixture
def d0(monkeypatch):
    monkeypatch.setenv("PANEL_DURABLE_MODULE", "faynt_d0_durable_game")


@pytest.mark.parametrize("dispatch,minimum", [(False, 2130), (True, 2430)])
def test_precise_admission_boundary(d0, dispatch, minimum):
    for remaining in (0, minimum - 0.1, minimum):
        result = app.late_admission("game", 1000 + remaining, now=1000, dispatch=dispatch)
        assert result["status"] == "budget-admission-deferred"
        assert result["minimum_remaining_seconds"] == minimum
        assert result["score_admitted"] is False
    assert app.late_admission("game", 1000 + minimum + 0.1, now=1000, dispatch=dispatch) is None


def test_legacy_admission_unchanged(monkeypatch):
    monkeypatch.setenv("PANEL_DURABLE_MODULE", "modal_panel_durable_game")
    assert app.late_admission("game", 1000, now=1000, dispatch=True) is None


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), True])
def test_d0_admission_requires_finite_deadline(d0, expiry):
    with pytest.raises(ValueError, match="finite"):
        app.late_admission("game", expiry, now=1000)


def test_late_first_dispatch_preserves_pending_without_owner_or_hold(dispatcher, d0, monkeypatch):
    worker, store, _, launched, sha, _ = dispatcher
    monkeypatch.setattr(worker, "frozen_game_session_seconds", lambda row: 2430)
    result = worker.dispatch_cloud()
    assert launched == []
    assert len(result["admission_deferred"]) == 2
    assert result["errors"] == []
    assert result["held"] == 0
    mutation_prefixes = ("/dispatch/", "/owner/", "/result/", "/hold/")
    assert not any(key.startswith(sha + kind) for key in store for kind in mutation_prefixes)
    assert store[sha + "/control"]["enabled"] is True


def test_late_dispatch_keeps_observing_active_call(dispatcher, d0, monkeypatch):
    worker, store, calls, launched, sha, _ = dispatcher
    store[sha + "/dispatch/a"] = {"call_id": "fc-active", "expires_unix": 9000}
    store[sha + "/owner/a"] = {"call_id": "fc-active"}
    calls["fc-active"] = None
    monkeypatch.setattr(worker, "frozen_game_session_seconds", lambda row: 2430)
    result = worker.dispatch_cloud()
    assert result["waiting"] == 1
    assert launched == []
    assert result["errors"] == []


def test_late_failed_call_still_gets_read_only_recovery(dispatcher, d0, monkeypatch):
    worker, store, calls, launched, sha, _ = dispatcher
    store[sha + "/dispatch/a"] = {"call_id": "fc-old", "expires_unix": 9000}
    store[sha + "/owner/a"] = {"call_id": "fc-old"}
    calls["fc-old"] = RuntimeError("return interrupted")
    monkeypatch.setattr(worker, "frozen_game_session_seconds", lambda row: 2430)
    result = worker.dispatch_cloud()
    assert len(launched) == 1
    assert launched[0][1][1]["recover_only"] is True
    assert result["held"] == 0


@pytest.mark.parametrize("recovery", [False, True])
def test_deferred_response_is_settled_without_new_hold_or_recovery(dispatcher, d0, monkeypatch, recovery):
    worker, store, calls, launched, sha, _ = dispatcher
    store[sha + "/dispatch/a"] = {"call_id": "fc-old", "expires_unix": 9000}
    store[sha + "/owner/a"] = {"call_id": "fc-old"}
    deferred = {"status": "budget-admission-deferred", "label": "a"}
    if recovery:
        calls["fc-old"] = RuntimeError("return interrupted")
        store[sha + "/recovery/a"] = {"call_id": "fc-recovery", "submissions": 1}
        calls["fc-recovery"] = deferred
    else:
        calls["fc-old"] = deferred
    monkeypatch.setattr(worker, "frozen_game_session_seconds", lambda row: 2430)
    for _ in range(2):
        result = worker.dispatch_cloud()
        assert deferred in result["admission_deferred"]
        assert result["held"] == 0
        assert result["errors"] == []
    assert launched == []


@pytest.mark.parametrize("saved", [None, {"status": "full-policy-durable-game-passed"}])
def test_late_worker_recovers_saved_result_without_binding_or_physical_start(d0, monkeypatch, saved):
    monkeypatch.setattr(app.time, "time", lambda: 1000)
    calls = []
    native = types.SimpleNamespace(
        recover_saved=lambda *args: saved,
        remote_durable=lambda *args: calls.append("physical"),
    )
    manifest, deferred = app.admitted_native_result(
        native, None, None, "sha", None, 3130, "image", "game", lambda: calls.append("owner")
    )
    assert manifest is saved
    assert calls == []
    assert (deferred is None) == (saved is not None)


def test_worker_binds_only_after_admission_and_before_physical_start(d0, monkeypatch):
    monkeypatch.setattr(app.time, "time", lambda: 1000)
    calls = []
    native = types.SimpleNamespace(remote_durable=lambda *args: calls.append("physical"))
    app.admitted_native_result(
        native, None, None, "sha", None, 3131, "image", "game", lambda: (calls.append("owner") or True)
    )
    assert calls == ["owner", "physical"]


def test_existing_owner_prevents_duplicate_physical_start(d0, monkeypatch):
    monkeypatch.setattr(app.time, "time", lambda: 1000)
    native = types.SimpleNamespace()
    manifest, result = app.admitted_native_result(
        native, None, None, "sha", None, 4000, "image", "game", lambda: False
    )
    assert manifest is None
    assert result["status"] == "duplicate-dispatch-suppressed"


def test_new_retry_uses_unchanged_tranche_deadline_and_dispatch_margin(d0):
    tranche = {"activation": {"deadline_unix": 3430}}
    result = app.retry_late_admission("game", 1500, 9090, tranche, now=1000)
    assert result["expires_unix"] == 3430
    assert tranche == {"activation": {"deadline_unix": 3430}}
    assert app.retry_late_admission("game", 4000, 9090, tranche, now=1000) is None


def test_existing_retry_authorization_retains_idempotent_reuse(d0):
    existing = {"expires_unix": 500}
    assert app.retry_late_admission("game", 500, 9090, None, existing=existing, now=1000) is None
    assert existing == {"expires_unix": 500}
