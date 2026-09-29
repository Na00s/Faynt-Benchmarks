"""The reviewed D0 Mewtwo bootstrap deadline failure retains its exact inputs."""
import copy
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_cloud_queue as q

ROOT = Path(__file__).resolve().parents[2] / (
    "artifacts/integration/frisson_ai/faynt-d0-632-980-v1/recovery-audit/fox24_mewtwo_fod"
)
REASON = "reviewed-zero-gameplay-runtime-bootstrap-deadline"
RAW_SHA = "7e231af64feb708b1ae44fc5476889844e91e79017e3e48573194042fa2097cf"
CANONICAL_SHA = "659be2fb03c1a9cb583bd0628e65727166f91eac229e9e2d0cb24efa1cff77b4"
FILES = {
    "receipt.json", "runtime/receipt.json", "runtime/logs/000.log", "runtime/logs/001.log",
    "supervisor/receipt.json", "supervisor/owner.log",
}
APT_PREFIX = ["apt-get", "-o", "Acquire::Retries=0", "-o", "Acquire::http::Timeout=30"]


@pytest.fixture
def failure():
    if not (ROOT / "manifest.json").is_file():
        pytest.skip("requires the exact externally retained bootstrap deadline fixture")
    raw = (ROOT / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    assert q.digest(raw) == RAW_SHA
    assert q.digest(manifest) == CANONICAL_SHA
    return manifest, ROOT


def test_classifier_key_is_the_canonical_digest_of_the_parsed_manifest():
    assert q.REVIEWED_FAILURE_RETRIES[CANONICAL_SHA] == REASON
    assert RAW_SHA not in q.REVIEWED_FAILURE_RETRIES
    assert list(q.REVIEWED_FAILURE_RETRIES.values()).count(REASON) == 1


def test_reviewed_bootstrap_deadline_has_exact_zero_gameplay_evidence(failure):
    manifest, root = failure
    rows = {row["name"]: row for row in manifest["files"]}
    assert set(rows) == FILES and len(manifest["files"]) == len(FILES)
    verification = json.loads((root / "REMOTE_VERIFICATION.json").read_bytes())
    remote = {row["name"]: row for row in verification["remote_files"]}
    assert set(remote) == FILES
    assert verification["verified_file_count"] == len(FILES)
    assert verification["remote_writes"] == 0
    assert verification["plan_sha256"] == manifest["plan_sha256"]
    assert verification["storage_path"] == manifest["storage_path"]
    assert verification["call_binding"]["call_id"] == manifest["call_id"]
    starts = verification["physical_starts"]
    assert len(starts) == manifest["scope"]["max_physical_executions"] == 3
    assert starts[0]["execution"] == manifest["execution"]
    assert starts[0]["call_id"] == manifest["call_id"]
    assert starts[1:] == [None, None]
    for name, row in rows.items():
        assert (remote[name]["bytes"], remote[name]["sha256"]) == (row["bytes"], row["sha256"])
        path = root / "files" / name
        if path.exists():
            body = path.read_bytes()
            assert (len(body), q.digest(body)) == (row["bytes"], row["sha256"])
    owner = json.loads((root / "files/receipt.json").read_bytes())
    runtime = json.loads((root / "files/runtime/receipt.json").read_bytes())
    supervisor = json.loads((root / "files/supervisor/receipt.json").read_bytes())
    assert manifest["status"] == "failed"
    assert manifest["physical_number"] == 1
    assert manifest["retryable_startup"] is False
    assert owner["plan_sha256"] == manifest["plan_sha256"]
    assert owner["scope"] == manifest["scope"]
    assert owner["status"] == "failed"
    assert owner["games"] == []
    assert owner["fresh_runtime_status"] == "failed"
    assert owner["error"] == "ValueError: fresh runtime failed"
    assert runtime["status"] == "failed"
    assert runtime["error"] == "TimeoutError: pilot absolute deadline exhausted"
    assert runtime["scope"]["gameplay_admitted"] is False
    assert all(runtime["scope"][name] == 0 for name in (
        "dolphin_launches", "policy_inference", "policy_instances", "rom_reads"
    ))
    commands = runtime["commands"]
    assert len(commands) == 2
    assert commands[0]["command"] == [*APT_PREFIX, "update"]
    assert commands[1]["command"][:8] == [*APT_PREFIX, "install", "-y", "--no-install-recommends"]
    for command in commands:
        assert command["exit_code"] == 0
        assert "error" not in command
    assert commands[0]["log_sha256"] == rows["runtime/logs/000.log"]["sha256"]
    assert commands[1]["log_sha256"] == rows["runtime/logs/001.log"]["original_sha256"]
    assert commands[1]["log_bytes"] == rows["runtime/logs/001.log"]["original_bytes"]
    assert commands[-1]["ended_unix"] <= runtime["expires_unix"] < runtime["ended_unix"]
    assert supervisor["exit_code"] == 0
    assert "error" not in supervisor
    assert supervisor["expires_unix"] == owner["expires_unix"]
    assert supervisor["command"][2:5] == ["--owner", "--sha", manifest["plan_sha256"]]
    assert float(supervisor["command"][6]) == owner["expires_unix"]
    assert supervisor["command"][8] == owner["runtime_image_id"]
    assert supervisor["log_bytes"] == 0 and supervisor["log_sha256"] == q.digest(b"")
    for proof in (*commands, supervisor):
        assert proof["terminated_owned_process_group"] is True
        assert proof["unreaped_leader_group_guard"] is True
    assert q.verified_failure_retry_reason(manifest, root) == REASON


def test_generic_classifiers_do_not_cover_bootstrap_deadline(failure):
    manifest, root = failure
    assert q.zero_runtime_timeout(manifest, root) is False
    assert q.zero_menu_disconnect(manifest, root) is False
    assert q.interrupted_native_owner(manifest, root) is False
    assert q.interrupted_empty_owner(manifest, root) is False
    assert q.exhausted_prebootstrap_budget(manifest, root) is False
    assert q.zero_gameplay_public_source_fetch(manifest, root) is False
    assert q.zero_gameplay_retry_reason(manifest, root) is None


@pytest.mark.parametrize("field", ["call_id", "plan_sha256", "execution", "scope", "status"])
def test_bootstrap_exception_rejects_changed_original_provenance(failure, field):
    manifest, root = failure
    manifest = copy.deepcopy(manifest)
    manifest[field] = {} if isinstance(manifest[field], dict) else "changed"
    assert q.verified_failure_retry_reason(manifest, root) is None


@pytest.mark.parametrize("name", sorted(FILES))
@pytest.mark.parametrize("field", ["sha256", "original_sha256"])
def test_bootstrap_exception_rejects_any_changed_file_hash(failure, name, field):
    manifest, root = failure
    manifest = copy.deepcopy(manifest)
    row = next(row for row in manifest["files"] if row["name"] == name)
    row[field] = "0" * 64
    assert q.digest(manifest) != CANONICAL_SHA
    assert q.verified_failure_retry_reason(manifest, root) is None


def test_bootstrap_exception_rejects_added_or_removed_files(failure):
    manifest, root = failure
    added = copy.deepcopy(manifest)
    added["files"].append({"name": "game-1/result.json", "bytes": 0, "sha256": q.digest(b"")})
    assert q.verified_failure_retry_reason(added, root) is None
    removed = copy.deepcopy(manifest)
    removed["files"] = [row for row in removed["files"] if row["name"] != "runtime/receipt.json"]
    assert q.verified_failure_retry_reason(removed, root) is None


def test_reviewed_bootstrap_deadline_failure_stays_unscored(failure):
    manifest, root = failure
    native = types.SimpleNamespace(
        MOUNT=root.parent, SUCCESS="full-policy-durable-game-passed", check_manifest=lambda *args: None
    )
    manifest = {**manifest, "storage_path": root.name}
    with pytest.raises(ValueError, match="without score admission"):
        q.native_acceptance(native, None, None, manifest)


def test_retry_execution_plan_accepts_reviewed_bootstrap_deadline_reason(failure):
    manifest, _ = failure
    original = {"artifact_labels": ["game-a1", "peer-a1"], "scope": {"selected_game_indices": [1]}, "seed": 31}
    body = q.encoded(original)
    row = {"label": "game", "plan_sha256": q.digest(body)}
    replacement = copy.deepcopy(original)
    replacement["artifact_labels"][1] = "game-a2"
    auth = {
        "queue_plan_sha256": row["plan_sha256"],
        "original_execution_plan_sha256": row["plan_sha256"],
        "queue_plan_body": body.decode(),
        "reason": REASON,
        "attempt_number": 2,
        "plan_sha256": q.digest(q.encoded(replacement)),
        "prior_failures": [{"physical_history": [{"execution": manifest["execution"]}]}],
    }
    plan, _, sha = q.retry_execution_plan(row, body, auth)
    assert plan == replacement and sha == auth["plan_sha256"]
    assert plan["artifact_labels"] == ["game-a1", "game-a2"] and plan["seed"] == 31
    with pytest.raises(ValueError, match="does not bind original frozen inputs"):
        q.retry_execution_plan(row, body, {**auth, "reason": "reviewed-zero-gameplay-runtime-bootstrap"})
    with pytest.raises(ValueError, match="three-attempt"):
        q.retry_execution_plan(row, body, {**auth, "attempt_number": 4})
