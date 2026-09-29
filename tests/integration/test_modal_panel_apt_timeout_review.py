"""Exact reviewed D0 apt failures retain their original inputs and start bound."""
import copy
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_cloud_queue as q

ROOT = Path(__file__).resolve().parents[2] / (
    "artifacts/integration/frisson_ai/faynt-d0-632-980-v1/recovery-audit"
)
CASES = {
    "fox21_ylink": "9c4e72445da8d5711332a6e8048ad1840d8fe8cb82f2400af04f27d9559ab102",
    "fox24_ganondorf": "c266a6fd5015d6dfe25816d4e1a4f9edc14502c69e6d5cc79026d773d793bdcc",
    "fox24_marth": "ef95e11f7b2906f2104c22efa99be3a2e9ac19baf41f5c2cb73f21cf1488a881",
    "falco21_bowser": "91465f64dab009f64207c40d8e5953afe2bd99fb8da05793dd56467e60a86592",
}


@pytest.fixture(params=CASES)
def failure(request):
    root = ROOT / request.param
    if not (root / "manifest.json").is_file():
        pytest.skip("requires the exact externally retained recovery audit fixture")
    raw = (root / "manifest.json").read_bytes()
    assert q.digest(raw) == CASES[request.param]
    return json.loads(raw), root


def test_reviewed_apt_timeout_has_exact_zero_gameplay_evidence(failure):
    manifest, root = failure
    names = {row["name"] for row in manifest["files"]}
    assert names == {
        "receipt.json", "runtime/receipt.json", "runtime/logs/000.log", "runtime/logs/001.log",
        "supervisor/receipt.json", "supervisor/owner.log",
    }
    verification = json.loads((root / "REMOTE_VERIFICATION.json").read_bytes())
    assert set(verification["verified_files"]) == names
    assert verification["manifest_sha256"] == q.digest((root / "manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / "files" / row["name"]
        if path.exists():
            body = path.read_bytes()
            assert (len(body), q.digest(body)) == (row["bytes"], row["sha256"])
    owner = json.loads((root / "files/receipt.json").read_bytes())
    runtime = json.loads((root / "files/runtime/receipt.json").read_bytes())
    supervisor = json.loads((root / "files/supervisor/receipt.json").read_bytes())
    assert manifest["physical_number"] == 1
    assert owner["plan_sha256"] == manifest["plan_sha256"]
    assert owner["scope"] == manifest["scope"]
    assert owner["games"] == []
    assert owner["error"] == "ValueError: fresh runtime failed"
    assert runtime["error"] == "TimeoutError: total build deadline exhausted"
    assert runtime["scope"]["gameplay_admitted"] is False
    assert all(runtime["scope"][name] == 0 for name in (
        "dolphin_launches", "policy_inference", "policy_instances", "rom_reads"
    ))
    assert len(runtime["commands"]) == 2
    assert runtime["commands"][0]["exit_code"] == 0
    last = runtime["commands"][-1]
    assert last["command"][:8] == [
        "apt-get", "-o", "Acquire::Retries=0", "-o", "Acquire::http::Timeout=30",
        "install", "-y", "--no-install-recommends",
    ]
    assert last["exit_code"] == -15
    assert last["error"] == runtime["error"]
    for proof in (last, supervisor):
        assert proof["terminated_owned_process_group"] is True
        assert proof["unreaped_leader_group_guard"] is True
    assert q.zero_runtime_timeout(manifest, root) is False
    assert q.verified_failure_retry_reason(manifest, root) == (
        "reviewed-zero-gameplay-dependency-acquisition-failure"
    )


@pytest.mark.parametrize("field", ["call_id", "plan_sha256", "execution", "files", "scope"])
def test_apt_exception_rejects_changed_original_provenance(failure, field):
    manifest, root = failure
    manifest = copy.deepcopy(manifest)
    manifest[field] = {} if isinstance(manifest[field], dict) else [] if field == "files" else "changed"
    assert q.verified_failure_retry_reason(manifest, root) is None


def test_reviewed_apt_failure_stays_unscored(failure):
    manifest, root = failure
    native = types.SimpleNamespace(
        MOUNT=root.parent, SUCCESS="full-policy-durable-game-passed", check_manifest=lambda *args: None
    )
    manifest = {**manifest, "storage_path": root.name}
    with pytest.raises(ValueError, match="without score admission"):
        q.native_acceptance(native, None, None, manifest)


def test_ness_third_start_retains_prebootstrap_failure_and_exhausted_bound():
    root = ROOT / "fox24_ness_a3"
    if not (root / "manifest.json").is_file():
        pytest.skip("requires the exact externally retained Ness recovery fixture")
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert manifest["physical_number"] == 3
    assert q.verified_failure_retry_reason(manifest, root) == "verified-prebootstrap-budget-exhaustion"
    plan = {"artifact_labels": ["game-a1"], "scope": {"selected_game_indices": [0]}}
    body = q.encoded(plan)
    row = {"label": "game", "plan_sha256": q.digest(body), "initial_status": "pending"}
    with pytest.raises(ValueError, match="three-attempt"):
        q.authorize_failure_retry(
            row, body, [{"manifest": manifest, "physical_history": [{}, {}, {}]}],
            reason="verified-prebootstrap-budget-exhaustion", expires_unix=100, now=200,
            session_seconds=1000, resume_tranche={"activation": {"deadline_unix": 5000}},
        )
