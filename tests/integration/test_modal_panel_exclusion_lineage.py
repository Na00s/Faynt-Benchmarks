"""Unscored exclusions preserve each published logical call's physical starts."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_cloud_queue as q
import report_modal_cloud_queue as report


def exclusion(distribution=(2, 1), *, cross_expiry=False, retained_bindings=True):
    original = {"artifact_labels": ["game-a1", "peer-a1"], "scope": {"selected_game_indices": [0]}}
    body = q.encoded(original)
    row = {
        "label": "game", "initial_status": "pending", "plan_sha256": q.digest(body),
        "initial_row": {"status": "pending", "result": None}, "phase": 0, "worker_slot": 0,
    }
    failures = []
    prior = []
    for group, count in enumerate(distribution):
        attempt = len(failures) + 1
        plan = copy.deepcopy(original)
        plan["artifact_labels"][0] = f"game-a{attempt}"
        plan_sha = q.digest(plan)
        expiry = 10000 * (group + 1) if cross_expiry else 10000
        binding = {"call_id": f"fc-group{group}", "expires_unix": expiry, "image_id": "im-original"}
        history = []
        for _ in range(count):
            number = len(failures) + 1
            start = {
                "call_id": binding["call_id"], "expires_unix": expiry, "started_unix": number * 10,
                "execution": {"execution_id": f"execution{number}", "task_id": f"ta-task{number}"},
            }
            manifest = {
                "status": "failed", "plan_sha256": plan_sha, "physical_number": number,
                "execution": copy.deepcopy(start["execution"]), "call_id": binding["call_id"],
                "files": [{"name": "receipt.json", "sha256": str(number) * 64, "bytes": 1}],
            }
            failure = {"manifest": manifest, "physical_start": start}
            if retained_bindings:
                failure["call_binding"] = copy.deepcopy(binding)
            failures.append(failure)
            history.append(copy.deepcopy(start))
        if group < len(distribution) - 1:
            proof = {"manifest": copy.deepcopy(manifest), "physical_history": history}
            if retained_bindings:
                proof["call_binding"] = copy.deepcopy(binding)
            prior.append(proof)
    retry = {
        "queue_plan_sha256": q.digest(body), "original_execution_plan_sha256": q.digest(body),
        "queue_plan_body": body.decode(), "reason": "reviewed-countdown-enet-disconnect",
        "attempt_number": attempt, "plan_sha256": plan_sha, "expires_unix": expiry,
        "prior_failures": prior,
    }
    if cross_expiry:
        retry["original_expires_unix"] = prior[-1]["call_binding"]["expires_unix"]
        retry["budget_resume"] = {
            "tranche_sha256": "a" * 64, "authorization_sha256": "b" * 64,
            "activation_sha256": "c" * 64,
        }
    snapshot = {"games": [row]}
    receipt = {
        "schema": q.SCHEMA + ".infrastructure-exclusion", "status": "quarantined", "label": "game",
        "queue_sha256": q.digest(snapshot), "queue_plan_sha256": q.digest(body),
        "user_authorization": "Test fixture authorization to record three failed starts without a score.",
        "reason": "infrastructure-failure-three-starts-exhausted", "retry_authorization": retry,
        "failed_attempts": failures,
    }
    return snapshot, row, receipt


@pytest.mark.parametrize("distribution", [(1, 2), (2, 1), (1, 1, 1)])
@pytest.mark.parametrize("cross_expiry", [False, True])
def test_three_start_distributions_keep_published_plans_calls_and_expiries(distribution, cross_expiry):
    snapshot, row, receipt = exclusion(distribution, cross_expiry=cross_expiry)
    retained = copy.deepcopy(receipt)
    assert q.validate_exclusion(row, receipt, receipt["queue_sha256"]) == retained
    assert receipt == retained
    projected = report.project(snapshot, {"game": receipt})
    assert projected["game"]["status"] == "quarantined"
    assert projected["game"]["result"] is None
    later = {"label": "later", "initial_status": "pending", "phase": 1, "worker_slot": 0}
    assert q.eligible({"games": [row, later]}, {"game": receipt}, strict_phase_order=True) == [later]


@pytest.mark.parametrize("distribution", [(1, 2), (2, 1), (1, 1, 1)])
def test_legacy_fixed_deadline_publications_retain_complete_history_without_binding_field(distribution):
    _, row, receipt = exclusion(distribution, retained_bindings=False)
    assert q.validate_exclusion(row, receipt, receipt["queue_sha256"]) == receipt


@pytest.mark.parametrize("mutation", [
    "prior-start", "prior-manifest", "prior-plan", "prior-call", "prior-image", "prior-expiry",
    "published-history", "latest-plan", "latest-call", "latest-expiry", "different-latest-calls",
    "different-latest-images",
    "duplicate-execution", "physical-number", "boolean-number", "missing-start", "fourth-start",
    "successful-prior", "successful-latest", "score", "result", "authorization", "queue", "label",
])
def test_changed_exclusion_provenance_and_score_fields_are_rejected(mutation):
    distribution = (1, 2) if mutation in {"different-latest-calls", "different-latest-images"} else (1, 1, 1)
    _, row, receipt = exclusion(distribution, cross_expiry=True)
    prior = receipt["failed_attempts"][0]
    latest = receipt["failed_attempts"][-1]
    if mutation == "prior-start":
        prior["physical_start"]["started_unix"] += 1
    if mutation == "prior-manifest":
        prior["manifest"]["files"][0]["sha256"] = "f" * 64
    if mutation == "prior-plan":
        prior["manifest"]["plan_sha256"] = "f" * 64
    if mutation == "prior-call":
        prior["call_binding"]["call_id"] = "fc-different"
    if mutation == "prior-image":
        prior["call_binding"]["image_id"] = "im-different"
    if mutation == "prior-expiry":
        prior["physical_start"]["expires_unix"] += 1
        prior["call_binding"]["expires_unix"] += 1
    if mutation == "published-history":
        receipt["retry_authorization"]["prior_failures"][0]["physical_history"][0]["started_unix"] += 1
    if mutation == "latest-plan":
        latest["manifest"]["plan_sha256"] = row["plan_sha256"]
    if mutation == "latest-call":
        latest["call_binding"]["call_id"] = "fc-different"
    if mutation == "latest-expiry":
        latest["physical_start"]["expires_unix"] += 1
        latest["call_binding"]["expires_unix"] += 1
    if mutation == "different-latest-calls":
        for item in (latest["manifest"], latest["physical_start"], latest["call_binding"]):
            item["call_id"] = "fc-other-latest-call"
    if mutation == "different-latest-images":
        latest["call_binding"]["image_id"] = "im-other-latest-image"
    if mutation == "duplicate-execution":
        latest["manifest"]["execution"] = copy.deepcopy(prior["manifest"]["execution"])
        latest["physical_start"]["execution"] = copy.deepcopy(prior["manifest"]["execution"])
    if mutation == "physical-number":
        latest["manifest"]["physical_number"] = 2
    if mutation == "boolean-number":
        prior["manifest"]["physical_number"] = True
        receipt["retry_authorization"]["prior_failures"][0]["manifest"]["physical_number"] = True
    if mutation == "missing-start":
        receipt["failed_attempts"].pop()
    if mutation == "fourth-start":
        receipt["failed_attempts"].append(copy.deepcopy(latest))
    if mutation == "successful-prior":
        prior["manifest"]["status"] = "full-policy-durable-game-passed"
        receipt["retry_authorization"]["prior_failures"][0]["manifest"]["status"] = prior["manifest"]["status"]
    if mutation == "successful-latest":
        latest["manifest"]["status"] = "full-policy-durable-game-passed"
    if mutation == "score":
        receipt["native_acceptance"] = {"win": True}
    if mutation == "result":
        receipt["result"] = None
    if mutation == "authorization":
        receipt["user_authorization"] = ""
    if mutation == "queue":
        receipt["queue_plan_sha256"] = "f" * 64
    if mutation == "label":
        receipt["label"] = "other"
    with pytest.raises(ValueError):
        q.validate_exclusion(row, receipt, receipt["queue_sha256"])


@pytest.mark.parametrize("distribution", [(1, 2), (2, 1), (1, 1, 1)])
def test_cross_expiry_legacy_publications_keep_each_original_history_deadline(distribution):
    _, row, receipt = exclusion(distribution, cross_expiry=True)
    for prior in receipt["retry_authorization"]["prior_failures"]:
        prior.pop("call_binding")
    assert q.validate_exclusion(row, receipt, receipt["queue_sha256"]) == receipt
    receipt["failed_attempts"][0]["call_binding"]["expires_unix"] = receipt["retry_authorization"]["expires_unix"]
    with pytest.raises(ValueError):
        q.validate_exclusion(row, receipt, receipt["queue_sha256"])


def test_legacy_publication_rejects_inconsistent_expiries_within_one_logical_call():
    _, row, receipt = exclusion((2, 1), cross_expiry=True)
    prior = receipt["retry_authorization"]["prior_failures"][0]
    prior.pop("call_binding")
    prior["physical_history"][0]["expires_unix"] += 1
    receipt["failed_attempts"][0]["physical_start"]["expires_unix"] += 1
    receipt["failed_attempts"][0]["call_binding"]["expires_unix"] += 1
    with pytest.raises(ValueError):
        q.validate_exclusion(row, receipt, receipt["queue_sha256"])


def test_missing_final_prior_manifest_cannot_be_replaced_by_history_only():
    _, row, receipt = exclusion()
    receipt["retry_authorization"]["prior_failures"][0].pop("manifest")
    with pytest.raises(ValueError, match="complete retained"):
        q.validate_exclusion(row, receipt, receipt["queue_sha256"])


def test_retained_ness_offline_candidate_requires_explicit_authorization():
    root = Path(__file__).resolve().parents[2] / "artifacts/integration/frisson_ai/faynt-d0-632-980-v1"
    if not (root / "recovery-audit/ness-exhausted-candidate.json").is_file():
        pytest.skip("requires the externally retained Ness recovery audit fixture")
    receipt = json.loads((root / "recovery-audit/ness-exhausted-candidate.json").read_bytes())
    snapshot = json.loads((root / "research/cloud-v1/metadata/snapshot.json").read_bytes())
    row = next(row for row in snapshot["games"] if row["label"] == receipt["label"])
    with pytest.raises(ValueError, match="explicit unscored"):
        q.validate_exclusion(row, receipt, receipt["queue_sha256"])
    # Exercise validation in memory only. The retained candidate stays unsigned.
    authorized_fixture = {**receipt, "user_authorization": "Test-only in-memory validation fixture."}
    assert q.validate_exclusion(row, authorized_fixture, receipt["queue_sha256"]) == authorized_fixture
    assert receipt["user_authorization"] == ""
