"""Local-only queue packaging, paired assignment and immutable file identities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

import prepare_faynt_d0_cloud as prepare


@pytest.fixture(scope="module")
def panel():
    if not prepare.PANEL.is_file():
        pytest.skip("fresh local panel manifest unavailable")
    return json.loads(prepare.PANEL.read_bytes())


def test_all_fresh_pairs_use_original_phases_and_round_robin_slots(panel):
    assignments, phases = prepare.pair_assignments(panel)
    assert phases == [122, 267, 267, 122, 267, 267]
    assert len(assignments) == 2624
    assert len({pair["pair_id"] for pair in assignments.values()}) == 1312
    counters = [0] * 6
    seen = set()
    for game in panel["games"]:
        pair = assignments[game["label"]]
        if pair["pair_id"] in seen:
            continue
        seen.add(pair["pair_id"])
        assert pair["worker_slot"] == counters[pair["phase"]] % 64
        counters[pair["phase"]] += 1
        assert assignments[pair["labels"][0]] == assignments[pair["labels"][1]]
    assert counters == phases


@pytest.mark.parametrize("change", ["duplicate", "stage", "port", "video", "ordinal", "old-label"])
def test_assignment_rejects_changed_or_reused_game_scope(panel, change):
    changed = copy.deepcopy(panel)
    first, second = changed["games"][:2]
    if change == "duplicate": second["label"] = first["label"]
    if change == "stage": second["stage"] = "another-stage"
    if change == "port": second["frisson_port"] = first["frisson_port"]
    if change == "video": first["save_video"] = True
    if change == "ordinal": first["ordinal"] = 99
    if change == "old-label": first["label"] = "sp21-old"
    with pytest.raises(ValueError):
        prepare.pair_assignments(changed)


def test_file_hashes_are_cached_across_all_game_plans(tmp_path, monkeypatch):
    path = tmp_path / "large-checkpoint.pt"
    path.write_bytes(b"checkpoint" * 100)
    actual = hashlib.file_digest
    calls = []
    def counted(stream, algorithm):
        calls.append(stream.name)
        return actual(stream, algorithm)
    monkeypatch.setattr(hashlib, "file_digest", counted)
    records = prepare.FileRecords()
    first = records.record(path, "/opt/d0-root/weights.pt")
    for _ in range(2624):
        assert records.record(path, "/opt/d0-root/weights.pt", expected=first) == first
    records.verify_unchanged()
    assert calls == [str(path)]
    assert len(records.blobs) == 1


def test_file_record_rejects_changes_and_wrong_reference_identity(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"frozen")
    records = prepare.FileRecords()
    record = records.record(path, "/opt/runtime-root/source.py")
    with pytest.raises(ValueError, match="frozen input identity"):
        records.record(path, record["remote"], expected={**record, "sha256": "0" * 64})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during preparation"):
        records.record(path, record["remote"])
    with pytest.raises(ValueError, match="changed during preparation"):
        records.verify_unchanged()


def test_file_record_rejects_symlinks_and_caps(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_bytes(b"source")
    link = tmp_path / "link.py"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular input"):
        prepare.FileRecords().record(link, "/opt/runtime-root/source.py")
    monkeypatch.setattr(prepare, "FILE_CAP", 2)
    with pytest.raises(ValueError, match="byte cap"):
        prepare.FileRecords().record(path, "/opt/runtime-root/source.py")


def test_old_and_new_policy_use_distinct_local_and_remote_paths(tmp_path):
    old = tmp_path / "reference-bootstrap/frisson_policy.py"
    old.parent.mkdir()
    old.write_bytes(b"frozen reference")
    new = tmp_path / "src/frisson_policy.py"
    new.parent.mkdir()
    new.write_bytes(b"new loader")
    records = prepare.FileRecords()
    before = records.record(old, prepare.REFERENCE_POLICY_REMOTE)
    after = records.record(new, "/opt/d0-root/src/melee_policy/integration/frisson_policy.py")
    assert before["local"] != after["local"]
    assert before["remote"] != after["remote"]
    assert before["sha256"] != after["sha256"]
    assert len(records.blobs) == 2


def test_authorization_rebinds_only_scope_resources_models_and_panel():
    original = {"scope": {"old": True}, "limits": {"nonpreemptible": True},
                "bindings": {"checkpoints": {"old": {}}, "panel_manifest_sha256": "old",
                             "tape_plan_sha256": "retained", "proof": "unchanged"}}
    before = copy.deepcopy(original)
    limits = {"nonpreemptible": False, "cpu": [4, 4], "memory_mib": [16384, 16384]}
    bound = SimpleNamespace(fixed_scope=lambda: {"selected_game_indices": [0]},
        limits=lambda: limits, CHECKPOINTS={"new": {"sha256": "new"}})
    value = prepare.authorization(original, bound, {"prior_tape_plan_sha256": "retained"})
    assert original == before
    assert value["scope"] == bound.fixed_scope()
    assert value["bindings"]["proof"] == "unchanged"
    assert value["bindings"]["checkpoints"] == bound.CHECKPOINTS
    assert value["bindings"]["panel_manifest_sha256"] == prepare.PANEL_SHA
    limits["nonpreemptible"] = True
    with pytest.raises(ValueError, match="standard preemptible"):
        prepare.authorization(original, bound, {"prior_tape_plan_sha256": "retained"})


def test_metadata_archive_has_only_regular_relative_entries_and_stable_bytes(tmp_path):
    meta = tmp_path / "metadata"
    (meta / "plans").mkdir(parents=True)
    (meta / "plans/one.json").write_text('{"pending":true}')
    (meta / "snapshot.json").write_text("{}")
    one, two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    prepare._archive(meta, one)
    prepare._archive(meta, two)
    assert one.read_bytes() == two.read_bytes()
    with tarfile.open(one) as archive:
        assert archive.getnames() == ["plans/one.json", "snapshot.json"]
        assert all(item.isfile() and item.mode == 0o644 and item.mtime == 0 for item in archive)
    (meta / "bad").symlink_to(meta / "snapshot.json")
    with pytest.raises(ValueError, match="regular metadata"):
        prepare._archive(meta, tmp_path / "bad.tar.gz")


def test_output_requires_fresh_canonical_run_path(tmp_path):
    with pytest.raises(ValueError, match="fresh canonical D0"):
        prepare.prepare(tmp_path / "cloud-v1")


@pytest.fixture(scope="module")
def packaged():
    if not (prepare.OUTPUT / "deployment.json").is_file():
        pytest.skip("fresh local cloud package unavailable")
    deployment = prepare.q.read(prepare.OUTPUT / "deployment.json")
    snapshot = prepare.q.read(prepare.OUTPUT / "metadata/snapshot.json")
    return deployment, snapshot


def test_packaged_snapshot_is_entirely_pending_and_main_budget_includes_prior_spend(packaged):
    deployment, snapshot = packaged
    assert deployment["fresh_run"] is True
    assert deployment["environment_name"] == "main"
    assert deployment["app_name"] == "faynt-d0-632-980-v1"
    assert deployment["state_name"] == "faynt-d0-632-980-state-v1"
    main_original = (
        prepare.q.read(prepare.OUTPUT / "deployment-main-original.json")
        if deployment.get("budget_resume_authorization") else deployment
    )
    assert main_original["budget"] == prepare.budget.make_policy(
        total_usd=150, reserve_usd=10, prior_spend_usd=1
    )
    if deployment.get("budget_resume_authorization"):
        authorization_path = (prepare.OUTPUT / deployment["budget_resume_authorization"]).resolve()
        assert authorization_path.is_relative_to(prepare.fresh.OUTPUT)
        authorization = prepare.budget.validate_resume_authorization(prepare.q.read(authorization_path))
        assert deployment["budget"] == authorization["budget_policy"] == prepare.budget.make_policy(
            total_usd=150, reserve_usd=10, prior_spend_usd=80
        )
        assert authorization["queue_sha256"] == deployment["snapshot_sha256"]
        assert authorization["app_name"] == deployment["app_name"]
        assert authorization["environment_name"] == deployment["environment_name"]
    assert "environment_id" not in deployment["budget"]
    if snapshot["budget"] != deployment["budget"]:
        migration = deployment["deployment_migration"]
        assert migration["from"]["environment_name"] == "faynt-d0-632-980"
        assert migration["from"]["budget"] == snapshot["budget"]
        assert migration["to"] == {"environment_name": "main", "budget": main_original["budget"]}
        assert migration["snapshot_budget_role"] == "frozen preparation metadata; deployment.budget is the execution override"
    assert deployment["counts"] == {"pending": 2624, "complete": 0, "quarantined": 0}
    assert deployment["adopted_calls"] == snapshot["adopted_calls"] == 0
    assert snapshot["imported_completed_games"] == []
    assert len(snapshot["games"]) == 2624
    for game in snapshot["games"]:
        assert game["initial_status"] == "pending"
        assert game["initial_row"] == {"status": "pending", "attempts": []}
        assert "existing_claim" not in game and "external_submission" not in game


def test_every_packaged_plan_hash_scope_and_resource_setting(packaged):
    _, snapshot = packaged
    sources = {row["sha256"] for row in snapshot["source_inputs"]}
    for row in snapshot["games"]:
        body = prepare.q.read_bytes(prepare.OUTPUT / "metadata" / row["plan_path"])
        assert prepare.cloud.digest(body) == row["plan_sha256"]
        plan = json.loads(body)
        index, = plan["scope"]["selected_game_indices"]
        assert plan["games"][index]["label"] == row["label"]
        assert plan["artifact_labels"][index] == row["label"] + "-a1"
        assert plan["scope"]["benchmark_games"] == 1
        assert plan["scope"]["run_id"] == prepare.fresh.RUN_ID
        assert plan["scope"]["save_slp"] is True and plan["scope"]["save_video"] is False
        assert plan["limits"]["nonpreemptible"] is False
        assert plan["limits"]["cpu"] == [4, 4] and plan["limits"]["memory_mib"] == [16384, 16384]
        assert len(plan["uploads"]) == len({record["remote"] for record in plan["uploads"]})
        by_remote = {record["remote"]: record for record in plan["uploads"]}
        old = by_remote[prepare.REFERENCE_POLICY_REMOTE]
        new = by_remote[prepare.REFERENCE_POLICY_REMOTE.replace("/opt/runtime-root/", "/opt/d0-root/")]
        assert old["sha256"] == prepare.REFERENCE_POLICY_SHA
        assert old["local"] == str(prepare.REFERENCE_POLICY)
        assert new["local"] != old["local"] and new["sha256"] != old["sha256"]
        for record in plan["uploads"]:
            assert record["sha256"] in sources or (
                prepare.OUTPUT / "metadata/blobs" / record["sha256"]
            ).is_file()


def test_packaged_archive_and_snapshot_digests_and_regular_entries(packaged):
    deployment, _ = packaged
    assert prepare.fresh.sha256(prepare.OUTPUT / "metadata/snapshot.json") == deployment["snapshot_sha256"]
    assert prepare.fresh.sha256(Path(deployment["archive"])) == deployment["archive_sha256"]
    with tarfile.open(deployment["archive"]) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names))
        assert len([name for name in names if name.startswith("plans/")]) == 2624
        assert all(member.isfile() and not Path(member.name).is_absolute()
                   and ".." not in Path(member.name).parts for member in members)


def test_main_migration_preserves_original_deployment_and_frozen_inputs(packaged):
    deployment, _ = packaged
    if deployment.get("budget_resume_authorization"):
        main_original = prepare.q.read(prepare.OUTPUT / "deployment-main-original.json")
        budget_only = copy.deepcopy(deployment)
        budget_only.pop("budget_resume_authorization")
        budget_only["budget"] = main_original["budget"]
        assert budget_only == main_original
        deployment = main_original
    migration = deployment.get("deployment_migration")
    if migration is None:
        pytest.skip("fresh main deployment has no isolated-environment predecessor")
    path = prepare.OUTPUT / migration["from"]["preserved_filename"]
    assert prepare.fresh.sha256(path) == migration["from"]["deployment_sha256"]
    original = prepare.q.read(path)
    assert original["environment_name"] == migration["from"]["environment_name"]
    assert original["budget"] == migration["from"]["budget"]
    recovered = copy.deepcopy(deployment)
    recovered.pop("deployment_migration")
    recovered["environment_name"] = original["environment_name"]
    recovered["budget"] = original["budget"]
    recovered["preparation"] = original["preparation"]
    assert recovered == original
    assert migration["frozen_inputs_unchanged"] == {
        "snapshot_sha256": original["snapshot_sha256"],
        "archive_sha256": original["archive_sha256"],
        "plans": "all 2624 plan files unchanged",
    }
