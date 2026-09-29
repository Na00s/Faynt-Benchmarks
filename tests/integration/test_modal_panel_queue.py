"""Synthetic queue/artifact fixtures. No real queue, model, process or cloud."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_queue as q


def synthetic():
    games, rows = [], {}
    for phase, (profile, block) in enumerate(q.PHASES):
        for pair in range(4 if phase == 0 else 2):
            pair_id = f"{phase * 100 + pair:016x}"
            for port in (1, 2):
                label = f"test-{profile}-{block}-{pair}-p{port}"
                game = {"profile": profile, "block": block, "release": "test", "stage": "FINAL_DESTINATION",
                        "seed": phase * 100 + pair, "frisson_character": "FOX", "slippi_character": "FOX",
                        "player_name": "Player", "frisson_port": port, "slippi_port": 3 - port,
                        "ordinal": len(games) + 1, "pair_id": pair_id, "label": label,
                        "save_slp": True, "save_video": False}
                status = "pending" if phase or pair >= 2 else "quarantined" if pair == 1 and port == 1 else "complete"
                rows[label] = {"status": status, "attempts": []}
                if status == "complete": rows[label]["result"] = {"win": True, "stocks_taken": 4, "stocks_conceded": 1}
                if status == "quarantined": rows[label]["observed_outcome"] = {"win": False, "retained": True}
                games.append(game)
    manifest = {"games": games, "releases": {"test": {}}}
    body = q.encoded(manifest)
    state = {"schema": "e011.slippi_public_panel.state.v1", "manifest_sha256": q.digest(body), "games": rows,
             "counts": dict(Counter(r["status"] for r in rows.values())), "status": "paused-local",
             "current_label": None, "worker_pid": None, "supervisor_pid": None,
             "local_pause": {"new_local_launches_authorized": False}}
    return manifest, state


RUNTIME = {"runtime_image": "im-synthetic", "admission_sha256": "a" * 64}
EXECUTION = {"app_id": "ap-test", "call_id": "fc-test", "task_id": "ta-test", "execution_id": "b" * 32}


@pytest.fixture
def queue(tmp_path, monkeypatch):
    root = tmp_path / "project"; artifacts = root / "artifacts"; output = artifacts / "panel"
    (output / "migrations").mkdir(parents=True)
    (output / ".supervisor.lock").touch(); (artifacts / ".global.lock").touch()
    monkeypatch.setattr(q.job, "ROOT", root); monkeypatch.setattr(q.job, "ARTIFACTS", artifacts)
    monkeypatch.setattr(q.job, "OUTPUT", output); monkeypatch.setattr(q.job, "GLOBAL_LOCK", artifacts / ".global.lock")
    monkeypatch.setattr(q.job, "validate_deployment", lambda: {"synthetic": True})
    monkeypatch.setattr(q.job, "revalidate_completed", lambda *a: None)
    calls = []
    def validate(game, attempt):
        calls.append((copy.deepcopy(game), copy.deepcopy(attempt)))
        summary = artifacts / attempt["label"] / "summary.json"
        data = q.read(summary)
        if data.get("fail"): raise ValueError("synthetic native gate failure")
        return {"win": True, "stocks_taken": 4, "stocks_conceded": 1,
                "summary": str(summary), "summary_sha256": q.digest(q.read_bytes(summary))}
    monkeypatch.setattr(q.job, "validate_result", validate)
    manifest, state = synthetic()
    q.new_file(output / "manifest.json", manifest); q.new_file(output / "state.json", state)
    q.new_file(output / "STATUS.json", {}); q.new_file(output / "RESULTS.md", b"prior report\n")
    directory = output / "migrations/modal-synthetic"
    frozen = q.freeze_plan(directory, runtime_binding=RUNTIME, max_workers=2)
    return {"root": root, "output": output, "artifacts": artifacts, "directory": directory,
            "manifest": manifest, "state": state, "frozen": frozen, "calls": calls}


def claim(queue, pair_index=0):
    return q.claim_pair(queue["directory"], queue["frozen"]["plan_sha256"],
                        queue["frozen"]["plan"]["pairs"][pair_index]["pair_id"],
                        expected_state_sha256=q.digest(q.read_bytes(queue["output"] / "state.json")))


def transport(queue, claim, index=0, *, fail=False, empty=False):
    attempt = claim["attempts"][index]; folder = queue["artifacts"] / attempt["attempt_label"]
    folder.mkdir()
    if not empty:
        q.new_file(folder / "summary.json", {"fail": fail, "execution": {"natural_game_end": bool(fail)}})
        q.new_file(folder / "controller_trace.jsonl", b"synthetic canonical trace\n")
        (folder / "replays").mkdir(); q.new_file(folder / "replays/game.slp", b"synthetic SLP")
    files = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size, "sha256": q.digest(p.read_bytes())}
             for p in sorted(folder.rglob("*")) if p.is_file()]
    receipt = {"schema": q.SCHEMA, "plan_sha256": queue["frozen"]["plan_sha256"], "claim_token": claim["claim_token"],
               "attempt_token": attempt["attempt_token"], "execution": EXECUTION, "files": files, "exit_code": None}
    path = queue["root"] / f"transport-{attempt['attempt_token']}.json"; q.new_file(path, receipt)
    return path


def bind(queue, claim, execution=EXECUTION):
    return q.bind_execution(queue["directory"], queue["frozen"]["plan_sha256"], claim["claim_token"], execution)


def ingest(queue, receipt):
    return q.ingest(queue["directory"], queue["frozen"]["plan_sha256"], receipt)


def test_plan_contains_only_pending_intact_pairs_and_exact_phase_slots(queue):
    plan = queue["frozen"]["plan"]
    assert plan["pending_games"] == 24 and plan["phase_pair_counts"] == [2] * 6
    assert len(plan["protected_rows"]) == 4
    assert [(p["phase"], p["worker_slot"]) for p in plan["pairs"]] == [(phase, slot) for phase in range(6) for slot in (0, 1)]
    assert q.read(queue["output"] / "state.json") == queue["state"]
    assert q.load_plan(queue["directory"], queue["frozen"]["plan_sha256"])[0] == plan


def test_zero_gameplay_socket_failure_is_retained_then_retryable(queue):
    from test_modal_panel_empty_startup import summary
    claimed = claim(queue); bind(queue, claimed)
    attempt = claimed["attempts"][0]; folder = queue["artifacts"] / attempt["attempt_label"]
    folder.mkdir(); s = summary(); s["fail"] = True
    q.new_file(folder / "summary.json", s); q.new_file(folder / "controller_trace.jsonl", b"")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": q.digest(p.read_bytes())}
             for p in sorted(folder.iterdir())]
    receipt = {"schema": q.SCHEMA, "plan_sha256": queue["frozen"]["plan_sha256"],
        "claim_token": claimed["claim_token"], "attempt_token": attempt["attempt_token"],
        "execution": EXECUTION, "files": files, "exit_code": 0}
    path = queue["root"] / "startup-transport.json"; q.new_file(path, receipt)
    assert ingest(queue, path)["status"] == "pending"
    assert q.read(folder / "summary.json") == s
    assert ingest(queue, path)["idempotent"] is True
    assert ingest(queue, transport(queue, claimed, 1, empty=True))["status"] == "pending"
    retry = claim(queue)
    assert [a["attempt_number"] for a in retry["attempts"]] == [2, 2]


@pytest.mark.parametrize("change", ["running", "local-resume", "pending-attempt", "split-pair", "ports", "seed", "video", "ordinal", "worker-cap"])
def test_plan_rejects_unsafe_snapshot(change):
    manifest, state = synthetic(); options = {"runtime_binding": RUNTIME}
    first = manifest["games"][4]; second = manifest["games"][5]
    if change == "running": state["current_label"] = first["label"]
    if change == "local-resume": state["local_pause"]["new_local_launches_authorized"] = True
    if change == "pending-attempt": state["games"][first["label"]]["attempts"] = [{"pending": True}]
    if change == "split-pair": state["games"][first["label"]]["status"] = "complete"
    if change == "ports": second["frisson_port"] = 1
    if change == "seed": second["seed"] += 1
    if change == "video": first["save_video"] = True
    if change == "ordinal": first["ordinal"] = 99
    if change == "worker-cap": options["max_workers"] = 65
    manifest_body = q.encoded(manifest); state["manifest_sha256"] = q.digest(manifest_body)
    with pytest.raises(ValueError): q.make_plan(manifest_body, q.encoded(state), **options)


def test_claim_is_durable_disjoint_and_cannot_resubmit(queue):
    first, second = claim(queue, 0), claim(queue, 1)
    assert first["worker_slot"] != second["worker_slot"]
    assert {a["attempt_token"] for a in first["attempts"]}.isdisjoint(a["attempt_token"] for a in second["attempts"])
    assert [a["game"]["frisson_port"] for a in first["attempts"]] == [1, 2]
    assert all(a["attempt_number"] == 1 for a in first["attempts"])
    assert q.read(queue["output"] / "state.json") == queue["state"]
    with pytest.raises(RuntimeError, match="reconciled"): claim(queue, 0)
    with pytest.raises(ValueError, match="earlier phase"): claim(queue, 2)


def test_provider_identity_is_once_bound(queue):
    value = claim(queue); first = bind(queue, value)
    assert bind(queue, value) == first
    with pytest.raises(ValueError, match="execution changed"):
        bind(queue, value, {**EXECUTION, "task_id": "ta-replacement"})


def test_ingestion_is_idempotent_preserves_protected_rows_and_unknown_exit(queue):
    value = claim(queue); bind(queue, value); path = transport(queue, value)
    first = ingest(queue, path); before = q.read_bytes(queue["output"] / "state.json")
    duplicate = ingest(queue, path)
    assert first["committed"] and not first["idempotent"] and duplicate["idempotent"]
    assert q.read_bytes(queue["output"] / "state.json") == before
    state = json.loads(before); label = value["attempts"][0]["game"]["label"]
    assert len(state["games"][label]["attempts"]) == 1 and state["games"][label]["status"] == "complete"
    assert "exit_code" not in state["games"][label]["attempts"][0]
    for key, row in queue["state"]["games"].items():
        if key != label: assert state["games"][key] == row
    assert (queue["directory"] / "ingests" / value["attempts"][0]["attempt_token"] / "transport.json").read_bytes() == path.read_bytes()
    assert queue["calls"]


def test_second_port_ingestion_waits_for_first(queue):
    value = claim(queue); bind(queue, value); second = transport(queue, value, 1)
    with pytest.raises(ValueError, match="earlier assigned port"): ingest(queue, second)
    ingest(queue, transport(queue, value, 0)); ingest(queue, second)
    with pytest.raises(ValueError, match="already has terminal"): claim(queue)


@pytest.mark.parametrize("point", ["before-publish", "after-state-publish", "before-done"])
def test_interrupted_commit_recovers_without_duplicate_game(queue, monkeypatch, point):
    value = claim(queue); bind(queue, value); path = transport(queue, value)
    original_publish, original_new = q.job.publish, q.new_file
    def publish(state, panel):
        if point == "before-publish": raise RuntimeError("interrupted")
        original_publish(state, panel)
        if point == "after-state-publish": raise RuntimeError("interrupted")
    def new_file(path, data):
        if point == "before-done" and Path(path).name == "done.json": raise RuntimeError("interrupted")
        return original_new(path, data)
    monkeypatch.setattr(q.job, "publish", publish); monkeypatch.setattr(q, "new_file", new_file)
    with pytest.raises(RuntimeError, match="interrupted"): ingest(queue, path)
    with pytest.raises(RuntimeError, match="unfinished ingestion"): claim(queue, 1)
    monkeypatch.setattr(q.job, "publish", original_publish); monkeypatch.setattr(q, "new_file", original_new)
    assert ingest(queue, path)["committed"]
    row = q.read(queue["output"] / "state.json")["games"][value["attempts"][0]["game"]["label"]]
    assert len(row["attempts"]) == 1


def test_partial_transport_journal_can_resume(queue, monkeypatch):
    value = claim(queue); bind(queue, value); path = transport(queue, value)
    original = q.new_file
    def fail(path, data):
        if Path(path).name == "intent.json": raise RuntimeError("journal interrupted")
        return original(path, data)
    monkeypatch.setattr(q, "new_file", fail)
    with pytest.raises(RuntimeError): ingest(queue, path)
    monkeypatch.setattr(q, "new_file", original)
    assert ingest(queue, path)["committed"]


def test_natural_end_failure_uses_original_quarantine_rule(queue):
    value = claim(queue); bind(queue, value)
    result = ingest(queue, transport(queue, value, fail=True))
    assert result["status"] == "quarantined"
    row = q.read(queue["output"] / "state.json")["games"][value["attempts"][0]["game"]["label"]]
    assert "natural-end game requires audit" in row["attempts"][0]["error"]


def test_empty_infrastructure_failure_retained_and_retried_to_three(queue):
    for number in (1, 2, 3):
        value = claim(queue); bind(queue, value)
        assert all(a["attempt_number"] == number for a in value["attempts"])
        for index in range(2):
            result = ingest(queue, transport(queue, value, index, empty=True))
            assert result["status"] == ("pending" if number < 3 else "quarantined")
    with pytest.raises(ValueError, match="already has terminal"): claim(queue)


@pytest.mark.parametrize("change", ["body", "extra-file", "symlink", "video", "wrong-task", "wrong-token", "traversal", "missing-file"])
def test_transport_tampering_never_changes_state(queue, change):
    value = claim(queue); bind(queue, value); path = transport(queue, value)
    receipt = q.read(path); folder = queue["artifacts"] / value["attempts"][0]["attempt_label"]
    if change == "body": (folder / "summary.json").write_bytes(b"changed")
    if change == "extra-file": (folder / "unlisted.json").write_bytes(b"{}")
    if change == "symlink": (folder / "link").symlink_to(path)
    if change == "video": receipt["files"][0]["path"] = "game.mp4"
    if change == "wrong-task": receipt["execution"]["task_id"] = "ta-other"
    if change == "wrong-token": receipt["attempt_token"] = "f" * 32
    if change == "traversal": receipt["files"][0]["path"] = "../summary.json"
    if change == "missing-file": receipt["files"].pop()
    path.write_bytes(q.encoded(receipt)); before = q.read_bytes(queue["output"] / "state.json")
    with pytest.raises((ValueError, FileNotFoundError)): ingest(queue, path)
    assert q.read_bytes(queue["output"] / "state.json") == before


def test_protected_original_outcome_mutation_blocks(queue):
    state = q.read(queue["output"] / "state.json")
    label = next(iter(queue["frozen"]["plan"]["protected_rows"]))
    state["games"][label]["changed"] = True
    (queue["output"] / "state.json").write_bytes(q.encoded(state))
    with pytest.raises(ValueError, match="original results"): claim(queue)


def test_lock_contention_blocks_all_mutation(queue):
    with q.existing_lock(queue["output"] / ".supervisor.lock"):
        with pytest.raises(BlockingIOError): claim(queue)


def test_new_helper_has_no_launch_or_cloud_api():
    import ast
    tree = ast.parse(Path(q.__file__).read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls) & {"Popen", "run_game", "spawn", "remote", "remote_gen", "kill", "killpg", "supervise"}


def test_progress_is_read_only_and_distinguishes_intent_from_acceptance(queue):
    before = q.read_bytes(queue["output"] / "state.json")
    initial = q.progress(queue["directory"], queue["frozen"]["plan_sha256"])
    assert initial["original_accepted"] == 3 and initial["original_quarantined"] == 1
    assert initial["phases"][0]["games"] == {"queued": 4, "running_or_awaiting": 0, "accepted": 0, "quarantined": 0}
    value = claim(queue)
    projected = q.progress(queue["directory"], queue["frozen"]["plan_sha256"])
    assert projected["phases"][0]["games"] == {"queued": 2, "running_or_awaiting": 2, "accepted": 0, "quarantined": 0}
    assert projected["phases"][0]["attempts"]["execution_bound"] == 0
    bind(queue, value)
    assert q.read_bytes(queue["output"] / "state.json") == before
    ingest(queue, transport(queue, value))
    after = q.read_bytes(queue["output"] / "state.json")
    projected = q.progress(queue["directory"], queue["frozen"]["plan_sha256"])
    phase = projected["phases"][0]
    assert phase["games"] == {"queued": 2, "running_or_awaiting": 1, "accepted": 1, "quarantined": 0}
    assert phase["attempts"] == {"claimed": 2, "execution_bound": 2, "reconciled": 1, "awaiting_ingest": 1}
    assert projected["read_only"] and "unobserved" in projected["provider_liveness"]
    assert q.read_bytes(queue["output"] / "state.json") == after


def test_progress_identifies_unfinished_publish_recovery(queue, monkeypatch):
    value = claim(queue); bind(queue, value); path = transport(queue, value)
    original = q.job.publish
    def fail(state, panel): original(state, panel); raise RuntimeError("interrupted")
    monkeypatch.setattr(q.job, "publish", fail)
    with pytest.raises(RuntimeError): ingest(queue, path)
    projected = q.progress(queue["directory"], queue["frozen"]["plan_sha256"])
    assert projected["unfinished_ingest_tokens"] == [value["attempts"][0]["attempt_token"]]
    assert projected["phases"][0]["games"]["accepted"] == 1
    assert projected["phases"][0]["attempts"]["awaiting_ingest"] == 2


def test_changed_done_marker_cannot_claim_idempotent_success(queue):
    value = claim(queue); bind(queue, value); path = transport(queue, value); ingest(queue, path)
    done = queue["directory"] / "ingests" / value["attempts"][0]["attempt_token"] / "done.json"
    changed = q.read(done); changed["receipt_sha256"] = "f" * 64; done.write_bytes(q.encoded(changed))
    with pytest.raises(ValueError, match="committed ingestion"): ingest(queue, path)
