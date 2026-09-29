# Imported pytest fixtures intentionally share names with fixture parameters.
# ruff: noqa: F811
import copy
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_cloud_queue as q
from test_modal_panel_cloud_queue import dispatcher  # noqa: F401
from test_modal_panel_durable_game import rig  # noqa: F401


@pytest.fixture
def interrupted(tmp_path):
    scope = {"selected_game_indices": [0]}
    values = {
        "receipt.json": {
            "status": "failed",
            "games": [],
            "error": "KeyboardInterrupt: owner termination requested",
            "plan_sha256": "a" * 64,
            "scope": scope,
        },
        "supervisor/receipt.json": {
            "error": "KeyboardInterrupt: ",
            "exit_code": 0,
            "terminated_owned_process_group": True,
            "unreaped_leader_group_guard": True,
        },
    }
    rows = []
    for name, value in values.items():
        path = tmp_path / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(q.encoded(value))
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": q.digest(path.read_bytes())})
    name = "project/artifacts/integration/frisson_ai/game-a1/replays/partial.slp"
    path = tmp_path / "files" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial replay")
    rows.append({"name": name, "bytes": path.stat().st_size, "sha256": q.digest(path.read_bytes())})
    manifest = {"status": "failed", "scope": scope, "plan_sha256": "a" * 64, "files": rows}
    return manifest, tmp_path, values


def test_clean_interruption_allows_retry_without_score(interrupted):
    manifest, root, _ = interrupted
    assert q.verified_failure_retry_reason(manifest, root) == "verified-interrupted-native-owner"
    assert any(row["name"].endswith(".slp") for row in manifest["files"])
    d = types.SimpleNamespace(
        MOUNT=root.parent, SUCCESS="full-policy-durable-game-passed", check_manifest=lambda *args: None
    )
    manifest.update(storage_path=root.name, native_schema="native")
    with pytest.raises(ValueError, match="without score admission"):
        q.native_acceptance(d, None, None, manifest)


@pytest.fixture
def empty_owner(tmp_path):
    supervisor = {
        "command": [
            "/usr/local/bin/python",
            "/opt/d0-root/scripts/faynt_d0_durable_game.py",
            "--owner",
            "--sha",
            "a" * 64,
            "--expires",
            "9000",
            "--image-id",
            "im-test",
        ],
        "started_unix": 1000,
        "ended_unix": 1002,
        "expires_unix": 9000,
        "error": "KeyboardInterrupt: ",
        "exit_code": -15,
        "terminated_owned_process_group": True,
        "unreaped_leader_group_guard": True,
        "log_bytes": 0,
        "log_sha256": q.digest(b""),
    }
    directory = tmp_path / "files/supervisor"
    directory.mkdir(parents=True)
    (directory / "owner.log").write_bytes(b"")
    (directory / "receipt.json").write_bytes(q.encoded(supervisor))
    rows = []
    for name in ("supervisor/owner.log", "supervisor/receipt.json"):
        body = (tmp_path / "files" / name).read_bytes()
        rows.append(
            {
                "name": name,
                "bytes": len(body),
                "original_bytes": len(body),
                "sha256": q.digest(body),
                "original_sha256": q.digest(body),
                "truncated_tail": False,
            }
        )
    return (
        {"status": "failed", "plan_sha256": "a" * 64, "scope": {"selected_game_indices": [0]}, "files": rows},
        tmp_path,
        supervisor,
    )


def test_exact_early_empty_owner_interruption_is_retryable(empty_owner):
    manifest, root, _ = empty_owner
    assert q.interrupted_empty_owner(manifest, root)
    assert q.verified_failure_retry_reason(manifest, root) == "verified-preowner-interruption"
    assert authorize(reason="verified-preowner-interruption")["attempt_number"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-replay",
        "owner-receipt",
        "summary",
        "missing-log",
        "nonempty-log",
        "truncated-log",
        "wrong-error",
        "normal-exit",
        "cleanup",
        "unreaped",
        "command-plan",
        "command-entry",
        "command-expiry",
        "elapsed",
        "expired",
        "nonfinite",
        "log-size",
        "log-hash",
        "passed",
    ],
)
def test_empty_owner_classifier_refuses_incomplete_or_other_failures(empty_owner, mutation):
    manifest, root, supervisor = empty_owner
    if mutation == "extra-replay":
        manifest["files"].append({"name": "project/replay.slp"})
    if mutation == "owner-receipt":
        manifest["files"].append({"name": "receipt.json"})
    if mutation == "summary":
        manifest["files"].append({"name": "project/summary.json"})
    if mutation == "missing-log":
        manifest["files"] = manifest["files"][1:]
    if mutation == "nonempty-log":
        (root / "files/supervisor/owner.log").write_bytes(b"started")
    if mutation == "truncated-log":
        manifest["files"][0]["truncated_tail"] = True
    if mutation == "wrong-error":
        supervisor["error"] = "TimeoutError: owner absolute deadline exhausted"
    if mutation == "normal-exit":
        supervisor["exit_code"] = 0
    if mutation == "cleanup":
        supervisor["terminated_owned_process_group"] = False
    if mutation == "unreaped":
        supervisor["unreaped_leader_group_guard"] = False
    if mutation == "command-plan":
        supervisor["command"][4] = "b" * 64
    if mutation == "command-entry":
        supervisor["command"][1] = "/opt/other.py"
    if mutation == "command-expiry":
        supervisor["command"][6] = "9001"
    if mutation == "elapsed":
        supervisor["ended_unix"] = 1011
    if mutation == "expired":
        supervisor["expires_unix"] = 1001
    if mutation == "nonfinite":
        supervisor["started_unix"] = float("inf")
    if mutation == "log-size":
        supervisor["log_bytes"] = 1
    if mutation == "log-hash":
        supervisor["log_sha256"] = "c" * 64
    if mutation == "passed":
        manifest["status"] = "full-policy-durable-game-passed"
    (root / "files/supervisor/receipt.json").write_text(json.dumps(supervisor))
    assert q.interrupted_empty_owner(manifest, root) is False


@pytest.fixture
def prebootstrap(empty_owner):
    manifest, root, supervisor = empty_owner
    supervisor.update(expires_unix=1400, ended_unix=1002, exit_code=0)
    supervisor.pop("error")
    supervisor["command"][6] = "1400"
    owner = {
        "status": "failed",
        "games": [],
        "error": "TimeoutError: pilot absolute deadline exhausted",
        "plan_sha256": manifest["plan_sha256"],
        "scope": manifest["scope"],
        "started_unix": 1001,
        "ended_unix": 1001.01,
        "expires_unix": 1400,
        "runtime_image_id": "im-test",
    }
    (root / "files/supervisor/receipt.json").write_bytes(q.encoded(supervisor))
    (root / "files/receipt.json").write_bytes(q.encoded(owner))
    manifest["files"].append({"name": "receipt.json"})
    return manifest, root, owner, supervisor


def test_exact_prebootstrap_budget_abort_needs_new_authorized_tranche(prebootstrap):
    manifest, root, _, _ = prebootstrap
    assert q.verified_failure_retry_reason(manifest, root) == "verified-prebootstrap-budget-exhaustion"
    assert authorize(reason="verified-prebootstrap-budget-exhaustion")["attempt_number"] == 2
    with pytest.raises(ValueError, match="approved current tranche"):
        authorize(reason="verified-prebootstrap-budget-exhaustion", resume_tranche=None)


@pytest.mark.parametrize(
    "mutation", ["runtime", "game", "replay", "error", "long-run", "enough-budget", "expiry"]
)
def test_prebootstrap_classifier_excludes_possible_gameplay_and_other_failures(prebootstrap, mutation):
    manifest, root, owner, _ = prebootstrap
    if mutation == "runtime":
        owner["fresh_runtime_status"] = "runtime-qualified-gameplay-unadmitted"
    if mutation == "game":
        owner["games"] = [{"index": 0}]
    if mutation == "replay":
        manifest["files"].append({"name": "partial.slp"})
    if mutation == "error":
        owner["error"] = "RuntimeError: model failure"
    if mutation == "long-run":
        owner["ended_unix"] = 1003
    if mutation == "enough-budget":
        owner["expires_unix"] = 9000
    if mutation == "expiry":
        owner["expires_unix"] = 1399
    (root / "files/receipt.json").write_bytes(q.encoded(owner))
    assert q.exhausted_prebootstrap_budget(manifest, root) is False


@pytest.fixture
def public_fetch_failure(prebootstrap):
    manifest, root, owner, supervisor = prebootstrap
    owner.update(
        error="RuntimeError: command exited 128; inspect 004.log",
        fresh_runtime_status="runtime-qualified-gameplay-unadmitted",
    )
    scope = {
        "gameplay_admitted": False,
        "dolphin_launches": 0,
        "rom_reads": 0,
        "policy_instances": 0,
        "policy_inference": 0,
    }
    qualified = {
        "status": "runtime-child-qualified",
        "scope": scope,
        "parent_terminal_cleanup_verified": False,
    }
    runtime = {
        "status": "runtime-qualified-gameplay-unadmitted",
        "scope": scope,
        "qualification": copy.deepcopy(qualified),
        "cleanup": {"parent_terminal_cleanup_verified": True},
    }
    work = "/work/research/faynt-d0-game"
    public = work + "/source/melee-policy/.e001-cache/slippi-ai-source"
    argv = [
        [
            "apt-get",
            "-o",
            "Acquire::Retries=0",
            "-o",
            "Acquire::http::Timeout=30",
            "install",
            "-y",
            "--no-install-recommends",
            "git",
        ],
        ["git", "init", "--initial-branch=main", public],
        ["git", "-C", public, "remote", "add", "origin", "https://github.com/vladfi1/slippi-ai"],
        [
            "git",
            "-C",
            public,
            "fetch",
            "--depth",
            "1",
            "--no-tags",
            "origin",
            "577965a7731dc53e3472ea63d9e9853a4e9d65fa",
        ],
    ]
    log = (
        b"fatal: could not read Username for 'https://github.com': terminal prompts disabled\n"
        b"fatal: the remote end hung up unexpectedly\n"
    )
    assert (
        len(log) == 126
        and q.digest(log) == "a3ed1bca695f5d110bed3bb470877f16707bd532b22040cb10393b8033300f4c"
    )
    commands = []
    values = {
        "receipt.json": owner,
        "runtime/receipt.json": runtime,
        "runtime/qualification.json": qualified,
        "supervisor/receipt.json": supervisor,
    }
    blobs = {"supervisor/owner.log": b""}
    for number, command in enumerate(argv, 1):
        body = log if number == 4 else b"ok\n"
        blobs[f"source-command/logs/{number:03d}.log"] = body
        row = {
            "command": command,
            "cwd": work,
            "log": f"logs/{number:03d}.log",
            "exit_code": 128 if number == 4 else 0,
            "terminated_owned_process_group": True,
            "unreaped_leader_group_guard": True,
            "log_bytes": len(body),
            "log_sha256": q.digest(body),
        }
        if number == 4:
            row["error"] = owner["error"]
        commands.append(row)
    values["source-command/commands.json"] = commands
    blobs.update({name: q.encoded(value) for name, value in values.items()})
    manifest["files"] = []
    for name, body in blobs.items():
        path = root / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        manifest["files"].append(
            {
                "name": name,
                "bytes": len(body),
                "original_bytes": len(body),
                "sha256": q.digest(body),
                "original_sha256": q.digest(body),
                "truncated_tail": False,
            }
        )
    return manifest, root, values


def test_exact_public_source_fetch_failure_is_zero_gameplay_retry(public_fetch_failure):
    manifest, root, _ = public_fetch_failure
    assert q.zero_gameplay_public_source_fetch(manifest, root)
    assert q.verified_failure_retry_reason(manifest, root) == "verified-zero-gameplay-public-source-fetch"
    assert authorize(reason="verified-zero-gameplay-public-source-fetch")["attempt_number"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "game",
        "replay",
        "source-ready",
        "owner-error",
        "owner-plan",
        "owner-scope",
        "runtime-failed",
        "gameplay",
        "dolphin_launches",
        "rom_reads",
        "policy_instances",
        "policy_inference",
        "qualification",
        "runtime-cleanup",
        "supervisor-cleanup",
        "supervisor-error",
        "supervisor-plan",
        "url",
        "commit",
        "fetch-args",
        "earlier-failure",
        "command-cleanup",
        "command-log",
        "extra-command",
        "different-log",
    ],
)
def test_public_source_fetch_classifier_rejects_other_failure_evidence(public_fetch_failure, mutation):
    manifest, root, values = public_fetch_failure
    owner = values["receipt.json"]
    runtime = values["runtime/receipt.json"]
    supervisor = values["supervisor/receipt.json"]
    commands = values["source-command/commands.json"]
    if mutation == "game":
        owner["games"] = [{"index": 0}]
    if mutation in {"replay", "source-ready"}:
        manifest["files"].append(
            {"name": "source.json" if mutation == "source-ready" else "project/game.slp"}
        )
    if mutation == "owner-error":
        owner["error"] = "RuntimeError: policy error"
    if mutation == "owner-plan":
        owner["plan_sha256"] = "b" * 64
    if mutation == "owner-scope":
        owner["scope"] = {"selected_game_indices": [1]}
    if mutation == "runtime-failed":
        runtime["status"] = "failed"
    if mutation == "gameplay":
        runtime["scope"]["gameplay_admitted"] = True
    if mutation in {"dolphin_launches", "rom_reads", "policy_instances", "policy_inference"}:
        runtime["scope"][mutation] = 1
    if mutation == "qualification":
        values["runtime/qualification.json"]["status"] = "failed"
    if mutation == "runtime-cleanup":
        runtime["cleanup"]["parent_terminal_cleanup_verified"] = False
    if mutation == "supervisor-cleanup":
        supervisor["terminated_owned_process_group"] = False
    if mutation == "supervisor-error":
        supervisor["error"] = "KeyboardInterrupt: "
    if mutation == "supervisor-plan":
        supervisor["command"][4] = "b" * 64
    if mutation == "url":
        commands[2]["command"][-1] += "-other"
    if mutation == "commit":
        commands[3]["command"][-1] = "0" * 40
    if mutation == "fetch-args":
        commands[3]["command"][5] = "2"
    if mutation == "earlier-failure":
        commands[1]["exit_code"] = 128
    if mutation == "command-cleanup":
        commands[3]["unreaped_leader_group_guard"] = False
    if mutation == "command-log":
        commands[3]["log_sha256"] = "b" * 64
    if mutation == "extra-command":
        commands.append(copy.deepcopy(commands[-1]))
    if mutation == "different-log":
        (root / "files/source-command/logs/004.log").write_bytes(b"fatal: different failure\n")
    for name, value in values.items():
        (root / "files" / name).write_bytes(q.encoded(value))
    assert q.zero_gameplay_public_source_fetch(manifest, root) is False


@pytest.mark.parametrize(
    "mutation",
    [
        "summary",
        "child",
        "owner-error",
        "supervisor-error",
        "cleanup",
        "unreaped",
        "exit",
        "plan",
        "scope",
        "games",
        "success",
        "missing-proof",
    ],
)
def test_interruption_retry_refuses_other_failures(interrupted, mutation):
    manifest, root, values = interrupted
    owner = values["receipt.json"]
    supervisor = values["supervisor/receipt.json"]
    if mutation == "summary":
        manifest["files"].append({"name": "project/example/summary.json"})
    if mutation == "child":
        manifest["files"].append({"name": "game-0/result.json"})
    if mutation == "owner-error":
        owner["error"] = "RuntimeError: policy mismatch"
    if mutation == "supervisor-error":
        supervisor["error"] = "TimeoutError: owner absolute deadline exhausted"
    if mutation == "cleanup":
        supervisor["terminated_owned_process_group"] = False
    if mutation == "unreaped":
        supervisor["unreaped_leader_group_guard"] = False
    if mutation == "exit":
        supervisor["exit_code"] = 1
    if mutation == "plan":
        owner["plan_sha256"] = "b" * 64
    if mutation == "scope":
        owner["scope"] = {"selected_game_indices": [1]}
    if mutation == "games":
        owner["games"] = [{"status": "complete"}]
    if mutation == "success":
        manifest["status"] = "full-policy-durable-game-passed"
    if mutation == "missing-proof":
        manifest["files"] = manifest["files"][1:]
    for name, value in values.items():
        (root / "files" / name).write_bytes(q.encoded(value))
    assert q.interrupted_native_owner(manifest, root) is False


def retry_fixture(number=1):
    plan = {"artifact_labels": ["game-a1", "peer-a1"], "scope": {"selected_game_indices": [0]}, "seed": 31}
    body = q.encoded(plan)
    row = {"label": "game", "plan_sha256": q.digest(body), "initial_status": "pending"}
    binding = {"call_id": "fc-first", "expires_unix": 100, "image_id": "im-original"}
    history = [
        {"execution": {"execution_id": str(n)}, "call_id": "fc-first", "expires_unix": 100}
        for n in range(number)
    ]
    failure = {
        "manifest": {"status": "failed", "plan_sha256": q.digest(body), "call_id": "fc-first"},
        "physical_history": history,
        "call_binding": binding,
    }
    tranche = {
        "activation": {"deadline_unix": 5000},
        "authorization_sha256": "d" * 64,
        "stopped_no_active_proof_sha256": "e" * 64,
    }
    return row, body, [failure], tranche


def authorize(number=1, **changes):
    row, body, failures, tranche = retry_fixture(number)
    args = {
        "reason": "verified-interrupted-native-owner",
        "expires_unix": 100,
        "now": 200,
        "session_seconds": 1000,
        "resume_tranche": tranche,
    }
    args.update(changes)
    return q.authorize_failure_retry(row, body, failures, **args)


def test_new_tranche_creates_fresh_plan_and_retains_original_expiry():
    row, body, failures, tranche = retry_fixture()
    auth = authorize()
    plan, _, sha = q.retry_execution_plan(row, body, auth)
    assert sha != row["plan_sha256"]
    assert plan == {
        "artifact_labels": ["game-a2", "peer-a1"],
        "scope": {"selected_game_indices": [0]},
        "seed": 31,
    }
    assert auth["expires_unix"] == 1200 and auth["original_expires_unix"] == 100
    assert auth["prior_failures"] == failures
    q.validate_retry_budget_binding(auth, tranche)
    q.validate_prior_call_binding(failures[0], failures[0]["call_binding"], auth)


def test_authorization_reuses_first_deadline_after_lost_publication_response():
    first = authorize(now=200)
    later = authorize(now=250, existing_authorization=first)
    assert later == first
    assert later is not first
    assert later["expires_unix"] == 1200
    # Even an expired retained authorization is preserved for audit. Reuse
    # cannot extend it or grant a new physical-start budget.
    assert authorize(now=1500, existing_authorization=first) == first


@pytest.mark.parametrize("mutation", ["manifest", "history", "call", "reason", "original-expiry", "plan"])
def test_authorization_reuse_requires_exact_original_failure_proofs(mutation):
    original = authorize()
    changed = copy.deepcopy(original)
    if mutation == "manifest":
        changed["prior_failures"][0]["manifest"]["call_id"] = "fc-other"
    if mutation == "history":
        changed["prior_failures"][0]["physical_history"][0]["execution"]["execution_id"] = "other"
    if mutation == "call":
        changed["prior_failures"][0]["call_binding"]["image_id"] = "im-other"
    if mutation == "reason":
        changed["reason"] = "verified-zero-gameplay-runtime-install-timeout"
    if mutation == "original-expiry":
        changed["original_expires_unix"] += 1
    if mutation == "plan":
        changed["queue_plan_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        authorize(now=250, existing_authorization=changed)


def test_authorization_reuse_requires_current_tranche_and_nonterminal_game():
    original = authorize()
    different = copy.deepcopy(retry_fixture()[3])
    different["authorization_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="current hash-bound"):
        authorize(existing_authorization=original, resume_tranche=different)
    with pytest.raises(ValueError, match="terminal"):
        authorize(existing_authorization=original, accepted_record={"status": "complete"})


def test_original_live_deadline_needs_no_resume():
    auth = authorize(expires_unix=1300, resume_tranche=None)
    assert auth["expires_unix"] == 1300 and "budget_resume" not in auth
    q.validate_retry_budget_binding(auth, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"resume_tranche": None},
        {"resume_tranche": {"activation": {"deadline_unix": 250}, "authorization_sha256": "d" * 64}},
        {"accepted_record": {"status": "complete"}},
        {"reason": "arbitrary-failure"},
    ],
)
def test_resume_requires_explicit_budget_and_eligible_failure(changes):
    with pytest.raises(ValueError):
        authorize(**changes)


def test_three_start_cap_includes_prior_platform_executions():
    assert authorize(number=2)["attempt_number"] == 3
    with pytest.raises(ValueError, match="three-attempt"):
        authorize(number=3)


@pytest.mark.parametrize("mutation", ["tranche", "authorization", "activation", "deadline", "missing"])
def test_changed_resume_binding_is_refused(mutation):
    auth = authorize()
    tranche = retry_fixture()[3]
    if mutation == "tranche":
        tranche["extra"] = "changed"
    if mutation == "authorization":
        tranche["authorization_sha256"] = "f" * 64
    if mutation == "activation":
        tranche["activation"]["deadline_unix"] = 5001
    if mutation == "deadline":
        auth["expires_unix"] = 5001
    if mutation == "missing":
        tranche = None
    with pytest.raises(ValueError, match="current hash-bound"):
        q.validate_retry_budget_binding(auth, tranche)


def test_original_history_and_call_binding_cannot_be_rewritten():
    _, _, failures, _ = retry_fixture()
    auth = authorize()
    changed = copy.deepcopy(failures[0]["call_binding"])
    changed["expires_unix"] = 1200
    with pytest.raises(ValueError, match="prior call or deadline"):
        q.validate_prior_call_binding(failures[0], changed, auth)
    failures[0]["physical_history"][0]["expires_unix"] = 1200
    with pytest.raises(ValueError, match="prior call or deadline"):
        q.validate_prior_call_binding(failures[0], failures[0]["call_binding"], auth)


def test_authorized_a2_runs_fresh_physical_with_three_total_cap(rig):
    import modal_panel_durable_game as d

    bound, _, volume, store, _ = rig
    auth = authorize()
    row, body, _, _ = retry_fixture()
    plan, _, sha = q.retry_execution_plan(row, body, auth)

    def crash():
        raise RuntimeError("synthetic interrupted owner")

    for number in (2, 3):
        with pytest.raises(RuntimeError, match="synthetic"):
            d.run_durable(
                bound,
                plan,
                sha,
                90,
                "im-resume",
                volume,
                store,
                "fc-resume",
                {"task_id": f"ta-test{number}", "execution_id": f"{number:032x}"},
                run_owner=crash,
            )
    with pytest.raises(RuntimeError, match="three physical"):
        d.run_durable(
            bound,
            plan,
            sha,
            90,
            "im-resume",
            volume,
            store,
            "fc-resume",
            {"task_id": "ta-test4", "execution_id": f"{4:032x}"},
            run_owner=lambda: pytest.fail("fourth logical physical start"),
        )
    assert len([key for key in store if "/physical/" in key]) == 2
    assert auth["attempt_number"] == 2


def test_scoped_reinspection_uses_fresh_readonly_counter(dispatcher, monkeypatch):
    app, store, calls, launched, sha, _ = dispatcher
    tranche = retry_fixture()[3]
    monkeypatch.setattr(app, "active_resume_tranche", lambda *args: tranche)
    store[sha + "/control"]["held_reinspection"] = {
        "tranche_sha256": q.budget_digest(tranche),
        "labels": ["a"],
    }
    store[sha + "/hold/a"] = {"error": "original interrupted attempt"}
    store[sha + "/hold/c"] = {"error": "unrelated retained hold"}
    store[sha + "/owner/a"] = {"call_id": "fc-old"}
    store[sha + "/dispatch/a"] = {"expires_unix": 100, "call_id": "fc-old"}
    store[sha + "/recovery/a"] = {"submissions": 3}
    calls["fc-old"] = RuntimeError("retained interrupted native result")
    app.dispatch_cloud()
    assert len(launched) == 1 and launched[0][0] == "a"
    assert launched[0][1][1] == {"recover_only": True, "attempt": 1}
    assert store[sha + "/recovery/a"] == {"submissions": 3}
    assert store[sha + "/hold/a"] == {"error": "original interrupted attempt"}


def test_failed_reinspection_archives_old_hold_and_stops_repeating(dispatcher, monkeypatch):
    app, store, calls, launched, sha, _ = dispatcher
    tranche = retry_fixture()[3]
    tranche_sha = q.budget_digest(tranche)
    monkeypatch.setattr(app, "active_resume_tranche", lambda *args: tranche)
    store[sha + "/control"]["held_reinspection"] = {"tranche_sha256": tranche_sha, "labels": ["a"]}
    old = {"error": "original interrupted attempt"}
    store[sha + "/hold/a"] = old
    store[sha + "/owner/a"] = {"call_id": "fc-old"}
    store[sha + "/dispatch/a"] = {"expires_unix": 100, "call_id": "fc-old"}
    store[sha + "/recovery/a/resume-" + tranche["authorization_sha256"]] = {"submissions": 3}
    calls["fc-old"] = RuntimeError("unrelated failure remains unscored")
    app.dispatch_cloud()
    assert store[sha + "/hold-history/" + tranche_sha + "/a"] == old
    assert store[sha + "/hold/a"]["reinspection_tranche_sha256"] == tranche_sha
    app.dispatch_cloud()
    assert not any(label == "a" for label, _, _ in launched)
    assert sha + "/result/a" not in store


def test_reinspection_excludes_accepted_and_unapproved_labels():
    tranche = retry_fixture()[3]
    control = {"held_reinspection": {"tranche_sha256": q.budget_digest(tranche), "labels": ["a", "b"]}}
    definitions = {label: {"label": label, "initial_status": "pending"} for label in ("a", "b")}
    assert q.held_reinspection_labels(control, tranche, definitions, {"a": {"status": "complete"}}) == {"b"}
    control["allowlist"] = ["a"]
    with pytest.raises(ValueError, match="active allowlist"):
        q.held_reinspection_labels(control, tranche, definitions, {})


def test_reviewed_revision_rearms_only_exact_named_hold(dispatcher, monkeypatch):
    app, store, calls, launched, sha, _ = dispatcher
    tranche = retry_fixture()[3]
    tranche_sha = q.budget_digest(tranche)
    monkeypatch.setattr(app, "active_resume_tranche", lambda *args: tranche)
    old = {"error": "unclassified early interruption", "reinspection_tranche_sha256": tranche_sha}
    revision = {
        "classifier_sha256": q.digest(Path(q.__file__).read_bytes()),
        "prior_hold_sha256": q.digest(old),
    }
    store[sha + "/control"]["held_reinspection"] = {
        "tranche_sha256": tranche_sha,
        "labels": ["a", "c"],
        "revisions": {"a": revision},
    }
    store[sha + "/hold/a"] = old
    store[sha + "/hold/c"] = copy.deepcopy(old)
    store[sha + "/owner/a"] = {"call_id": "fc-old"}
    store[sha + "/dispatch/a"] = {"expires_unix": 100, "call_id": "fc-old"}
    store[sha + "/recovery/a/resume-" + tranche["authorization_sha256"]] = {"submissions": 3}
    calls["fc-old"] = RuntimeError("retained early owner failure")
    app.dispatch_cloud()
    assert len(launched) == 1 and launched[0][0] == "a"
    assert launched[0][1][1] == {"recover_only": True, "attempt": 1}
    assert store[sha + "/hold/a"] == old
    assert sha + "/retry-latest/a" not in store


@pytest.mark.parametrize("mutation", ["code", "prior-hold", "extra-label"])
def test_reviewed_revision_requires_exact_source_and_retained_hold(mutation):
    tranche = retry_fixture()[3]
    hold = {"error": "reviewed failure"}
    revision = {
        "classifier_sha256": q.digest(Path(q.__file__).read_bytes()),
        "prior_hold_sha256": q.digest(hold),
    }
    control = {
        "held_reinspection": {
            "tranche_sha256": q.budget_digest(tranche),
            "labels": ["a"],
            "revisions": {"a": revision},
        }
    }
    if mutation == "code":
        revision["classifier_sha256"] = "f" * 64
    if mutation == "prior-hold":
        hold["error"] = "changed"
    if mutation == "extra-label":
        control["held_reinspection"]["revisions"]["other"] = copy.deepcopy(revision)
    definitions = {"a": {"label": "a", "initial_status": "pending"}}
    with pytest.raises(ValueError):
        q.held_reinspection_labels(control, tranche, definitions, {})
        q.held_reinspection_token(control, "a", q.budget_digest(tranche), hold)


@pytest.fixture
def reviewed_dependency_failure():
    root = (
        Path(__file__).resolve().parents[2] / "artifacts/integration/frisson_ai/"
        "faynt-d0-632-980-v1/recovery-audit/extended_gm_v1_samus"
    )
    if not (root / "manifest.json").is_file():
        pytest.skip("requires the exact externally retained dependency failure fixture")
    manifest = json.loads((root / "manifest.json").read_bytes())
    return manifest, root


def test_reviewed_dependency_failure_requires_exact_retained_manifest(reviewed_dependency_failure):
    manifest, root = reviewed_dependency_failure
    assert q.digest(manifest) == "818cd7acdcbaf7513cd9dd265e15ff2655bc50085584cd0401852999f41f371b"
    assert q.verified_failure_retry_reason(manifest, root) == (
        "reviewed-zero-gameplay-dependency-acquisition-failure"
    )
    assert authorize(reason="reviewed-zero-gameplay-dependency-acquisition-failure")["attempt_number"] == 2
    assert (
        authorize(number=2, reason="reviewed-zero-gameplay-dependency-acquisition-failure")["attempt_number"]
        == 3
    )
    with pytest.raises(ValueError, match="three-attempt"):
        authorize(number=3, reason="reviewed-zero-gameplay-dependency-acquisition-failure")


@pytest.mark.parametrize(
    "mutation", ["file-hash", "file-length", "plan", "call", "scope", "execution", "status"]
)
def test_dependency_exception_does_not_generalize_to_other_failures(reviewed_dependency_failure, mutation):
    manifest, root = reviewed_dependency_failure
    if mutation == "file-hash":
        manifest["files"][0]["sha256"] = "0" * 64
    if mutation == "file-length":
        manifest["files"][0]["bytes"] += 1
    if mutation == "plan":
        manifest["plan_sha256"] = "0" * 64
    if mutation == "call":
        manifest["call_id"] = "fc-other"
    if mutation == "scope":
        manifest["scope"]["seed"] += 1
    if mutation == "execution":
        manifest["execution"]["execution_id"] = "0" * 32
    if mutation == "status":
        manifest["status"] = "full-policy-durable-game-passed"
    assert q.verified_failure_retry_reason(manifest, root) is None


def test_dependency_exception_cannot_admit_failed_result(reviewed_dependency_failure):
    manifest, root = reviewed_dependency_failure
    retained = copy.deepcopy(manifest)
    retained["storage_path"] = root.name
    d = types.SimpleNamespace(
        MOUNT=root.parent, SUCCESS="full-policy-durable-game-passed", check_manifest=lambda *args: None
    )
    with pytest.raises(ValueError, match="without score admission"):
        q.native_acceptance(d, None, None, retained)
