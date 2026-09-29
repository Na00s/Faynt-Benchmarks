"""Synthetic live wrapper checks. No ROM, network, package install or Dolphin."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_live_pilot as p


def row(remote, data=b"x"):
    return {"local": "/fixture/" + hashlib.sha256(remote.encode()).hexdigest(), "remote": remote,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fake_plan():
    return p.definition([row(n) for n in sorted(p.expected_destinations())], "/research/test", "a"*64, "b"*64)


def test_scope_complete_sources_and_fixed_resources():
    plan = fake_plan(); p.check_layout(plan)
    assert set(p.runtime.SOURCE_PATHS) <= set(plan["source_paths"])
    assert set(p.tape.SOURCE_PATHS) <= set(plan["source_paths"])
    assert "scripts/modal_panel_live_pilot.py" in plan["source_paths"]
    assert len(plan["source_paths"]) == len(set(plan["source_paths"]))
    assert plan["scope"]["controller_cases"] == 2
    assert plan["scope"]["command_frames_per_case"] == 320
    assert plan["scope"]["policy_inference_calls"] == plan["scope"]["checkpoints"] == 0
    assert plan["scope"]["save_slp"] is True and plan["scope"]["save_video"] is False
    assert plan["limits"]["cpu"] == [4,4] and plan["limits"]["memory_mib"] == [16384,16384]
    assert plan["limits"]["nonpreemptible"] is True
    assert plan["fresh_runtime"]["compiled_extension_byte_equality_to_prior"] is False
    assert p.WORK.parts.count("research") == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "oversize", "flag", "boolbytes"])
def test_layout_rejects_malformed_or_widened_upload(change):
    plan = fake_plan()
    if change == "missing": plan["uploads"].pop()
    elif change == "duplicate": plan["uploads"][0] = plan["uploads"][1]
    elif change == "extra": plan["uploads"].append(row("/opt/secret"))
    elif change == "oversize": plan["uploads"][0]["bytes"] = p.UPLOAD_CAP
    elif change == "flag": plan["scope"]["save_video"] = True
    else: plan["uploads"][0]["bytes"] = True
    with pytest.raises(ValueError): p.check_layout(plan)


def approval(plan):
    return {"schema": p.APPROVAL_SCHEMA, "decision": "authorized", "scope": p.fixed_scope(), "bindings": {
        "game_image_sha256": p.tape.IMAGE["sha256"], "package_archive_sha256": p.host.ARCHIVE[1],
        "runtime_plan_sha256": plan["prior_runtime_plan_sha256"], "policy_plan_sha256": plan["policy_plan_sha256"]}}


@pytest.mark.parametrize("field", [None, "decision", "scope", "bindings"])
def test_explicit_authorization_bound_to_exact_inputs(field):
    plan = fake_plan(); value = approval(plan)
    if field is None: p.authorization(value, plan)
    else:
        value[field] = {}
        with pytest.raises(ValueError): p.authorization(value, plan)


def test_no_model_or_checkpoint_apis():
    tree = ast.parse(Path(p.__file__).read_text())
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"FrissonPolicySession", "load_policy", "torch_load", "run_game", "record_video"}
    assert "run_prepared_plan" in calls and "perform" in calls
    assert not any(n.endswith(".pt") for n in p.expected_destinations())


@pytest.mark.parametrize("name,allowed", [
    ("receipt.json", True), ("runtime/logs/001.log", True), ("admissions/tape-plan.json", True),
    ("controller-tape/case-0/replays/Game_20260101T000000.slp", True),
    ("controller-tape/case-1/controller_tape_trace.jsonl", True),
    ("../secret", False), ("/receipt.json", False), ("controller-tape/case-2/case.json", False),
    ("controller-tape/case-0/User/Config/Dolphin.ini", False), ("checkpoint.pt", False),
    ("controller-tape/case-0/replays/game.mp4", False), ("controller-tape/case-0/replays/../secret.slp", False)])
def test_output_allowlist(name, allowed):
    assert p.allowed_output(name) is allowed


EXECUTION = {"task_id": "ta-test", "execution_id": "a"*32}


def frames(body=b"{}", name="receipt.json"):
    result = {"schema": p.SCHEMA, "status": "failed", "plan_sha256": "a"*64, "scope": p.fixed_scope(), "execution": EXECUTION,
              "files": [{"name": name, "body": body, "sha256": hashlib.sha256(body).hexdigest(),
                 "original_bytes": len(body), "original_sha256": hashlib.sha256(body).hexdigest(), "truncated_tail": False}]}
    return [{"kind": "hello", "seq": 0, "plan_sha256": "a"*64, "call_id": "fc-test", "execution": EXECUTION}, *p.wire.output_frames(result)]


def test_shared_stream_saves_exact_bytes_without_admission(tmp_path):
    terminal, files, records = p.receive_files(tmp_path, frames(), "a"*64, "ap-test", p.time.time()+60)
    assert terminal["status"] == "failed" and files["receipt.json"].read_bytes() == b"{}"
    assert records[0]["sha256"] == hashlib.sha256(b"{}").hexdigest()


@pytest.mark.parametrize("change", ["duplicate", "booloffset", "wronghash", "truncatedslp", "execution", "missingdone", "path", "total", "file"])
def test_stream_rejects_malformed(tmp_path, change):
    value = frames(); kwargs = {}
    if change == "duplicate":
        value = value[:-1] + copy.deepcopy(value[1:])
        for i, f in enumerate(value): f["seq"] = i
    elif change == "booloffset": value[2]["offset"] = False
    elif change == "wronghash": value[1]["file"]["original_sha256"] = "f"*64
    elif change == "truncatedslp": value[1]["file"].update(name="controller-tape/case-0/replays/x.slp", truncated_tail=True)
    elif change == "execution": value[-1]["result"]["execution"] = {**EXECUTION, "task_id": "ta-other"}
    elif change == "missingdone": value.pop()
    elif change == "path": value[1]["file"]["name"] = "../secret"
    elif change == "total": kwargs["total_cap"] = 1
    else: kwargs["file_cap"] = 1
    with pytest.raises(ValueError): p.receive_files(tmp_path, value, "a"*64, "ap-test", p.time.time()+60, **kwargs)


def test_sigterm_unwinds_and_restores_handler():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with p.terminate_cleanup(): signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) == before


def test_success_without_actual_tape_evidence_rejected():
    terminal = {"schema": p.SCHEMA, "status": "controller-tape-qualified", "scope": p.fixed_scope(), "plan_sha256": "a"*64}
    with pytest.raises((ValueError, KeyError)): p.validate_terminal(terminal, {}, "a"*64, fake_plan())


def test_actual_successful_runtime_receipts_bind_without_rom():
    directory = p.COMPAT / "runtime-pilot-v2"
    if not (directory / "result.json").exists(): pytest.skip("retained qualification absent")
    plan = p.policy.bounded_json(directory / "plan.json")
    data = {name: p.policy.bounded_json(directory / name) for name in p.PRIOR_NAMES}
    p.validate_prior(plan, data)
    changed = copy.deepcopy(data); changed["result.json"]["execution"] = {**changed["result.json"]["execution"], "task_id": "ta-wrong"}
    with pytest.raises(ValueError): p.validate_prior(plan, changed)


def test_sdk_configuration_is_lazy_and_single_worker(tmp_path):
    modal = pytest.importorskip("modal")
    app, function, image = p.configure_app(modal, tmp_path / "plan.json", {"uploads": []})
    assert app.name == p.APP_NAME
    assert function.get_raw_f().__name__ == p.remote_live.__name__
    assert image is not None


def test_fresh_runtime_failure_does_not_call_tape(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "WORK", tmp_path / "research")
    monkeypatch.setattr(p.policy, "bounded_json", lambda path: {})
    monkeypatch.setattr(p.runtime, "perform", lambda *a: {"status": "failed"})
    monkeypatch.setattr(p, "assemble_admissions", lambda *a, **k: pytest.fail("tape reached"))
    result = p.owner(fake_plan(), "a"*64, p.time.time()+1000, "im-test")
    assert result["status"] == "failed" and "fresh runtime" in result["error"]


def test_owner_supervision_reaps_after_group_cleanup(tmp_path, monkeypatch):
    events = []
    class Process:
        pid = 12345
        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True
            assert kwargs["env"]["HOME"] == p.os.environ["HOME"]
        def wait(self, timeout): events.append("wait"); return 0
    monkeypatch.setattr(p.platform, "system", lambda: "Linux")
    monkeypatch.setattr(p.subprocess, "Popen", Process)
    monkeypatch.setattr(p.os, "waitid", lambda *a: SimpleNamespace(si_pid=12345), raising=False)
    for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT"): monkeypatch.setattr(p.os, name, 1, raising=False)
    monkeypatch.setattr(p.os, "killpg", lambda pid, sig: events.append(sig))
    value = p.supervise_owner(["python", "owner.py"], p.time.time()+10, tmp_path / "owner")
    assert events == [signal.SIGTERM, signal.SIGKILL, "wait"]
    assert value["unreaped_leader_group_guard"] and value["terminated_owned_process_group"]
    assert value["native_user_environment"] == {"HOME": p.os.environ["HOME"]}


@pytest.mark.parametrize("state", ["valid", "absent", "empty", "different", "symlink"])
def test_actual_account_home_is_preserved_without_repurposing(tmp_path, monkeypatch, state):
    import pwd
    account = tmp_path / "account"; account.mkdir()
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(account)))
    monkeypatch.setenv("HOME", str(account))
    if state == "absent": monkeypatch.delenv("HOME")
    elif state == "empty": monkeypatch.setenv("HOME", "")
    elif state == "different": monkeypatch.setenv("HOME", str(tmp_path))
    elif state == "symlink":
        alias = tmp_path / "alias"; alias.symlink_to(account, target_is_directory=True)
        monkeypatch.setenv("HOME", str(alias))
        monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(alias)))
    before = dict(p.os.environ)
    if state == "valid": assert p.native_user_environment() == {"HOME": str(account)}
    else:
        with pytest.raises(ValueError): p.native_user_environment()
    assert dict(p.os.environ) == before


def test_both_native_child_launches_preserve_account_environment():
    import modal_panel_game_pilot as full_game
    import inspect
    assert "native_user_environment()" in inspect.getsource(p.owner)
    assert "live.native_user_environment()" in inspect.getsource(full_game.owner)


def test_assemble_fresh_evidence_expanded_sources_and_native_tape(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"; inputs.mkdir()
    remote = tmp_path / "root"; remote.mkdir()
    monkeypatch.setattr(p, "INPUT", inputs); monkeypatch.setattr(p, "REMOTE", remote)
    entry = remote / "scripts/modal_panel_live_pilot.py"; entry.parent.mkdir(); entry.write_text("# owner\n")
    monkeypatch.setattr(p, "ENTRY", entry)
    runtime_inputs = tmp_path / "runtime"; runtime_inputs.mkdir()
    monkeypatch.setattr(p.runtime, "INPUT", runtime_inputs)
    (runtime_inputs / "lock.json").write_text("{}"); (runtime_inputs / "pins.lock").write_text("pin==1\n")
    prior = tmp_path / "prior.json"; prior.write_text('{"uploads": []}')
    monkeypatch.setattr(p.runtime, "PLAN", prior)
    monkeypatch.setattr(p.runtime, "validate_qualified_evidence", lambda *a: None)
    for name in p.context.LIMITED_EVIDENCE_SHA256:
        path = inputs / "policy" / (name + ".json"); path.parent.mkdir(exist_ok=True); path.write_text("{}")
    monkeypatch.setattr(p.context, "LIMITED_EVIDENCE_SHA256", {name: hashlib.sha256(b"{}").hexdigest()
                       for name in p.context.LIMITED_EVIDENCE_SHA256})
    (inputs / "authorization.json").write_text("{}")
    plan = fake_plan()
    plan.update(runtime_image_id="im-fixture", outer_plan_sha256="c"*64)
    fresh = {"status": "runtime-qualified-gameplay-unadmitted", "qualification": {"evidence": {
        "installed_runtime": {"fresh": True}, "ubjson_extension": {"fresh_identity": True}, "os_packages": {"a": "1"}}}}
    prepared, descriptor = p.assemble_admissions(plan, fresh, tmp_path / "admissions",
        created_at=1000, expires_at=1600, parent_pid=42)
    p.tape.validate_plan(prepared, expected_sha256=p.tape.identity(prepared)["sha256"], now=1000)
    assert prepared["runtime_image"]["manifest_sha256"] == "c"*64
    assert prepared["scientific"]["policy_inference_calls"] == 0
    assert [c["a_port"] for c in prepared["cases"]] == [1,2]
    assert descriptor["owner_declaration"]["parent_pid"] == 42
    assert descriptor["extension_binary_identical_to_prior"] == "unassessed"
    paths = descriptor["evidence_paths"]["linux_runtime"]
    assert set(paths) == {"dependency_lock", "dependency_pins", "source_manifest", "installed_runtime", "ubjson_extension", "os_packages"}
    assert json.loads(Path(paths["source_manifest"]).read_text()) == p.source_manifest(plan)
    assert json.loads(Path(paths["ubjson_extension"]).read_text()) == {"fresh_identity": True}
