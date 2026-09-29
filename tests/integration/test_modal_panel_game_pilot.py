"""Inert full-policy pilot contracts. No checkpoints, ROM or emulator loaded."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_game_pilot as p


def games():
    return [{"label": f"sp21-75m-fox_d21_ditto_v4-supported-mirror-fox-final_destination-p{port}",
        "profile": "75m", "release": "fox_d21_ditto_v4", "block": "supported-mirror",
        "frisson_character": "FOX", "slippi_character": "FOX", "player_name": "Cody",
        "stage": "FINAL_DESTINATION", "seed": 1971985417, "frisson_port": port, "slippi_port": 3-port,
        "save_slp": True, "save_video": False, "pair_id": "11c2a0db80872079",
        "slippi_character_in_deployed_roster": True} for port in (2,1)]


def row(name):
    identity = p.CHECKPOINTS.get(name.removeprefix(str(p.REMOTE)+"/"), {"bytes": 1, "sha256": "a"*64})
    if name == str(p.INPUT / "source.json"): identity = {"bytes": 198071, "sha256": p.SNAPSHOT_SHA}
    return {"local": "/fixture/" + hashlib.sha256(name.encode()).hexdigest(), "remote": name, **identity}


def plan():
    proof = {"result_sha256": "b"*64, "files": {"receipt.json": {"bytes": 1, "sha256": "c"*64}}}
    return p.definition([row(n) for n in sorted(p.destinations(proof))], "/research/game", "d"*64, proof, "e"*64, games())


def test_retained_successful_tape_preparation_accepts_json_string_paths():
    # Optional real retained metadata regression. This hashes explicit local
    # inputs and builds an in-memory plan, with no output or runtime execution.
    request = p.COMPAT / "FULL_GAME_INPUT_PATHS.json"
    if not request.exists(): pytest.skip("retained full-game input request unavailable")
    spec = json.loads(request.read_bytes())
    if not (Path(spec["tape_directory"]) / "result.json").exists():
        pytest.skip("successful live tape unavailable")
    destination = p.COMPAT / "test-preparation-no-output"
    assert not destination.exists()
    value = p.make_plan(str(destination), **spec)
    p.check_layout(value)
    assert value["scope"] == p.fixed_scope()
    assert not destination.exists()


def test_explicit_scope_layout_and_native_sources():
    value = plan(); p.check_layout(value)
    assert p.SOURCE_PATHS == tuple(dict.fromkeys(p.SOURCE_PATHS))
    assert set(p.live.SOURCE_PATHS) <= set(p.SOURCE_PATHS)
    for name in ("frisson_slippi_match", "game_bundle", "slippi_compatibility", "play", "replay_result"):
        assert f"src/melee_policy/integration/{name}.py" in p.SOURCE_PATHS
    assert "requirements-e010.lock" in p.SOURCE_PATHS
    assert "scripts/run_final_75m_vs_slippi_ai_character_sweep.py" in p.SOURCE_PATHS
    assert value["scope"]["competitive_games"] == 0
    assert value["scope"]["save_slp"] and not value["scope"]["save_video"]
    assert value["scope"]["policy_inference"] is True
    assert value["scope"]["frisson_physical_ports"] == [2,1]
    assert value["limits"]["cpu"] == [4,4]
    assert value["limits"]["memory_mib"] == [16384,16384]
    assert p.PROJECT.parent == p.PARENT and p.PROJECT.name == "melee-policy"
    assert p.PARENT.is_relative_to(p.WORK)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "boolean", "bytes", "model", "scope", "ports", "seed", "proofpath"])
def test_plan_rejects_drift(change):
    value = plan()
    if change == "missing": value["uploads"].pop()
    elif change == "duplicate": value["uploads"][0] = value["uploads"][1]
    elif change == "extra": value["uploads"].append(row("/opt/secret"))
    elif change == "boolean": value["uploads"][0]["bytes"] = True
    elif change == "bytes": value["uploads"][0]["bytes"] = p.UPLOAD_CAP
    elif change == "model":
        next(r for r in value["uploads"] if r["remote"].endswith("step222.pt"))["sha256"] = "0"*64
    elif change == "scope": value["scope"]["save_video"] = True
    elif change == "ports": value["games"] = value["games"][::-1]
    elif change == "seed": value["games"][0]["seed"] += 1
    else: value["tape_prerequisite"]["files"]["../secret"] = {"bytes": 1, "sha256": "a"*64}
    with pytest.raises(ValueError): p.check_layout(value)


def test_empty_prior_log_and_full_trace_identity_are_allowed():
    value = plan()
    for name, size in (("runtime/logs/002.log", 0), ("controller-tape/case-0/controller_tape_trace.jsonl", 3*1024**2)):
        value["tape_prerequisite"]["files"][name] = {"bytes": size, "sha256": "f"*64}
        value["uploads"].append({**row(str(p.INPUT / "tape/returned" / name)), "bytes": size, "sha256": "f"*64})
    p.check_layout(value)
    next(r for r in value["uploads"] if r["remote"] == str(p.INPUT / "source.json"))["bytes"] = 0
    with pytest.raises(ValueError): p.check_layout(value)


def test_pair_exact_current_manifest_read_only():
    path = ROOT / p.PANEL
    assert p.wire.digest(path) == p.PANEL_SHA
    selected = p.selected_games(json.loads(path.read_bytes()))
    assert [g["frisson_port"] for g in selected] == [2,1]
    assert selected[0]["pair_id"] == selected[1]["pair_id"]
    assert all(g["profile"] == "75m" and g["save_video"] is False for g in selected)


def approval(value):
    return {"schema": p.APPROVAL_SCHEMA, "decision": "authorized", "scope": p.fixed_scope(), "limits": p.limits(),
        "bindings": {"tape_plan_sha256": value["prior_tape_plan_sha256"], "tape_result_sha256": value["tape_prerequisite"]["result_sha256"],
          "policy_plan_sha256": value["policy_plan_sha256"], "checkpoints": p.CHECKPOINTS,
          "source_snapshot_sha256": p.SNAPSHOT_SHA, "panel_manifest_sha256": p.PANEL_SHA,
          "game_image_sha256": p.tape.IMAGE["sha256"], "package_archive_sha256": p.host.ARCHIVE[1]}}


@pytest.mark.parametrize("field", [None, "decision", "scope", "limits", "bindings"])
def test_saved_explicit_approval(field):
    value = plan(); auth = approval(value)
    if field is None: p.authorization(auth, value)
    else:
        auth[field] = {}
        with pytest.raises(ValueError): p.authorization(auth, value)


@pytest.mark.parametrize("name,expected", [
    ("receipt.json", True), ("game-0/result.json", True), ("game-command-1/logs/001.log", True),
    ("project/artifacts/integration/frisson_ai/modal-compat-75m-d21-fd-p1/summary.json", True),
    ("project/artifacts/integration/frisson_ai/modal-compat-75m-d21-fd-p2/replays/Game_1.slp", True),
    ("project/artifacts/integration/frisson_ai/modal-compat-75m-d21-fd-p1/game/game.slp", True),
    ("project/artifacts/integration/frisson_ai/modal-compat-75m-d21-fd-p1/User/Config/Dolphin.ini", False),
    ("../secret", False), ("/receipt.json", False), ("game-2/result.json", False),
    ("game-0/checkpoint.pt", False), ("project/.e010-cache/model.pt", False),
    ("project/artifacts/integration/frisson_ai/modal-compat-75m-d21-fd-p1/game/game.mp4", False),
    ("project/artifacts/integration/frisson_ai/sp21-existing/summary.json", False)])
def test_result_allowlist(name, expected):
    assert p.allowed_output(name) is expected


def test_copy_exact_exclusive_bounded(tmp_path):
    source = tmp_path / "input"; source.write_bytes(b"example")
    expected = {"bytes": 7, "sha256": hashlib.sha256(b"example").hexdigest()}
    output = tmp_path / "owned/asset"
    p.checked_copy(source, output, expected, p.time.time()+30)
    assert output.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError): p.checked_copy(source, output, expected, p.time.time()+30)
    with pytest.raises(ValueError): p.checked_copy(source, tmp_path / "wrong", {**expected, "sha256": "a"*64}, p.time.time()+30)


def test_copy_refuses_symlink_and_expiry(tmp_path):
    source = tmp_path / "input"; source.write_bytes(b"x")
    expected = {"bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
    parent = tmp_path / "outside"; parent.mkdir()
    (tmp_path / "link").symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError): p.checked_copy(source, tmp_path / "link/x", expected, p.time.time()+30)
    with pytest.raises(TimeoutError): p.checked_copy(source, tmp_path / "expired", expected, p.time.time()-1)


def test_remote_tape_projection_preserves_function_and_plan(monkeypatch):
    original = {"uploads": [{"local": "/mac/lock", "remote": "/linux/lock", "bytes": 1, "sha256": "a"*64}]}
    prior = {"uploads": [{"local": "/mac/runtime-plan", "remote": str(p.runtime.PLAN)}], "prior_runtime_plan_sha256": "a"*64}
    saved = copy.deepcopy(original)
    monkeypatch.setattr(p.runtime, "verify_plan", lambda path, sha, remote: copy.deepcopy(original))
    # Execute a synthetic frozen validator through the same isolated-code path.
    environment = {"policy": p.policy}
    exec("def validator(result, files, sha, prior):\n"
         "    value = policy.bounded_json('/mac/runtime-plan')\n"
         "    assert value['uploads'][0]['local'] == '/linux/lock'\n"
         "    if result.get('status') != 'passed': raise ValueError('scientific gate')\n", environment)
    function = environment["validator"]
    monkeypatch.setattr(p.live, "validate_terminal", function)
    global_before = function.__globals__["policy"]
    p.validate_tape_terminal({"status": "passed"}, {}, "b"*64, prior, remote=True)
    with pytest.raises(ValueError, match="scientific gate"):
        p.validate_tape_terminal({"status": "failed"}, {}, "b"*64, prior, remote=True)
    assert original == saved and function.__globals__["policy"] is global_before
    assert prior["uploads"][0]["local"] == "/mac/runtime-plan"


def test_native_acceptance_module_preserves_source_and_host_globals(tmp_path):
    source = tmp_path / "validator.py"
    source.write_text("from pathlib import Path\nROOT=Path('/original')\n"
                      "def validate_result(game, attempt):\n"
                      "    return {'root':str(ROOT), 'manifest':str(OUTPUT), 'label':attempt['label'], 'game':game}\n")
    before = source.read_bytes()
    result = p.native_validation({"x": 1}, "label", tmp_path / "returned/project", tmp_path / "inputs/manifest.json", source)
    assert result == {"root": str(tmp_path / "returned/project"), "manifest": str(tmp_path / "inputs"), "label": "label", "game": {"x":1}}
    assert source.read_bytes() == before


def test_admission_assembly_is_full_policy_and_bound_to_explicit_parent(tmp_path, monkeypatch):
    runtime_input, live_input, game_input = (tmp_path / n for n in ("runtime", "live", "game"))
    for path in (runtime_input, live_input / "policy", game_input): path.mkdir(parents=True)
    for name in p.context.LIMITED_EVIDENCE_SHA256: (live_input / "policy" / (name+".json")).write_text("{}")
    for name in ("lock.json", "pins.lock"): (runtime_input / name).write_text("{}")
    (game_input / "authorization.json").write_text("{}")
    prior = tmp_path / "runtime-plan.json"; prior.write_text('{"uploads":[]}')
    entry = tmp_path / "entry.py"; entry.write_text("# exact owner\n")
    monkeypatch.setattr(p.runtime, "INPUT", runtime_input); monkeypatch.setattr(p.runtime, "PLAN", prior)
    monkeypatch.setattr(p.live, "INPUT", live_input); monkeypatch.setattr(p, "INPUT", game_input)
    monkeypatch.setattr(p, "ENTRY", entry)
    monkeypatch.setattr(p.runtime, "validate_qualified_evidence", lambda *a: None)
    monkeypatch.setattr(p.context, "LIMITED_EVIDENCE_SHA256", {n: hashlib.sha256(b"{}").hexdigest() for n in p.context.LIMITED_EVIDENCE_SHA256})
    value = {**plan(), "runtime_image_id": "im-fixture", "outer_plan_sha256": "f"*64}
    fresh = {"status": "runtime-qualified-gameplay-unadmitted", "qualification": {"evidence": {
        "installed_runtime": {}, "ubjson_extension": {}, "os_packages": {}}}}
    prepared, descriptor = p.assemble_admissions(value, fresh, tmp_path / "admissions", index=1,
                                               created_at=1000, expires_at=1900, parent_pid=1234)
    assert prepared["schema"] == p.SCHEMA + ".execution"
    assert prepared["agent_kind"] == "native-frisson-versus-slippi-policy"
    assert prepared["game"] == value["games"][1] and prepared["artifact_label"] == p.LABELS[1]
    assert prepared["cases"] == [{"slippi_port": 51442}]
    assert prepared["runtime_image"]["manifest_sha256"] == "f"*64
    assert prepared["benchmark_ledger_writes"] is False
    assert descriptor["owner_declaration"]["parent_pid"] == 1234
    assert descriptor["owner_declaration"]["plan_sha256"] == p.tape.identity(prepared)["sha256"]
    assert set(descriptor["admission_paths"]) == set(p.context.KINDS)
    policy_admission = json.loads(Path(descriptor["admission_paths"]["policy_parity"]).read_bytes())
    assert policy_admission["decision"] == p.context.LIMITED_DECISION
    assert set(policy_admission["evidence"]) == set(p.context.LIMITED_EVIDENCE_SHA256)
    with pytest.raises(FileExistsError):
        p.assemble_admissions(value, fresh, tmp_path / "admissions", index=1, created_at=1000, expires_at=1900, parent_pid=1234)
    with pytest.raises(ValueError):
        p.assemble_admissions(value, fresh, tmp_path / "too-long", index=1, created_at=1000, expires_at=1901, parent_pid=1234)


def test_shared_large_stream_retains_exact_science_files(tmp_path):
    name = f"project/artifacts/integration/frisson_ai/{p.LABELS[0]}/controller_trace.jsonl"
    body = b'{"example":1}\n'*10000
    execution = {"task_id": "ta-test", "execution_id": "a"*32}
    result = {"schema": p.SCHEMA, "status": "failed", "scope": p.fixed_scope(), "plan_sha256": "a"*64, "execution": execution,
              "files": [{"name": name, "body": body, "sha256": hashlib.sha256(body).hexdigest(),
                         "original_bytes": len(body), "original_sha256": hashlib.sha256(body).hexdigest(), "truncated_tail": False}]}
    frames = [{"kind": "hello", "seq": 0, "plan_sha256": "a"*64, "call_id": "fc-test", "execution": execution},
              *p.wire.output_frames(result)]
    terminal, files, records = p.live.receive_files(tmp_path, frames, "a"*64, "ap-test", p.time.time()+30,
        name_check=p.allowed_output, total_cap=p.RETURN_CAP, file_cap=p.FILE_CAP, file_count=p.FILE_COUNT)
    assert files[name].read_bytes() == body and records[0]["truncated_tail"] is False
    assert terminal["status"] == "failed"


def completed_pair_fixture(tmp_path, monkeypatch):
    """Real receipt/hash/cleanup schemas with synthetic policy/runtime evidence."""
    same = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(p.context, "LIMITED_EVIDENCE_SHA256", {n: same for n in p.context.LIMITED_EVIDENCE_SHA256})
    qualification = {"classification": p.context.LIMITED_CLASSIFICATION, "strict_policy_parity_passed": False}
    monkeypatch.setattr(p.context, "bind_policy_evidence", lambda *a, **k: qualification)
    runtime_calls, native_calls = [], []
    monkeypatch.setattr(p.runtime, "validate_terminal", lambda *a: runtime_calls.append(a))
    expected_acceptance = {"winner": "frisson-ai", "win": True, "stocks_taken": 4, "stocks_conceded": 2,
                           "summary": "/cloud/summary.json", "replay": "/cloud/game.slp"}
    def native(game, label, project, manifest, validator):
        assert project == tmp_path / "returned/project"
        native_calls.append((game, label))
        return {**expected_acceptance, "summary": str(project / "summary.json"), "replay": str(project / "game.slp")}
    monkeypatch.setattr(p, "native_validation", native)
    value = plan(); files = {}
    def put(name, body):
        path = tmp_path / "returned" / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body); files[name] = path
        return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    def save(name, body): return put(name, p.wire.canonical(body))
    for index, record in enumerate(value["uploads"]):
        path = tmp_path / "inputs" / str(index); path.parent.mkdir(exist_ok=True)
        body = (p.wire.canonical({"prior_runtime_plan_sha256": "r"*64}) if record["remote"] == str(p.live.PLAN) else b"{}")
        path.write_bytes(body); record.update(local=str(path), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    inputs = {r["remote"]: Path(r["local"]) for r in value["uploads"]}
    source = {str(Path(r["remote"]).relative_to(p.REMOTE)): {k:r[k] for k in ("bytes", "sha256")}
              for r in value["uploads"] if r["remote"] in {str(p.REMOTE / n) for n in p.SOURCE_PATHS}}
    fresh = {"status": "runtime-qualified-gameplay-unadmitted", "qualification": {"evidence": {
        "installed_runtime": {}, "ubjson_extension": {}, "os_packages": {}}}}
    save("runtime/receipt.json", fresh); save("runtime/qualification.json", fresh["qualification"])
    supervisor = {"pid": 1234, "exit_code": 0, "unreaped_leader_group_guard": True, "terminated_owned_process_group": True}
    save("supervisor/receipt.json", supervisor)
    result = {"schema": p.SCHEMA, "status": "full-policy-compatibility-pair-passed", "plan_sha256": "a"*64,
        "scope": p.fixed_scope(), "fresh_runtime_status": fresh["status"], "runtime_image_id": "im-fixture", "games": []}
    for i in (0,1):
        image = {"image_id": "im-fixture", "manifest_sha256": "a"*64, "platform": "linux_x86_64"}
        bindings = {"policy_plan_sha256": value["policy_plan_sha256"], "package_archive_sha256": p.host.ARCHIVE[1],
                    "game_image_sha256": p.tape.IMAGE["sha256"], "runtime_image": image}
        evidences = {"policy_parity": {n: {"bytes": 2, "sha256": same} for n in p.context.LIMITED_EVIDENCE_SHA256},
                    "game_image_access": {"authorization": {"bytes": 2, "sha256": same}},
                    "live_execution": {"authorization": {"bytes": 2, "sha256": same}, "owner_entrypoint": {"bytes": 2, "sha256": same}},
                    "linux_runtime": {}}
        bodies = {"dependency_lock": inputs[str(p.runtime.INPUT / "lock.json")].read_bytes(),
                  "dependency_pins": inputs[str(p.runtime.INPUT / "pins.lock")].read_bytes(),
                  "source_manifest": p.wire.canonical(p.source_manifest(value)),
                  **{n: p.wire.canonical(fresh["qualification"]["evidence"][n]) for n in ("installed_runtime", "ubjson_extension", "os_packages")}}
        for name, body in bodies.items(): evidences["linux_runtime"][name] = put(f"game-{i}/{name}.json", body)
        admissions = {}
        for kind, evidence in evidences.items():
            body = {"schema": p.context.SCHEMA, "kind": kind, "decision": p.context.LIMITED_DECISION if kind=="policy_parity" else p.context.KINDS[kind],
                    "bindings": bindings, "evidence": evidence}
            admissions[kind] = save(f"game-{i}/{kind}-admission.json", body)
        prepared = {"schema": p.SCHEMA+".execution", "index": i, "game": value["games"][i], "scope": p.fixed_scope(),
                    "artifact_label": p.LABELS[i], "cases": [{"slippi_port":51442}], "agent_kind":"native-frisson-versus-slippi-policy",
                    "source_identities": source, "runtime_image": image, "required_admissions": admissions}
        plan_sha = p.tape.identity(prepared)["sha256"]
        owner = {"status":"owned-linux-tape-context-ready", "parent_terminal_cleanup_verified":False,
                 "process":{"pid":2000+i, "parent_pid":1234, "pgid":2000+i, "sid":2000+i},
                 "parent_declaration":{"expires_at":1900}, "plan_sha256":plan_sha, "admissions":admissions,
                 "runtime_image":image, "policy_qualification":qualification}
        command = {"pid":2000+i, "command":[str(p.runtime.WORK/"venv/bin/python"),str(p.ENTRY),"--game-child",str(i)],
                   "exit_code":0, "unreaped_leader_group_guard":True, "terminated_owned_process_group":True,
                   "log_sha256":"f"*64, "started_unix":1000, "ended_unix":1500}
        child = {"schema":p.SCHEMA+".game", "status":"native-full-policy-game-passed", "game":value["games"][i],
                 "artifact_label":p.LABELS[i], "plan_sha256":plan_sha, "owner":owner, "native_acceptance":expected_acceptance,
                 "launch_evidence":{"adjacent_rust_loaded":True, "pgid":2000+i,"sid":2000+i,
                                    "executable":{"bytes":p.host.EXECUTABLE[0],"sha256":p.host.EXECUTABLE[1]},"udp":[{"port":51442}]},
                 "host_adapter":{"schema":p.match_host.SCHEMA,"source_identities":{n:source[n] for n in p.match_host.SOURCE_HASHES},
                                 "hooks":list(p.match_host.HOOKS),"policy_loop_changes":False,"parent_terminal_cleanup_verified":False,
                                 "policy_qualification":qualification}}
        save(f"game-{i}/result.json",child); save(f"game-{i}/execution-plan.json",prepared)
        save(f"game-command-{i}/commands.json",[command])
        save(f"project/artifacts/integration/frisson_ai/{p.LABELS[i]}/summary.json",{})
        result["games"].append({"index":i,"child":child,"commands":[command],"cleanup":p.context.validate_parent_terminal(owner,command)})
    save("receipt.json",result)
    terminal = {"schema":p.SCHEMA,"plan_sha256":"a"*64,"scope":p.fixed_scope(),"status":result["status"],
                "execution":{"task_id":"ta-test","execution_id":"a"*32}}
    return terminal, files, value, runtime_calls, native_calls


def test_successful_terminal_reproduces_native_acceptance_for_both_ports(tmp_path, monkeypatch):
    terminal, files, value, runtime_calls, native_calls = completed_pair_fixture(tmp_path, monkeypatch)
    p.validate_terminal(terminal, files, "a"*64, value)
    assert len(runtime_calls)==1 and [x[0]["frisson_port"] for x in native_calls]==[2,1]


@pytest.mark.parametrize("change", ["missing_second", "scope", "port", "source", "cleanup", "image", "admission", "runtime_evidence", "udp", "host_hook", "native_score"])
def test_terminal_rejects_each_bound_pair_mutation(tmp_path, monkeypatch, change):
    terminal, files, value, _, _ = completed_pair_fixture(tmp_path, monkeypatch)
    result = json.loads(files["receipt.json"].read_bytes())
    if change == "missing_second": result["games"].pop()
    elif change == "scope": result["scope"]["competitive_games"] = 2
    else:
        name = "game-0/result.json"; child = result["games"][0]["child"]
        if change == "port": child["game"]["frisson_port"] = 1
        elif change == "source":
            name = "game-0/execution-plan.json"; body=json.loads(files[name].read_bytes()); body["source_identities"]={}
        elif change == "cleanup": result["games"][0]["cleanup"]["parent_terminal_cleanup_verified"]=False
        elif change == "image": child["owner"]["runtime_image"]["image_id"]="im-other"
        elif change == "admission":
            name="game-0/live_execution-admission.json"; body=json.loads(files[name].read_bytes()); body["decision"]="pending"
        elif change == "runtime_evidence": name="game-0/dependency_lock.json"; body={"changed":True}
        elif change == "udp": child["launch_evidence"]["udp"][0]["port"]=51441
        elif change == "host_hook": child["host_adapter"]["hooks"].append("_run_exact_frame")
        elif change == "native_score": child["native_acceptance"]["stocks_taken"]=3
        files[name].write_bytes(p.wire.canonical(body if change in {"source","admission","runtime_evidence"} else child))
    files["receipt.json"].write_bytes(p.wire.canonical(result))
    with pytest.raises((ValueError, KeyError)): p.validate_terminal(terminal, files, "a"*64, value)


def test_no_queue_write_and_native_entrypoint_contract():
    tree = ast.parse(Path(p.__file__).read_text())
    attrs = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "run_game" in attrs and "validate_result" in attrs
    assert not set(attrs) & {"publish", "ingest", "claim_pair", "Popen", "run_frisson_slippi_match", "torch_load", "record_video"}
    assert "supervise_owner" in attrs and "receive_files" in attrs and "materialize" in attrs
    assert "_run_exact_frame" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    assert p.LABELS == ("modal-compat-75m-d21-fd-p2", "modal-compat-75m-d21-fd-p1")


def test_execution_intent_prevents_resubmission(tmp_path, monkeypatch):
    value = plan(); value["output"] = str(tmp_path)
    (tmp_path / "intent.json").write_text("{}")
    monkeypatch.setattr(p, "verify_plan", lambda path, sha: value)
    monkeypatch.setattr(p, "configure_app", lambda *args: pytest.fail("duplicate cloud submit"))
    assert p.execute(tmp_path / "plan.json", "a"*64) == {"status": "intent-retained-no-resubmission"}


def test_actual_sdk_constructs_one_nonpreemptible_worker(tmp_path):
    modal = pytest.importorskip("modal")
    value = plan(); value["uploads"] = []
    path = tmp_path / "plan.json"; path.write_text("{}")
    app, function, image = p.configure_app(modal, path, value)
    assert function.spec.scheduler_placement.nonpreemptible is True
    assert function.spec.cpu == (4,4) and function.spec.memory == (16384,16384)
    assert function.spec.secrets == [] and function.spec.volumes == {}


@pytest.mark.parametrize("status", ["failed", "full-policy-compatibility-pair-passed", "passed", None])
def test_terminal_requires_complete_pair(status):
    terminal = {"schema": p.SCHEMA, "plan_sha256": "a"*64, "scope": p.fixed_scope(), "status": status}
    if status == "failed": p.validate_terminal(terminal, {}, "a"*64, plan())
    else:
        with pytest.raises((ValueError, KeyError)): p.validate_terminal(terminal, {}, "a"*64, plan())
