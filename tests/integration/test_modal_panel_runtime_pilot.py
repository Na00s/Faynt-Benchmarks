"""Synthetic runtime-only plans/transport. No install, network or real process."""
import ast
import copy
import hashlib
import json
import pickle
import os
import shutil
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_runtime_pilot as p


def test_exact_scope_and_deadlines():
    plan = p.plan_definition([], "/research/output")
    assert plan["limits"]["function_seconds"] == 900
    assert plan["limits"]["session_seconds"] == 1200
    assert plan["limits"]["cpu"] == [4, 4] and plan["limits"]["memory_mib"] == [16384, 16384]
    assert plan["limits"]["nonpreemptible"] is True
    assert plan["scope"]["dolphin_launches"] == plan["scope"]["rom_reads"] == plan["scope"]["policy_instances"] == plan["scope"]["policy_inference"] == 0
    assert len([n for n in p.SOURCE_PATHS if n.startswith("src/")]) == 14
    assert len(p.SOURCE_PATHS) == 24 and len(set(p.SOURCE_PATHS)) == 24
    assert all(n.endswith(".py") for n in p.SOURCE_PATHS)
    assert p.DATA_ASSETS == {"src/melee_policy/integration/mimic_native_bundles.v1.json": {
        "bytes": 31259, "sha256": "adb8fa9ef8c6fd5f054fb68b22552c2c92cacce3f8cc0ac0ca2f72bed9ebcef3"}}
    assert not any(n.endswith((".pt", ".slp", ".iso", ".toml")) for n in p.SOURCE_PATHS)
    assert set(p.INPUT_PATHS) == {"lock.json", "pins.lock", "manifest.json", "audit.json"}


def test_no_live_or_policy_execution_calls_in_runtime_helper():
    tree = ast.parse(Path(p.__file__).read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls) & {"verify_and_prepare", "_resolve_game_image_path", "create_console", "_load_config",
                             "FrissonPolicySession", "load_policy", "torch_load", "drive_tape", "run_prepared_plan"}
    assert "_load_native" in calls and "_start_display" in calls and "verify_installed" in calls
    assert "materialize_package" not in calls


@pytest.fixture
def fake_inputs(tmp_path, monkeypatch):
    root = tmp_path / "root"; compat = root / "research"; compat.mkdir(parents=True)
    monkeypatch.setattr(p, "ROOT", root); monkeypatch.setattr(p, "COMPAT", compat)
    for name in p.SOURCE_PATHS:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"# test\n")
    for name in p.DATA_ASSETS:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, path)
    expected = {}
    for name, relative in p.INPUT_PATHS.items():
        path = compat / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if name == "lock.json" else b"synthetic input")
        expected[name] = (path.stat().st_size, p.wire.digest(path))
    monkeypatch.setattr(p, "expected_inputs", lambda: expected)
    monkeypatch.setattr(p.policy, "validate_lock", lambda *a, **k: {"artifacts": [{"bytes": 1, "distribution": f"pkg{i}", "version": "1.0"} for i in range(94)]})
    monkeypatch.setattr(p, "PACKAGE_BYTES", 94)
    return root, compat


def test_preparation_exact_uploads_without_process(fake_inputs, monkeypatch):
    root, compat = fake_inputs
    monkeypatch.setattr(p.guard.subprocess, "Popen", lambda *a, **k: pytest.fail("prepared a process"))
    plan = p.make_plan(compat / "candidate")
    assert len(plan["uploads"]) == 29
    assert {r["remote"] for r in plan["uploads"]} == {str(p.REMOTE / n) for n in (*p.SOURCE_PATHS, *p.DATA_ASSETS)} | {str(p.INPUT / n) for n in p.INPUT_PATHS}
    assert p.data_manifest(plan) == p.DATA_ASSETS
    assert not set(p.source_manifest(plan)) & set(p.DATA_ASSETS)
    assert not (compat / "candidate").exists()
    assert p.source_manifest(plan)[p.SOURCE_PATHS[0]]["bytes"] == 7


def test_frozen_plan_reconstruction_and_input_hashes(fake_inputs):
    root, compat = fake_inputs; out = compat / "candidate"; out.mkdir()
    plan = p.make_plan(out); path = out / "plan.json"; p.wire.create_once(path, plan)
    sha = p.wire.digest(path)
    assert p.verify_plan(path, sha) == plan
    (root / p.SOURCE_PATHS[0]).write_bytes(b"changed")
    with pytest.raises(ValueError): p.verify_plan(path, sha)


def test_wrong_platform_prevents_install_and_import(monkeypatch):
    monkeypatch.setattr(p.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(p.guard.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(RuntimeError): p.perform({}, 1000)
    with pytest.raises(RuntimeError): p.qualification({}, 1000, 123)


def failed_result():
    plan = p.plan_definition([], "/research/output")
    result = {"schema": p.SCHEMA, "plan_sha256": "a" * 64, "scope": plan["scope"], "status": "failed",
              "error": "synthetic failure"}
    return plan, result


def frames(result, files):
    execution = {"task_id": "ta-test", "execution_id": "b" * 32}
    start = {"kind": "hello", "seq": 0, "plan_sha256": "a" * 64, "call_id": "fc-test", "execution": execution}
    rows = []
    for name, body in files.items():
        sha = hashlib.sha256(body).hexdigest()
        rows.append({"name": name, "body": body, "sha256": sha, "original_bytes": len(body), "original_sha256": sha, "truncated_tail": False})
    return [start, *p.wire.output_frames({**result, "execution": execution, "files": rows})]


def test_failure_transport_retains_hash_bound_receipt(tmp_path):
    plan, result = failed_result(); data = p.wire.canonical(result)
    got = p.receive(tmp_path, frames(result, {"receipt.json": data}), "a" * 64, "ap-test", p.time.time() + 30, plan)
    assert got["status"] == "failed" and (tmp_path / "returned/receipt.json").read_bytes() == data
    assert json.loads((tmp_path / "result.json").read_text())["gameplay_admitted"] is False


@pytest.mark.parametrize("change", ["sequence", "offset-bool", "hash", "path", "execution", "no-terminal", "original-sha", "truncated-json"])
def test_transport_rejects_malformed_stream(tmp_path, change):
    plan, result = failed_result(); values = frames(result, {"receipt.json": p.wire.canonical(result)})
    if change == "sequence": values[1]["seq"] = 99
    if change == "offset-bool": values[2]["offset"] = False
    if change == "hash": values[2]["body"] = b"x" * len(values[2]["body"])
    if change == "path": values[1]["file"]["name"] = "../escape"
    if change == "execution": values[-1]["result"]["execution"] = {"task_id": "ta-test", "execution_id": "c" * 32}
    if change == "no-terminal": values.pop()
    if change == "original-sha": values[1]["file"]["original_sha256"] = "d" * 64
    if change == "truncated-json": values[1]["file"]["truncated_tail"] = True
    with pytest.raises(ValueError): p.receive(tmp_path, values, "a" * 64, "ap-test", p.time.time() + 30, plan)


def test_success_cannot_be_claimed_without_evidence(tmp_path):
    plan, result = failed_result(); result["status"] = "runtime-qualified-gameplay-unadmitted"
    with pytest.raises(ValueError): p.receive(tmp_path, frames(result, {"receipt.json": p.wire.canonical(result)}), "a" * 64, "ap-test", p.time.time() + 30, plan)


def test_partial_file_retained_on_interruption(tmp_path):
    plan, result = failed_result(); values = frames(result, {"receipt.json": p.wire.canonical(result)})
    with pytest.raises(ValueError): p.receive(tmp_path, values[:3], "a" * 64, "ap-test", p.time.time() + 30, plan)
    assert (tmp_path / "returned/receipt.json.partial").exists()


def test_app_construction_uses_exact_caps_and_only_plan_files(tmp_path):
    try: import modal
    except ImportError: pytest.skip("Modal SDK optional locally")
    app, function = p.configure_app(modal, tmp_path / "plan.json", {"uploads": []})
    assert app.name == p.APP_NAME
    assert function is not None


def test_stable_context_and_package_identities():
    values = p.expected_inputs()
    assert values["lock.json"][1] == "0083888d28351524408a49f9c73a4b47886886339af79004219460f6fc9f729e"
    assert "code.tar.gz" not in values
    assert values["manifest.json"] == p.host.MANIFEST
    assert p.PACKAGE_BYTES == 950805582


def success(monkeypatch):
    plan, result = failed_result()
    plan["uploads"] = [{"remote": str(p.REMOTE / name), **row} for name, row in p.DATA_ASSETS.items()]
    versions = {f"pkg{i}": "1.0" for i in range(94)}
    artifacts = [{"distribution": name, "version": value, "filename": name + ".whl", "bytes": 100, "sha256": f"{i:064x}"}
                 for i, (name, value) in enumerate(versions.items())]
    monkeypatch.setattr(p, "qualified_lock", lambda plan: {"artifacts": artifacts})
    packages = {name: "version" for name in p.context.RUNTIME_PACKAGES}
    result.pop("error"); result.update(status="runtime-qualified-gameplay-unadmitted", expires_unix=300)
    child = {"schema": p.SCHEMA, "status": "runtime-child-qualified", "scope": plan["scope"],
             "parent_terminal_cleanup_verified": False, "pid": 222, "pgid": 222, "sid": 222, "parent_pid": 111,
             "loaded_project_sources": sorted(n for n in p.SOURCE_PATHS if n.startswith("src/")),
             "runtime": {"pip_check_passed": True, "versions": versions, "os_packages": packages},
             "os_release": {"ID": "ubuntu", "VERSION_ID": '"22.04"'},
             "loader": {"libc.so.6": "/lib/x86_64-linux-gnu/libc.so.6", "libslippi_rust_extensions.so": str(p.WORK / "package/libslippi_rust_extensions.so")},
             "display": {"software_renderer_verified": True, "environment": p.context.HOST_ENV}}
    child["evidence"] = {"source_manifest": p.source_manifest(plan), "data_manifest": p.data_manifest(plan), "os_packages": packages,
        "installed_runtime": {"schema": "e011.modal-live-installed-runtime.v1", "versions": versions, "pip_check_passed": True},
        "ubjson_extension": {"version": "0.16.1", "identity": {"bytes": 100, "sha256": "d" * 64}}}
    h = p.host
    child["package"] = {"schema": h.SCHEMA, "status": "linux-package-bytes-verified", "target_platform": "linux_x86_64",
        "verification_host": {"system": "Linux", "machine": "x86_64"}, "release": h.RELEASE, "source_revision": h.REVISION,
        "frame_patch_sha256": h.FRAME_PATCH_SHA, "build_patch_sha256": h.BUILD_PATCH_SHA, "outer_plan_sha256": h.OUTER_PLAN_SHA,
        "archive_sha256": h.ARCHIVE[1], "manifest_sha256": h.MANIFEST[1], "package_audit_sha256": h.AUDIT_RECEIPT[1],
        "files": h.MEMBER_COUNT, "bytes": h.MEMBER_BYTES, "package_root": str(p.WORK / "package"), "gameplay_admitted": False, "executed": False,
        "executable": {"path": str(p.WORK / "package/dolphin-emu"), "bytes": h.EXECUTABLE[0], "sha256": h.EXECUTABLE[1], "format": "ELF64-x86_64"},
        "rust_library": {"path": str(p.WORK / "package/libslippi_rust_extensions.so"), "bytes": h.RUST_LIBRARY[0], "sha256": h.RUST_LIBRARY[1]}}
    row = {"pid": 222, "unreaped_leader_group_guard": True, "terminated_owned_process_group": True,
           "exit_code": 0, "started_unix": 100, "ended_unix": 201, "log_sha256": "c" * 64,
           "command": [str(p.WORK / "venv/bin/python"), str(p.REMOTE / "scripts/modal_panel_runtime_pilot.py"),
                       "--qualify", "--sha", "a" * 64, "--expires", "200", "--parent-pid", "111"]}
    owner = {"status": "owned-linux-tape-context-ready", "parent_terminal_cleanup_verified": False,
             "process": {"pid": 222}, "parent_declaration": {"expires_at": 200}}
    result.update(qualification=child, package_verified=copy.deepcopy(child["package"]), commands=[row], cleanup=p.context.validate_parent_terminal(owner, row))
    return plan, result


def test_successful_transport_still_leaves_gameplay_unadmitted(tmp_path, monkeypatch):
    plan, result = success(monkeypatch)
    files = {"receipt.json": p.wire.canonical(result), "qualification.json": p.wire.canonical(result["qualification"])}
    got = p.receive(tmp_path, frames(result, files), "a" * 64, "ap-test", p.time.time() + 30, plan)
    assert got["status"] == "runtime-qualified-gameplay-unadmitted"
    assert got["cleanup"]["parent_terminal_cleanup_verified"] is True
    assert got["scope"]["gameplay_admitted"] is False


@pytest.mark.parametrize("change", ["scope", "display", "closure", "cleanup", "parent-command", "deadline", "child-receipt", "package", "loader", "versions", "source-evidence", "data-evidence", "data-sha", "parent-package-missing", "parent-package-sha", "parent-package-size", "parent-package-path"])
def test_claimed_success_requires_exact_runtime_and_parent_evidence(tmp_path, monkeypatch, change):
    plan, result = success(monkeypatch)
    child = result["qualification"]
    if change == "scope": child["scope"] = {**child["scope"], "rom_reads": 1}
    if change == "display": child["display"]["software_renderer_verified"] = False
    if change == "closure": child["loaded_project_sources"].pop()
    if change == "cleanup": result["commands"][-1]["terminated_owned_process_group"] = False
    if change == "parent-command": result["commands"][-1]["command"][1] = "/other.py"
    if change == "deadline": result["commands"][-1]["command"][6] = "301"
    if change == "package": child.pop("package")
    if change == "loader": child.pop("loader")
    if change == "versions": child["runtime"]["versions"] = {"pkg0": "changed"}
    if change == "source-evidence": child["evidence"].pop("source_manifest")
    if change == "data-evidence": child["evidence"].pop("data_manifest")
    if change == "data-sha": child["evidence"]["data_manifest"] = {next(iter(p.DATA_ASSETS)): {"bytes": 31259, "sha256": "0" * 64}}
    if change == "parent-package-missing": result.pop("package_verified")
    if change == "parent-package-sha": result["package_verified"]["executable"]["sha256"] = "f" * 64
    if change == "parent-package-size": result["package_verified"]["bytes"] += 1
    if change == "parent-package-path": result["package_verified"]["package_root"] = "/other"
    files = {"receipt.json": p.wire.canonical(result), "qualification.json": p.wire.canonical(child)}
    if change == "child-receipt": files["qualification.json"] = b"{}"
    with pytest.raises(ValueError): p.receive(tmp_path, frames(result, files), "a" * 64, "ap-test", p.time.time() + 30, plan)


def test_qualification_with_fake_native_context_never_constructs_game(tmp_path, monkeypatch):
    work, remote, inputs = tmp_path / "work", tmp_path / "root", tmp_path / "inputs"
    work.mkdir(); remote.mkdir(); inputs.mkdir()
    monkeypatch.setattr(p, "WORK", work); monkeypatch.setattr(p, "REMOTE", remote); monkeypatch.setattr(p, "INPUT", inputs)
    monkeypatch.setattr(p.platform, "system", lambda: "Linux"); monkeypatch.setattr(p.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(p.os, "getpid", lambda: 222); monkeypatch.setattr(p.os, "getpgid", lambda n: 222)
    monkeypatch.setattr(p.os, "getsid", lambda n: 222); monkeypatch.setattr(p.os, "getppid", lambda: 111)
    monkeypatch.setattr(p.os, "waitid", lambda *a: None, raising=False)
    for name in ("lock.json", "pins.lock"): (inputs / name).write_bytes(b"{}")
    extension = tmp_path / "_ubjson.so"; extension.write_bytes(b"fake extension")
    monkeypatch.setitem(sys.modules, "ubjson", SimpleNamespace(EXTENSION_ENABLED=True))
    monkeypatch.setitem(sys.modules, "_ubjson", SimpleNamespace(__file__=str(extension)))
    monkeypatch.setattr(p.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(p.importlib.metadata, "version", lambda name: "0.16.1")
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(p.os, "environ", dict(p.os.environ))
    rows = []
    for name in p.SOURCE_PATHS:
        path = remote / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"# fixture\n")
        rows.append({"remote": str(path), "bytes": 10, "sha256": p.wire.digest(path)})
    for name, identity in p.DATA_ASSETS.items():
        path = remote / name; shutil.copyfile(ROOT / name, path)
        rows.append({"remote": str(path), **identity})
    calls = []
    class FakeContext:
        def __init__(self, **kwargs): calls.append("constructed")
        def check_deadline(self, expiry): pass
        def _runtime(self, evidence):
            assert set(evidence) == {"dependency_lock", "dependency_pins", "source_manifest", "installed_runtime", "ubjson_extension", "os_packages"}
            calls.append("runtime"); return {"pip_check_passed": True}
        def _capture(self, argv):
            calls.append(argv)
            if "dpkg-query" in argv[0]: return ""
            return ""
        def _load_native(self, manifest):
            calls.append("native imports")
            for name in tuple(sys.modules):
                if name == "melee_policy" or name.startswith("melee_policy."):
                    monkeypatch.delitem(sys.modules, name)
            for name in p.SOURCE_PATHS:
                if not name.startswith("src/"): continue
                module_name = name[4:-3].replace("/", ".").removesuffix(".__init__")
                monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__file__=str(remote / name)))
        def _start_display(self): calls.append("display"); return {"software_renderer_verified": True}
    monkeypatch.setattr(p.context, "OwnedLinuxContext", FakeContext)
    monkeypatch.setattr(p.context, "parse_loader", lambda *a: {"lib": "fixture"})
    monkeypatch.setattr(p.context, "proc_text", lambda *a: 'ID=ubuntu\nVERSION_ID="22.04"\n')
    monkeypatch.setattr(p.host, "verify_installed", lambda *a: {"verified": True})
    monkeypatch.setattr(p.guard.subprocess, "Popen", lambda *a, **k: pytest.fail("real process"))
    plan = p.plan_definition(rows, "/research/output")
    value = p.qualification(plan, p.time.time() + 30, 111)
    assert value["status"] == "runtime-child-qualified", value
    assert value["parent_terminal_cleanup_verified"] is False
    assert "native imports" in calls and "display" in calls
    assert value["scope"]["rom_reads"] == value["scope"]["dolphin_launches"] == 0


def test_qualified_version_inventory_is_bound_to_actual_lock(fake_inputs, monkeypatch):
    _, compat = fake_inputs; plan = p.make_plan(compat / "candidate")
    lock = compat / p.INPUT_PATHS["lock.json"]
    monkeypatch.setattr(p.context, "LOCK_SHA", p.wire.digest(lock))
    assert p.qualified_versions(plan) == {f"pkg{i}": "1.0" for i in range(94)}
    lock.write_bytes(b"changed lock")
    with pytest.raises(ValueError): p.qualified_versions(plan)


def test_plausible_complete_terminal_fits_generator_frame(monkeypatch):
    plan, result = success(monkeypatch)
    lock = p.policy.bounded_json(ROOT / p.policy.REFERENCE / p.INPUT_PATHS["lock.json"])
    result["acquired"] = [{k: row[k] for k in ("filename", "bytes", "sha256")} for row in lock["artifacts"]]
    versions = {row["distribution"]: row["version"] for row in lock["artifacts"]}
    child = result["qualification"]
    child["runtime"]["versions"] = versions
    child["evidence"]["installed_runtime"]["versions"] = versions
    child["evidence"]["source_manifest"] = {name: {"bytes": 100000, "sha256": hashlib.sha256(name.encode()).hexdigest()} for name in p.SOURCE_PATHS}
    retained_log = ROOT / p.policy.REFERENCE / "pilot-v7/returned/logs/030.log"
    if not retained_log.is_file():
        pytest.skip("requires the externally retained Linux loader log")
    text = retained_log.read_text()
    child["loader"] = p.context.parse_loader(text, Path("/work/build/install/Binaries"))
    child["loader"]["libslippi_rust_extensions.so"] = str(p.WORK / "package/libslippi_rust_extensions.so")
    result["package_materialized"] = {**child["package"], "materialized": True}
    result["commands"] = [{**result["commands"][0], "command": ["python", "-c", str(i) + "x" * 1500],
                            "cwd": str(p.WORK), "log": f"logs/{i:03d}.log", "log_bytes": 100000,
                            "log_sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(12)]
    result["tls"] = {"ca_bundle": "/etc/ssl/certs/ca-certificates.crt", "ca_bundle_sha256": "a" * 64,
                     "cert_store_stats": {"x509_ca": 144}, "default_verify_paths": {"cafile": "/opt/python/cert.pem"}}
    # Remove incidental shared fixture references before measuring the envelope.
    frame = json.loads(json.dumps({"kind": "done", "seq": 99, "result": result}))
    p.wire.checked_frame(frame)
    assert len(pickle.dumps(frame, protocol=4)) < p.wire.FRAME_BYTES


@pytest.mark.parametrize("change", ["missing", "changed", "symlink"])
def test_import_asset_identity_is_required_before_plan(fake_inputs, change):
    root, compat = fake_inputs
    path = root / next(iter(p.DATA_ASSETS))
    if change == "missing": path.unlink()
    if change == "changed": path.write_bytes(b"{}")
    if change == "symlink":
        path.unlink(); path.symlink_to(ROOT / next(iter(p.DATA_ASSETS)))
    with pytest.raises((ValueError, FileNotFoundError)):
        p.make_plan(compat / "candidate")


def test_data_inventory_cannot_be_dropped_or_mixed_with_source(fake_inputs):
    _, compat = fake_inputs
    plan = p.make_plan(compat / "candidate")
    assert set(p.source_manifest(plan)) == set(p.SOURCE_PATHS)
    plan["uploads"] = [r for r in plan["uploads"] if not r["remote"].endswith("mimic_native_bundles.v1.json")]
    with pytest.raises(ValueError): p.data_manifest(plan)


def test_real_staged_native_imports_with_only_upload_allowlist(tmp_path):
    """Import exact staged files using installed native dependencies, without constructing policies."""
    python = ROOT / ".e010-env/bin/python"
    if not python.is_file(): pytest.skip("installed native import environment unavailable")
    stage = tmp_path / "stage"; stage.mkdir()
    for name in (*p.SOURCE_PATHS, *p.DATA_ASSETS):
        target = stage / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        assert p.wire.digest(target) == p.wire.digest(ROOT / name)
    code = r'''
import json, pathlib, sys
stage = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(stage / "scripts"), str(stage / "src")]
def restricted(event, args):
    if event in {"socket.connect", "subprocess.Popen", "os.system"}:
        raise RuntimeError("unexpected external action during import")
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = str(args[0]).lower()
        if name.endswith((".pt", ".pth", ".slp", ".iso", ".gcm", ".rvz")):
            raise RuntimeError("unexpected model or game asset access")
sys.addaudithook(restricted)
import modal_panel_runtime_pilot as pilot
manifest = {name: {"bytes": (stage / name).stat().st_size, "sha256": pilot.wire.digest(stage / name)} for name in pilot.SOURCE_PATHS}
ctx = object.__new__(pilot.context.OwnedLinuxContext)
ctx.project_root = stage
ctx._load_native(manifest)
loaded = sorted(str(pathlib.Path(m.__file__).resolve().relative_to(stage)) for name, m in sys.modules.items()
                if name == "melee_policy" or name.startswith("melee_policy."))
assert loaded == sorted(n for n in pilot.SOURCE_PATHS if n.startswith("src/"))
print(json.dumps({"loaded": loaded, "data_assets": pilot.DATA_ASSETS, "policy_instances": 0, "dolphin_launches": 0}))
'''
    result = subprocess.run([str(python), "-I", "-B", "-c", code, str(stage)], cwd=stage,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-5000:]
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["data_assets"] == p.DATA_ASSETS
    assert len(receipt["loaded"]) == 14


def test_prior_missing_static_asset_reproduces_in_isolated_import(tmp_path):
    stage = tmp_path / "stage"
    for name in ("src/melee_policy/__init__.py", "src/melee_policy/integration/__init__.py",
                 "src/melee_policy/integration/mimic_bundle_manifest.py"):
        target = stage / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    code = "import sys;sys.path.insert(0,sys.argv[1]);import melee_policy.integration.mimic_bundle_manifest"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(stage / "src")],
        cwd=stage, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "FileNotFoundError" in result.stderr
    assert "mimic_native_bundles.v1.json" in result.stderr
