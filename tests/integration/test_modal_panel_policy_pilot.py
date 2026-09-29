"""Local synthetic tests: no cloud, network, package install or policy loading."""
from __future__ import annotations

import copy
import builtins
import hashlib
import io
import json
from pathlib import Path
import ssl
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_policy_pilot as p

EXECUTION = {"task_id": "ta-fixture", "execution_id": "a" * 32}
PLAN_SHA = "b" * 64


def artifact(name="numpy", version="1.0", role="runtime", format="wheel", host="files.pythonhosted.org"):
    filename = name.replace("-", "_") + "-" + version + "-py3-none-any.whl"
    if format == "sdist": filename = "py-ubjson-0.16.1.tar.gz"
    return dict(distribution=name, version=version, role=role, format=format, filename=filename,
                url=f"https://{host}/whl/cpu/{filename}", bytes=3, sha256=hashlib.sha256(b"abc").hexdigest())


def lock(complete=True):
    rows = [artifact(n, role="bootstrap") for n in ("pip", "setuptools", "wheel")]
    rows += [artifact(n, version="2.7.1+cpu", host="download-r2.pytorch.org") if n == "torch" else artifact(n)
             for n in p.native.DEPENDENCIES]
    return dict(schema=p.LOCK_SCHEMA, python="3.12", platform="linux_x86_64", complete=complete,
                artifacts=rows, provenance={"candidate_sha256": "1" * 64, "source_build_audit_sha256": "2" * 64})


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = (tmp_path / "root").resolve(); root.mkdir()
    def put(rel, data=b"source"):
        path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        return {"path": str(rel), "bytes": len(data), "sha256": p.transport.digest(path)}
    common = [put(s) for s in p.native.SOURCE_FILES]
    for s in p.EXTRA_SOURCES: put(s)
    replay = put("replay.slp"); manifest = put("manifest.json")
    prefix = p.REFERENCE / "frisson-source-git/melee-policy-frisson-ai"
    sources = common + [{"path": str(prefix / n), "sha256": sha, "bytes": 1} for n, sha in p.snapshot.FILES.items()]
    for kind, key, port in p.CASES:
        checkpoint = put(f"checkpoints/{kind}-{key}.pt")
        native = dict(kind=kind, key=key, port=port, frisson_git=None, frisson_source=str(prefix),
                      slippi_source=p.native.SLIPPI_SOURCE, checkpoint=checkpoint, replay=replay,
                      manifest=manifest, sources=sources)
        put(p.REFERENCE / f"native-canary-{kind}-{key}-p{port}-plan-v3.json", p.canonical(native))
    snap = put(p.REFERENCE / "frisson-source-snapshot.json", b"{}")
    monkeypatch.setattr(p, "SNAPSHOT_SHA", snap["sha256"])
    monkeypatch.setattr(p.snapshot, "validate", lambda obj: obj)
    monkeypatch.setattr(p.native, "validate_plan", lambda obj, root: obj)
    lp = root / "execution-lock.json"; lp.write_bytes(p.canonical(lock()))
    output = tmp_path.resolve() / "pilot"
    plan = p.make_plan(root, lp, output); output.mkdir()
    path = output / "plan.json"; path.write_bytes(p.canonical(plan))
    return SimpleNamespace(root=root, lock=lp, out=output, plan=plan, path=path, sha=p.transport.digest(path))


def test_exact_plan_layout_and_no_reference_upload(frozen):
    assert p.verify_plan(frozen.path, frozen.sha)[0] == frozen.plan
    assert len(frozen.plan["cases"]) == 10
    assert len(frozen.plan["generated_sources"]) == 3
    names = [r["remote"] for r in frozen.plan["uploads"]]
    assert all("reference" not in name and "/.git/" not in name for name in names)
    assert len(names) == len(set(names))
    assert frozen.plan["limits"]["cpu"] == [4, 4]
    assert frozen.plan["runtime"]["nonpreemptible"] is True
    assert frozen.plan["runtime"]["native_environment_flags_unchanged"] is False
    assert frozen.plan["runtime"]["native_environment_overrides"] == {"TF_ENABLE_ONEDNN_OPTS": "0"}
    assert frozen.plan["retry_semantics"]["platform_restarts_possible"] is True


@pytest.mark.parametrize("change", ["upload", "case", "limits", "source", "backend"])
def test_tampered_plan_rejected(frozen, change):
    plan = frozen.plan
    if change == "upload": plan["uploads"].append(plan["uploads"][0])
    if change == "case": plan["cases"].reverse()
    if change == "limits": plan["limits"]["cpu"] = [16, 16]
    if change == "source": Path(plan["uploads"][0]["local"]).write_text("changed")
    if change == "backend": plan["runtime"]["native_environment_overrides"] = {}
    frozen.path.write_bytes(p.canonical(plan))
    with pytest.raises(ValueError): p.verify_plan(frozen.path, p.transport.digest(frozen.path))


def test_lock_complete_and_official_cpu_hosts():
    for host in ("download.pytorch.org", "download-r2.pytorch.org"):
        value = lock(); row = next(x for x in value["artifacts"] if x["distribution"] == "torch")
        row["url"] = row["url"].replace("download-r2.pytorch.org", host)
        assert p.validate_lock(value, require_complete=True) is value


@pytest.mark.parametrize("change", ["incomplete", "evidence", "duplicate", "filename", "url", "host", "cpu", "provenance", "traversal", "redirect-query"])
def test_invalid_lock_rejected(change):
    value = lock(); row = value["artifacts"][3]
    if change == "incomplete": value["complete"] = False
    if change == "evidence": value["evidence"] = {"headers": "Set-Cookie: private"}
    if change == "duplicate": value["artifacts"].append(artifact("tf.nightly"))
    if change == "filename": row["filename"] = value["artifacts"][0]["filename"]
    if change == "url": row["url"] += "wrong"
    if change == "host": row["url"] = row["url"].replace("files.pythonhosted.org", "example.com")
    if change == "cpu": next(x for x in value["artifacts"] if x["distribution"] == "torch")["version"] = "2.7.1"
    if change == "provenance": value["provenance"]["source_build_audit_sha256"] = None
    if change == "traversal": row["filename"] = "../escape.whl"
    if change == "redirect-query": row["url"] += "?token=x"
    with pytest.raises(ValueError): p.validate_lock(value, require_complete=True)


@pytest.mark.parametrize("name", ["../a", "/etc/passwd", ".git/config", "a//b", "a\\b", "a\n"])
def test_relative_rejects_unsafe(name):
    with pytest.raises(ValueError): p.relative(name)






def cases():
    return [dict(kind=k, key=n, port=port, plan=f"plans/{i}.json", plan_sha256=str(i) * 64) for i, (k, n, port) in enumerate(p.CASES)]


def terminal(success=False):
    rows = [{"index": i, **case, "output": f"case-{i:02d}.json", "passed": success} for i, case in enumerate(cases())]
    return dict(schema=p.SCHEMA, status="native-cases-passed-awaiting-local-comparison" if success else "native-cases-failed",
                cases=rows, commands=[], no_gameplay=True, admitted=False)


def frames(payloads=None, result=None):
    values = [dict(kind="hello", plan_sha256=PLAN_SHA, call_id="fc-fixture", execution=EXECUTION)]
    for name, payload in (payloads or {"case-00.json": b"{\"numeric\":[1,2,3]}"}).items():
        sha = hashlib.sha256(payload).hexdigest()
        values += [dict(kind="file", execution=EXECUTION, file=dict(name=name, bytes=len(payload), sha256=sha,
                   original_bytes=len(payload), original_sha256=sha, truncated_tail=False)),
                   dict(kind="data", offset=0, body=payload), dict(kind="end-file", name=name, sha256=sha)]
    values.append(dict(kind="done", plan_sha256=PLAN_SHA, execution=EXECUTION, result=result or terminal()))
    return [{**value, "seq": i} for i, value in enumerate(values)]


def receive(tmp_path, values):
    return p.receive(tmp_path.resolve(), values, PLAN_SHA, "ap-fixture", time.time() + 5, cases())


def test_full_receipt_stream_and_failed_result(tmp_path):
    result = receive(tmp_path, frames())
    assert result["admitted"] is False and result["comparison_required"] is True
    assert (tmp_path / "returned/case-00.json").read_bytes() == b'{"numeric":[1,2,3]}'


@pytest.mark.parametrize("change", ["sequence", "execution", "task", "sha", "original-sha", "offset", "traversal", "missing-success", "terminal-case", "terminal-schema"])
def test_stream_faults_rejected(tmp_path, change):
    values = frames()
    if change == "sequence": values[2]["seq"] = 99
    if change in {"execution", "task"}:
        values[-1]["execution"] = {**EXECUTION, ("execution_id" if change == "execution" else "task_id"): ("c" * 32 if change == "execution" else "ta-other")}
    if change == "sha": values[-2]["sha256"] = "d" * 64
    if change == "original-sha": values[1]["file"]["original_sha256"] = "d" * 64
    if change == "offset": values[2]["offset"] = False
    if change == "traversal": values[1]["file"]["name"] = "../escape"
    if change == "missing-success": values[-1]["result"] = terminal(True)
    if change == "terminal-case": values[-1]["result"]["cases"][0]["port"] = 2
    if change == "terminal-schema": values[-1]["result"] = {"status": "passed"}
    with pytest.raises(ValueError): receive(tmp_path, values)


def test_interrupted_transfer_retains_partial(tmp_path):
    with pytest.raises(ValueError): receive(tmp_path, frames()[:3])
    assert (tmp_path / "returned/case-00.json.partial").is_file()
    assert (tmp_path / "call.json").is_file()


def test_success_requires_all_native_receipts(tmp_path):
    payloads = {name: b"{}" for name in ["intent.json", "prepared-runtime.json", "ubjson-extension.json"]}
    payloads["tensorflow-backend.json"] = p.canonical({"schema": "e011.modal-policy-tensorflow-backend.v1",
        "environment_before_import": p.NATIVE_BACKEND_ENV, "mkl_enabled": False, "distribution_version": "2.21.0.dev20260203"})
    for i, case in enumerate(cases()):
        payloads[f"case-{i:02d}.json"] = p.canonical(native_receipt(case))
        payloads[f"case-{i:02d}-status.json"] = p.canonical(terminal(True)["cases"][i])
    result = receive(tmp_path, frames(payloads, terminal(True)))
    assert result["worker"]["status"] == "native-cases-passed-awaiting-local-comparison"


def native_receipt(case):
    return dict(schema=p.native.RESULT_SCHEMA, passed=True, policy_steps=384, rows=[{}] * 384,
                approved_plan_sha256=case["plan_sha256"], live_games=0, source_assets_unchanged=True,
                checks={"native": True}, plan={key: case[key] for key in ("kind", "key", "port")},
                environment={"environment_flags": {"TF_ENABLE_ONEDNN_OPTS": "0"}})


@pytest.mark.parametrize("change", ["empty", "schema", "rows", "plan", "sha", "checks", "games", "backend"])
def test_native_receipt_completeness(change):
    value = native_receipt(cases()[0])
    if change == "empty": value = {}
    if change == "schema": value["schema"] = "other"
    if change == "rows": value["rows"].pop()
    if change == "plan": value["plan"]["port"] = 2
    if change == "sha": value["approved_plan_sha256"] = "c" * 64
    if change == "checks": value["checks"] = {"native": False}
    if change == "games": value["live_games"] = 1
    if change == "backend": value["environment"]["environment_flags"]["TF_ENABLE_ONEDNN_OPTS"] = None
    with pytest.raises(ValueError): p.check_native_receipt(value, cases()[0])


@pytest.mark.parametrize("scope", ["receipt", "aggregate"])
def test_transport_caps(tmp_path, monkeypatch, scope):
    monkeypatch.setattr(p, "RECEIPT_BYTES" if scope == "receipt" else "RETURN_BYTES", 2)
    with pytest.raises(ValueError): receive(tmp_path, frames())


def test_bootstrap_failure_retains_truthful_result(tmp_path, monkeypatch):
    work = tmp_path.resolve() / "work"; plan = tmp_path / "plan.json"; plan.write_text("{}")
    monkeypatch.setattr(p, "WORK", work); monkeypatch.setattr(p, "REMOTE_PLAN", plan)
    monkeypatch.setattr(p.platform, "system", lambda: "Linux"); monkeypatch.setattr(p.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(p.os, "waitid", lambda: None, raising=False)
    monkeypatch.setattr(p, "run_command", lambda *a: (_ for _ in ()).throw(TimeoutError("expired")))
    expires = time.time() + 30
    result = p.perform({}, lock(), expires)
    assert result["status"] == "failed" and result["cases"] == []
    assert result["worker_deadline_unix"] == expires
    assert result["expires_unix"] == expires and (work / "intent.json").exists()


def test_perform_isolated_sequential_native_commands(tmp_path, monkeypatch):
    root = tmp_path.resolve(); work = root / "work"; remote = root / "remote"; remote.mkdir()
    planpath = root / "plan.json"; planpath.write_text("{}")
    monkeypatch.setattr(p, "WORK", work); monkeypatch.setattr(p, "REMOTE_ROOT", remote)
    monkeypatch.setattr(p, "REMOTE_PLAN", planpath)
    monkeypatch.setattr(p.platform, "system", lambda: "Linux"); monkeypatch.setattr(p.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(p.os, "waitid", lambda: None, raising=False)
    interpreter = root / "provided-python"
    supplied_source = root / "provided-slippi"; supplied_source.mkdir()
    monkeypatch.setattr(p.prepared_runtime, "require_prepared_runtime", lambda _: interpreter)
    monkeypatch.setattr(p.prepared_runtime, "slippi_source", lambda _: supplied_source)
    monkeypatch.setattr(p, "REMOTE_LOCK", planpath)
    calls = []
    def command(argv, directory, seconds, env):
        calls.append((argv, seconds, env)); directory.mkdir()
        if "--output" in argv:
            out = Path(argv[argv.index("--output") + 1])
            out.write_text(json.dumps({"passed": True, "policy_steps": 384,
                                      "approved_plan_sha256": argv[argv.index("--approved-plan-sha256") + 1]}))
    monkeypatch.setattr(p, "run_command", command)
    value = lock(); value["artifacts"].append(artifact("py-ubjson", "0.16.1", format="sdist"))
    result = p.perform(dict(cases=cases(), slippi={"destination": "slippi"}, frisson_destination="frisson"), value, time.time() + 4500)
    assert result["status"] == "native-cases-passed-awaiting-local-comparison"
    canaries = [argv for argv, _, _ in calls if "--output" in argv]
    assert len(canaries) == 10
    assert [a[a.index("--approved-plan-sha256") + 1] for a in canaries] == [c["plan_sha256"] for c in cases()]
    assert all(a[0] == str(interpreter) for a in canaries)
    installs = [a for a, _, _ in calls if "install" in a and "pip" in a]
    assert installs == []
    assert any("ubjson.EXTENSION_ENABLED is True" in " ".join(a) for a, _, _ in calls)
    backend_calls = [(a, env) for a, _, env in calls if "TF_ENABLE_ONEDNN_OPTS" in env]
    assert len(backend_calls) == 11
    assert all(env["TF_ENABLE_ONEDNN_OPTS"] == "0" for _, env in backend_calls)
    assert "assert 'tensorflow' not in sys.modules" in backend_calls[0][0][-1]
    assert all("--output" in a for a, _ in backend_calls[1:])
    assert all(not any(k.startswith("OMP_") or (k.startswith("TF_") and k != "TF_ENABLE_ONEDNN_OPTS") for k in env) for _, _, env in calls)
    assert all("TF_ENABLE_ONEDNN_OPTS" not in env for a, _, env in calls if "pip" in a or a[0] == "git")
    assert all("CC" not in env and "CXX" not in env for a, _, env in calls if "--output" in a)
    assert max(seconds for _, seconds, _ in calls) <= 250
    assert result["worker_deadline_unix"] <= result["started_unix"] + 3300
    assert (work / "prepared-runtime.json").is_file()
    # Exercise the generated pre-import backend gate with import stubs only.
    import importlib.metadata
    original_import = builtins.__import__
    imports = []
    def backend_import(name, *args, **kwargs):
        if name == "tensorflow":
            assert p.os.environ.get("TF_ENABLE_ONEDNN_OPTS") == "0"
            imports.append(name)
            return SimpleNamespace(__version__="fixture", version=SimpleNamespace(GIT_VERSION="fixture-git"))
        if name == "tensorflow.python.util":
            return SimpleNamespace(_pywrap_util_port=SimpleNamespace(IsMklEnabled=lambda: False))
        return original_import(name, *args, **kwargs)
    probe = backend_calls[0][0][-1]
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", backend_import)
        patch.delitem(sys.modules, "tensorflow", raising=False)
        patch.delenv("TF_ENABLE_ONEDNN_OPTS", raising=False)
        with pytest.raises(AssertionError): exec(compile(probe, "<generated-backend-probe>", "exec"), {})
        assert imports == []
        patch.setenv("TF_ENABLE_ONEDNN_OPTS", "0")
        patch.setattr(importlib.metadata, "version", lambda name: "2.21.0.dev20260203")
        exec(compile(probe, "<generated-backend-probe>", "exec"), {})
        assert imports == ["tensorflow"]
        backend = json.loads((work / "tensorflow-backend.json").read_bytes())
        assert backend["mkl_enabled"] is False and backend["environment_before_import"] == p.NATIVE_BACKEND_ENV
    # Execute the actual generated receipt writer with module stubs only.
    extension = root / "fixture.so"; extension.write_bytes(b"fixture extension")
    monkeypatch.setitem(sys.modules, "ubjson", SimpleNamespace(EXTENSION_ENABLED=True))
    monkeypatch.setitem(sys.modules, "_ubjson", SimpleNamespace(__file__=str(extension)))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.16.1")
    writer = next(a[-1] for a, _, _ in calls if "ubjson.EXTENSION_ENABLED is True" in " ".join(a))
    exec(compile(writer, "<generated-extension-check>", "exec"), {})
    data = (work / "ubjson-extension.json").read_bytes()
    assert data.endswith(b"\n") and json.loads(data)["sha256"] == p.transport.digest(extension)












def test_old_intent_prevents_submission(frozen, monkeypatch):
    (frozen.out / "intent.json").write_text("{}")
    monkeypatch.setattr(p, "configure_app", lambda *_: pytest.fail("must not resubmit"))
    assert p.execute(frozen.path, frozen.sha) == {"status": "intent-retained-no-resubmission"}


def test_actual_sdk_constructs_without_cloud(frozen):
    modal = pytest.importorskip("modal")
    app, function = p.configure_app(modal, frozen.path, frozen.plan)
    assert app.name == p.APP_NAME and function is not None
