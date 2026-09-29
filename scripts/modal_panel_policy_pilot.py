#!/usr/bin/env python3
"""Staged, source-bound Linux policy canaries. Preparation never loads a model."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import uuid
import shutil
import prepared_runtime

import modal_panel_build_pilot as transport
import modal_panel_linux_build as process_guard
import modal_panel_policy_canary as native
import modal_panel_source_snapshot as snapshot

SCHEMA = "e011.modal-policy-pilot.v1"
LOCK_SCHEMA = "e011.modal-policy-linux-dependencies.v1"
ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/policy-root")
REMOTE_PLAN = Path("/opt/policy-plan.json")
REMOTE_LOCK = Path("/opt/policy-lock.json")
REMOTE_SNAPSHOT = Path("/opt/frisson-source-snapshot.json")
WORK = Path("/work/policy-pilot")
REFERENCE = Path("artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/research/modal-compat-v1")
CASES = tuple((kind, key, port) for kind, key in native.CASES for port in (1, 2))
EXTRA_SOURCES = ("scripts/modal_panel_policy_pilot.py", "scripts/modal_panel_source_snapshot.py",
                 "scripts/modal_panel_build_pilot.py", "scripts/modal_panel_linux_build.py", "scripts/prepared_runtime.py")
SLIPPI_URL = "https://github.com/vladfi1/slippi-ai"
SLIPPI_REVISION = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
SNAPSHOT_SHA = "3a4a3972734124ec02ccdc217b3ec63abb13368bbcd9fffbc8e0793a0b1e2203"
UPLOAD_BYTES = 1024**3
DEPENDENCY_BYTES = 4 * 1024**3
RECEIPT_BYTES = 128 * 1024**2
RETURN_BYTES = 10 * RECEIPT_BYTES + 32 * 1024**2
SESSION_SECONDS = 4500
FUNCTION_SECONDS = 3600
BOOTSTRAP_SECONDS = 600
CASE_SECONDS = 250
OUTPUT_RESERVE_SECONDS = 300
APP_NAME = "frisson-linux-native-policy-canary-v1"
HTTP_USER_AGENT = "frisson-policy-metadata-audit/1"
SYSTEM_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
SOURCE_BUILD_ENV = {"CC": "/usr/bin/gcc"}
NATIVE_BACKEND_ENV = {"TF_ENABLE_ONEDNN_OPTS": "0"}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return transport.canonical(value)


def relative(value):
    if (not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value
            or any(part in {"", ".", "..", ".git"} for part in value.split("/"))
            or any(ord(c) < 32 for c in value)):
        raise ValueError("unsafe relative input path")
    return value


def bounded_json(path, cap=8 * 1024**2):
    transport.regular(path, cap)
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError("JSON read exceeds limit")
    return json.loads(data)


def file_record(local, remote, cap=UPLOAD_BYTES):
    local = Path(local).absolute()
    return {"local": str(local), "remote": str(remote), "bytes": transport.regular(local, cap),
            "sha256": transport.digest(local)}


def validate_lock(lock, require_complete=False):
    if (not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA
            or set(lock) != {"schema", "python", "platform", "complete", "artifacts", "provenance"}
            or lock.get("python") != "3.12" or lock.get("platform") != "linux_x86_64"
            or type(lock.get("complete")) is not bool):
        raise ValueError("dependency lock schema/platform differs")
    provenance = lock["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {"candidate_sha256", "source_build_audit_sha256"}
            or any(not isinstance(v, str) or not SHA.fullmatch(v) for v in provenance.values())):
        raise ValueError("local metadata and source-build review hashes are required")
    if require_complete and not lock["complete"]:
        raise ValueError("dependency closure is incomplete; cloud submission is blocked")
    rows = lock.get("artifacts")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("bounded dependency artifact inventory required")
    seen, filenames, total = set(), set(), 0
    for row in rows:
        required = {"distribution", "version", "filename", "url", "bytes", "sha256", "role", "format"}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("dependency artifact schema differs")
        name, filename = row["distribution"], row["filename"]
        url = urllib.parse.urlsplit(row["url"])
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or re.sub(r"[-_.]+", "-", name.lower()) in seen or not isinstance(row["version"], str)
                or not re.fullmatch(r"[A-Za-z0-9_.+!-]+", row["version"])
                or not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", filename)
                or filename in filenames or urllib.parse.unquote(url.path.rsplit("/", 1)[-1]) != filename
                or url.scheme != "https" or url.hostname not in {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}
                or url.username or url.password or url.fragment or url.query
                or row["role"] not in {"bootstrap", "runtime"} or row["format"] not in {"wheel", "sdist"}
                or type(row["bytes"]) is not int or not 0 < row["bytes"] <= 2 * 1024**3
                or not isinstance(row["sha256"], str) or not SHA.fullmatch(row["sha256"])):
            raise ValueError("unsafe dependency artifact identity")
        if row["format"] == "wheel" and not filename.endswith(".whl"):
            raise ValueError("wheel filename required")
        if row["format"] == "sdist" and (name != "py-ubjson" or filename != "py-ubjson-0.16.1.tar.gz"):
            raise ValueError("only the declared py-ubjson source build is permitted")
        if row["role"] == "bootstrap" and (name not in {"pip", "setuptools", "wheel", "packaging"} or row["format"] != "wheel"):
            raise ValueError("bootstrap is restricted to pinned pip/setuptools/wheel wheels")
        if name == "torch" and (row["version"] != "2.7.1+cpu" or url.hostname not in {"download.pytorch.org", "download-r2.pytorch.org"} or not url.path.startswith("/whl/cpu/")):
            raise ValueError("the declared Linux CPU PyTorch build is required")
        seen.add(re.sub(r"[-_.]+", "-", name.lower()))
        filenames.add(filename)
        total += row["bytes"]
    if total > DEPENDENCY_BYTES:
        raise ValueError("dependency aggregate exceeds 4 GiB")
    if lock["complete"] and not {"pip", "setuptools", "wheel"} <= seen:
        raise ValueError("complete lock must include bootstrap tool identities")
    if lock["complete"] and not {re.sub(r"[-_.]+", "-", name.lower()) for name in native.DEPENDENCIES} <= seen:
        raise ValueError("complete lock lacks native top-level packages")
    return lock


def make_plan(root: Path, lock_path: Path, output: Path, *, existing=False):
    root, lock_path, output = root.absolute(), lock_path.absolute(), output.absolute()
    if root.resolve(strict=True) != root or (output.exists() and not existing) or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("canonical source root and new output are required")
    lock = validate_lock(bounded_json(lock_path))
    uploads, cases, generated = {}, [], {}
    def add(local, remote):
        row = file_record(local, remote)
        if str(remote) in uploads and uploads[str(remote)] != row:
            raise ValueError("conflicting upload destinations")
        uploads[str(remote)] = row
    source_parent = REFERENCE / "frisson-source-git"
    generated_prefix = (source_parent / "melee-policy-frisson-ai").as_posix()
    for kind, key, port in CASES:
        rel = REFERENCE / f"native-canary-{kind}-{key}-p{port}-plan-v3.json"
        plan = bounded_json(root / rel)
        native.validate_plan(plan, root)
        if ((plan["kind"], plan["key"], plan["port"]) != (kind, key, port)
                or plan["frisson_git"] is not None or plan["frisson_source"] != generated_prefix
                or plan["slippi_source"] != native.SLIPPI_SOURCE):
            raise ValueError("native plan case or unchanged source layout differs")
        add(root / rel, REMOTE_ROOT / rel)
        cases.append({"kind": kind, "key": key, "port": port, "plan": rel.as_posix(),
                      "plan_sha256": transport.digest(root / rel)})
        for record in [plan["checkpoint"], plan["manifest"], plan["replay"], *plan["sources"]]:
            path = relative(record["path"])
            if path.startswith(generated_prefix + "/"):
                if Path(path).name not in snapshot.FILES or record["sha256"] != snapshot.FILES[Path(path).name]:
                    raise ValueError("unapproved generated source blob")
                generated[path] = record
            else:
                add(root / path, REMOTE_ROOT / path)
    for path in EXTRA_SOURCES:
        add(root / path, REMOTE_ROOT / path)
    snap_path = root / REFERENCE / "frisson-source-snapshot.json"
    if transport.digest(snap_path) != SNAPSHOT_SHA:
        raise ValueError("authentic source snapshot identity differs")
    snapshot.validate(bounded_json(snap_path, snapshot.MAX_BYTES))
    add(snap_path, REMOTE_SNAPSHOT)
    add(lock_path, REMOTE_LOCK)
    result = {"schema": SCHEMA, "root": str(root), "output": str(output), "uploads": list(uploads.values()), "cases": cases,
              "generated_sources": list(generated.values()), "frisson_destination": source_parent.as_posix(),
              "slippi": {"url": SLIPPI_URL, "revision": SLIPPI_REVISION, "branch": "main", "destination": native.SLIPPI_SOURCE},
              "dependencies_complete": lock["complete"], "dependency_sha256": transport.digest(lock_path),
              "limits": {"upload_bytes": UPLOAD_BYTES, "dependency_bytes": DEPENDENCY_BYTES, "receipt_bytes": RECEIPT_BYTES,
                         "return_bytes": RETURN_BYTES, "session_seconds": SESSION_SECONDS, "function_seconds": FUNCTION_SECONDS,
                         "bootstrap_seconds": BOOTSTRAP_SECONDS, "case_seconds": CASE_SECONDS, "output_reserve_seconds": OUTPUT_RESERVE_SECONDS,
                         "cpu": [4, 4], "memory_mib": [16384, 16384], "concurrent_workers": 1},
              "runtime": {"sdk": transport.SDK_VERSION, "image": "ubuntu:22.04", "python": "3.12", "nonpreemptible": True,
                          "cpu_float32": True, "native_environment_flags_unchanged": False, "sequential_fresh_processes": True,
                          "native_environment_overrides": NATIVE_BACKEND_ENV,
                          "linux_tensorflow_backend": "standard operations; explicit oneDNN disable for numeric compatibility",
                          "artifact_http_user_agent": HTTP_USER_AGENT,
                          "tls": {"ca_package": "ca-certificates", "ca_bundle": str(SYSTEM_CA_BUNDLE),
                                  "check_hostname": True, "verify_mode": "CERT_REQUIRED"},
                          "py_ubjson_source_build_environment": SOURCE_BUILD_ENV},
              "retry_semantics": {"client_resubmissions": 0, "user_exception_retry_policy": None, "platform_restarts_possible": True},
              "comparison": "Complete receipts return for local native comparison at unchanged atol=rtol=1e-5; cloud success alone is not parity.",
              "no_gameplay": True, "no_rom_secrets_volumes": True, "admitted": False}
    if sum(row["bytes"] for row in uploads.values()) + len(canonical(result)) > UPLOAD_BYTES:
        raise ValueError("explicit upload exceeds 1 GiB")
    return result


def verify_plan(path, sha, remote=False):
    if not SHA.fullmatch(sha) or transport.digest(path) != sha:
        raise ValueError("frozen policy plan SHA differs")
    plan = bounded_json(path)
    if plan.get("schema") != SCHEMA or len(plan.get("cases", [])) != 10:
        raise ValueError("invalid frozen policy plan")
    if [(r["kind"], r["key"], r["port"]) for r in plan["cases"]] != list(CASES):
        raise ValueError("case order differs from frozen ten-case panel")
    total, seen = 0, set()
    for row in plan["uploads"]:
        if set(row) != {"local", "remote", "bytes", "sha256"} or row["remote"] in seen:
            raise ValueError("invalid upload identity")
        remote_path = Path(row["remote"])
        if remote_path not in {REMOTE_LOCK, REMOTE_SNAPSHOT} and (REMOTE_ROOT not in remote_path.parents or relative(remote_path.relative_to(REMOTE_ROOT).as_posix()) is None):
            raise ValueError("upload escaped isolated layout")
        actual = remote_path if remote else Path(row["local"])
        if transport.regular(actual, UPLOAD_BYTES) != row["bytes"] or transport.digest(actual) != row["sha256"]:
            raise ValueError("uploaded input changed")
        seen.add(row["remote"])
        total += row["bytes"]
    if total + path.stat().st_size > UPLOAD_BYTES:
        raise ValueError("upload aggregate differs")
    lock_row = next(row for row in plan["uploads"] if row["remote"] == str(REMOTE_LOCK))
    lock = validate_lock(bounded_json(REMOTE_LOCK if remote else Path(lock_row["local"])), require_complete=True)
    if not plan["dependencies_complete"] or lock_row["sha256"] != plan["dependency_sha256"]:
        raise ValueError("incomplete or mismatched dependency gate")
    if not remote and make_plan(Path(plan["root"]), Path(lock_row["local"]), Path(plan["output"]), existing=True) != plan:
        raise ValueError("plan differs from reconstructed exact local inventory")
    return plan, lock








def run_command(argv, directory, seconds, env):
    directory.mkdir()
    (directory / "logs").mkdir()
    runner = process_guard.Runner(directory, seconds, 1)
    runner.env = env
    try:
        runner.run(argv, WORK, build=True)
    finally:
        transport.create_once(directory / "command.json", runner.commands)




def output_names(result):
    names = ["intent.json", "prepared-runtime.json", "ubjson-extension.json", "tensorflow-backend.json"]
    for index in range(10):
        names.extend((f"case-{index:02d}.json", f"case-{index:02d}.json.error.json", f"case-{index:02d}-status.json"))
    for name in result["commands"]:
        if not re.fullmatch(r"command-[0-9]{3}", name):
            raise ValueError("invalid command output directory")
        names.extend((f"{name}/command.json", f"{name}/logs/001.log"))
    return names


def output_cap(name):
    if re.fullmatch(r"case-0[0-9]\.json", name): return RECEIPT_BYTES
    if re.fullmatch(r"(?:intent|prepared-runtime|ubjson-extension|tensorflow-backend|case-0[0-9]-status|case-0[0-9]\.json\.error)\.json", name): return 1024**2
    if re.fullmatch(r"command-[0-9]{3}/command\.json", name): return 1024**2
    if re.fullmatch(r"command-[0-9]{3}/logs/001\.log", name): return 256 * 1024
    raise ValueError("unapproved policy output name")


def check_native_receipt(receipt, case):
    if (receipt.get("schema") != native.RESULT_SCHEMA or receipt.get("passed") is not True
            or receipt.get("policy_steps") != 384 or receipt.get("approved_plan_sha256") != case["plan_sha256"]
            or receipt.get("live_games") != 0 or receipt.get("source_assets_unchanged") is not True
            or not isinstance(receipt.get("rows"), list) or len(receipt["rows"]) != 384
            or not isinstance(receipt.get("checks"), dict) or not receipt["checks"]
            or any(value is not True for value in receipt["checks"].values())
            or receipt.get("environment", {}).get("environment_flags", {}).get("TF_ENABLE_ONEDNN_OPTS") != "0"
            or not isinstance(receipt.get("plan"), dict)
            or any(receipt["plan"].get(key) != case[key] for key in ("kind", "key", "port"))):
        raise ValueError("native receipt schema, case identity or completeness differs")




def checked_terminal(result, cases, files):
    if (result.get("schema") != SCHEMA or result.get("no_gameplay") is not True or result.get("admitted") is not False
            or result.get("status") not in {"failed", "native-cases-failed", "native-cases-passed-awaiting-local-comparison"}
            or not isinstance(result.get("cases"), list) or len(result["cases"]) > 10
            or not isinstance(result.get("commands"), list) or len(result["commands"]) > 40
            or result["commands"] != [f"command-{i:03d}" for i in range(len(result["commands"]))]):
        raise ValueError("invalid worker terminal schema or inventory")
    for index, row in enumerate(result["cases"]):
        if (not isinstance(row, dict) or row.get("index") != index or type(row.get("passed")) is not bool
                or any(row.get(k) != v for k, v in cases[index].items())
                or row.get("output") != f"case-{index:02d}.json"):
            raise ValueError("terminal case identities differ")
    if not files <= set(output_names(result)):
        raise ValueError("returned files differ from terminal command inventory")
    if result["status"] == "native-cases-passed-awaiting-local-comparison":
        required = {"intent.json", "prepared-runtime.json", "ubjson-extension.json", "tensorflow-backend.json"}
        required.update(f"case-{i:02d}.json" for i in range(10))
        required.update(f"case-{i:02d}-status.json" for i in range(10))
        if len(result["cases"]) != 10 or not all(r["passed"] for r in result["cases"]) or not required <= files:
            raise ValueError("successful terminal lacks ten complete native receipts")
    return result


def receive(output, frames, plan_sha, app_id, expires, cases):
    sequence, aggregate, current, stream, terminal = 0, 0, None, None, None
    files, seen, execution = [], set(), None
    try:
        for frame in frames:
            transport.remaining(expires)
            transport.checked_frame(frame)
            if frame.get("seq") != sequence or type(frame.get("seq")) is not int or terminal is not None:
                raise ValueError("policy stream order differs")
            sequence += 1
            kind = frame.get("kind")
            if sequence == 1:
                execution = transport.checked_execution(frame.get("execution"))
                if kind != "hello" or frame.get("plan_sha256") != plan_sha or not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", frame.get("call_id", "")):
                    raise ValueError("policy stream start identity differs")
                transport.create_once(output / "call.json", {"call_id": frame["call_id"], "app_id": app_id, "execution": execution, "expires_unix": expires})
            elif kind == "file":
                row = frame.get("file")
                if (current is not None or frame.get("execution") != execution or not isinstance(row, dict)
                        or set(row) != {"name", "bytes", "sha256", "original_bytes", "original_sha256", "truncated_tail"}
                        or len(seen) >= 100 or row["name"] in seen):
                    raise ValueError("invalid policy file header")
                cap = output_cap(row["name"])
                if (type(row["bytes"]) is not int or not 0 <= row["bytes"] <= cap
                        or not isinstance(row["sha256"], str) or not SHA.fullmatch(row["sha256"])
                        or type(row["original_bytes"]) is not int or not row["bytes"] <= row["original_bytes"] <= RECEIPT_BYTES
                        or not isinstance(row["original_sha256"], str) or not SHA.fullmatch(row["original_sha256"])
                        or type(row["truncated_tail"]) is not bool
                        or row["truncated_tail"] != (row["bytes"] < row["original_bytes"])
                        or (not row["truncated_tail"] and row["original_sha256"] != row["sha256"])
                        or (row["truncated_tail"] and not row["name"].endswith(".log"))):
                    raise ValueError("invalid policy file bounds or SHA")
                aggregate += row["bytes"]
                if aggregate > RETURN_BYTES: raise ValueError("policy return aggregate exceeds cap")
                destination = output / "returned" / row["name"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.parent.resolve() != destination.parent or destination.exists() or destination.is_symlink():
                    raise ValueError("unsafe or existing policy destination")
                partial = destination.with_name(destination.name + ".partial")
                stream = partial.open("xb")
                current = {"row": row, "destination": destination, "partial": partial, "hash": hashlib.sha256(), "count": 0}
                seen.add(row["name"])
            elif kind == "data":
                data = frame.get("body")
                if (current is None or type(data) is not bytes or not 0 < len(data) <= transport.CHUNK_BYTES
                        or type(frame.get("offset")) is not int or frame.get("offset") != current["count"] or current["count"] + len(data) > current["row"]["bytes"]):
                    raise ValueError("invalid policy stream chunk")
                if stream.write(data) != len(data): raise OSError("short policy receipt write")
                current["count"] += len(data)
                current["hash"].update(data)
            elif kind == "end-file":
                if (current is None or frame.get("name") != current["row"]["name"]
                        or frame.get("sha256") != current["row"]["sha256"]
                        or current["count"] != current["row"]["bytes"] or current["hash"].hexdigest() != current["row"]["sha256"]):
                    raise ValueError("policy receipt length/hash differs")
                stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
                current["partial"].rename(current["destination"])
                files.append(current["row"])
                transport.create_once(output / f"received-{len(files):03d}.json", current["row"])
                current = None
            elif kind == "done":
                if current is not None or frame.get("execution") != execution or frame.get("plan_sha256") != plan_sha or not isinstance(frame.get("result"), dict):
                    raise ValueError("policy terminal identity differs")
                terminal = checked_terminal(frame["result"], cases, seen)
            else:
                raise ValueError("unknown policy stream kind")
        if terminal is None: raise ValueError("incomplete policy stream; partial files retained")
        if terminal["status"] == "native-cases-passed-awaiting-local-comparison":
            backend = bounded_json(output / "returned/tensorflow-backend.json", 1024**2)
            if (backend.get("schema") != "e011.modal-policy-tensorflow-backend.v1"
                    or backend.get("environment_before_import") != NATIVE_BACKEND_ENV
                    or backend.get("mkl_enabled") is not False
                    or backend.get("distribution_version") != "2.21.0.dev20260203"):
                raise ValueError("Linux TensorFlow standard-operation backend receipt differs")
            for index, case in enumerate(cases):
                transport.remaining(expires)
                check_native_receipt(bounded_json(output / "returned" / f"case-{index:02d}.json", RECEIPT_BYTES), case)
                if bounded_json(output / "returned" / f"case-{index:02d}-status.json", 1024**2) != terminal["cases"][index]:
                    raise ValueError("case status file differs from worker terminal")
        result = {"schema": SCHEMA, "plan_sha256": plan_sha, "execution": execution,
                  "worker": terminal, "files": files, "admitted": False, "comparison_required": True}
        transport.create_once(output / "result.json", result)
        return result
    finally:
        if stream is not None:
            stream.flush(); os.fsync(stream.fileno()); stream.close()










def perform(plan, lock, expires):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not hasattr(os, "waitid"):
        raise ValueError("policy pilot requires Linux x86_64 with owned process cleanup")
    transport.remaining(expires)
    WORK.mkdir(parents=True, exist_ok=False)
    transport.create_once(WORK / "intent.json", {"plan_sha256": transport.digest(REMOTE_PLAN), "expires_unix": expires})
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "DEBIAN_FRONTEND": "noninteractive",
           "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
    commands, completed = [], []
    def command(argv, deadline, cap=250, *, source_build=False, native_backend=False):
        if source_build and native_backend: raise ValueError("source-build and native-backend scopes are separate")
        directory = WORK / f"command-{len(commands):03d}"
        commands.append(directory)
        overrides = SOURCE_BUILD_ENV if source_build else NATIVE_BACKEND_ENV if native_backend else {}
        run_command(argv, directory, min(cap, transport.remaining(deadline)), {**env, **overrides})
    result = {"schema": SCHEMA, "status": "failed", "cases": completed, "expires_unix": expires,
              "no_gameplay": True, "admitted": False, "started_unix": time.time()}
    try:
        worker_deadline = min(expires, result["started_unix"] + FUNCTION_SECONDS - OUTPUT_RESERVE_SECONDS)
        result["worker_deadline_unix"] = worker_deadline
        bootstrap = min(worker_deadline, time.time() + BOOTSTRAP_SECONDS)
        python = str(prepared_runtime.require_prepared_runtime(Path("/work/runtime-pilot")))
        transport.create_once(WORK / "prepared-runtime.json", {
            "source": "caller-provided-image", "interpreter": python,
            "dependency_lock_sha256": transport.digest(REMOTE_LOCK),
            "installation_performed": False})
        command([python, "-m", "pip", "check"], bootstrap)
        versions = {row["distribution"]: row["version"] for row in lock["artifacts"]}
        check_versions = "import importlib.metadata as m; expected=" + repr(versions) + "; actual={n:m.version(n) for n in expected}; print(actual); assert actual==expected"
        command([python, "-c", check_versions], bootstrap)
        extension_check = (
            "import ubjson,_ubjson,importlib.metadata as m,hashlib,json; from pathlib import Path; "
            "assert ubjson.EXTENSION_ENABLED is True; assert m.version('py-ubjson')=='0.16.1'; "
            "p=Path(_ubjson.__file__); assert p.is_file() and p.suffix=='.so'; "
            "assert 0<p.stat().st_size<16777216; "
            "r={'version':m.version('py-ubjson'),'extension_enabled':True,'path':str(p),"
            "'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}; "
            f"Path({str(WORK / 'ubjson-extension.json')!r}).write_text(json.dumps(r,sort_keys=True)+chr(10)); print(r)"
        )
        command([python, "-c", extension_check], bootstrap)
        command([python, "-m", "pip", "list", "--format=json"], bootstrap)
        source = REMOTE_ROOT / plan["slippi"]["destination"]
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prepared_runtime.slippi_source(SLIPPI_REVISION), source)
        command([sys.executable, str(REMOTE_ROOT / "scripts/modal_panel_source_snapshot.py"), "materialize",
                 "--snapshot", str(REMOTE_SNAPSHOT), "--destination", str(REMOTE_ROOT / plan["frisson_destination"])], bootstrap, 120)
        backend_probe = (
            "import os,sys,json,importlib.metadata as m; from pathlib import Path; "
            "assert os.environ.get('TF_ENABLE_ONEDNN_OPTS')=='0'; assert 'tensorflow' not in sys.modules; "
            "import tensorflow as tf; from tensorflow.python.util import _pywrap_util_port; "
            "r={'schema':'e011.modal-policy-tensorflow-backend.v1','environment_before_import':{'TF_ENABLE_ONEDNN_OPTS':'0'},"
            "'tensorflow_version':tf.__version__,'distribution_version':m.version('tf-nightly'),'tensorflow_git_version':tf.version.GIT_VERSION,"
            "'mkl_enabled':bool(_pywrap_util_port.IsMklEnabled())}; "
            f"Path({str(WORK / 'tensorflow-backend.json')!r}).write_text(json.dumps(r,sort_keys=True)+chr(10)); print(r); "
            "assert r['mkl_enabled'] is False"
        )
        command([python, "-c", backend_probe], worker_deadline, 120, native_backend=True)
        for index, case in enumerate(plan["cases"]):
            output = WORK / f"case-{index:02d}.json"
            row = {"index": index, **case, "output": output.name, "passed": False}
            completed.append(row)
            try:
                command([python, str(REMOTE_ROOT / "scripts/modal_panel_policy_canary.py"),
                         "--plan", str(REMOTE_ROOT / case["plan"]), "--approved-plan-sha256", case["plan_sha256"],
                         "--root", str(REMOTE_ROOT), "--output", str(output)], worker_deadline, CASE_SECONDS, native_backend=True)
                receipt = bounded_json(output, RECEIPT_BYTES)
                row["passed"] = receipt.get("passed") is True and receipt.get("policy_steps") == 384 and receipt.get("approved_plan_sha256") == case["plan_sha256"]
                del receipt
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"[:2048]
            transport.create_once(WORK / f"case-{index:02d}-status.json", row)
        result["status"] = "native-cases-passed-awaiting-local-comparison" if all(row["passed"] for row in completed) else "native-cases-failed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"[:2048]
    result["ended_unix"] = time.time()
    result["commands"] = [p.name for p in commands]
    return result


def remote_cases(plan_sha, expires):
    plan, lock = verify_plan(REMOTE_PLAN, plan_sha, remote=True)
    if transport.remaining(expires) > SESSION_SECONDS + 1:
        raise ValueError("policy expiry exceeds approved session")
    import modal
    call_id = modal.current_function_call_id()
    if not isinstance(call_id, str) or not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", call_id):
        raise ValueError("native call ID unavailable")
    execution = transport.checked_execution({"task_id": os.environ.get("MODAL_TASK_ID"), "execution_id": uuid.uuid4().hex})
    print(json.dumps({"stage": "policy-worker-start", "execution": execution, "call_id": call_id,
                      "plan_sha256": plan_sha, "expires_unix": expires}), flush=True)
    yield transport.checked_frame({"kind": "hello", "seq": 0, "plan_sha256": plan_sha,
                                   "call_id": call_id, "execution": execution})
    result = perform(plan, lock, expires)
    sequence, total = 1, 0
    for name in output_names(result):
        transport.remaining(expires)
        path = WORK / name
        if not path.exists(): continue
        original_bytes = transport.regular(path, RECEIPT_BYTES)
        cap = output_cap(name)
        if original_bytes > cap and not name.endswith(".log"):
            raise ValueError("native output exceeds its declared cap")
        count = min(cap, original_bytes)
        total += count
        if total > RETURN_BYTES: raise ValueError("policy return aggregate exceeds cap")
        value = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(original_bytes - count)
            while data := stream.read(transport.CHUNK_BYTES):
                transport.remaining(expires)
                value.update(data)
        row = {"name": name, "bytes": count, "sha256": value.hexdigest(), "original_bytes": original_bytes,
               "original_sha256": transport.digest(path), "truncated_tail": count < original_bytes}
        yield transport.checked_frame({"kind": "file", "seq": sequence, "file": row, "execution": execution})
        sequence += 1
        with path.open("rb") as stream:
            stream.seek(original_bytes - count)
            offset = 0
            while data := stream.read(transport.CHUNK_BYTES):
                transport.remaining(expires)
                yield transport.checked_frame({"kind": "data", "seq": sequence, "offset": offset, "body": data})
                sequence += 1
                offset += len(data)
        yield transport.checked_frame({"kind": "end-file", "seq": sequence, "name": name, "sha256": row["sha256"]})
        sequence += 1
    yield transport.checked_frame({"kind": "done", "seq": sequence, "result": result,
                                   "plan_sha256": plan_sha, "execution": execution})


def configure_app(modal, plan_path, plan):
    app = modal.App(APP_NAME, include_source=False)
    image = modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), "PYTHONPATH": str(REMOTE_ROOT / "scripts")})
    for row in plan["uploads"]:
        image = image.add_local_file(row["local"], row["remote"], copy=False)
    image = image.add_local_file(str(plan_path), str(REMOTE_PLAN), copy=False)
    function = app.function(image=image, include_source=False, serialized=False, is_generator=True,
                            cpu=(4, 4), memory=(16384, 16384), max_containers=1, min_containers=0,
                            buffer_containers=0, timeout=FUNCTION_SECONDS, startup_timeout=300,
                            scaledown_window=2, nonpreemptible=True, single_use_containers=True,
                            restrict_modal_access=True)(remote_cases)
    return app, function


def execute(plan_path, sha):
    plan, _lock = verify_plan(plan_path, sha)
    output = Path(plan["output"])
    if plan_path != output / "plan.json" or output.resolve() != output:
        raise ValueError("policy plan must remain in its canonical output")
    with (output / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "intent.json").exists():
            return {"status": "intent-retained-no-resubmission"}
        expires = time.time() + SESSION_SECONDS
        transport.create_once(output / "intent.json", {"plan_sha256": sha, "expires_unix": expires, "logical_submissions": 1})
        try:
            if os.environ.get("MODAL_PROFILE", "frisson") != "frisson" or any(key.startswith("MODAL_") and key != "MODAL_PROFILE" for key in os.environ):
                raise ValueError("unexpected Modal environment override")
            os.environ["MODAL_PROFILE"] = "frisson"
            import importlib.metadata
            if importlib.metadata.version("modal") != transport.SDK_VERSION:
                raise ValueError("Modal SDK differs from frozen version")
            import modal
            app, function = configure_app(modal, plan_path, plan)
            with transport.local_deadline(transport.remaining(expires)), modal.enable_output(), app.run(detach=False, environment_name="main"):
                transport.create_once(output / "app.json", {"app_id": app.app_id})
                return receive(output, function.remote_gen(sha, expires), sha, app.app_id, expires, plan["cases"])
        except BaseException as error:
            transport.create_once(output / "failure.json", {"error": f"{type(error).__name__}: {error}"[:2048],
                                                           "intent_retained": True, "client_resubmission": False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--run", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dependency-lock", type=Path)
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    if args.prepare:
        if not args.dependency_lock or args.approved_plan_sha256: parser.error("prepare requires dependency lock only")
        plan = make_plan(args.root, args.dependency_lock, args.prepare)
        output = Path(plan["output"])
        output.mkdir()
        transport.create_once(output / "plan.json", plan)
        print(json.dumps({"plan": str(output / "plan.json"), "sha256": transport.digest(output / "plan.json"),
                          "ready_for_review": plan["dependencies_complete"], "uploaded": False}))
    else:
        if not args.approved_plan_sha256 or args.dependency_lock: parser.error("run requires approved plan SHA only")
        result = execute(args.run.absolute(), args.approved_plan_sha256)
        print(json.dumps({"status": result.get("worker", {}).get("status", result.get("status")), "admitted": False}))


if __name__ == "__main__":
    main()
