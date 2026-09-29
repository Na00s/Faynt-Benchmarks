#!/usr/bin/env python3
"""One bounded Linux runtime qualification, without a game image or policies.

Preparation is local and inert. The execution path checks a caller-provided environment against the exact
public dependency lock, native imports, loader and virtual display.
The Dolphin executable is inspected by its loader; it is never launched.
"""
from __future__ import annotations

import prepared_runtime

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import uuid

import modal_panel_build_pilot as wire
import modal_panel_linux_build as guard
import modal_panel_linux_host as host
import modal_panel_live_context as context
import modal_panel_policy_pilot as policy

SCHEMA = "e011.modal-runtime-pilot.v1"
ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / policy.REFERENCE
REMOTE = Path("/opt/runtime-root")
PLAN = Path("/opt/runtime-plan.json")
INPUT = Path("/opt/runtime-input")
WORK = Path("/work/runtime-pilot")
APP_NAME = "frisson-linux-runtime-qualification-v1"
FUNCTION_SECONDS = 900
SESSION_SECONDS = 1200
WORK_SECONDS = 810
QUALIFY_SECONDS = 180
UPLOAD_CAP = 32 * 1024**2
RETURN_CAP = 8 * 1024**2
PACKAGE_BYTES = 950805582
SOURCE_PATHS = (
    "src/melee_policy/__init__.py", "src/melee_policy/integration/__init__.py",
    *(f"src/melee_policy/integration/{name}.py" for name in (
        "frame_watchdog", "frisson_match", "frisson_policy", "faynt_d0_checkpoints", "mimic_bundle_manifest", "match_runtime",
        "natural_game_end", "post_rl_checkpoints", "runtime_identity", "slippi_ai_policy", "slippi_match", "state_identity")),
    *(f"scripts/{name}.py" for name in (
        "modal_panel_runtime_pilot", "modal_panel_policy_pilot", "modal_panel_build_pilot", "modal_panel_linux_build",
        "modal_panel_policy_canary", "modal_panel_source_snapshot", "modal_panel_live_context", "modal_panel_linux_host",
        "modal_panel_package_audit", "prepared_runtime")),
)
# Import-time static metadata stays separate from the Python code closure.
DATA_ASSETS = {
    "src/melee_policy/integration/mimic_native_bundles.v1.json": {
        "bytes": 31259, "sha256": "adb8fa9ef8c6fd5f054fb68b22552c2c92cacce3f8cc0ac0ca2f72bed9ebcef3"},
}
INPUT_PATHS = {
    "lock.json": "live-dependencies-v1/candidate-lock-94.json",
    "pins.lock": "live-dependencies-v1/requirements-linux-live-94.lock",
    "manifest.json": "pilot-v7/returned/package-manifest.json",
    "audit.json": "pilot-v7/PACKAGE_AUDIT.json",
}


def expected_inputs():
    return {"lock.json": (None, context.LOCK_SHA), "pins.lock": (None, context.PINS_SHA),
            "manifest.json": host.MANIFEST, "audit.json": host.AUDIT_RECEIPT}


def plan_definition(rows, output):
    return {"schema": SCHEMA, "uploads": rows, "output": str(output),
            "runtime": context.runtime_spec(), "source_paths": list(SOURCE_PATHS), "data_assets": DATA_ASSETS,
            "package_acquisition_bytes": 0, "historical_package_acquisition_bytes": PACKAGE_BYTES,
            "limits": {"function_seconds": FUNCTION_SECONDS, "session_seconds": SESSION_SECONDS,
                "work_seconds": WORK_SECONDS, "qualification_seconds": QUALIFY_SECONDS,
                "cpu": [4, 4], "memory_mib": [16384, 16384], "nonpreemptible": True,
                "max_containers": 1, "client_resubmissions": 0, "platform_restarts": "possible; exact expiry retained",
                "upload_bytes": UPLOAD_CAP, "return_bytes": RETURN_CAP},
            "scope": {"verify_existing_exact94": True, "native_imports": True, "loader_inspection": True,
                "xvfb_software_display": True, "dolphin_launches": 0, "rom_reads": 0, "policy_instances": 0,
                "policy_inference": 0, "gameplay_admitted": False}}


def make_plan(output):
    output = Path(output).absolute()
    if output.resolve(strict=False) != output or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("canonical new research output required")
    if not output.is_relative_to(COMPAT): raise ValueError("isolated compatibility research output required")
    rows = [policy.file_record(ROOT / name, REMOTE / name, UPLOAD_CAP) for name in SOURCE_PATHS]
    for name, identity in DATA_ASSETS.items():
        row = policy.file_record(ROOT / name, REMOTE / name, UPLOAD_CAP)
        if {k: row[k] for k in ("bytes", "sha256")} != identity:
            raise ValueError("pinned import-time data asset differs")
        rows.append(row)
    for name, relative in INPUT_PATHS.items():
        row = policy.file_record(COMPAT / relative, INPUT / name, UPLOAD_CAP)
        size, sha = expected_inputs()[name]
        if row["sha256"] != sha or size is not None and row["bytes"] != size:
            raise ValueError("pinned runtime input differs")
        rows.append(row)
    lock = policy.validate_lock(policy.bounded_json(COMPAT / INPUT_PATHS["lock.json"]), require_complete=True)
    if len(lock["artifacts"]) != 94 or sum(r["bytes"] for r in lock["artifacts"]) != PACKAGE_BYTES:
        raise ValueError("exact public94-package acquisition inventory required")
    plan = plan_definition(rows, output)
    if sum(r["bytes"] for r in rows) + len(wire.canonical(plan)) > UPLOAD_CAP:
        raise ValueError("bounded exact upload inventory exceeded")
    return plan


def verify_plan(path, sha, remote=False):
    if wire.digest(path) != sha: raise ValueError("frozen runtime plan SHA differs")
    plan = policy.bounded_json(path)
    allowed = {str(REMOTE / p) for p in (*SOURCE_PATHS, *DATA_ASSETS)} | {str(INPUT / p) for p in INPUT_PATHS}
    rows = plan.get("uploads", [])
    if (not isinstance(rows, list) or len(rows) != len(allowed)
            or {r["remote"] for r in rows} != allowed or plan != plan_definition(rows, plan.get("output"))):
        raise ValueError("runtime plan layout or fixed scope differs")
    for row in rows:
        if set(row) != {"local", "remote", "bytes", "sha256"}: raise ValueError("exact upload row required")
        file = Path(row["remote"] if remote else row["local"])
        if wire.regular(file, UPLOAD_CAP) != row["bytes"] or wire.digest(file) != row["sha256"]:
            raise ValueError("runtime upload changed")
        if Path(row["remote"]).is_relative_to(REMOTE):
            asset = str(Path(row["remote"]).relative_to(REMOTE))
            if asset in DATA_ASSETS and {k: row[k] for k in ("bytes", "sha256")} != DATA_ASSETS[asset]:
                raise ValueError("fixed import-time data identity differs")
        if file.name in expected_inputs() and row["remote"] == str(INPUT / file.name):
            size, expected_sha = expected_inputs()[file.name]
            if row["sha256"] != expected_sha or size is not None and row["bytes"] != size:
                raise ValueError("fixed runtime asset identity differs")
    if sum(r["bytes"] for r in rows) + wire.regular(path, UPLOAD_CAP) > UPLOAD_CAP:
        raise ValueError("runtime upload aggregate bound")
    if not remote and plan != make_plan(Path(plan["output"])): raise ValueError("local plan reconstruction differs")
    return plan


def source_manifest(plan):
    return {str(Path(r["remote"]).relative_to(REMOTE)): {"bytes": r["bytes"], "sha256": r["sha256"]}
            for r in plan["uploads"] if r["remote"] in {str(REMOTE / name) for name in SOURCE_PATHS}}


def data_manifest(plan):
    result = {str(Path(r["remote"]).relative_to(REMOTE)): {k: r[k] for k in ("bytes", "sha256")}
              for r in plan["uploads"] if r["remote"] in {str(REMOTE / name) for name in DATA_ASSETS}}
    if result != DATA_ASSETS: raise ValueError("exact import-time data inventory required")
    return result


def qualification(plan, expires, parent_pid):
    """Runs only inside the single reviewed Runner-owned child session."""
    if (platform.system() != "Linux" or platform.machine() != "x86_64" or os.getpid() != os.getpgid(0)
            or os.getpid() != os.getsid(0) or os.getppid() != parent_pid or not hasattr(os, "waitid")):
        raise RuntimeError("owned Linux qualification session required")
    directory = WORK / "qualification"; directory.mkdir()
    ctx = context.OwnedLinuxContext(project_root=REMOTE, research_output_root=WORK, package_root=WORK / "package",
        package_manifest=INPUT / "manifest.json", package_audit=INPUT / "audit.json", game_image_path=WORK / "unused-image",
        admission_paths={}, evidence_paths={}, policy_plan_sha256="0" * 64, owner_declaration={"parent_pid": parent_pid})
    ctx.directory, ctx.expiry = directory, expires
    ctx.monotonic_end = time.monotonic() + max(0, expires - time.time())
    ctx.check_deadline(expires)
    result = {"schema": SCHEMA, "status": "failed", "pid": os.getpid(), "pgid": os.getpgid(0), "sid": os.getsid(0),
              "parent_pid": parent_pid, "scope": plan["scope"], "parent_terminal_cleanup_verified": False}
    try:
        result["os_release"] = dict(line.split("=", 1) for line in context.proc_text("/etc/os-release", 65536).splitlines() if "=" in line)
        if result["os_release"].get("ID", "").strip('"') != "ubuntu" or result["os_release"].get("VERSION_ID", "").strip('"') != "22.04":
            raise RuntimeError("Ubuntu22.04 qualification host required")
        for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DISPLAY", "WAYLAND_DISPLAY"):
            if os.environ.get(key): raise RuntimeError("fresh display/loader environment required")
        os.environ.update(context.HOST_ENV)
        versions = {}
        for d in importlib.metadata.distributions():
            name = re.sub("[-_.]+", "-", d.metadata["Name"]).lower()
            if name in versions: raise ValueError("duplicate installed distribution")
            versions[name] = d.version
        ctx._capture([sys.executable, "-m", "pip", "check"])
        packages = dict(line.split("\t", 1) for line in ctx._capture(
            ["/usr/bin/dpkg-query", "-W", "-f=${Package}\t${Version}\n", *context.RUNTIME_PACKAGES]).splitlines())
        import ubjson, _ubjson
        if ubjson.EXTENSION_ENABLED is not True: raise ValueError("compiled UBJSON required")
        extension = Path(_ubjson.__file__).resolve(strict=True)
        size = wire.regular(extension, context.MAX_METADATA)
        evidence = {"dependency_lock": (INPUT / "lock.json").read_bytes(), "dependency_pins": (INPUT / "pins.lock").read_bytes(),
            "source_manifest": wire.canonical(source_manifest(plan)),
            "installed_runtime": wire.canonical({"schema": "e011.modal-live-installed-runtime.v1", "versions": versions, "pip_check_passed": True}),
            "ubjson_extension": wire.canonical({"version": importlib.metadata.version("py-ubjson"), "identity": {"bytes": size, "sha256": wire.digest(extension)}}),
            "os_packages": wire.canonical(packages)}
        result["runtime"] = ctx._runtime(evidence)
        evidence["data_manifest"] = wire.canonical(data_manifest(plan))
        result["evidence"] = {k: json.loads(v) for k, v in evidence.items() if k not in ("dependency_pins", "dependency_lock")}
        manifest = source_manifest(plan)
        for name, row in manifest.items(): context.read_bound(REMOTE / name, row)
        for name, row in data_manifest(plan).items(): context.read_bound(REMOTE / name, row)
        ctx._load_native(manifest)
        loaded = sorted(str(Path(m.__file__).resolve().relative_to(REMOTE)) for name, m in sys.modules.items()
                        if name == "melee_policy" or name.startswith("melee_policy."))
        expected = sorted(name for name in SOURCE_PATHS if name.startswith("src/"))
        if loaded != expected: raise ValueError("actual imported project closure differs")
        result["loaded_project_sources"] = loaded
        result["package"] = host.verify_installed(WORK / "package", INPUT / "manifest.json", INPUT / "audit.json")
        result["loader"] = context.parse_loader(ctx._capture([
            "/lib64/ld-linux-x86-64.so.2", "--list", str(WORK / "package/dolphin-emu")]), WORK / "package")
        result["display"] = ctx._start_display()
        ctx.check_deadline(expires)
        result["status"] = "runtime-child-qualified"
    except Exception as error: result["error"] = f"{type(error).__name__}: {error}"[:2048]
    wire.create_once(directory / "receipt.json", result)
    return result


def perform(plan, expires):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not hasattr(os, "waitid"):
        raise RuntimeError("Linux x86_64 with WNOWAIT required")
    wire.remaining(expires); WORK.mkdir(parents=True, exist_ok=True)
    deadline = min(expires, time.time() + WORK_SECONDS)
    wire.create_once(WORK / "intent.json", {"plan_sha256": wire.digest(PLAN), "expires_unix": expires})
    commands = []
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "DEBIAN_FRONTEND": "noninteractive",
           "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}
    def command(argv, seconds=240, source=False):
        d = WORK / f"command-{len(commands):03d}"; d.mkdir(); (d / "logs").mkdir()
        runner = guard.Runner(d, min(seconds, wire.remaining(deadline)), 1)
        runner.env = {**env, **(policy.SOURCE_BUILD_ENV if source else {})}
        try: runner.run(argv, WORK, build=True)
        finally:
            commands.extend(runner.commands); wire.create_once(d / "command.json", runner.commands)
        return runner.commands[-1]
    result = {"schema": SCHEMA, "status": "failed", "scope": plan["scope"], "started_unix": time.time(),
              "expires_unix": expires, "plan_sha256": wire.digest(PLAN), "commands": commands}
    try:
        python = str(prepared_runtime.require_prepared_runtime(WORK))
        policy.validate_lock(policy.bounded_json(INPUT / "lock.json"), require_complete=True)
        result["package_verified"] = host.verify_installed(WORK / "package", INPUT / "manifest.json", INPUT / "audit.json")
        until = min(deadline - 5, time.time() + QUALIFY_SECONDS)
        if until <= time.time(): raise TimeoutError("qualification budget exhausted")
        row = command([python, str(REMOTE / "scripts/modal_panel_runtime_pilot.py"), "--qualify", "--sha", wire.digest(PLAN),
                       "--expires", str(until), "--parent-pid", str(os.getpid())], seconds=until - time.time())
        child = policy.bounded_json(WORK / "qualification/receipt.json")
        result["qualification"] = child
        if child.get("status") != "runtime-child-qualified": raise ValueError("runtime child did not qualify")
        owner = {"status": "owned-linux-tape-context-ready", "parent_terminal_cleanup_verified": False,
                 "process": {"pid": child["pid"]}, "parent_declaration": {"expires_at": until}}
        result["cleanup"] = context.validate_parent_terminal(owner, row)
        result["status"] = "runtime-qualified-gameplay-unadmitted"
    except Exception as error: result["error"] = f"{type(error).__name__}: {error}"[:2048]
    result["ended_unix"] = time.time()
    wire.create_once(WORK / "receipt.json", result)
    return result


def collect_outputs(result):
    candidates = [(WORK / "receipt.json", "receipt.json"), (WORK / "qualification/receipt.json", "qualification.json")]
    logs = sorted(WORK.glob("command-[0-9][0-9][0-9]/logs/001.log"))
    logs += sorted((WORK / "qualification").glob("host-[0-9][0-9].log"))
    logs += [WORK / "qualification/xvfb.log"]
    candidates += [(path, f"logs/{index:03d}.log") for index, path in enumerate(logs)]
    output, count = [], 0
    for path, name in candidates:
        if not path.exists(): continue
        if len(output) >= 64: raise ValueError("runtime return file bound")
        size = wire.regular(path, 32 * 1024**2 if name.endswith(".log") else RETURN_CAP)
        cap = 64 * 1024 if name.endswith(".log") else RETURN_CAP
        with path.open("rb") as stream:
            stream.seek(max(0, size - cap)); body = stream.read(cap + 1)
        count += len(body)
        if len(body) > cap or count > RETURN_CAP: raise ValueError("runtime return byte bound")
        output.append({"name": name, "body": body, "sha256": hashlib.sha256(body).hexdigest(),
            "original_bytes": size, "original_sha256": wire.digest(path), "truncated_tail": size > cap})
    return {**result, "files": output}


def remote_runtime(sha, expires):
    plan = verify_plan(PLAN, sha, remote=True)
    if wire.remaining(expires) > SESSION_SECONDS + 1: raise ValueError("runtime session expiry exceeds approved bound")
    import modal
    call = modal.current_function_call_id()
    if not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(call)): raise ValueError("function call identity unavailable")
    execution = wire.checked_execution({"task_id": os.environ.get("MODAL_TASK_ID"), "execution_id": uuid.uuid4().hex})
    print(json.dumps({"stage": "runtime-start", "execution": execution, "expires_unix": expires}), flush=True)
    yield wire.checked_frame({"kind": "hello", "seq": 0, "plan_sha256": sha, "call_id": call, "execution": execution})
    result = perform(plan, expires)
    print(json.dumps({"stage": "runtime-ended", "status": result["status"], "error": result.get("error")}), flush=True)
    yield from wire.output_frames({**collect_outputs(result), "execution": execution})


def validate_terminal(result, sha, plan, files):
    if (result.get("schema") != SCHEMA or result.get("plan_sha256") != sha or result.get("scope") != plan["scope"]
            or result.get("status") not in {"failed", "runtime-qualified-gameplay-unadmitted"}
            or "receipt.json" not in files):
        raise ValueError("bound runtime terminal receipt required")
    body = json.loads(files["receipt.json"])
    if body != {k: v for k, v in result.items() if k != "execution"}:
        raise ValueError("runtime status differs from returned receipt")
    if result["status"] == "runtime-qualified-gameplay-unadmitted":
        child = result.get("qualification", {})
        if (child.get("schema") != SCHEMA or child.get("status") != "runtime-child-qualified"
                or child.get("scope") != plan["scope"] or child.get("parent_terminal_cleanup_verified") is not False
                or child.get("pid") != child.get("pgid") or child.get("pid") != child.get("sid")
                or child.get("loaded_project_sources") != sorted(p for p in SOURCE_PATHS if p.startswith("src/"))
                or child.get("runtime", {}).get("pip_check_passed") is not True
                or child.get("display", {}).get("software_renderer_verified") is not True
                or child.get("display", {}).get("environment") != context.HOST_ENV
                or "qualification.json" not in files or json.loads(files["qualification.json"]) != child):
            raise ValueError("full native import/runtime/display evidence required")
        validate_qualified_evidence(child, plan)
        if result.get("package_verified") != child.get("package"):
            raise ValueError("parent and child installed package evidence differ")
        rows = result.get("commands", [])
        if not rows: raise ValueError("missing parent command evidence")
        argv = rows[-1].get("command", [])
        if (not isinstance(argv, list) or len(argv) != 9
                or argv[:6] != [str(WORK / "venv/bin/python"), str(REMOTE / "scripts/modal_panel_runtime_pilot.py"),
                                "--qualify", "--sha", sha, "--expires"]
                or argv[7:] != ["--parent-pid", str(child.get("parent_pid"))]):
            raise ValueError("qualification parent command binding differs")
        until = float(argv[6])
        if not 0 < until <= result.get("expires_unix", 0): raise ValueError("qualification expiry binding differs")
        owner = {"status": "owned-linux-tape-context-ready", "parent_terminal_cleanup_verified": False,
                 "process": {"pid": child["pid"]}, "parent_declaration": {"expires_at": until}}
        if result.get("cleanup") != context.validate_parent_terminal(owner, rows[-1]):
            raise ValueError("parent cleanup does not reproduce")


def qualified_lock(plan):
    rows = [r for r in plan["uploads"] if r["remote"] == str(INPUT / "lock.json")]
    if len(rows) != 1: raise ValueError("exact approved runtime lock required")
    body, identity = context.read_bound(rows[0]["local"], {k: rows[0][k] for k in ("bytes", "sha256")})
    if identity["sha256"] != context.LOCK_SHA: raise ValueError("approved runtime lock SHA differs")
    return policy.validate_lock(json.loads(body), require_complete=True)


def qualified_versions(plan):
    lock = qualified_lock(plan)
    return {re.sub("[-_.]+", "-", r["distribution"]).lower(): r["version"] for r in lock["artifacts"]}


def validate_qualified_evidence(child, plan):
    expected = qualified_versions(plan)
    evidence, runtime = child.get("evidence", {}), child.get("runtime", {})
    packages = runtime.get("os_packages", {})
    if (len(expected) != 94 or set(packages) != set(context.RUNTIME_PACKAGES)
            or any(not isinstance(v, str) or not v for v in packages.values())
            or runtime != {"versions": expected, "os_packages": packages, "pip_check_passed": True}
            or set(evidence) != {"source_manifest", "data_manifest", "installed_runtime", "ubjson_extension", "os_packages"}
            or evidence["source_manifest"] != source_manifest(plan)
            or evidence["data_manifest"] != data_manifest(plan)
            or evidence["installed_runtime"] != {"schema": "e011.modal-live-installed-runtime.v1", "versions": expected, "pip_check_passed": True}
            or evidence["os_packages"] != packages):
        raise ValueError("exact94 installed/source/runtime evidence required")
    extension = evidence["ubjson_extension"]
    if not isinstance(extension, dict) or set(extension) != {"version", "identity"} or extension["version"] != "0.16.1":
        raise ValueError("compiled UBJSON evidence required")
    context.identity_shape(extension["identity"])
    package = child.get("package", {})
    required = {"schema": host.SCHEMA, "status": "linux-package-bytes-verified", "target_platform": "linux_x86_64",
        "verification_host": {"system": "Linux", "machine": "x86_64"}, "release": host.RELEASE,
        "source_revision": host.REVISION, "frame_patch_sha256": host.FRAME_PATCH_SHA, "build_patch_sha256": host.BUILD_PATCH_SHA,
        "outer_plan_sha256": host.OUTER_PLAN_SHA, "archive_sha256": host.ARCHIVE[1], "manifest_sha256": host.MANIFEST[1],
        "package_audit_sha256": host.AUDIT_RECEIPT[1], "files": host.MEMBER_COUNT, "bytes": host.MEMBER_BYTES,
        "package_root": str(WORK / "package"), "gameplay_admitted": False, "executed": False,
        "executable": {"path": str(WORK / "package/dolphin-emu"), "bytes": host.EXECUTABLE[0], "sha256": host.EXECUTABLE[1], "format": "ELF64-x86_64"},
        "rust_library": {"path": str(WORK / "package/libslippi_rust_extensions.so"), "bytes": host.RUST_LIBRARY[0], "sha256": host.RUST_LIBRARY[1]}}
    if package != required: raise ValueError("qualified native package evidence differs")
    loader = child.get("loader")
    if not isinstance(loader, dict) or not 1 <= len(loader) <= 256: raise ValueError("qualified loader evidence required")
    lines = "\n".join(f"{k} => {v} (0x0)" for k, v in loader.items())
    if context.parse_loader(lines, WORK / "package") != loader: raise ValueError("qualified loader mapping differs")
    if child.get("os_release", {}).get("ID", "").strip('"') != "ubuntu" or child.get("os_release", {}).get("VERSION_ID", "").strip('"') != "22.04":
        raise ValueError("qualified Ubuntu release evidence missing")


def receive(output, frames, sha, app_id, expires, plan):
    sequence, current, stream, terminal, execution = 0, None, None, None, None
    files, records, used, total = {}, [], set(), 0
    try:
        for frame in frames:
            wire.remaining(expires); wire.checked_frame(frame)
            if type(frame.get("seq")) is not int or frame["seq"] != sequence or terminal is not None:
                raise ValueError("runtime stream order differs")
            sequence += 1
            kind = frame.get("kind")
            if sequence == 1:
                if kind != "hello" or frame.get("plan_sha256") != sha or not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(frame.get("call_id"))):
                    raise ValueError("runtime stream identity missing")
                execution = wire.checked_execution(frame.get("execution"))
                wire.create_once(output / "call.json", {"app_id": app_id, "call_id": frame["call_id"], "execution": execution, "expires_unix": expires})
                continue
            if kind == "file":
                row = frame["file"]
                if current is not None or len(used) >= 64 or set(row) != {"name", "bytes", "sha256", "original_bytes", "original_sha256", "truncated_tail"}:
                    raise ValueError("invalid runtime file header")
                name = row["name"]
                if (not isinstance(name, str) or not re.fullmatch(r"(?:receipt\.json|qualification\.json|logs/[0-9]{3}\.log)", name)
                        or name in used or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= RETURN_CAP
                        or type(row["original_bytes"]) is not int or not row["bytes"] <= row["original_bytes"] <= 32 * 1024**2
                        or not wire.SHA.fullmatch(str(row["sha256"])) or not wire.SHA.fullmatch(str(row["original_sha256"]))
                        or type(row["truncated_tail"]) is not bool):
                    raise ValueError("invalid runtime file identity")
                if not row["truncated_tail"] and (row["original_bytes"] != row["bytes"] or row["original_sha256"] != row["sha256"]):
                    raise ValueError("untruncated file original identity differs")
                if row["truncated_tail"] and not name.endswith(".log"): raise ValueError("runtime receipts cannot be truncated")
                if name.endswith(".log") and row["bytes"] > 64 * 1024: raise ValueError("runtime log return bound")
                total += row["bytes"]
                if total > RETURN_CAP: raise ValueError("runtime total return bound")
                path = output / "returned" / name; path.parent.mkdir(parents=True, exist_ok=True)
                if path.parent.resolve() != path.parent or path.exists() or path.is_symlink(): raise ValueError("unsafe runtime output path")
                partial = path.with_suffix(path.suffix + ".partial")
                stream = partial.open("xb")
                current = {"row": row, "path": path, "partial": partial, "bytes": 0, "hash": hashlib.sha256()}
                used.add(name)
            elif kind == "data":
                if (current is None or type(frame.get("offset")) is not int or frame["offset"] != current["bytes"]
                        or type(frame.get("body")) is not bytes or not 0 < len(frame["body"]) <= wire.CHUNK_BYTES
                        or current["bytes"] + len(frame["body"]) > current["row"]["bytes"]):
                    raise ValueError("runtime chunk mismatch")
                if stream.write(frame["body"]) != len(frame["body"]): raise OSError("short runtime output write")
                current["hash"].update(frame["body"]); current["bytes"] += len(frame["body"])
            elif kind == "end-file":
                if (current is None or frame.get("name") != current["row"]["name"] or frame.get("sha256") != current["row"]["sha256"]
                        or current["bytes"] != current["row"]["bytes"] or current["hash"].hexdigest() != current["row"]["sha256"]):
                    raise ValueError("runtime file end mismatch")
                stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
                current["partial"].rename(current["path"])
                files[current["row"]["name"]] = current["path"].read_bytes()
                records.append(current["row"]); current = None
            elif kind == "done":
                if current is not None or not isinstance(frame.get("result"), dict): raise ValueError("incomplete runtime stream")
                terminal = frame["result"]
                if wire.checked_execution(terminal.get("execution")) != execution: raise ValueError("runtime execution changed")
            else: raise ValueError("unknown runtime stream frame")
        if terminal is None: raise ValueError("runtime stream lacks terminal receipt")
        validate_terminal(terminal, sha, plan, files)
        wire.create_once(output / "result.json", {**terminal, "files": records, "gameplay_admitted": False})
        return terminal
    finally:
        if stream is not None: stream.flush(); os.fsync(stream.fileno()); stream.close()


def configure_app(modal, path, plan):
    app = modal.App(APP_NAME, include_source=False)
    image = modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), "PYTHONPATH": str(REMOTE / "scripts")})
    for row in plan["uploads"]: image = image.add_local_file(row["local"], row["remote"], copy=False)
    image = image.add_local_file(str(path), str(PLAN), copy=False)
    function = app.function(image=image, include_source=False, serialized=False, is_generator=True,
        cpu=(4, 4), memory=(16384, 16384), max_containers=1, min_containers=0, buffer_containers=0,
        timeout=FUNCTION_SECONDS, startup_timeout=300, scaledown_window=2, nonpreemptible=True,
        single_use_containers=True, restrict_modal_access=True)(remote_runtime)
    return app, function


def execute(path, sha):
    plan = verify_plan(path, sha); output = Path(plan["output"])
    if path != output / "plan.json" or output.resolve() != output: raise ValueError("canonical prepared plan required")
    with (output / "coordinator.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "intent.json").exists(): return {"status": "intent-retained-no-resubmission"}
        expires = time.time() + SESSION_SECONDS
        wire.create_once(output / "intent.json", {"plan_sha256": sha, "expires_unix": expires, "logical_submissions": 1})
        try:
            if os.environ.get("MODAL_PROFILE", "frisson") != "frisson" or any(k.startswith("MODAL_") and k != "MODAL_PROFILE" for k in os.environ):
                raise ValueError("unexpected Modal environment override")
            os.environ["MODAL_PROFILE"] = "frisson"
            if importlib.metadata.version("modal") != wire.SDK_VERSION: raise ValueError("pinned Modal SDK required")
            import modal
            app, fn = configure_app(modal, path, plan)
            with wire.local_deadline(wire.remaining(expires)), modal.enable_output(), app.run(detach=False, environment_name="main"):
                wire.create_once(output / "app.json", {"app_id": app.app_id})
                return receive(output, fn.remote_gen(sha, expires), sha, app.app_id, expires, plan)
        except BaseException as error:
            wire.create_once(output / "failure.json", {"error": f"{type(error).__name__}: {error}"[:2048], "intent_retained": True})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path); mode.add_argument("--run", type=Path); mode.add_argument("--qualify", action="store_true")
    parser.add_argument("--sha"); parser.add_argument("--expires", type=float); parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.prepare:
        plan = make_plan(args.prepare); args.prepare.mkdir(exist_ok=False)
        wire.create_once(args.prepare / "plan.json", plan)
        print(json.dumps({"plan": str(args.prepare / "plan.json"), "sha256": wire.digest(args.prepare / "plan.json")}))
    elif args.qualify:
        plan = verify_plan(PLAN, args.sha, remote=True)
        if args.expires is None or not 0 < wire.remaining(args.expires) <= QUALIFY_SECONDS + 1:
            raise ValueError("qualification absolute deadline required")
        qualification(plan, args.expires, args.parent_pid)
    else: print(json.dumps(execute(args.run, args.sha), default=str))
    return 0


if __name__ == "__main__": raise SystemExit(main())
