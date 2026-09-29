#!/usr/bin/env python3
"""Explicit, bounded Modal controller-tape qualification with native SLP output.

The caller supplies a successful frozen runtime job and the exact approved
research input paths. Preparation hashes those explicit inputs. Execution is a
separate command. No policy constructor, checkpoint or inference is used here.
"""
from __future__ import annotations

import prepared_runtime

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import uuid

import modal_panel_build_pilot as wire
import modal_panel_linux_build as guard
import modal_panel_linux_host as host
import modal_panel_live_canary as tape
import modal_panel_live_context as context
import modal_panel_policy_pilot as policy
import modal_panel_runtime_pilot as runtime

SCHEMA = "e011.modal-live-pilot.v1"
ROOT, COMPAT, REMOTE = runtime.ROOT, runtime.COMPAT, runtime.REMOTE
PLAN = Path("/opt/live-plan.json")
INPUT = Path("/opt/live-input")
WORK = Path("/work/research/live-pilot")
ENTRY = REMOTE / "scripts/modal_panel_live_pilot.py"
ROM = INPUT / "melee-ntsc-1.02.iso"
FUNCTION_SECONDS, SESSION_SECONDS, OWNER_SECONDS, TAPE_SECONDS = 1800, 2100, 1710, 600
UPLOAD_CAP = tape.IMAGE["byte_length"] + 40 * 1024**2
RETURN_CAP, FILE_CAP, FILE_COUNT = 128 * 1024**2, 32 * 1024**2, 192
APP_NAME = "frisson-linux-controller-tape-v1"
SOURCE_PATHS = tuple(dict.fromkeys((*runtime.SOURCE_PATHS, *tape.SOURCE_PATHS,
                                  "scripts/modal_panel_live_pilot.py")))
PRIOR_NAMES = ("result.json", "call.json", "returned/receipt.json", "returned/qualification.json")
APPROVAL_SCHEMA = "e011.modal-live-pilot-authorization.v1"


def fixed_scope():
    return {"controller_cases": 2, "command_frames_per_case": 320, "reversed_physical_ports": True,
            "save_slp": True, "save_video": False, "checkpoints": 0, "policy_inference_calls": 0,
            "benchmark_ledger_writes": False, "numeric_classification": context.LIMITED_CLASSIFICATION,
            "cross_host_trajectory_equivalence": "unassessed"}


def definition(rows, output, runtime_sha, authorization_sha):
    return {"schema": SCHEMA, "uploads": rows, "output": str(output), "scope": fixed_scope(),
            "prior_runtime_plan_sha256": runtime_sha, "authorization_sha256": authorization_sha,
            "policy_plan_sha256": context.LIMITED_EVIDENCE_SHA256["policy_plan"],
            "runtime_spec": context.runtime_spec(), "source_paths": list(SOURCE_PATHS),
            "data_assets": runtime.DATA_ASSETS, "game_image": tape.IMAGE,
            "limits": {"function_seconds": FUNCTION_SECONDS, "session_seconds": SESSION_SECONDS,
                "owner_seconds": OWNER_SECONDS, "tape_seconds": TAPE_SECONDS, "cpu": [4, 4],
                "memory_mib": [16384, 16384], "nonpreemptible": True, "max_containers": 1,
                "client_resubmissions": 0, "platform_restarts": "possible; same absolute expiry retained",
                "upload_bytes": UPLOAD_CAP, "return_bytes": RETURN_CAP, "return_file_bytes": FILE_CAP},
            "fresh_runtime": {"same_locked94_packages": True, "requalify_install_loader_display": True,
                "expanded_sources_checked_in_tape_child": True, "compiled_extension_byte_equality_to_prior": False},
            "native_account_environment": "preserve existing HOME after matching the operating-system account"}


def expected_destinations():
    return ({str(REMOTE / p) for p in (*SOURCE_PATHS, *runtime.DATA_ASSETS)}
            | {str(runtime.INPUT / p) for p in runtime.INPUT_PATHS}
            | {str(runtime.PLAN), str(ROM), str(INPUT / "authorization.json")}
            | {str(INPUT / "prior" / p) for p in PRIOR_NAMES}
            | {str(INPUT / "policy" / (p + ".json")) for p in context.LIMITED_EVIDENCE_SHA256})


def authorization(value, plan):
    if (not isinstance(value, dict) or set(value) != {"schema", "decision", "scope", "bindings"}
            or value["schema"] != APPROVAL_SCHEMA or value["decision"] != "authorized"
            or value["scope"] != fixed_scope() or value["bindings"] != {
                "game_image_sha256": tape.IMAGE["sha256"], "package_archive_sha256": host.ARCHIVE[1],
                "runtime_plan_sha256": plan["prior_runtime_plan_sha256"],
                "policy_plan_sha256": plan["policy_plan_sha256"]}):
        raise ValueError("saved explicit exact live pilot authorization required")


def make_plan(output, runtime_directory, policy_paths, authorization_path, game_image_path):
    """Explicit preparation only. This function reads the named game image."""
    output = Path(output).absolute()
    authorization_path = Path(authorization_path).absolute()
    if output.resolve(strict=False) != output or not output.is_relative_to(COMPAT):
        raise ValueError("canonical isolated research output required")
    runtime_directory = context.canonical(runtime_directory)
    prior_path = runtime_directory / "plan.json"
    prior = runtime.verify_plan(prior_path, wire.digest(prior_path))
    validate_prior(prior, {p: policy.bounded_json(runtime_directory / p) for p in PRIOR_NAMES})
    if set(policy_paths) != set(context.LIMITED_EVIDENCE_SHA256):
        raise ValueError("exact14 approved V4 evidence paths required")
    rows = list(prior["uploads"])
    known = {r["remote"] for r in rows}
    for name in SOURCE_PATHS:
        if str(REMOTE / name) not in known:
            rows.append(policy.file_record(ROOT / name, REMOTE / name, FILE_CAP))
    rows.append(policy.file_record(prior_path, runtime.PLAN, FILE_CAP))
    for name in PRIOR_NAMES:
        rows.append(policy.file_record(runtime_directory / name, INPUT / "prior" / name, FILE_CAP))
    for name, expected in context.LIMITED_EVIDENCE_SHA256.items():
        row = policy.file_record(Path(policy_paths[name]), INPUT / "policy" / (name + ".json"), context.MAX_METADATA)
        if row["sha256"] != expected: raise ValueError("approved V4 evidence identity differs")
        rows.append(row)
    rows.append(policy.file_record(Path(authorization_path), INPUT / "authorization.json", context.MAX_METADATA))
    image = policy.file_record(Path(game_image_path), ROM, tape.IMAGE["byte_length"])
    if (image["bytes"], image["sha256"]) != (tape.IMAGE["byte_length"], tape.IMAGE["sha256"]):
        raise ValueError("explicit owned NTSC1.02 image differs")
    rows.append(image)
    result = definition(rows, output, wire.digest(prior_path), wire.digest(authorization_path))
    check_layout(result)
    authorization(policy.bounded_json(authorization_path), result)
    return result


def check_layout(plan):
    rows = plan.get("uploads", [])
    if (not isinstance(rows, list) or len(rows) != len(expected_destinations())
            or {r.get("remote") for r in rows} != expected_destinations()
            or plan != definition(rows, plan.get("output"), plan.get("prior_runtime_plan_sha256"), plan.get("authorization_sha256"))):
        raise ValueError("exact live input layout and fixed scope required")
    total = len(wire.canonical(plan))
    for row in rows:
        if (set(row) != {"local", "remote", "bytes", "sha256"} or type(row["bytes"]) is not int
                or not 0 < row["bytes"] <= (tape.IMAGE["byte_length"] if row["remote"] == str(ROM) else FILE_CAP)
                or not wire.SHA.fullmatch(str(row["sha256"]))):
            raise ValueError("bounded exact input identity required")
        total += row["bytes"]
    if total > UPLOAD_CAP: raise ValueError("live upload aggregate bound")


def verify_plan(path, sha, remote=False):
    if wire.digest(path) != sha: raise ValueError("frozen live plan SHA differs")
    plan = policy.bounded_json(path); check_layout(plan)
    for row in plan["uploads"]:
        file = Path(row["remote"] if remote else row["local"])
        if wire.regular(file, row["bytes"]) != row["bytes"] or wire.digest(file) != row["sha256"]:
            raise ValueError("live input bytes changed")
    lookup = {r["remote"]: Path(r["remote"] if remote else r["local"]) for r in plan["uploads"]}
    prior = runtime.verify_plan(lookup[str(runtime.PLAN)], plan["prior_runtime_plan_sha256"], remote=remote)
    prior_files = {name: policy.bounded_json(lookup[str(INPUT / "prior" / name)]) for name in PRIOR_NAMES}
    validate_prior(prior, prior_files, remote=remote)
    for name, expected in context.LIMITED_EVIDENCE_SHA256.items():
        if wire.digest(lookup[str(INPUT / "policy" / (name + ".json"))]) != expected:
            raise ValueError("frozen approved policy evidence differs")
    if wire.digest(lookup[str(INPUT / "authorization.json")]) != plan["authorization_sha256"]:
        raise ValueError("authorization identity differs")
    authorization(policy.bounded_json(lookup[str(INPUT / "authorization.json")]), plan)
    image = next(r for r in plan["uploads"] if r["remote"] == str(ROM))
    if (image["bytes"], image["sha256"]) != (tape.IMAGE["byte_length"], tape.IMAGE["sha256"]):
        raise ValueError("native image identity differs")
    return plan


def validate_prior(prior_plan, files, remote=False):
    """Retain prior qualification and independently verify its returned receipts."""
    result, call, receipt, child = (files[p] for p in PRIOR_NAMES)
    if (result.get("status") != "runtime-qualified-gameplay-unadmitted"
            or wire.checked_execution(result.get("execution")) != wire.checked_execution(call.get("execution"))
            or result.get("qualification") != child):
        raise ValueError("successful bound prior runtime result required")
    expected = {"receipt.json": wire.canonical(receipt), "qualification.json": wire.canonical(child)}
    records = {r["name"]: r for r in result.get("files", [])}
    for name, body in expected.items():
        row = records.get(name, {})
        if row.get("bytes") != len(body) or row.get("sha256") != hashlib.sha256(body).hexdigest() or row.get("truncated_tail") is not False:
            raise ValueError("prior returned runtime file identity differs")
    # The immutable prior plan's local paths belong to the coordinator. The
    # remote copy uses the identical mounted bytes for the lock reader only.
    view = {**prior_plan, "uploads": [{**r, "local": r["remote"]} if remote else r for r in prior_plan["uploads"]]}
    runtime.validate_terminal({**receipt, "execution": result["execution"]}, result["plan_sha256"], view, expected)
    if result["plan_sha256"] != hashlib.sha256(wire.canonical(prior_plan)).hexdigest():
        raise ValueError("prior runtime result/plan binding differs")


def source_manifest(plan):
    return {str(Path(r["remote"]).relative_to(REMOTE)): {k: r[k] for k in ("bytes", "sha256")}
            for r in plan["uploads"] if r["remote"] in {str(REMOTE / p) for p in SOURCE_PATHS if p.endswith(".py")}}


def assemble_admissions(plan, fresh, directory, *, created_at, expires_at, parent_pid):
    """Save fresh runtime evidence and derive the exact downstream tape plan."""
    if fresh.get("status") != "runtime-qualified-gameplay-unadmitted": raise ValueError("fresh runtime qualification required")
    prior = policy.bounded_json(runtime.PLAN)
    view = {**prior, "uploads": [{**r, "local": r["remote"]} for r in prior["uploads"]]}
    runtime.validate_qualified_evidence(fresh["qualification"], view)
    runtime_image = {"image_id": plan["runtime_image_id"], "manifest_sha256": plan["outer_plan_sha256"], "platform": "linux_x86_64"}
    bindings = {"policy_plan_sha256": plan["policy_plan_sha256"], "package_archive_sha256": host.ARCHIVE[1],
                "game_image_sha256": tape.IMAGE["sha256"], "runtime_image": runtime_image}
    evidence_paths = {kind: {} for kind in context.KINDS}
    evidence_paths["policy_parity"] = {name: INPUT / "policy" / (name + ".json") for name in context.LIMITED_EVIDENCE_SHA256}
    evidence_paths["game_image_access"] = {"authorization": INPUT / "authorization.json"}
    evidence_paths["live_execution"] = {"authorization": INPUT / "authorization.json", "owner_entrypoint": ENTRY}
    runtime_body = {"dependency_lock": (runtime.INPUT / "lock.json").read_bytes(),
                    "dependency_pins": (runtime.INPUT / "pins.lock").read_bytes(),
                    "source_manifest": wire.canonical(source_manifest(plan))}
    runtime_body.update({key: wire.canonical(fresh["qualification"]["evidence"][key])
                         for key in ("installed_runtime", "ubjson_extension", "os_packages")})
    directory.mkdir(exist_ok=False)
    for name, body in runtime_body.items():
        path = directory / (name + ".json")
        with path.open("xb") as stream:
            if stream.write(body) != len(body): raise OSError("short runtime evidence write")
        evidence_paths["linux_runtime"][name] = path
    admissions, admission_paths = {}, {}
    for kind, evidence in evidence_paths.items():
        value = {"schema": context.SCHEMA, "kind": kind,
                 "decision": context.LIMITED_DECISION if kind == "policy_parity" else context.KINDS[kind],
                 "bindings": bindings, "evidence": {name: context.read_bound(path)[1] for name, path in evidence.items()}}
        context.validate_admission(value, kind, bindings)
        path = directory / (kind + "-admission.json"); wire.create_once(path, value)
        admission_paths[kind] = path; admissions[kind] = context.read_bound(path)[1]
    source = {str(Path(r["remote"]).relative_to(REMOTE)): {k: r[k] for k in ("bytes", "sha256")}
              for r in plan["uploads"] if r["remote"] in {str(REMOTE / p) for p in tape.SOURCE_PATHS}}
    prepared = tape.make_plan(label="controller-tape", created_at=created_at, expires_at=expires_at,
        source_identities=source, admissions=admissions, runtime_image=runtime_image, game_image=tape.IMAGE,
        udp_ports=[51441, 51442])
    wire.create_once(directory / "tape-plan.json", prepared)
    owner = {"schema": "e011.modal-live-parent-declaration.v1", "parent_pid": parent_pid,
             "plan_sha256": tape.identity(prepared)["sha256"], "expires_at": expires_at,
             "entrypoint": {"path": str(ENTRY), "identity": context.read_bound(ENTRY)[1]}, "runner_sha256": context.RUNNER_SHA}
    descriptor = {"admission_paths": {k: str(v) for k, v in admission_paths.items()},
        "evidence_paths": {k: {n: str(p) for n, p in v.items()} for k, v in evidence_paths.items()},
        "owner_declaration": owner, "policy_plan_sha256": plan["policy_plan_sha256"],
        "prior_runtime_plan_sha256": plan["prior_runtime_plan_sha256"],
        "fresh_runtime_requalified": True, "extension_binary_identical_to_prior": "unassessed"}
    wire.create_once(directory / "context.json", descriptor)
    return prepared, descriptor


@contextmanager
def terminate_cleanup():
    """SIGTERM unwinds the reviewed Runner's child-group cleanup first."""
    old = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("owner termination requested")
    signal.signal(signal.SIGTERM, stop)
    try: yield
    finally: signal.signal(signal.SIGTERM, old)


def native_user_environment():
    """Preserve the real account home needed by Dolphin's GetHomeDirectory.

    Research output remains in WORK and libmelee retains its temporary User
    directory. No task path is assigned to the system home variable.
    """
    import pwd
    value = os.environ.get("HOME")
    account_directory = pwd.getpwuid(os.geteuid()).pw_dir
    if (not value or value != account_directory or not Path(value).is_absolute()
            or Path(value).resolve(strict=True) != Path(value) or not Path(value).is_dir()):
        raise ValueError("existing HOME must identify the actual operating-system account directory")
    return {"HOME": value}


def tape_child():
    descriptor = policy.bounded_json(WORK / "admissions/context.json")
    prepared = policy.bounded_json(WORK / "admissions/tape-plan.json")
    ctx = context.OwnedLinuxContext(project_root=REMOTE, research_output_root=WORK,
        package_root=runtime.WORK / "package", package_manifest=runtime.INPUT / "manifest.json",
        package_audit=runtime.INPUT / "audit.json", game_image_path=ROM,
        admission_paths=descriptor["admission_paths"], evidence_paths=descriptor["evidence_paths"],
        policy_plan_sha256=descriptor["policy_plan_sha256"], owner_declaration=descriptor["owner_declaration"])
    result = tape.run_prepared_plan(prepared, expected_sha256=tape.identity(prepared)["sha256"], context=ctx)
    wire.create_once(WORK / "tape-result.json", result)
    return result


def owner(plan, sha, expires, image_id):
    wire.remaining(expires); WORK.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "status": "failed", "plan_sha256": sha, "scope": fixed_scope(),
              "started_unix": time.time(), "expires_unix": expires, "runtime_image_id": image_id}
    with terminate_cleanup():
        try:
            fresh = runtime.perform(policy.bounded_json(runtime.PLAN), expires - 650)
            result["fresh_runtime_status"] = fresh["status"]
            if fresh["status"] != "runtime-qualified-gameplay-unadmitted": raise ValueError("fresh runtime setup failed")
            now = int(time.time()); until = min(int(expires - 15), now + TAPE_SECONDS)
            if until - now < 60: raise TimeoutError("insufficient bounded tape window")
            prepared, descriptor = assemble_admissions({**plan, "runtime_image_id": image_id, "outer_plan_sha256": sha}, fresh, WORK / "admissions",
                created_at=now, expires_at=until, parent_pid=os.getpid())
            command_dir = WORK / "tape-command"; command_dir.mkdir(); (command_dir / "logs").mkdir()
            runner = guard.Runner(command_dir, until - time.time(), 1)
            runner.env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                          "PYTHONPATH": str(REMOTE / "scripts"), "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
                          **native_user_environment()}
            try: runner.run([str(runtime.WORK / "venv/bin/python"), str(ENTRY), "--tape-child"], WORK, build=True)
            finally:
                result["tape_commands"] = runner.commands
                wire.create_once(command_dir / "commands.json", runner.commands)
            child = policy.bounded_json(WORK / "tape-result.json")
            result["tape"] = child
            result["tape_plan_sha256"] = tape.identity(prepared)["sha256"]
            result["cleanup"] = context.validate_parent_terminal(child["owner"], runner.commands[-1])
            if child.get("status") != "controller-tape-local-transport-passed": raise ValueError("native tape gate did not pass")
            result["status"] = "controller-tape-qualified"
        except BaseException as error: result["error"] = f"{type(error).__name__}: {error}"[:2048]
        finally:
            result["ended_unix"] = time.time(); wire.create_once(WORK / "receipt.json", result)
    return result


def progress_markers():
    return {"dependency-install": runtime.WORK / "command-003", "native-imports": runtime.WORK / "qualification",
            "runtime-display": runtime.WORK / "qualification/xvfb.log", "tape-display": WORK / "controller-tape/xvfb.log",
            **{f"tape-case-{i}": WORK / f"controller-tape/case-{i}/controller_tape_trace.jsonl" for i in (0, 1)},
            **{f"replay-audit-{i}-complete": WORK / f"controller-tape/case-{i}/case.json" for i in (0, 1)}}


def supervise_owner(argv, expires, directory, *, markers=None):
    """One actual owner CLI, with time reserved for its nested Runner cleanup.

    An exited leader is kept unreaped while its group is signalled. SIGTERM
    gives the owner seven seconds to unwind its own five-second Runner cleanup.
    The container remains single-use if infrastructure kills this supervisor.
    """
    if platform.system() != "Linux" or not hasattr(os, "waitid"): raise RuntimeError("Linux WNOWAIT required")
    directory.mkdir(parents=True, exist_ok=False)
    row = {"command": argv, "started_unix": time.time(), "expires_unix": expires}
    account_environment = native_user_environment()
    row["native_user_environment"] = account_environment
    process = None; seen = set(); ends = time.monotonic() + wire.remaining(expires)
    path = directory / "owner.log"
    try:
        with path.open("xb") as output:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                    "PYTHONPATH": str(REMOTE / "scripts"), "PYTHONDONTWRITEBYTECODE": "1", **account_environment, **prepared_runtime.image_environment()})
            row["pid"] = process.pid
            while True:
                observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if observed is not None and observed.si_pid: break
                if time.time() >= expires or time.monotonic() >= ends: raise TimeoutError("owner absolute deadline exhausted")
                if path.stat().st_size > FILE_CAP: raise RuntimeError("owner log bound exceeded")
                for name, candidate in (markers or {}).items():
                    if name not in seen and candidate.exists():
                        print(json.dumps({"stage": name, "observed_unix": time.time()}), flush=True); seen.add(name)
                time.sleep(0.1)
    except BaseException as error: row["error"] = f"{type(error).__name__}: {error}"[:2048]
    finally:
        if process is not None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            grace = time.monotonic() + 7
            while time.monotonic() < grace:
                observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if observed is not None and observed.si_pid: break
                time.sleep(0.05)
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            row["exit_code"] = process.wait(timeout=3)
            row.update(unreaped_leader_group_guard=True, terminated_owned_process_group=True)
        row["ended_unix"] = time.time()
        if path.exists(): row.update(log_bytes=path.stat().st_size, log_sha256=wire.digest(path))
        wire.create_once(directory / "receipt.json", row)
    return row


def allowed_output(name):
    if not isinstance(name, str) or len(name) > 240 or ".." in Path(name).parts: return False
    return bool(re.fullmatch(
        r"(?:receipt\.json|tape-result\.json|supervisor/(?:receipt\.json|owner\.log)|"
        r"runtime/(?:receipt\.json|qualification\.json|logs/[0-9]{3}\.log)|"
        r"admissions/[a-z_-]+\.json|tape-command/(?:commands\.json|logs/001\.log)|"
        r"controller-tape/(?:intent\.json|result\.json|xvfb\.log|host-[0-9]{2}\.log|"
        r"case-[01]/(?:case\.json|controller_tape_trace\.jsonl|controller_tape_canonical_schema_audit_projection\.jsonl|"
        r"replays/[A-Za-z0-9_. -]+\.slp)))", name))


def collect_outputs(supervisor):
    candidates = []
    for base, prefix in ((WORK, ""), (Path("/work/live-supervisor"), "supervisor/")):
        if not base.exists(): continue
        for count, path in enumerate(base.rglob("*")):
            if count >= 2000: raise ValueError("bounded owned output inventory exceeded")
            name = prefix + path.relative_to(base).as_posix()
            if allowed_output(name): candidates.append((path, name))
    result = policy.bounded_json(WORK / "receipt.json") if (WORK / "receipt.json").exists() else {
        "schema": SCHEMA, "status": "failed", "error": "owner produced no terminal receipt"}
    output, total = [], 0
    if (runtime.WORK / "receipt.json").exists():
        for row in runtime.collect_outputs(policy.bounded_json(runtime.WORK / "receipt.json"))["files"]:
            row = {**row, "name": "runtime/" + row["name"]}; output.append(row); total += len(row["body"])
    for path, name in sorted(candidates, key=lambda row: row[1]):
        size = wire.regular(path, FILE_CAP)
        cap = 64 * 1024 if name.endswith(".log") else FILE_CAP
        with path.open("rb") as stream:
            stream.seek(max(0, size - cap)); body = stream.read(cap + 1)
        total += len(body)
        if len(body) > cap or total > RETURN_CAP or len(output) >= FILE_COUNT: raise ValueError("bounded live output exceeded")
        output.append({"name": name, "body": body, "sha256": hashlib.sha256(body).hexdigest(),
            "original_bytes": size, "original_sha256": wire.digest(path), "truncated_tail": size > cap})
    status = result["status"] if supervisor.get("exit_code") == 0 and not supervisor.get("error") else "failed"
    return {"schema": SCHEMA, "status": status, "plan_sha256": result.get("plan_sha256"),
            "scope": fixed_scope(), "files": output}


def remote_live(sha, expires, image_id):
    plan = verify_plan(PLAN, sha, remote=True)
    if not 0 < wire.remaining(expires) <= SESSION_SECONDS + 1: raise ValueError("live session expiry exceeds bound")
    if not re.fullmatch("im-[A-Za-z0-9]{1,100}", image_id): raise ValueError("bound actual Modal image id required")
    import modal
    call = modal.current_function_call_id()
    if not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(call)): raise ValueError("function call identity unavailable")
    execution = wire.checked_execution({"task_id": os.environ.get("MODAL_TASK_ID"), "execution_id": uuid.uuid4().hex})
    print(json.dumps({"stage": "live-start", "execution": execution, "expires_unix": expires}), flush=True)
    yield wire.checked_frame({"kind": "hello", "seq": 0, "plan_sha256": sha, "call_id": call, "execution": execution})
    until = min(expires - 30, time.time() + OWNER_SECONDS)
    supervisor = supervise_owner([sys.executable, str(ENTRY), "--owner", "--sha", sha, "--expires", str(until),
                                   "--image-id", image_id], until, Path("/work/live-supervisor"), markers=progress_markers())
    result = collect_outputs(supervisor)
    if result["plan_sha256"] is None: result["plan_sha256"] = sha
    print(json.dumps({"stage": "live-ended", "status": result["status"]}), flush=True)
    yield from wire.output_frames({**result, "execution": execution})


def receive_files(output, frames, sha, app_id, expires, *, name_check=allowed_output,
                  total_cap=RETURN_CAP, file_cap=FILE_CAP, file_count=FILE_COUNT):
    """Shared verified stream transport; scientific validation happens afterward."""
    sequence, current, terminal, execution, total = 0, None, None, None, 0
    records, files = [], {}
    try:
        for frame in frames:
            wire.remaining(expires); wire.checked_frame(frame)
            if type(frame.get("seq")) is not int or frame["seq"] != sequence or terminal is not None: raise ValueError("live stream order differs")
            sequence += 1; kind = frame.get("kind")
            if sequence == 1:
                if kind != "hello" or frame.get("plan_sha256") != sha or not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(frame.get("call_id"))):
                    raise ValueError("live stream identity missing")
                execution = wire.checked_execution(frame.get("execution"))
                wire.create_once(output / "call.json", {"app_id": app_id, "call_id": frame["call_id"], "execution": execution, "expires_unix": expires})
                continue
            if kind == "file":
                row = frame.get("file", {})
                if current is not None or len(files) >= file_count or set(row) != {"name", "bytes", "sha256", "original_bytes", "original_sha256", "truncated_tail"}:
                    raise ValueError("invalid live file header")
                name = row["name"]
                if (not name_check(name) or name in files or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= file_cap
                        or type(row["original_bytes"]) is not int or not row["bytes"] <= row["original_bytes"] <= file_cap
                        or not wire.SHA.fullmatch(str(row["sha256"])) or not wire.SHA.fullmatch(str(row["original_sha256"]))
                        or type(row["truncated_tail"]) is not bool): raise ValueError("invalid bounded live file identity")
                if row["truncated_tail"]:
                    if not name.endswith(".log") or row["bytes"] > 65536: raise ValueError("only bounded log tails may be truncated")
                elif row["bytes"] != row["original_bytes"] or row["sha256"] != row["original_sha256"]: raise ValueError("original live file identity differs")
                total += row["bytes"]
                if total > total_cap: raise ValueError("live aggregate return bound")
                path = output / "returned" / name; path.parent.mkdir(parents=True, exist_ok=True)
                if path.parent.resolve() != path.parent or path.exists() or path.is_symlink(): raise ValueError("unsafe live output path")
                partial = path.with_suffix(path.suffix + ".partial")
                current = {"row": row, "path": path, "partial": partial, "stream": partial.open("xb"), "bytes": 0, "hash": hashlib.sha256()}
            elif kind == "data":
                if (current is None or type(frame.get("offset")) is not int or frame["offset"] != current["bytes"]
                        or type(frame.get("body")) is not bytes or not 0 < len(frame["body"]) <= wire.CHUNK_BYTES
                        or current["bytes"] + len(frame["body"]) > current["row"]["bytes"]): raise ValueError("live stream chunk differs")
                if current["stream"].write(frame["body"]) != len(frame["body"]): raise OSError("short live stream write")
                current["hash"].update(frame["body"]); current["bytes"] += len(frame["body"])
            elif kind == "end-file":
                if (current is None or frame.get("name") != current["row"]["name"] or frame.get("sha256") != current["row"]["sha256"]
                        or current["bytes"] != current["row"]["bytes"] or current["hash"].hexdigest() != current["row"]["sha256"]): raise ValueError("live stream file end differs")
                current["stream"].flush(); os.fsync(current["stream"].fileno()); current["stream"].close()
                current["partial"].rename(current["path"])
                files[current["row"]["name"]] = current["path"]; records.append(current["row"]); current = None
            elif kind == "done":
                if current is not None or not isinstance(frame.get("result"), dict): raise ValueError("incomplete live stream")
                terminal = frame["result"]
                if wire.checked_execution(terminal.get("execution")) != execution: raise ValueError("live provider execution changed")
            else: raise ValueError("unknown live frame")
        if terminal is None: raise ValueError("live stream lacks terminal")
        return terminal, files, records
    finally:
        if current is not None and not current["stream"].closed:
            current["stream"].flush(); os.fsync(current["stream"].fileno()); current["stream"].close()


def validate_terminal(terminal, files, sha, plan):
    if (terminal.get("schema") != SCHEMA or terminal.get("plan_sha256") != sha or terminal.get("scope") != fixed_scope()
            or terminal.get("status") not in {"failed", "controller-tape-qualified"}): raise ValueError("bound live terminal required")
    if terminal["status"] == "failed": return
    required = {"runtime/receipt.json", "runtime/qualification.json", "receipt.json", "tape-result.json",
                "admissions/tape-plan.json", "supervisor/receipt.json"}
    if not required <= set(files): raise ValueError("complete retained live/runtime receipts required")
    prior_row = next(r for r in plan["uploads"] if r["remote"] == str(runtime.PLAN))
    prior = policy.bounded_json(Path(prior_row["local"]))
    fresh = policy.bounded_json(files["runtime/receipt.json"])
    runtime.validate_terminal({**fresh, "execution": terminal["execution"]}, plan["prior_runtime_plan_sha256"], prior,
        {"receipt.json": files["runtime/receipt.json"].read_bytes(), "qualification.json": files["runtime/qualification.json"].read_bytes()})
    if fresh.get("status") != "runtime-qualified-gameplay-unadmitted": raise ValueError("fresh runtime success required")
    result = policy.bounded_json(files["receipt.json"])
    child = policy.bounded_json(files["tape-result.json"])
    prepared = policy.bounded_json(files["admissions/tape-plan.json"])
    supervisor = policy.bounded_json(files["supervisor/receipt.json"])
    tape.validate_plan(prepared, expected_sha256=tape.identity(prepared)["sha256"], now=prepared["created_at"])
    source = {str(Path(r["remote"]).relative_to(REMOTE)): {k: r[k] for k in ("bytes", "sha256")}
              for r in plan["uploads"] if r["remote"] in {str(REMOTE / name) for name in tape.SOURCE_PATHS}}
    if (prepared["source_identities"] != source or prepared["runtime_image"]["manifest_sha256"] != sha
            or prepared["runtime_image"]["image_id"] != result.get("runtime_image_id")):
        raise ValueError("returned tape source/image differs from frozen outer plan")
    bindings = {"policy_plan_sha256": plan["policy_plan_sha256"], "package_archive_sha256": host.ARCHIVE[1],
                "game_image_sha256": tape.IMAGE["sha256"], "runtime_image": prepared["runtime_image"]}
    for kind in context.KINDS:
        path = files[f"admissions/{kind}-admission.json"]
        body, actual = context.read_bound(path, prepared["required_admissions"][kind])
        context.validate_admission(json.loads(body), kind, bindings)
    if policy.bounded_json(files["admissions/source_manifest.json"]) != source_manifest(plan):
        raise ValueError("expanded source closure differs")
    if (result.get("schema") != SCHEMA or result.get("status") != terminal["status"] or result.get("plan_sha256") != sha
            or result.get("scope") != fixed_scope() or result.get("tape") != child
            or result.get("fresh_runtime_status") != "runtime-qualified-gameplay-unadmitted"
            or child.get("status") != "controller-tape-local-transport-passed"
            or child.get("policy_inference_calls") != 0 or child.get("competitive_games") != 0
            or child.get("plan_sha256") != tape.identity(prepared)["sha256"]
            or supervisor.get("exit_code") != 0 or supervisor.get("error") is not None
            or supervisor.get("unreaped_leader_group_guard") is not True
            or supervisor.get("terminated_owned_process_group") is not True): raise ValueError("complete native tape/owner success required")
    rows = result.get("tape_commands", [])
    if len(rows) != 1 or rows[0].get("command") != [str(runtime.WORK / "venv/bin/python"), str(ENTRY), "--tape-child"]:
        raise ValueError("exact native tape command required")
    if result.get("cleanup") != context.validate_parent_terminal(child["owner"], rows[0]): raise ValueError("tape cleanup does not reproduce")
    if supervisor.get("pid") != child["owner"]["process"].get("parent_pid"):
        raise ValueError("actual supervised owner PID differs")
    if len(child.get("cases", [])) != 2: raise ValueError("both reversed tape cases required")
    for index, item in enumerate(child["cases"]):
        case = policy.bounded_json(files[f"controller-tape/case-{index}/case.json"])
        checks = tape.tape_receipt_checks(observed_frames=case.get("loop", {}).get("observed_frames"),
            controller_audit=case.get("native_controller_audit", {}), transport_checks=case.get("transport_checks", {}),
            policy_calls=case.get("policy_inference_calls"), shutdown_method=case.get("shutdown_method"))
        checks.update(no_runtime_error=case.get("error") is None, exact_frozen_trace_rows=case.get("trace_rows") == 320,
                      native_replay_gate=bool(case.get("native_replay_checks")) and all(v is True for v in case["native_replay_checks"].values()))
        if (item != {"index": index, "status": "controller-tape-case-passed", "receipt": tape.identity(case)}
                or case.get("status") != "controller-tape-case-passed" or case.get("case") != prepared["cases"][index]
                or case.get("plan_sha256") != tape.identity(prepared)["sha256"] or case.get("trace_rows") != 320
                or case.get("checks") != checks or not all(v is True for v in checks.values())):
            raise ValueError("complete exact-case controller checks required")
        for role, name in (("primary", "controller_tape_trace.jsonl"), ("schema_projection", "controller_tape_canonical_schema_audit_projection.jsonl")):
            path = files[f"controller-tape/case-{index}/" + name]
            if {"bytes": wire.regular(path, FILE_CAP), "sha256": wire.digest(path)} != case["trace_identity"][role]:
                raise ValueError("returned native trace identity differs")
        with files[f"controller-tape/case-{index}/controller_tape_trace.jsonl"].open("rb") as stream:
            count = 0
            for count, body in enumerate(stream, 1):
                if count > 320 or len(body) > 65536: raise ValueError("bounded native tape trace rows required")
                tape.validate_tape_row(prepared, index, count - 1, json.loads(body))
            if count != 320: raise ValueError("complete native tape trace required")
        replays = [p for name, p in files.items() if name.startswith(f"controller-tape/case-{index}/replays/")]
        if not any({"bytes": wire.regular(p, FILE_CAP), "sha256": wire.digest(p)} == case.get("game_start_rng", {}).get("replay") for p in replays):
            raise ValueError("native RNG-bound SLP missing")


def configure_app(modal, path, plan):
    app = modal.App(APP_NAME, include_source=False)
    image = modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), "PYTHONPATH": str(REMOTE / "scripts")})
    for row in plan["uploads"]: image = image.add_local_file(row["local"], row["remote"], copy=False)
    image = image.add_local_file(str(path), str(PLAN), copy=False)
    function = app.function(image=image, include_source=False, serialized=False, is_generator=True,
        cpu=(4, 4), memory=(16384, 16384), max_containers=1, min_containers=0, buffer_containers=0,
        timeout=FUNCTION_SECONDS, startup_timeout=300, scaledown_window=2, nonpreemptible=True,
        single_use_containers=True, restrict_modal_access=True)(remote_live)
    return app, function, image


def execute(path, sha):
    plan = verify_plan(path, sha); output = Path(plan["output"])
    if path != output / "plan.json" or output.resolve() != output: raise ValueError("canonical prepared live plan required")
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
            app, fn, image = configure_app(modal, path, plan)
            with wire.local_deadline(wire.remaining(expires)), modal.enable_output(), app.run(detach=False, environment_name="main"):
                wire.create_once(output / "app.json", {"app_id": app.app_id, "image_id": image.object_id})
                terminal, files, records = receive_files(output, fn.remote_gen(sha, expires, image.object_id), sha, app.app_id, expires)
                validate_terminal(terminal, files, sha, plan)
                wire.create_once(output / "result.json", {**terminal, "files": records, "gameplay_admitted": False})
                return terminal
        except BaseException as error:
            wire.create_once(output / "failure.json", {"error": f"{type(error).__name__}: {error}"[:2048], "intent_retained": True})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path); mode.add_argument("--run", type=Path)
    mode.add_argument("--owner", action="store_true"); mode.add_argument("--tape-child", action="store_true")
    parser.add_argument("--sha"); parser.add_argument("--expires", type=float); parser.add_argument("--image-id")
    parser.add_argument("--input-paths", type=Path)
    args = parser.parse_args()
    if args.prepare:
        spec = policy.bounded_json(args.input_paths)
        if set(spec) != {"runtime_directory", "policy_paths", "authorization_path", "game_image_path"}: raise ValueError("exact explicit preparation paths required")
        plan = make_plan(args.prepare, **spec); args.prepare.mkdir(exist_ok=False)
        wire.create_once(args.prepare / "plan.json", plan)
        print(json.dumps({"plan": str(args.prepare / "plan.json"), "sha256": wire.digest(args.prepare / "plan.json")}))
    elif args.owner:
        if not 0 < wire.remaining(args.expires) <= OWNER_SECONDS + 1: raise ValueError("bounded owner expiry required")
        owner(verify_plan(PLAN, args.sha, remote=True), args.sha, args.expires, args.image_id)
    elif args.tape_child: tape_child()
    else: print(json.dumps(execute(args.run, args.sha)))
    return 0


if __name__ == "__main__": raise SystemExit(main())
