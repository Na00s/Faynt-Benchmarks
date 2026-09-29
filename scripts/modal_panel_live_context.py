#!/usr/bin/env python3
"""Explicit owned Linux context for the separately admitted controller tape.

Import and construction are inert. There is no CLI, installer or cloud call.
verify_and_prepare is an execution boundary requiring saved admissions and a
reviewed parent entrypoint. Parent cleanup is independently checked afterward.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import select
import signal
import subprocess
import sys
import threading
import time
import types

import modal_panel_linux_host as host

SCHEMA = "e011.modal-live-admission.v1"
LOCK_SHA = "0083888d28351524408a49f9c73a4b47886886339af79004219460f6fc9f729e"
PINS_SHA = "1d9922123beee8953bd6c2234a28062b8f5a4b89012371ca7198c1d713438dd9"
RUNNER_SHA = "0eefacae251003d992a78ecba008315d2c20330a8ff45d0773ed1dbd1743c66d"
MAX_METADATA = 2 * 1024**2
MAX_SOURCES = 100
RUNTIME_PACKAGES = (
    "libgtk-3-0", "libasound2", "libxi6", "libsfml-network2.5", "libusb-1.0-0",
    "libmbedtls14", "libsoil1", "libsm6", "liblzo2-2", "libudev1", "libcurl4",
    "libopengl0", "libglx0", "libxrandr2", "libxxf86vm1", "xvfb",
    "libgl1-mesa-dri", "mesa-utils", "x11-utils",
)
HOST_ENV = {"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
            "TF_ENABLE_ONEDNN_OPTS": "0"}
POLICY_CASES = tuple(f"{kind}-{key}-p{port}" for kind, key in (
    ("frisson", "75m"), ("frisson", "10m"), ("slippi", "fox_d21_ditto_v4"),
    ("slippi", "fox_d24_ditto_v4"), ("slippi", "gm")) for port in (1, 2))
KINDS = {"policy_parity": "pass", "linux_runtime": "pass",
         "game_image_access": "authorized", "live_execution": "authorized"}
LIMITED_DECISION = "authorized-portability-limited-continuation"
LIMITED_CLASSIFICATION = "portability-limited-v4-approved-exact-command-parity"
# This exception identifies one retained experiment and one explicit approval.
# Every other policy experiment continues through comparison_passed unchanged.
LIMITED_EVIDENCE_SHA256 = dict(zip(POLICY_CASES, (
    "744ce21e9117feebbb23f024b8d9a411c9e4d7a80adc0e8017c1d3e63f4f9bb0",
    "aafbd834eceeb6bf2523848006613694d7408d6db4055e09f15e05548b6af106",
    "52ad78c4c8882f3074dd902f0c2a4395dd79f25ca38959ae1699914785dec8b2",
    "394acec127c3595b515876c101459938ee67e4ef74ae5a977bab49dd77ace20e",
    "a5736b2a1895214b8eeb520d35f303dcab2aa9150638a5543ee7c555c0e148f8",
    "03cd1ddf4e183ee8c9c2b40d240af3492b6545fcb75b667e7c6156c8dde7074f",
    "85d30642fa1a0d5f263cc6a942bf9e3d5ab40e226e6d2966a290f371cdcfad5b",
    "f3608fcfcc10b1fb78fbc83dd37885b9667426288e94ec8161868edd6a951139",
    "4c7d48d60e75a6ccd635053aa01ccd01fddf10a5282a6910f7e8844178bea04c",
    "c2ea424761269e8c595c876a960f0a8be7f394c956aa2c2024cd0f9df9051ae2",
), strict=True)) | {
    "policy_plan": "c03d57507438f08240e49e76cf9b8ce14ba600520395a3ebf9b1ac0ef208e5e9",
    "policy_result": "f1a18ee4655e449951f7bc5131aca9a8d17191eef1e9ae03da894c184d6e270c",
    "numeric_diagnostic": "2701ec7bc33af5d79e83aba6e82a41e1127c96f609914fe1b5b6efd4a0443d47",
    "user_approval": "d378d15fdf3ea0541a16e83af79f014b55553aab94e10419b6fe3c2626a4fd78",
}
LIMITED_FAILURE_PATHS = {
    5: ("numeric[231].sample.outputs[1][3].cell", "numeric[231].sample.outputs[1][3].hidden",
        "numeric[232].sample.inputs.args[1][3].cell", "numeric[232].sample.inputs.args[1][3].hidden"),
    7: ("numeric[136].sample.outputs[0].logits.main_stick.x",),
    9: ("numeric[110].sample.outputs[1][3].cell", "numeric[110].sample.outputs[1][3].hidden",
        "numeric[111].sample.inputs.args[1][3].cell", "numeric[111].sample.inputs.args[1][3].hidden"),
}


def runtime_spec():
    return {"os": "ubuntu:22.04", "architecture": "x86_64", "packages": list(RUNTIME_PACKAGES),
            "dependency_lock_sha256": LOCK_SHA, "dependency_pins_sha256": PINS_SHA,
            "environment": dict(HOST_ENV), "display": "fresh inherited Xvfb, TCP disabled, 640x480x24",
            "renderer": "Mesa llvmpipe software OpenGL", "native_disable_audio": False,
            "audio": "native audio enabled; virtual host has no promised physical sound device",
            "package_versions": "exact dpkg receipt required from the admitted runtime image",
            "runtime_admitted": False}


def canonical(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("canonical existing path required")
    return path


def read_bound(path, expected=None, cap=MAX_METADATA):
    path = canonical(path)
    if not path.is_file() or path.stat().st_size > cap:
        raise ValueError("bounded regular evidence file required")
    with path.open("rb") as stream:
        body = stream.read(cap + 1)
    actual = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    if len(body) > cap or (expected is not None and actual != expected):
        raise ValueError("evidence bytes differ or exceed bound")
    return body, actual


def identity_shape(value):
    if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int or not 0 < value["bytes"] <= MAX_METADATA
            or not isinstance(value["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", value["sha256"])):
        raise ValueError("bounded evidence identity required")


def validate_admission(value, kind, bindings):
    limited = kind == "policy_parity" and isinstance(value, dict) and value.get("decision") == LIMITED_DECISION
    if (not isinstance(value, dict) or set(value) != {"schema", "kind", "decision", "bindings", "evidence"}
            or value["schema"] != SCHEMA or value["kind"] != kind or (not limited and value["decision"] != KINDS[kind])
            or value["bindings"] != bindings or not isinstance(value["evidence"], dict)
            or not 1 <= len(value["evidence"]) <= 32):
        raise ValueError("successful explicit admission and exact bindings required")
    for name, row in value["evidence"].items():
        if not re.fullmatch("[a-zA-Z0-9_.-]{1,100}", name):
            raise ValueError("bounded evidence role required")
        identity_shape(row)
    if kind == "policy_parity":
        roles = set(LIMITED_EVIDENCE_SHA256) if limited else set(POLICY_CASES) | {"policy_plan", "policy_result"}
        if set(value["evidence"]) != roles:
            raise ValueError("all exact policy evidence roles required")
        if limited and (bindings.get("policy_plan_sha256") != LIMITED_EVIDENCE_SHA256["policy_plan"]
                or any(value["evidence"][name]["sha256"] != expected for name, expected in LIMITED_EVIDENCE_SHA256.items())):
            raise ValueError("only the explicitly approved retained v4 evidence is eligible")


def comparison_passed(value):
    checks = {"identical_input_bindings", "exact_commands", "frame_context_equal", "reset_barrier_counts_equal",
              "numeric_parity", "native_checks_match", "runtime_contract_equal"}
    numeric = value.get("numeric", {})
    if (value.get("schema") != "e011.modal_policy_canary.comparison.v1" or value.get("passed") is not True
            or set(value.get("checks", {})) != checks or not all(v is True for v in value["checks"].values())
            or value.get("command_mismatch_indices") != [] or numeric.get("passed") is not True
            or numeric.get("mismatch_count") != 0 or numeric.get("atol") != 1e-5 or numeric.get("rtol") != 1e-5):
        raise ValueError("unchanged numeric and exact-command policy parity required")


def limited_comparison_checked(value, index):
    """Recognize the exact numeric-only signature of the approved v4 cases."""
    if index not in LIMITED_FAILURE_PATHS:
        comparison_passed(value)
        return
    expected = {"identical_input_bindings": True, "exact_commands": True, "frame_context_equal": True,
                "reset_barrier_counts_equal": True, "numeric_parity": False,
                "native_checks_match": True, "runtime_contract_equal": True}
    numeric = value.get("numeric", {})
    mismatches = numeric.get("first_mismatches", [])
    if (value.get("schema") != "e011.modal_policy_canary.comparison.v1" or value.get("passed") is not False
            or value.get("checks") != expected or any(type(v) is not bool for v in value["checks"].values())
            or value.get("command_mismatch_indices") != [] or numeric.get("passed") is not False
            or numeric.get("atol") != 1e-5 or numeric.get("rtol") != 1e-5
            or type(numeric.get("mismatch_count")) is not int
            or numeric["mismatch_count"] != len(LIMITED_FAILURE_PATHS[index])
            or tuple(row.get("path") for row in mismatches) != LIMITED_FAILURE_PATHS[index]
            or any(row.get("reason") != "float tolerance exceeded" for row in mismatches)):
        raise ValueError("approved numeric-only v4 failure signature differs")


def limited_evidence_checked(evidence, policy_sha):
    if (set(evidence) != set(LIMITED_EVIDENCE_SHA256) or policy_sha != LIMITED_EVIDENCE_SHA256["policy_plan"]
            or any(hashlib.sha256(evidence[name]).hexdigest() != expected
                   for name, expected in LIMITED_EVIDENCE_SHA256.items())):
        raise ValueError("exact retained v4 evidence and explicit user approval required")
    approval = json.loads(evidence["user_approval"])
    diagnostic = json.loads(evidence["numeric_diagnostic"])
    required_scope = {"proceed_toward_modal_benchmark": True, "preserve_original_failed_comparisons": True,
                      "change_original_numeric_tolerance": False, "waive_other_behavior_source_controller_or_replay_checks": False,
                      "require_live_synchronization_and_replay_gates": True, "claim_identical_future_trajectories": False,
                      "restart_local_games": False, "save_slp": True, "save_video": False}
    summary = diagnostic.get("summary", {})
    if (approval.get("schema") != "e011.modal-policy-portability-limitation-approval.v1"
            or approval.get("decision") != LIMITED_DECISION or approval.get("user_message") != "great let's just proceed then"
            or approval.get("bound_evidence") != {name + "_sha256": LIMITED_EVIDENCE_SHA256[name]
                                                   for name in ("policy_plan", "policy_result", "numeric_diagnostic")}
            or any(approval.get("scope", {}).get(key) is not value for key, value in required_scope.items())
            or diagnostic.get("schema") != "e011.modal-policy-v4-retained-numeric-diagnostic.v1"
            or any(summary.get(key) != value for key, value in {
                "passed_cases": 7, "total_cases": 10, "failed_case_indices": [5, 7, 9],
                "exceeding_tensor_paths": 9, "exceeding_recorded_scalar_occurrences": 9,
                "unique_computed_exceeding_output_scalars": 5, "matched_controller_commands": 3840,
                "all_cases_recorded_float_occurrences": 23245056,
                "encoded_state_action_name_equal": True, "reset_inputs_equal": True}.items())):
        raise ValueError("approved portability limitation or disclosed scope differs")
    return {"classification": LIMITED_CLASSIFICATION, "decision": LIMITED_DECISION,
            "strict_policy_parity_passed": False, "exact_commands_passed": True,
            "strict_numeric_cases_passed": 7, "strict_numeric_cases_total": 10,
            "numeric_failure_case_indices": [5, 7, 9], "numeric_failure_path_counts": [4, 1, 4],
            "atol": 1e-5, "rtol": 1e-5, "future_closed_loop_trajectory_equivalence": "unproven",
            "benchmark_win_rate_effect": "unmeasured", "original_failed_comparisons_preserved": True,
            "evidence_sha256": dict(LIMITED_EVIDENCE_SHA256)}


def bind_policy_evidence(evidence, policy_sha, *, decision="pass"):
    if decision == LIMITED_DECISION:
        qualification = limited_evidence_checked(evidence, policy_sha)
    elif decision == "pass":
        qualification = {"classification": "strict-fixed-observation-policy-parity",
                         "decision": "pass", "strict_policy_parity_passed": True, "exact_commands_passed": True}
    else:
        raise ValueError("explicit policy admission decision required")
    if hashlib.sha256(evidence["policy_plan"]).hexdigest() != policy_sha:
        raise ValueError("admitted policy plan bytes differ")
    plan, result = (json.loads(evidence[name]) for name in ("policy_plan", "policy_result"))
    worker = result.get("worker", {})
    if (plan.get("schema") != "e011.modal-policy-pilot.v1" or result.get("schema") != plan["schema"]
            or result.get("plan_sha256") != policy_sha or worker.get("schema") != plan["schema"]
            or worker.get("status") != "native-cases-passed-awaiting-local-comparison"
            or result.get("admitted") is not False or result.get("comparison_required") is not True
            or len(plan.get("cases", [])) != 10 or len(worker.get("cases", [])) != 10):
        raise ValueError("complete bound policy execution result required")
    files = result.get("files", [])
    if not isinstance(files, list) or len(files) > 128: raise ValueError("bounded policy output inventory required")
    names = [r["name"] for r in files]
    if len(set(names)) != len(names): raise ValueError("duplicate policy output inventory")
    inventory = {r["name"]: r for r in files}
    used = set()
    for index, role in enumerate(POLICY_CASES):
        case, finished = plan["cases"][index], worker["cases"][index]
        if (f"{case['kind']}-{case['key']}-p{case['port']}" != role
                or finished != {**case, "index": index, "output": f"case-{index:02d}.json", "passed": True}):
            raise ValueError("policy case/port inventory differs")
        row = inventory.get(f"case-{index:02d}.json", {})
        if (row.get("truncated_tail") is not False or type(row.get("bytes")) is not int
                or not 0 < row["bytes"] <= 128 * 1024**2 or row.get("original_bytes") != row["bytes"]
                or row.get("original_sha256") != row.get("sha256")
                or not re.fullmatch("[0-9a-f]{64}", str(row.get("sha256")))):
            raise ValueError("complete hash-bound native case output required")
        value = json.loads(evidence[role])
        if decision == LIMITED_DECISION: limited_comparison_checked(value, index)
        else: comparison_passed(value)
        sources = value.get("source_receipts", [])
        if (not isinstance(sources, list) or len(sources) != 2
                or sources[1].get("sha256") != row["sha256"]
                or not re.fullmatch("[0-9a-f]{64}", str(sources[0].get("sha256")))
                or row["sha256"] in used):
            raise ValueError("comparison must bind its distinct mapped native case receipt")
        used.add(row["sha256"])
    return qualification


def parse_loader(text, package_root):
    if len(text) > MAX_METADATA or "not found" in text:
        raise ValueError("dynamic loader dependency failure")
    rows = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\S+) => (/[^\s]+) \(0x[0-9a-fA-F]+\)\s*", line)
        if match:
            name, path = match.groups()
            if name in rows: raise ValueError("duplicate loader dependency")
            rows[name] = path
        elif line.strip() and not re.fullmatch(r"\s*(?:linux-vdso\.so\.1|/[^\s]+) \(0x[0-9a-fA-F]+\)\s*", line):
            raise ValueError("unrecognized dynamic loader line")
    if (rows.get("libslippi_rust_extensions.so") != str(package_root / "libslippi_rust_extensions.so")
            or "libc.so.6" not in rows):
        raise ValueError("adjacent pinned Rust library and libc required")
    if any(name != "libslippi_rust_extensions.so" and not path.startswith(("/lib/", "/usr/lib/"))
           for name, path in rows.items()):
        raise ValueError("unexpected dynamic library search path")
    return rows


def proc_text(path, cap=MAX_METADATA):
    # Linux proc files commonly report st_size=0. Bound the actual read.
    with Path(path).open("rb") as stream: body = stream.read(cap + 1)
    if len(body) > cap: raise ValueError("process metadata byte bound")
    return body.decode("utf-8", errors="strict")


def socket_inodes(pid):
    result = set()
    for index, entry in enumerate(Path(f"/proc/{pid}/fd").iterdir()):
        if index >= 512: raise ValueError("owned process descriptor bound")
        try: target = os.readlink(entry)
        except FileNotFoundError: continue
        match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
        if match: result.add(match[1])
    return result


def owned_udp_rows(text, ports, inodes):
    result = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10: raise ValueError("invalid Linux UDP table")
        port = int(fields[1].split(":")[1], 16)
        if port in ports:
            if fields[9] not in inodes: raise RuntimeError("Slippi port belongs to another process")
            result.append({"port": port, "inode": fields[9]})
    return result


def validate_owner(value, plan_sha, expires, parent_pid):
    if (not isinstance(value, dict) or set(value) != {"schema", "parent_pid", "plan_sha256", "expires_at", "entrypoint", "runner_sha256"}
            or value["schema"] != "e011.modal-live-parent-declaration.v1"
            or type(value["parent_pid"]) is not int or value["parent_pid"] != parent_pid
            or value["plan_sha256"] != plan_sha or value["expires_at"] != expires
            or value["runner_sha256"] != RUNNER_SHA):
        raise ValueError("reviewed parent entrypoint and immutable deadline declaration required")
    row = value["entrypoint"]
    if not isinstance(row, dict) or set(row) != {"path", "identity"}: raise ValueError("explicit owner source required")
    identity_shape(row["identity"])


def validate_parent_terminal(owner_receipt, row):
    """Called by the outer coordinator on its own Runner row after cleanup.

    The child cannot establish this proof. The returned value binds the exact
    observed child PID to the parent-owned WNOWAIT and group-cleanup result.
    """
    if (owner_receipt.get("status") != "owned-linux-tape-context-ready"
            or owner_receipt.get("parent_terminal_cleanup_verified") is not False
            or row.get("pid") != owner_receipt.get("process", {}).get("pid")
            or type(row.get("pid")) is not int or row["pid"] <= 1
            or row.get("unreaped_leader_group_guard") is not True
            or row.get("terminated_owned_process_group") is not True
            or type(row.get("exit_code")) is not int or row["exit_code"] != 0
            or row.get("error") is not None
            or not isinstance(row.get("log_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", row["log_sha256"])
            or type(row.get("ended_unix")) not in (int, float)
            or type(row.get("started_unix")) not in (int, float)
            or not 0 < row["started_unix"] <= row["ended_unix"] <= owner_receipt["parent_declaration"]["expires_at"] + 5):
        raise ValueError("authoritative successful parent Runner cleanup receipt required")
    return {"pid": row["pid"], "parent_terminal_cleanup_verified": True,
            "log_sha256": row["log_sha256"], "ended_unix": row["ended_unix"]}


def _codes(code):
    yield code
    for child in code.co_consts:
        if isinstance(child, types.CodeType): yield from _codes(child)


class OwnedLinuxContext:
    def __init__(self, *, project_root, research_output_root, package_root, package_manifest,
                 package_audit, game_image_path, admission_paths, evidence_paths,
                 policy_plan_sha256, owner_declaration):
        self.project_root = Path(project_root).absolute()
        self.research_output_root = Path(research_output_root).absolute()
        self.package_root = Path(package_root).absolute()
        self.package_manifest, self.package_audit = Path(package_manifest), Path(package_audit)
        self.game_image_path = Path(game_image_path).absolute()
        self.admission_paths, self.evidence_paths = admission_paths, evidence_paths
        self.policy_plan_sha256, self.owner = policy_plan_sha256, owner_declaration
        self.version_lock = threading.Lock()
        self.native = None
        self.policy_qualification = None
        self.directory = self.xvfb = self.expiry = self.monotonic_end = None
        self.command_count = 0

    def check_deadline(self, expiry):
        if (self.expiry is None or expiry != self.expiry or time.time() >= self.expiry
                or time.monotonic() >= self.monotonic_end or os.getppid() != self.owner["parent_pid"]):
            raise TimeoutError("owned host absolute deadline or parent identity changed")
        if self.xvfb is not None and self.xvfb.poll() is not None:
            raise RuntimeError("owned display exited")
        if self.directory is not None:
            log = self.directory / "xvfb.log"
            if log.exists() and log.stat().st_size > MAX_METADATA:
                raise RuntimeError("owned display log bound exceeded")

    def _capture(self, argv, seconds=30):
        """Diagnostic child inherits the parent-owned session, with bounded output."""
        self.check_deadline(self.expiry)
        self.command_count += 1
        if self.command_count > 32: raise RuntimeError("host diagnostic command bound")
        path = self.directory / f"host-{self.command_count:02d}.log"
        ends = min(self.monotonic_end, time.monotonic() + seconds)
        with path.open("xb") as stream:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            try:
                self._same_session(child.pid)
                while child.poll() is None:
                    self.check_deadline(self.expiry)
                    if time.monotonic() >= ends or path.stat().st_size > MAX_METADATA:
                        raise TimeoutError("diagnostic command deadline or byte bound")
                    time.sleep(0.02)
                if child.returncode: raise RuntimeError(f"host diagnostic failed: {path.name}")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=2)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=2)
        return read_bound(path)[0].decode("utf-8", errors="strict")

    def _same_session(self, pid):
        if os.getpgid(pid) != os.getpid() or os.getsid(pid) != os.getpid():
            raise RuntimeError("child escaped the Runner-owned process session")

    def _admissions(self, plan):
        bindings = {"policy_plan_sha256": self.policy_plan_sha256, "package_archive_sha256": host.ARCHIVE[1],
                    "game_image_sha256": plan["game_image"]["sha256"], "runtime_image": plan["runtime_image"]}
        if not re.fullmatch("[0-9a-f]{64}", self.policy_plan_sha256): raise ValueError("explicit policy plan identity required")
        if set(self.admission_paths) != set(KINDS) or set(self.evidence_paths) != set(KINDS):
            raise ValueError("four exact admissions required")
        evidence = {}
        for kind in KINDS:
            value = json.loads(read_bound(self.admission_paths[kind], plan["required_admissions"][kind])[0])
            validate_admission(value, kind, bindings)
            if set(self.evidence_paths[kind]) != set(value["evidence"]): raise ValueError("exact evidence file inventory required")
            evidence[kind] = {}
            for role, row in value["evidence"].items():
                body, _ = read_bound(self.evidence_paths[kind][role], row)
                evidence[kind][role] = body
        policy_admission = json.loads(read_bound(self.admission_paths["policy_parity"], plan["required_admissions"]["policy_parity"])[0])
        self.policy_qualification = bind_policy_evidence(evidence["policy_parity"], self.policy_plan_sha256,
                                                        decision=policy_admission["decision"])
        return evidence

    def _sources(self, plan, body):
        manifest = json.loads(body)
        if not isinstance(manifest, dict) or not 1 <= len(manifest) <= MAX_SOURCES:
            raise ValueError("bounded explicit full source closure required")
        for name, row in manifest.items():
            if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                    or not name.endswith(".py") or not name.startswith(("scripts/", "src/melee_policy/"))):
                raise ValueError("project code-only source closure required")
            identity_shape(row); read_bound(self.project_root / name, row)
        for name, row in plan["source_identities"].items():
            read_bound(self.project_root / name, row)
            if name.endswith(".py") and manifest.get(name) != row:
                raise ValueError("entry source absent from full closure")
        runner = plan["source_identities"].get("scripts/modal_panel_linux_build.py")
        if runner is not None and runner["sha256"] != RUNNER_SHA:
            raise ValueError("reviewed parent Runner source differs")
        return manifest

    def _load_native(self, manifest):
        if any(n == "melee_policy" or n.startswith("melee_policy.") for n in sys.modules):
            raise RuntimeError("fresh native module namespace required")
        sys.path.insert(0, str(self.project_root / "src"))
        modules = {role: importlib.import_module("melee_policy.integration." + name) for role, name in (
            ("mimic", "match_runtime"), ("dispatch", "slippi_ai_policy"),
            ("frisson", "frisson_match"), ("replay", "slippi_match"))}
        modules["melee"] = importlib.import_module("melee")
        modules["console_module"] = importlib.import_module("melee.console")
        for name, module in tuple(sys.modules.items()):
            if name != "melee_policy" and not name.startswith("melee_policy."): continue
            path = canonical(module.__file__)
            relative = path.relative_to(self.project_root).as_posix()
            if relative not in manifest: raise ValueError("loaded project module absent from admitted closure")
            body, _ = read_bound(path, manifest[relative])
            compiled = list(_codes(compile(body, str(path), "exec", dont_inherit=True)))
            for item in vars(module).values():
                if isinstance(item, types.FunctionType) and item.__module__ == name and item.__code__ not in compiled:
                    raise ValueError("native function bytecode differs from source")
        n = types.SimpleNamespace(**modules)
        for alias in ("_ControllerPipeLockstep", "_stop_console", "_load_config", "_resolve_game_image_path"):
            if getattr(n.frisson, alias) is not getattr(n.mimic, alias) or getattr(n.replay, alias) is not getattr(n.mimic, alias):
                raise ValueError("native shared aliases differ")
        if n.frisson._legacy_replay_gate_checks is not n.mimic._legacy_replay_gate_checks:
            raise ValueError("native replay gate alias differs")
        for cls in (n.mimic._ControllerPipeLockstep, n.console_module.Console, n.melee.Controller, n.melee.MenuHelper):
            provider = sys.modules[cls.__module__]
            source = canonical(provider.__file__)
            body, _ = read_bound(source)
            compiled = list(_codes(compile(body, str(source), "exec", dont_inherit=True)))
            for value in vars(cls).values():
                method = value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
                if isinstance(method, types.FunctionType) and method.__code__ not in compiled:
                    raise ValueError("native controller/menu/transport method differs from source")
        host.verify_console_module(n.console_module, host.audit.Budget(30))
        self.native = n

    def _runtime(self, evidence):
        required = {"dependency_lock", "dependency_pins", "source_manifest", "installed_runtime", "ubjson_extension", "os_packages"}
        if set(evidence) != required: raise ValueError("complete saved runtime evidence required")
        if hashlib.sha256(evidence["dependency_lock"]).hexdigest() != LOCK_SHA or hashlib.sha256(evidence["dependency_pins"]).hexdigest() != PINS_SHA:
            raise ValueError("exact frozen94-package lock required")
        lock = json.loads(evidence["dependency_lock"])
        expected = {re.sub("[-_.]+", "-", r["distribution"]).lower(): r["version"] for r in lock["artifacts"]}
        actual = {}
        for distribution in importlib.metadata.distributions():
            name = re.sub("[-_.]+", "-", distribution.metadata["Name"]).lower()
            if name in actual: raise ValueError("duplicate installed distribution")
            actual[name] = distribution.version
        if len(expected) != 94 or actual != expected: raise ValueError("installed exact94-package environment differs")
        installed = json.loads(evidence["installed_runtime"])
        if installed != {"schema": "e011.modal-live-installed-runtime.v1", "versions": actual, "pip_check_passed": True}:
            raise ValueError("saved installed runtime gate differs")
        self._capture([sys.executable, "-m", "pip", "check"])
        package_rows = dict(line.split("\t", 1) for line in self._capture(
            ["/usr/bin/dpkg-query", "-W", "-f=${Package}\t${Version}\n", *RUNTIME_PACKAGES]).splitlines())
        if package_rows != json.loads(evidence["os_packages"]): raise ValueError("admitted Ubuntu runtime packages differ")
        ubjson, extension = importlib.import_module("ubjson"), importlib.import_module("_ubjson")
        extension_row = json.loads(evidence["ubjson_extension"])
        if (not isinstance(extension_row, dict) or set(extension_row) != {"version", "identity"}
                or ubjson.EXTENSION_ENABLED is not True or extension_row["version"] != "0.16.1"):
            raise ValueError("compiled native UBJSON extension required")
        identity_shape(extension_row["identity"])
        read_bound(extension.__file__, extension_row["identity"])
        return {"versions": actual, "os_packages": package_rows, "pip_check_passed": True}

    def _start_display(self):
        read_fd, write_fd = os.pipe()
        log = (self.directory / "xvfb.log").open("xb")
        try:
            self.xvfb = subprocess.Popen(["/usr/bin/Xvfb", "-displayfd", str(write_fd), "-screen", "0", "640x480x24", "-nolisten", "tcp", "-noreset"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, pass_fds=(write_fd,))
        finally:
            os.close(write_fd); log.close()
        try:
            self._same_session(self.xvfb.pid)
            ends = min(self.monotonic_end, time.monotonic() + 20)
            body = b""
            while not body.endswith(b"\n"):
                self.check_deadline(self.expiry)
                if time.monotonic() >= ends or len(body) > 16: raise TimeoutError("owned Xvfb startup bound")
                if select.select([read_fd], [], [], 0.05)[0]:
                    chunk = os.read(read_fd, 17 - len(body))
                    if not chunk: raise RuntimeError("Xvfb display channel closed")
                    body += chunk
            if not re.fullmatch(rb"[0-9]{1,5}\n", body): raise ValueError("invalid owned display number")
        finally: os.close(read_fd)
        display = ":" + body.decode().strip()
        if Path(f"/proc/{self.xvfb.pid}/exe").resolve(strict=True) != Path("/usr/bin/Xvfb"):
            raise ValueError("owned Xvfb executable differs")
        inodes = socket_inodes(self.xvfb.pid)
        socket_path = "/tmp/.X11-unix/X" + body.decode().strip()
        rows = [line.split() for line in proc_text("/proc/net/unix").splitlines()[1:]]
        sockets = [row for row in rows if len(row) == 8 and row[-1] in (socket_path, "@" + socket_path)]
        if not sockets or any(row[6] not in inodes for row in sockets):
            raise RuntimeError("display socket is not owned by this Xvfb")
        os.environ["DISPLAY"] = display
        info = self._capture(["/usr/bin/xdpyinfo", "-display", display])
        gl = self._capture(["/usr/bin/glxinfo", "-B"])
        if "640x480 pixels" not in info or not re.search(r"OpenGL renderer string:.*llvmpipe", gl):
            raise RuntimeError("declared software display renderer unavailable")
        return {"pid": self.xvfb.pid, "display": display, "software_renderer_verified": True,
                "tcp_listener_disabled": True, "environment": dict(HOST_ENV)}

    def verify_and_prepare(self, plan, directory):
        import modal_panel_live_canary as tape
        self.directory = canonical(directory)
        self.expiry = plan["expires_at"]
        self.monotonic_end = time.monotonic() + max(0, self.expiry - time.time())
        plan_sha = tape.identity(plan)["sha256"]
        validate_owner(self.owner, plan_sha, self.expiry, os.getppid())
        if (platform.system() != "Linux" or platform.machine() != "x86_64" or not hasattr(os, "waitid")
                or os.getpid() != os.getpgid(0) or os.getpid() != os.getsid(0)):
            raise RuntimeError("fresh Linux Runner-owned child session required")
        self.check_deadline(self.expiry)
        os_release = dict(line.split("=", 1) for line in proc_text("/etc/os-release", 65536).splitlines() if "=" in line)
        if os_release.get("ID", "").strip('"') != "ubuntu" or os_release.get("VERSION_ID", "").strip('"') != "22.04":
            raise RuntimeError("admitted Ubuntu22.04 runtime required")
        evidence = self._admissions(plan)
        owner_source = self.owner["entrypoint"]
        if evidence["live_execution"].get("owner_entrypoint") is None:
            raise ValueError("saved owner entrypoint approval required")
        owner_body, _ = read_bound(owner_source["path"], owner_source["identity"])
        if owner_body != evidence["live_execution"]["owner_entrypoint"]:
            raise ValueError("approved parent entrypoint differs")
        parent_args = proc_text(f"/proc/{os.getppid()}/cmdline", 65536)
        if str(canonical(owner_source["path"])) not in parent_args.split("\0"):
            raise RuntimeError("actual parent command does not name reviewed owner entrypoint")
        sources = self._sources(plan, evidence["linux_runtime"]["source_manifest"])
        for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DISPLAY", "WAYLAND_DISPLAY"):
            if os.environ.get(key): raise RuntimeError("fresh loader/display environment required")
        os.environ.update(HOST_ENV)
        runtime = self._runtime(evidence["linux_runtime"])
        self._load_native(sources)
        identity = host.verify_installed(self.package_root, self.package_manifest, self.package_audit)
        loader = parse_loader(self._capture(["/lib64/ld-linux-x86-64.so.2", "--list", str(self.package_root / "dolphin-emu")]), self.package_root)
        display = self._start_display()
        config, root = self.native.mimic._load_config(self.project_root / "configs/integration.toml")
        if root != self.project_root or {k: config["game_image"][k] for k in plan["game_image"]} != plan["game_image"]:
            raise ValueError("native game image configuration differs")
        self.game_image_path = self.native.mimic._resolve_game_image_path(config, root, self.game_image_path)
        for case in plan["cases"]: self.native.mimic._assert_udp_port_available(case["slippi_port"])
        return {"status": "owned-linux-tape-context-ready", "plan_sha256": plan_sha,
                "admissions": plan["required_admissions"], "runtime_image": plan["runtime_image"],
                "policy_qualification": self.policy_qualification,
                "process": {"pid": os.getpid(), "pgid": os.getpgid(0), "sid": os.getsid(0), "parent_pid": os.getppid()},
                "parent_declaration": self.owner, "parent_terminal_cleanup_verified": False,
                "package": identity, "runtime": runtime, "display": display, "loader": loader,
                "host_differences": runtime_spec(), "gameplay_admitted": False}

    def verify_inherited_process(self, process, plan):
        self.check_deadline(plan["expires_at"])
        if process is None or process.poll() is not None: raise RuntimeError("live owned Dolphin child required")
        self._same_session(process.pid)
        executable = Path(f"/proc/{process.pid}/exe").resolve(strict=True)
        if executable != self.package_root / "dolphin-emu": raise ValueError("Dolphin executable differs")
        actual = host.audit.digest(executable, host.audit.Budget(30), host.audit.MAX_ELF)
        if actual != host.EXECUTABLE: raise ValueError("loaded Dolphin byte identity differs")
        library = self.package_root / "libslippi_rust_extensions.so"
        if host.audit.digest(library, host.audit.Budget(30), host.audit.MAX_ELF) != host.RUST_LIBRARY:
            raise ValueError("loaded Rust library byte identity differs")
        ends = min(self.monotonic_end, time.monotonic() + 20)
        while True:
            self.check_deadline(self.expiry)
            if process.poll() is not None: raise RuntimeError("Dolphin exited during loaded-library/UDP gate")
            self._same_session(process.pid)
            maps = proc_text(f"/proc/{process.pid}/maps")
            loaded = any(line.split()[-1:] == [str(library)] for line in maps.splitlines())
            inodes = socket_inodes(process.pid)
            ports = {case["slippi_port"] for case in plan["cases"]}
            udp = owned_udp_rows(proc_text("/proc/net/udp"), ports, inodes)
            udp += owned_udp_rows(proc_text("/proc/net/udp6"), ports, inodes)
            if loaded and len(udp) == 1: break
            if time.monotonic() >= ends: raise TimeoutError("loaded Rust library or exclusive Slippi UDP gate failed")
            time.sleep(0.02)
        return {"pid": process.pid, "pgid": os.getpgid(process.pid), "sid": os.getsid(process.pid),
                "executable": {"bytes": actual[0], "sha256": actual[1]}, "adjacent_rust_loaded": True,
                "udp": udp, "parent_terminal_cleanup_verified": False}

    def read_game_start_rng(self, path):
        self.check_deadline(self.expiry)
        path = canonical(path)
        if not path.is_relative_to(self.directory): raise ValueError("RNG replay must belong to this owned run")
        _, row = read_bound(path, cap=32 * 1024**2)
        game = importlib.import_module("peppi_py").read_slippi(str(path), skip_frames=True)
        seed = game.start.random_seed
        if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError("native game-start RNG missing")
        return {"game_random_seed": seed, "replay": row, "parser": "peppi-py-vladfi==0.9.2",
                "cross_host_equivalence": "requires-matched-start-and-RNG"}
