"""Synthetic host/admission fixtures only. No child, emulator, ROM or network."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_live_context as c


def encoded(value): return json.dumps(value, sort_keys=True).encode()
def ident(body): return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
def bound(tmp_path, name, value):
    path = tmp_path / name; body = encoded(value); path.write_bytes(body)
    return path, ident(body)


def comparison():
    return {"schema": "e011.modal_policy_canary.comparison.v1", "passed": True,
            "checks": {n: True for n in ("identical_input_bindings", "exact_commands", "frame_context_equal",
                "reset_barrier_counts_equal", "numeric_parity", "native_checks_match", "runtime_contract_equal")},
            "command_mismatch_indices": [], "numeric": {"passed": True, "mismatch_count": 0, "atol": 1e-5, "rtol": 1e-5}}


def context(tmp_path, **extra):
    kwargs = dict(project_root=tmp_path, research_output_root=tmp_path, package_root=tmp_path,
                  package_manifest=tmp_path / "manifest", package_audit=tmp_path / "audit",
                  game_image_path=tmp_path / "never-read.iso", admission_paths={}, evidence_paths={},
                  policy_plan_sha256="a" * 64, owner_declaration={})
    kwargs.update(extra)
    return c.OwnedLinuxContext(**kwargs)


def test_construction_and_spec_have_no_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("constructor read file"))
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **k: pytest.fail("constructor launched process"))
    monkeypatch.setattr(c.importlib, "import_module", lambda *a, **k: pytest.fail("constructor imported runtime"))
    x = context(tmp_path)
    assert x.native is None and x.xvfb is None
    spec = c.runtime_spec()
    assert spec["native_disable_audio"] is False and spec["runtime_admitted"] is False
    assert spec["environment"] == {"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe", "TF_ENABLE_ONEDNN_OPTS": "0"}
    assert not any("-dev" in name for name in spec["packages"])


def test_bound_read_rejects_growth_symlink_and_hash(tmp_path):
    p = tmp_path / "record"; p.write_bytes(b"abc")
    assert c.read_bound(p, ident(b"abc")) == (b"abc", ident(b"abc"))
    with pytest.raises(ValueError): c.read_bound(p, ident(b"abd"))
    with pytest.raises(ValueError): c.read_bound(p, cap=2)
    link = tmp_path / "link"; link.symlink_to(p)
    with pytest.raises(ValueError): c.read_bound(link)


@pytest.mark.parametrize("change", [
    lambda x: x.update(passed=False), lambda x: x["checks"].update(numeric_parity=False),
    lambda x: x["checks"].pop("exact_commands"), lambda x: x.update(command_mismatch_indices=[0]),
    lambda x: x["numeric"].update(atol=1e-4), lambda x: x["numeric"].update(rtol=1e-4),
    lambda x: x["numeric"].update(mismatch_count=1), lambda x: x["numeric"].update(passed=False),
])
def test_existing_comparison_gate_remains_strict(change):
    value = comparison(); c.comparison_passed(value); change(value)
    with pytest.raises(ValueError): c.comparison_passed(value)


def admission(kind, bindings):
    roles = (*c.POLICY_CASES, "policy_plan", "policy_result") if kind == "policy_parity" else ("evidence",)
    return {"schema": c.SCHEMA, "kind": kind, "decision": c.KINDS[kind], "bindings": bindings,
            "evidence": {name: ident(b"{}") for name in roles}}


@pytest.mark.parametrize("kind", list(c.KINDS))
def test_admission_decisions_and_bindings(kind):
    bindings = {"policy_plan_sha256": "a" * 64}
    value = admission(kind, bindings); c.validate_admission(value, kind, bindings)
    changed = copy.deepcopy(value); changed["decision"] = "pending"
    with pytest.raises(ValueError): c.validate_admission(changed, kind, bindings)
    with pytest.raises(ValueError): c.validate_admission(value, kind, {"policy_plan_sha256": "b" * 64})
    changed = copy.deepcopy(value); changed["extra"] = True
    with pytest.raises(ValueError): c.validate_admission(changed, kind, bindings)


def test_all_ten_unique_policy_evidence_roles_required():
    value = admission("policy_parity", {})
    value["evidence"].pop(c.POLICY_CASES[-1])
    with pytest.raises(ValueError): c.validate_admission(value, "policy_parity", {})


def test_read_saved_admissions_checks_bodies_before_native_import(tmp_path, monkeypatch):
    x = context(tmp_path)
    policy = policy_evidence()
    x.policy_plan_sha256 = hashlib.sha256(policy["policy_plan"]).hexdigest()
    plan = {"game_image": {"sha256": "c" * 64}, "runtime_image": {"image_id": "im-fake"}, "required_admissions": {}}
    bindings = {"policy_plan_sha256": x.policy_plan_sha256, "package_archive_sha256": c.host.ARCHIVE[1],
                "game_image_sha256": "c" * 64, "runtime_image": plan["runtime_image"]}
    for kind in c.KINDS:
        value = admission(kind, bindings); x.evidence_paths[kind] = {}
        for role in value["evidence"]:
            p, row = bound(tmp_path, kind + role + ".json", json.loads(policy[role]) if kind == "policy_parity" else {})
            value["evidence"][role] = row; x.evidence_paths[kind][role] = p
        path, row = bound(tmp_path, kind + ".json", value)
        x.admission_paths[kind] = path; plan["required_admissions"][kind] = row
    monkeypatch.setattr(c.importlib, "import_module", lambda *a: pytest.fail("admission imported policy"))
    assert set(x._admissions(plan)) == set(c.KINDS)
    x.evidence_paths["policy_parity"][c.POLICY_CASES[0]].write_bytes(b"{}")
    with pytest.raises(ValueError): x._admissions(plan)


def policy_evidence():
    plan = {"schema": "e011.modal-policy-pilot.v1", "cases": []}
    files, finished, evidence = [], [], {}
    for index, role in enumerate(c.POLICY_CASES):
        kind, key, port = role.split("-")
        case = {"kind": kind, "key": key, "port": int(port[1:]), "plan": f"plan-{index}", "plan_sha256": f"{index:064x}"}
        plan["cases"].append(case)
        finished.append({**case, "index": index, "output": f"case-{index:02d}.json", "passed": True})
        digest = f"{index + 100:064x}"
        files.append({"name": f"case-{index:02d}.json", "bytes": 100, "original_bytes": 100,
                      "sha256": digest, "original_sha256": digest, "truncated_tail": False})
        value = comparison(); value["source_receipts"] = [{"sha256": f"{index + 200:064x}"}, {"sha256": digest}]
        evidence[role] = encoded(value)
    evidence["policy_plan"] = encoded(plan)
    evidence["policy_result"] = encoded({"schema": plan["schema"], "admitted": False, "comparison_required": True,
        "plan_sha256": hashlib.sha256(evidence["policy_plan"]).hexdigest(), "files": files,
        "worker": {"schema": plan["schema"], "status": "native-cases-passed-awaiting-local-comparison", "cases": finished}})
    return evidence


@pytest.mark.parametrize("change", ["duplicate-comparison", "swapped-comparison", "wrong-plan", "wrong-result-plan", "wrong-port", "duplicate-file", "truncated-case"])
def test_policy_admission_binds_each_actual_case(change):
    evidence = policy_evidence(); sha = hashlib.sha256(evidence["policy_plan"]).hexdigest()
    c.bind_policy_evidence(evidence, sha)
    first, second = c.POLICY_CASES[:2]
    if change == "duplicate-comparison": evidence[second] = evidence[first]
    if change == "swapped-comparison": evidence[first], evidence[second] = evidence[second], evidence[first]
    if change == "wrong-plan": evidence["policy_plan"] += b" "
    result = json.loads(evidence["policy_result"])
    if change == "wrong-result-plan": result["plan_sha256"] = "f" * 64
    if change == "wrong-port": result["worker"]["cases"][0]["port"] = 2
    if change == "duplicate-file": result["files"].append(result["files"][0])
    if change == "truncated-case": result["files"][0]["truncated_tail"] = True
    evidence["policy_result"] = encoded(result)
    with pytest.raises(ValueError): c.bind_policy_evidence(evidence, sha)


def owner():
    return {"schema": "e011.modal-live-parent-declaration.v1", "parent_pid": 111,
            "plan_sha256": "a" * 64, "expires_at": 200, "runner_sha256": c.RUNNER_SHA,
            "entrypoint": {"path": "/approved/owner.py", "identity": ident(b"source")}}


@pytest.mark.parametrize("change", [lambda x: x.update(parent_pid=112), lambda x: x.update(parent_pid=True),
    lambda x: x.update(expires_at=201), lambda x: x.update(plan_sha256="b" * 64),
    lambda x: x.update(runner_sha256="b" * 64), lambda x: x.pop("entrypoint")])
def test_parent_declared_bindings_are_exact(change):
    value = owner(); c.validate_owner(value, "a" * 64, 200, 111); change(value)
    with pytest.raises(ValueError): c.validate_owner(value, "a" * 64, 200, 111)


def test_monotonic_and_wall_deadlines_cannot_refresh(tmp_path, monkeypatch):
    x = context(tmp_path, owner_declaration=owner()); x.expiry = 200; x.monotonic_end = 150
    monkeypatch.setattr(c.os, "getppid", lambda: 111)
    monkeypatch.setattr(c.time, "time", lambda: 199)
    monkeypatch.setattr(c.time, "monotonic", lambda: 149)
    x.check_deadline(200)
    with pytest.raises(TimeoutError): x.check_deadline(201)
    monkeypatch.setattr(c.time, "monotonic", lambda: 150)
    with pytest.raises(TimeoutError): x.check_deadline(200)
    monkeypatch.setattr(c.time, "monotonic", lambda: 149)
    monkeypatch.setattr(c.time, "time", lambda: 200)
    with pytest.raises(TimeoutError): x.check_deadline(200)


def loader(root):
    return f"linux-vdso.so.1 (0xabc)\nlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x123)\nlibslippi_rust_extensions.so => {root}/libslippi_rust_extensions.so (0x456)\n/lib64/ld-linux-x86-64.so.2 (0x789)\n"


@pytest.mark.parametrize("change", [lambda t: t.replace("/package/libslippi", "/other/libslippi"),
    lambda t: t + "libbad.so => not found\n", lambda t: t + "garbage\n",
    lambda t: t + "libc.so.6 => /lib/libc.so.6 (0x12)\n",
    lambda t: t.replace("/lib/x86_64-linux-gnu/libc.so.6", "/tmp/libc.so.6")])
def test_loader_is_adjacent_and_fail_closed(change):
    text = loader("/package")
    assert c.parse_loader(text, Path("/package"))["libc.so.6"].startswith("/lib/")
    with pytest.raises(ValueError): c.parse_loader(change(text), Path("/package"))


def test_udp_inode_ownership():
    header = "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
    row = "1: 00000000:C8F1 00000000:0000 07 00000000:00000000 00:00000000 00000000 1000 0 12345 2\n"
    assert c.owned_udp_rows(header + row, {51441}, {"12345"}) == [{"port": 51441, "inode": "12345"}]
    assert c.owned_udp_rows(header + row, {51442}, set()) == []
    with pytest.raises(RuntimeError): c.owned_udp_rows(header + row, {51441}, {"other"})
    with pytest.raises(ValueError): c.owned_udp_rows(header + "short", {51441}, set())


def test_source_manifest_is_explicit_and_hash_bound(tmp_path):
    x = context(tmp_path)
    (tmp_path / "scripts").mkdir(); p = tmp_path / "scripts/a.py"; p.write_bytes(b"x=1\n")
    row = ident(p.read_bytes()); plan = {"source_identities": {"scripts/a.py": row}}
    manifest = {"scripts/a.py": row}
    assert x._sources(plan, encoded(manifest)) == manifest
    with pytest.raises(ValueError): x._sources(plan, encoded({"scripts/../a.py": row}))
    with pytest.raises((ValueError, FileNotFoundError)): x._sources(plan, encoded({"scripts/b.py": row}))
    p.write_bytes(b"x=2\n")
    with pytest.raises(ValueError): x._sources(plan, encoded(manifest))


def test_rng_reads_only_run_owned_replay_with_native_parser(tmp_path, monkeypatch):
    x = context(tmp_path); x.directory = tmp_path; x.check_deadline = lambda *a: None
    p = tmp_path / "small.slp"; p.write_bytes(b"synthetic replay")
    seen = []
    def read(path, **kwargs):
        seen.append((path, kwargs)); return SimpleNamespace(start=SimpleNamespace(random_seed=42))
    monkeypatch.setattr(c.importlib, "import_module", lambda name: SimpleNamespace(read_slippi=read))
    value = x.read_game_start_rng(p)
    assert value["game_random_seed"] == 42 and value["replay"] == ident(p.read_bytes())
    assert seen == [(str(p), {"skip_frames": True})]


def test_wrong_platform_blocks_before_rom_or_child(tmp_path, monkeypatch):
    import modal_panel_live_canary as tape
    x = context(tmp_path)
    plan = {"expires_at": 200}; x.owner = owner(); x.owner["plan_sha256"] = tape.identity(plan)["sha256"]
    monkeypatch.setattr(c.os, "getppid", lambda: 111)
    monkeypatch.setattr(c.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **k: pytest.fail("started child"))
    monkeypatch.setattr(c.importlib, "import_module", lambda *a, **k: pytest.fail("imported native"))
    with pytest.raises(RuntimeError, match="fresh Linux"): x.verify_and_prepare(plan, tmp_path)


def test_diagnostic_process_inherits_session_and_is_cleaned_on_failure(tmp_path, monkeypatch):
    x = context(tmp_path); x.directory = tmp_path; x.expiry = 200; x.monotonic_end = 150
    x.check_deadline = lambda *a: None
    seen = []
    class Child:
        pid = 123
        returncode = None
        def poll(self): return self.returncode
        def terminate(self): seen.append("terminate")
        def kill(self): seen.append("kill")
        def wait(self, timeout): self.returncode = -15; seen.append("wait")
    def start(argv, **kwargs):
        seen.append((argv, kwargs))
        assert "start_new_session" not in kwargs and "process_group" not in kwargs
        return Child()
    monkeypatch.setattr(c.subprocess, "Popen", start)
    x._same_session = lambda pid: (_ for _ in ()).throw(RuntimeError("wrong group"))
    with pytest.raises(RuntimeError, match="wrong group"): x._capture(["/approved/probe"])
    assert seen[-2:] == ["terminate", "wait"]


def test_child_group_and_session_both_required(tmp_path, monkeypatch):
    x = context(tmp_path)
    monkeypatch.setattr(c.os, "getpid", lambda: 100)
    monkeypatch.setattr(c.os, "getpgid", lambda pid: 100)
    monkeypatch.setattr(c.os, "getsid", lambda pid: 100)
    x._same_session(123)
    monkeypatch.setattr(c.os, "getsid", lambda pid: 101)
    with pytest.raises(RuntimeError): x._same_session(123)


@pytest.mark.parametrize("seed", [None, True, -1, 2**32])
def test_missing_or_invalid_replay_rng_fails(tmp_path, monkeypatch, seed):
    x = context(tmp_path); x.directory = tmp_path; x.check_deadline = lambda *a: None
    path = tmp_path / "owned.slp"; path.write_bytes(b"test")
    monkeypatch.setattr(c.importlib, "import_module", lambda name: SimpleNamespace(
        read_slippi=lambda *a, **k: SimpleNamespace(start=SimpleNamespace(random_seed=seed))))
    with pytest.raises(ValueError): x.read_game_start_rng(path)


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    x = context(tmp_path)
    versions = {f"package{i}": "1.0" for i in range(94)}
    body = encoded({"artifacts": [{"distribution": n, "version": v} for n, v in versions.items()]})
    pins = b"fixture pins\n"
    monkeypatch.setattr(c, "LOCK_SHA", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(c, "PINS_SHA", hashlib.sha256(pins).hexdigest())
    distributions = [SimpleNamespace(metadata={"Name": n}, version=v) for n, v in versions.items()]
    monkeypatch.setattr(c.importlib.metadata, "distributions", lambda: distributions)
    extension = tmp_path / "_ubjson.so"; extension.write_bytes(b"synthetic compiled extension")
    modules = {"ubjson": SimpleNamespace(EXTENSION_ENABLED=True), "_ubjson": SimpleNamespace(__file__=str(extension))}
    monkeypatch.setattr(c.importlib, "import_module", lambda name: modules[name])
    os_packages = {name: "test-version" for name in c.RUNTIME_PACKAGES}
    commands = []
    def capture(argv):
        commands.append(argv)
        if argv[0] == sys.executable: return "No broken requirements found."
        return "\n".join(f"{n}\t{v}" for n, v in os_packages.items())
    x._capture = capture
    evidence = {"dependency_lock": body, "dependency_pins": pins, "source_manifest": b"{}",
                "installed_runtime": encoded({"schema": "e011.modal-live-installed-runtime.v1", "versions": versions, "pip_check_passed": True}),
                "os_packages": encoded(os_packages),
                "ubjson_extension": encoded({"version": "0.16.1", "identity": ident(extension.read_bytes())})}
    return x, evidence, distributions, modules, commands


def test_exact_runtime_gate_checks_pip_os_and_compiled_extension(runtime_fixture):
    x, evidence, _, _, commands = runtime_fixture
    result = x._runtime(evidence)
    assert len(result["versions"]) == 94 and result["pip_check_passed"] is True
    assert commands[0] == [sys.executable, "-m", "pip", "check"]
    assert commands[1][-len(c.RUNTIME_PACKAGES):] == list(c.RUNTIME_PACKAGES)


@pytest.mark.parametrize("change", ["missing", "duplicate", "version", "extension", "saved-pip", "pins", "os"])
def test_runtime_gate_rejects_changed_installation(runtime_fixture, change):
    x, evidence, distributions, modules, _ = runtime_fixture
    if change == "missing": distributions.pop()
    if change == "duplicate": distributions.append(distributions[0])
    if change == "version": distributions[0].version = "2.0"
    if change == "extension": modules["ubjson"].EXTENSION_ENABLED = False
    if change == "saved-pip": evidence["installed_runtime"] = evidence["installed_runtime"].replace(b"true", b"false")
    if change == "pins": evidence["dependency_pins"] += b"x"
    if change == "os": evidence["os_packages"] = b"{}"
    with pytest.raises(ValueError): x._runtime(evidence)


def parent_result():
    owner_receipt = {"status": "owned-linux-tape-context-ready", "parent_terminal_cleanup_verified": False,
                     "process": {"pid": 222}, "parent_declaration": {"expires_at": 200}}
    row = {"pid": 222, "unreaped_leader_group_guard": True, "terminated_owned_process_group": True,
           "exit_code": 0, "started_unix": 100, "ended_unix": 201, "log_sha256": "c" * 64}
    return owner_receipt, row


@pytest.mark.parametrize("change", [lambda r: r.update(pid=223), lambda r: r.update(exit_code=1),
    lambda r: r.update(terminated_owned_process_group=False), lambda r: r.update(unreaped_leader_group_guard=False),
    lambda r: r.update(error="timeout"), lambda r: r.update(ended_unix=206),
    lambda r: r.update(ended_unix=float("nan")), lambda r: r.pop("log_sha256")])
def test_parent_cleanup_requires_authoritative_terminal_row(change):
    owner_receipt, row = parent_result()
    assert c.validate_parent_terminal(owner_receipt, row)["parent_terminal_cleanup_verified"] is True
    change(row)
    with pytest.raises(ValueError): c.validate_parent_terminal(owner_receipt, row)


@pytest.fixture
def retained_v4_evidence():
    directory = ROOT / "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/research/modal-compat-v1/policy-pilot-v4"
    names = {role: f"comparison-{index:02d}.json" for index, role in enumerate(c.POLICY_CASES)}
    names.update(policy_plan="plan.json", policy_result="result.json", numeric_diagnostic="NUMERIC_DIAGNOSTIC.json",
                 user_approval="PORTABILITY_LIMITATION_APPROVAL.json")
    if not all((directory / name).is_file() for name in names.values()):
        pytest.skip("retained local v4 evidence is unavailable")
    return {role: (directory / name).read_bytes() for role, name in names.items()}


def limited_admission(evidence):
    bindings = {"policy_plan_sha256": c.LIMITED_EVIDENCE_SHA256["policy_plan"]}
    value = {"schema": c.SCHEMA, "kind": "policy_parity", "decision": c.LIMITED_DECISION,
             "bindings": bindings, "evidence": {role: ident(body) for role, body in evidence.items()}}
    return value, bindings


def test_retained_v4_requires_explicit_limited_route_and_preserves_failures(retained_v4_evidence):
    evidence = retained_v4_evidence; original = dict(evidence)
    value, bindings = limited_admission(evidence)
    c.validate_admission(value, "policy_parity", bindings)
    with pytest.raises(ValueError, match="numeric and exact-command"):
        c.bind_policy_evidence(evidence, bindings["policy_plan_sha256"])
    result = c.bind_policy_evidence(evidence, bindings["policy_plan_sha256"], decision=c.LIMITED_DECISION)
    assert result["classification"] == c.LIMITED_CLASSIFICATION
    assert result["strict_policy_parity_passed"] is False and result["exact_commands_passed"] is True
    assert result["numeric_failure_case_indices"] == [5, 7, 9] and result["numeric_failure_path_counts"] == [4, 1, 4]
    assert result["atol"] == result["rtol"] == 1e-5
    assert result["future_closed_loop_trajectory_equivalence"] == "unproven"
    assert evidence == original


@pytest.mark.parametrize("role", tuple(c.LIMITED_EVIDENCE_SHA256))
def test_limited_route_requires_every_exact_evidence_body(retained_v4_evidence, role):
    evidence = retained_v4_evidence; evidence[role] += b" "
    value, bindings = limited_admission(evidence)
    with pytest.raises(ValueError): c.validate_admission(value, "policy_parity", bindings)
    with pytest.raises(ValueError):
        c.bind_policy_evidence(evidence, bindings["policy_plan_sha256"], decision=c.LIMITED_DECISION)


@pytest.mark.parametrize("change", ["missing-approval", "missing-diagnostic", "other-plan", "extra-evidence", "other-kind", "unknown-decision"])
def test_limited_route_cannot_generalize_authorization(retained_v4_evidence, change):
    evidence = retained_v4_evidence; value, bindings = limited_admission(evidence)
    kind = "policy_parity"
    if change == "missing-approval": value["evidence"].pop("user_approval")
    if change == "missing-diagnostic": value["evidence"].pop("numeric_diagnostic")
    if change == "other-plan": bindings["policy_plan_sha256"] = "f" * 64
    if change == "extra-evidence": value["evidence"]["other"] = ident(b"{}")
    if change == "other-kind": kind = value["kind"] = "linux_runtime"
    if change == "unknown-decision": value["decision"] = "waive-numerics"
    with pytest.raises(ValueError): c.validate_admission(value, kind, bindings)


def failing_comparison(index):
    value = comparison(); value["passed"] = value["checks"]["numeric_parity"] = value["numeric"]["passed"] = False
    paths = c.LIMITED_FAILURE_PATHS[index]
    value["numeric"].update(mismatch_count=len(paths), first_mismatches=[
        {"path": path, "reason": "float tolerance exceeded", "max_abs": 1.1e-5} for path in paths])
    return value


@pytest.mark.parametrize("index", (5, 7, 9))
@pytest.mark.parametrize("change", ["commands", "inputs", "context", "reset", "native", "runtime", "atol", "rtol",
                                    "count", "path", "reason", "passed", "numeric-passed", "bool-type"])
def test_limited_failure_signature_retains_every_other_gate(index, change):
    value = failing_comparison(index); c.limited_comparison_checked(value, index)
    gates = {"commands": "exact_commands", "inputs": "identical_input_bindings", "context": "frame_context_equal",
             "reset": "reset_barrier_counts_equal", "native": "native_checks_match", "runtime": "runtime_contract_equal"}
    if change in gates: value["checks"][gates[change]] = False
    if change in ("atol", "rtol"): value["numeric"][change] = 1e-4
    if change == "count": value["numeric"]["mismatch_count"] += 1
    if change == "path": value["numeric"]["first_mismatches"][0]["path"] += "changed"
    if change == "reason": value["numeric"]["first_mismatches"][0]["reason"] = "dtype mismatch"
    if change == "passed": value["passed"] = True
    if change == "numeric-passed": value["numeric"]["passed"] = True
    if change == "bool-type": value["checks"]["exact_commands"] = 1
    with pytest.raises(ValueError): c.limited_comparison_checked(value, index)


def test_previously_passing_case_stays_strict_in_limited_route():
    for index in (0, 1, 2, 3, 4, 6, 8):
        c.limited_comparison_checked(comparison(), index)
        with pytest.raises(ValueError): c.limited_comparison_checked(failing_comparison(5), index)


def test_saved_limited_admission_propagates_truthful_qualification(tmp_path, monkeypatch, retained_v4_evidence):
    evidence = retained_v4_evidence; x = context(tmp_path)
    x.policy_plan_sha256 = c.LIMITED_EVIDENCE_SHA256["policy_plan"]
    plan = {"game_image": {"sha256": "c" * 64}, "runtime_image": {"image_id": "im-fake"}, "required_admissions": {}}
    bindings = {"policy_plan_sha256": x.policy_plan_sha256, "package_archive_sha256": c.host.ARCHIVE[1],
                "game_image_sha256": "c" * 64, "runtime_image": plan["runtime_image"]}
    for kind in c.KINDS:
        value = admission(kind, bindings); x.evidence_paths[kind] = {}
        if kind == "policy_parity":
            value.update(decision=c.LIMITED_DECISION, evidence={role: ident(body) for role, body in evidence.items()})
        for role in value["evidence"]:
            path = tmp_path / (kind + role + ".json")
            body = evidence[role] if kind == "policy_parity" else b"{}"
            path.write_bytes(body); value["evidence"][role] = ident(body); x.evidence_paths[kind][role] = path
        path, row = bound(tmp_path, kind + ".json", value)
        x.admission_paths[kind] = path; plan["required_admissions"][kind] = row
    monkeypatch.setattr(c.importlib, "import_module", lambda *a: pytest.fail("admission imported a runtime"))
    assert set(x._admissions(plan)) == set(c.KINDS)
    assert x.policy_qualification["classification"] == c.LIMITED_CLASSIFICATION
    assert x.policy_qualification["strict_policy_parity_passed"] is False
