"""Synthetic host binding tests. No real Console, policy, process or game image."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_match_host as h


def plan_hash(plan):
    return hashlib.sha256((json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    calls = []
    native = SimpleNamespace(**{name: (lambda *a, **k: None) for name in h.HOOKS})
    for name in ("_stop_console", "_load_config", "_ControllerPipeLockstep"):
        setattr(native, name, object())
    native._disable_attested_dolphin_stop_hotkey = lambda console: calls.append("hotkey")
    native._LIBMELEE_VERSION_ATTESTATION_LOCK = object()
    native._file_identity = lambda path, root: {"path": str(path), "sha256": "1" * 64, "byte_length": 1}
    match = SimpleNamespace(**{name: getattr(native, name) for name in h.HOOKS})
    for name in h.PROTECTED: setattr(match, name, getattr(native, name, object()))
    dispatch = SimpleNamespace(send_canonical_controller=match.send_canonical_controller)
    image = tmp_path / "synthetic-image"; image.write_bytes(b"fixture")
    ctx = SimpleNamespace(project_root=tmp_path, package_root=tmp_path / "package", package_manifest=tmp_path / "manifest",
        package_audit=tmp_path / "audit", game_image_path=image, owner={"parent_pid": 111},
        native=SimpleNamespace(mimic=native, dispatch=dispatch, console_module=object()),
        policy_qualification={"strict_policy_parity_passed": False, "decision": "authorized-portability-limited-continuation"},
        check_deadline=lambda expiry: calls.append("deadline"),
        verify_inherited_process=lambda process, plan: {"owned": True, "pid": process.pid})
    plan = {"expires_at": 900, "cases": [{"slippi_port": 51441}, {"slippi_port": 51442}]}
    ready = {"status": "owned-linux-tape-context-ready", "plan_sha256": plan_hash(plan), "process": {"pid": 222},
        "parent_declaration": ctx.owner, "policy_qualification": ctx.policy_qualification,
        "display": {"software_renderer_verified": True}, "runtime": {"pip_check_passed": True}}
    runtime = SimpleNamespace(runtime_environment_record=lambda root, path: {
        "dependency_lock_validation": {"environment_lock_gate_passed": True}, "actual_linux_pins": str(path)})
    monkeypatch.setattr(h.platform, "system", lambda: "Linux")
    monkeypatch.setattr(h.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(h.os, "getpid", lambda: 222)
    monkeypatch.setattr(h.os, "getpgid", lambda pid: 222)
    monkeypatch.setattr(h.os, "getsid", lambda pid: 222)
    monkeypatch.setattr(h, "verify_sources", lambda root: {"source.py": {"sha256": "a" * 64, "bytes": 1}})
    monkeypatch.setattr(h, "verify_function", lambda *args: None)
    monkeypatch.setattr(h.context, "read_bound", lambda *a: (b"pins", {"sha256": h.context.PINS_SHA, "bytes": 4}))
    monkeypatch.setattr(h.host, "verify_installed", lambda *a: {"source_revision": h.host.REVISION, "release": h.host.RELEASE})
    class Console:
        _process = None
        def run(self, **kw): calls.append(("run", kw)); self._process = SimpleNamespace(pid=333)
        def connect(self): calls.append("connect"); return True
    def create(*args, **kwargs): calls.append(("create", kwargs)); return Console()
    monkeypatch.setattr(h.host, "create_console", create)
    value = h.MatchHostAdapter(match=match, owned_context=ctx, preparation=ready, execution_plan=plan,
        linux_pins_path=tmp_path / "linux-pins", runtime_identity=runtime)
    value.calls = calls
    return value


def test_five_host_hooks_preserve_every_protected_reference(adapter):
    before = {name: getattr(adapter.match, name) for name in h.PROTECTED}
    original = {name: getattr(adapter.match, name) for name in h.HOOKS}
    row = adapter.install()
    assert row["policy_loop_changes"] is False
    assert row["policy_qualification"]["strict_policy_parity_passed"] is False
    assert all(getattr(adapter.match, n) is value for n, value in before.items())
    assert all(getattr(adapter.match, n) is not value for n, value in original.items())
    adapter.restore()
    assert all(getattr(adapter.match, n) is value for n, value in original.items())


@pytest.mark.parametrize("change", ["platform", "plan", "pid", "parent", "policy", "display", "runtime", "alias", "sender", "pins"])
def test_install_fails_closed(adapter, monkeypatch, change):
    if change == "platform": monkeypatch.setattr(h.platform, "system", lambda: "Darwin")
    if change == "plan": adapter.preparation["plan_sha256"] = "0" * 64
    if change == "pid": adapter.preparation["process"]["pid"] = 333
    if change == "parent": adapter.preparation["parent_declaration"] = {}
    if change == "policy": adapter.preparation["policy_qualification"] = {}
    if change == "display": adapter.preparation["display"]["software_renderer_verified"] = False
    if change == "runtime": adapter.preparation["runtime"]["pip_check_passed"] = False
    if change == "alias": adapter.match._emulator_application_identity = object()
    if change == "sender": adapter.match.send_canonical_controller = object()
    if change == "pins": monkeypatch.setattr(h.context, "read_bound", lambda *a: (b"x", {"sha256": "0" * 64}))
    with pytest.raises((ValueError, RuntimeError)): adapter.install()
    assert not adapter.hooks


def setup_console(adapter):
    adapter.install()
    config = {"emulator": {"slippi_port": 51441, "application": "original Mac control literal"}}
    identity = adapter.match._emulator_application_identity(config, adapter.root)
    options = h.host.native_options(adapter.root / "replays", 51441)
    console = adapter.match._create_attested_dolphin_console(config, adapter.root, identity, **options)
    return config, identity, options, console


def test_linux_identity_native_options_and_owned_launch(adapter):
    config, identity, options, console = setup_console(adapter)
    assert identity["host_contract"] == h.SCHEMA
    assert identity["original_control_emulator"] == config["emulator"]
    assert "matches_frozen_configuration" not in identity
    assert adapter.match._attested_emulator_release(identity) == h.host.RELEASE
    create = next(row for row in adapter.calls if isinstance(row, tuple) and row[0] == "create")
    assert create[1]["console_options"] == options
    assert adapter.match._launch_and_connect_attested_dolphin(console, adapter.ctx.game_image_path) is True
    assert adapter.launch_evidence == {"owned": True, "pid": 333}
    assert adapter.calls.index("hotkey") < next(i for i, row in enumerate(adapter.calls) if isinstance(row, tuple) and row[0] == "run")
    with pytest.raises(RuntimeError): adapter.launch_and_connect(console, adapter.ctx.game_image_path)


@pytest.mark.parametrize("change", ["identity", "port", "root", "duplicate", "image", "existing-process", "loop"])
def test_wrong_host_console_and_launch_rejected(adapter, change):
    config, identity, options, console = setup_console(adapter)
    if change == "identity":
        with pytest.raises(ValueError): adapter.release({**identity, "release": "other"})
    elif change == "port":
        adapter.console = None; options["slippi_port"] = 55555
        with pytest.raises(ValueError): adapter.create_console(config, adapter.root, identity, **options)
    elif change == "root":
        with pytest.raises(ValueError): adapter.application_identity(config, adapter.root.parent)
    elif change == "duplicate":
        with pytest.raises(RuntimeError): adapter.create_console(config, adapter.root, identity, **options)
    elif change == "image":
        with pytest.raises(ValueError): adapter.launch_and_connect(console, adapter.root)
    elif change == "existing-process":
        console._process = object()
        with pytest.raises(RuntimeError): adapter.launch_and_connect(console, adapter.ctx.game_image_path)
    else:
        adapter.match._run_exact_frame = object()
        with pytest.raises(RuntimeError): adapter.launch_and_connect(console, adapter.ctx.game_image_path)
    assert "connect" not in adapter.calls


def test_panel_composition_retains_native_public_panel_wrapper(adapter):
    adapter.install()
    original = adapter.match._runtime_reproducibility_record
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs); result["public_panel"] = {"kept": True}; return result
    adapter.match._runtime_reproducibility_record = wrapper
    result = wrapper(adapter.root, adapter.root / "config.toml", "requirements-e010.lock", ("src/example.py",))
    assert adapter.composed is wrapper
    assert result["public_panel"]["kept"] is True
    assert result["dependency_lock"]["path"] == str(adapter.pins)
    assert result["original_control_dependency_lock"]["path"].endswith("requirements-e010.lock")
    assert result["runtime_environment"]["actual_linux_pins"] == str(adapter.pins)
    assert result["linux_host"]["policy_qualification"]["strict_policy_parity_passed"] is False
    adapter.ctx.check_deadline = lambda expiry: (_ for _ in ()).throw(TimeoutError("expired"))
    adapter.restore()  # Restoration remains available after the execution deadline.


def test_wrong_composition_and_runtime_versions_rejected(adapter):
    adapter.install()
    adapter.match._runtime_reproducibility_record = lambda *a, **k: {}
    with pytest.raises(ValueError): adapter.after_panel_configuration()
    adapter.match._runtime_reproducibility_record = adapter.hooks["_runtime_reproducibility_record"]
    adapter.runtime_identity.runtime_environment_record = lambda *a: {"dependency_lock_validation": {"environment_lock_gate_passed": False}}
    with pytest.raises(ValueError): adapter.reproducibility(adapter.root, adapter.root / "config", "lock", ())


def test_source_and_compiled_function_binding_without_importing_policies():
    identities = h.verify_sources(ROOT)
    assert len(identities) == 7
    relative = "src/melee_policy/integration/frisson_slippi_match.py"
    path = ROOT / relative
    module_code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    code = next(c for c in h.code_tree(module_code) if c.co_name == "_run_exact_frame")
    function = FunctionType(code, {})
    h.verify_function(function, ROOT, relative)
    with pytest.raises(ValueError): h.verify_function(lambda: None, ROOT, relative)


def test_no_default_execution_or_policy_construction_entrypoint():
    tree = ast.parse(Path(h.__file__).read_text())
    assert not any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) for n in tree.body)
    names = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(any(token in name for token in ("Popen", "remote", "FrissonPolicySession", "SlippiAIPolicySession", "run_prepared_plan")) for name in names)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr"]
    assert len(assignments) == 2
