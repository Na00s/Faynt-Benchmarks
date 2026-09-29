"""Synthetic packages and Console classes only; no real archive or emulator."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import sys
import tarfile
import threading
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_linux_host as h

ELF = b"\x7fELF\x02\x01\x01" + b"\0" * 9 + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def write_json(path, obj):
    body = (json.dumps(obj, sort_keys=True) + "\n").encode()
    path.write_bytes(body)
    return len(body), digest(body)


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    bodies = {"dolphin-emu": ELF, "libslippi_rust_extensions.so": ELF,
              "portable.txt": b"", "license.txt": b"test license", "Sys/GameSettings/GALE01.ini": b"test settings"}
    manifest = [{"path": p, "bytes": len(b), "sha256": digest(b)} for p, b in bodies.items()]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, body in bodies.items():
            member = tarfile.TarInfo("Binaries/" + name)
            member.size = len(body); member.mode = 0o755 if name == "dolphin-emu" else 0o644
            tar.addfile(member, io.BytesIO(body))
    archive = root / "code.tar.gz"; archive.write_bytes(gzip.compress(stream.getvalue()))
    manifest_path = root / "manifest.json"; manifest_id = write_json(manifest_path, manifest)
    archive_id = (archive.stat().st_size, digest(archive.read_bytes()))
    record = dict(schema="frisson.modal-panel-package-audit.v1", status="package-audit-passed", gameplay_admitted=False,
                  outer_plan_sha256=h.OUTER_PLAN_SHA, returned_files={
                      "linux-dolphin-code.tar.gz": {"bytes": archive_id[0], "sha256": archive_id[1]},
                      "package-manifest.json": {"bytes": manifest_id[0], "sha256": manifest_id[1]}})
    audit_path = root / "audit.json"
    for key, value in dict(ARCHIVE=archive_id, MANIFEST=manifest_id, AUDIT_RECEIPT=write_json(audit_path, record),
                           MEMBER_COUNT=len(bodies), MEMBER_BYTES=sum(map(len, bodies.values())), EXPANDED_TAR_BYTES=len(stream.getvalue()),
                           EXECUTABLE=(len(ELF), digest(ELF)), RUST_LIBRARY=(len(ELF), digest(ELF))).items():
        monkeypatch.setattr(h, key, value)
    return types.SimpleNamespace(root=root, archive=archive, manifest=manifest_path, audit=audit_path,
                                 destination=root / "installed", bodies=bodies)


def materialize(p):
    """Provide a synthetic installed fixture without invoking an installer."""
    p.destination.mkdir()
    for name, body in p.bodies.items():
        path = p.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        path.chmod(0o755 if name == "dolphin-emu" else 0o644)
    return h.verify_installed(p.destination, p.manifest, p.audit)


def test_verify_exact_provided_regular_assets_only(package, monkeypatch):
    monkeypatch.setattr(tarfile.TarFile, "extract", lambda *a, **k: pytest.fail("unsafe tar extraction API"))
    monkeypatch.setattr(tarfile.TarFile, "extractall", lambda *a, **k: pytest.fail("unsafe tar extraction API"))
    identity = materialize(package)
    assert identity["status"] == "linux-package-bytes-verified" and identity["target_platform"] == "linux_x86_64"
    assert not identity["executed"] and not identity["gameplay_admitted"]
    assert "application_tree_sha256" not in identity and "MacOS" not in identity["executable"]["path"]
    for name, body in package.bodies.items():
        p = package.destination / name
        assert p.read_bytes() == body
        assert bool(stat.S_IMODE(p.stat().st_mode) & 0o111) == (name == "dolphin-emu")
    assert h.verify_installed(package.destination, package.manifest, package.audit)["files"] == 5


@pytest.mark.parametrize("which", ["manifest", "audit"])
def test_changed_input_rejected_before_materialization(package, which):
    path = getattr(package, which); path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError): materialize(package)




@pytest.mark.parametrize("change", ["elf", "library", "sys", "extra", "directory", "symlink", "hardlink", "mode", "missing"])
def test_installed_tampering_rejected(package, change):
    materialize(package)
    root = package.destination
    if change == "elf": (root / "dolphin-emu").write_bytes(b"changed")
    if change == "library": (root / "libslippi_rust_extensions.so").write_bytes(b"changed")
    if change == "sys": (root / "Sys/GameSettings/GALE01.ini").write_bytes(b"changed")
    if change == "extra": (root / "Sys/extra").write_bytes(b"extra")
    if change == "directory": (root / "Sys/extra").mkdir()
    if change == "symlink":
        (root / "license.txt").unlink(); (root / "license.txt").symlink_to(package.manifest)
    if change == "hardlink": (package.root / "outside-link").hardlink_to(root / "license.txt")
    if change == "mode": (root / "license.txt").chmod(0o700)
    if change == "missing": (root / "portable.txt").unlink()
    with pytest.raises(ValueError): h.verify_installed(root, package.manifest, package.audit)




def test_deadline_and_filesystem_bounds(package, monkeypatch):
    materialize(package)
    monkeypatch.setattr(h, "MAX_FILESYSTEM_ENTRIES", 1)
    with pytest.raises(ValueError, match="entry bound"):
        h.verify_installed(package.destination, package.manifest, package.audit)


def test_native_options_equal_frozen_function_without_framework_imports(tmp_path):
    tree = ast.parse((ROOT / "src/melee_policy/integration/frisson_match.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_console_options")
    namespace = {"Path": Path, "Any": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native-options-only", "exec"), namespace)
    assert h.native_options(tmp_path, 51441) == namespace["_console_options"](tmp_path, 51441)


CONSOLE_FIXTURE = '''
from pathlib import Path
import tempfile
class DolphinBuild:
    NETPLAY = object()
class DolphinVersion:
    def __init__(self, mainline, version, build):
        self.mainline, self.version, self.build = mainline, version, build
def get_dolphin_version(path):
    raise AssertionError("native version subprocess must be intercepted")
class Console:
    def __init__(self, path, **options):
        self.exe_path = path
        self.dolphin_version = get_dolphin_version(path if MODE != "wrong-request" else OTHER_PATH)
        CALLS.append((path, options))
        if MODE == "raise":
            raise ValueError("fixture constructor failed")
        self.temp_dir = tempfile.mkdtemp(prefix="libmelee_", dir=PROFILE_PARENT)
        self.dolphin_home_path = str(Path(self.temp_dir) / "User")
        Path(self.dolphin_home_path).mkdir()
        if MODE == "wrong-result":
            self.exe_path = OTHER_PATH
'''


@pytest.fixture
def console(package, monkeypatch):
    materialize(package)
    source = package.root / "console.py"; source.write_text(CONSOLE_FIXTURE)
    module = types.ModuleType("melee.console"); module.__file__ = str(source)
    module.MODE = "good"; module.OTHER_PATH = str(package.destination / "license.txt")
    module.PROFILE_PARENT = str(package.root); module.CALLS = []
    exec(compile(CONSOLE_FIXTURE, str(source), "exec"), module.__dict__)
    monkeypatch.setattr(h, "CONSOLE_SOURCE_SHA", digest(source.read_bytes()))
    monkeypatch.setattr(h.platform, "system", lambda: "Linux")
    monkeypatch.setattr(h.platform, "machine", lambda: "x86_64")
    replay = package.root / "replays"; replay.mkdir()
    kwargs = dict(replay_directory=replay, slippi_port=51441, console_module=module,
                  version_lock=threading.Lock(), console_options=h.native_options(replay, 51441))
    return package, module, kwargs


def construct(value):
    package, module, kwargs = value
    return h.create_console(package.destination, package.manifest, package.audit, **kwargs)


def test_direct_elf_exact_options_and_hook_restoration(console):
    package, module, kwargs = console
    original = module.get_dolphin_version
    actual = construct(console)
    assert module.get_dolphin_version is original
    assert module.CALLS == [(str(package.destination / "dolphin-emu"), kwargs["console_options"])]
    assert actual.dolphin_version.mainline is False
    assert actual.dolphin_version.version == "3.6.4"
    record = dict(actual._modal_panel_linux_host)
    assert record["temporary_version_hook_restored"] and not record["gui_version_process_used"]
    assert record["verification_host"] == {"system": "Linux", "machine": "x86_64"}
    assert record["native_emulation_speed_default"] == 1.0


@pytest.mark.parametrize("mode", ["raise", "wrong-request", "wrong-result"])
def test_hook_restored_on_constructor_or_path_failure(console, mode):
    _, module, _ = console
    original = module.get_dolphin_version; module.MODE = mode
    with pytest.raises(ValueError): construct(console)
    assert module.get_dolphin_version is original


@pytest.mark.parametrize("key,value", [("disable_audio", True), ("online_delay", 2), ("enable_ffw", True),
                                      ("tmp_home_directory", False), ("copy_home_directory", True),
                                      ("gfx_backend", "Null"), ("path", "/other"), ("emulation_speed", 2.0),
                                      ("disable_audio", 0), ("polling_timeout", 1)])
def test_scientific_options_cannot_change(console, key, value):
    _, module, kwargs = console
    kwargs["console_options"][key] = value
    with pytest.raises(ValueError, match="scientific"): construct(console)
    assert not module.CALLS


def test_changed_original_hook_is_rejected(console):
    _, module, _ = console
    changed = lambda path: None
    module.get_dolphin_version = changed
    with pytest.raises(ValueError, match="already changed"): construct(console)
    assert module.get_dolphin_version is changed


def test_shared_hook_lock_cannot_block_forever(console):
    _, module, kwargs = console
    original = module.get_dolphin_version
    with kwargs["version_lock"]:
        with pytest.raises(RuntimeError, match="already in use"):
            construct(console)
    assert module.get_dolphin_version is original


def test_true_host_and_separate_replay_required(console, monkeypatch):
    package, _, kwargs = console
    monkeypatch.setattr(h.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="actually be Linux"): construct(console)
    monkeypatch.setattr(h.platform, "system", lambda: "Linux")
    kwargs["replay_directory"] = package.destination
    with pytest.raises(ValueError, match="separate replay"): construct(console)
