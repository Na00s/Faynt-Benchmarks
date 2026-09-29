"""Small local archive fixtures. No extraction, emulator, network, or cloud."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import struct
import zlib

import pytest

SPEC = importlib.util.spec_from_file_location("package_audit_tested", Path(__file__).resolve().parents[2] / "scripts/modal_panel_package_audit.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
ELF = b"\x7fELF\x02\x01\x01" + b"\0" * 9 + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def bodies():
    return {"dolphin-emu": ELF, "libslippi_rust_extensions.so": ELF,
            "license.txt": b"fixture license", "portable.txt": b"", "Sys/GameSettings/GALE01.ini": b"fixture"}


def archive(tmp_path, entries=None, mutate=None):
    entries = bodies() if entries is None else entries
    manifest = [{"path": name, "bytes": len(body), "sha256": sha(body)} for name, body in entries.items()]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, body in entries.items():
            info = tarfile.TarInfo("Binaries/" + name)
            info.size, info.mode = len(body), 0o755 if name == "dolphin-emu" else 0o644
            if mutate:
                mutate(info)
            tar.addfile(info, io.BytesIO(body) if info.isfile() else None)
    path = tmp_path / "code.tar.gz"
    path.write_bytes(gzip.compress(stream.getvalue()))
    return path, manifest


def test_streams_every_member_without_extraction(tmp_path, monkeypatch):
    path, manifest = archive(tmp_path)
    monkeypatch.setattr(tarfile.TarFile, "extract", lambda *x, **kw: pytest.fail("extraction forbidden"))
    monkeypatch.setattr(tarfile.TarFile, "extractall", lambda *x, **kw: pytest.fail("extraction forbidden"))
    result = a.audit_archive(path, manifest, a.Budget())
    assert result["files"] == 5 and not result["extracted"] and not result["executed"]
    assert result["uncompressed_bytes"] == sum(row["bytes"] for row in manifest)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "Sys/../escape", "Sys//x", "Sys/./x", "Sys\\x", "other", "Sys/line\nx"])
def test_unsafe_manifest_paths_rejected(tmp_path, name):
    path, manifest = archive(tmp_path)
    manifest[4]["path"] = name
    with pytest.raises(ValueError):
        a.audit_archive(path, manifest, a.Budget())


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_nonregular_archive_member_rejected(tmp_path, kind):
    def mutate(info):
        if info.name.endswith("license.txt"):
            info.type, info.size, info.linkname = kind, 0, "target"
    path, manifest = archive(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        a.audit_archive(path, manifest, a.Budget())


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "hash", "size", "extra", "elf", "mode", "portable"])
def test_package_integrity_rejections(tmp_path, mutation):
    entries = bodies()
    if mutation == "elf": entries["libslippi_rust_extensions.so"] = b"bad ELF"
    if mutation == "portable": entries["portable.txt"] = b"bad"
    def mutate(info):
        if mutation == "extra" and info.name.endswith("license.txt"): info.name = "Binaries/Sys/extra"
        if mutation == "mode" and info.name.endswith("dolphin-emu"): info.mode = 0o644
    path, manifest = archive(tmp_path, entries, mutate)
    if mutation == "duplicate": manifest.append(manifest[0])
    if mutation == "missing": manifest.pop()
    if mutation == "hash": manifest[2]["sha256"] = "0" * 64
    if mutation == "size": manifest[2]["bytes"] += 1
    with pytest.raises(ValueError):
        a.audit_archive(path, manifest, a.Budget())


def test_duplicate_tar_member_rejected(tmp_path):
    path, manifest = archive(tmp_path, mutate=lambda info: setattr(info, "name", "Binaries/dolphin-emu") if info.name.endswith("libslippi_rust_extensions.so") else None)
    with pytest.raises(ValueError, match="duplicate"):
        a.audit_archive(path, manifest, a.Budget())


@pytest.mark.parametrize("mutation", ["truncated", "crc", "extra-archive", "empty-member", "zero-member"])
def test_gzip_trailer_and_hidden_data_rejected(tmp_path, mutation):
    path, manifest = archive(tmp_path)
    data = path.read_bytes()
    if mutation == "truncated": data = data[:-5]
    if mutation == "crc": data = data[:-8] + bytes([data[-8] ^ 1]) + data[-7:]
    if mutation == "extra-archive": data += gzip.compress(b"extra hidden payload")
    if mutation == "empty-member": data += gzip.compress(b"")
    if mutation == "zero-member": data += gzip.compress(b"\0" * 512)
    path.write_bytes(data)
    with pytest.raises((ValueError, EOFError, zlib.error)):
        a.audit_archive(path, manifest, a.Budget())


def test_bounds_deadline_and_forward_short_reads(tmp_path, monkeypatch):
    path, manifest = archive(tmp_path)
    monkeypatch.setattr(a, "MAX_ELF", len(ELF) - 1)
    with pytest.raises(ValueError): a.audit_archive(path, manifest, a.Budget())
    reader = a.ForwardReader(io.BytesIO(b"x"), a.Budget())
    with pytest.raises(ValueError, match="short"): reader.seek(4)
    with pytest.raises(ValueError, match="unsafe seek"): reader.seek(0)
    with pytest.raises(ValueError, match="allocation"): reader.read(a.MAX_READ + 1)
    budget = a.Budget()
    monkeypatch.setattr(a.time, "monotonic", lambda: budget.ends)
    with pytest.raises(TimeoutError): budget.check()


@pytest.mark.parametrize("limit", ["MAX_FILES", "MAX_ARCHIVE", "MAX_TAR"])
def test_manifest_and_expansion_caps(tmp_path, monkeypatch, limit):
    path, manifest = archive(tmp_path)
    monkeypatch.setattr(a, limit, 1)
    with pytest.raises(ValueError): a.audit_archive(path, manifest, a.Budget())


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha(path.read_bytes())


@pytest.fixture
def pilot(tmp_path):
    tmp_path = tmp_path.resolve()
    returned = tmp_path / "returned"
    returned.mkdir()
    package_path, manifest = archive(returned)
    package_path.rename(returned / "linux-dolphin-code.tar.gz")
    package_path = returned / "linux-dolphin-code.tar.gz"
    manifest_sha = put(returned / "package-manifest.json", manifest)
    constants = {"REVISION": "a" * 40, "FRAME_PATCH_SHA256": "b" * 64, "BUILD_PATCH_SHA256": "c" * 64}
    plan = {"schema": "frisson.modal-panel-build-pilot.v1", "sources": [{"name": "modal_panel_linux_build.py", "sha256": "d" * 64}], "builder_constants": constants}
    plan_sha = put(tmp_path / "plan.json", plan)
    build = {"revision": constants["REVISION"], "patch_sha256": constants["FRAME_PATCH_SHA256"], "build_patch_sha256": constants["BUILD_PATCH_SHA256"], "target": "dolphin-emu", "gameplay_admitted": False}
    build_sha = put(returned / "plan.json", build)
    package = {"archive_sha256": sha(package_path.read_bytes()), "archive_bytes": package_path.stat().st_size,
               "manifest_sha256": manifest_sha, "gameplay_admitted": False, "binary_sha256": sha(ELF),
               "binary_bytes": len(ELF), "rust_library_sha256": sha(ELF), "rust_library_bytes": len(ELF),
               "files": 5, "uncompressed_bytes": sum(row["bytes"] for row in manifest)}
    helper = {"status": "code-build-passed", "gameplay_admitted": False, "helper_sha256": "d" * 64,
              "plan_sha256": build_sha, "package": package}
    put(returned / "receipt.json", helper)
    execution = {"task_id": "ta-fixture", "execution_id": "a" * 32}
    put(tmp_path / "call.json", {"execution": execution})
    result = {"schema": plan["schema"], "status": "code-build-passed", "plan_sha256": plan_sha,
              "execution": execution, "gameplay_admitted": False, "no_gameplay": True, "files": []}
    for path in sorted(returned.iterdir()):
        value = sha(path.read_bytes())
        result["files"].append({"name": path.name, "sha256": value, "original_bytes": path.stat().st_size,
                                "original_sha256": value, "truncated_tail": False})
    put(tmp_path / "result.json", result)
    return tmp_path, plan_sha


def test_full_receipt_chain_passes_without_writes(pilot):
    path, plan_sha = pilot
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    result = a.audit(path, plan_sha)
    assert result["status"] == "package-audit-passed" and not result["gameplay_admitted"]
    assert before == {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mutation", ["outer-failed", "execution", "file-hash", "missing", "extra", "symlink", "helper-provenance"])
def test_full_receipt_chain_rejections(pilot, mutation):
    path, plan_sha = pilot
    result = json.loads((path / "result.json").read_text())
    if mutation == "outer-failed": result["status"] = "failed"
    if mutation == "execution": result["execution"]["task_id"] = "ta-changed"
    if mutation == "file-hash": result["files"][0]["sha256"] = "f" * 64
    if mutation == "missing": result["files"].pop()
    if mutation == "extra": (path / "returned/unregistered").write_text("extra")
    if mutation == "symlink": (path / "returned/link").symlink_to(path / "plan.json")
    if mutation == "helper-provenance":
        helperpath = path / "returned/receipt.json"
        helper = json.loads(helperpath.read_text())
        helper["helper_sha256"] = "f" * 64
        value = put(helperpath, helper)
        row = next(row for row in result["files"] if row["name"] == "receipt.json")
        row.update(sha256=value, original_sha256=value, original_bytes=helperpath.stat().st_size)
    put(path / "result.json", result)
    with pytest.raises((ValueError, FileNotFoundError)):
        a.audit(path, plan_sha)


def test_json_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"status": "passed", "status": "failed"}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        a.read_json(path.resolve(), a.Budget())


def test_json_growth_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "growing.json"
    path.write_text(json.dumps("a" * 200))
    monkeypatch.setattr(a, "MAX_TEXT", 100)
    monkeypatch.setattr(a, "regular", lambda *args: 10)
    with pytest.raises(ValueError, match="grew"):
        a.read_json(path.resolve(), a.Budget())


@pytest.mark.parametrize("pax", [{"mtime": "123.456"}, {"SCHILY.xattr.user.test": "extra"}])
def test_pax_metadata_policy(tmp_path, pax):
    path, manifest = archive(tmp_path, mutate=lambda info: setattr(info, "pax_headers", pax))
    if "mtime" in pax:
        assert a.audit_archive(path, manifest, a.Budget())["files"] == 5
    else:
        with pytest.raises(ValueError): a.audit_archive(path, manifest, a.Budget())


@pytest.mark.parametrize("mutation", ["short", "version", "type", "tables"])
def test_complete_elf_header_required(mutation):
    header = bytearray(ELF)
    if mutation == "short": header = header[:20]
    if mutation == "version": header[20:24] = b"\0" * 4
    if mutation == "type": header[16:18] = b"\x02\0"
    if mutation == "tables":
        header[32:40] = struct.pack("<Q", 64)
        header[54:58] = struct.pack("<HH", 56, 2)
    with pytest.raises(ValueError): a.elf_header(bytes(header), len(header), "libslippi_rust_extensions.so")


def test_multichunk_archive_and_short_gzip_source(tmp_path):
    entries = bodies()
    entries["Sys/large"] = b"payload" * (a.MAX_READ // 2)
    path, manifest = archive(tmp_path, entries)
    assert a.audit_archive(path, manifest, a.Budget())["files"] == 6
    class Short(io.BytesIO):
        def read(self, count):
            return super().read(min(count, 7))
    reader = a.SingleGzip(Short(gzip.compress(b"abcdef" * 5000)), a.Budget())
    assert reader.read(a.MAX_READ) == b"abcdef" * 5000
    assert reader.read(1) == b""


def test_final_recheck_detects_same_size_mutation(pilot, monkeypatch):
    path, plan_sha = pilot
    original = a.audit_archive
    def mutate_after_archive(*args):
        result = original(*args)
        receipt = path / "returned/receipt.json"
        receipt.write_bytes(receipt.read_bytes().replace(b"code-build-passed", b"code-build-failex"))
        return result
    monkeypatch.setattr(a, "audit_archive", mutate_after_archive)
    with pytest.raises(ValueError, match="changed during audit"):
        a.audit(path, plan_sha)


@pytest.mark.parametrize("end_bytes", [0, 512, 1024, 1025])
def test_tar_requires_complete_aligned_end_marker(tmp_path, end_bytes):
    path, manifest = archive(tmp_path)
    raw = gzip.decompress(path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as source:
        last = list(source)[-1]
        data_end = last.offset_data + ((last.size + 511) // 512) * 512
    path.write_bytes(gzip.compress(raw[:data_end] + b"\0" * end_bytes))
    if end_bytes == 1024:
        assert a.audit_archive(path, manifest, a.Budget())["files"] == 5
    else:
        with pytest.raises(ValueError, match="two zero end blocks"):
            a.audit_archive(path, manifest, a.Budget())
