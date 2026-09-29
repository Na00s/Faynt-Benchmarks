#!/usr/bin/env python3
"""Read-only bounded audit of one returned Linux code package. Never executes it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import tarfile
import time
import zlib

MAX_ARCHIVE = 256 * 1024**2
MAX_ELF = 128 * 1024**2
MAX_TEXT = 8 * 1024**2
MAX_FILES = 100_000
MAX_READ = 1024**2
MAX_TAR = MAX_ARCHIVE + MAX_FILES * 4096 + 10240
SHA = re.compile(r"[0-9a-f]{64}\Z")
RETURNED = re.compile(r"(?:receipt\.json|plan\.json|package-manifest\.json|linux-dolphin-code\.tar\.gz|(?:bootstrap|logs)/[0-9]{3}\.log)\Z")
ELFS = {"dolphin-emu", "libslippi_rust_extensions.so"}


class Budget:
    def __init__(self, seconds: int = 60):
        if type(seconds) is not int or not 1 <= seconds <= 120:
            raise ValueError("audit deadline must be an integer from 1 to 120 seconds")
        self.ends = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.ends:
            raise TimeoutError("package audit monotonic deadline exhausted")


def regular(path: Path, cap: int) -> int:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("audit input must be a canonical regular path")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= cap:
        raise ValueError("audit input exceeds regular file cap")
    return info.st_size


def digest(path: Path, budget: Budget, cap: int) -> tuple[int, str]:
    expected = regular(path, cap)
    total, value = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            budget.check()
            block = stream.read(MAX_READ)
            if not block:
                break
            total += len(block)
            if total > cap:
                raise ValueError("file grew past audit cap")
            value.update(block)
    if total != expected or regular(path, cap) != expected:
        raise ValueError("audit input changed length or ended early")
    return total, value.hexdigest()


def read_json(path: Path, budget: Budget) -> dict | list:
    regular(path, MAX_TEXT)
    budget.check()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    with path.open("rb") as stream:
        body = stream.read(MAX_TEXT + 1)
    if len(body) > MAX_TEXT:
        raise ValueError("JSON input grew past text cap")
    value = json.loads(body, object_pairs_hook=unique)
    budget.check()
    return value


def member_path(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode()) <= 1024 or "\\" in value:
        raise ValueError("invalid package path")
    parts = value.split("/")
    if any(p in {"", ".", ".."} for p in parts) or any(ord(c) < 32 for c in value):
        raise ValueError("unsafe package path")
    if value not in ELFS | {"license.txt", "portable.txt"} and not value.startswith("Sys/"):
        raise ValueError("package path outside native assets")
    return value


class ForwardReader:
    """Bound gzip expansion and metadata allocations, with forward seeks only."""
    def __init__(self, stream, budget: Budget):
        self.stream, self.budget, self.position = stream, budget, 0

    def tell(self):
        return self.position

    def read(self, size):
        self.budget.check()
        if not 0 <= size <= MAX_READ:
            raise ValueError("tar requested oversized metadata/read allocation")
        data = self.stream.read(size)
        self.position += len(data)
        if self.position > MAX_TAR:
            raise ValueError("expanded tar exceeds bounded metadata allowance")
        return data

    def seek(self, offset, whence=0):
        if whence != 0 or offset < self.position or offset > MAX_TAR:
            raise ValueError("tar attempted unsafe seek")
        while self.position < offset:
            if not self.read(min(MAX_READ, offset - self.position)):
                raise ValueError("short tar data during seek")
        return self.position


class SingleGzip:
    """One gzip member, bounded input/output chunks, verified checksum trailer."""
    def __init__(self, raw, budget: Budget):
        self.raw, self.budget = raw, budget
        self.decoder, self.pending, self.finished = zlib.decompressobj(31), b"", False

    def read(self, size):
        if not 0 <= size <= MAX_READ:
            raise ValueError("invalid gzip read size")
        chunks, total = [], 0
        while total < size and not self.finished:
            self.budget.check()
            if not self.pending:
                self.pending = self.raw.read(MAX_READ)
                if not self.pending:
                    raise ValueError("short gzip stream")
            data = self.decoder.decompress(self.pending, size - total)
            self.pending = self.decoder.unconsumed_tail
            chunks.append(data)
            total += len(data)
            if self.decoder.eof:
                if self.decoder.unused_data or self.raw.read(1):
                    raise ValueError("additional gzip member or trailing compressed data")
                self.finished = True
        return b"".join(chunks)


def elf_header(header: bytes, size: int, name: str):
    if len(header) != 64 or header[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("package executable/library lacks complete ELF64 little-endian header")
    kind, machine, version, _entry, phoff, shoff, _flags, ehsize, phsize, phnum, shsize, shnum, shstr = struct.unpack("<HHIQQQIHHHHHH", header[16:64])
    if machine != 62 or version != 1 or ehsize != 64 or kind not in ({3} if name.endswith(".so") else {2, 3}):
        raise ValueError("ELF architecture, version, or executable/library type differs")
    if ((phnum and (phsize != 56 or phoff < 64 or phoff + phsize * phnum > size))
            or (shnum and (shsize != 64 or shoff < 64 or shoff + shsize * shnum > size))
            or (shstr and shstr >= shnum)):
        raise ValueError("ELF header tables exceed member bounds")


def audit_archive(path: Path, manifest: list, budget: Budget) -> dict:
    regular(path, MAX_ARCHIVE)
    if not isinstance(manifest, list) or not 1 <= len(manifest) <= MAX_FILES:
        raise ValueError("package manifest count exceeds limit")
    expected, total = {}, 0
    for row in manifest:
        budget.check()
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid package manifest schema")
        name = member_path(row["path"])
        count, sha = row["bytes"], row["sha256"]
        cap = MAX_ELF if name in ELFS else MAX_ARCHIVE
        if "Binaries/" + name in expected or type(count) is not int or not 0 <= count <= cap or not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError("invalid or duplicate manifest member")
        total += count
        if total > MAX_ARCHIVE:
            raise ValueError("package aggregate exceeds 256 MiB")
        expected["Binaries/" + name] = row
    if not ELFS | {"license.txt", "portable.txt"} <= {r["path"] for r in manifest}:
        raise ValueError("package lacks native binary/library/license/portable marker")
    if not any(r["path"].startswith("Sys/") for r in manifest):
        raise ValueError("package lacks Sys assets")
    if expected["Binaries/portable.txt"]["bytes"] != 0:
        raise ValueError("portable marker must be empty")
    seen = set()
    with path.open("rb") as raw:
        stream = ForwardReader(SingleGzip(raw, budget), budget)
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for member in archive:
                budget.check()
                if (member.name in seen or member.name not in expected or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                        or member.issparse() or member.linkname or member.mode & 0o7000
                        or set(member.pax_headers) - {"mtime", "path", "uname", "gname"}):
                    raise ValueError("extra, duplicate, linked, sparse, or unsafe tar member")
                row = expected[member.name]
                if member.size != row["bytes"]:
                    raise ValueError("tar member length differs from manifest")
                seen.add(member.name)
                body = archive.extractfile(member)
                if body is None:
                    raise ValueError("tar regular member has no body")
                count, value, header = 0, hashlib.sha256(), b""
                with body:
                    while count < member.size:
                        budget.check()
                        block = body.read(min(MAX_READ, member.size - count))
                        if not block:
                            raise ValueError("short tar member body")
                        header = (header + block[:64])[:64]
                        count += len(block)
                        value.update(block)
                if value.hexdigest() != row["sha256"]:
                    raise ValueError("tar member SHA differs from manifest")
                if row["path"] in ELFS:
                    elf_header(header, member.size, row["path"])
                if row["path"] == "dolphin-emu" and not member.mode & 0o111:
                    raise ValueError("Dolphin archive member lacks executable mode")
        # Consume the entire gzip stream, checking its trailer and excluding a
        # hidden second tar archive after the first archive's end marker.
        zero_tail_bytes = 0
        while True:
            block = stream.read(MAX_READ)
            if not block:
                break
            if any(block):
                raise ValueError("nonzero data follows tar end marker")
            zero_tail_bytes += len(block)
        if zero_tail_bytes < 512 or stream.position % 512:
            raise ValueError("tar requires two zero end blocks and block-aligned length")
    if seen != set(expected):
        raise ValueError("archive lacks manifest members")
    return {"files": len(seen), "uncompressed_bytes": total, "expanded_tar_bytes": stream.position,
            "elf64_x86_64": sorted(ELFS), "extracted": False, "executed": False}


def audit(pilot: Path, expected_plan_sha: str, seconds: int = 60) -> dict:
    budget = Budget(seconds)
    if not SHA.fullmatch(expected_plan_sha):
        raise ValueError("expected frozen plan SHA is required")
    pilot = pilot.absolute()
    if pilot.resolve(strict=True) != pilot or not pilot.is_dir():
        raise ValueError("pilot must be a canonical existing directory")
    snapshots = {name: digest(pilot / name, budget, MAX_TEXT) for name in ("plan.json", "result.json", "call.json")}
    if snapshots["plan.json"][1] != expected_plan_sha:
        raise ValueError("outer plan SHA differs")
    plan, result, call = (read_json(pilot / name, budget) for name in ("plan.json", "result.json", "call.json"))
    execution = result.get("execution")
    if (not isinstance(execution, dict) or set(execution) != {"task_id", "execution_id"}
            or not isinstance(execution["task_id"], str) or not re.fullmatch(r"ta-[A-Za-z0-9_-]{1,128}", execution["task_id"])
            or not isinstance(execution["execution_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", execution["execution_id"])):
        raise ValueError("invalid worker execution identity")
    if (result.get("status") != "code-build-passed" or result.get("plan_sha256") != expected_plan_sha
            or result.get("schema") != plan.get("schema") or result.get("execution") != call.get("execution")
            or result.get("gameplay_admitted") is not False or result.get("no_gameplay") is not True):
        raise ValueError("outer build receipt identity or pass gate differs")
    rows = result.get("files")
    if not isinstance(rows, list) or not 4 <= len(rows) <= 100:
        raise ValueError("invalid returned file manifest")
    files, text_total = {}, 0
    for row in rows:
        budget.check()
        name = row.get("name")
        if not isinstance(name, str) or not RETURNED.fullmatch(name) or name in files:
            raise ValueError("unsafe or duplicate returned file")
        count, sha = digest(pilot / "returned" / name, budget, MAX_ARCHIVE if name.endswith(".tar.gz") else MAX_TEXT)
        if sha != row.get("sha256") or type(row.get("original_bytes")) is not int or not isinstance(row.get("original_sha256"), str) or not SHA.fullmatch(row["original_sha256"]):
            raise ValueError("returned file hash or size metadata differs")
        if row.get("truncated_tail") is True:
            if not name.startswith(("logs/", "bootstrap/")) or not count <= 64 * 1024 < row["original_bytes"]:
                raise ValueError("invalid retained log tail")
        elif row.get("truncated_tail") is not False or count != row["original_bytes"] or sha != row["original_sha256"]:
            raise ValueError("returned full-file identity differs")
        if not name.endswith(".tar.gz"):
            text_total += count
            if text_total > MAX_TEXT:
                raise ValueError("returned text exceeds 8 MiB aggregate")
        files[name] = {"bytes": count, "sha256": sha}
    required = {"receipt.json", "plan.json", "package-manifest.json", "linux-dolphin-code.tar.gz"}
    if not required <= set(files):
        raise ValueError("successful return lacks required evidence")
    actual = set()
    for index, path in enumerate((pilot / "returned").rglob("*")):
        budget.check()
        if index >= 202:
            raise ValueError("too many returned filesystem entries")
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("unsafe returned filesystem member")
        if path.is_file():
            actual.add(path.relative_to(pilot / "returned").as_posix())
        if len(actual) > 100:
            raise ValueError("too many returned files")
    if actual != set(files):
        raise ValueError("unregistered or missing returned files")
    helper = read_json(pilot / "returned/receipt.json", budget)
    build = read_json(pilot / "returned/plan.json", budget)
    package = helper.get("package", {})
    source_rows = {r["name"]: r for r in plan["sources"]}
    if (helper.get("status") != "code-build-passed" or helper.get("gameplay_admitted") is not False
            or helper.get("plan_sha256") != files["plan.json"]["sha256"]
            or helper.get("helper_sha256") != source_rows["modal_panel_linux_build.py"]["sha256"]
            or build.get("revision") != plan["builder_constants"]["REVISION"]
            or build.get("patch_sha256") != plan["builder_constants"]["FRAME_PATCH_SHA256"]
            or build.get("build_patch_sha256") != plan["builder_constants"]["BUILD_PATCH_SHA256"]
            or build.get("target") != "dolphin-emu" or build.get("gameplay_admitted") is not False
            or package.get("gameplay_admitted") is not False
            or package.get("archive_sha256") != files["linux-dolphin-code.tar.gz"]["sha256"]
            or package.get("archive_bytes") != files["linux-dolphin-code.tar.gz"]["bytes"]
            or package.get("manifest_sha256") != files["package-manifest.json"]["sha256"]):
        raise ValueError("helper source, plan, or package identity differs")
    manifest = read_json(pilot / "returned/package-manifest.json", budget)
    inspected = audit_archive(pilot / "returned/linux-dolphin-code.tar.gz", manifest, budget)
    by_name = {row["path"]: row for row in manifest}
    for label, name in (("binary", "dolphin-emu"), ("rust_library", "libslippi_rust_extensions.so")):
        if package.get(label + "_sha256") != by_name[name]["sha256"] or package.get(label + "_bytes") != by_name[name]["bytes"]:
            raise ValueError("helper ELF identity differs from archived member")
    if package.get("files") != inspected["files"] or package.get("uncompressed_bytes") != inspected["uncompressed_bytes"]:
        raise ValueError("helper package totals differ")
    for name, identity in snapshots.items():
        if digest(pilot / name, budget, MAX_TEXT) != identity:
            raise ValueError("outer receipt changed during audit")
    for name, identity in files.items():
        if digest(pilot / "returned" / name, budget, MAX_ARCHIVE if name.endswith(".tar.gz") else MAX_TEXT) != (identity["bytes"], identity["sha256"]):
            raise ValueError("returned file changed during audit")
    return {"schema": "frisson.modal-panel-package-audit.v1", "status": "package-audit-passed",
            "outer_plan_sha256": expected_plan_sha, "execution": result["execution"], "returned_files": files,
            "outer_files": {name: {"bytes": item[0], "sha256": item[1]} for name, item in snapshots.items()},
            "archive": inspected, "gameplay_admitted": False,
            "scope": "Byte/provenance and archive safety audit only. Truncated logs verify returned tail bytes; unavailable full-log bytes remain unaudited."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--expected-plan-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=60)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("audit receipt requires a new canonical output path")
    if args.pilot.absolute() / "returned" in output.parents:
        raise ValueError("audit output must be outside returned inputs")
    try:
        receipt = audit(args.pilot, args.expected_plan_sha, args.deadline_seconds)
    except Exception as error:
        receipt = {"schema": "frisson.modal-panel-package-audit.v1", "status": "failed",
                   "error": f"{type(error).__name__}: {error}", "gameplay_admitted": False}
    with output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return 0 if receipt["status"] == "package-audit-passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
