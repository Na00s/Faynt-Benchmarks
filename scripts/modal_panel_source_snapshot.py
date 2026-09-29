#!/usr/bin/env python3
"""Export three pinned Frisson blobs with authentic, shallow Git provenance.

Other file contents, parent history, remotes and local Git configuration never
enter this snapshot. Original commit and tree objects retain their Git IDs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

REVISION = "2c535c04693dc9a232fc81f244cbd24cb2359771"
FILES = {
    "model.py": "2e61e7b27b1412c4becbd2dd93ee8d8bebad15c14cb7a2babaf954993205854d",
    "controller_codec.py": "4a51a6eb2bd1da48a1a7c852194d914ab6e5072b7d353096f351e89b24252286",
    "tensor_batch.py": "07c39e9e5d5152b9d8d73aa74edf77c690db53f45b6ec03c6d4e64c80f4814ed",
}
SCHEMA = "e011.modal.frisson-source-snapshot.v1"
PROVENANCE_SCOPE = "original commit, all tree objects, exactly three runtime blobs; shallow at original commit"
MAX_BYTES = 2 * 1024 * 1024
MAX_OBJECTS = 512


def git(repository: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *args], input=data, capture_output=True,
        check=True, timeout=30,
    ).stdout


def object_id(kind: str, value: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(value)}\0".encode() + value).hexdigest()


def safe_path(value: str) -> bool:
    p = Path(value)
    return bool(value) and not p.is_absolute() and all(x not in (".", "..", ".git") for x in p.parts) and "\n" not in value and "\r" not in value and "\x00" not in value


def export(repository: Path) -> dict:
    if git(repository, "rev-parse", "--verify", REVISION + "^{commit}").decode().strip() != REVISION:
        raise ValueError("pinned original commit is unavailable")
    prefix = git(repository, "rev-parse", "--show-prefix").decode().strip()
    if prefix and (not prefix.endswith("/") or not safe_path(prefix)):
        raise ValueError("unsafe original source prefix")
    paths_to_names = {prefix + name: name for name in FILES}
    rows = git(repository, "ls-tree", "--full-tree", "-r", "-t", "-z", REVISION).split(b"\0")
    commit = git(repository, "cat-file", "commit", REVISION)
    root_tree = commit.splitlines()[0].decode().removeprefix("tree ")
    wanted = {REVISION: "commit", root_tree: "tree"}
    file_ids = {}
    omitted = []
    for row in rows:
        if not row:
            continue
        metadata, path_bytes = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = path_bytes.decode()
        if not safe_path(path):
            raise ValueError("unexpected source tree path")
        if kind == "tree":
            wanted[oid] = kind
        elif path in paths_to_names:
            if kind != "blob" or mode != "100644":
                raise ValueError("runtime source must be an ordinary Git file")
            file_ids[paths_to_names[path]] = oid
            wanted[oid] = kind
        else:
            omitted.append(path)
    if set(file_ids) != set(FILES) or len(wanted) > MAX_OBJECTS:
        raise ValueError("source snapshot inventory differs")
    objects = []
    for oid, kind in sorted(wanted.items()):
        raw = git(repository, "cat-file", kind, oid)
        if object_id(kind, raw) != oid:
            raise ValueError("Git object identity differs")
        objects.append({"oid": oid, "kind": kind, "data": base64.b64encode(raw).decode()})
    value = {
        "schema": SCHEMA, "revision": REVISION, "files": FILES,
        "file_objects": file_ids, "objects": objects, "source_subdirectory": prefix.rstrip("/"),
        "omitted_file_contents": sorted(omitted),
        "provenance_scope": PROVENANCE_SCOPE,
    }
    validate(value)
    return value


def validate(value: dict) -> dict[str, tuple[str, bytes]]:
    expected = {"schema", "revision", "files", "file_objects", "objects", "source_subdirectory", "omitted_file_contents", "provenance_scope"}
    if type(value) is not dict or set(value) != expected or value.get("provenance_scope") != PROVENANCE_SCOPE:
        raise ValueError("source snapshot schema or provenance scope differs")
    if len(json.dumps(value).encode()) > MAX_BYTES:
        raise ValueError("source snapshot byte limit")
    if value.get("schema") != SCHEMA or value.get("revision") != REVISION or value.get("files") != FILES:
        raise ValueError("source snapshot contract differs")
    file_ids = value.get("file_objects", {})
    if set(file_ids) != set(FILES):
        raise ValueError("runtime file inventory differs")
    subdir = value.get("source_subdirectory")
    if not isinstance(subdir, str) or (subdir and not safe_path(subdir)):
        raise ValueError("unsafe source subdirectory")
    source_paths = {(subdir + "/" if subdir else "") + name: oid for name, oid in file_ids.items()}
    rows = value.get("objects", [])
    if not isinstance(rows, list) or not 4 <= len(rows) <= MAX_OBJECTS:
        raise ValueError("invalid object count")
    objects = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"oid", "kind", "data"}:
            raise ValueError("source object schema differs")
        kind, oid = row["kind"], row["oid"]
        raw = base64.b64decode(row["data"], validate=True)
        if kind not in ("commit", "tree", "blob") or object_id(kind, raw) != oid or oid in objects:
            raise ValueError("invalid or duplicate object identity")
        objects[oid] = (kind, raw)
    if [oid for oid, (kind, _) in objects.items() if kind == "commit"] != [REVISION]:
        raise ValueError("only the pinned original commit is permitted")
    if {oid for oid, (kind, _) in objects.items() if kind == "blob"} != set(file_ids.values()):
        raise ValueError("unapproved file content in snapshot")
    for name, oid in file_ids.items():
        kind, raw = objects[oid]
        if kind != "blob" or hashlib.sha256(raw).hexdigest() != FILES[name]:
            raise ValueError("runtime blob SHA differs")
    commit = objects[REVISION][1]
    first = commit.splitlines()[0].decode()
    if not first.startswith("tree "):
        raise ValueError("original commit has no root tree")
    visited = set()
    paths = {}

    def visit(oid: str, prefix: str, depth: int) -> None:
        if depth > 32 or len(paths) > 10000:
            raise ValueError("tree traversal limit")
        if oid not in objects or objects[oid][0] != "tree":
            raise ValueError("required original tree object is missing")
        visited.add(oid)
        data = objects[oid][1]
        offset = 0
        sibling_names = set()
        while offset < len(data):
            end = data.index(b"\0", offset)
            mode, name = data[offset:end].split(b" ", 1)
            name = name.decode()
            if "/" in name or name in sibling_names or not safe_path(name):
                raise ValueError("unsafe or duplicate tree entry")
            sibling_names.add(name)
            raw_id = data[end + 1:end + 21]
            if len(raw_id) != 20:
                raise ValueError("truncated tree entry")
            child = raw_id.hex()
            path = prefix + name
            offset = end + 21
            if mode == b"40000":
                visit(child, path + "/", depth + 1)
            elif mode in (b"100644", b"100755", b"120000", b"160000"):
                paths[path] = (mode, child)
            else:
                raise ValueError("unsupported tree entry mode")
    visit(first[5:], "", 0)
    if visited != {oid for oid, (kind, _) in objects.items() if kind == "tree"}:
        raise ValueError("unreferenced tree object in upload")
    if any(paths.get(name) != (b"100644", oid) for name, oid in source_paths.items()):
        raise ValueError("runtime paths differ from original tree")
    if sorted(set(paths) - set(source_paths)) != value.get("omitted_file_contents"):
        raise ValueError("omitted source inventory differs")
    return objects


def materialize(value: dict, destination: Path) -> dict:
    objects = validate(value)
    subdir = value["source_subdirectory"]
    source_paths = [(subdir + "/" if subdir else "") + name for name in FILES]
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("snapshot destination must be new")
    destination.mkdir(parents=True)
    git(destination, "init", "--initial-branch=main")
    for oid, (kind, raw) in objects.items():
        p = destination / ".git" / "objects" / oid[:2] / oid[2:]
        p.parent.mkdir(exist_ok=True)
        with p.open("xb") as f:
            f.write(zlib.compress(f"{kind} {len(raw)}\0".encode() + raw))
    with (destination / ".git" / "shallow").open("x") as f:
        f.write(REVISION + "\n")
    git(destination, "update-ref", "refs/heads/main", REVISION)
    git(destination, "read-tree", REVISION)
    names = git(destination, "ls-files", "-z").split(b"\0")
    excluded = []
    for name in names:
        if not name:
            continue
        text = name.decode()
        if not safe_path(text):
            raise ValueError("unsafe Git index path")
        if text not in source_paths:
            excluded.append(name)
    if excluded:
        git(destination, "update-index", "--skip-worktree", "-z", "--stdin", data=b"\0".join(excluded) + b"\0")
    git(destination, "checkout-index", "--", *source_paths)
    status = git(destination, "status", "--short", "--untracked-files=all").decode().strip()
    runtime_revision = git(destination, "log", "-1", "--format=%H", REVISION, "--", *source_paths).decode().strip()
    materialized = [str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file() and ".git" not in p.relative_to(destination).parts]
    checks = {
        "original_commit": git(destination, "rev-parse", "HEAD").decode().strip() == REVISION,
        "native_runtime_log": runtime_revision == REVISION,
        "clean_partial_worktree": not status,
        "three_worktree_files": sorted(materialized) == sorted(source_paths),
        "exact_runtime_bytes": all(hashlib.sha256((destination / subdir / name).read_bytes()).hexdigest() == sha for name, sha in FILES.items()),
        "no_remote": not git(destination, "remote").strip(),
    }
    if not all(checks.values()):
        raise ValueError(f"materialized source check failed: {checks}")
    return {"schema": SCHEMA, "checks": checks, "passed": True, "shallow": True, "object_count": len(objects), "omitted_blob_count": len(excluded), "model_source_directory": str(destination / subdir)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("export"); e.add_argument("--repository", type=Path, required=True); e.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("materialize"); m.add_argument("--snapshot", type=Path, required=True); m.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "export":
        result = export(args.repository)
        with args.output.open("x") as f:
            json.dump(result, f, sort_keys=True, indent=2)
            f.write("\n")
        print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "objects": len(result["objects"])}))
    else:
        if args.snapshot.stat().st_size > MAX_BYTES:
            raise ValueError("snapshot input too large")
        print(json.dumps(materialize(json.loads(args.snapshot.read_text()), args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
