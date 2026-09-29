#!/usr/bin/env python3
"""Bounded authentic source history for the unchanged full-game config.

Contains the selected commit's history through the runtime-producing commit,
their tree objects, and exactly three runtime file bodies. Export is local;
materialization creates one new isolated sparse repository. No cloud calls.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import modal_panel_source_snapshot as base

SCHEMA = "e011.modal.frisson-match-source.v1"
SELECTED = "268031e7bddebb4e8c7a40026cd0b95f824d9d0d"
RUNTIME = base.REVISION
SUBDIRECTORY = "melee-policy-frisson-ai"
MAX_BYTES = 8 * 1024**2
MAX_OBJECTS = 2048
COMMIT_COUNT = 11
SCOPE = "eleven authentic commits and their trees; exactly three runtime blobs; shallow at runtime commit"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def headers(body):
    lines = body.split(b"\n\n", 1)[0].splitlines()
    roots = [line[5:].decode() for line in lines if line.startswith(b"tree ")]
    parents = [line[7:].decode() for line in lines if line.startswith(b"parent ")]
    if len(roots) != 1 or len(parents) != 1:
        raise ValueError("pinned single-parent commit history required")
    return roots[0], parents[0]


def export(repository):
    repository = Path(repository).resolve(strict=True)
    paths = {f"{SUBDIRECTORY}/{name}": name for name in base.FILES}
    objects, files, chain = {}, {}, []
    revision = SELECTED
    while True:
        if len(chain) >= COMMIT_COUNT or revision in chain:
            raise ValueError("unexpected source history length")
        body = base.git(repository, "cat-file", "commit", revision)
        if base.object_id("commit", body) != revision:
            raise ValueError("commit bytes differ")
        objects[revision] = ("commit", body); chain.append(revision)
        tree, parent = headers(body)
        wanted = {tree: "tree"}; current_files = {}
        for row in base.git(repository, "ls-tree", "--full-tree", "-r", "-t", "-z", revision).split(b"\0"):
            if not row: continue
            meta, raw_path = row.split(b"\t", 1)
            mode, kind, oid = meta.decode().split(); name = raw_path.decode()
            if not base.safe_path(name): raise ValueError("unsafe original tree path")
            if kind == "tree": wanted[oid] = kind
            elif name in paths:
                if mode != "100644" or kind != "blob": raise ValueError("regular runtime blob required")
                current_files[paths[name]] = oid; wanted[oid] = kind
        if set(current_files) != set(base.FILES) or files and current_files != files:
            raise ValueError("runtime file identities changed within selected history")
        files = current_files
        for oid, kind in wanted.items():
            if oid not in objects: objects[oid] = (kind, base.git(repository, "cat-file", kind, oid))
        if len(objects) > MAX_OBJECTS: raise ValueError("source object bound")
        if revision == RUNTIME: break
        revision = parent
    value = {"schema": SCHEMA, "selected_revision": SELECTED, "runtime_revision": RUNTIME,
             "source_subdirectory": SUBDIRECTORY, "scope": SCOPE, "files": base.FILES,
             "file_objects": files, "commits": chain,
             "objects": [{"oid": oid, "kind": kind, "data": base64.b64encode(body).decode()}
                         for oid, (kind, body) in sorted(objects.items())]}
    validate(value)
    return value


def tree_paths(objects, root, visited):
    paths = {}
    def walk(oid, prefix, depth):
        if depth > 32 or len(paths) > 50000 or oid not in objects or objects[oid][0] != "tree":
            raise ValueError("bounded complete original tree required")
        visited.add(oid); body = objects[oid][1]; offset = 0; names = set()
        while offset < len(body):
            end = body.index(b"\0", offset)
            mode, raw_name = body[offset:end].split(b" ", 1)
            name = raw_name.decode(); raw_oid = body[end + 1:end + 21]
            if len(raw_oid) != 20 or "/" in name or name in names or not base.safe_path(name):
                raise ValueError("invalid original tree entry")
            names.add(name); child = raw_oid.hex(); path = prefix + name; offset = end + 21
            if mode == b"40000": walk(child, path + "/", depth + 1)
            elif mode in (b"100644", b"100755", b"120000", b"160000"): paths[path] = (mode, child)
            else: raise ValueError("unsupported original tree mode")
    walk(root, "", 0)
    return paths


def validate(value):
    keys = {"schema", "selected_revision", "runtime_revision", "source_subdirectory", "scope",
            "files", "file_objects", "commits", "objects"}
    if (not isinstance(value, dict) or set(value) != keys or value["schema"] != SCHEMA
            or value["selected_revision"] != SELECTED or value["runtime_revision"] != RUNTIME
            or value["source_subdirectory"] != SUBDIRECTORY or value["scope"] != SCOPE
            or value["files"] != base.FILES or len(encoded(value)) > MAX_BYTES
            or not isinstance(value["commits"], list) or len(value["commits"]) != COMMIT_COUNT
            or len(set(value["commits"])) != COMMIT_COUNT
            or not isinstance(value["objects"], list) or not 4 <= len(value["objects"]) <= MAX_OBJECTS
            or set(value["file_objects"]) != set(base.FILES)):
        raise ValueError("exact full-match source snapshot required")
    objects = {}
    for row in value["objects"]:
        if not isinstance(row, dict) or set(row) != {"oid", "kind", "data"}:
            raise ValueError("source object schema differs")
        body = base64.b64decode(row["data"], validate=True); oid, kind = row["oid"], row["kind"]
        if kind not in {"commit", "tree", "blob"} or base.object_id(kind, body) != oid or oid in objects:
            raise ValueError("source object identity differs")
        objects[oid] = (kind, body)
    if {oid for oid, (kind, _) in objects.items() if kind == "blob"} != set(value["file_objects"].values()):
        raise ValueError("only three approved runtime file bodies allowed")
    for name, oid in value["file_objects"].items():
        kind, body = objects[oid]
        if kind != "blob" or hashlib.sha256(body).hexdigest() != base.FILES[name]:
            raise ValueError("runtime blob bytes differ")
    expected_paths = {f"{SUBDIRECTORY}/{name}": (b"100644", oid) for name, oid in value["file_objects"].items()}
    seen, trees, revision = [], set(), SELECTED
    for index in range(COMMIT_COUNT):
        if revision not in objects or objects[revision][0] != "commit" or revision in seen:
            raise ValueError("authentic selected-to-runtime history incomplete")
        seen.append(revision); tree, parent = headers(objects[revision][1])
        paths = tree_paths(objects, tree, trees)
        if any(paths.get(name) != pair for name, pair in expected_paths.items()):
            raise ValueError("runtime paths differ in retained history")
        if index == COMMIT_COUNT - 1 and revision != RUNTIME:
            raise ValueError("runtime shallow boundary differs")
        revision = parent
    if (seen != value["commits"] or set(seen) != {oid for oid, (kind, _) in objects.items() if kind == "commit"}
            or trees != {oid for oid, (kind, _) in objects.items() if kind == "tree"}):
        raise ValueError("unreferenced source object or history inventory differs")
    return objects


def materialize(value, destination):
    objects = validate(value); destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or destination.resolve(strict=False) != destination:
        raise ValueError("new canonical sparse repository required")
    destination.mkdir(parents=True)
    base.git(destination, "init", "--initial-branch=main")
    for oid, (kind, body) in objects.items():
        path = destination / ".git/objects" / oid[:2] / oid[2:]; path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream: stream.write(zlib.compress(f"{kind} {len(body)}\0".encode() + body))
    with (destination / ".git/shallow").open("x") as stream: stream.write(RUNTIME + "\n")
    base.git(destination, "update-ref", "refs/heads/main", SELECTED)
    base.git(destination, "read-tree", SELECTED)
    paths = [f"{SUBDIRECTORY}/{name}" for name in base.FILES]
    excluded = [name for name in base.git(destination, "ls-files", "-z").split(b"\0")
                if name and name.decode() not in paths]
    if excluded: base.git(destination, "update-index", "--skip-worktree", "-z", "--stdin", data=b"\0".join(excluded) + b"\0")
    base.git(destination, "checkout-index", "--", *paths)
    selected = base.git(destination, "rev-parse", "HEAD").decode().strip()
    runtime = base.git(destination, "log", "-1", "--format=%H", SELECTED, "--", *paths).decode().strip()
    if selected != SELECTED or runtime != RUNTIME or base.git(destination, "status", "--short", "--untracked-files=all").strip():
        raise ValueError("materialized native Git source contract differs")
    for name, sha in base.FILES.items():
        if hashlib.sha256((destination / SUBDIRECTORY / name).read_bytes()).hexdigest() != sha:
            raise ValueError("materialized runtime file differs")
    return {"schema": SCHEMA, "selected_revision": selected, "runtime_revision": runtime,
            "source_subdirectory": SUBDIRECTORY, "objects": len(objects), "commits": COMMIT_COUNT,
            "materialized_file_contents": list(base.FILES), "scope": SCOPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    exp = modes.add_parser("export"); exp.add_argument("--repository", type=Path, required=True); exp.add_argument("--output", type=Path, required=True)
    mat = modes.add_parser("materialize"); mat.add_argument("--snapshot", type=Path, required=True); mat.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "export":
        value = export(args.repository); body = encoded(value)
        with args.output.open("xb") as stream: stream.write(body)
        print(json.dumps({"path": str(args.output), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}))
    else:
        with args.snapshot.open("rb") as stream: body = stream.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES: raise ValueError("snapshot byte bound")
        print(json.dumps(materialize(json.loads(body), args.destination)))


if __name__ == "__main__": main()
