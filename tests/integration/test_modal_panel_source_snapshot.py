"""Authentic Git provenance with an explicit three-file content allowlist."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("snapshot_under_test", Path(__file__).resolve().parents[2] / "scripts/modal_panel_source_snapshot.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(params=["", "project"])
def original(tmp_path, monkeypatch, request):
    repo = tmp_path / "original"
    repo.mkdir()
    m.git(repo, "init", "--initial-branch=main")
    m.git(repo, "config", "user.email", "fixture@example.invalid")
    m.git(repo, "config", "user.name", "Fixture")
    contents = {"model.py": b"model=1\n", "controller_codec.py": b"codec=2\n", "tensor_batch.py": b"tensor=3\n"}
    source = repo / request.param
    source.mkdir(exist_ok=True)
    for name, raw in contents.items():
        (source / name).write_bytes(raw)
    (repo / "other").mkdir()
    (repo / "other/private.txt").write_text("OMITTED_PRIVATE_CONTENT_7753")
    (repo / "readme.txt").write_text("OTHER_OMITTED_CONTENT_1184")
    m.git(repo, "add", ".")
    m.git(repo, "commit", "-m", "Original fixture")
    revision = m.git(repo, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(m, "REVISION", revision)
    monkeypatch.setattr(m, "FILES", {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()})
    return source


def test_original_commit_and_only_approved_contents(original, tmp_path):
    before = m.git(original, "status", "--short", "--untracked-files=all")
    value = m.export(original)
    decoded = b"\n".join(base64.b64decode(x["data"]) for x in value["objects"])
    assert b"OMITTED_PRIVATE_CONTENT" not in decoded
    assert b"OTHER_OMITTED_CONTENT" not in decoded
    assert sum(x["kind"] == "blob" for x in value["objects"]) == 3
    report = m.materialize(value, tmp_path / "partial")
    assert report["passed"] and all(report["checks"].values())
    assert report["omitted_blob_count"] == 2
    assert m.git(original, "status", "--short", "--untracked-files=all") == before


def test_existing_destination_never_overwritten(original, tmp_path):
    target = tmp_path / "existing"; target.mkdir()
    with pytest.raises(FileExistsError):
        m.materialize(m.export(original), target)


def test_symlink_destination_rejected(original, tmp_path):
    target = tmp_path / "link"; target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        m.materialize(m.export(original), target)


@pytest.mark.parametrize("mutation", ["commit", "extra_blob", "duplicate", "blob_bytes", "omitted", "extra_tree", "missing_tree"])
def test_rejects_changed_or_expanded_snapshot(original, mutation):
    v = copy.deepcopy(m.export(original))
    if mutation == "commit":
        v["revision"] = "0" * 40
    elif mutation == "extra_blob":
        raw = b"private extra data"
        v["objects"].append({"kind": "blob", "oid": m.object_id("blob", raw), "data": base64.b64encode(raw).decode()})
    elif mutation == "duplicate":
        v["objects"].append(v["objects"][0])
    elif mutation == "blob_bytes":
        next(x for x in v["objects"] if x["kind"] == "blob")["data"] = base64.b64encode(b"changed").decode()
    elif mutation == "omitted":
        v["omitted_file_contents"] = []
    elif mutation == "extra_tree":
        raw = b""
        v["objects"].append({"kind": "tree", "oid": m.object_id("tree", raw), "data": ""})
    elif mutation == "missing_tree":
        v["objects"] = [x for x in v["objects"] if x["kind"] != "tree"]
    with pytest.raises(ValueError):
        m.validate(v)


def test_size_cap_before_decode(original, monkeypatch):
    value = m.export(original)
    monkeypatch.setattr(m, "MAX_BYTES", 20)
    with pytest.raises(ValueError, match="byte limit"):
        m.validate(value)


@pytest.mark.parametrize("mutation", ["top", "object", "scope"])
def test_unapproved_metadata_rejected(original, mutation):
    value = m.export(original)
    if mutation == "top":
        value["unapproved_payload"] = "synthetic"
    elif mutation == "object":
        value["objects"][0]["unapproved_payload"] = "synthetic"
    else:
        value["provenance_scope"] += " expanded"
    with pytest.raises(ValueError, match="schema|scope"):
        m.validate(value)


@pytest.mark.parametrize("path", ["../x", "/x", "x\ny", "x\ry", ".git/config", "x/../y", ""])
def test_unsafe_path(path):
    assert not m.safe_path(path)
