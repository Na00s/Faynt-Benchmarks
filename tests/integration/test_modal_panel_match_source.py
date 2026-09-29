"""Local sparse-Git source packaging. No models, emulator or network."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_panel_match_source as p


@pytest.fixture
def repository(tmp_path, monkeypatch):
    source = tmp_path / "source"; source.mkdir()
    def git(*args):
        return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid", "-C", str(source), *args],
            check=True, capture_output=True, text=True).stdout.strip()
    git("init", "--initial-branch=main")
    git("commit", "--allow-empty", "-m", "older omitted history")
    sub = source / p.SUBDIRECTORY; sub.mkdir()
    files = {}
    for name in p.base.FILES:
        body = f"# {name} fixture\n".encode(); (sub / name).write_bytes(body)
        files[name] = hashlib.sha256(body).hexdigest()
    (source / "unrelated.txt").write_text("PRIVATE OTHER BODY MUST BE OMITTED\n")
    git("add", "."); git("commit", "-m", "runtime")
    runtime = git("rev-parse", "HEAD")
    for i in range(2):
        (source / "unrelated.txt").write_text(f"PRIVATE OTHER BODY {i}\n")
        git("add", "."); git("commit", "-m", f"unrelated {i}")
    monkeypatch.setattr(p, "SELECTED", git("rev-parse", "HEAD"))
    monkeypatch.setattr(p, "RUNTIME", runtime)
    monkeypatch.setattr(p, "COMMIT_COUNT", 3)
    monkeypatch.setattr(p.base, "FILES", files)
    return source


def test_sparse_selected_and_runtime_history_roundtrip(repository, tmp_path):
    value = p.export(repository)
    assert len(value["commits"]) == 3 and value["commits"][0] == p.SELECTED and value["commits"][-1] == p.RUNTIME
    objects = p.validate(value)
    assert len([1 for kind, _ in objects.values() if kind == "blob"]) == 3
    assert not any(b"PRIVATE OTHER BODY" in body for kind, body in objects.values())
    destination = tmp_path / "materialized"
    receipt = p.materialize(value, destination)
    assert receipt["selected_revision"] == p.SELECTED and receipt["runtime_revision"] == p.RUNTIME
    assert (destination / ".git/shallow").read_text().strip() == p.RUNTIME
    assert not (destination / "unrelated.txt").exists()
    assert sorted(x.name for x in (destination / p.SUBDIRECTORY).iterdir()) == sorted(p.base.FILES)
    with pytest.raises(ValueError): p.materialize(value, destination)


@pytest.mark.parametrize("change", ["selected", "runtime", "scope", "extra_blob", "bad_object", "missing_commit", "order", "file_map", "bound"])
def test_unapproved_snapshot_changes_fail(repository, change, monkeypatch):
    value = copy.deepcopy(p.export(repository))
    if change == "selected": value["selected_revision"] = "a" * 40
    if change == "runtime": value["runtime_revision"] = "b" * 40
    if change == "scope": value["scope"] = "unbounded"
    if change == "extra_blob":
        body = b"PRIVATE EXTRA BODY"; oid = p.base.object_id("blob", body)
        value["objects"].append({"oid": oid, "kind": "blob", "data": p.base.base64.b64encode(body).decode()})
    if change == "bad_object": value["objects"][0]["data"] = "Y2hhbmdlZA=="
    if change == "missing_commit": value["objects"] = [r for r in value["objects"] if r["oid"] != p.RUNTIME]
    if change == "order": value["commits"].reverse()
    if change == "file_map": value["file_objects"]["extra.py"] = "c" * 40
    if change == "bound": monkeypatch.setattr(p, "MAX_BYTES", 64)
    with pytest.raises(ValueError): p.validate(value)


def test_changed_runtime_in_selected_history_rejected(repository, monkeypatch):
    path = repository / p.SUBDIRECTORY / "model.py"; path.write_text("# changed runtime\n")
    p.base.git(repository, "add", ".")
    p.base.git(repository, "-c", "commit.gpgsign=false", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "changed")
    monkeypatch.setattr(p, "SELECTED", p.base.git(repository, "rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(p, "COMMIT_COUNT", 4)
    with pytest.raises(ValueError): p.export(repository)
