from pathlib import Path
from types import SimpleNamespace
import hashlib
import shutil
import time

import pytest
import prepared_runtime
import modal_panel_game_pilot as game
from melee_policy.integration import frisson_policy


def test_image_requires_digest(monkeypatch):
    monkeypatch.delenv('FAYNT_BENCHMARK_IMAGE', raising=False)
    with pytest.raises(ValueError):
        prepared_runtime.image_reference()
    monkeypatch.setenv('FAYNT_BENCHMARK_IMAGE', 'registry.example/faynt@sha256:'+'a'*64)
    assert prepared_runtime.image_reference().endswith('a'*64)


def test_stage_project_preserves_authenticated_faynt_bundle(tmp_path, monkeypatch):
    release = Path(__file__).resolve().parents[2]
    remote = tmp_path/'remote'; remote.mkdir()
    work = tmp_path/'work'; work.mkdir()
    parent = work/'source'; parent.mkdir()
    project = parent/'melee-policy'
    supplied_slippi = tmp_path/'slippi'; supplied_slippi.mkdir()
    (supplied_slippi/'local-source.txt').write_text('caller supplied')
    uploads=[]
    for relative in game.FAYNT_BUNDLE_PATHS:
        assert relative in game.SOURCE_PATHS
        source = release/relative
        target = remote/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        uploads.append({'remote':str(target),'bytes':target.stat().st_size,
                        'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    for name,value in {'WORK':work,'PARENT':parent,'PROJECT':project,'REMOTE':remote}.items():
        monkeypatch.setattr(game,name,value)
    monkeypatch.setattr(game.policy,'bounded_json',lambda _: {})
    monkeypatch.setattr(game.match_source,'materialize',lambda *_: {'source':'authenticated fixture'})
    monkeypatch.setattr(game.guard,'Runner',lambda *_: SimpleNamespace(commands=[]))
    monkeypatch.setattr(game.prepared_runtime,'slippi_source',lambda _: supplied_slippi)
    game.stage_project({'uploads':uploads},time.time()+60)
    authenticated=frisson_policy._bundled_source_manifest(project,
        frisson_policy.FRISSON_SOURCE_REVISION,game.FAYNT_BUNDLE_PREFIX)
    assert authenticated is not None
    (project/game.FAYNT_BUNDLE_PREFIX/'model.py').write_text('tampered')
    with pytest.raises(RuntimeError,match='hash mismatch'):
        frisson_policy._bundled_source_manifest(project,
            frisson_policy.FRISSON_SOURCE_REVISION,game.FAYNT_BUNDLE_PREFIX)
