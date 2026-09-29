from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from melee_rl import release_cli


@pytest.mark.parametrize("opponent", ["phillip", "slippi_ai"])
def test_stored_schedules_preserve_games_and_zero_faynt_delay(opponent):
    fd = release_cli.plan(f"video_{opponent}")
    stages = release_cli.plan(f"video_{opponent}_stages")
    assert fd["video"]["clips"] == 16
    assert stages["video"]["clips"] == 18
    assert fd["actor"]["delay_frames"] == stages["actor"]["delay_frames"] == 0
    assert fd["environment"]["dolphin"]["infinite_time"] is False
    assert fd["launches_games"] is False
    assert len(stages["environment"]["dolphin"]["stages"]) == 6


def test_run_rejects_missing_assets_before_importing_runner(tmp_path):
    with pytest.raises(SystemExit) as error:
        release_cli.main(["run", "--config", "video_phillip", "--output", str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", [("--device", "cpu"), ("--checkpoint", "model.pt"),
                                       ("--override", "actor.delay_frames=0")])
def test_plan_rejects_ignored_run_options(flag, value):
    with pytest.raises(SystemExit) as error:
        release_cli.main(["plan", "--config", "video_phillip", flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize("clip_error,expected", [(None, 0), ("controller failure", 1)])
def test_run_preserves_overrides_and_reports_game_errors(monkeypatch, tmp_path, clip_error, expected):
    paths = [tmp_path / name for name in ("weights.pt", "emulator", "game.iso")]
    for path in paths:
        path.write_text("test fixture")
    calls = []

    def run(config, **kwargs):
        calls.append((config, kwargs))
        return {"clips": [{"error": clip_error}]}

    monkeypatch.setitem(sys.modules, "melee_rl.release_runner", SimpleNamespace(run_record_video=run))
    output = tmp_path / "out"
    status = release_cli.main([
        "run", "--config", "video_phillip", "--checkpoint", str(paths[0]),
        "--dolphin", str(paths[1]), "--game", str(paths[2]), "--output", str(output),
        "--override", 'phillip.agent="FalcoFD"',
    ])
    assert status == expected
    assert calls[0][1]["render"] is False
    assert 'phillip.agent="FalcoFD"' in calls[0][1]["overrides"]
    assert json.loads((output / "release-run.json").read_text())["clips"][0]["error"] == clip_error


def test_cloud_source_contains_no_acquisition_calls():
    path = Path(__file__).resolve().parents[2] / "melee_rl/modal_app.py"
    tree = ast.parse(path.read_text())
    forbidden = {"urlopen", "urlretrieve", "snapshot_download", "hf_hub_download", "run_commands"}
    calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in calls)
    assert not any(isinstance(n, ast.Name) and n.id in forbidden for n in calls)
