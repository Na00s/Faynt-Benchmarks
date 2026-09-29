"""Local-only D0 binding and native-admission checks; no emulator is started."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
d = importlib.import_module("faynt_d0_durable_game")


@pytest.fixture(scope="module")
def panel():
    if not (ROOT / d.PANEL).is_file():
        pytest.skip("fresh local D0 manifest is unavailable")
    return d._panel(ROOT)[0]


@pytest.mark.parametrize("profile", ["75m", "10m"])
@pytest.mark.parametrize("index", [0, 1])
def test_fresh_binding_is_one_game_and_preserves_proven_globals(panel, profile, index):
    before = {name: getattr(d.proven, name) for name in ("ROOT", "REMOTE", "PANEL", "SOURCE_PATHS", "owner")}
    games = [game for game in panel["games"] if game["profile"] == profile][:2]
    labels = [game["label"] for game in games]
    bound = d.bind(labels, [label + "-a1" for label in labels], index)
    assert bound.fixed_scope()["benchmark_games"] == 1
    assert bound.fixed_scope()["selected_game_indices"] == [index]
    assert bound.fixed_scope()["checkpoint_step"] == {"10m": 632, "75m": 980}[profile]
    assert bound.fixed_scope()["trained_delay"] == 0
    assert bound.limits()["nonpreemptible"] is False
    assert Path("/opt/d0-root") == bound.REMOTE
    assert bound.PLAN == d.PLAN
    assert set(d.proven.runtime.DATA_ASSETS) <= set(bound.SOURCE_PATHS)
    assert "src/melee_policy/integration/faynt_d0_checkpoints.py" in bound.SOURCE_PATHS
    assert bound.allowed_output(f"project/artifacts/integration/frisson_ai/{labels[index]}-a1/summary.json")
    assert not bound.allowed_output("project/artifacts/integration/frisson_ai/old-run/summary.json")
    assert before == {name: getattr(d.proven, name) for name in before}
    assert d.run_durable.__code__ is d.durable.run_durable.__code__
    assert d.run_durable.__globals__["VOLUME"] != d.durable.VOLUME


def test_import_keeps_native_package_namespace_fresh():
    code = (
        "import sys;sys.path.insert(0,'scripts');import faynt_d0_durable_game;"
        "assert not any(n=='melee_policy' or n.startswith('melee_policy.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_reference_views_bind_old_and_new_sources_independently(tmp_path):
    old = tmp_path / "old.py"
    new = tmp_path / "current.py"
    old.write_text("old reference bytes\n")
    new.write_text("new gameplay bytes\n")

    def row(path, remote):
        return {"local": str(path), "remote": remote, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    old_remote = "/opt/runtime-root/src/melee_policy/integration/frisson_policy.py"
    new_remote = "/opt/d0-root/src/melee_policy/integration/frisson_policy.py"
    original = row(new, old_remote)
    original.update(bytes=old.stat().st_size, sha256=hashlib.sha256(old.read_bytes()).hexdigest())
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"uploads": [original]}))
    runtime_row = row(runtime, str(d.proven.runtime.PLAN))
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"uploads": [original, runtime_row]}))
    plan = {"uploads": [row(old, old_remote), row(new, new_remote), runtime_row,
                        row(live, str(d.proven.live.PLAN))]}
    view = d._reference_views(plan, remote=False)
    assert view["wire"].digest(Path(old_remote)) == hashlib.sha256(old.read_bytes()).hexdigest()
    assert view["wire"].digest(new) == hashlib.sha256(old.read_bytes()).hexdigest()
    assert view["wire"].digest(Path(new_remote)) == hashlib.sha256(new.read_bytes()).hexdigest()
    assert d.proven.wire.digest(new) == hashlib.sha256(new.read_bytes()).hexdigest()
    changed = copy.deepcopy(plan)
    changed["uploads"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reference input was replaced"):
        d._reference_views(changed, remote=False)


@pytest.mark.parametrize("profile", ["75m", "10m"])
def test_real_checkpoint_native_admission_without_gameplay(panel, profile, monkeypatch):
    from melee_policy.integration import frisson_match as fm
    from melee_policy.integration import frisson_policy as policy
    from melee_policy.integration import frisson_slippi_match as match
    from melee_policy.integration.faynt_d0_checkpoints import CHECKPOINTS, FORMAT

    record = CHECKPOINTS[profile]
    path = ROOT / record["relative_path"]
    if not path.is_file():
        pytest.skip("staged new checkpoint is unavailable")
    identity = policy.inspect_frisson_checkpoint(path)
    game = next(game for game in panel["games"] if game["profile"] == profile)
    for module, name in (
        (match, "FINAL_CHECKPOINT_SPECS"), (match, "_selected_checkpoint_contract"),
        (fm, "POST_RL_CHECKPOINTS"), (fm, "_INSPECTED_CHECKPOINT_PARAMETER_COUNTS"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    old_registry = copy.deepcopy(policy.POST_RL_CHECKPOINTS)
    calls = []

    def inspect_only(selected, label):
        calls.append((selected, label))
        accepted = match._final_checkpoint_contract(identity, path)
        assert accepted["format"] == FORMAT
        assert accepted["step"] == record["step"]
        assert accepted["display_name"] == f"Faynt {profile.upper()} RL D0 step-{record['step']}"
        config, root = match._load_config(ROOT / "configs/integration.toml")
        assert fm._resolve_checkpoint(config["frisson_ai"], root, path) == path
        wrong = {**identity, "trained_delay": 18}
        with pytest.raises(ValueError, match="trained_delay"):
            match._final_checkpoint_contract(wrong, path)
        return {"checked": True}

    runtime = types.SimpleNamespace(run_game=inspect_only)
    result = d._run_native_d0(runtime, game, game["label"] + "-a1", ROOT, ROOT / d.PANEL)
    assert result == {"checked": True}
    assert len(calls) == 1
    assert runtime.ROOT == ROOT and (ROOT / d.PANEL).parent == runtime.OUTPUT
    assert old_registry == policy.POST_RL_CHECKPOINTS
