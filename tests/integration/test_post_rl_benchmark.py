"""P21 export identity, baseline preservation, and SLP-only schedule checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from melee_policy.integration import frisson_policy as policy
from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS, FORMAT, ROOT


@pytest.fixture(scope="module", params=tuple(CHECKPOINTS))
def checkpoint(request):
    row = CHECKPOINTS[request.param]
    path = ROOT / row["relative_path"]
    if not path.is_file():
        pytest.skip("local P21 export unavailable")
    return row, path, policy.inspect_frisson_checkpoint(path)


def test_real_export_uses_rl_step_and_keeps_inherited_counters_separate(checkpoint):
    row, path, identity = checkpoint
    assert identity["format"] == FORMAT
    assert identity["step"] == row["step"]
    assert identity["processed_target_frames"] is None
    assert identity["parameter_count"] == row["parameter_count"]
    assert identity["sha256"] == row["sha256"]
    assert identity["training"]["wandb_run_id"] == row["wandb_run_id"]
    assert identity["post_rl"]["initialization_training_state"]["optimizer_steps"] != row["step"]
    assert identity["post_rl"]["native_context_mode"] == "prefix"
    assert identity["deployed_actor"]["actor"]["context_mode"] == "ring"
    from melee_policy.integration.frisson_match import _selected_checkpoint_contract
    from melee_policy.integration.frisson_cpu_match import _validate_final_cpu_checkpoint
    from melee_policy.integration.frisson_slippi_match import _final_checkpoint_contract
    assert _selected_checkpoint_contract(identity)["sha256"] == row["sha256"]
    assert _validate_final_cpu_checkpoint(identity, ROOT)["step"] == row["step"]
    assert _final_checkpoint_contract(identity, path)["validation_nll"] is None
    from melee_policy.integration.frisson_match import _resolve_checkpoint
    from melee_policy.integration.match_runtime import _load_config
    config, root = _load_config(ROOT / "configs/integration.toml")
    assert _resolve_checkpoint(config["frisson_ai"], root, path) == path


@pytest.mark.parametrize("field,value", [("step", 195248), ("sha256", "0" * 64), ("processed_target_frames", 8337096704)])
def test_changed_rl_identity_rejected_by_all_boundaries(checkpoint, field, value):
    row, path, identity = checkpoint
    changed = copy.deepcopy(identity)
    changed[field] = value
    from melee_policy.integration.frisson_match import _selected_checkpoint_contract
    from melee_policy.integration.frisson_cpu_match import _validate_final_cpu_checkpoint
    from melee_policy.integration.frisson_slippi_match import _final_checkpoint_contract
    for check in (lambda: _selected_checkpoint_contract(changed),
                  lambda: _validate_final_cpu_checkpoint(changed, ROOT),
                  lambda: _final_checkpoint_contract(changed, path)):
        with pytest.raises(ValueError):
            check()


def test_postrl_schedule_preserves_every_original_matchup_and_disables_video():
    code = '''
import json
from types import SimpleNamespace
from melee_policy.integration import final_benchmark_suite as s
import run_final_winner_benchmark as r
print(json.dumps({"id":s.SUITE_ID,"schema":s.SUITE_SCHEMA_VERSION,
"games":[[x.kind,x.seed,x.frisson_profile,x.player_1_character,x.opponent_character,x.opponent_evaluation_mode,x.mimic_directory,x.head_to_head_swap_ports] for x in s.SCHEDULE],
"labels":[x.label for x in s.SCHEDULE],
"commands":[s.command_for_game(x,"PYTHON") for x in s.SCHEDULE],
"order":[x.frisson_profile if x.kind != "head-to-head" else "dual" for x in r._selected_specs(SimpleNamespace(label=None,group=None))],
"imports":s.import_semantically_valid_v1_artifacts()}))
'''
    snapshots = {}
    for variant in ("posttraining", "postrl"):
        env = dict(os.environ, MELEE_FINAL_BENCHMARK_VARIANT=variant,
                   PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")]))
        snapshots[variant] = json.loads(subprocess.check_output([sys.executable,"-c",code],env=env,cwd=ROOT,text=True))
    old, new = snapshots["posttraining"], snapshots["postrl"]
    assert new["games"] == old["games"]
    assert len(new["games"]) == 329
    assert set(new["labels"]).isdisjoint(old["labels"])
    assert new["order"] == ["75m"] * 162 + ["dual"] * 5 + ["10m"] * 162
    assert new["imports"] == []
    for game, cmd in zip(new["games"], new["commands"]):
        assert "--save-video" not in cmd
        if game[0] not in {"head-to-head", "ancestor"}:
            assert "--no-save-video" in cmd and "--save-slp" in cmd
            assert cmd[cmd.index("--p1-checkpoint") + 1] == str(ROOT / CHECKPOINTS[game[2]]["relative_path"])
