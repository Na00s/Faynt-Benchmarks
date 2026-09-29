import copy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
import modal_panel_game_pilot as proven
import modal_panel_pair_binding as p
import modal_panel_batch_pilot as batch


def inputs():
    panel=proven.policy.bounded_json(ROOT/proven.PANEL)
    games=panel["games"][154:156]
    labels=[g["label"] for g in games]
    return panel,labels,[label+"-a1" for label in labels]


def bound():
    panel,labels,attempts=inputs()
    return p.bind(panel,labels,attempts,project_root=ROOT,entry_relative=batch.ENTRY_RELATIVE)


def test_binding_preserves_every_proven_function_code_and_original_globals():
    before=dict(vars(proven)); b=bound()
    for name in ("game_child","owner","validate_terminal","verify_plan","stage_project","assemble_admissions","native_validation","remote_game"):
        assert getattr(b,name).__code__ is getattr(proven,name).__code__
        assert getattr(b,name).__globals__ is not vars(proven)
    assert all(vars(proven)[k] is v for k,v in before.items())
    assert b.fixed_scope()["characters"]==["LUIGI","LUIGI"]
    assert b.GAME_SECONDS==7200
    assert b.ENTRY==proven.REMOTE/batch.ENTRY_RELATIVE
    assert b.fixed_scope()["benchmark_games"]==2
    assert b.fixed_scope()["save_video"] is False


@pytest.mark.parametrize("mutation",["reversed","duplicate","one","other_pair","seed","port","video","attempt","entry","manifest"])
def test_binding_rejects_pair_or_attempt_drift(mutation):
    panel,labels,attempts=inputs(); entry=batch.ENTRY_RELATIVE
    if mutation=="reversed": labels.reverse(); attempts.reverse()
    elif mutation=="duplicate": labels[1]=labels[0]
    elif mutation=="one": labels.pop()
    elif mutation=="other_pair": labels[1]=panel["games"][156]["label"]
    elif mutation=="seed": panel["games"][154]["seed"]+=1
    elif mutation=="port": panel["games"][154]["frisson_port"]=1
    elif mutation=="video": panel["games"][154]["save_video"]=True
    elif mutation=="attempt": attempts[0]="../../elsewhere"
    elif mutation=="entry": entry="scripts/other.py"
    else: panel["schema"]="other"
    with pytest.raises((ValueError,KeyError)):
        p.bind(panel,labels,attempts,project_root=ROOT,entry_relative=entry)


def test_selected_games_must_be_exact_frozen_values():
    b=bound(); panel,labels,_=inputs()
    games=p.select_pair(panel,labels)
    assert b.selected_games({"games":games})==games
    games[0]["seed"]+=1
    with pytest.raises(ValueError): b.selected_games({"games":games})


def test_checkpoint_binding_matches_both_frozen_native_records():
    b=bound(); panel,_,_=inputs()
    assert b.CHECKPOINTS[panel["frisson_checkpoints"]["75m"]["relative_path"]]["sha256"]==panel["frisson_checkpoints"]["75m"]["sha256"]
    assert b.CHECKPOINTS[".e001-cache/slippi-ai/models/medium-v2"]["sha256"]==panel["releases"]["medium-v2"]["sha256"]


@pytest.mark.parametrize("tail,allowed",[("summary.json",True),("controller_trace.jsonl",True),("replays/Game_1.slp",True),("game/game.slp",True),("video.mp4",False),("../elsewhere.json",False)])
def test_only_this_pairs_artifacts_are_returned(tail,allowed):
    b=bound()
    assert b.allowed_output("project/artifacts/integration/frisson_ai/"+b.LABELS[0]+"/"+tail) is allowed
    assert not b.allowed_output("project/artifacts/integration/frisson_ai/"+proven.LABELS[0]+"/"+tail)


def test_scope_return_is_not_mutable_binding_state():
    b=bound(); value=b.fixed_scope(); value["seed"]=1
    assert b.fixed_scope()["seed"]!=1


def test_actual_modal_function_is_direct_importable_and_single_worker(tmp_path):
    modal=pytest.importorskip("modal")
    b=bound(); app,fn,_=batch.configure_app(modal,tmp_path/"plan.json",{"uploads":[]},b)
    assert fn.get_raw_f().__name__ == batch.remote_pair.__name__
    assert fn.spec.cpu==(4,4) and fn.spec.memory==(16384,16384)
    assert fn.spec.scheduler_placement.nonpreemptible is True
    assert fn.spec.secrets==[] and fn.spec.volumes=={}
