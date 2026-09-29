import ast
import copy
import inspect
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import modal_panel_batch_pilot as batch
import modal_panel_single_pilot as single
from modal_panel_prefix_validation import validator


def bound():
    p=batch.base.policy.bounded_json(batch.base.ROOT/batch.base.PANEL)
    labels=[g["label"] for g in p["games"][154:156]]
    return single.bind(labels,[labels[0]+"-a1",labels[1]+"-a2"])


def test_selected_second_game_preserves_native_functions_and_global_state():
    before=dict(vars(batch.base)); b=bound()
    assert b.fixed_scope()["selected_game_indices"]==[1]
    assert b.fixed_scope()["benchmark_games"]==1
    assert b.ENTRY==batch.base.REMOTE/single.ENTRY
    for name in ["game_child","native_validation","stage_project","assemble_admissions","remote_game"]:
        assert getattr(b,name).__code__ is getattr(batch.base,name).__code__
    assert all(vars(batch.base)[k] is v for k,v in before.items())
    assert (1,) in b.owner.__code__.co_consts
    assert (0,1) not in b.owner.__code__.co_consts
    assert single.SUCCESS in b.owner.__code__.co_consts
    assert frozenset({'failed',single.SUCCESS}) in b.validate_terminal.__code__.co_consts
    assert b.GAME_SECONDS==7200


def test_prefix_validator_keeps_all_original_gate_names():
    b=batch.local_binding(*([g["label"] for g in batch.base.policy.bounded_json(batch.base.ROOT/batch.base.PANEL)["games"][154:156]],
        [g["label"]+"-a1" for g in batch.base.policy.bounded_json(batch.base.ROOT/batch.base.PANEL)["games"][154:156]]))
    prefix=validator(b)
    assert set(prefix.__code__.co_names)==set(b.validate_terminal.__code__.co_names)
    assert 'native_validation' in prefix.__code__.co_names
    assert 'validate_parent_terminal' in prefix.__code__.co_names


def incompatible():
    return None


def test_unknown_orchestration_is_rejected():
    with pytest.raises(ValueError): single.select_function(incompatible,{})
