import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_cohort as c

def value():
    return {"schema": c.SCHEMA, "journal": "x", "journal_sha256": "0"*64,
        "max_workers": 16, "pairs": [{"pair_id": "p", "directory": "d", "plan_sha256": "1"*64, "claim_token": "c"}]}

def test_exact_bounded_plan():
    assert c.validate(value()) == value()

@pytest.mark.parametrize("change", ["workers", "empty", "many", "duplicate", "extra", "command"])
def test_bad_plan(change):
    v = value()
    if change == "workers": v["max_workers"] = 65
    if change == "empty": v["pairs"] = []
    if change == "many": v["pairs"] *= 65
    if change == "duplicate": v["pairs"] *= 2
    if change == "extra": v["refresh"] = True
    if change == "command": v["pairs"][0]["command"] = "anything"
    with pytest.raises(ValueError): c.validate(v)

def test_accepts_exact_64_worker_cohort():
    v = value()
    v["max_workers"] = 64
    v["pairs"] = [{"pair_id": str(i), "directory": str(i), "claim_token": str(i),
                   "plan_sha256": "1"*64} for i in range(64)]
    assert c.validate(v) == v
