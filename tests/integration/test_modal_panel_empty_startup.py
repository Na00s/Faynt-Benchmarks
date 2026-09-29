import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import modal_panel_queue as q

def summary():
    return {"error": "RuntimeError: Slippi port belongs to another process", "result": "failed",
        "execution": {"termination": "not-started", "processed_policy_frames": 0,
            "first_game_frame": None, "last_game_frame": None, "first_game_context": None,
            "natural_game_end": False, "game_end_observed": False, "winner": None,
            "inference_counts": {"frisson-ai": 0, "slippi-ai": 0},
            "dispatch_counts": {"frisson-ai": 0, "slippi-ai": 0},
            "controller_transport": {"installed": False}},
        "gate": {"checks": {"entered_gameplay": False}}}

@pytest.mark.parametrize("change", [None, "frames", "inference", "dispatch", "first", "last", "context",
    "end", "winner", "trace", "replay", "missing", "error", "entered", "transport"])
def test_only_proven_empty_startup_is_retryable(tmp_path, monkeypatch, change):
    monkeypatch.setattr(q.job, "ARTIFACTS", tmp_path)
    d = tmp_path / "attempt"; d.mkdir(); s = summary(); e = s["execution"]
    if change == "frames": e["processed_policy_frames"] = 1
    if change == "inference": e["inference_counts"]["frisson-ai"] = 1
    if change == "dispatch": e["dispatch_counts"]["slippi-ai"] = 1
    if change == "first": e["first_game_frame"] = -123
    if change == "last": e["last_game_frame"] = -123
    if change == "context": e["first_game_context"] = {}
    if change == "end": e["natural_game_end"] = True
    if change == "winner": e["winner"] = "frisson-ai"
    if change == "missing": del s["execution"]
    if change == "error": s["error"] = "different failure"
    if change == "entered": s["gate"]["checks"]["entered_gameplay"] = True
    if change == "transport": e["controller_transport"]["installed"] = True
    q.new_file(d / "summary.json", s)
    (d / "controller_trace.jsonl").write_bytes(b"gameplay" if change == "trace" else b"")
    if change == "replay": (d / "game.slp").write_bytes(b"replay")
    assert q.proven_empty_socket_startup("attempt") is (change is None)
