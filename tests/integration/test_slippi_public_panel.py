from __future__ import annotations

import copy
import dataclasses
import io
import json
import fcntl
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import slippi_panel_audit as audit
import slippi_panel_plan as plan
import slippi_panel_runtime as runtime
import run_slippi_public_panel as job


@pytest.fixture(scope="module")
def panel():
    return runtime.manifest()


def test_frozen_inventory_and_counts(panel):
    assert len(panel["releases"]) == 14
    assert len({r["sha256"] for r in panel["releases"].values()}) == 14
    assert panel["aliases"] == {"medium": "medium-v2"}
    assert Counter(g["profile"] for g in panel["games"]) == {"75m": 1312, "10m": 1312}
    assert Counter(g["block"] for g in panel["games"]) == {
        "supported-mirror": 488, "extended-roster": 1068, "forced-mirror": 1068}
    assert [g["profile"] for g in panel["games"]] == ["75m"] * 1312 + ["10m"] * 1312
    assert len({g["label"] for g in panel["games"]}) == 2624
    assert plan.make_manifest(job.read(audit.OUTPUT / "audit-summary.json")) == panel


def test_every_cell_has_both_ports_same_seed_and_native_settings(panel):
    pairs = defaultdict(list)
    for game in panel["games"]:
        pairs[game["pair_id"]].append(game)
        assert game["save_slp"] is True and game["save_video"] is False
        assert game["player_name"] == panel["releases"][game["release"]]["selected_name"]
        assert game["frisson_port"] + game["slippi_port"] == 3
    assert len(pairs) == 1312
    for pair in pairs.values():
        assert {g["frisson_port"] for g in pair} == {1, 2}
        for key in ("seed", "stage", "frisson_character", "slippi_character", "player_name"):
            assert len({g[key] for g in pair}) == 1


@pytest.mark.parametrize("release", plan.PRIORITY)
def test_full_roster_and_distinct_transfer_semantics(panel, release):
    supported = set(panel["releases"][release]["deployed_characters"])
    games = [g for g in panel["games"] if g["profile"] == "75m" and g["release"] == release]
    for block in ("supported-mirror", "extended-roster", "forced-mirror"):
        rows = [g for g in games if g["block"] == block]
        expected = supported if block == "supported-mirror" else set(plan.ROSTER) - supported
        assert {g["frisson_character"] for g in rows} == expected
        for g in rows:
            if block == "extended-roster":
                assert g["slippi_character"] in supported
            else:
                assert g["frisson_character"] == g["slippi_character"]
                assert g["slippi_character_in_deployed_roster"] == (block == "supported-mirror")
    if release in {"gm-v1", "medium-v1"}:
        assert len(supported) == 4
        assert len(panel["releases"][release]["bc_declared_characters"]) == 26


def test_executable_pickle_global_is_rejected():
    with pytest.raises(Exception, match="(global|permitted|allow|unsupported)"):
        audit.RestrictedUnpickler(io.BytesIO(b"cos\nsystem\n.")).load()


@pytest.mark.parametrize("relative", ("controller_trace.jsonl", "replays/retained.slp"))
def test_missing_summary_with_gameplay_evidence_prevents_resampling(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(job, "ARTIFACTS", tmp_path)
    path = tmp_path / "finished-attempt" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"retained game evidence")
    status, reason = job.classify_failure({"label": "finished-attempt"})
    assert status == "quarantined"
    assert "audit before any resampling" in reason


def test_missing_summary_without_gameplay_keeps_bounded_startup_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ARTIFACTS", tmp_path)
    path = tmp_path / "startup-attempt" / "controller_trace.jsonl"
    path.parent.mkdir(parents=True)
    path.touch()
    assert job.classify_failure({"label": "startup-attempt"})[0] == "pending"


@pytest.mark.parametrize("stage", plan.STAGES)
@pytest.mark.parametrize("port", (1, 2))
def test_physical_stage_port_and_name_contract(monkeypatch, panel, stage, port):
    from melee_policy.integration import frisson_slippi_match as match, frisson_match as fm, game_bundle
    import melee
    # Restore every process-local override at the end of this test.
    for module, attrs in ((match, ("FRISSON_PORT", "SLIPPI_PORT", "MATCH_STAGE", "DEFAULT_PLAYER_NAME", "SlippiAIPolicySession", "_frisson_request", "_trace_row", "_first_context", "_validate_slippi_release_config_contract", "_selected_replay_identity_audit", "_runtime_reproducibility_record")),
                          (fm, ("FRISSON_PORT", "MIMIC_PORT", "MATCH_STAGE")),
                          (melee, ("MenuHelper",)), (game_bundle, ("_player_record",))):
        for attr in attrs:
            monkeypatch.setattr(module, attr, getattr(module, attr))
    record = panel["releases"]["fox_d21_ditto_v4"]
    calls = []
    class Menu:
        def menu_helper_simple(self, *args, **kwargs):
            calls.append((args, kwargs))
    monkeypatch.setattr(melee, "MenuHelper", Menu)
    game = {"frisson_port": port, "slippi_port": 3 - port, "stage": stage, "player_name": "Cody"}
    runtime.configure_game(game, record)
    req = match.FrissonSlippiMatchRequest(player_1_character="YOSHI", player_2_character="FOX", stage=stage,
        player_2_slippi_release="fox_d21_ditto_v4", player_2_name="Cody",
        player_1_checkpoint=plan.ROOT / panel["frisson_checkpoints"]["75m"]["relative_path"])
    req.validate()
    assert match._frisson_request(req).stage == stage
    assert (fm.FRISSON_PORT, fm.MIMIC_PORT) == (port, 3 - port)
    config, root = match._load_config(plan.ROOT / "configs/integration.toml")
    canonical = match._canonicalize_request(root, config, req)
    policy = match._slippi_config(config, root, canonical)
    assert (policy.port, policy.opponent_port, policy.name) == (3 - port, port, "Cody")
    assert policy.console_delay_frames == 0
    assert policy.policy_delay_frames == 21
    players = {port: SimpleNamespace(character=melee.Character.YOSHI, costume=0),
               3 - port: SimpleNamespace(character=melee.Character.FOX, costume=0)}
    state = SimpleNamespace(frame=-123, stage=melee.Stage[stage], players=players)
    assert all(match._first_context(state, req)["checks"].values())
    state.players = dict(reversed(list(players.items())))
    assert all(match._first_context(state, req)["checks"].values())
    melee.MenuHelper().menu_helper_simple(state, None, melee.Character.YOSHI, melee.Stage.FINAL_DESTINATION)
    assert calls[0][0][3] == melee.Stage[stage]
    summary = {"configuration": {"player_1": {"model": "frisson-ai", "character": "YOSHI", "port": port},
                                  "player_2": {"model": "slippi-ai", "character": "FOX", "port": 3 - port}}}
    assert game_bundle._player_record(summary, port)["model"] == "frisson-ai"
    assert game_bundle._player_record(summary, 3 - port)["model"] == "slippi-ai"
    from test_frisson_slippi_match import _Session, _Transport, _command, _player
    completed = set()
    sync = threading.Barrier(2)
    sessions = [_Session("frisson", _command("A"), sync, completed), _Session("slippi", _command("B"), sync, completed)]
    transport = _Transport(completed)
    sent = []
    def send(controller, command, *, flush):
        assert completed == {"frisson", "slippi"}
        assert flush is False
        sent.append(controller)
        return SimpleNamespace(as_dict=lambda: {"port": controller})
    monkeypatch.setattr(match, "send_canonical_controller", send)
    state.players = {port: _player("YOSHI"), 3 - port: _player("FOX")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = match._run_exact_frame(gamestate=state, processed_frames=0, frisson_session=sessions[0],
            slippi_session=sessions[1], executor=pool, controllers={1: 1, 2: 2}, transport=transport)
    assert sent == [port, 3 - port]
    trace = match._trace_row(state, req, result)
    assert trace["slots"][f"p{port}"]["model"] == "frisson-ai"
    assert trace["slots"][f"p{3-port}"]["model"] == "slippi-ai"


@pytest.fixture
def fake_result(tmp_path, monkeypatch, panel):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", tmp_path / "panel")
    monkeypatch.setattr(job, "ARTIFACTS", tmp_path / "games")
    game = copy.deepcopy(panel["games"][0])
    attempt = {"label": "attempt1", "accepted": True}
    directory = job.ARTIFACTS / attempt["label"]
    directory.mkdir(parents=True)
    replay = directory / "game.slp"
    replay.write_bytes(b"synthetic-test-replay")
    fp, sp = game["frisson_port"], game["slippi_port"]
    summary = {"gate": {"decision": "pass", "checks": {"all": True}}, "result": "complete",
        "configuration": {"player_1": {"model": "frisson-ai", "character": game["frisson_character"], "port": fp},
            "player_2": {"model": "slippi-ai", "character": game["slippi_character"], "port": sp},
            "seed": game["seed"], "stage": game["stage"]},
        "execution": {"natural_game_end": True, "processed_policy_frames": 1000, "wall_seconds": 20.0,
            "last_stocks": {"frisson-ai": 2, "slippi-ai": 0}, "winner": "frisson-ai"},
        "artifacts": {"replays": [{"tournament_result_replay": True, "path": str(replay.relative_to(tmp_path)),
            "sha256": job.sha(replay), "byte_length": replay.stat().st_size}]},
        "contract": {"frisson": {"selected_checkpoint": {"sha256": panel["frisson_checkpoints"][game["profile"]]["sha256"]}},
            "slippi_ai": {"checkpoint_sha256": panel["releases"][game["release"]]["sha256"], "player_name": game["player_name"]}},
        "policies": {"p1": {"metadata": {"runtime_contract": {"port": fp, "opponent_port": sp}}},
                     "p2": {"metadata": {"runtime_contract": {"port": sp, "opponent_port": fp}}}}}
    summary["replay_identity_audit"] = {"audit": {
        "rules_passed": True, "failures": [], "checks": {"all": True},
        "replay": {"sha256": job.sha(replay)}, "audit_passed": True, "tournament_result_ready": True,
        "end": {"present": True}, "terminal": {"frame_id": 876, "slots": {str(fp): {"stocks": 2}, str(sp): {"stocks": 0}}},
        "outcome": {"status": "win", "winner_port": fp, "conclusive": True, "game_complete": True, "reason": "stock_out"}}}
    job.write(directory / "summary.json", summary)
    job.write(job.OUTPUT / "manifest.json", panel)
    return game, attempt, summary, directory, replay


def test_acceptance_and_resume_evidence(fake_result):
    game, attempt, summary, directory, replay = fake_result
    result = job.validate_result(game, attempt)
    assert result["win"] and (result["stocks_taken"], result["stocks_conceded"]) == (4, 2)
    state = {"games": {game["label"]: {"status": "complete", "attempts": [attempt], "result": result}}}
    job.revalidate_completed(state, {"games": [game]})
    replay.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="replay identity"):
        job.revalidate_completed(state, {"games": [game]})


@pytest.mark.parametrize("fault", ("port", "seed", "stage", "checkpoint", "name", "perspective", "stocks", "gate", "winner"))
def test_rejects_invalid_game_without_resampling_natural_end(fake_result, fault):
    game, attempt, summary, directory, replay = fake_result
    if fault == "port": summary["configuration"]["player_1"]["port"] = game["slippi_port"]
    if fault == "seed": summary["configuration"]["seed"] += 1
    if fault == "stage": summary["configuration"]["stage"] = "OTHER"
    if fault == "checkpoint": summary["contract"]["slippi_ai"]["checkpoint_sha256"] = "wrong"
    if fault == "name": summary["contract"]["slippi_ai"]["player_name"] = "wrong"
    if fault == "perspective": summary["policies"]["p2"]["metadata"]["runtime_contract"]["port"] = game["frisson_port"]
    if fault == "stocks": summary["execution"]["last_stocks"]["frisson-ai"] = 5
    if fault == "gate": summary["gate"]["checks"]["all"] = False
    if fault == "winner": summary["execution"]["winner"] = None
    job.write(directory / "summary.json", summary)
    with pytest.raises(ValueError): job.validate_result(game, attempt)
    assert job.classify_failure(attempt)[0] == "quarantined"


def test_failure_recovery_classification(fake_result):
    game, attempt, summary, directory, replay = fake_result
    summary["execution"]["natural_game_end"] = False
    summary["error"] = "connection timeout"
    job.write(directory / "summary.json", summary)
    assert job.classify_failure(attempt)[0] == "pending"
    summary["error"] = "checkpoint identity wrong"
    job.write(directory / "summary.json", summary)
    assert job.classify_failure(attempt)[0] == "quarantined"
    (directory / "summary.json").write_text("{")
    assert job.classify_failure(attempt)[0] == "quarantined"


def test_process_identity_requires_exact_script_and_arguments(monkeypatch):
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="/resolved/python -u /scope/job.py --watch"))
    assert job.exact_process(4321, ["/env/python", "-u", "/scope/job.py", "--watch"])
    assert not job.exact_process(4321, ["/env/python", "-u", "/unrelated/job.py", "--watch"])
    assert not job.exact_process(1, ["/env/python", "-u", "/scope/job.py", "--watch"])


def test_deployment_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.setattr(job, "source_binding", lambda: {"script": "original"})
    job.write(tmp_path / "deployment.json", {"source_binding": {"script": "original"}})
    job.write(tmp_path / "canary-run.json", {"passed": True})
    job.write(tmp_path / "tests.json", {"passed": True})
    job.validate_deployment()
    monkeypatch.setattr(job, "source_binding", lambda: {"script": "changed"})
    with pytest.raises(RuntimeError, match="changed"): job.validate_deployment()


def test_crash_reconciles_completed_attempt_without_new_launch(fake_result, monkeypatch):
    game, attempt, summary, directory, replay = fake_result
    panel = job.read(job.OUTPUT / "manifest.json")
    panel["games"] = [game]
    job.write(job.OUTPUT / "manifest.json", panel)
    state = job.fresh_state(panel)
    attempt.pop("accepted")
    attempt["command"] = ["/python", "/test/worker"]
    state["games"][game["label"]].update(status="running", attempts=[attempt])
    job.write(job.OUTPUT / "state.json", state)
    monkeypatch.setattr(job, "validate_deployment", lambda: {})
    monkeypatch.setattr(job, "predecessor_complete", lambda: True)
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate game launched"))
    assert job.supervise() == 0
    saved = job.read(job.OUTPUT / "state.json")
    assert saved["status"] == "complete"
    assert len(saved["games"][game["label"]]["attempts"]) == 1
    assert job.supervise() == 0


def test_retry_cap_prevents_fourth_attempt(fake_result, monkeypatch):
    game, _, _, _, _ = fake_result
    panel = job.read(job.OUTPUT / "manifest.json")
    panel["games"] = [game]
    job.write(job.OUTPUT / "manifest.json", panel)
    state = job.fresh_state(panel)
    state["games"][game["label"]]["attempts"] = [{"reconciled": True} for _ in range(3)]
    job.write(job.OUTPUT / "state.json", state)
    monkeypatch.setattr(job, "validate_deployment", lambda: {})
    monkeypatch.setattr(job, "predecessor_complete", lambda: True)
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: pytest.fail("fourth attempt launched"))
    assert job.supervise() == 0
    assert job.read(job.OUTPUT / "state.json")["status"] == "finished-with-quarantined-games"


def test_inherited_lock_survives_coordinator_handle_close(tmp_path):
    path = tmp_path / "workflow.lock"
    handle = path.open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.readline()"],
        stdin=subprocess.PIPE, pass_fds=(handle.fileno(),))
    handle.close()
    try:
        with path.open("a+") as other:
            with pytest.raises(BlockingIOError): fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.communicate(input=b"done\n", timeout=10)
        with path.open("a+") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


def test_worker_registration_recovers_identity_before_parent_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    attempt = {"label": "test-a1", "command": ["python", "worker"], "started_unix": 1}
    assert job.registered_attempt(attempt) == attempt
    job.write(tmp_path / "workers/test-a1.json", {"label": "test-a1", "command": attempt["command"], "pid": 3456})
    assert job.registered_attempt(attempt)["pid"] == 3456
    job.write(tmp_path / "workers/test-a1.json", {"label": "test-a1", "command": ["different"], "pid": 3456})
    with pytest.raises(RuntimeError, match="registration differs"):
        job.registered_attempt(attempt)


def test_workflow_lock_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    assert job.workflow_lock_held() is False
    with (tmp_path / ".supervisor.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert job.workflow_lock_held() is True
    assert job.workflow_lock_held() is False


def test_orphan_watchdog_progress_and_bounded_exact_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    attempt = {"label": "test-a1", "command": ["python", "worker"], "started_unix": 0, "pid": 3456}
    previous = {}
    monkeypatch.setattr(job, "attempt_progress", lambda a: [])
    monkeypatch.setattr(job, "exact_process", lambda pid, cmd: pid == 3456 and cmd == attempt["command"])
    monkeypatch.setattr(job.os, "getpgid", lambda pid: pid)
    signals = []
    monkeypatch.setattr(job.os, "kill", lambda pid, sig: signals.append(("pid", pid, sig)))
    monkeypatch.setattr(job.os, "killpg", lambda pid, sig: signals.append(("group", pid, sig)))
    assert job.guard_orphan(attempt, previous, 899) == "coordinator absent; surviving worker monitored"
    assert signals == []
    job.guard_orphan(attempt, previous, 900)
    job.guard_orphan(attempt, previous, 930)
    job.guard_orphan(attempt, previous, 960)
    job.guard_orphan(attempt, previous, 1020)
    job.guard_orphan(attempt, previous, 2000)
    assert signals == [("pid", 3456, job.signal.SIGINT), ("group", 3456, job.signal.SIGTERM), ("group", 3456, job.signal.SIGKILL)]
    saved = job.read(tmp_path / "WATCH.json")
    assert saved["orphan_guard"]["sigkill_unix"] == 1020


def test_orphan_progress_resets_idle_but_preserves_absolute_wall_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    attempt = {"label": "test-a1", "command": ["python", "worker"], "started_unix": 0, "pid": 3456}
    previous = {}
    progress = [["trace", 10, int(800e9)]]
    monkeypatch.setattr(job, "attempt_progress", lambda a: progress)
    monkeypatch.setattr(job, "exact_process", lambda *a: True)
    signals = []
    monkeypatch.setattr(job.os, "kill", lambda pid, sig: signals.append(sig))
    job.guard_orphan(attempt, previous, 900)
    assert signals == []
    progress[0] = ["trace", 20, int(1650e9)]
    job.guard_orphan(attempt, previous, 1650)
    assert signals == []
    progress[0] = ["trace", 30, int(7200e9)]
    job.guard_orphan(attempt, previous, 7200)
    assert signals == [job.signal.SIGINT]


def test_orphan_group_escalation_requires_owned_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    attempt = {"label": "test-a1", "command": ["python", "worker"], "started_unix": 0, "pid": 3456}
    previous = {}
    monkeypatch.setattr(job, "attempt_progress", lambda a: [])
    monkeypatch.setattr(job, "exact_process", lambda *a: True)
    monkeypatch.setattr(job.os, "kill", lambda *a: None)
    monkeypatch.setattr(job.os, "getpgid", lambda pid: 9876)
    monkeypatch.setattr(job.os, "killpg", lambda *a: pytest.fail("unowned process group signaled"))
    job.guard_orphan(attempt, previous, 900)
    assert "ownership cannot be verified" in job.guard_orphan(attempt, previous, 960)


@pytest.mark.parametrize("fault", ("reversed_winner", "terminal_stocks", "replay_hash", "raw_rules", "raw_winner"))
def test_replay_scoring_is_independently_cross_checked(fake_result, fault):
    game, attempt, summary, directory, replay = fake_result
    raw = summary["replay_identity_audit"]["audit"]
    if fault == "reversed_winner": summary["execution"]["winner"] = "slippi-ai"
    if fault == "terminal_stocks": raw["terminal"]["slots"][str(game["frisson_port"])]["stocks"] = 1
    if fault == "replay_hash": raw["replay"]["sha256"] = "other"
    if fault == "raw_rules": raw["checks"]["all"] = False
    if fault == "raw_winner": raw["outcome"]["winner_port"] = game["slippi_port"]
    job.write(directory / "summary.json", summary)
    with pytest.raises(ValueError): job.validate_result(game, attempt)


@pytest.mark.parametrize("fault", (None, "trace_stocks", "causal_pair", "partial_trace", "raw_terminal_frame"))
def test_missing_end_requires_labeled_causal_terminal_stockout(fake_result, fault):
    game, attempt, summary, directory, replay = fake_result
    raw = summary["replay_identity_audit"]["audit"]
    raw.update(audit_passed=False, tournament_result_ready=False, end={"present": False},
               outcome={"reason": "missing_game_end", "conclusive": False, "game_complete": False, "status": "incomplete", "winner_port": None})
    summary["execution"].update(last_game_frame=876, termination="natural-game-end")
    summary["controller_boundary_audit"] = {"gate": {"decision": "pass", "checks": {"all": True}},
        "alignment": {"last_aligned_pair": [875, 876], "missing_internal_pairs": [], "permitted_terminal_unobservable_trace_frames": [876]}}
    summary["artifacts"]["replays"][0]["validation"] = {"required_trace_coverage": {"complete": True, "missing_frame_count": 0, "last_frame": 876}}
    trace = {"game_frame": 876, "slots": {
        f"p{game['frisson_port']}": {"port": game["frisson_port"], "model": "frisson-ai", "player_state": {"stocks": 2}},
        f"p{game['slippi_port']}": {"port": game["slippi_port"], "model": "slippi-ai", "player_state": {"stocks": 0}}}}
    if fault == "trace_stocks": trace["slots"][f"p{game['frisson_port']}"]["player_state"]["stocks"] = 3
    if fault == "causal_pair": summary["controller_boundary_audit"]["alignment"]["last_aligned_pair"] = [874, 875]
    if fault == "partial_trace": summary["artifacts"]["replays"][0]["validation"]["required_trace_coverage"]["missing_frame_count"] = 1
    if fault == "raw_terminal_frame": raw["terminal"]["frame_id"] = 875
    (directory / "controller_trace.jsonl").write_text(json.dumps(trace) + "\n")
    job.write(directory / "summary.json", summary)
    if fault is None:
        assert job.validate_result(game, attempt)["end_evidence"] == "causal-terminal-stockout-without-game-end-event"
    else:
        with pytest.raises(ValueError): job.validate_result(game, attempt)
