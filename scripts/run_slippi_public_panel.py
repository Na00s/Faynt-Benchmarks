#!/usr/bin/env python3
"""Crash-resumable, SLP-only public checkpoint benchmark and independent watch."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1"
EXISTING = ROOT / "artifacts/integration/frisson_ai/postrl-p21-step1318-step222-v1"
ARTIFACTS = ROOT / "artifacts/integration/frisson_ai"
GLOBAL_LOCK = ARTIFACTS / ".final-winner-benchmark.lock"
SERVICE = f"gui/{os.getuid()}/com.frisson.slippi-public-panel-p21"
PYTHON = ROOT / ".e010-env/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_binding():
    paths = [*sorted((ROOT / "src/melee_policy/integration").glob("*.py")),
             *sorted((ROOT / "scripts").glob("slippi_panel_*.py")), Path(__file__).resolve(),
             ROOT / "configs/integration.toml", ROOT / "requirements-e010.lock", OUTPUT / "manifest.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def exact_process(pid, command):
    if not isinstance(pid, int) or pid <= 1:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=10)
    # macOS may resolve the interpreter symlink; preserve exact script and args.
    words = shlex.split(result.stdout.strip()) if result.returncode == 0 else []
    return len(words) == len(command) and words[1:] == command[1:]


def predecessor_complete():
    """Release admission uses the selected frozen panel's complete schedule."""
    panel = read(OUTPUT / "manifest.json")
    games = panel["games"]
    return (len(games) == 2624 and len({g["label"] for g in games}) == 2624
            and Counter(g["profile"] for g in games) == {"75m": 1312, "10m": 1312}
            and all(g["frisson_port"] in (1, 2) and g["slippi_port"] == 3 - g["frisson_port"] for g in games))


def verify_replay_outcome(summary, game, directory, replay, stocks, winner):
    """Compare live scoring with independently parsed physical replay stocks.

    The inherited engine sometimes stops on a decisive zero-stock frame before
    Dolphin writes Game End. Such files require a narrowly labeled, fully
    causal terminal-stockout proof; their raw parser outcome remains unchanged.
    """
    raw = summary["replay_identity_audit"]["audit"]
    if not raw["rules_passed"] or raw.get("failures") or not raw["checks"] or not all(raw["checks"].values()):
        raise ValueError("independent replay rules audit differs")
    if raw["replay"]["sha256"] != replay["sha256"]:
        raise ValueError("independent outcome belongs to a different replay")
    mapping = {game["frisson_port"]: "frisson-ai", game["slippi_port"]: "slippi-ai"}
    terminal = raw["terminal"]
    if {int(p) for p in terminal["slots"]} != set(mapping):
        raise ValueError("independent terminal ports differ")
    for port, role in mapping.items():
        if terminal["slots"][str(port)]["stocks"] != stocks[role]:
            raise ValueError("independent replay terminal stocks differ")
    expected_port = next(port for port, role in mapping.items() if role == winner)
    other_role = "slippi-ai" if winner == "frisson-ai" else "frisson-ai"
    if stocks[winner] < stocks[other_role]:
        raise ValueError("winner contradicts remaining stocks")
    outcome = raw["outcome"]
    if raw["end"]["present"]:
        if not raw["audit_passed"] or not raw["tournament_result_ready"] or not outcome["conclusive"] or not outcome["game_complete"] or outcome["status"] != "win" or outcome["winner_port"] != expected_port:
            raise ValueError("independent replay winner differs")
        return "formal-game-end"
    if outcome["reason"] != "missing_game_end" or stocks[other_role] != 0 or stocks[winner] <= 0:
        raise ValueError("missing Game End lacks decisive terminal stockout")
    execution = summary["execution"]
    last_frame = execution["last_game_frame"]
    boundary = summary["controller_boundary_audit"]
    coverage = replay["validation"]["required_trace_coverage"]
    if (execution["termination"] != "natural-game-end" or terminal["frame_id"] != last_frame
            or boundary["gate"]["decision"] != "pass" or not all(boundary["gate"]["checks"].values())
            or boundary["alignment"]["last_aligned_pair"] != [last_frame - 1, last_frame]
            or boundary["alignment"]["missing_internal_pairs"]
            or boundary["alignment"]["permitted_terminal_unobservable_trace_frames"] != [last_frame]
            or not coverage["complete"] or coverage["missing_frame_count"] != 0 or coverage["last_frame"] != last_frame):
        raise ValueError("terminal stockout lacks complete causal replay proof")
    trace_path = directory / "controller_trace.jsonl"
    with trace_path.open("rb") as stream:
        size = trace_path.stat().st_size
        stream.seek(max(0, size - 1024 * 1024))
        last = json.loads(stream.read().splitlines()[-1])
    if last["game_frame"] != last_frame:
        raise ValueError("terminal trace frame differs")
    for port, role in mapping.items():
        slot = last["slots"][f"p{port}"]
        if slot["port"] != port or slot["model"] != role or slot["player_state"]["stocks"] != stocks[role]:
            raise ValueError("terminal trace identity or stocks differ")
    return "causal-terminal-stockout-without-game-end-event"


def validate_result(game, attempt):
    directory = ARTIFACTS / attempt["label"]
    path = directory / "summary.json"
    summary = read(path)
    if summary.get("gate", {}).get("decision") != "pass" or summary.get("result") != "complete":
        raise ValueError(f"saved game failed acceptance: {summary.get('error')}")
    if not summary["gate"]["checks"] or not all(summary["gate"]["checks"].values()):
        raise ValueError("a required game gate is false")
    cfg, execution = summary["configuration"], summary["execution"]
    expected = [("player_1", "frisson-ai", game["frisson_character"], game["frisson_port"]),
                ("player_2", "slippi-ai", game["slippi_character"], game["slippi_port"])]
    for key, model, character, port in expected:
        if any(cfg[key].get(k) != v for k, v in {"model": model, "character": character, "port": port}.items()):
            raise ValueError("physical player identity differs")
    if cfg["seed"] != game["seed"] or cfg["stage"] != game["stage"]:
        raise ValueError("seed or stage differs")
    if not execution["natural_game_end"] or execution["processed_policy_frames"] <= 0:
        raise ValueError("natural end evidence missing")
    records = [r for r in summary["artifacts"]["replays"] if r.get("tournament_result_replay") is True]
    if len(records) != 1:
        raise ValueError("expected one authoritative replay")
    replay = records[0]
    replay_path = ROOT / replay["path"]
    if not replay_path.resolve().is_relative_to(directory.resolve()) or sha(replay_path) != replay["sha256"] or replay_path.stat().st_size != replay["byte_length"]:
        raise ValueError("replay identity differs")
    stocks = execution["last_stocks"]
    if set(stocks) != {"frisson-ai", "slippi-ai"} or any(type(v) is not int or not 0 <= v <= 4 for v in stocks.values()):
        raise ValueError("invalid stocks")
    winner = execution["winner"]
    if winner not in {"frisson-ai", "slippi-ai"}:
        raise ValueError("ambiguous winner")
    end_evidence = verify_replay_outcome(summary, game, directory, replay, stocks, winner)
    panel = read(OUTPUT / "manifest.json")
    own = panel["frisson_checkpoints"][game["profile"]]
    other = panel["releases"][game["release"]]
    contract = summary["contract"]
    if contract["frisson"]["selected_checkpoint"]["sha256"] != own["sha256"] or contract["slippi_ai"]["checkpoint_sha256"] != other["sha256"]:
        raise ValueError("wrong policy checkpoint")
    if contract["slippi_ai"]["player_name"] != game["player_name"]:
        raise ValueError("wrong conditioning name")
    own_policy = summary["policies"]["p1"]["metadata"]["runtime_contract"]
    other_policy = summary["policies"]["p2"]["metadata"]["runtime_contract"]
    if (own_policy["port"], own_policy["opponent_port"]) != (game["frisson_port"], game["slippi_port"]) or (other_policy["port"], other_policy["opponent_port"]) != (game["slippi_port"], game["frisson_port"]):
        raise ValueError("policy perspective differs from physical ports")
    return {"winner": winner, "win": winner == "frisson-ai", "stocks_remaining": stocks, "end_evidence": end_evidence,
            "stocks_taken": 4 - stocks["slippi-ai"], "stocks_conceded": 4 - stocks["frisson-ai"],
            "summary": str(path), "summary_sha256": sha(path), "replay": str(replay_path),
            "replay_sha256": replay["sha256"], "replay_bytes": replay["byte_length"],
            "wall_seconds": execution["wall_seconds"], "game_frames": execution["processed_policy_frames"]}


def fresh_state(panel):
    return {"schema": "e011.slippi_public_panel.state.v1", "manifest_sha256": sha(OUTPUT / "manifest.json"),
            "status": "waiting-predecessor", "games": {g["label"]: {"status": "pending", "attempts": []} for g in panel["games"]}}


def publish(state, panel):
    state["updated_unix"] = time.time()
    state["counts"] = dict(Counter(g["status"] for g in state["games"].values()))
    write(OUTPUT / "state.json", state)
    status = {k: v for k, v in state.items() if k != "games"}
    status.update(total=len(panel["games"]), save_slp=True, save_video=False)
    write(OUTPUT / "STATUS.json", status)
    lines = ["# Slippi-AI public checkpoint benchmark", "", f"Status: {state['status']}. Accepted {state['counts'].get('complete', 0)}/{len(panel['games'])}.", "",
             "Stocks are taken / conceded from Frisson's perspective. Every result requires replay and controller gates. Paired ports; native delays; ring-context Frisson runtime.", ""]
    for profile in ("75m", "10m"):
        lines.extend([f"## {profile.upper()} RL", "", "| Opponent | Block | Complete | W-L | Stocks taken / conceded |", "| --- | --- | ---: | ---: | ---: |"])
        for release in panel["releases"]:
            for block in ("supported-mirror", "extended-roster", "forced-mirror"):
                specs = [g for g in panel["games"] if g["profile"] == profile and g["release"] == release and g["block"] == block]
                done = [state["games"][g["label"]]["result"] for g in specs if state["games"][g["label"]]["status"] == "complete"]
                wins = sum(r["win"] for r in done)
                lines.append(f"| {release} | {block} | {len(done)}/{len(specs)} | {wins}-{len(done)-wins} | {sum(r['stocks_taken'] for r in done)} / {sum(r['stocks_conceded'] for r in done)} |")
        lines.append("")
    temp = OUTPUT / ".RESULTS.md.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(OUTPUT / "RESULTS.md")


def attempt_progress(attempt):
    directory = ARTIFACTS / attempt["label"]
    paths = [directory / "controller_trace.jsonl", directory / "summary.json", *sorted((directory / "replays").glob("*.slp"))]
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()]


def registered_attempt(attempt):
    """Recover child identity even if its parent died before saving Popen.pid."""
    path = OUTPUT / "workers" / f"{attempt['label']}.json"
    if not path.exists():
        return attempt
    receipt = read(path)
    if receipt.get("label") != attempt["label"] or receipt.get("command") != attempt["command"]:
        raise RuntimeError("worker registration differs from persisted launch intent")
    return {**attempt, "pid": receipt["pid"], "registration": str(path)}


def workflow_lock_held():
    with (OUTPUT / ".supervisor.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def guard_orphan(attempt, previous, now):
    """Track an orphan using the same progress/wall limits as its coordinator.

    Signals are persisted before dispatch. Group escalation requires the exact
    registered worker to remain its own session/process-group leader.
    """
    identity = {"label": attempt["label"], "pid": attempt["pid"], "command": attempt["command"]}
    guard = previous.get("orphan_guard", {})
    observed = [list(row) for row in attempt_progress(attempt)]
    if guard.get("identity") != identity:
        latest = max((row[2] / 1e9 for row in observed), default=attempt["started_unix"])
        guard = {"identity": identity, "first_seen_unix": now, "progress_unix": min(latest, now), "progress": observed}
    elif guard.get("progress") != observed:
        guard.update(progress=observed, progress_unix=now)
    guard.update(checked_unix=now, idle_seconds=now - guard["progress_unix"],
                 wall_seconds=now - attempt["started_unix"])
    previous["orphan_guard"] = guard
    expired = guard["idle_seconds"] >= 900 or guard["wall_seconds"] >= 7200
    if not expired:
        return "coordinator absent; surviving worker monitored"
    if not exact_process(attempt["pid"], attempt["command"]):
        return "orphan exited during timeout check"
    action = None
    if "sigint_unix" not in guard:
        action = "sigint"
    elif now - guard["sigint_unix"] >= 60 and "sigterm_unix" not in guard:
        action = "sigterm"
    elif "sigterm_unix" in guard and now - guard["sigterm_unix"] >= 60 and "sigkill_unix" not in guard:
        action = "sigkill"
    if action is not None:
        if action != "sigint" and os.getpgid(attempt["pid"]) != attempt["pid"]:
            return "stalled orphan process-group ownership cannot be verified"
        guard[action + "_unix"] = now
        write(OUTPUT / "WATCH.json", previous)
        signal_value = {"sigint": signal.SIGINT, "sigterm": signal.SIGTERM, "sigkill": signal.SIGKILL}[action]
        if action == "sigint":
            os.kill(attempt["pid"], signal_value)
        else:
            os.killpg(attempt["pid"], signal_value)
    return "stalled orphan recovery in progress"


def classify_failure(attempt):
    path = ARTIFACTS / attempt["label"] / "summary.json"
    if path.exists():
        try:
            summary = read(path)
        except (ValueError, OSError) as error:
            return "quarantined", f"unreadable summary retained for audit: {error}"
        if summary.get("execution", {}).get("natural_game_end"):
            return "quarantined", "natural-end game requires audit; retained without resampling"
        error = summary.get("error") or summary.get("execution", {}).get("termination", "unknown")
        transient = any(word in str(error).lower() for word in ("connect", "timeout", "timed out", "keyboardinterrupt", "no-frame", "no frame"))
        return ("pending" if transient else "quarantined"), str(error)
    # A postgame serialization error can lose the summary after a full match.
    # Retained gameplay evidence must be audited before any further attempt.
    directory = ARTIFACTS / attempt["label"]
    evidence = [directory / "controller_trace.jsonl", *directory.glob("replays/*.slp")]
    if any(item.is_file() and item.stat().st_size > 0 for item in evidence):
        return "quarantined", "worker ended without summary but retained gameplay evidence; audit before any resampling"
    # No summary and no gameplay evidence can be an interpreter/startup crash.
    return "pending", "worker ended without summary; attempt retained"


def validate_deployment():
    deployment = read(OUTPUT / "deployment.json")
    if deployment["source_binding"] != source_binding():
        raise RuntimeError("frozen panel source or manifest changed")
    for relative_path, expected in deployment.get("validation_artifacts", {}).items():
        if sha(ROOT / relative_path) != expected:
            raise RuntimeError("frozen validation evidence changed")
    if not read(OUTPUT / "canary-run.json")["passed"]:
        raise RuntimeError("checkpoint canaries have not all passed")
    if not read(OUTPUT / "tests.json")["passed"]:
        raise RuntimeError("focused deployment tests have not passed")
    return deployment


def revalidate_completed(state, panel):
    """A restart cannot silently trust a stale result or changed replay."""
    for game in panel["games"]:
        row = state["games"][game["label"]]
        if row["status"] != "complete":
            continue
        accepted = [a for a in row["attempts"] if a.get("accepted")]
        if len(accepted) != 1:
            raise RuntimeError("completed row lacks exactly one accepted attempt")
        current = validate_result(game, accepted[0])
        if current != row["result"]:
            raise RuntimeError(f"accepted evidence changed: {game['label']}")


def worker(label, attempt_label):
    command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--worker", label, "--attempt", attempt_label]
    write(OUTPUT / "workers" / f"{attempt_label}.json", {
        "label": attempt_label, "pid": os.getpid(), "command": command,
        "registered_unix": time.time(), "parent_pid": os.getppid()})
    panel = read(OUTPUT / "manifest.json")
    game = next(g for g in panel["games"] if g["label"] == label)
    validate_deployment()
    with GLOBAL_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75)
        from slippi_panel_runtime import run_game
        run_game(game, attempt_label)


def supervise():
    panel = read(OUTPUT / "manifest.json")
    validate_deployment()
    with (OUTPUT / ".supervisor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        state = read(OUTPUT / "state.json") if (OUTPUT / "state.json").exists() else fresh_state(panel)
        if state["manifest_sha256"] != sha(OUTPUT / "manifest.json") or set(state["games"]) != {g["label"] for g in panel["games"]}:
            raise RuntimeError("queue identity differs")
        state["supervisor_pid"] = os.getpid()
        revalidate_completed(state, panel)
        while not predecessor_complete():
            state["status"] = "waiting-predecessor"
            publish(state, panel)
            time.sleep(30)
            validate_deployment()
        for game in panel["games"]:
            row = state["games"][game["label"]]
            if row["status"] in {"complete", "quarantined"}:
                continue
            state["current_label"] = game["label"]
            # Reconcile a previous attempt before spending another game.
            for attempt in row["attempts"]:
                if attempt.get("accepted"):
                    row.update(status="complete", result=validate_result(game, attempt))
                    break
                if attempt.get("reconciled"):
                    continue
                if exact_process(attempt.get("pid"), attempt["command"]):
                    raise RuntimeError("surviving worker unexpectedly lacks inherited workflow lock")
                try:
                    result = validate_result(game, attempt)
                    attempt["accepted"] = True
                    row.update(status="complete", result=result)
                    break
                except (OSError, ValueError, KeyError) as error:
                    row["status"], attempt["error"] = classify_failure(attempt)
                    attempt["reconciled"] = True
            while row["status"] not in {"complete", "quarantined"}:
                if len(row["attempts"]) >= 3:
                    row.update(status="quarantined", error="three retained infrastructure attempts exhausted")
                    break
                validate_deployment()
                if os.statvfs(OUTPUT).f_bavail * os.statvfs(OUTPUT).f_frsize < 8 * 1024**3:
                    raise RuntimeError("less than 8 GiB free; no new game launched")
                number = len(row["attempts"]) + 1
                attempt_label = f"{game['label']}-a{number}"
                if (ARTIFACTS / attempt_label).exists():
                    raise RuntimeError("unregistered attempt directory exists; retain for audit")
                command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--worker", game["label"], "--attempt", attempt_label]
                attempt = {"label": attempt_label, "command": command, "started_unix": time.time()}
                row["attempts"].append(attempt)
                row["status"] = state["status"] = "running"
                publish(state, panel)  # Intent precedes child launch.
                log = OUTPUT / "logs" / f"{attempt_label}.log"
                log.parent.mkdir(exist_ok=True)
                with log.open("a") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lock.fileno(),), env={**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/scripts"})
                    attempt["pid"] = child.pid
                    state["worker_pid"] = child.pid
                    publish(state, panel)
                    progress, progress_at = None, time.time()
                    while child.poll() is None:
                        observed = attempt_progress(attempt)
                        if observed != progress:
                            progress, progress_at = observed, time.time()
                        if time.time() - progress_at > 900 or time.time() - attempt["started_unix"] > 7200:
                            if exact_process(child.pid, command):
                                os.kill(child.pid, signal.SIGINT)
                                attempt["watchdog_interrupt"] = time.time()
                                try:
                                    child.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    if exact_process(child.pid, command):
                                        os.killpg(child.pid, signal.SIGTERM)
                                    child.wait(timeout=30)
                            break
                        state["progress"] = observed
                        state["progress_unix"] = progress_at
                        publish(state, panel)
                        time.sleep(15)
                attempt["exit_code"] = child.returncode
                attempt["ended_unix"] = time.time()
                state["worker_pid"] = None
                if child.returncode == 75 and not (ARTIFACTS / attempt_label).exists():
                    row["attempts"].pop()
                    row["status"] = "pending"
                    state["status"] = "waiting-dolphin-lock"
                    publish(state, panel)
                    time.sleep(30)
                    continue
                try:
                    row.update(status="complete", result=validate_result(game, attempt))
                    attempt["accepted"] = True
                except (OSError, ValueError, KeyError) as error:
                    row["status"], reason = classify_failure(attempt)
                    attempt.update(reconciled=True, error=f"{type(error).__name__}: {error}; {reason}")
                publish(state, panel)
        state["current_label"] = None
        state["status"] = "complete" if all(r["status"] == "complete" for r in state["games"].values()) else "finished-with-quarantined-games"
        publish(state, panel)
        return 0


def watch():
    path = OUTPUT / "WATCH.json"
    previous = read(path) if path.exists() else {}
    command = [str(PYTHON), "-u", str(Path(__file__).resolve())]
    with (OUTPUT / ".watch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        while True:
            try:
                state = read(OUTPUT / "STATUS.json") if (OUTPUT / "STATUS.json").exists() else {}
                alive = exact_process(state.get("supervisor_pid"), command)
                orphan_alive = False
                orphan_alert = None
                identity_pending = False
                if not alive and state.get("current_label"):
                    ledger = read(OUTPUT / "state.json")
                    current = ledger.get("games", {}).get(state.get("current_label"), {})
                    for recorded in current.get("attempts", []):
                        attempt = registered_attempt(recorded)
                        if exact_process(attempt.get("pid"), attempt["command"]):
                            orphan_alive = True
                            orphan_alert = guard_orphan(attempt, previous, time.time())
                            break
                    identity_pending = not orphan_alive and workflow_lock_held()
                done = state.get("status") in {"complete", "finished-with-quarantined-games"}
                signature = json.dumps(state.get("counts", {}), sort_keys=True)
                if signature != previous.get("signature"):
                    previous["restart_count"] = 0
                previous.update(checked_unix=time.time(), watcher_pid=os.getpid(), signature=signature,
                    supervisor_alive=alive, surviving_worker_alive=orphan_alive, worker_identity_pending=identity_pending,
                    status=state.get("status"), counts=state.get("counts"),
                    current_label=state.get("current_label"), heartbeat_age=time.time() - state.get("updated_unix", 0))
                if not alive and not orphan_alive and not identity_pending and not done and time.time() - previous.get("last_restart", 0) >= 90 and previous.get("restart_count", 0) < 3:
                    previous.update(last_restart=time.time(), restart_count=previous.get("restart_count", 0) + 1)
                    write(path, previous)
                    result = subprocess.run(["launchctl", "kickstart", SERVICE], capture_output=True, text=True, timeout=15)
                    previous["recovery"] = {"returncode": result.returncode, "stderr": result.stderr}
                previous["alert"] = (orphan_alert if orphan_alive else "workflow lock held; worker identity pending" if identity_pending
                                     else "supervisor absent; recovery exhausted" if not alive and not done and previous.get("restart_count", 0) >= 3
                                     else "supervisor heartbeat stale" if alive and previous["heartbeat_age"] > 180 else None)
                write(path, previous)
                if done:
                    return 0
            except Exception as error:
                previous.update(checked_unix=time.time(), alert=f"{type(error).__name__}: {error}")
                write(path, previous)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--attempt")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not args.attempt or not args.attempt.startswith(args.worker + "-a"):
            raise ValueError("invalid attempt identity")
        worker(args.worker, args.attempt)
        return 0
    return watch() if args.watch else supervise()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write(OUTPUT / "SUPERVISOR_ERROR.json", {"time": time.time(), "error": f"{type(error).__name__}: {error}"})
        raise
