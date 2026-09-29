#!/usr/bin/env python3
"""Local-only frozen pair assignments and exactly-once benchmark ingestion.

There is no cloud or process launch API. An external reviewed coordinator
submits durable claims, binds the first physical execution and returns complete
artifact folders. All scoring uses the existing panel acceptance functions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

import run_slippi_public_panel as job
from finalize_paused_slippi_panel_attempt import existing_lock

SCHEMA = "e011.modal-panel-queue.v1"
MAX_JSON = 16 * 1024**2
MAX_FILES = 64
MAX_FILE = 256 * 1024**2
MAX_ATTEMPT = 512 * 1024**2
PHASES = tuple((profile, block) for profile in ("75m", "10m")
               for block in ("supported-mirror", "extended-roster", "forced-mirror"))
SHA = re.compile(r"[0-9a-f]{64}")
TOKEN = re.compile(r"[0-9a-f]{32}")


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else encoded(value)).hexdigest()


def read_bytes(path, cap=MAX_JSON):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("canonical regular file required")
    with path.open("rb") as stream: body = stream.read(cap + 1)
    if len(body) > cap: raise ValueError("bounded metadata input required")
    return body


def read(path):
    return json.loads(read_bytes(path))


def new_file(path, value):
    body = value if isinstance(value, bytes) else encoded(value)
    if len(body) > MAX_JSON: raise ValueError("bounded journal record required")
    with Path(path).open("xb") as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    # Make newly created names durable before any caller may submit cloud work.
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def make_plan(manifest_body, state_body, *, runtime_binding, max_workers=64):
    if type(max_workers) is not int or not 1 <= max_workers <= 64:
        raise ValueError("worker limit must be in 1..64")
    if (not isinstance(runtime_binding, dict) or set(runtime_binding) != {"runtime_image", "admission_sha256"}
            or not isinstance(runtime_binding["runtime_image"], str)
            or not re.fullmatch(r"im-[A-Za-z0-9_-]{1,128}", runtime_binding["runtime_image"])
            or not SHA.fullmatch(str(runtime_binding["admission_sha256"]))):
        raise ValueError("explicit externally reviewed Linux runtime binding required")
    manifest, state = json.loads(manifest_body), json.loads(state_body)
    games = manifest["games"]
    if (state.get("schema") != "e011.slippi_public_panel.state.v1"
            or state.get("manifest_sha256") != digest(manifest_body)
            or len(games) != len({g["label"] for g in games})
            or set(state["games"]) != {g["label"] for g in games}
            or state.get("current_label") is not None or state.get("worker_pid") is not None
            or state.get("supervisor_pid") is not None
            or state.get("local_pause", {}).get("new_local_launches_authorized") is not False):
        raise ValueError("exact stopped local queue and manifest required")
    if any(r["status"] not in {"pending", "complete", "quarantined"} for r in state["games"].values()):
        raise ValueError("running or unresolved local rows cannot migrate")
    if any(r.get("attempts") for r in state["games"].values() if r["status"] == "pending"):
        raise ValueError("initial pending attempts require reconciliation first")
    pairs, protected, prior_phase = {}, {}, -1
    for ordinal, game in enumerate(games, 1):
        phase = PHASES.index((game["profile"], game["block"]))
        if (game["ordinal"] != ordinal or phase < prior_phase or game.get("save_slp") is not True
                or game.get("save_video") is not False or not re.fullmatch(r"[a-z0-9_-]{1,200}", game["label"])
                or not re.fullmatch(r"[0-9a-f]{16}", game["pair_id"])):
            raise ValueError("original ordered SLP-only game specification required")
        prior_phase = phase
        row = state["games"][game["label"]]
        if row["status"] != "pending": protected[game["label"]] = digest(row)
        pairs.setdefault(game["pair_id"], []).append(game)
    pending = []
    phase_counts = [0] * len(PHASES)
    for pair_id, specs in pairs.items():
        if len(specs) != 2: raise ValueError("exact two-game physical-port pairs required")
        first, second = specs
        invariant = ("profile", "block", "release", "stage", "seed", "frisson_character", "slippi_character", "player_name")
        if (any(first[k] != second[k] for k in invariant)
                or {g["frisson_port"] for g in specs} != {1, 2}
                or any(g["slippi_port"] != 3 - g["frisson_port"] for g in specs)):
            raise ValueError("paired game context differs")
        statuses = [state["games"][g["label"]]["status"] for g in specs]
        if "pending" not in statuses: continue
        if statuses != ["pending", "pending"]: raise ValueError("port pair crosses existing and pending outcomes")
        phase = PHASES.index((first["profile"], first["block"]))
        pending.append({"pair_id": pair_id, "phase": phase, "worker_slot": phase_counts[phase] % max_workers,
                        "games": [{"label": g["label"], "spec_sha256": digest(g)} for g in specs]})
        phase_counts[phase] += 1
    return {"schema": SCHEMA, "manifest_sha256": digest(manifest_body), "state_snapshot_sha256": digest(state_body),
            "runtime_binding": copy.deepcopy(runtime_binding), "max_workers": max_workers,
            "phase_order": [list(row) for row in PHASES], "phase_pair_counts": phase_counts,
            "pairs": pending, "protected_rows": protected,
            "pending_games": 2 * len(pending), "existing_counts": state.get("counts"),
            "save_slp": True, "save_video": False, "maximum_attempts": 3,
            "assignment": "whole pairs; original order within each slot; phase barrier before later phases",
            "external_runtime_admission_required": True, "cloud_launches": 0}


def journal_path(value):
    path = Path(value).absolute()
    if (path.resolve(strict=False) != path or path.parent != job.OUTPUT / "migrations"
            or not re.fullmatch(r"modal-[a-z0-9-]{1,80}", path.name)):
        raise ValueError("canonical named modal migration journal required")
    return path


def freeze_plan(directory, *, runtime_binding, max_workers=64):
    directory = journal_path(directory)
    with existing_lock(job.OUTPUT / ".supervisor.lock"), existing_lock(job.GLOBAL_LOCK):
        job.validate_deployment()
        manifest_body, state_body = (read_bytes(job.OUTPUT / name) for name in ("manifest.json", "state.json"))
        plan = make_plan(manifest_body, state_body, runtime_binding=runtime_binding, max_workers=max_workers)
        job.revalidate_completed(json.loads(state_body), json.loads(manifest_body))
        if read_bytes(job.OUTPUT / "state.json") != state_body: raise RuntimeError("state changed during snapshot")
        directory.mkdir(mode=0o700)
        for name in ("claims", "executions", "ingests"): (directory / name).mkdir()
        new_file(directory / "manifest.json", manifest_body)
        new_file(directory / "state-snapshot.json", state_body)
        new_file(directory / "plan.json", plan)
        return {"plan": plan, "plan_sha256": digest(plan)}


def load_plan(directory, expected_sha256):
    directory = journal_path(directory)
    body = read_bytes(directory / "plan.json")
    if digest(body) != expected_sha256: raise ValueError("exact frozen migration plan required")
    plan = json.loads(body)
    manifest_body, state_body = (read_bytes(directory / name) for name in ("manifest.json", "state-snapshot.json"))
    if plan != make_plan(manifest_body, state_body, runtime_binding=plan["runtime_binding"], max_workers=plan["max_workers"]):
        raise ValueError("migration plan reconstruction differs")
    return plan, json.loads(manifest_body)


def current_state(plan, manifest):
    job.validate_deployment()
    body = read_bytes(job.OUTPUT / "state.json"); state = json.loads(body)
    if (digest(read_bytes(job.OUTPUT / "manifest.json")) != plan["manifest_sha256"]
            or state.get("manifest_sha256") != plan["manifest_sha256"]
            or set(state["games"]) != {g["label"] for g in manifest["games"]}
            or state.get("current_label") is not None or state.get("worker_pid") is not None
            or state.get("supervisor_pid") is not None
            or state.get("local_pause", {}).get("new_local_launches_authorized") is not False
            or any(digest(state["games"][label]) != expected for label, expected in plan["protected_rows"].items())):
        raise ValueError("local pause, original results or queue identity changed")
    return state, digest(body)


def no_unfinished_commit(directory, allowed=None):
    for index, path in enumerate(sorted((directory / "ingests").iterdir())):
        if index >= 3 * 2624: raise ValueError("journal commit count bound")
        if path.name != allowed and not (path / "done.json").is_file():
            raise RuntimeError("recover unfinished ingestion before another mutation")


def committed_record(directory, token, plan_sha):
    path = directory / "ingests" / token
    done, intent = read(path / "done.json"), read(path / "intent.json")
    required = {"schema", "plan_sha256", "attempt_token", "receipt_sha256", "status", "state_after_sha256",
                "committed", "new_cloud_submissions", "new_local_launches"}
    if (set(done) != required or done["schema"] != SCHEMA or done["plan_sha256"] != plan_sha
            or done["attempt_token"] != token or intent["plan_sha256"] != plan_sha
            or intent["attempt"]["attempt_token"] != token or done["receipt_sha256"] != intent["receipt_sha256"]
            or done["receipt_sha256"] != digest(read_bytes(path / "transport.json"))
            or done["status"] != intent["row_after"]["status"] or done["status"] not in {"complete", "pending", "quarantined"}
            or not SHA.fullmatch(str(done["state_after_sha256"])) or done["committed"] is not True
            or done["new_cloud_submissions"] != 0 or done["new_local_launches"] != 0):
        raise ValueError("committed ingestion identity differs")
    return done


def claims(directory, pair_id):
    if not re.fullmatch(r"[0-9a-f]{16}", pair_id): raise ValueError("exact pair identifier required")
    paths = sorted((directory / "claims").glob(pair_id + "-*.json"))
    if len(paths) > 6: raise ValueError("pair claim bound")
    return [read(path) for path in paths]


def claim_pair(directory, expected_plan_sha256, pair_id, *, expected_state_sha256, first_pending_only=False):
    """Persist intent before external submission; never resubmit a saved claim."""
    directory = journal_path(directory)
    if type(first_pending_only) is not bool:
        raise ValueError("explicit boolean game-granular claim required")
    with existing_lock(job.OUTPUT / ".supervisor.lock"), existing_lock(job.GLOBAL_LOCK):
        plan, manifest = load_plan(directory, expected_plan_sha256)
        state, sha = current_state(plan, manifest)
        if sha != expected_state_sha256: raise ValueError("claim requires current exact state SHA")
        no_unfinished_commit(directory)
        pair = next((p for p in plan["pairs"] if p["pair_id"] == pair_id), None)
        if pair is None: raise ValueError("pair absent from frozen pending assignment")
        lookup = {g["label"]: g for g in manifest["games"]}
        pair_index = plan["pairs"].index(pair)
        for other_index, other in enumerate(plan["pairs"]):
            earlier = other["phase"] < pair["phase"] or (other["phase"] == pair["phase"]
                and other["worker_slot"] == pair["worker_slot"] and other_index < pair_index)
            if earlier and any(state["games"][g["label"]]["status"] not in {"complete", "quarantined"} for g in other["games"]):
                raise ValueError("earlier phase or assigned slot pair remains unresolved")
        previous = claims(directory, pair_id)
        for prior in previous:
            if any(not (directory / "ingests" / a["attempt_token"] / "done.json").is_file() for a in prior["attempts"]):
                raise RuntimeError("existing claim must be reconciled before any new submission")
            for a in prior["attempts"]: committed_record(directory, a["attempt_token"], expected_plan_sha256)
        attempts = []
        for member in pair["games"]:
            game = lookup[member["label"]]; row = state["games"][game["label"]]
            if row["status"] in {"complete", "quarantined"}: continue
            if row["status"] != "pending" or len(row["attempts"]) >= 3 or digest(game) != member["spec_sha256"]:
                raise ValueError("exact pending game below attempt limit required")
            number = len(row["attempts"]) + 1
            label = f"{game['label']}-a{number}"
            if (job.ARTIFACTS / label).exists(): raise ValueError("unregistered artifact folder requires reconciliation")
            attempts.append({"game": game, "spec_sha256": member["spec_sha256"], "attempt_number": number,
                             "attempt_label": label, "attempt_token": uuid.uuid4().hex})
            if first_pending_only:
                break
        if not attempts: raise ValueError("pair already has terminal outcomes")
        token = uuid.uuid4().hex
        claim = {"schema": SCHEMA, "plan_sha256": expected_plan_sha256, "claim_token": token,
                 "pair_id": pair_id, "worker_slot": pair["worker_slot"], "phase": pair["phase"],
                 "runtime_binding": plan["runtime_binding"], "attempts": attempts,
                 "created_unix": time.time(), "state_before_sha256": sha, "automatic_resubmission": False}
        new_file(directory / "claims" / f"{pair_id}-{len(previous) + 1:02d}.json", claim)
        return claim


def checked_execution(execution):
    keys = {"app_id": r"ap-[A-Za-z0-9_-]{1,128}", "call_id": r"fc-[A-Za-z0-9_-]{1,128}",
            "task_id": r"ta-[A-Za-z0-9_-]{1,128}", "execution_id": r"[0-9a-f]{32}"}
    if not isinstance(execution, dict) or set(execution) != set(keys) or any(
            not re.fullmatch(pattern, str(execution[key])) for key, pattern in keys.items()):
        raise ValueError("exact physical Modal execution identity required")


def find_claim(directory, token):
    if not TOKEN.fullmatch(str(token)): raise ValueError("exact claim token required")
    found = []
    for index, path in enumerate((directory / "claims").iterdir()):
        if index >= 3 * 2624: raise ValueError("claim count bound")
        row = read(path)
        if row["claim_token"] == token: found.append(row)
    if len(found) != 1: raise ValueError("exactly one durable claim required")
    return found[0]


def validate_claim(claim, plan, manifest):
    pair = next((p for p in plan["pairs"] if p["pair_id"] == claim.get("pair_id")), None)
    if (pair is None or not TOKEN.fullmatch(str(claim.get("claim_token")))
            or claim.get("worker_slot") != pair["worker_slot"] or claim.get("phase") != pair["phase"]
            or claim.get("runtime_binding") != plan["runtime_binding"] or claim.get("automatic_resubmission") is not False
            or not isinstance(claim.get("attempts"), list) or not 1 <= len(claim["attempts"]) <= 2):
        raise ValueError("claim differs from whole-pair assignment")
    lookup = {g["label"]: g for g in manifest["games"]}
    labels, tokens = [], []
    for attempt in claim["attempts"]:
        game = attempt["game"]; label = game["label"]; number = attempt.get("attempt_number")
        if (game != lookup.get(label) or label not in {g["label"] for g in pair["games"]}
                or attempt.get("spec_sha256") != digest(game) or type(number) is not int or not 1 <= number <= 3
                or attempt.get("attempt_label") != f"{label}-a{number}"
                or not TOKEN.fullmatch(str(attempt.get("attempt_token")))):
            raise ValueError("immutable attempt specification differs")
        labels.append(label); tokens.append(attempt["attempt_token"])
    if len(set(tokens)) != len(tokens) or labels != [g["label"] for g in pair["games"] if g["label"] in labels]:
        raise ValueError("attempt order or identity differs")


def bind_execution(directory, expected_plan_sha256, claim_token, execution):
    """Bind first provider identity. A replacement requires explicit recovery."""
    directory = journal_path(directory); checked_execution(execution)
    with existing_lock(job.OUTPUT / ".supervisor.lock"), existing_lock(job.GLOBAL_LOCK):
        plan, manifest = load_plan(directory, expected_plan_sha256)
        claim = find_claim(directory, claim_token)
        validate_claim(claim, plan, manifest)
        if claim["plan_sha256"] != expected_plan_sha256: raise ValueError("claim plan differs")
        record = {"claim_sha256": digest(claim), "execution": execution}
        path = directory / "executions" / f"{claim_token}.json"
        if path.exists():
            if read(path) != record: raise ValueError("physical execution changed; reconcile preserved attempt")
        else: new_file(path, record)
        return record


def verify_artifacts(attempt_label, files):
    directory = (job.ARTIFACTS / attempt_label).absolute()
    if directory.resolve(strict=True) != directory or not directory.is_dir(): raise ValueError("canonical received attempt folder required")
    if not isinstance(files, list) or len(files) > MAX_FILES: raise ValueError("bounded full artifact inventory required")
    expected, total = {}, 0
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}: raise ValueError("exact artifact identity required")
        if not isinstance(row["path"], str): raise ValueError("artifact path must be a string")
        path = Path(row["path"])
        if (path.is_absolute() or str(path) != row["path"]
                or any(part in {".", ".."} for part in path.parts) or not path.parts
                or not re.fullmatch(r"[A-Za-z0-9_./-]{1,240}", row["path"])
                or path.suffix not in {".json", ".jsonl", ".slp", ".log"}
                or row["path"] in expected or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_FILE
                or not SHA.fullmatch(str(row["sha256"]))):
            raise ValueError("safe SLP-only artifact identity required")
        total += row["bytes"]
        if total > MAX_ATTEMPT: raise ValueError("attempt artifact byte bound")
        expected[row["path"]] = row
    actual = set()
    for index, path in enumerate(directory.rglob("*")):
        if index >= MAX_FILES * 3 or path.is_symlink(): raise ValueError("artifact tree bound or symlink")
        if path.is_dir(): continue
        relative = path.relative_to(directory).as_posix(); actual.add(relative)
        if relative not in expected or not stat.S_ISREG(path.lstat().st_mode): raise ValueError("unexpected artifact entry")
        row = expected[relative]
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno()); size = 0; sha = hashlib.sha256()
            if not stat.S_ISREG(before.st_mode) or before.st_size != row["bytes"]:
                raise ValueError("received regular artifact size differs")
            while True:
                chunk = stream.read(min(1024**2, row["bytes"] + 1 - size))
                if not chunk: break
                size += len(chunk); sha.update(chunk)
                if size > row["bytes"]: raise ValueError("received artifact grew beyond bound")
            after = os.fstat(stream.fileno()); named = path.lstat()
        if (size != row["bytes"] or sha.hexdigest() != row["sha256"]
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                   (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)):
            raise ValueError("received artifact identity differs")
    if actual != set(expected): raise ValueError("artifact inventory is incomplete")
    return directory


def proven_empty_socket_startup(attempt_label):
    """A failed pre-game ownership check is eligible for a bounded fresh boot.

    This never accepts gameplay and never changes the retained native summary.
    Require independent zero-work counters and an empty trace with no replay.
    """
    directory = job.ARTIFACTS / attempt_label
    try:
        summary = read(directory / "summary.json")
        execution = summary["execution"]
        trace = directory / "controller_trace.jsonl"
        return (summary.get("error") == "RuntimeError: Slippi port belongs to another process"
            and summary.get("result") == "failed"
            and execution.get("termination") == "not-started"
            and execution.get("processed_policy_frames") == 0
            and execution.get("first_game_frame") is None
            and execution.get("last_game_frame") is None
            and execution.get("first_game_context") is None
            and execution.get("natural_game_end") is False
            and execution.get("game_end_observed") is False
            and execution.get("winner") is None
            and execution.get("inference_counts") == {"frisson-ai": 0, "slippi-ai": 0}
            and execution.get("dispatch_counts") == {"frisson-ai": 0, "slippi-ai": 0}
            and execution.get("controller_transport", {}).get("installed") is False
            and summary.get("gate", {}).get("checks", {}).get("entered_gameplay") is False
            and read_bytes(trace, 0) == b""
            and not any(directory.rglob("*.slp")))
    except (KeyError, OSError, ValueError):
        return False


def ingest(directory, expected_plan_sha256, receipt_path):
    """Score an already received attempt, then publish once under local locks."""
    directory = journal_path(directory); receipt_body = read_bytes(receipt_path); receipt = json.loads(receipt_body)
    required = {"schema", "plan_sha256", "claim_token", "attempt_token", "execution", "files", "exit_code"}
    if (set(receipt) != required or receipt["schema"] != SCHEMA or receipt["plan_sha256"] != expected_plan_sha256
            or not TOKEN.fullmatch(str(receipt["attempt_token"]))
            or receipt["exit_code"] is not None and type(receipt["exit_code"]) is not int):
        raise ValueError("exact attempt transport receipt required")
    checked_execution(receipt["execution"])
    with existing_lock(job.OUTPUT / ".supervisor.lock"), existing_lock(job.GLOBAL_LOCK):
        plan, manifest = load_plan(directory, expected_plan_sha256)
        state, before_sha = current_state(plan, manifest)
        token = receipt["attempt_token"]; no_unfinished_commit(directory, token)
        claim = find_claim(directory, receipt["claim_token"])
        validate_claim(claim, plan, manifest)
        if claim["plan_sha256"] != expected_plan_sha256 or claim["runtime_binding"] != plan["runtime_binding"]:
            raise ValueError("claim plan or admitted runtime differs")
        execution = read(directory / "executions" / f"{claim['claim_token']}.json")
        if execution != {"claim_sha256": digest(claim), "execution": receipt["execution"]}:
            raise ValueError("receipt execution differs from first bound identity")
        candidates = [a for a in claim["attempts"] if a["attempt_token"] == token]
        if len(candidates) != 1: raise ValueError("exact durable attempt required")
        attempt = candidates[0]; game = attempt["game"]; label = game["label"]
        for earlier in claim["attempts"][:claim["attempts"].index(attempt)]:
            if not (directory / "ingests" / earlier["attempt_token"] / "done.json").is_file():
                raise ValueError("earlier assigned port game requires ingestion first")
            committed_record(directory, earlier["attempt_token"], expected_plan_sha256)
        if next((g for g in manifest["games"] if g["label"] == label), None) != game or digest(game) != attempt["spec_sha256"]:
            raise ValueError("attempt game differs from original specification")
        if label in plan["protected_rows"]: raise ValueError("original terminal row is immutable")
        verify_artifacts(attempt["attempt_label"], receipt["files"])
        record = {"label": attempt["attempt_label"], "attempt_token": token, "backend": "modal",
                  "plan_sha256": expected_plan_sha256, "claim_token": claim["claim_token"],
                  "execution": receipt["execution"], "transport_receipt_sha256": digest(receipt_body)}
        if receipt["exit_code"] is not None: record["exit_code"] = receipt["exit_code"]
        row = state["games"][label]
        destination = directory / "ingests" / token
        intent_path, done_path = destination / "intent.json", destination / "done.json"
        if intent_path.exists():
            intent = read(intent_path)
            if (intent["receipt_sha256"] != digest(receipt_body) or intent["attempt"] != attempt
                    or intent["plan_sha256"] != expected_plan_sha256 or row not in (intent["row_before"], intent["row_after"])):
                raise ValueError("ingestion recovery identity differs")
        else:
            if row["status"] != "pending" or len(row["attempts"]) + 1 != attempt["attempt_number"]:
                raise ValueError("attempt has already been consumed or state changed")
            after = copy.deepcopy(row)
            try:
                result = job.validate_result(game, record)
                record["accepted"] = True; after.update(status="complete", result=result)
            except (OSError, ValueError, KeyError) as error:
                status, reason = job.classify_failure(record)
                if status == "quarantined" and proven_empty_socket_startup(attempt["attempt_label"]):
                    status, reason = "pending", "retained socket-ownership startup failure; independently verified zero gameplay"
                if status == "pending" and attempt["attempt_number"] >= 3:
                    status, reason = "quarantined", "three retained infrastructure attempts exhausted"
                record.update(reconciled=True, error=f"{type(error).__name__}: {error}; {reason}")
                after.update(status=status)
            after["attempts"].append(record)
            intent = {"schema": SCHEMA, "plan_sha256": expected_plan_sha256, "attempt": attempt,
                      "receipt_sha256": digest(receipt_body), "state_before_sha256": before_sha,
                      "row_before": row, "row_after": after}
            destination.mkdir(exist_ok=True)
            if destination.resolve(strict=True) != destination: raise ValueError("canonical commit directory required")
            transport = destination / "transport.json"
            if transport.exists():
                if read_bytes(transport) != receipt_body: raise ValueError("partial transport receipt differs")
            else: new_file(transport, receipt_body)
            new_file(intent_path, intent)
        if intent["row_after"]["status"] == "complete":
            if job.validate_result(game, record) != intent["row_after"]["result"]:
                raise ValueError("accepted evidence changed during ingestion")
        verify_artifacts(attempt["attempt_label"], receipt["files"])
        if done_path.exists():
            if row != intent["row_after"]: raise ValueError("committed result disappeared")
            return {**committed_record(directory, token, expected_plan_sha256), "idempotent": True}
        updated = copy.deepcopy(state); updated["games"][label] = intent["row_after"]
        job.publish(updated, manifest)
        if read(job.OUTPUT / "state.json") != updated: raise RuntimeError("published ingestion differs")
        result = {"schema": SCHEMA, "plan_sha256": expected_plan_sha256, "attempt_token": token,
                  "receipt_sha256": digest(receipt_body), "status": intent["row_after"]["status"],
                  "state_after_sha256": digest(read_bytes(job.OUTPUT / "state.json")), "committed": True,
                  "new_cloud_submissions": 0, "new_local_launches": 0}
        new_file(done_path, result)
        return {**result, "idempotent": False}


def progress(directory, expected_plan_sha256):
    """Read-only local ledger projection; provider liveness is unobserved here.

    Game categories are disjoint. Attempt counters separately include previous
    infrastructure attempts and a commit that may need recovery after publish.
    """
    directory = journal_path(directory)
    with existing_lock(job.OUTPUT / ".supervisor.lock"), existing_lock(job.GLOBAL_LOCK):
        plan, manifest = load_plan(directory, expected_plan_sha256)
        state, state_sha = current_state(plan, manifest)
        phases = [{"phase": i, "profile": profile, "block": block,
                   "games": {"queued": 0, "running_or_awaiting": 0, "accepted": 0, "quarantined": 0},
                   "attempts": {"claimed": 0, "execution_bound": 0, "reconciled": 0, "awaiting_ingest": 0},
                   "original_accepted": 0, "original_quarantined": 0}
                  for i, (profile, block) in enumerate(PHASES)]
        active, used, unfinished = set(), set(), []
        for index, path in enumerate(sorted((directory / "claims").iterdir())):
            if index >= 3 * 2624: raise ValueError("progress claim count bound")
            claim = read(path); validate_claim(claim, plan, manifest)
            if claim["plan_sha256"] != expected_plan_sha256: raise ValueError("progress claim plan differs")
            execution_path = directory / "executions" / f"{claim['claim_token']}.json"
            bound = execution_path.is_file()
            if bound:
                execution = read(execution_path); checked_execution(execution["execution"])
                if execution["claim_sha256"] != digest(claim): raise ValueError("progress execution binding differs")
            phase = phases[claim["phase"]]
            for attempt in claim["attempts"]:
                token = attempt["attempt_token"]
                if token in used: raise ValueError("attempt token appears in multiple claims")
                used.add(token); phase["attempts"]["claimed"] += 1
                phase["attempts"]["execution_bound"] += int(bound)
                commit = directory / "ingests" / token
                if (commit / "done.json").is_file():
                    committed_record(directory, token, expected_plan_sha256)
                    phase["attempts"]["reconciled"] += 1
                else:
                    active.add(attempt["game"]["label"])
                    phase["attempts"]["awaiting_ingest"] += 1
                    if commit.exists(): unfinished.append(token)
        for game in manifest["games"]:
            label = game["label"]; status = state["games"][label]["status"]
            phase = phases[PHASES.index((game["profile"], game["block"]))]
            if label in plan["protected_rows"]:
                phase["original_accepted" if status == "complete" else "original_quarantined"] += 1
            elif status == "complete": phase["games"]["accepted"] += 1
            elif status == "quarantined": phase["games"]["quarantined"] += 1
            elif status == "pending": phase["games"]["running_or_awaiting" if label in active else "queued"] += 1
            else: raise ValueError("unexpected scientific row state in migration")
        return {"schema": SCHEMA, "plan_sha256": expected_plan_sha256, "state_sha256": state_sha,
                "phases": phases, "original_accepted": sum(p["original_accepted"] for p in phases),
                "original_quarantined": sum(p["original_quarantined"] for p in phases),
                "pending_snapshot_games": plan["pending_games"], "unfinished_ingest_tokens": unfinished,
                "provider_liveness": "unobserved; running_or_awaiting includes saved submission intent and result transport",
                "new_local_launches_authorized": False, "read_only": True}
