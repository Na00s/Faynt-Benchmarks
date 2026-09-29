#!/usr/bin/env python3
"""Accept one finished attempt after local coordinators have been stopped.

The operator must disable both launchd services and remove the old coordinator
processes first. This helper never starts, resumes or terminates a process.
Without --commit it validates and returns a preview. Commit preserves the three
published files before using the existing acceptance gate and publisher.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

import run_slippi_public_panel as job

SCHEMA = "e011.slippi-panel.one-attempt-finalization.v1"
PAUSED = "paused-local-awaiting-validated-modal-runtime"
MAX_BYTES = 16 * 1024**2
PUBLISHED = ("state.json", "STATUS.json", "RESULTS.md")


def bounded_bytes(path):
    path = Path(path)
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"regular canonical file required: {path}")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("input exceeds 16 MiB")
    return data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_absent(pid):
    if type(pid) is not int or pid <= 1:
        raise ValueError("specific process PID required")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise RuntimeError(f"cannot prove process {pid} has exited") from error
    raise RuntimeError(f"process {pid} remains present")


@contextmanager
def existing_lock(path):
    path = Path(path)
    if path.resolve(strict=True) != path:
        raise ValueError("lock path must be canonical")
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("existing regular lock required")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = path.stat()
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("lock inode changed")
        yield
    finally:
        os.close(fd)


def new_file(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(data) > MAX_BYTES:
        raise ValueError("receipt exceeds 16 MiB")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def receipt_destination(value):
    path = Path(value).absolute()
    parent = job.OUTPUT / "reconciliations"
    if (path.parent != parent or path.resolve(strict=False) != path or path.exists()
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,99}", path.name)):
        raise ValueError("new named receipt directory beneath panel/reconciliations required")
    return path


def finalize(*, game_label, attempt_label, worker_pid, supervisor_pid, watcher_pid,
             expected_state_sha256, commit=False, receipt_dir=None):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256):
        raise ValueError("exact expected state SHA-256 required")
    if len({worker_pid, supervisor_pid, watcher_pid}) != 3:
        raise ValueError("worker and coordinator PIDs must be distinct")
    for pid in (supervisor_pid, watcher_pid, worker_pid):
        require_absent(pid)
    destination = receipt_destination(receipt_dir) if commit else None
    with existing_lock(job.OUTPUT / ".supervisor.lock"), existing_lock(job.GLOBAL_LOCK):
        for pid in (supervisor_pid, watcher_pid, worker_pid):
            require_absent(pid)
        before = {name: bounded_bytes(job.OUTPUT / name) for name in PUBLISHED}
        if digest(before["state.json"]) != expected_state_sha256:
            raise ValueError("state changed from exact approved snapshot")
        state = json.loads(before["state.json"])
        panel = json.loads(bounded_bytes(job.OUTPUT / "manifest.json"))
        watch = json.loads(bounded_bytes(job.OUTPUT / "WATCH.json"))
        if (state.get("schema") != "e011.slippi_public_panel.state.v1"
                or state.get("manifest_sha256") != job.sha(job.OUTPUT / "manifest.json")
                or set(state.get("games", {})) != {g["label"] for g in panel["games"]}
                or state.get("current_label") != game_label
                or state.get("worker_pid") != worker_pid or state.get("supervisor_pid") != supervisor_pid
                or watch.get("watcher_pid") != watcher_pid):
            raise ValueError("manifest, queue or coordinator identity differs")
        games = [g for g in panel["games"] if g["label"] == game_label]
        if len(games) != 1:
            raise ValueError("exactly one manifest game required")
        row = state["games"][game_label]
        attempts = [a for a in row.get("attempts", []) if a.get("label") == attempt_label]
        expected_command = [str(job.PYTHON), "-u", str(job.ROOT / "scripts/run_slippi_public_panel.py"),
                            "--worker", game_label, "--attempt", attempt_label]
        if (row.get("status") != "running" or len(attempts) != 1
                or row["attempts"][-1] != attempts[0] or attempts[0].get("pid") != worker_pid
                or attempts[0].get("command") != expected_command
                or any(a.get("accepted") for a in row["attempts"])
                or attempts[0].get("reconciled")
                or not re.fullmatch(re.escape(game_label) + r"-a[1-9][0-9]*", attempt_label)):
            raise ValueError("exact unreconciled current attempt required")
        job.validate_deployment()
        result = job.validate_result(games[0], attempts[0])
        if any(bounded_bytes(job.OUTPUT / name) != value for name, value in before.items()):
            raise RuntimeError("published files changed during validation")
        for pid in (supervisor_pid, watcher_pid, worker_pid):
            require_absent(pid)
        report = {"schema": SCHEMA, "game": game_label, "attempt": attempt_label,
                  "pids": {"worker": worker_pid, "supervisor": supervisor_pid, "watcher": watcher_pid},
                  "state_before_sha256": expected_state_sha256, "manifest_sha256": state["manifest_sha256"],
                  "accepted_result": result, "new_local_games_launched": 0,
                  "worker_exit_code_observed": False, "existing_exit_code_preserved": "exit_code" in attempts[0],
                  "operator_precondition": "Both local launchd services remain disabled.",
                  "status_after": PAUSED, "committed": False}
        if not commit:
            return report
        destination.parent.mkdir(exist_ok=True)
        destination.mkdir()
        for name, data in before.items():
            new_file(destination / ("before-" + name), data)
        new_file(destination / "intent.json", report)
        updated = copy.deepcopy(state)
        updated_row = updated["games"][game_label]
        updated_row.update(status="complete", result=result)
        updated_row["attempts"][-1]["accepted"] = True
        updated.update(status=PAUSED, current_label=None, worker_pid=None, supervisor_pid=None)
        updated["local_pause"] = {"reason": "user-requested-stop-after-current-game", "after_attempt": attempt_label,
                                 "new_local_launches_authorized": False, "finalizer_receipt": str(destination)}
        try:
            job.publish(updated, panel)
            actual = json.loads(bounded_bytes(job.OUTPUT / "state.json"))
            if actual != updated:
                raise RuntimeError("published state differs from finalizer result")
            report.update(committed=True, committed_unix=time.time(), counts=actual["counts"],
                          after_sha256={name: digest(bounded_bytes(job.OUTPUT / name)) for name in PUBLISHED})
            new_file(destination / "result.json", report)
        except BaseException as error:
            new_file(destination / "failure.json", {"error": f"{type(error).__name__}: {error}"[:2048],
                                                    "original_files_retained": True,
                                                    "publish_may_be_partial": True,
                                                    "automatic_retry_permitted": False})
            raise
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("game-label", "attempt-label", "expected-state-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("worker-pid", "supervisor-pid", "watcher-pid"):
        parser.add_argument("--" + name, required=True, type=int)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--receipt-dir", type=Path)
    args = vars(parser.parse_args())
    if args["commit"] and args["receipt_dir"] is None:
        parser.error("--commit requires a new --receipt-dir")
    print(json.dumps(finalize(**args), sort_keys=True))


if __name__ == "__main__":
    main()
