#!/usr/bin/env python3
"""One explicitly approved, source-only Modal Linux Dolphin build pilot.

Preparation and imports perform no network work. --run requires the exact frozen
plan SHA and records an irreversible one-submission intent before touching Modal.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pickle
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
REMOTE = Path("/opt/pilot")
WORK = Path("/work")
SCHEMA = "frisson.modal-panel-build-pilot.v1"
SDK_VERSION = "1.5.4"
APP_NAME = "frisson-slippi-linux-code-pilot-v1"
PROFILE = "frisson"
ENVIRONMENT = "main"
IMAGE = "ubuntu:22.04"
PYTHON_VERSION = "3.12"
UPLOAD_CAP = 1024**2
SESSION_SECONDS = 2700
FUNCTION_SECONDS = 1800
WORK_SECONDS = 1740
BOOTSTRAP_SECONDS = 300
BUILD_SECONDS = 1500
RUST_URL = "https://static.rust-lang.org/dist/rust-1.88.0-x86_64-unknown-linux-gnu.tar.xz"
RUST_SHA256 = "7b5437c1d18a174faae253a18eac22c32288dccfc09ff78d5ee99b7467e21bca"
RUST_ARCHIVE_CAP = 512 * 1024**2
RUST_UNPACK_CAP = 2 * 1024**3
RUST_FILE_CAP = 100_000
ARCHIVE_CAP = 256 * 1024**2
TEXT_CAP = 8 * 1024**2
LOG_TAIL_CAP = 64 * 1024
CHUNK_BYTES = 60 * 1024
FRAME_BYTES = 64 * 1024
SAFE_NAME = re.compile(r"(?:receipt\.json|plan\.json|package-manifest\.json|linux-dolphin-code\.tar\.gz|(?:bootstrap|logs)/[0-9]{3}\.log)\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def create_once(path: Path, value: object) -> None:
    payload = canonical(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def regular(path: Path, cap: int) -> int:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("path must be canonical and contain no symlinks")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
        raise ValueError("expected bounded regular file")
    return info.st_size


def constants(path: Path) -> dict:
    """Read only literal builder constants, without importing executable source."""
    wanted = {"APT_PACKAGES", "RUST_VERSION", "REVISION", "UPSTREAM", "FRAME_PATCH_SHA256", "BUILD_PATCH_SHA256"}
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                result[name] = ast.literal_eval(node.value)
    if set(result) != wanted or result["RUST_VERSION"] != "1.88.0":
        raise ValueError("builder literal contract missing or wrong Rust version")
    packages = result["APT_PACKAGES"]
    if not isinstance(packages, tuple) or not 1 <= len(packages) <= 64 or any(
            not isinstance(x, str) or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", x) for x in packages):
        raise ValueError("invalid bounded apt dependency allowlist")
    result["APT_PACKAGES"] = list(packages)
    return result


def source_paths() -> list[Path]:
    return [SELF, ROOT / "scripts/modal_panel_linux_build.py",
            ROOT / "patches/slippi-dolphin-two-pipe-frame-sync.patch",
            ROOT / "patches/slippi-dolphin-linux-headless-build.patch"]


def plan_definition(output: str, rows: list[dict], metadata: dict) -> dict:
    for row, expected in ((rows[2], metadata["FRAME_PATCH_SHA256"]), (rows[3], metadata["BUILD_PATCH_SHA256"])):
        if row["sha256"] != expected:
            raise ValueError("builder patch identity mismatch")
    plan = {
        "schema": SCHEMA, "output": output, "sources": rows, "builder_constants": metadata,
        "modal": {"sdk": SDK_VERSION, "profile": PROFILE, "environment": ENVIRONMENT,
                  "app_name": APP_NAME, "image": IMAGE, "python": PYTHON_VERSION,
                  "nonpreemptible": True},
        "limits": {"files": 5, "upload_bytes": UPLOAD_CAP, "cpu_request_limit": [4, 4],
                   "memory_request_limit_mib": [16384, 16384], "containers": 1,
                   "session_seconds": SESSION_SECONDS, "function_seconds": FUNCTION_SECONDS,
                   "process_seconds": WORK_SECONDS, "bootstrap_seconds": BOOTSTRAP_SECONDS,
                   "build_seconds": BUILD_SECONDS, "rust_archive_bytes": RUST_ARCHIVE_CAP,
                   "rust_unpack_bytes": RUST_UNPACK_CAP, "return_archive_bytes": ARCHIVE_CAP,
                   "return_text_bytes": TEXT_CAP, "stream_chunk_bytes": CHUNK_BYTES,
                   "stream_frame_bytes": FRAME_BYTES},
        "transport": "One native remote_gen invocation, early call-ID frame, bounded inline chunks; no configured generator retry policy.",
        "retry_semantics": {"client_resubmissions": 0, "user_exception_retry_policy": None,
                            "platform_restarts_possible": True,
                            "container_limit_scope": "simultaneous containers",
                            "deadline_scope": "One immutable expires_unix shared by all physical executions."},
        "rust": {"url": RUST_URL, "sha256": RUST_SHA256, "checksum_url": RUST_URL + ".sha256",
                 "metadata_evidence": "Root read the official HTTP 200 checksum line; raw response SHA was not captured."},
        "network_note": "Public Ubuntu packages, exact public source git fetches, locked Cargo dependencies and one checksum-bound official Rust archive. Rust body is capped; apt/git/Cargo traffic has deadlines and no aggregate byte counter.",
        "billing_note": "One nonpreemptible CPU Function with hard resource tuples and the documented 3x CPU/memory price multiplier; base image setup is additional. Expected below $2 subject to root's current pricing review. No SDK dollar stop exists.",
        "no_rom_models_secrets_volumes": True, "no_gameplay": True, "no_auto_resubmit": True,
    }
    if sum(r["bytes"] for r in rows) + len(canonical(plan)) > UPLOAD_CAP:
        raise ValueError("five-file upload exceeds 1 MiB")
    return plan


def make_plan(output: Path) -> dict:
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("preparation requires a new output with canonical existing parent")
    rows = []
    for path in source_paths():
        rows.append({"path": str(path), "name": path.name, "bytes": regular(path, UPLOAD_CAP), "sha256": digest(path)})
    return plan_definition(str(output), rows, constants(source_paths()[1]))


def verify_plan(plan_path: Path, expected: str, *, remote: bool = False) -> dict:
    regular(plan_path, UPLOAD_CAP)
    if not SHA.fullmatch(expected) or digest(plan_path) != expected:
        raise ValueError("plan SHA mismatch")
    plan = json.loads(plan_path.read_text())
    # Rebuild the exact contract without allowing preparation output reuse.
    if plan.get("schema") != SCHEMA or not isinstance(plan.get("sources"), list) or len(plan["sources"]) != 4:
        raise ValueError("invalid pilot contract")
    paths = source_paths() if not remote else [REMOTE / x.name for x in source_paths()]
    total = regular(plan_path, UPLOAD_CAP)
    for row, path in zip(plan["sources"], paths, strict=True):
        if set(row) != {"path", "name", "bytes", "sha256"} or row["name"] != path.name:
            raise ValueError("upload identity differs from exact allowlist")
        if not remote and row["path"] != str(path):
            raise ValueError("local source path differs")
        count = regular(path, UPLOAD_CAP)
        if row["bytes"] != count or row["sha256"] != digest(path):
            raise ValueError("source bytes changed")
        total += count
    if total > UPLOAD_CAP or plan["builder_constants"] != constants(paths[1]):
        raise ValueError("upload cap or builder constants changed")
    if plan_definition(plan["output"], plan["sources"], constants(paths[1])) != plan:
        raise ValueError("plan does not match exact frozen constants")
    return plan


def remaining(deadline: float) -> float:
    if not isinstance(deadline, (float, int)) or not math.isfinite(deadline):
        raise ValueError("finite absolute deadline required")
    value = deadline - time.time()
    if value <= 0:
        raise TimeoutError("pilot absolute deadline exhausted")
    return value


def stop_owned(process) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=7)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=3)


def command(argv: list[str], log: Path, deadline: float, env: dict) -> dict:
    started = time.time()
    process = None
    try:
        with log.open("xb") as stream:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                       env=env, cwd=WORK, start_new_session=True)
            while process.poll() is None:
                remaining(deadline)
                if log.stat().st_size > TEXT_CAP:
                    raise ValueError("command log exceeds sampled 8 MiB bound")
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f"command failed with {process.returncode}: {argv[0]}")
    finally:
        if process is not None:
            stop_owned(process)
    if log.stat().st_size > TEXT_CAP:
        raise ValueError("command log exceeds 8 MiB")
    return {"argv": argv, "started_unix": started, "ended_unix": time.time(), "sha256": digest(log), "bytes": log.stat().st_size}








def collect_files(build: Path, bootstrap: Path) -> list[dict]:
    output = []
    text_bytes = 0
    candidates = [(build / x, x) for x in ("receipt.json", "plan.json", "package-manifest.json", "linux-dolphin-code.tar.gz")]
    candidates += [(p, "logs/" + p.name) for p in sorted((build / "logs").glob("[0-9][0-9][0-9].log"))]
    candidates += [(p, "bootstrap/" + p.name) for p in sorted(bootstrap.glob("[0-9][0-9][0-9].log"))]
    for path, name in candidates:
        if not path.exists():
            continue
        cap = ARCHIVE_CAP if name.endswith(".tar.gz") else 32 * 1024**2
        size = regular(path, cap)
        if not SAFE_NAME.fullmatch(name) or len(output) >= 100:
            raise ValueError("unexpected output file")
        with path.open("rb") as stream:
            truncated = name.endswith(".log") and size > LOG_TAIL_CAP
            if truncated:
                stream.seek(size - LOG_TAIL_CAP)
            body = stream.read(cap + 1)
        if not name.endswith(".tar.gz"):
            text_bytes += len(body)
            if text_bytes > TEXT_CAP:
                raise ValueError("returned text exceeds 8 MiB")
        output.append({"name": name, "body": body, "sha256": hashlib.sha256(body).hexdigest(),
                       "original_bytes": size, "original_sha256": digest(path), "truncated_tail": truncated})
    return output


def print_failure(build: Path, bootstrap: Path, error: str) -> None:
    """Emit a bounded diagnosis independently of final artifact transport."""
    detail = {"stage": "failure-detail", "error": error[:1024]}
    try:
        receipt_path = build / "receipt.json"
        logs = sorted(bootstrap.glob("[0-9][0-9][0-9].log"))
        if receipt_path.exists() and regular(receipt_path, TEXT_CAP) <= TEXT_CAP:
            receipt = json.loads(receipt_path.read_text())
            detail["helper_error"] = str(receipt.get("error", ""))[:1024]
            commands = receipt.get("commands", [])
            if isinstance(commands, list) and commands:
                last = commands[-1]
                detail["last_command"] = str(last.get("command", ""))[:1024]
                relative = last.get("log", "")
                if isinstance(relative, str) and re.fullmatch(r"logs/[0-9]{3}\.log", relative):
                    logs.append(build / relative)
        if logs:
            log = logs[-1]
            size = regular(log, 32 * 1024**2)
            with log.open("rb") as stream:
                stream.seek(max(0, size - 4096))
                detail["log_tail"] = stream.read(4096).decode(errors="replace")
            detail["log_name"] = log.name
    except Exception as extra:
        detail["diagnostic_error"] = str(extra)[:256]
    print(json.dumps(detail), flush=True)




def checked_frame(frame: dict) -> dict:
    # The SDK's native generator channel has a default 2 MiB inline threshold.
    # This conservative 64 KiB serialized envelope leaves substantial margin.
    if not isinstance(frame, dict) or len(pickle.dumps(frame, protocol=4)) > FRAME_BYTES:
        raise ValueError("stream frame exceeds 64 KiB serialized bound")
    return frame


def output_frames(result: dict, first_sequence: int = 1):
    sequence = first_sequence
    for row in result["files"]:
        metadata = {key: value for key, value in row.items() if key != "body"}
        metadata["bytes"] = len(row["body"])
        yield checked_frame({"kind": "file", "seq": sequence, "file": metadata})
        sequence += 1
        for offset in range(0, len(row["body"]), CHUNK_BYTES):
            data = row["body"][offset:offset + CHUNK_BYTES]
            yield checked_frame({"kind": "data", "seq": sequence, "offset": offset, "body": data})
            sequence += 1
        yield checked_frame({"kind": "end-file", "seq": sequence, "name": row["name"], "sha256": row["sha256"]})
        sequence += 1
    terminal = {key: value for key, value in result.items() if key != "files"}
    yield checked_frame({"kind": "done", "seq": sequence, "result": terminal})






def save_result(output: Path, result: dict, plan_sha256: str, *, files_already_saved: bool = False) -> None:
    if result.get("schema") != SCHEMA or result.get("plan_sha256") != plan_sha256:
        raise ValueError("returned receipt identity mismatch")
    rows = result.get("files")
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError("invalid returned files")
    total_text = 0
    seen = set()
    validated = []
    for row in rows:
        name, body = row["name"], row["body"]
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in seen or type(body) is not bytes:
            raise ValueError("unsafe or repeated returned file")
        seen.add(name)
        cap = ARCHIVE_CAP if name.endswith(".tar.gz") else TEXT_CAP
        if len(body) > cap or hashlib.sha256(body).hexdigest() != row["sha256"]:
            raise ValueError("returned file size/hash mismatch")
        if not name.endswith(".tar.gz"):
            total_text += len(body)
        if total_text > TEXT_CAP:
            raise ValueError("returned text aggregate exceeded")
        validated.append((row, output / "returned" / name))
    archive_disposition = "absent"
    by_name = {row["name"]: row for row in rows}
    if result.get("status") == "code-build-passed":
        required = {"receipt.json", "plan.json", "package-manifest.json", "linux-dolphin-code.tar.gz"}
        if not required.issubset(by_name):
            raise ValueError("successful build lacks required package evidence")
        build_receipt = json.loads(by_name["receipt.json"]["body"])
        package = build_receipt.get("package", {})
        if (build_receipt.get("status") != "code-build-passed"
                or build_receipt.get("plan_sha256") != by_name["plan.json"]["sha256"]
                or package.get("archive_bytes") != len(by_name["linux-dolphin-code.tar.gz"]["body"])
                or package.get("archive_sha256") != by_name["linux-dolphin-code.tar.gz"]["sha256"]
                or package.get("manifest_sha256") != by_name["package-manifest.json"]["sha256"]
                or package.get("gameplay_admitted") is not False):
            raise ValueError("successful build package/plan identity mismatch")
    if "linux-dolphin-code.tar.gz" in by_name:
        archive_disposition = "retained-unextracted; archive members require bounded validation before execution"
    for row, path in validated:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.resolve() != path.parent or path.is_symlink():
            raise ValueError("returned destination traverses link")
        if files_already_saved:
            if regular(path, ARCHIVE_CAP) != len(row["body"]) or digest(path) != row["sha256"]:
                raise ValueError("retained streamed file changed")
        else:
            with path.open("xb") as stream:
                stream.write(row["body"])
    receipt = {key: value for key, value in result.items() if key != "files"}
    receipt["files"] = [{key: value for key, value in row.items() if key != "body"} for row in rows]
    receipt["archive_disposition"] = archive_disposition
    receipt["gameplay_admitted"] = False
    create_once(output / "result.json", receipt)


def checked_execution(value: object) -> dict:
    if (not isinstance(value, dict) or set(value) != {"task_id", "execution_id"}
            or not isinstance(value["task_id"], str) or not re.fullmatch(r"ta-[A-Za-z0-9_-]{1,128}", value["task_id"])
            or not isinstance(value["execution_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["execution_id"])):
        raise ValueError("invalid bounded worker execution identity")
    return dict(value)


def receive_stream(output: Path, frames, plan_sha256: str, app_id: str, expires_unix: float) -> dict:
    """Accept one sequence, keeping every incomplete file and progress receipt."""
    sequence = 0
    current = None
    stream = None
    seen = set()
    returned = []
    text_reserved = 0
    terminal = None
    execution = None
    try:
        for frame in frames:
            remaining(expires_unix)
            checked_frame(frame)
            if type(frame.get("seq")) is not int or frame["seq"] != sequence or terminal is not None:
                raise ValueError("stream order/terminal mismatch")
            sequence += 1
            kind = frame.get("kind")
            if sequence == 1:
                if (kind != "hello" or frame.get("plan_sha256") != plan_sha256
                        or not isinstance(frame.get("call_id"), str)
                        or not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", frame["call_id"])):
                    raise ValueError("first stream frame must carry exact call identity")
                execution = checked_execution(frame.get("execution"))
                create_once(output / "call.json", {"app_id": app_id, "call_id": frame["call_id"],
                                                  "execution": execution, "expires_unix": expires_unix,
                                                  "recorded_unix": time.time(), "transport": "native-generator"})
                continue
            if kind == "file":
                row = frame["file"]
                if current is not None or not isinstance(row, dict) or len(seen) >= 100:
                    raise ValueError("unexpected file header")
                if set(row) != {"name", "bytes", "sha256", "original_bytes", "original_sha256", "truncated_tail"}:
                    raise ValueError("unexpected streamed file metadata")
                name = row.get("name")
                count = row.get("bytes")
                if (not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in seen
                        or type(count) is not int or count < 0 or not isinstance(row.get("sha256"), str)
                        or not SHA.fullmatch(row["sha256"])
                        or type(row["original_bytes"]) is not int or row["original_bytes"] < count
                        or not isinstance(row["original_sha256"], str) or not SHA.fullmatch(row["original_sha256"])
                        or type(row["truncated_tail"]) is not bool):
                    raise ValueError("invalid streamed file identity")
                cap = ARCHIVE_CAP if name.endswith(".tar.gz") else TEXT_CAP
                if count > cap:
                    raise ValueError("streamed file exceeds cap")
                if not name.endswith(".tar.gz"):
                    text_reserved += count
                    if text_reserved > TEXT_CAP:
                        raise ValueError("streamed text aggregate exceeds cap")
                final = output / "returned" / name
                partial = final.with_name(final.name + ".partial")
                final.parent.mkdir(parents=True, exist_ok=True)
                if final.parent.resolve() != final.parent or final.exists() or final.is_symlink() or partial.is_symlink():
                    raise ValueError("unsafe or existing stream destination")
                stream = partial.open("xb")
                current = {"row": dict(row), "partial": partial, "final": final, "written": 0, "hash": hashlib.sha256()}
                seen.add(name)
            elif kind == "data":
                data = frame.get("body")
                if (current is None or type(data) is not bytes or not 0 < len(data) <= CHUNK_BYTES
                        or type(frame.get("offset")) is not int or frame["offset"] != current["written"]
                        or current["written"] + len(data) > current["row"]["bytes"]):
                    raise ValueError("invalid stream chunk")
                if stream.write(data) != len(data):
                    raise OSError("short stream write")
                current["written"] += len(data)
                current["hash"].update(data)
            elif kind == "end-file":
                if (current is None or frame.get("name") != current["row"]["name"]
                        or frame.get("sha256") != current["row"]["sha256"]
                        or current["written"] != current["row"]["bytes"]
                        or current["hash"].hexdigest() != current["row"]["sha256"]):
                    raise ValueError("streamed final file size/hash mismatch")
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                stream = None
                current["partial"].rename(current["final"])
                progress = output / "stream-progress"
                progress.mkdir(exist_ok=True)
                if progress.resolve() != progress:
                    raise ValueError("unsafe stream progress directory")
                create_once(progress / f"{len(returned):03d}.json", current["row"])
                returned.append(current["row"])
                current = None
            elif kind == "done":
                if current is not None or not isinstance(frame.get("result"), dict):
                    raise ValueError("stream ended with an incomplete file")
                terminal = frame["result"]
                if checked_execution(terminal.get("execution")) != execution:
                    raise ValueError("worker execution changed within the returned stream")
            else:
                raise ValueError("unknown stream frame kind")
        if terminal is None or sequence < 2:
            raise ValueError("stream ended without a terminal receipt")
        terminal = dict(terminal)
        terminal["files"] = []
        for row in returned:
            path = output / "returned" / row["name"]
            if regular(path, ARCHIVE_CAP) != row["bytes"]:
                raise ValueError("streamed output identity changed")
            terminal["files"].append({key: value for key, value in row.items() if key != "bytes"} | {"body": path.read_bytes()})
        save_result(output, terminal, plan_sha256, files_already_saved=True)
        return terminal
    finally:
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()


@contextmanager
def local_deadline(seconds: float):
    def expired(_sig, _frame):
        raise TimeoutError("local 2700-second pilot deadline exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)






