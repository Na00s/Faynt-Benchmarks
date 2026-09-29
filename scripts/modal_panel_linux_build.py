#!/usr/bin/env python3
"""Isolated, bounded Linux build pilot for the frozen public Slippi panel.

This helper builds code only. It never launches an emulator, reads a game image
or policy checkpoint, changes the local panel, or installs system packages.
The returned binary remains unadmitted until separate gameplay canaries pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
REVISION = "e7711b104b339a99385f2bb12b472d46140a7bc7"
UPSTREAM = "https://github.com/project-slippi/Ishiiruka.git"
FRAME_PATCH_SHA256 = "bb0e8885b33e6f3bb5a43e5ef936fce4a0b79459cde926a97d02aed9a7fd388e"
BUILD_PATCH_SHA256 = "e276165fa35ab2c8d63f0cc97df3aa643543088453f13ba23391a6902f41c682"
GITLINKS = {
    "Externals/SlippiRustExtensions": {
        "revision": "2d29e794de8497582675fb70877851f2cdd2f256",
        "url": "https://github.com/project-slippi/slippi-rust-extensions.git",
    },
    "Externals/corrosion": {
        "revision": "b1fab721655c5c4b1b08a083d3cd29f163af75d0",
        "url": "https://github.com/corrosion-rs/corrosion",
    },
}
RUST_VERSION = "1.88.0"
MAX_JOBS = 4
MAX_SECONDS = 1500
MAX_COMMAND_SECONDS = 180
MAX_LOG_BYTES = 32 * 1024**2
MAX_SOURCE_BYTES = 2 * 1024**3
MAX_BINARY_BYTES = 128 * 1024**2
MAX_PACKAGE_BYTES = 256 * 1024**2
MAX_FILES = 100_000
MIN_FREE_BYTES = 8 * 1024**3
TARGET = "dolphin-emu"
RUST_LIBRARY = "libslippi_rust_extensions.so"
OPTIONS = (
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DLINUX_LOCAL_DEV=ON",
    "-DCMAKE_BUILD_RPATH=$ORIGIN", "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
    "-DENABLE_HEADLESS=OFF", "-DDISABLE_WX=OFF", "-DENABLE_QT2=OFF",
    "-DTRY_X11=ON", "-DUSE_EGL=OFF", "-DIS_PLAYBACK=OFF",
    "-DENABLE_PCH=ON", "-DENABLE_LTO=OFF", "-DENABLE_GENERIC=OFF",
    "-DENCODE_FRAMEDUMPS=OFF", "-DENABLE_ANALYTICS=OFF",
    "-DENABLE_ALSA=OFF", "-DENABLE_AO=OFF", "-DENABLE_PULSEAUDIO=OFF",
    "-DENABLE_OPENAL=OFF", "-DENABLE_BLUEZ=OFF", "-DENABLE_EVDEV=OFF",
    "-DENABLE_SDL=OFF", "-DENABLE_LLVM=OFF", "-DUSE_UPNP=OFF",
    f"-DDOLPHIN_WC_REVISION={REVISION}",
    "-DDOLPHIN_WC_DESCRIBE=3.6.4-frisson-linux-compat-v1",
    "-DDOLPHIN_WC_BRANCH=slippi", "-DDISTRIBUTOR=FrissonResearchCompatibility",
)
APT_PACKAGES = (
    "build-essential", "cmake", "ninja-build", "pkg-config", "git",
    "ca-certificates", "curl", "libx11-dev", "libxext-dev", "libxrandr-dev",
    "libxi-dev", "libxxf86vm-dev", "libgl1-mesa-dev", "libegl1-mesa-dev",
    "libudev-dev", "libusb-1.0-0-dev", "libcurl4-openssl-dev", "libssl-dev",
    "libpng-dev", "zlib1g-dev", "liblzo2-dev", "libsfml-dev", "libsoil-dev",
    "libmbedtls-dev", "libasound2-dev", "portaudio19-dev",
    "libgtk-3-dev", "libsoup2.4-dev", "libfontconfig1-dev", "libxinerama-dev",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_plan(source: str | Path | None, patch: str | Path, output: str | Path,
               jobs: int = 4, deadline_seconds: int = 1500,
               build_patch: str | Path | None = None) -> dict:
    """Prepare the exact contract locally, without network, spawning or writes.

    ``source=None`` permits three exact public Git fetches during run_build.
    A supplied source uses read-only git archives for it and its two gitlinks.
    """
    if type(jobs) is not int or not 1 <= jobs <= MAX_JOBS:
        raise ValueError("jobs must be an integer in 1..4")
    if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= MAX_SECONDS:
        raise ValueError("deadline_seconds must be an integer in 1..1500")
    patch = Path(patch).resolve(strict=True)
    build_patch = Path(build_patch or ROOT / "patches/slippi-dolphin-linux-headless-build.patch").resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.parent.resolve() != output.parent:
        raise ValueError("output must be a new path with a canonical parent")
    if source is not None:
        source = Path(source).resolve(strict=True)
        if output == source or source in output.parents:
            raise ValueError("output must be separate from the source checkout")
    for path, expected in ((patch, FRAME_PATCH_SHA256), (build_patch, BUILD_PATCH_SHA256)):
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"patch identity differs: {path.name}")
    return {
        "schema": "frisson.modal-panel-linux-build-plan.v1",
        "source": str(source) if source is not None else None,
        "upstream": UPSTREAM, "revision": REVISION,
        "gitlinks": {path: dict(link) for path, link in GITLINKS.items()},
        "patch": str(patch), "patch_sha256": FRAME_PATCH_SHA256,
        "build_patch": str(build_patch), "build_patch_sha256": BUILD_PATCH_SHA256,
        "output": str(output), "jobs": jobs, "deadline_seconds": deadline_seconds,
        "cmake_options": list(OPTIONS), "target": TARGET,
        "rust_version": RUST_VERSION, "apt_packages": list(APT_PACKAGES),
        "limits": {"command_seconds": MAX_COMMAND_SECONDS, "log_bytes_each": MAX_LOG_BYTES,
                   "source_bytes_each": MAX_SOURCE_BYTES, "binary_bytes": MAX_BINARY_BYTES,
                   "package_bytes": MAX_PACKAGE_BYTES, "files": MAX_FILES,
                   "minimum_free_bytes": MIN_FREE_BYTES},
        "no_retry": True, "emulator_launch": False, "gameplay_admitted": False,
        "network_note": "Three explicit Git fetches if source is null; Cargo may acquire locked dependencies. No transfer-byte cap; all work has the total deadline.",
    }


class Runner:
    """Owns each child process group and stops it at the absolute deadline."""
    def __init__(self, output: Path, seconds: int, jobs: int):
        self.output = output
        self.ends = time.monotonic() + seconds
        self.commands: list[dict] = []
        self.env = {key: value for key, value in os.environ.items()
                    if key in {"PATH", "LANG", "LC_ALL", "RUSTUP_HOME", "CARGO_HOME"}}
        self.env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                         "GIT_CEILING_DIRECTORIES": str(output),
                         "GIT_TERMINAL_PROMPT": "0", "CARGO_NET_RETRY": "0",
                         "CARGO_BUILD_JOBS": str(jobs), "RUSTUP_TOOLCHAIN": RUST_VERSION,
                         "CARGO_TERM_COLOR": "never", "CMAKE_BUILD_PARALLEL_LEVEL": str(jobs)})

    def check(self) -> float:
        remaining = self.ends - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("total build deadline exhausted")
        if shutil.disk_usage(self.output).free < MIN_FREE_BYTES:
            raise RuntimeError("build free disk below 8 GiB")
        return remaining

    def run(self, command: list[str], cwd: Path, *, build: bool = False,
            binary_output: Path | None = None) -> str:
        remaining = self.check()
        row = {"command": command, "cwd": str(cwd), "started_unix": time.time()}
        self.commands.append(row)
        log = self.output / "logs" / f"{len(self.commands):03d}.log"
        row["log"] = str(log.relative_to(self.output))
        process = None
        waitid_guard = hasattr(os, "waitid")
        leader_exited = False
        try:
            with log.open("xb") as stream:
                output_stream = binary_output.open("xb") if binary_output else stream
                try:
                    process = subprocess.Popen(command, cwd=cwd, env=self.env,
                                               stdin=subprocess.DEVNULL, stdout=output_stream,
                                               stderr=stream, start_new_session=True)
                    row["pid"] = process.pid
                    until = time.monotonic() + min(remaining, remaining if build else MAX_COMMAND_SECONDS)
                    while True:
                        if waitid_guard:
                            # Keep an exited group leader unreaped until all of its
                            # children have been signalled. Its PID stays reserved.
                            observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            if observed is not None and observed.si_pid:
                                code = observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
                                leader_exited = True
                                break
                        else:
                            # Portable test path. Production run_build requires
                            # Linux waitid(WNOWAIT), including cleanup after exit.
                            code = process.poll()
                            if code is not None:
                                leader_exited = True
                                break
                        self.check()
                        if time.monotonic() >= until:
                            raise TimeoutError("command deadline exhausted")
                        if log.stat().st_size > MAX_LOG_BYTES:
                            raise RuntimeError("command log exceeded 32 MiB sampled bound")
                        if binary_output and binary_output.stat().st_size > MAX_SOURCE_BYTES:
                            raise RuntimeError("source archive exceeded 2 GiB sampled bound")
                        time.sleep(0.1)
                    row["exit_code"] = code
                    if code:
                        raise RuntimeError(f"command exited {code}; inspect {log.name}")
                finally:
                    if binary_output:
                        output_stream.close()
            if log.stat().st_size > MAX_LOG_BYTES:
                raise RuntimeError("command log exceeded 32 MiB")
            if binary_output and binary_output.stat().st_size > MAX_SOURCE_BYTES:
                raise RuntimeError("source archive exceeded 2 GiB")
            # The log bound also bounds captured identity and metadata output.
            with log.open("rb") as stream:
                return stream.read(MAX_LOG_BYTES).decode(errors="replace").strip()
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if process is not None and (waitid_guard or process.poll() is None):
                # With WNOWAIT the leader remains reserved even after it exits.
                # Always finish the process group before calling wait/reaping it.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if waitid_guard:
                    time.sleep(0.02 if leader_exited else 2)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
                else:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=3)
                row["exit_code"] = process.returncode
                row["terminated_owned_process_group"] = True
            row["unreaped_leader_group_guard"] = waitid_guard
            row["ended_unix"] = time.time()
            if log.exists():
                row.update(log_bytes=log.stat().st_size, log_sha256=digest(log))








def native_pch_compile_contract(compile_commands: list[dict], build: Path) -> dict:
    """Verify the upstream PCH target and the two previously failing consumers."""
    expected_header = str(build / "Source/CMakeFiles/pch.dir/pch.h")
    selected = {}
    expected_sources = {"pch": "/Source/PCH/pch.h",
                        "slippipad": "/Source/Core/Core/Slippi/SlippiPad.cpp",
                        "dpl2decoder": "/Source/Core/AudioCommon/DPL2Decoder.cpp"}
    for name, suffix in expected_sources.items():
        rows = [row for row in compile_commands if row["file"].endswith(suffix)]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one native PCH contract source: {name}")
        row = rows[0]
        tokens = row.get("arguments") or shlex.split(row["command"])
        expected = ["-x", "c++-header"] if name == "pch" else ["-include", expected_header]
        if sum(tokens[i:i + 2] == expected for i in range(len(tokens) - 1)) != 1:
            raise ValueError(f"native PCH compilation or forced-header contract differs: {name}")
        selected[name] = row
    return {"forced_header": expected_header, "compile_commands": selected,
            "scope": "Native upstream header compilation and forced inclusion; compiler reuse of the precompiled object is not asserted."}


def native_gui_compile_contract(compile_commands: list[dict], source: Path, build: Path) -> dict:
    """Require the native GUI path, bundled wx, and source-local include order."""
    core = source / "Source/Core"
    paths = {"main": core / "DolphinWX/Main.cpp", "x11": core / "DolphinWX/X11Utils.cpp",
             "controller": core / "InputCommon/ControllerInterface/ControllerInterface.cpp",
             "timer": core / "Core/Slippi/SlippiTimer.cpp",
             "exi": core / "Core/HW/EXI_DeviceSlippi.cpp"}
    pch = str(build / "Source/CMakeFiles/pch.dir/pch.h")
    device = str(core / "InputCommon/ControllerInterface/Device.h")
    pipes = str(core / "InputCommon/ControllerInterface/Pipes/Pipes.h")
    wx_root = str(source / "Externals/wxWidgets3")
    selected, wx_rows = {}, []
    for row in compile_commands:
        tokens = row.get("arguments") or shlex.split(row["command"])
        includes = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "-include"]
        file = row["file"]
        if file.endswith("/DolphinWX/MainNoGUI.cpp"):
            raise ValueError("native GUI contract unexpectedly includes MainNoGUI.cpp")
        if file.startswith(wx_root + "/"):
            wx_rows.append(row)
        for special, owner in ((device, paths["controller"]), (pipes, paths["controller"]),
                               ("cstring", paths["x11"])):
            if special in includes and file != str(owner):
                raise ValueError("native GUI source-local forced header leaked to another source")
        for name, path in paths.items():
            if file != str(path):
                continue
            if name in selected:
                raise ValueError(f"native GUI contract has duplicate source: {name}")
            if name in {"main", "timer", "exi"}:
                if "-DHAVE_WX=1" not in tokens or not any(wx_root + "/include" in t for t in tokens):
                    raise ValueError(f"native GUI bundled wx headers are missing: {name}")
            if name == "main" and not any("CMakeFiles/dolphin-emu.dir/" in t for t in tokens):
                raise ValueError("native GUI entry point belongs to a different target")
            if name == "x11" and includes != ["cstring"]:
                raise ValueError("native GUI X11 string include contract differs")
            if name == "controller" and includes != [pch, device, pipes]:
                raise ValueError("native GUI controller includes must be PCH then Device then Pipes")
            selected[name] = row
    if set(selected) != set(paths) or not wx_rows:
        raise ValueError("native GUI or bundled wx compilation is missing")
    return {"target": TARGET, "compile_commands": selected,
            "bundled_wx_source_count": len(wx_rows), "bundled_wx_example": wx_rows[0],
            "controller_forced_headers": [pch, device, pipes],
            "scope": "Native GUI entry and callbacks, bundled wx, and Linux source-local declaration ordering. No runtime or gameplay admission."}


def compile_failure_summary(log: Path) -> dict:
    """Extract recognized errors with one line of context under fixed caps."""
    patterns = ("FAILED:", "fatal error:", "error:", "undefined reference")
    with log.open("rb") as stream:
        data = stream.read(MAX_LOG_BYTES)
    lines = data.decode(errors="replace").splitlines()
    matches = [i for i, line in enumerate(lines) if any(pattern in line for pattern in patterns)]
    indices = sorted({j for i in matches for j in range(max(0, i - 1), min(len(lines), i + 2))})
    retained = indices[:256]
    truncated = len(indices) > 256 or log.stat().st_size > len(data)
    records = []
    for i in retained:
        truncated = truncated or len(lines[i]) > 1024
        records.append({"line": i + 1, "text": lines[i][:1024]})
    return {"log": str(log), "log_bytes": log.stat().st_size, "log_sha256": digest(log),
            "recognized_patterns": list(patterns), "recognized_matching_lines": len(matches),
            "lines": records, "truncated": truncated, "maximum_lines": 256,
            "maximum_characters_per_line": 1024,
            "scope": "Recognized textual compiler/linker errors with adjacent context; other failure formats may require the full retained log."}










