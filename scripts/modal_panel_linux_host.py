#!/usr/bin/env python3
"""Preparatory Linux package and Console boundary. No launch entry point.

Materialization is a separate explicit call that accepts only the v7 package.
Console construction requires an admitted Linux environment and an explicitly
shared version-hook lock. This module never launches Dolphin or a display,
opens a ROM, changes queue state, or imports a policy framework.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import stat
import tarfile
import types

import modal_panel_package_audit as audit

SCHEMA = "e011.modal-panel-linux-host.v1"
REVISION = "e7711b104b339a99385f2bb12b472d46140a7bc7"
RELEASE = "3.6.4"
FRAME_PATCH_SHA = "bb0e8885b33e6f3bb5a43e5ef936fce4a0b79459cde926a97d02aed9a7fd388e"
BUILD_PATCH_SHA = "e276165fa35ab2c8d63f0cc97df3aa643543088453f13ba23391a6902f41c682"
OUTER_PLAN_SHA = "aea6a4743a53589411c4d460467e7df6ec5e0a10d7f9e707be049f943da2a483"
ARCHIVE = (23284133, "ab2b1e04d8d3bd9beac1211299ca9e5038a06d03d36ad0efb148084c5b473c19")
MANIFEST = (55049, "c9be9b3d84401487f8f7aa03f608dfe4a7a294804a92ef2ab1038f0d8d44e700")
AUDIT_RECEIPT = (6365, "f5366740c90fa53a01d5b3aba2ac2c136a34802c814156e4e72c7aefe5b0cb05")
MEMBER_COUNT = 344
MEMBER_BYTES = 81401746
EXPANDED_TAR_BYTES = 82032640
EXECUTABLE = (19996344, "de76b9e141d35b9f81bc335a5d5a0ccb875ad43d9dcb91f27b412b0a07d17a43")
RUST_LIBRARY = (56535656, "cbee3d3750f9690cb5beb630a362656ae0ada6a47bfedefa530d4b688a0e1a69")
CONSOLE_SOURCE_SHA = "5402f432595c0edcc6db5323c5efa234a3bf4b5884772f496b5524ceb8a8c9d5"
MAX_FILESYSTEM_ENTRIES = 2000


def bound_file(path, expected, budget, cap):
    path = Path(path).absolute()
    if audit.digest(path, budget, cap) != expected:
        raise ValueError(f"pinned Linux host input differs: {path.name}")
    return path


def package_inputs(manifest_path, audit_path, budget):
    manifest_path = bound_file(manifest_path, MANIFEST, budget, audit.MAX_TEXT)
    audit_path = bound_file(audit_path, AUDIT_RECEIPT, budget, audit.MAX_TEXT)
    manifest = audit.read_json(manifest_path, budget)
    record = audit.read_json(audit_path, budget)
    if (not isinstance(record, dict) or record.get("schema") != "frisson.modal-panel-package-audit.v1"
            or record.get("status") != "package-audit-passed" or record.get("gameplay_admitted") is not False
            or record.get("outer_plan_sha256") != OUTER_PLAN_SHA
            or record.get("returned_files", {}).get("linux-dolphin-code.tar.gz") != {"bytes": ARCHIVE[0], "sha256": ARCHIVE[1]}
            or record.get("returned_files", {}).get("package-manifest.json") != {"bytes": MANIFEST[0], "sha256": MANIFEST[1]}):
        raise ValueError("v7 package-audit provenance differs")
    if not isinstance(manifest, list) or len(manifest) != MEMBER_COUNT:
        raise ValueError("exact native manifest file count required")
    expected = {}
    for row in manifest:
        budget.check()
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid native manifest row")
        name = audit.member_path(row["path"])
        if (name in expected or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= audit.MAX_ELF
                or not isinstance(row["sha256"], str) or not audit.SHA.fullmatch(row["sha256"])):
            raise ValueError("duplicate or invalid native manifest identity")
        expected[name] = row
    if sum(row["bytes"] for row in manifest) != MEMBER_BYTES:
        raise ValueError("exact native manifest aggregate required")
    for name, identity in (("dolphin-emu", EXECUTABLE), ("libslippi_rust_extensions.so", RUST_LIBRARY)):
        row = expected.get(name, {})
        if (row.get("bytes"), row.get("sha256")) != identity:
            raise ValueError("native ELF manifest identity differs")
    if expected.get("portable.txt", {}).get("bytes") != 0 or "license.txt" not in expected or not any(n.startswith("Sys/") for n in expected):
        raise ValueError("native assets/license/portable marker missing")
    return manifest, expected


def allowed_directories(expected):
    return {p.as_posix() for name in expected for p in Path(name).parents if p != Path(".")}


def verify_installed(package_root, manifest_path, audit_path, *, budget=None):
    budget = budget or audit.Budget(120)
    root = Path(package_root).absolute()
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("canonical installed package directory required")
    _, expected = package_inputs(manifest_path, audit_path, budget)
    directories = allowed_directories(expected)
    seen, expanded = set(), 0
    for index, path in enumerate(root.rglob("*")):
        budget.check()
        if index >= MAX_FILESYSTEM_ENTRIES:
            raise ValueError("installed directory-entry bound exceeded")
        info = path.lstat()
        name = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            if name not in directories:
                raise ValueError("extra installed directory")
            continue
        if not stat.S_ISREG(info.st_mode) or name not in expected:
            raise ValueError("extra, linked or nonregular installed asset")
        if info.st_nlink != 1 or info.st_mode & 0o7000:
            raise ValueError("installed hardlink or special mode forbidden")
        row = expected[name]
        if audit.digest(path, budget, audit.MAX_ELF) != (row["bytes"], row["sha256"]):
            raise ValueError("installed native file bytes changed")
        if bool(info.st_mode & 0o111) != (name == "dolphin-emu"):
            raise ValueError("only exact Dolphin ELF receives executable mode")
        if name in audit.ELFS:
            with path.open("rb") as stream:
                audit.elf_header(stream.read(64), row["bytes"], name)
        expanded += row["bytes"]
        seen.add(name)
    if seen != set(expected) or expanded != MEMBER_BYTES:
        raise ValueError("installed native file inventory differs")
    package_inputs(manifest_path, audit_path, budget)
    return {"schema": SCHEMA, "status": "linux-package-bytes-verified", "package_root": str(root),
            "target_platform": "linux_x86_64", "verification_host": {"system": platform.system(), "machine": platform.machine()},
            "release": RELEASE, "source_revision": REVISION, "frame_patch_sha256": FRAME_PATCH_SHA,
            "build_patch_sha256": BUILD_PATCH_SHA, "outer_plan_sha256": OUTER_PLAN_SHA,
            "archive_sha256": ARCHIVE[1], "manifest_sha256": MANIFEST[1], "package_audit_sha256": AUDIT_RECEIPT[1],
            "files": MEMBER_COUNT, "bytes": MEMBER_BYTES,
            "executable": {"path": str(root / "dolphin-emu"), "bytes": EXECUTABLE[0], "sha256": EXECUTABLE[1], "format": "ELF64-x86_64"},
            "rust_library": {"path": str(root / "libslippi_rust_extensions.so"), "bytes": RUST_LIBRARY[0], "sha256": RUST_LIBRARY[1]},
            "gameplay_admitted": False, "executed": False}




def native_options(replay_directory, slippi_port):
    """Exact current frisson_match._console_options mapping, without imports."""
    return {"is_dolphin": True, "tmp_home_directory": True, "copy_home_directory": False,
            "blocking_input": True, "polling_mode": True, "polling_timeout": 1.0, "online_delay": 0,
            "setup_gecko_codes": True, "fullscreen": False, "gfx_backend": "", "disable_audio": False,
            "use_exi_inputs": False, "enable_ffw": False, "save_replays": True,
            "replay_dir": str(replay_directory), "replay_monthly_folders": False, "slippi_port": slippi_port}


def code_named(code, name):
    return next(item for item in code.co_consts if isinstance(item, types.CodeType) and item.co_name == name)


def verify_console_module(module, budget):
    source = Path(module.__file__).absolute()
    if module.__name__ != "melee.console" or audit.digest(source, budget, audit.MAX_TEXT)[1] != CONSOLE_SOURCE_SHA:
        raise ValueError("pinned libmelee console source required")
    with source.open("rb") as stream:
        body = stream.read(audit.MAX_TEXT + 1)
    if len(body) > audit.MAX_TEXT:
        raise ValueError("libmelee source size bound")
    compiled = compile(body, str(source), "exec", dont_inherit=True)
    if (module.get_dolphin_version.__code__ != code_named(compiled, "get_dolphin_version")
            or module.Console.__init__.__code__ != code_named(code_named(compiled, "Console"), "__init__")):
        raise ValueError("libmelee constructor or version hook was already changed")


def create_console(package_root, manifest_path, audit_path, *, replay_directory, slippi_port,
                   console_module, version_lock, console_options):
    """Construct through the native class, with an attested direct ELF path.

    The caller supplies the shared version lock and original native options.
    The tape coordinator owns any temporary profile cleanup. No launch occurs.
    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Console host must actually be Linux x86_64")
    if type(slippi_port) is not int or not 1024 <= slippi_port <= 65535:
        raise ValueError("explicit nonprivileged Slippi UDP port required")
    root, replay = Path(package_root).absolute(), Path(replay_directory).absolute()
    if replay.resolve(strict=True) != replay or not replay.is_dir() or replay.is_relative_to(root) or root.is_relative_to(replay):
        raise ValueError("canonical separate replay directory required")
    expected_options = native_options(replay, slippi_port)
    if (set(console_options) != set(expected_options)
            or any(type(console_options[k]) is not type(v) or console_options[k] != v for k, v in expected_options.items())):
        raise ValueError("native scientific Console options changed")
    budget = audit.Budget(120)
    identity = verify_installed(root, manifest_path, audit_path, budget=budget)
    executable = root / "dolphin-emu"
    if not version_lock.acquire(blocking=False):
        raise RuntimeError("libmelee version-hook lock is already in use")
    try:
        verify_console_module(console_module, budget)
        original = console_module.get_dolphin_version
        def attested_version(path):
            if Path(path).absolute() != executable or Path(path).resolve(strict=True) != executable:
                raise ValueError("libmelee requested another executable")
            bound_file(executable, EXECUTABLE, budget, audit.MAX_ELF)
            return console_module.DolphinVersion(mainline=False, version=RELEASE, build=console_module.DolphinBuild.NETPLAY)
        console_module.get_dolphin_version = attested_version
        try:
            console = console_module.Console(path=str(executable), **dict(console_options))
        finally:
            console_module.get_dolphin_version = original
    finally:
        version_lock.release()
    if Path(console.exe_path).absolute() != executable or Path(console.exe_path).resolve(strict=True) != executable:
        raise ValueError("constructed Console executable differs")
    user = Path(console.dolphin_home_path).resolve(strict=True)
    temporary = Path(console.temp_dir).resolve(strict=True)
    if (not user.is_dir() or user != temporary / "User" or not temporary.name.startswith("libmelee_")
            or any(user.is_relative_to(p) or p.is_relative_to(user) for p in (root, replay))):
        raise ValueError("native fresh User profile is not separate")
    version = console.dolphin_version
    if version.mainline is not False or version.version != RELEASE or version.build is not console_module.DolphinBuild.NETPLAY:
        raise ValueError("constructed Console release classification differs")
    verify_installed(root, manifest_path, audit_path, budget=budget)
    identity.update(console_constructed=True, user_directory=str(user), console_source_sha256=CONSOLE_SOURCE_SHA,
                    console_options=dict(console_options), native_emulation_speed_default=1.0,
                    gui_version_process_used=False, temporary_version_hook_restored=console_module.get_dolphin_version is original)
    console._modal_panel_linux_host = types.MappingProxyType(identity)
    return console
