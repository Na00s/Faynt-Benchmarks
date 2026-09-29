"""Bindings for an existing benchmark image and caller-supplied local sources."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


def image_reference() -> str:
    value = os.environ.get("FAYNT_BENCHMARK_IMAGE", "")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value):
        raise ValueError("Set FAYNT_BENCHMARK_IMAGE to your prepared registry image pinned by @sha256.")
    return value


def image_environment() -> dict[str, str]:
    return {"FAYNT_PREPARED_SLIPPI_SOURCE": os.environ.get(
        "FAYNT_PREPARED_SLIPPI_SOURCE", "/opt/prepared/slippi-ai")}


def require_prepared_runtime(work: Path) -> Path:
    python = work / "venv/bin/python"
    package = work / "package"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Provide the pinned existing Python environment at {python}.")
    if not package.is_dir() or package.resolve(strict=True) != package or not (package / "dolphin-emu").is_file():
        raise FileNotFoundError(f"Provide the previously verified emulator package at {package}.")
    return python


def slippi_source(revision: str) -> Path:
    source = Path(image_environment()["FAYNT_PREPARED_SLIPPI_SOURCE"]).absolute()
    if not source.is_dir() or source.resolve(strict=True) != source:
        raise FileNotFoundError(f"Provide a canonical local Slippi-AI checkout at {source}.")
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True, timeout=15)
    if result.stdout.strip() != revision:
        raise ValueError("The supplied Slippi-AI source revision differs from the frozen benchmark.")
    changed = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, check=True, timeout=15)
    if changed.stdout.strip():
        raise ValueError("The supplied Slippi-AI checkout contains tracked changes.")
    return source
