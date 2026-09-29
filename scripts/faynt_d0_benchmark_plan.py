"""Fresh, source-bound Slippi panel for the selected 632/980 D0 checkpoints.

This module prepares metadata only. It never loads a model or starts gameplay.
The original native manifest shapes remain available to the existing runners.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "faynt-d0-632-980-v1"
OUTPUT = ROOT / "artifacts/integration/frisson_ai" / RUN_ID
SCHEMA = "integration.faynt_d0_benchmark.v1"
PROFILES = ("75m", "10m")
SOURCE_FILES = {
    "expanded_slippi": (
        "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/manifest.json",
        "f80624560cc51283dc894787984de6089a42fd1e9f451ed2f1e845f4d33a6115",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_sources(root: Path = ROOT) -> tuple[dict, dict]:
    """Read exactly the audited source receipts, failing closed on byte changes."""
    sources, provenance = {}, {}
    for key, (relative, expected) in SOURCE_FILES.items():
        path = root / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen source receipt changed: {relative}")
        source = json.loads(path.read_text())
        sources[key] = source
        provenance[key] = {
            "path": relative, "sha256": actual, "byte_length": path.stat().st_size,
            "selected_field": "whole manifest",
        }
    return sources, provenance


def validate_learners(learners: dict) -> None:
    if set(learners) != {"10m", "75m"}:
        raise ValueError("exactly the 10m and 75m learner records are required")
    for profile, expected_step in (("10m", 632), ("75m", 980)):
        row = learners[profile]
        if row["profile"] != profile or row["step"] != expected_step:
            raise ValueError(f"learner profile or selected step differs: {profile}")
        path = Path(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"learner path must be repository-relative: {profile}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"invalid learner digest: {profile}")
        for key in ("byte_length", "parameter_count"):
            if type(row[key]) is not int or row[key] <= 0:
                raise ValueError(f"invalid learner {key}: {profile}")
        for key in ("delay", "delay_frames", "policy_delay_frames", "evaluation_delay_frames", "trained_delay"):
            if key in row and row[key] != 0:
                raise ValueError(f"selected learner must use D0: {profile}/{key}")
        for key, expected in (
            ("context_mode", "ring"), ("actor_context_frames", 128),
            ("action_offset_frames", 1), ("temperature", 1.0),
        ):
            if key in row and row[key] != expected:
                raise ValueError(f"selected learner deployment differs: {profile}/{key}")


def verify_learner_files(learners: dict, root: Path = ROOT) -> None:
    validate_learners(learners)
    for profile, row in learners.items():
        path = root / row["relative_path"]
        if path.stat().st_size != row["byte_length"] or sha256(path) != row["sha256"]:
            raise ValueError(f"learner checkpoint bytes differ: {profile}")


def fresh_label(source_label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_label):
        raise ValueError("source game label contains unsafe characters")
    label = f"{RUN_ID}-{source_label}"
    if len(label.encode()) > 240:
        raise ValueError("fresh game label leaves insufficient space for attempt suffix")
    return label


def _clone_games(games: list[dict]) -> tuple[list[dict], dict]:
    cloned, labels = [], {}
    for original in games:
        old = original["label"]
        if old in labels:
            raise ValueError(f"duplicate source game label: {old}")
        row = copy.deepcopy(original)
        row["label"] = labels[old] = fresh_label(old)
        if "artifact_root" in row:
            source_root = Path(row["artifact_root"])
            if source_root.name != old:
                raise ValueError("source artifact root does not match source label")
            row["artifact_root"] = str(source_root.with_name(row["label"]))
        cloned.append(row)
    return cloned, labels


def build_manifest(learners: dict, *, sources: dict | None = None, provenance: dict | None = None) -> dict:
    """Clone exactly the authorized Slippi panel with no accepted-game imports."""
    validate_learners(learners)
    if sources is None:
        sources, provenance = load_sources()
    if provenance is None:
        raise ValueError("source receipt provenance is required")
    if set(sources) != {"expanded_slippi"} or set(provenance) != {"expanded_slippi"}:
        raise ValueError("only the expanded Slippi panel is authorized")
    expanded = copy.deepcopy(sources["expanded_slippi"])
    if len(expanded["games"]) != 2624:
        raise ValueError("frozen fixed schedule counts differ")
    if len(expanded["releases"]) != 14 or len({r["sha256"] for r in expanded["releases"].values()}) != 14:
        raise ValueError("expanded opponent checkpoint inventory differs")
    if expanded["configuration"]["frisson_delay"] != 0:
        raise ValueError("source learner timing differs from the authorized D0 schedule")
    expanded["games"], expanded_labels = _clone_games(expanded["games"])
    expanded["frisson_checkpoints"] = copy.deepcopy(learners)
    expanded["experiment_id"] = RUN_ID
    expanded["configuration"]["gameplay_order"] = "all 75M, then all 10M; supported before transfer"

    fixed_games = []
    for old, new in zip(sources["expanded_slippi"]["games"], expanded["games"], strict=True):
        fixed_games.append({
            "component": "expanded_slippi", "source_label": old["label"], "label": new["label"],
            "learner_profiles": [new["profile"]], "save_slp": True, "save_video": False, "game": new,
        })
    counts = Counter(profile for game in fixed_games for profile in game["learner_profiles"])
    if counts != {"10m": 1312, "75m": 1312} or len(fixed_games) != 2624:
        raise ValueError("fixed physical-game or per-learner appearance counts differ")
    if len({row["label"] for row in fixed_games}) != 2624:
        raise ValueError("fresh fixed-game labels are not unique")
    for row in expanded["games"]:
        if row["save_slp"] is not True or row["save_video"] is not False:
            raise ValueError("expanded recording settings differ")
    for profile in PROFILES:
        if Counter(g["block"] for g in expanded["games"] if g["profile"] == profile) != {
            "supported-mirror": 244, "extended-roster": 534, "forced-mirror": 534,
        }:
            raise ValueError(f"expanded coverage differs: {profile}")
    manifest = {
        "schema": SCHEMA, "run_id": RUN_ID,
        "authorization": {
            "included_components": ["expanded_slippi"],
            "excluded_components": ["legacy_external", "pretraining_ancestors", "head_to_head", "hal_bo5"],
            "hard_spend_limit_usd": 150,
            "compute_policy": "standard-preemptible",
            "budget_stopping_rule": "runner must stop before cumulative run cost exceeds the hard cap",
        },
        "experiment": {
            "changed_variable": "focal learner checkpoint identity",
            "control": "frozen preceding expanded Slippi matchups, opponents, seeds, ports, stages and timing",
            "hypothesis": "the selected newer checkpoints change paired closed-loop outcomes",
            "metrics": ["wins", "losses", "stocks", "per-character and opponent slices", "paired changes"],
            "gate": "fresh checkpoint and deployment acceptance, then fixed Slippi schedule within the hard budget",
            "execution_authority": "preparation only; gameplay requires the separate authorized runner",
        },
        "source_manifests": copy.deepcopy(provenance), "learners": copy.deepcopy(learners),
        "expanded_slippi": expanded,
        "fixed_games": fixed_games,
        "old_to_new_labels": {"expanded_slippi": expanded_labels},
        "recording": {"save_slp": True, "save_video": False, "canaries_count_as_games": False},
        "counts": {
            "expanded_slippi_physical": 2624, "fixed_physical": 2624,
            "fixed_appearances_per_learner": dict(counts),
            "supported_mirrors_per_learner": 244,
            "extended_roster_per_learner": 534,
            "forced_mirrors_per_learner": 534,
        },
        "imported_completed_games": [],
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    return manifest


def game_index(manifest: dict) -> dict[str, dict]:
    rows = manifest["fixed_games"]
    index = {row["label"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate fixed-game label")
    return index


def lookup_game(manifest: dict, label: str) -> dict:
    """Return a fresh wrapper containing the native row and its component."""
    return copy.deepcopy(game_index(manifest)[label])


def fresh_state(manifest: dict) -> dict:
    return {
        "schema": SCHEMA, "run_id": manifest["run_id"], "status": "prepared",
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "games": {}, "attempts": [], "imported_completed_games": [],
    }


def _write_once(path: Path, value: dict, *, compare_existing: bool = True) -> None:
    """Creation is exclusive; preparing again cannot reset progress or receipts."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if compare_existing and path.read_bytes() != payload:
            raise ValueError(f"existing frozen output differs: {path}")


def write_prepared(manifest: dict, output: Path = OUTPUT) -> dict:
    if output.name != RUN_ID:
        raise ValueError(f"output directory must use the fresh run ID {RUN_ID}")
    if canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_content_sha256"}) != manifest["manifest_content_sha256"]:
        raise ValueError("prepared manifest content digest differs")
    # Check existing immutable files before creating anything new.
    payloads = {
        "manifest.json": manifest,
        "expanded-slippi/manifest.json": manifest["expanded_slippi"],
    }
    for relative, value in payloads.items():
        path = output / relative
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"existing frozen output differs: {path}")
    state_path = output / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("run_id") != RUN_ID or state.get("manifest_content_sha256") != manifest["manifest_content_sha256"]:
            raise ValueError("existing progress state belongs to a different manifest")
    for relative, value in payloads.items():
        _write_once(output / relative, value)
    _write_once(output / "state.json", fresh_state(manifest), compare_existing=False)
    return {
        "output": str(output), "manifest_sha256": sha256(output / "manifest.json"),
        "hard_spend_limit_usd": manifest["authorization"]["hard_spend_limit_usd"],
        "compute_policy": manifest["authorization"]["compute_policy"], **manifest["counts"],
    }
