"""Shared fail-closed contracts for the frozen final-winner match batches."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from melee_policy.integration.frisson_policy import inspect_frisson_checkpoint  # noqa: E402
from melee_policy.integration.match_runtime import _validate_mimic_bundle  # noqa: E402


@dataclass(frozen=True, slots=True)
class FinalCheckpoint:
    name: str
    path: Path
    sha256: str
    byte_length: int
    profile: str
    step: int
    processed_target_frames: int | None
    parameter_count: int


BENCHMARK_VARIANT = os.environ.get(
    "MELEE_FINAL_BENCHMARK_VARIANT",
    "pretraining",
).strip().lower()
if BENCHMARK_VARIANT not in {"pretraining", "posttraining", "postrl"}:
    raise RuntimeError(
        "MELEE_FINAL_BENCHMARK_VARIANT must be 'pretraining' or 'posttraining'"
    )
POSTTRAINING_BENCHMARK = BENCHMARK_VARIANT in {"posttraining", "postrl"}

PRETRAINING_10M = FinalCheckpoint(
    name="Frisson-AI 10M v2 step-122064",
    path=ROOT / ".e011-cache/final-winners/remote-verification/10m-step-122064.pt",
    sha256="5a54ea4ecfa150198180dff4d06ac4fe6ecec5d9433803cc13b527e41f41d14e",
    byte_length=96_451_691,
    profile="10m",
    step=122_064,
    processed_target_frames=7_999_586_304,
    parameter_count=10_163_629,
)
PRETRAINING_75M = FinalCheckpoint(
    name="Frisson-AI 75M step-86016",
    path=ROOT / ".e011-cache/final-winners/remote-verification/75m-step-86016.pt",
    sha256="8662de0c4c0deaae2879548d6373def792dbb52adec44f2b474adb7fe155c648",
    byte_length=644_291_987,
    profile="75m",
    step=86_016,
    processed_target_frames=5_637_144_576,
    parameter_count=75_305_709,
)
POSTTRAINING_10M = FinalCheckpoint(
    name="Frisson-AI 10M post-trained step-195248",
    path=(
        ROOT
        / ".e013-cache/posttraining-winners/frisson-melee-10m-posttrained-best-val.pt"
    ),
    sha256="63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b",
    byte_length=96_452_075,
    profile="10m",
    step=195_248,
    processed_target_frames=12_795_772_928,
    parameter_count=10_163_629,
)
POSTTRAINING_75M = FinalCheckpoint(
    name="Frisson-AI 75M post-trained step-127214",
    path=(
        ROOT
        / ".e013-cache/posttraining-winners/frisson-melee-75m-posttrained-best-val.pt"
    ),
    sha256="8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077",
    byte_length=644_291_987,
    profile="75m",
    step=127_214,
    processed_target_frames=8_337_096_704,
    parameter_count=75_305_709,
)
FINAL_10M = POSTTRAINING_10M if POSTTRAINING_BENCHMARK else PRETRAINING_10M
FINAL_75M = POSTTRAINING_75M if POSTTRAINING_BENCHMARK else PRETRAINING_75M

from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS as POST_RL_CHECKPOINTS

POST_RL_FINALS = {
    profile: FinalCheckpoint(
        name=f"Frisson-AI {profile.upper()} post-RL step-{row['step']}",
        path=ROOT / row["relative_path"], sha256=row["sha256"],
        byte_length=row["byte_length"], profile=profile, step=row["step"],
        processed_target_frames=None, parameter_count=row["parameter_count"],
    ) for profile, row in POST_RL_CHECKPOINTS.items()
}
if BENCHMARK_VARIANT == "postrl":
    FINAL_10M, FINAL_75M = POST_RL_FINALS["10m"], POST_RL_FINALS["75m"]

STANDARD_TEN_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)
HEAD_TO_HEAD_SEEDS = (101, 211, 307, 401, 503)
ADDITIVE_SWEEP_SEEDS = (1201, 1301)

MIMIC_BUNDLES = (
    ("FOX", "fox", "fox-master"),
    ("FALCO", "falco", "falco"),
    ("MARTH", "marth", "marth"),
    ("SHEIK", "sheik", "sheik"),
    ("CPTFALCON", "cptfalcon", "cptfalcon"),
    ("LUIGI", "luigi", "luigi"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_checkpoint(expected: FinalCheckpoint) -> dict[str, Any]:
    path = expected.path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required final checkpoint is missing: {path}")
    observed = inspect_frisson_checkpoint(path)
    required = {
        "sha256": expected.sha256,
        "byte_length": expected.byte_length,
        "profile": expected.profile,
        "step": expected.step,
        "processed_target_frames": expected.processed_target_frames,
        "parameter_count": expected.parameter_count,
        "all_state_tensors_finite": True,
    }
    mismatches = {
        key: {"observed": observed.get(key), "required": value}
        for key, value in required.items()
        if observed.get(key) != value
    }
    safe_load = observed.get("safe_load")
    allowlisted = (
        ["builtins.frozenset"]
        if expected in {POSTTRAINING_10M, POSTTRAINING_75M, *POST_RL_FINALS.values()}
        else []
    )
    required_safe_load = {
        "weights_only": True,
        "mmap": True,
        "map_location": "cpu",
        "unsafe_globals": allowlisted,
        "allowlisted_safe_builtins": allowlisted,
    }
    if safe_load != required_safe_load:
        mismatches["safe_load"] = {
            "observed": safe_load,
            "required": required_safe_load,
        }
    if mismatches:
        raise RuntimeError(f"{expected.name} checkpoint identity mismatch: {mismatches}")
    return observed


def verify_mimic_bundle(character: str, directory_name: str) -> dict[str, Any]:
    directory = (ROOT / ".e001-cache" / "mimic" / directory_name).resolve()
    checkpoint = directory / "model.pt"
    bundle, _, _ = _validate_mimic_bundle(checkpoint, directory)
    if bundle.get("character") != character:
        raise RuntimeError(
            f"MIMIC {directory_name} character mismatch: {bundle.get('character')!r}"
        )
    checks = bundle.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RuntimeError(f"MIMIC {directory_name} bundle checks did not all pass")
    return bundle


def mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} is missing or malformed")
    return value


def resolve_artifact_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name}.path is missing or malformed")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise RuntimeError(f"{name}.path escapes the project: {resolved}")
    return resolved


def require_selected_replay(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    artifacts = mapping(payload.get("artifacts"), "summary.artifacts")
    records = artifacts.get("replays")
    if not isinstance(records, list):
        raise RuntimeError(f"seed {seed} replay records are missing or malformed")
    selected = [
        record
        for record in records
        if isinstance(record, dict) and record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1:
        raise RuntimeError(f"seed {seed} has {len(selected)} selected scoring replays")
    record = selected[0]
    validation = mapping(record.get("validation"), "selected replay validation")
    coverage = mapping(
        validation.get("required_trace_coverage"),
        "selected replay required trace coverage",
    )
    required_validation = {
        "libmelee_parseable": validation.get("libmelee_parseable") is True,
        "parsed_frames_strictly_consecutive": (
            validation.get("parsed_frames_strictly_consecutive") is True
        ),
        "trace_coverage_complete": coverage.get("complete") is True,
        "no_missing_trace_frames": coverage.get("missing_frame_count") == 0,
    }
    if not all(required_validation.values()):
        raise RuntimeError(f"seed {seed} replay validation failed: {required_validation}")
    replay_path = resolve_artifact_path(record.get("path"), "selected replay")
    if replay_path.suffix.lower() != ".slp" or not replay_path.is_file():
        raise RuntimeError(f"seed {seed} scoring replay is missing: {replay_path}")
    if replay_path.stat().st_size <= 0 or record.get("byte_length") != replay_path.stat().st_size:
        raise RuntimeError(f"seed {seed} scoring replay byte length changed")
    digest = record.get("sha256")
    if not isinstance(digest, str) or sha256_file(replay_path) != digest:
        raise RuntimeError(f"seed {seed} scoring replay SHA-256 changed")
    return record


def require_completed_match_summary(
    summary_path: Path,
    *,
    seed: int,
    opponent_model: str,
    opponent_character: str,
    player_1_character: str = "FOX",
    final_checkpoint: FinalCheckpoint = FINAL_75M,
    mimic_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    payload = mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary")
    configuration = mapping(payload.get("configuration"), "summary.configuration")
    required_configuration = {
        "seed": seed,
        "stage": "FINAL_DESTINATION",
        "require_natural_end": True,
        "player_1": {
            "model": "frisson-ai",
            "character": player_1_character,
            "port": 1,
        },
    }
    for key, required in required_configuration.items():
        if configuration.get(key) != required:
            raise RuntimeError(
                f"seed {seed} configuration {key} mismatch: {configuration.get(key)!r}"
            )
    player_2 = mapping(configuration.get("player_2"), "configuration.player_2")
    required_player_2: dict[str, Any] = {
        "model": opponent_model,
        "character": opponent_character,
        "port": 2,
    }
    if opponent_model == "cpu":
        required_player_2.update(cpu_level=9, gameplay_owner="melee-native-cpu")
    if player_2 != required_player_2:
        raise RuntimeError(f"seed {seed} player 2 contract mismatch: {player_2}")

    contract = mapping(payload.get("contract"), "summary.contract")
    selected = mapping(contract.get("selected_checkpoint"), "contract.selected_checkpoint")
    required_checkpoint = {
        "sha256": final_checkpoint.sha256,
        "byte_length": final_checkpoint.byte_length,
        "profile": final_checkpoint.profile,
        "step": final_checkpoint.step,
        "processed_target_frames": final_checkpoint.processed_target_frames,
        "parameter_count": final_checkpoint.parameter_count,
        "inspected": True,
    }
    checkpoint_mismatches = {
        key: {"observed": selected.get(key), "required": value}
        for key, value in required_checkpoint.items()
        if selected.get(key) != value
    }
    checkpoint_path = Path(str(selected.get("path", ""))).resolve()
    if checkpoint_path != final_checkpoint.path.resolve():
        checkpoint_mismatches["path"] = {
            "observed": str(checkpoint_path),
            "required": str(final_checkpoint.path.resolve()),
        }
    if checkpoint_mismatches:
        raise RuntimeError(f"seed {seed} checkpoint mismatch: {checkpoint_mismatches}")

    execution = mapping(payload.get("execution"), "summary.execution")
    required_execution = {
        "game_end_observed": execution.get("game_end_observed") is True,
        "natural_termination": execution.get("termination") == "natural-game-end",
        "processed_frames": isinstance(execution.get("processed_policy_frames"), int)
        and execution["processed_policy_frames"] > 0,
        "first_frame_minus_123": execution.get("first_game_frame") == -123,
    }
    if not all(required_execution.values()):
        raise RuntimeError(f"seed {seed} gameplay completion failed: {required_execution}")
    gate = mapping(payload.get("gate"), "summary.gate")
    gate_checks = mapping(gate.get("checks"), "summary.gate.checks")
    common_required = (
        "final_destination",
        "natural_game_end_observed",
        "entered_gameplay",
        "processed_at_least_one_frame",
        "first_policy_frame_minus_123",
        "strict_consecutive_policy_frames",
        "one_frisson_inference_per_frame",
        "one_frisson_dispatch_per_frame",
        "current_run_replay_covers_every_policy_frame",
        "exactly_one_tournament_result_replay",
        "parseable_current_run_replay",
    )
    opponent_required = (
        (
            "both_inferences_barrier_per_frame",
            "all_commands_current_frame",
            "one_mimic_inference_per_frame",
            "one_mimic_dispatch_per_frame",
        )
        if opponent_model == "mimic"
        else (
            "one_neutral_cpu_pipe_transaction_per_frame",
            "one_cpu_neutral_pipe_dispatch_per_frame",
            "cpu_replay.p2_cpu_level_9",
            "cpu_replay.p2_declared_cpu",
        )
    )
    failed_core = [
        name
        for name in (*common_required, *opponent_required)
        if gate_checks.get(name) is not True
    ]
    if failed_core:
        raise RuntimeError(f"seed {seed} core gameplay gates failed: {failed_core}")

    replay = require_selected_replay(payload, seed)
    if opponent_model == "cpu":
        cpu_contract = mapping(replay.get("cpu_contract"), "selected replay CPU contract")
        if cpu_contract.get("decision") != "pass":
            raise RuntimeError(f"seed {seed} native CPU9 replay contract failed")
    if opponent_model == "mimic":
        mimic = mapping(payload.get("mimic"), "summary.mimic")
        if mimic.get("controlled_character") != opponent_character:
            raise RuntimeError(f"seed {seed} MIMIC controlled character changed")
        if mimic.get("checkpoint_sha256") != mimic_checkpoint_sha256:
            raise RuntimeError(f"seed {seed} MIMIC checkpoint identity changed")
        identity = mapping(mimic.get("bundle_identity"), "summary.mimic.bundle_identity")
        state = mapping(identity.get("state_dictionary"), "MIMIC state dictionary")
        if state != {"strict": True, "missing_keys": [], "unexpected_keys": []}:
            raise RuntimeError(f"seed {seed} MIMIC strict state load failed: {state}")
    return payload


def next_attempt_label(base_label: str) -> str:
    output = ROOT / "artifacts" / "integration" / "frisson_ai"
    if not (output / base_label).exists():
        return base_label
    attempt = 2
    while (output / f"{base_label}-attempt{attempt}").exists():
        attempt += 1
    return f"{base_label}-attempt{attempt}"


def completed_attempt(
    base_label: str,
    validator: Any,
) -> tuple[Path, dict[str, Any]] | None:
    output = ROOT / "artifacts" / "integration" / "frisson_ai"
    candidates = [output / base_label / "summary.json"]
    candidates.extend(sorted(output.glob(f"{base_label}-attempt*/summary.json")))
    for summary in candidates:
        if not summary.is_file():
            continue
        try:
            return summary, validator(summary)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    return None


__all__ = [
    "ADDITIVE_SWEEP_SEEDS",
    "FINAL_10M",
    "FINAL_75M",
    "HEAD_TO_HEAD_SEEDS",
    "MIMIC_BUNDLES",
    "ROOT",
    "STANDARD_TEN_SEEDS",
    "completed_attempt",
    "next_attempt_label",
    "require_completed_match_summary",
    "require_selected_replay",
    "sha256_file",
    "verify_final_checkpoint",
    "verify_mimic_bundle",
]
