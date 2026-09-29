from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _run_variant(source: str, *, variant: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MELEE_FINAL_BENCHMARK_VARIANT"] = variant
    python_path = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((python_path, inherited)) if inherited else python_path
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _snapshot(variant: str) -> dict[str, Any]:
    completed = _run_variant(
        r'''
import json
from collections import Counter
from types import SimpleNamespace

import final_winner_batch_common as common
import run_final_winner_benchmark as runner
import run_frisson_dual_oneoff as dual
from melee_policy.integration import final_benchmark_suite as suite


def argument(command, name):
    return command[command.index(name) + 1]


commands = {spec.label: suite.command_for_game(spec, "PYTHON") for spec in suite.SCHEDULE}
for spec in suite.SCHEDULE:
    command = commands[spec.label]
    assert spec.label in command
    assert str(spec.seed) in command
    assert "--save-video" not in command
    assert "--save-slp" not in command
    if spec.kind not in {"head-to-head", "ancestor"}:
        expected = (
            suite.TEN_M_CHECKPOINT
            if spec.frisson_profile == "10m"
            else suite.SEVENTY_FIVE_M_CHECKPOINT
        )
        assert argument(command, "--p1-checkpoint") == str(expected)
        assert argument(command, "--p1-character") == spec.player_1_character

selected = runner._selected_specs(SimpleNamespace(label=None, group=None))
ancestors = []
for spec in suite.SCHEDULE:
    if spec.kind != "ancestor":
        continue
    command = commands[spec.label]
    ancestors.append(
        {
            "ordinal": spec.ordinal,
            "profile": spec.frisson_profile,
            "seed": spec.seed,
            "swap": spec.head_to_head_swap_ports,
            "matchup_class": spec.matchup_class,
            "characters": [spec.player_1_character, spec.opponent_character],
            "p1_checkpoint": argument(command, "--p1-checkpoint"),
            "p2_checkpoint": argument(command, "--p2-checkpoint"),
            "p1_contract": argument(command, "--p1-contract"),
            "p2_contract": argument(command, "--p2-contract"),
            "video_players": suite.expected_video_players(spec),
        }
    )

head_to_head = []
for spec in suite.SCHEDULE:
    if spec.kind != "head-to-head":
        continue
    command = commands[spec.label]
    head_to_head.append(
        {
            "seed": spec.seed,
            "swap": spec.head_to_head_swap_ports,
            "p1_checkpoint": argument(command, "--p1-checkpoint"),
            "p2_checkpoint": argument(command, "--p2-checkpoint"),
            "p1_contract": argument(command, "--p1-contract"),
            "p2_contract": argument(command, "--p2-contract"),
        }
    )

manifest = suite.schedule_manifest()
parser_defaults = dual._parser().parse_args([])
print(
    json.dumps(
        {
            "variant": suite.BENCHMARK_VARIANT,
            "runner_variant": runner.BENCHMARK_VARIANT,
            "suite_id": suite.SUITE_ID,
            "suite_schema": suite.SUITE_SCHEMA_VERSION,
            "state_schema": suite.STATE_SCHEMA_VERSION,
            "game_count": suite.GAME_COUNT,
            "schedule_length": len(suite.SCHEDULE),
            "schedule_sha256": manifest["schedule_sha256"],
            "labels": [spec.label for spec in suite.SCHEDULE],
            "artifact_roots": [str(spec.artifact_root) for spec in suite.SCHEDULE],
            "schedule_semantics": [
                {
                    "ordinal": spec.ordinal,
                    "group": spec.group,
                    "kind": spec.kind,
                    "seed": spec.seed,
                    "frisson_profile": spec.frisson_profile,
                    "matchup_class": spec.matchup_class,
                    "player_1_character": spec.player_1_character,
                    "opponent_character": spec.opponent_character,
                    "opponent_slug": spec.opponent_slug,
                    "opponent_evaluation_mode": spec.opponent_evaluation_mode,
                    "mimic_directory": spec.mimic_directory,
                    "head_to_head_swap_ports": spec.head_to_head_swap_ports,
                }
                for spec in suite.SCHEDULE
            ],
            "groups": Counter(spec.group for spec in suite.SCHEDULE),
            "kinds": Counter(spec.kind for spec in suite.SCHEDULE),
            "execution_partitions": [
                "head-to-head" if spec.kind == "head-to-head" else spec.frisson_profile
                for spec in selected
            ],
            "execution_ordinals": [spec.ordinal for spec in selected],
            "ancestors": ancestors,
            "head_to_head": head_to_head,
            "manifest_final_winners": manifest["final_winners"],
            "manifest_ancestors": manifest.get("pretraining_ancestors"),
            "initial_state_games": len(suite.initial_state()["games"]),
            "legacy_imports": suite.import_semantically_valid_v1_artifacts(),
            "ten_checkpoint": str(suite.TEN_M_CHECKPOINT),
            "seventy_five_checkpoint": str(suite.SEVENTY_FIVE_M_CHECKPOINT),
            "ten_identity": {
                "sha256": suite.TEN_M_IDENTITY.sha256,
                "bytes": suite.TEN_M_IDENTITY.byte_length,
                "step": suite.TEN_M_STEP,
                "frames": suite.TEN_M_FRAMES,
            },
            "seventy_five_identity": {
                "sha256": suite.SEVENTY_FIVE_M_IDENTITY.sha256,
                "bytes": suite.SEVENTY_FIVE_M_IDENTITY.byte_length,
                "step": suite.SEVENTY_FIVE_M_STEP,
                "frames": suite.SEVENTY_FIVE_M_FRAMES,
            },
            "dual_defaults": {
                "p1_checkpoint": str(parser_defaults.p1_checkpoint),
                "p2_checkpoint": str(parser_defaults.p2_checkpoint),
                "p1_contract": parser_defaults.p1_contract,
                "p2_contract": parser_defaults.p2_contract,
            },
            "common_finals": {
                "10m": common.FINAL_10M.name,
                "75m": common.FINAL_75M.name,
            },
        },
        sort_keys=True,
    )
)
''',
        variant=variant,
    )
    return json.loads(completed.stdout)


def test_pretraining_variant_remains_the_exact_persisted_309_game_suite() -> None:
    snapshot = _snapshot("pretraining")

    assert snapshot["variant"] == snapshot["runner_variant"] == "pretraining"
    assert snapshot["suite_id"] == "final-winners-expanded-step122064-step86016-v5"
    assert snapshot["suite_schema"] == "integration.final_winner_benchmark_suite.v5"
    assert snapshot["state_schema"] == "integration.final_winner_benchmark_state.v5"
    assert snapshot["game_count"] == snapshot["schedule_length"] == 309
    assert snapshot["schedule_sha256"] == (
        "6580bcf88d42f696ca8f882d5be8d008e6050c83600fca95b8f0c90e22d7c1e7"
    )
    assert snapshot["kinds"] == {
        "head-to-head": 5,
        "mimic": 124,
        "cpu9": 76,
        "slippi-ai": 104,
    }
    assert snapshot["ancestors"] == []
    assert snapshot["manifest_ancestors"] is None
    assert len(set(snapshot["labels"])) == 309
    assert len(set(snapshot["artifact_roots"])) == 309
    assert snapshot["execution_partitions"] == (
        ["head-to-head"] * 5 + ["75m"] * 152 + ["10m"] * 152
    )
    assert snapshot["initial_state_games"] == 309
    assert snapshot["dual_defaults"]["p1_contract"] == "10m-pretraining"
    assert snapshot["dual_defaults"]["p2_contract"] == "75m-pretraining"
    assert all(
        row["p1_contract"].endswith("-pretraining")
        and row["p2_contract"].endswith("-pretraining")
        for row in snapshot["head_to_head"]
    )


def test_posttraining_variant_is_exactly_329_unique_games_with_75m_first() -> None:
    pretraining = _snapshot("pretraining")
    posttraining = _snapshot("posttraining")

    assert posttraining["variant"] == posttraining["runner_variant"] == "posttraining"
    assert posttraining["suite_id"] == (
        "posttraining-winners-expanded-step195248-step127214-v1"
    )
    assert posttraining["suite_schema"] == (
        "integration.posttraining_winner_benchmark_suite.v1"
    )
    assert posttraining["state_schema"] == (
        "integration.posttraining_winner_benchmark_state.v1"
    )
    assert posttraining["game_count"] == posttraining["schedule_length"] == 329
    assert posttraining["schedule_sha256"] == (
        "566eb5e897492dedfa05f5233ad94f2c4933027aa75c55cbadbe55ebb42da2bd"
    )
    assert posttraining["kinds"] == {
        "head-to-head": 5,
        "ancestor": 20,
        "mimic": 124,
        "cpu9": 76,
        "slippi-ai": 104,
    }
    expected_groups = dict(pretraining["groups"])
    expected_groups.update(
        {
            "75m-posttraining-v-pretraining-ancestor-10": 10,
            "10m-posttraining-v-pretraining-ancestor-10": 10,
        }
    )
    assert posttraining["groups"] == expected_groups
    assert (
        posttraining["schedule_semantics"][:309]
        == pretraining["schedule_semantics"]
    )
    assert len(set(posttraining["labels"])) == 329
    assert len(set(posttraining["artifact_roots"])) == 329
    assert set(posttraining["artifact_roots"]).isdisjoint(pretraining["artifact_roots"])
    assert set(posttraining["labels"]).isdisjoint(pretraining["labels"])
    assert posttraining["execution_partitions"] == (
        ["75m"] * 162 + ["head-to-head"] * 5 + ["10m"] * 162
    )
    assert posttraining["initial_state_games"] == 329
    assert posttraining["legacy_imports"] == []
    assert posttraining["ten_identity"] == {
        "sha256": "63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b",
        "bytes": 96_452_075,
        "step": 195_248,
        "frames": 12_795_772_928,
    }
    assert posttraining["seventy_five_identity"] == {
        "sha256": "8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077",
        "bytes": 644_291_987,
        "step": 127_214,
        "frames": 8_337_096_704,
    }
    assert set(posttraining["manifest_ancestors"]) == {"10m", "75m"}
    assert posttraining["dual_defaults"]["p1_contract"] == "10m-posttraining"
    assert posttraining["dual_defaults"]["p2_contract"] == "75m-posttraining"


def test_posttraining_dual_commands_and_same_profile_roles_are_exact() -> None:
    snapshot = _snapshot("posttraining")
    seeds = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
    ancestors = snapshot["ancestors"]

    assert len(ancestors) == 20
    for profile, rows in (
        ("75m", ancestors[:10]),
        ("10m", ancestors[10:]),
    ):
        assert [row["profile"] for row in rows] == [profile] * 10
        assert [row["seed"] for row in rows] == seeds
        assert [row["swap"] for row in rows] == [False, True] * 5
        assert all(row["matchup_class"] == "head-to-head" for row in rows)
        assert all(row["characters"] == ["FOX", "FOX"] for row in rows)
        posttraining_checkpoint = snapshot[
            "ten_checkpoint" if profile == "10m" else "seventy_five_checkpoint"
        ]
        ancestor_checkpoint = str(
            ROOT
            / (
                ".e011-cache/final-winners/remote-verification/10m-step-122064.pt"
                if profile == "10m"
                else ".e011-cache/final-winners/remote-verification/75m-step-86016.pt"
            )
        )
        for row in rows:
            expected_checkpoints = (
                [ancestor_checkpoint, posttraining_checkpoint]
                if row["swap"]
                else [posttraining_checkpoint, ancestor_checkpoint]
            )
            expected_contracts = (
                [f"{profile}-pretraining", f"{profile}-posttraining"]
                if row["swap"]
                else [f"{profile}-posttraining", f"{profile}-pretraining"]
            )
            assert [row["p1_checkpoint"], row["p2_checkpoint"]] == expected_checkpoints
            assert [row["p1_contract"], row["p2_contract"]] == expected_contracts
            labels = [player["model"] for player in row["video_players"]]
            assert any("post-trained final" in label for label in labels)
            assert any("pretraining ancestor" in label for label in labels)

    for row in snapshot["head_to_head"]:
        expected_contracts = (
            ["75m-posttraining", "10m-posttraining"]
            if row["swap"]
            else ["10m-posttraining", "75m-posttraining"]
        )
        expected_checkpoints = (
            [snapshot["seventy_five_checkpoint"], snapshot["ten_checkpoint"]]
            if row["swap"]
            else [snapshot["ten_checkpoint"], snapshot["seventy_five_checkpoint"]]
        )
        assert [row["p1_contract"], row["p2_contract"]] == expected_contracts
        assert [row["p1_checkpoint"], row["p2_checkpoint"]] == expected_checkpoints

    completed = _run_variant(
        r'''
import copy
import hashlib
import tempfile
from pathlib import Path

import run_frisson_dual_oneoff as dual
from melee_policy.integration import final_benchmark_suite as suite


assert dual._scoring_role_keys(
    {1: "75m", 2: "75m"},
    {1: "75m-posttraining", 2: "75m-pretraining"},
) == {1: "75m-posttraining", 2: "75m-pretraining"}

try:
    dual._scoring_role_keys(
        {1: "75m", 2: "75m"},
        {1: "75m-posttraining", 2: "75m-posttraining"},
    )
except RuntimeError:
    pass
else:
    raise AssertionError("duplicate same-profile scoring roles were accepted")


def checkpoint_record(identity, step, frames):
    return {
        "path": str(identity.path.resolve()),
        "sha256": identity.sha256,
        "byte_length": identity.byte_length,
        "step": step,
        "processed_target_frames": frames,
    }


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory).resolve()
    suite.ROOT = root
    suite.ARTIFACT_OUTPUT = root / "artifacts"
    cases = [
        spec
        for spec in suite.SCHEDULE
        if spec.kind == "ancestor"
        and spec.seed in {101, 211}
    ]
    assert len(cases) == 4
    for spec in cases:
        if spec.frisson_profile == "10m":
            post = (
                suite.TEN_M_IDENTITY,
                195_248,
                12_795_772_928,
                "Frisson-AI 10M post-trained step-195248",
                "10m-posttraining",
            )
            ancestor = (
                suite.TEN_M_ANCESTOR_IDENTITY,
                122_064,
                7_999_586_304,
                "Frisson-AI 10M v2 step-122064",
                "10m-pretraining",
            )
        else:
            post = (
                suite.SEVENTY_FIVE_M_IDENTITY,
                127_214,
                8_337_096_704,
                "Frisson-AI 75M post-trained step-127214",
                "75m-posttraining",
            )
            ancestor = (
                suite.SEVENTY_FIVE_M_ANCESTOR_IDENTITY,
                86_016,
                5_637_144_576,
                "Frisson-AI 75M step-86016",
                "75m-pretraining",
            )
        p1, p2 = (ancestor, post) if spec.head_to_head_swap_ports else (post, ancestor)
        replay_path = spec.artifact_root / "replays" / "game.slp"
        replay_path.parent.mkdir(parents=True)
        replay_bytes = f"scoring replay {spec.label}".encode()
        replay_path.write_bytes(replay_bytes)
        summary = {
            "schema_version": "integration.frisson_vs_frisson.v2",
            "gate": {"decision": "pass", "checks": {"all": True}},
            "configuration": {
                "seed": spec.seed,
                "stage": "FINAL_DESTINATION",
                "require_natural_end": True,
                "player_1": {
                    "model": p1[3],
                    "profile": spec.frisson_profile,
                    "character": "FOX",
                    "port": 1,
                },
                "player_2": {
                    "model": p2[3],
                    "profile": spec.frisson_profile,
                    "character": "FOX",
                    "port": 2,
                },
                "model_to_port": {p1[4]: 1, p2[4]: 2},
                "policy_evaluation_seeds": {p1[4]: spec.seed, p2[4]: spec.seed},
            },
            "checkpoints": {
                "p1": checkpoint_record(p1[0], p1[1], p1[2]),
                "p2": checkpoint_record(p2[0], p2[1], p2[2]),
            },
            "execution": {
                "game_end_observed": True,
                "termination": "natural-game-end",
                "winner": p1[3],
                "last_stocks": {p1[4]: 2, p2[4]: 0},
            },
            "artifacts": {
                "replays": [
                    {
                        "path": str(replay_path),
                        "sha256": hashlib.sha256(replay_bytes).hexdigest(),
                        "byte_length": len(replay_bytes),
                        "tournament_result_replay": True,
                        "validation": {
                            "libmelee_parseable": True,
                            "parsed_frames_strictly_consecutive": True,
                            "first_parsed_frame": -123,
                            "last_parsed_frame": 10,
                            "required_trace_coverage": {"complete": True},
                        },
                    }
                ]
            },
        }
        result = suite.validate_completed_summary(
            summary,
            spec,
            summary_path=spec.summary_path,
        )
        assert result["winner"] == p1[3]
        assert result["stocks"] == {p1[4]: 2, p2[4]: 0}

        tampered = copy.deepcopy(summary)
        tampered["configuration"]["model_to_port"] = {p1[4]: 2, p2[4]: 1}
        try:
            suite.validate_completed_summary(tampered, spec, summary_path=spec.summary_path)
        except suite.SuiteArtifactError:
            pass
        else:
            raise AssertionError("swapped score attribution was accepted")

        tampered = copy.deepcopy(summary)
        tampered["execution"]["winner"] = p2[3]
        try:
            suite.validate_completed_summary(tampered, spec, summary_path=spec.summary_path)
        except suite.SuiteArtifactError:
            pass
        else:
            raise AssertionError("winner inconsistent with stocks was accepted")

        tampered = copy.deepcopy(summary)
        tampered["execution"]["last_stocks"] = {"unknown-a": 2, "unknown-b": 0}
        try:
            suite.validate_completed_summary(tampered, spec, summary_path=spec.summary_path)
        except suite.SuiteArtifactError:
            pass
        else:
            raise AssertionError("unknown score roles were accepted")
''',
        variant="posttraining",
    )
    assert completed.stdout == ""
