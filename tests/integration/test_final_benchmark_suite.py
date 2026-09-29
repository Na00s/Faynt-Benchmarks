from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from melee_policy.integration import final_benchmark_suite as suite
from melee_policy.integration import frisson_match as mimic_integration


def test_frozen_schedule_has_exact_requested_309_game_composition() -> None:
    assert len(suite.SCHEDULE) == 309
    assert len({spec.label for spec in suite.SCHEDULE}) == 309
    assert Counter(spec.group for spec in suite.SCHEDULE) == {
        "head-to-head-5": 5,
        "75m-mimic-current-master-fox-10": 10,
        "75m-cpu9-fox-10": 10,
        "75m-mimic-current-native-core-character-mirrors-2": 12,
        "75m-slippi-medium-v2-all-character-mirrors-2": 24,
        "10m-mimic-current-master-fox-10": 10,
        "10m-cpu9-fox-10": 10,
        "10m-mimic-current-native-core-character-mirrors-2": 12,
        "10m-slippi-medium-v2-all-character-mirrors-2": 24,
        "75m-remaining-cpu9-character-mirrors-2": 28,
        "75m-mimic-native-extension-character-mirrors-2": 34,
        "75m-mimic-current-fox-master-ood-character-mirrors-2": 6,
        "75m-slippi-native-specialist-character-mirrors-2": 4,
        "75m-slippi-medium-v2-ood-character-mirrors-2": 24,
        "10m-remaining-cpu9-character-mirrors-2": 28,
        "10m-mimic-native-extension-character-mirrors-2": 34,
        "10m-mimic-current-fox-master-ood-character-mirrors-2": 6,
        "10m-slippi-native-specialist-character-mirrors-2": 4,
        "10m-slippi-medium-v2-ood-character-mirrors-2": 24,
    }
    additive_fox = [
        spec
        for spec in suite.SCHEDULE
        if spec.group == "75m-mimic-current-native-core-character-mirrors-2"
        and spec.opponent_character == "FOX"
    ]
    assert [spec.seed for spec in additive_fox] == [1201, 1301]
    assert [spec.head_to_head_swap_ports for spec in suite.SCHEDULE[:5]] == [
        False,
        True,
        False,
        True,
        False,
    ]


def test_every_game_command_is_raw_replay_first_and_uses_its_exact_profile() -> None:
    for spec in suite.SCHEDULE:
        command = suite.command_for_game(spec, "PYTHON")
        text = " ".join(command)
        assert "--save-video" not in command
        assert "--save-slp" not in command
        assert "step83919" not in text
        assert spec.label in command
        assert str(spec.seed) in command
        if spec.kind != "head-to-head":
            expected = (
                suite.TEN_M_CHECKPOINT
                if spec.frisson_profile == "10m"
                else suite.SEVENTY_FIVE_M_CHECKPOINT
            )
            assert str(expected) in command
            assert command[command.index("--p1-character") + 1] == spec.player_1_character


def test_native_slippi_specialist_commands_select_the_exact_release_contract() -> None:
    specialists = [
        spec
        for spec in suite.SCHEDULE
        if "slippi-native-specialist" in spec.group
    ]
    assert len(specialists) == 8
    for spec in specialists:
        command = suite.command_for_game(spec, "PYTHON")
        release = suite.SLIPPI_SPECIALIST_BY_CHARACTER[spec.opponent_character]
        assert command[command.index("--p2-slippi-release") + 1] == release.release_name
        assert "--p2-checkpoint" not in command


def test_manifest_freezes_current_mimic_source_and_specialist_release_metadata() -> None:
    opponents = suite.schedule_manifest()["opponent_bundles"]
    mimic = opponents["mimic"]
    assert len(mimic) == 23
    assert all(row["source_repository"] == suite.MIMIC_SOURCE_REPOSITORY for row in mimic)
    assert all(row["source_revision"] == suite.MIMIC_SOURCE_REVISION for row in mimic)
    assert all(row["source_directory"] == suite.MIMIC_SOURCE_DIRECTORY for row in mimic)

    specialists = opponents["slippi_ai"]["native_specialists"]
    assert len(specialists) == 2
    for row in specialists:
        release = suite.SLIPPI_SPECIALIST_BY_CHARACTER[row["character"]]
        assert row["release_contract"] == suite._slippi_specialist_release_contract(release)


def test_ood_commands_are_explicit_same_character_transfers() -> None:
    transfers = [
        spec
        for spec in suite.SCHEDULE
        if spec.opponent_evaluation_mode == "forced-ood-character-transfer"
    ]
    assert len(transfers) == 60
    for spec in transfers:
        command = suite.command_for_game(spec, "PYTHON")
        assert "--allow-p2-ood-character" in command
        assert command[command.index("--p1-character") + 1] == spec.player_1_character
        assert command[command.index("--p2-character") + 1] == spec.player_1_character
        if spec.kind == "mimic":
            assert str(suite.MIMIC_BY_DIRECTORY["fox-master"].checkpoint) in command
            assert str(suite.MIMIC_BY_DIRECTORY["fox-master"].assets) in command
    supported = [
        spec
        for spec in suite.SCHEDULE
        if spec.opponent_evaluation_mode == "supported"
    ]
    assert all("--allow-p2-ood-character" not in suite.command_for_game(spec, "PYTHON") for spec in supported)


def test_schedule_labels_follow_the_stable_contract() -> None:
    assert suite.SCHEDULE[0].label == "final-10m-step122064-v-75m-step86016-s101"
    assert suite.SCHEDULE[5].label == (
        "final-75m-step86016-v-mimic-current-master-fox-s101"
    )
    assert suite.SCHEDULE[15].label == "final-75m-step86016-v-cpu9-fox-s101"
    assert any(
        spec.label
        == "final-75m-step86016-cptfalcon-v-mimic-current-cptfalcon-s1301"
        for spec in suite.SCHEDULE
    )
    assert suite.SCHEDULE[116].label == (
        "final-10m-step122064-samus-v-slippi-medium-v2-samus-s1301"
    )
    assert suite.SCHEDULE[-1].label == (
        "final-10m-step122064-samus-v-mimic-native-samus-s1501"
    )


def test_every_external_character_sweep_is_a_true_mirror() -> None:
    sweep = [
        spec
        for spec in suite.SCHEDULE
        if "all-character-mirrors" in spec.group
        or "mimic-current-native-core-character-mirrors" in spec.group
    ]
    assert len(sweep) == 72
    assert all(spec.player_1_character == spec.opponent_character for spec in sweep)
    assert {spec.frisson_profile for spec in sweep} == {"10m", "75m"}
    assert len(suite.FRISSON_ROSTER) == 26
    assert len(set(suite.FRISSON_ROSTER)) == 26
    manifest = suite.schedule_manifest()
    assert manifest["schema_version"] == "integration.final_winner_benchmark_suite.v5"
    assert manifest["suite_id"] == suite.SUITE_ID
    assert manifest["game_count"] == 309
    assert manifest["frisson_roster"] == list(suite.FRISSON_ROSTER)
    assert all("frisson_profile" in game for game in manifest["games"])
    assert all("player_1_character" in game for game in manifest["games"])
    assert all("matchup_class" in game for game in manifest["games"])
    assert all("opponent_evaluation_mode" in game for game in manifest["games"])


def test_extension_is_same_character_and_covers_each_opponents_untrained_roster() -> None:
    extension = suite.SCHEDULE[117:]
    assert len(extension) == 192
    by_profile = {
        profile: [spec for spec in extension if spec.frisson_profile == profile]
        for profile in ("75m", "10m")
    }
    assert all(len(games) == 96 for games in by_profile.values())
    signatures = {}
    extension_characters = {
        character for character, _ in suite.MIMIC_EXTENSION_CHARACTERS
    }
    slippi_released = {
        *{character for character, _ in suite.SLIPPI_CHARACTERS},
        *suite.SLIPPI_SPECIALIST_BY_CHARACTER,
    }
    for profile, games in by_profile.items():
        signatures[profile] = {
            (
                spec.kind,
                spec.player_1_character,
                spec.opponent_character,
                spec.seed,
                spec.opponent_evaluation_mode,
                spec.mimic_directory,
            )
            for spec in games
        }
        assert all(spec.player_1_character == spec.opponent_character for spec in games)
        cpu = [spec for spec in games if spec.kind == "cpu9"]
        mimic = [spec for spec in games if spec.kind == "mimic"]
        slippi = [spec for spec in games if spec.kind == "slippi-ai"]
        assert len(cpu) == 28
        assert len(mimic) == 40
        assert len(slippi) == 28
        assert {spec.player_1_character for spec in cpu} == {
            character for character, _ in suite.REMAINING_FRISSON_CHARACTERS
        }
        assert {spec.player_1_character for spec in mimic} == extension_characters
        assert {spec.player_1_character for spec in slippi} == (
            set(suite.FRISSON_ROSTER)
            - {character for character, _ in suite.SLIPPI_CHARACTERS}
        )
        assert all(spec.opponent_evaluation_mode == "supported" for spec in cpu)
        assert all(
            spec.opponent_evaluation_mode == "forced-ood-character-transfer"
            and spec.mimic_directory == "fox-master"
            for spec in mimic
            if spec.player_1_character in suite.MIMIC_OOD_CHARACTERS
        )
        assert all(
            spec.opponent_evaluation_mode == "supported"
            and spec.mimic_directory
            == suite.MIMIC_BY_CHARACTER[spec.player_1_character].directory
            for spec in mimic
            if spec.player_1_character not in suite.MIMIC_OOD_CHARACTERS
        )
        assert all(
            spec.opponent_evaluation_mode == "forced-ood-character-transfer"
            for spec in slippi
            if spec.player_1_character not in slippi_released
        )
        assert all(
            spec.opponent_evaluation_mode == "supported"
            for spec in slippi
            if spec.player_1_character in suite.SLIPPI_SPECIALIST_BY_CHARACTER
        )
        assert Counter((spec.kind, spec.player_1_character) for spec in games).most_common()[0][1] == 2
    assert signatures["75m"] == signatures["10m"]


def test_extension_matchup_class_requires_exact_ood_mirrors() -> None:
    transfer = next(
        spec
        for spec in suite.SCHEDULE[117:]
        if spec.kind == "mimic"
        and spec.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    summary = {
        "configuration": {
            "seed": transfer.seed,
            "stage": suite.STAGE,
            "require_natural_end": True,
            "player_1": {
                "model": "frisson-ai",
                "character": transfer.player_1_character,
                "port": 1,
            },
            "player_2": {
                "model": "mimic",
                "character": transfer.opponent_character,
                "port": 2,
            },
        }
    }
    suite._validate_players(summary, transfer)
    with pytest.raises(suite.SuiteArtifactError, match="contract differs"):
        suite._validate_players(summary, replace(transfer, matchup_class="cross-character"))

    original_mirror = next(
        spec
        for spec in suite.SCHEDULE[:117]
        if spec.kind == "mimic" and spec.opponent_character == "FALCO"
    )
    wrong = {
        "configuration": {
            "seed": original_mirror.seed,
            "stage": suite.STAGE,
            "require_natural_end": True,
            "player_1": {"model": "frisson-ai", "character": "FOX", "port": 1},
            "player_2": {"model": "mimic", "character": "FALCO", "port": 2},
        }
    }
    with pytest.raises(suite.SuiteArtifactError, match="character assignment"):
        suite._validate_players(wrong, original_mirror)


def test_v3_state_migration_authenticates_and_preserves_only_unchanged_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_directory = tmp_path / suite.V3_SUITE_ID
    legacy_directory.mkdir()
    manifest_path = legacy_directory / "manifest.json"
    state_path = legacy_directory / "state.json"
    current_games = suite.schedule_manifest()["games"]
    prefix = []
    for value in current_games[: suite.UNCHANGED_V3_PREFIX_GAME_COUNT]:
        record = dict(value)
        record.pop("opponent_evaluation_mode")
        prefix.append(record)
    legacy_manifest = {
        "schema_version": suite.V3_SUITE_SCHEMA_VERSION,
        "suite_id": suite.V3_SUITE_ID,
        "schedule_sha256": suite.V3_SCHEDULE_SHA256,
        "game_count": suite.V3_GAME_COUNT,
        "games": prefix + [{} for _ in range(suite.V3_GAME_COUNT - len(prefix))],
    }
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    first = suite.SCHEDULE[0]
    prefix_games = {
        spec.label: {
            "status": "complete" if spec == first else "pending",
            "attempts_started": 3 if spec == first else 0,
            "command": ["legacy-launcher"] if spec == first else None,
            "result": {"legacy": True} if spec == first else None,
        }
        for spec in suite.SCHEDULE[: suite.UNCHANGED_V3_PREFIX_GAME_COUNT]
    }
    legacy_state = {
        "schema_version": suite.V3_STATE_SCHEMA_VERSION,
        "suite_id": suite.V3_SUITE_ID,
        "schedule_sha256": suite.V3_SCHEDULE_SHA256,
        "status": "blocked",
        "updated_at_utc": "2026-09-03T12:00:00+00:00",
        "games": prefix_games,
        "deferred_postgame_validation_backlog": {"old": {"status": "unresolved"}},
    }
    state_path.write_text(json.dumps(legacy_state), encoding="utf-8")
    monkeypatch.setattr(suite, "V3_SUITE_DIRECTORY", legacy_directory)
    monkeypatch.setattr(suite, "V3_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(suite, "V3_STATE_PATH", state_path)

    migrated = suite._migrate_v3_state()

    assert migrated is not None
    assert migrated["schema_version"] == suite.STATE_SCHEMA_VERSION
    assert migrated["suite_id"] == suite.SUITE_ID
    assert len(migrated["games"]) == 309
    assert migrated["games"][first.label]["attempts_started"] == 3
    assert migrated["games"][first.label]["command"] == ["legacy-launcher"]
    assert migrated["games"][first.label]["status"] == "pending"
    assert migrated["games"][first.label]["result"] is None
    assert migrated["games"][suite.SCHEDULE[117].label]["attempts_started"] == 0
    migration = migrated["migration"]
    assert migration["migrated_label_count"] == 117
    assert migration["obsolete_v3_extension_games_imported"] is False
    assert migration["gameplay_reexecuted"] is False
    assert migration["legacy_deferred_postgame_validation_backlog"] == {
        "old": {"status": "unresolved"}
    }


def test_both_final_checkpoints_pass_full_roster_capability_gate() -> None:
    if not suite.TEN_M_CHECKPOINT.is_file() or not suite.SEVENTY_FIVE_M_CHECKPOINT.is_file():
        pytest.skip("requires caller-supplied exact Base checkpoint bytes")
    inputs = suite.verify_frozen_inputs(include_opponents=False)
    roster = inputs["frisson_full_roster"]
    assert roster["launchable_characters"] == list(suite.FRISSON_ROSTER)
    for key in ("10m_step122064", "75m_step86016"):
        capability = roster[key]
        assert capability["character_embedding_shape"] == [33, 128]
        assert capability["character_action_embedding_shape"] == [33, 399, 128]
        assert capability["character_action_nonzero_physical_ids"] == list(range(27))
        assert capability["character_action_zero_non_roster_ids"] == list(range(27, 33))
        assert all(capability["checks"].values())


def _checkpoint_record(identity: suite.FileIdentity, step: int, frames: int) -> dict[str, Any]:
    return {
        "path": str(identity.path.resolve()),
        "sha256": identity.sha256,
        "byte_length": identity.byte_length,
        "step": step,
        "processed_target_frames": frames,
    }


def _mimic_source_record(release: suite.MimicRelease) -> dict[str, Any]:
    return {
        "repository_url": suite.MIMIC_SOURCE_REPOSITORY,
        "revision": suite.MIMIC_SOURCE_REVISION,
        "tracked_tree_clean": True,
        "directory": str((suite.ROOT / suite.MIMIC_SOURCE_DIRECTORY).resolve()),
        "checks": {
            "repository_url_exact": True,
            "revision_exact": True,
            "tracked_tree_clean": True,
        },
        "bundle_character": release.character,
        "bundle_checkpoint_sha256": release.checkpoint_sha256,
    }


def _mimic_ood_identity_summary(spec: suite.GameSpec) -> dict[str, Any]:
    release = suite.MIMIC_BY_DIRECTORY["fox-master"]
    transfer = {
        "schema_version": suite.MIMIC_OOD_TRANSFER_SCHEMA_VERSION,
        "mode": suite.MIMIC_OOD_TRANSFER_MODE,
        "forced_ood_character_transfer": True,
        "explicit_opt_in_field": "allow_player_2_ood_character",
        "physical_port": 2,
        "physical_character": spec.opponent_character,
        "checkpoint_character": release.character,
        "physical_character_covered_by_released_checkpoint": False,
        "checkpoint_roster_unchanged": True,
        "checkpoint": {
            "path": ".e001-cache/mimic/fox-master/model.pt",
            "sha256": release.checkpoint_sha256,
        },
        "assets": {
            "directory": ".e001-cache/mimic/fox-master",
            "released_bundle_name": release.bundle_name,
            "released_run_name": "fox-mastfox-20260625",
        },
        "native_inference": suite._mimic_ood_native_inference_contract(),
        "scientific_interpretation": "out-of-distribution character transfer",
        "checks": {name: True for name in suite.MIMIC_OOD_PROVENANCE_CHECKS},
    }
    return {
        "configuration": {
            "allow_player_2_ood_character": True,
            "player_2": {
                "physical_character": spec.opponent_character,
                "checkpoint_character": release.character,
                "character_transfer_mode": suite.MIMIC_OOD_TRANSFER_MODE,
            },
        },
        "contract": {
            "selected_checkpoint": _checkpoint_record(
                suite.SEVENTY_FIVE_M_IDENTITY,
                86_016,
                5_637_144_576,
            ),
            "mimic_bundle_selection": "explicit-released-bundle-ood-character-transfer",
            "mimic_checkpoint_character": release.character,
            "mimic_physical_character": spec.opponent_character,
            "mimic_requested_character_covered_by_released_checkpoint": False,
            "mimic_forced_ood_character_transfer": True,
            "allow_player_2_ood_character": True,
            "mimic_ood_character_transfer": transfer,
        },
        "mimic": {
            "checkpoint_sha256": release.checkpoint_sha256,
            "controlled_character": release.character,
            "bundle_identity": {
                "checkpoint_sha256": release.checkpoint_sha256,
                "name": release.bundle_name,
                "character": release.character,
                "revision": suite.MIMIC_RELEASE_REVISION,
                "source_repository": suite.MIMIC_SOURCE_REPOSITORY,
                "source_revision": suite.MIMIC_SOURCE_REVISION,
                "source_directory": suite.MIMIC_SOURCE_DIRECTORY,
            },
            "ood_character_transfer": copy.deepcopy(transfer),
        },
        "sources": {"mimic": _mimic_source_record(release)},
    }


def _slippi_ood_identity_summary(spec: suite.GameSpec) -> dict[str, Any]:
    declared_characters = [character for character, _ in suite.SLIPPI_CHARACTERS]
    return {
        "contract": {
            "frisson": {
                "selected_checkpoint": _checkpoint_record(
                    suite.SEVENTY_FIVE_M_IDENTITY,
                    86_016,
                    5_637_144_576,
                )
            },
            "slippi_ai": {
                "source_revision": suite.SLIPPI_SOURCE_REVISION,
                "checkpoint_sha256": suite.SLIPPI_MEDIUM_V2_IDENTITY.sha256,
                "checkpoint_byte_length": suite.SLIPPI_MEDIUM_V2_IDENTITY.byte_length,
                "player_name": suite.SLIPPI_PLAYER_NAME,
                "sample_temperature": suite.SLIPPI_TEMPERATURE,
                "checkpoint_policy_delay_frames": suite.SLIPPI_POLICY_DELAY_FRAMES,
                "native_parser_observation_filter_recurrent_state_and_decoder": True,
                "requested_character": spec.opponent_character,
                "checkpoint_declared_characters": declared_characters,
                "requested_character_covered_by_training": False,
                "forced_ood_character_transfer": True,
                "ood_provenance": {
                    "explicit_opt_in": "allow_player_2_ood_character",
                    "physical_port": 2,
                    "checkpoint_roster_unchanged": True,
                    "playing_strength_interpretation": "out-of-training-roster transfer",
                },
            },
        },
        "policies": {
            "p2": {
                "metadata": {
                    "character_transfer": {
                        "requested_character": spec.opponent_character,
                        "checkpoint_declared_characters": declared_characters,
                        "requested_character_covered_by_training": False,
                        "forced_ood_character_transfer": True,
                        "checkpoint_bytes_unchanged": True,
                        "native_inference_path_unchanged": True,
                    }
                }
            }
        },
    }


def _slippi_specialist_identity_summary(spec: suite.GameSpec) -> dict[str, Any]:
    release = suite.SLIPPI_SPECIALIST_BY_CHARACTER[spec.opponent_character]
    release_contract = suite._slippi_specialist_release_contract(release)
    return {
        "contract": {
            "frisson": {
                "selected_checkpoint": _checkpoint_record(
                    suite.SEVENTY_FIVE_M_IDENTITY,
                    86_016,
                    5_637_144_576,
                )
            },
            "slippi_ai": {
                "release": release.release_name,
                "release_contract": release_contract,
                "source_revision": suite.SLIPPI_SOURCE_REVISION,
                "checkpoint_sha256": release.checkpoint_sha256,
                "checkpoint_byte_length": release.checkpoint_byte_length,
                "parameter_count": release.parameter_count,
                "player_name": suite.SLIPPI_PLAYER_NAME,
                "sample_temperature": suite.SLIPPI_TEMPERATURE,
                "checkpoint_policy_delay_frames": release.policy_delay_frames,
                "policy_delay_frames": release.policy_delay_frames,
                "console_delay_frames": 0,
                "effective_policy_delay_frames": release.policy_delay_frames,
                "configured_mixed_runtime": {
                    "exact_mode_only": True,
                    "inference_mode": "synchronous-concurrent",
                    "checkpoint_delay_frames": release.policy_delay_frames,
                    "mixed_runtime_console_delay_frames": 0,
                    "effective_policy_delay_frames": release.policy_delay_frames,
                    "release_contract": release_contract,
                },
                "async_inference": True,
                "compile": True,
                "native_parser_observation_filter_recurrent_state_and_decoder": True,
                "requested_character": release.character,
                "checkpoint_declared_characters": [release.character],
                "requested_character_covered_by_training": True,
                "forced_ood_character_transfer": False,
            },
        }
    }


def _tampered_value(value: Any) -> Any:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.5
    if isinstance(value, str):
        return f"{value}-tampered"
    if isinstance(value, list):
        return [*value, "TAMPERED"]
    raise AssertionError(f"test has no tamper rule for {value!r}")


def _tamper_nested(summary: dict[str, Any], path: tuple[str, ...]) -> None:
    cursor: Any = summary
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = _tampered_value(cursor[path[-1]])


def test_frozen_mimic_ood_provenance_contract_matches_the_summary_generator(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "fox-master"
    checkpoint = assets / "model.pt"
    release = suite.MIMIC_BY_DIRECTORY["fox-master"]
    request = mimic_integration.FrissonMatchRequest(
        player_2_character="KIRBY",
        player_2_checkpoint=checkpoint,
        player_2_assets=assets,
        allow_player_2_ood_character=True,
    )
    runtime = SimpleNamespace(
        checkpoint_path=checkpoint.resolve(),
        asset_directory=assets.resolve(),
        checkpoint_sha256=release.checkpoint_sha256,
        controlled_character=release.character,
        bundle_identity={
            "name": release.bundle_name,
            "run_name": "fox-mastfox-20260717-cstick9-nomirror",
            "character": release.character,
            "checkpoint_sha256": release.checkpoint_sha256,
            "checks": {"checkpoint_matches_asset_model": True},
            "state_dictionary": {
                "strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
            },
        },
    )

    provenance = mimic_integration._mimic_ood_character_transfer_provenance(
        request,
        runtime,
        tmp_path,
    )

    assert provenance is not None
    assert suite.MIMIC_OOD_TRANSFER_SCHEMA_VERSION == (
        mimic_integration.MIMIC_OOD_CHARACTER_TRANSFER_SCHEMA_VERSION
    )
    assert suite.MIMIC_OOD_TRANSFER_MODE == mimic_integration.MIMIC_OOD_CHARACTER_TRANSFER_MODE
    assert suite._mimic_ood_native_inference_contract() == provenance["native_inference"]
    assert tuple(provenance["checks"]) == suite.MIMIC_OOD_PROVENANCE_CHECKS


@pytest.mark.parametrize(
    ("character", "directory", "bundle_name"),
    (
        ("FOX", "fox-master", "fox-master"),
        ("FALCO", "falco", "falco"),
        ("MARTH", "marth", "marth"),
        ("SHEIK", "sheik", "sheik"),
        ("CPTFALCON", "cptfalcon", "cptfalcon"),
        ("LUIGI", "luigi", "luigi"),
    ),
)
def test_mimic_identity_uses_each_release_canonical_bundle_name(
    character: str,
    directory: str,
    bundle_name: str,
) -> None:
    release = suite.MIMIC_BY_CHARACTER[character]
    assert release.directory == directory
    assert release.bundle_name == bundle_name

    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.kind == "mimic"
        and candidate.frisson_profile == "75m"
        and candidate.opponent_character == character
    )
    summary = {
        "contract": {
            "selected_checkpoint": _checkpoint_record(
                suite.SEVENTY_FIVE_M_IDENTITY,
                86_016,
                5_637_144_576,
            )
        },
        "mimic": {
            "checkpoint_sha256": release.checkpoint_sha256,
            "controlled_character": release.character,
            "bundle_identity": {
                "checkpoint_sha256": release.checkpoint_sha256,
                "name": bundle_name,
                "character": release.character,
                "revision": suite.MIMIC_RELEASE_REVISION,
                "source_repository": suite.MIMIC_SOURCE_REPOSITORY,
                "source_revision": suite.MIMIC_SOURCE_REVISION,
                "source_directory": suite.MIMIC_SOURCE_DIRECTORY,
            },
        },
        "sources": {"mimic": _mimic_source_record(release)},
    }
    suite._validate_model_identity(summary, spec)

    summary["mimic"]["bundle_identity"]["name"] = f"{bundle_name}-tampered"
    with pytest.raises(suite.SuiteArtifactError, match="MIMIC release identity differs"):
        suite._validate_model_identity(summary, spec)


@pytest.mark.parametrize(
    "path",
    (
        ("mimic", "bundle_identity", "source_repository"),
        ("mimic", "bundle_identity", "source_revision"),
        ("mimic", "bundle_identity", "source_directory"),
        ("sources", "mimic", "repository_url"),
        ("sources", "mimic", "revision"),
        ("sources", "mimic", "tracked_tree_clean"),
        ("sources", "mimic", "directory"),
        ("sources", "mimic", "bundle_character"),
        ("sources", "mimic", "bundle_checkpoint_sha256"),
        ("sources", "mimic", "checks", "repository_url_exact"),
        ("sources", "mimic", "checks", "revision_exact"),
        ("sources", "mimic", "checks", "tracked_tree_clean"),
    ),
    ids=lambda path: "-".join(path),
)
def test_mimic_identity_rejects_wrong_current_source_provenance(
    path: tuple[str, ...],
) -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.kind == "mimic"
        and candidate.frisson_profile == "75m"
        and candidate.opponent_character == "FOX"
    )
    release = suite.MIMIC_BY_DIRECTORY["fox-master"]
    summary = {
        "contract": {
            "selected_checkpoint": _checkpoint_record(
                suite.SEVENTY_FIVE_M_IDENTITY,
                86_016,
                5_637_144_576,
            )
        },
        "mimic": {
            "checkpoint_sha256": release.checkpoint_sha256,
            "controlled_character": release.character,
            "bundle_identity": {
                "checkpoint_sha256": release.checkpoint_sha256,
                "name": release.bundle_name,
                "character": release.character,
                "revision": release.revision,
                "source_repository": suite.MIMIC_SOURCE_REPOSITORY,
                "source_revision": suite.MIMIC_SOURCE_REVISION,
                "source_directory": suite.MIMIC_SOURCE_DIRECTORY,
            },
        },
        "sources": {"mimic": _mimic_source_record(release)},
    }
    _tamper_nested(summary, path)
    with pytest.raises(suite.SuiteArtifactError):
        suite._validate_model_identity(summary, spec)


@pytest.mark.parametrize(
    "path",
    (
        ("contract", "slippi_ai", "release"),
        ("contract", "slippi_ai", "parameter_count"),
        ("contract", "slippi_ai", "policy_delay_frames"),
        ("contract", "slippi_ai", "console_delay_frames"),
        ("contract", "slippi_ai", "effective_policy_delay_frames"),
        ("contract", "slippi_ai", "async_inference"),
        ("contract", "slippi_ai", "compile"),
        ("contract", "slippi_ai", "requested_character"),
        ("contract", "slippi_ai", "release_contract", "checkpoint_kind"),
        ("contract", "slippi_ai", "release_contract", "checkpoint_tag"),
        ("contract", "slippi_ai", "release_contract", "checkpoint_config_version"),
        (
            "contract",
            "slippi_ai",
            "configured_mixed_runtime",
            "mixed_runtime_console_delay_frames",
        ),
        (
            "contract",
            "slippi_ai",
            "configured_mixed_runtime",
            "effective_policy_delay_frames",
        ),
    ),
    ids=lambda path: "-".join(path),
)
def test_slippi_specialist_identity_rejects_wrong_release_or_delay_contract(
    path: tuple[str, ...],
) -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.frisson_profile == "75m"
        and candidate.kind == "slippi-ai"
        and "slippi-native-specialist" in candidate.group
    )
    summary = _slippi_specialist_identity_summary(spec)
    suite._validate_model_identity(summary, spec)
    _tamper_nested(summary, path)
    with pytest.raises(suite.SuiteArtifactError):
        suite._validate_model_identity(summary, spec)


def test_ood_model_identity_binds_physical_character_separately_from_checkpoint() -> None:
    mimic_spec = next(
        spec
        for spec in suite.SCHEDULE
        if spec.kind == "mimic"
        and spec.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    mimic_summary = _mimic_ood_identity_summary(mimic_spec)
    suite._validate_model_identity(mimic_summary, mimic_spec)
    mimic_summary["contract"]["mimic_ood_character_transfer"]["physical_character"] = "FOX"
    with pytest.raises(suite.SuiteArtifactError, match="OOD character-transfer identity differs"):
        suite._validate_model_identity(mimic_summary, mimic_spec)

    slippi_spec = next(
        spec
        for spec in suite.SCHEDULE
        if spec.kind == "slippi-ai"
        and spec.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    slippi_summary = _slippi_ood_identity_summary(slippi_spec)
    suite._validate_model_identity(slippi_summary, slippi_spec)


MIMIC_OOD_TAMPER_PATHS = (
    ("configuration", "allow_player_2_ood_character"),
    ("configuration", "player_2", "physical_character"),
    ("configuration", "player_2", "checkpoint_character"),
    ("configuration", "player_2", "character_transfer_mode"),
    ("contract", "mimic_bundle_selection"),
    ("contract", "mimic_checkpoint_character"),
    ("contract", "mimic_physical_character"),
    ("contract", "mimic_requested_character_covered_by_released_checkpoint"),
    ("contract", "mimic_forced_ood_character_transfer"),
    ("contract", "allow_player_2_ood_character"),
    ("contract", "mimic_ood_character_transfer", "schema_version"),
    ("contract", "mimic_ood_character_transfer", "mode"),
    ("contract", "mimic_ood_character_transfer", "explicit_opt_in_field"),
    ("contract", "mimic_ood_character_transfer", "physical_port"),
    ("contract", "mimic_ood_character_transfer", "physical_character"),
    ("contract", "mimic_ood_character_transfer", "checkpoint_character"),
    (
        "contract",
        "mimic_ood_character_transfer",
        "physical_character_covered_by_released_checkpoint",
    ),
    ("contract", "mimic_ood_character_transfer", "forced_ood_character_transfer"),
    ("contract", "mimic_ood_character_transfer", "checkpoint_roster_unchanged"),
    ("contract", "mimic_ood_character_transfer", "checkpoint", "sha256"),
    ("contract", "mimic_ood_character_transfer", "assets", "released_bundle_name"),
    *(("contract", "mimic_ood_character_transfer", "native_inference", name) for name in (
        "live_policy_class",
        "observation_builder",
        "model_forward",
        "sampler_decoder",
        "decode_strategy",
        "temperature",
        "top_k",
        "top_p",
        "online_delay_frames",
        "inference_mode",
        "current_frame_barrier_before_controller_transaction",
    )),
    *(("contract", "mimic_ood_character_transfer", "checks", name) for name in (
        suite.MIMIC_OOD_PROVENANCE_CHECKS
    )),
    ("mimic", "ood_character_transfer", "explicit_opt_in_field"),
)


@pytest.mark.parametrize("path", MIMIC_OOD_TAMPER_PATHS, ids=lambda path: "-".join(path))
def test_mimic_ood_identity_rejects_each_tampered_opt_in_and_native_path_field(
    path: tuple[str, ...],
) -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.kind == "mimic"
        and candidate.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    summary = _mimic_ood_identity_summary(spec)
    _tamper_nested(summary, path)
    with pytest.raises(suite.SuiteArtifactError):
        suite._validate_model_identity(summary, spec)


@pytest.mark.parametrize("mapping_name", ("native_inference", "checks"))
def test_mimic_ood_identity_rejects_unexpected_provenance_fields(mapping_name: str) -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.kind == "mimic"
        and candidate.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    summary = _mimic_ood_identity_summary(spec)
    summary["contract"]["mimic_ood_character_transfer"][mapping_name]["unexpected"] = True
    with pytest.raises(suite.SuiteArtifactError):
        suite._validate_model_identity(summary, spec)


SLIPPI_OOD_TAMPER_PATHS = (
    ("contract", "slippi_ai", "requested_character"),
    ("contract", "slippi_ai", "checkpoint_declared_characters"),
    ("contract", "slippi_ai", "requested_character_covered_by_training"),
    ("contract", "slippi_ai", "forced_ood_character_transfer"),
    (
        "contract",
        "slippi_ai",
        "native_parser_observation_filter_recurrent_state_and_decoder",
    ),
    ("contract", "slippi_ai", "ood_provenance", "explicit_opt_in"),
    ("contract", "slippi_ai", "ood_provenance", "physical_port"),
    ("contract", "slippi_ai", "ood_provenance", "checkpoint_roster_unchanged"),
    (
        "contract",
        "slippi_ai",
        "ood_provenance",
        "playing_strength_interpretation",
    ),
    ("policies", "p2", "metadata", "character_transfer", "requested_character"),
    (
        "policies",
        "p2",
        "metadata",
        "character_transfer",
        "checkpoint_declared_characters",
    ),
    (
        "policies",
        "p2",
        "metadata",
        "character_transfer",
        "requested_character_covered_by_training",
    ),
    (
        "policies",
        "p2",
        "metadata",
        "character_transfer",
        "forced_ood_character_transfer",
    ),
    (
        "policies",
        "p2",
        "metadata",
        "character_transfer",
        "checkpoint_bytes_unchanged",
    ),
    (
        "policies",
        "p2",
        "metadata",
        "character_transfer",
        "native_inference_path_unchanged",
    ),
)


@pytest.mark.parametrize("path", SLIPPI_OOD_TAMPER_PATHS, ids=lambda path: "-".join(path))
def test_slippi_ood_identity_rejects_each_tampered_opt_in_and_native_path_field(
    path: tuple[str, ...],
) -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.kind == "slippi-ai"
        and candidate.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    summary = _slippi_ood_identity_summary(spec)
    _tamper_nested(summary, path)
    with pytest.raises(suite.SuiteArtifactError):
        suite._validate_model_identity(summary, spec)


@pytest.mark.parametrize(
    "path",
    (
        ("contract", "slippi_ai", "ood_provenance"),
        ("policies", "p2", "metadata", "character_transfer"),
    ),
    ids=("contract-opt-in", "native-character-transfer"),
)
def test_slippi_ood_identity_rejects_unexpected_provenance_fields(
    path: tuple[str, ...],
) -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE
        if candidate.kind == "slippi-ai"
        and candidate.opponent_evaluation_mode == "forced-ood-character-transfer"
    )
    summary = _slippi_ood_identity_summary(spec)
    cursor: Any = summary
    for key in path:
        cursor = cursor[key]
    cursor["unexpected"] = True
    with pytest.raises(suite.SuiteArtifactError):
        suite._validate_model_identity(summary, spec)


def test_supported_slippi_identity_does_not_require_ood_provenance() -> None:
    spec = next(
        candidate
        for candidate in suite.SCHEDULE[: suite.UNCHANGED_V3_PREFIX_GAME_COUNT]
        if candidate.kind == "slippi-ai"
    )
    summary = {
        "contract": {
            "frisson": {
                "selected_checkpoint": _checkpoint_record(
                    suite.SEVENTY_FIVE_M_IDENTITY,
                    86_016,
                    5_637_144_576,
                )
            },
            "slippi_ai": {
                "source_revision": suite.SLIPPI_SOURCE_REVISION,
                "checkpoint_sha256": suite.SLIPPI_MEDIUM_V2_IDENTITY.sha256,
                "checkpoint_byte_length": suite.SLIPPI_MEDIUM_V2_IDENTITY.byte_length,
                "player_name": suite.SLIPPI_PLAYER_NAME,
                "sample_temperature": suite.SLIPPI_TEMPERATURE,
                "checkpoint_policy_delay_frames": suite.SLIPPI_POLICY_DELAY_FRAMES,
            },
        }
    }
    suite._validate_model_identity(summary, spec)


def _head_to_head_summary(spec: suite.GameSpec, replay_path: Path) -> dict[str, Any]:
    replay_bytes = replay_path.read_bytes()
    ten = _checkpoint_record(suite.TEN_M_IDENTITY, 122_064, 7_999_586_304)
    seventy_five = _checkpoint_record(
        suite.SEVENTY_FIVE_M_IDENTITY, 86_016, 5_637_144_576
    )
    p1, p2 = (seventy_five, ten) if spec.head_to_head_swap_ports else (ten, seventy_five)
    p1_profile, p2_profile = (
        ("75m", "10m") if spec.head_to_head_swap_ports else ("10m", "75m")
    )
    model_names = {
        "10m": "Frisson-AI 10M v2 step-122064",
        "75m": "Frisson-AI 75M step-86016",
    }
    return {
        "schema_version": "integration.frisson_vs_frisson.v2",
        "gate": {"decision": "pass", "checks": {"all": True}},
        "configuration": {
            "seed": spec.seed,
            "stage": "FINAL_DESTINATION",
            "require_natural_end": True,
            "player_1": {
                "model": model_names[p1_profile],
                "profile": p1_profile,
                "character": "FOX",
                "port": 1,
            },
            "player_2": {
                "model": model_names[p2_profile],
                "profile": p2_profile,
                "character": "FOX",
                "port": 2,
            },
            "model_to_port": {p1_profile: 1, p2_profile: 2},
            "policy_evaluation_seeds": {
                p1_profile: spec.seed,
                p2_profile: spec.seed,
            },
        },
        "checkpoints": {"p1": p1, "p2": p2},
        "execution": {
            "game_end_observed": True,
            "termination": "natural-game-end",
            "winner": model_names["75m"],
            "last_stocks": {"10m": 0, "75m": 2},
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


def test_completed_game_validator_rejects_the_old_75m_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "ARTIFACT_OUTPUT", tmp_path)
    spec = suite.SCHEDULE[0]
    replay_path = spec.artifact_root / "replays" / "game.slp"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(b"synthetic scoring replay")
    summary = _head_to_head_summary(spec, replay_path)
    proposed = suite.validate_completed_summary(
        summary,
        spec,
        summary_path=spec.summary_path,
    )
    assert proposed["winner"] == "Frisson-AI 75M step-86016"
    assert not spec.summary_path.exists()
    spec.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = suite.validate_completed_game(spec)
    assert result["winner"] == "Frisson-AI 75M step-86016"

    summary["checkpoints"]["p2"]["sha256"] = (
        "08329fe349224be2dc92ff1332cc022000251622f4fffe2e76ff401ba3a8a99c"
    )
    spec.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(suite.SuiteArtifactError, match="identity differs"):
        suite.validate_completed_game(spec)


def test_completed_game_validator_rejects_failed_fidelity_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "ARTIFACT_OUTPUT", tmp_path)
    spec = suite.SCHEDULE[0]
    replay_path = spec.artifact_root / "replays" / "game.slp"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(b"synthetic scoring replay")
    summary = _head_to_head_summary(spec, replay_path)
    summary["gate"]["decision"] = "fail"
    spec.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(suite.SuiteArtifactError, match="fidelity gate"):
        suite.validate_completed_game(spec)
