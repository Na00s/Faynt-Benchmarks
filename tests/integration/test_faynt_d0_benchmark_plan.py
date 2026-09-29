"""Checkpoint-only schedule cloning, fresh state and frozen-source protection."""
from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

import faynt_d0_benchmark_plan as plan


@pytest.fixture(scope="module")
def sources():
    if any(not (plan.ROOT / path).is_file() for path, _ in plan.SOURCE_FILES.values()):
        pytest.skip("frozen local benchmark manifests unavailable")
    return plan.load_sources()


@pytest.fixture
def learners():
    return {
        profile: {
            "profile": profile, "step": step, "relative_path": f"checkpoints/{profile}.pt",
            "sha256": digit * 64, "byte_length": 1, "parameter_count": 100,
            "trained_delay": 0, "action_offset_frames": 1, "context_mode": "ring",
            "actor_context_frames": 128, "temperature": 1.0,
            "processed_target_frames": None, "validation_nll": None,
        }
        for profile, step, digit in (("10m", 632, "1"), ("75m", 980, "2"))
    }


@pytest.fixture
def manifest(sources, learners):
    source, provenance = sources
    return plan.build_manifest(learners, sources=source, provenance=provenance)


def test_all_fixed_rows_preserve_matchups_and_namespace_labels(sources, manifest):
    source, _ = sources
    for component in ("expanded_slippi",):
        original, new = source[component], manifest[component]
        assert len(original["games"]) == len(new["games"])
        assert {g["label"] for g in original["games"]}.isdisjoint(g["label"] for g in new["games"])
        for before, after in zip(original["games"], new["games"], strict=True):
            recovered = copy.deepcopy(after)
            recovered["label"] = before["label"]
            if "artifact_root" in recovered:
                assert Path(recovered["artifact_root"]).name == after["label"]
                recovered["artifact_root"] = str(Path(recovered["artifact_root"]).with_name(before["label"]))
            assert recovered == before
            assert manifest["old_to_new_labels"][component][before["label"]] == after["label"]
            assert after["label"].startswith(plan.RUN_ID + "-")
    assert manifest["expanded_slippi"]["releases"] == source["expanded_slippi"]["releases"]
    configuration = copy.deepcopy(manifest["expanded_slippi"]["configuration"])
    configuration["gameplay_order"] = source["expanded_slippi"]["configuration"]["gameplay_order"]
    assert configuration == source["expanded_slippi"]["configuration"]
    assert manifest["expanded_slippi"]["configuration"]["gameplay_order"] == "all 75M, then all 10M; supported before transfer"


def test_all_learner_projections_use_new_identity(manifest, learners):
    assert manifest["learners"] == manifest["expanded_slippi"]["frisson_checkpoints"] == learners


def test_counts_preserve_exact_slippi_panel_coverage(manifest):
    rows = manifest["fixed_games"]
    assert len(rows) == len(plan.game_index(manifest)) == 2624
    assert Counter(g["component"] for g in rows) == {"expanded_slippi": 2624}
    assert Counter(p for g in rows for p in g["learner_profiles"]) == {"10m": 1312, "75m": 1312}
    assert all(len(g["learner_profiles"]) == 1 for g in rows)
    assert manifest["counts"]["fixed_physical"] == 2624
    assert manifest["counts"]["fixed_appearances_per_learner"] == {"10m": 1312, "75m": 1312}
    expanded = manifest["expanded_slippi"]["games"]
    assert Counter(g["block"] for g in expanded) == {
        "supported-mirror": 488, "extended-roster": 1068, "forced-mirror": 1068,
    }
    for profile in plan.PROFILES:
        assert Counter(g["block"] for g in expanded if g["profile"] == profile) == {
            "supported-mirror": 244, "extended-roster": 534, "forced-mirror": 534,
        }
    pairs = defaultdict(list)
    for game in expanded:
        pairs[game["pair_id"]].append(game)
    assert len(pairs) == 1312
    for pair in pairs.values():
        assert {g["frisson_port"] for g in pair} == {1, 2}
        for key in ("seed", "stage", "frisson_character", "slippi_character", "player_name"):
            assert len({g[key] for g in pair}) == 1


def test_authorization_excludes_every_earlier_component_and_pins_budget(manifest):
    assert manifest["authorization"]["hard_spend_limit_usd"] == 150
    assert manifest["authorization"]["compute_policy"] == "standard-preemptible"
    assert manifest["authorization"]["included_components"] == ["expanded_slippi"]
    assert manifest["authorization"]["excluded_components"] == [
        "legacy_external", "pretraining_ancestors", "head_to_head", "hal_bo5",
    ]
    assert set(manifest["source_manifests"]) == {"expanded_slippi"}
    assert set(manifest["old_to_new_labels"]) == {"expanded_slippi"}
    for key in ("legacy", "hal_protocols", "hal_series"):
        assert key not in manifest


def test_all_games_save_replays_and_disable_videos(manifest):
    for game in manifest["fixed_games"] + manifest["expanded_slippi"]["games"]:
        assert game["save_slp"] is True and game["save_video"] is False
    assert manifest["recording"]["canaries_count_as_games"] is False


def test_sources_and_inputs_stay_unchanged(sources, learners):
    source, provenance = sources
    before = copy.deepcopy((source, provenance, learners))
    first = plan.build_manifest(learners, sources=source, provenance=provenance)
    second = plan.build_manifest(learners, sources=source, provenance=provenance)
    assert first == second
    assert (source, provenance, learners) == before
    for record in provenance.values():
        assert plan.sha256(plan.ROOT / record["path"]) == record["sha256"]
    assert provenance["expanded_slippi"]["selected_field"] == "whole manifest"


def test_lookup_requires_new_label_and_returns_independent_row(manifest):
    row = manifest["fixed_games"][0]
    assert plan.lookup_game(manifest, row["label"]) == row
    with pytest.raises(KeyError):
        plan.lookup_game(manifest, row["source_label"])
    found = plan.lookup_game(manifest, row["label"])
    found["game"]["seed"] = -1
    assert plan.lookup_game(manifest, row["label"])["game"]["seed"] != -1


def test_initial_state_has_no_completed_games_or_attempts(manifest):
    state = plan.fresh_state(manifest)
    assert manifest["imported_completed_games"] == []
    assert state["games"] == {} and state["attempts"] == []
    assert state["imported_completed_games"] == []
    assert "hal_series" not in state


def test_prepare_is_idempotent_preserves_progress_and_rejects_changed_plan(manifest, tmp_path):
    output = tmp_path / plan.RUN_ID
    first = plan.write_prepared(manifest, output)
    assert first["hard_spend_limit_usd"] == 150
    assert first["compute_policy"] == "standard-preemptible"
    state_path = output / "state.json"
    state = json.loads(state_path.read_text())
    state["games"] = {manifest["fixed_games"][0]["label"]: {"status": "complete"}}
    state_path.write_text(json.dumps(state))
    assert plan.write_prepared(manifest, output) == first
    assert json.loads(state_path.read_text()) == state
    assert len(json.loads((output / "expanded-slippi/manifest.json").read_text())["games"]) == 2624
    assert not (output / "legacy").exists() and not (output / "hal-bo5").exists()
    changed = copy.deepcopy(manifest)
    changed["fixed_games"][0]["game"]["seed"] = -1
    with pytest.raises(ValueError, match="content digest"):
        plan.write_prepared(changed, output)
    changed["manifest_content_sha256"] = plan.canonical_sha256({k: v for k, v in changed.items() if k != "manifest_content_sha256"})
    with pytest.raises(ValueError, match="existing frozen output"):
        plan.write_prepared(changed, output)
    assert json.loads(state_path.read_text()) == state


def test_prepare_rejects_prior_run_directory(manifest, tmp_path):
    with pytest.raises(ValueError, match="fresh run ID"):
        plan.write_prepared(manifest, tmp_path / "postrl-p21-step1318-step222-v1")


@pytest.mark.parametrize("key,value", [
    ("step", 1318), ("trained_delay", 18), ("context_mode", "prefix"),
    ("actor_context_frames", 256), ("action_offset_frames", 0), ("temperature", 0.5),
    ("relative_path", "../outside.pt"), ("relative_path", "/outside.pt"),
    ("sha256", "missing"), ("byte_length", 0),
])
def test_rejects_unselected_identity_or_changed_timing(learners, key, value):
    learners["10m"][key] = value
    with pytest.raises(ValueError):
        plan.validate_learners(learners)


def test_changed_frozen_source_rejected(tmp_path):
    relative, _ = next(iter(plan.SOURCE_FILES.values()))
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    with pytest.raises(ValueError, match="frozen source receipt changed"):
        plan.load_sources(tmp_path)


def test_rejects_sources_that_expand_the_authorized_scope(sources, learners):
    source, provenance = copy.deepcopy(sources)
    source["legacy"] = {"games": []}
    with pytest.raises(ValueError, match="only the expanded Slippi panel"):
        plan.build_manifest(learners, sources=source, provenance=provenance)
