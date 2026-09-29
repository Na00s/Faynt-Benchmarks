"""Frozen E011 public Slippi panel: paired coverage, separate transfer strata."""
from __future__ import annotations
import hashlib
import json
import random
import sys
from pathlib import Path

from slippi_panel_audit import ROOT, OUTPUT, atomic_json, digest
sys.path.insert(0, str(ROOT / "src"))
from melee_policy.integration.frisson_match import FRISSON_SUPPORTED_CHARACTERS
from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS

ROSTER = tuple(FRISSON_SUPPORTED_CHARACTERS)
STAGES = ("FINAL_DESTINATION", "BATTLEFIELD", "DREAMLAND", "FOUNTAIN_OF_DREAMS", "POKEMON_STADIUM", "YOSHIS_STORY")
PRIORITY = ("fox_d21_ditto_v4", "fox_d24_ditto_v4", "falco_d21_ditto_v4", "gm", "master", "SFIL", "fox_d21_ditto_hax_v3", "diamond", "gm-v1", "medium-v2", "plat", "gold", "silver", "medium-v1")


def checkpoint_rosters(record):
    import melee
    metadata = record["metadata"]
    dataset = metadata["config"]["dataset"]["allowed_characters"]
    bc = list(ROSTER) if dataset == "all" else [x.upper() for x in dataset.split(",")]
    actor = metadata.get("rl_config", {}).get("agent", metadata.get("agent_config", {}))
    chars = actor.get("char")
    deployed = bc if chars is None else [x["name"] if isinstance(x, dict) else melee.Character(x).name for x in chars]
    if not set(deployed).issubset(ROSTER) or not set(bc).issubset(ROSTER):
        raise ValueError("checkpoint declares non-launchable fighters")
    return bc, deployed


def make_manifest(audit):
    if audit["failures"]:
        raise ValueError("checkpoint audit has failures")
    records = audit["records"]
    if records["medium"]["sha256"] != records["medium-v2"]["sha256"]:
        raise ValueError("medium alias changed; a new inventory decision is required")
    releases = {}
    for key in PRIORITY:
        row = records[key]
        bc, deployed = checkpoint_rosters(row)
        actor = row["metadata"].get("rl_config", {}).get("agent", {})
        names = actor["name"]
        names = [names] if isinstance(names, str) else names
        selected_name = names[0]  # Exact native upstream fallback, declared before games.
        if selected_name not in row["metadata"]["name_map"]:
            raise ValueError(f"missing native name: {key}")
        releases[key] = {**row, "bc_declared_characters": bc, "deployed_characters": deployed,
                         "selected_name": selected_name, "all_rl_names": names,
                         "name_selection": "first RL-trained name, pinned build_delayed_agent default",
                         "delay": row["metadata"]["config"]["policy"]["delay"]}
    games = []
    # Every pair shares seed/stage/characters/name; physical ports are reversed.
    for profile in ("75m", "10m"):
        for block in ("supported-mirror", "extended-roster", "forced-mirror"):
            for release_index, (key, release) in enumerate(releases.items()):
                supported = release["deployed_characters"]
                chars = supported if block == "supported-mirror" else [c for c in ROSTER if c not in supported]
                pairs = []
                for char_index, character in enumerate(chars):
                    stages = STAGES if len(supported) == 1 and block == "supported-mirror" else (STAGES[(char_index + release_index) % len(STAGES)],)
                    for stage_index, stage in enumerate(stages):
                        opponent = supported[char_index % len(supported)] if block == "extended-roster" else character
                        # Common seed across Frisson sizes and baselines for a matched cell.
                        seed_key = f"p21-panel-v1:{block}:{character}:{opponent}:{stage}"
                        seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], "big") % 2**31
                        pair = f"{profile}:{key}:{block}:{character}:{opponent}:{stage}:{seed}"
                        pair_id = hashlib.sha256(pair.encode()).hexdigest()[:16]
                        pair_games = []
                        for port in (1, 2):
                            label = f"sp21-{profile}-{key.lower()}-{block}-{character.lower()}-{stage.lower()}-p{port}"
                            pair_games.append({"label": label, "pair_id": pair_id, "profile": profile,
                                "release": key, "block": block, "frisson_character": character,
                                "slippi_character": opponent, "stage": stage, "seed": seed,
                                "frisson_port": port, "slippi_port": 3 - port,
                                "player_name": release["selected_name"],
                                "slippi_character_in_bc_declaration": opponent in release["bc_declared_characters"],
                                "slippi_character_in_deployed_roster": opponent in supported,
                                "opponent_context_outside_rl_roster": character not in supported,
                                "save_slp": True, "save_video": False})
                        random.Random(seed).shuffle(pair_games)
                        pairs.append(pair_games)
                random.Random(20260910 + release_index).shuffle(pairs)
                games.extend(g for pair in pairs for g in pair)
    assert len(games) == 2624, len(games)
    assert len({g["label"] for g in games}) == len(games)
    for i, game in enumerate(games):
        game["ordinal"] = i + 1
    return {
        "schema": "e011.slippi_public_panel.p21.v1", "experiment_id": "E011-SLIPPI-P21-PUBLIC-V1",
        "frisson_checkpoints": CHECKPOINTS, "releases": releases, "games": games,
        "aliases": {"medium": "medium-v2"},
        "configuration": {"frisson_context": "ring", "actor_context_frames": 128,
            "frisson_delay": 0, "slippi_delay": "checkpoint native; mixed console delay zero",
            "temperature": 1.0, "stocks": 4, "maximum_game_frames": 30000,
            "maximum_attempts": 3, "no_progress_seconds": 900, "wall_limit_seconds": 7200,
            "gameplay_order": "finish existing P21 suite, then all 75M, then all 10M; supported before transfer",
            "stopping_rule": "fixed schedule; no outcome-dependent additions, removals or early stopping",
            "failed_attempts": "retain evidence, retry transient infrastructure only, quarantine scientific failures",
            "claims": "fixed-checkpoint descriptive comparison; limited per-cell sample; unequal native delays; no universal SOTA claim",
            "scope_exclusions": ["private or unpublished weights", "auto-* live model-bank routing unverified", "basic-* aliases lack a single immutable checkpoint", "other matchup-specific files outside named checkpoint panel"],
            "bc_vs_rl_coverage": "gm-v1 and medium-v1 outside four RL characters remains inside declared broad BC roster"},
    }


def main():
    manifest = make_manifest(json.loads((OUTPUT / "audit-summary.json").read_text()))
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("frozen manifest differs")
    atomic_json(path, manifest)
    print(json.dumps({"games": len(manifest["games"]), "per_model": len(manifest["games"]) // 2,
                      "checkpoints": len(manifest["releases"]), "manifest_sha256": digest(path)}))


if __name__ == "__main__":
    main()
