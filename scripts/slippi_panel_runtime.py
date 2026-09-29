"""Audited public-release adapter, isolated from the already-running benchmark.

The upstream builder, encoder, filter, sampling, recurrent state, FIFO and
controller decoder are reused. Only immutable release metadata, the selected
native name, physical ports and stage are parameterized in each fresh process.
"""
from __future__ import annotations
import copy
import dataclasses
import functools
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from slippi_panel_audit import ROOT, OUTPUT, digest, load_restricted, json_safe
sys.path.insert(0, str(ROOT / "src"))
from melee_policy.integration import slippi_ai_policy as boundary


def manifest():
    return json.loads((OUTPUT / "manifest.json").read_text())


def register_release(record):
    cfg = record["metadata"]["config"]
    contract = boundary.SlippiAIReleaseContract(
        key=record["name"], display_name=f"Slippi-AI {record['name']} ({record['selected_name']})",
        checkpoint_sha256=record["sha256"], checkpoint_bytes=record["bytes"],
        variable_count=record["variable_count"], parameter_count=record["parameter_count"],
        policy_delay_frames=record["delay"], supported_characters=tuple(record["deployed_characters"]),
        allowed_opponents=cfg["dataset"]["allowed_opponents"], checkpoint_kind="self-play-rl",
        checkpoint_tag=cfg.get("tag"), checkpoint_config_version=cfg["version"],
        checkpoint_step=record["metadata"].get("step"), rl_trained_names=tuple(record["all_rl_names"]),
        official_url=record["url"])
    boundary.SLIPPI_AI_RELEASE_CONTRACTS[record["name"]] = contract
    return contract


class PublicRuntime(boundary._NativePolicyRuntime):
    def __init__(self, config, capture, *, record):
        if digest(config.checkpoint_path) != record["sha256"]:
            raise RuntimeError("checkpoint changed after metadata audit")
        # Reject executable pickle globals before the pinned upstream loader.
        restricted = load_restricted(config.checkpoint_path)
        # Use the audit's portable metadata view after verifying original checkpoint bytes.
        raw_metadata = {k: json_safe(v) for k, v in restricted.items() if k != "state"}
        if raw_metadata != record["metadata"]:
            raise RuntimeError("checkpoint metadata differs from frozen audit")
        del restricted
        modules = boundary._activate_pinned_source(config.source_directory)
        self._utils = modules["utils"]
        self._policies = modules["policies"]
        self._mirror_lib = modules["mirror_lib"]
        saving, eval_lib = modules["saving"], modules["eval_lib"]
        state = saving.load_state_from_disk(str(config.checkpoint_path))
        original_config = copy.deepcopy(state["config"])
        if saving.get_platform(original_config) is not self._policies.Platform.TF:
            raise RuntimeError("public panel expects TensorFlow checkpoints")
        trained_names = eval_lib.get_name_from_rl_state(state)
        if list(trained_names) != record["all_rl_names"] or config.name != record["selected_name"]:
            raise RuntimeError("native name conditioning differs from frozen contract")
        policy = eval_lib.build_agent(
            state=state, controller=capture, port=config.port, opponent_port=config.opponent_port,
            name=config.name, mirror=False, console_delay=config.console_delay_frames,
            async_inference=True, sample_temperature=1.0, compile=True,
            batch_steps=0, tf={"jit_compile": False})
        if not isinstance(policy, eval_lib.Agent) or type(policy._agent).__name__ != "AsyncDelayedAgent":
            raise RuntimeError("unexpected native policy runtime")
        if policy._agent.policy.delay != record["delay"] or policy._agent.delay != record["delay"] - config.console_delay_frames:
            raise RuntimeError("native policy delay differs")
        name_code = int(np.asarray(policy._agent.name_code)[0])
        if name_code != int(state["name_map"][config.name]):
            raise RuntimeError("native name-code changed")
        tree = importlib.import_module("tree")
        loaded = tree.flatten(policy._agent.policy.variables)
        saved = tree.flatten(state["state"]["policy"])
        if len(loaded) != len(saved) or len(loaded) != record["variable_count"]:
            raise RuntimeError("restored variable count differs")
        for index, (variable, expected) in enumerate(zip(loaded, saved, strict=True)):
            actual = np.asarray(variable.numpy())
            if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected) or not np.isfinite(actual).all():
                raise RuntimeError(f"restored tensor {index} differs")
        if sum(int(np.asarray(x).size) for x in saved) != record["parameter_count"]:
            raise RuntimeError("restored parameter count differs")
        self._policy, self._mirror, self._started = policy, False, False
        self.metadata = {
            "release_contract": config.release_contract.as_dict(),
            "saving_loader": "slippi_ai.saving.load_state_from_disk",
            "runtime_class": f"{type(policy).__module__}.{type(policy).__name__}",
            "delayed_runtime_class": f"{type(policy._agent).__module__}.{type(policy._agent).__name__}",
            "parser_class": "slippi_db.parse_libmelee.Parser",
            "parser_source_path": str(modules["parser_file"]),
            "observation_filter_class": f"{type(policy._observation_filter).__module__}.{type(policy._observation_filter).__name__}",
            "controller_head_class": f"{type(policy._agent.policy.controller_head).__module__}.{type(policy._agent.policy.controller_head).__name__}",
            "platform": "tf", "checkpoint_config": boundary._json_safe(original_config),
            "upgraded_checkpoint_config": boundary._json_safe(state["config"]),
            "checkpoint_step": state.get("step"), "checkpoint_name_map_size": len(state["name_map"]),
            "requested_name": config.name, "effective_name": config.name,
            "requested_name_code": name_code, "effective_name_code": name_code,
            "selected_name": config.name, "selected_name_code": name_code, "name_code": name_code,
            "rl_trained_names": trained_names, "supported_characters": record["deployed_characters"],
            "dataset_supported_characters": record["bc_declared_characters"],
            "allowed_opponents": config.release_contract.allowed_opponents,
            "supported_rendered_stages": list(boundary.MEDIUM_V2_SUPPORTED_STAGES),
            "variable_count": len(loaded), "parameter_count": record["parameter_count"],
            "variable_shapes": [list(x.shape) for x in saved], "variable_dtypes": [str(x.dtype) for x in saved],
            "state_assignment": {"shape_mismatches": [], "dtype_mismatches": [], "value_mismatches": [], "nonfinite_variables": []},
            "recurrent_state_class": type(policy._agent.hidden_state).__name__,
            "policy_delay_frames": record["delay"], "console_delay_frames": config.console_delay_frames,
            "effective_policy_delay_frames": policy._agent.delay,
            "native_analog_shoulder": "L", "native_analog_r_emitted": False,
            "audit_sha256": record["sha256"], "policy_tensor_sha256": record["policy_sha256"],
        }


def session(config, *, record):
    return boundary.SlippiAIPolicySession(config, agent_factory=functools.partial(PublicRuntime, record=record))


def configure_game(game, record):
    """Configure one fresh worker; shared source files stay byte-identical."""
    from melee_policy.integration import frisson_slippi_match as match
    from melee_policy.integration import frisson_match as fm
    from melee_policy.integration import game_bundle
    import melee
    register_release(record)
    match.FRISSON_PORT = fm.FRISSON_PORT = game["frisson_port"]
    match.SLIPPI_PORT = fm.MIMIC_PORT = game["slippi_port"]
    match.MATCH_STAGE = fm.MATCH_STAGE = game["stage"]
    match.DEFAULT_PLAYER_NAME = game["player_name"]
    match.SlippiAIPolicySession = functools.partial(session, record=record)

    original_menu_helper = melee.MenuHelper
    selected_stage = melee.Stage[game["stage"]]
    class PanelMenuHelper(original_menu_helper):
        def menu_helper_simple(self, gamestate, controller, character, stage, *args, **kwargs):
            return super().menu_helper_simple(gamestate, controller, character, selected_stage, *args, **kwargs)
    melee.MenuHelper = PanelMenuHelper

    def physical_bundle_player(summary, port):
        for key in ("player_1", "player_2"):
            configured = summary.get("configuration", {}).get(key, {})
            if configured.get("port") == port:
                return {"port": port, "model": configured["model"], "character": configured["character"]}
        raise ValueError(f"physical bundle player {port} is missing")
    game_bundle._player_record = physical_bundle_player

    original_frisson_request = match._frisson_request
    match._frisson_request = lambda request: dataclasses.replace(original_frisson_request(request), stage=request.stage)

    original_trace = match._trace_row
    def trace(*args, **kwargs):
        row = original_trace(*args, **kwargs)
        row["slots"] = {f"p{slot['port']}": slot for slot in row["slots"].values()}
        return row
    match._trace_row = trace

    original_replay_audit = match._selected_replay_identity_audit
    def physical_replay_audit(*args, **kwargs):
        result = original_replay_audit(*args, **kwargs)
        for key in ("characters", "replay_character_names"):
            roles = result["expected"][key]
            result["expected"][key] = {f"p{game['frisson_port']}": roles["p1"],
                                        f"p{game['slippi_port']}": roles["p2"]}
        return result
    match._selected_replay_identity_audit = physical_replay_audit

    original_reproducibility = match._runtime_reproducibility_record
    def reproducibility(*args, **kwargs):
        result = original_reproducibility(*args, **kwargs)
        result["public_panel"] = {
            "manifest_sha256": digest(OUTPUT / "manifest.json"),
            "adapter_sha256": digest(Path(__file__)),
            "game": game,
            "bc_declared_characters": record["bc_declared_characters"],
            "rl_deployed_characters": record["deployed_characters"],
            "role_fields": "configuration.player_1 and policies.p1 describe Frisson; use their port fields for physical slots",
            "trace_slots": "controller_trace slots p1/p2 are physical ports",
            "legacy_gate_names": "final_destination means selected frozen stage; training-coverage flags refer to deployed roster",
        }
        return result
    match._runtime_reproducibility_record = reproducibility

    def first_context(gamestate, request):
        frame = int(gamestate.frame)
        stage = gamestate.stage.name
        chars = {f"p{port}": gamestate.players[port].character.name for port in (1, 2)}
        expected = {f"p{game['frisson_port']}": request.player_1_character,
                    f"p{game['slippi_port']}": request.player_2_character}
        checks = {"first_frame_minus_123": bool(frame == -123),
                  "selected_stage": bool(stage == request.stage), "physical_characters": bool(chars == expected)}
        if not all(checks.values()):
            raise RuntimeError(f"first physical context mismatch: {checks}; observed={chars}; expected={expected}")
        return {"frame": frame, "stage": stage, "characters": chars,
                "requested_characters": expected, "checks": checks,
                "costumes": {f"p{port}": int(gamestate.players[port].costume) for port in (1, 2)}}
    match._first_context = first_context

    # Validate all releases through the same explicit shared runtime contract.
    def release_config(config, root, request):
        source = config["slippi_ai"]
        checks = {
            "repository": source["repository_url"] == boundary.SLIPPI_AI_REPOSITORY_URL,
            "source": source["source_revision"] == boundary.SLIPPI_AI_SOURCE_REVISION,
            "async": source["async_inference"] is True, "compile": source["compile"] is True,
            "exact_mode": config["slippi_integration"]["exact_mode_only"] is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"shared Slippi runtime contract differs: {checks}")
        policy_config = match._slippi_config(config, root, request)
        policy_config.validate()
        return {"exact_mode_only": True, "inference_mode": match.EXACT_INFERENCE_MODE,
                "checkpoint_delay_frames": record["delay"], "mixed_runtime_console_delay_frames": 0,
                "effective_policy_delay_frames": record["delay"],
                "release_contract": policy_config.release_contract.as_dict(), "checks": checks}
    match._validate_slippi_release_config_contract = release_config
    return match


def run_game(game, attempt_label):
    panel = manifest()
    record = panel["releases"][game["release"]]
    match = configure_game(game, record)
    request = match.FrissonSlippiMatchRequest(
        player_1_character=game["frisson_character"], player_2_character=game["slippi_character"],
        player_1_checkpoint=ROOT / panel["frisson_checkpoints"][game["profile"]]["relative_path"],
        player_2_slippi_release=game["release"], player_2_name=game["player_name"],
        stage=game["stage"], seed=game["seed"], artifact_label=attempt_label,
        allow_player_2_ood_character=not game["slippi_character_in_deployed_roster"],
        save_slp=True, save_video=False, max_game_frames=30000)
    print(f"PHYSICAL PORTS: Frisson={game['frisson_port']}, Slippi={game['slippi_port']}; stage={game['stage']}; name={game['player_name']}", flush=True)
    summary = match.run_frisson_slippi_match(ROOT / "configs/integration.toml", request=request)
    return summary
