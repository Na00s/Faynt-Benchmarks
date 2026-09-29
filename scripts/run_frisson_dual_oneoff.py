#!/usr/bin/env python3
"""Run one frozen-final 10M-versus-75M Frisson Fox match in exact lockstep."""

from __future__ import annotations

import argparse
import json
import platform
import random
import signal
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch
from final_winner_batch_common import (
    FINAL_10M,
    FINAL_75M,
    POSTTRAINING_10M,
    POSTTRAINING_75M,
    POSTTRAINING_BENCHMARK,
    PRETRAINING_10M,
    PRETRAINING_75M,
    FinalCheckpoint,
    POST_RL_FINALS,
)

from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.frisson_match import (
    CONTROLLER_REPLAY_LAG_FRAMES,
    FIRST_POLICY_FRAME,
    MATCH_STAGE,
    FrissonMatchRequest,
    _collect_replays,
    _console_options,
    _frisson_policy_config,
    _player_state,
    _resolve_checkpoint,
    _selected_checkpoint_contract,
)
from melee_policy.integration.frisson_policy import (
    FrissonPolicySession,
    inspect_frisson_checkpoint,
)
from melee_policy.integration.match_runtime import (
    _assert_udp_port_available,
    _attested_emulator_release,
    _ControllerPipeLockstep,
    _create_attested_dolphin_console,
    _display_path,
    _emulator_application_identity,
    _file_identity,
    _game_image_identity,
    _launch_and_connect_attested_dolphin,
    _legacy_replay_gate_checks,
    _load_config,
    _require_exact_player_ports,
    _require_unused_artifact_label,
    _resolve_game_image_path,
    _runtime_reproducibility_record,
    _stop_console,
    _validate_evaluation_seed,
    _write_json,
)
from melee_policy.integration.natural_game_end import has_decisive_zero_stock
from melee_policy.integration.slippi_ai_policy import (
    CanonicalControllerCommand,
    send_canonical_controller,
)
from melee_policy.integration.slippi_match import (
    _audit_controller_boundary_records,
    _game_start_transport_proof,
    _read_replay_controller_states,
    _read_trace_rows,
)

SCHEMA_VERSION = "integration.frisson_vs_frisson.v2"
TRACE_SCHEMA_VERSION = "integration.frisson_vs_frisson.controller_trace.v1"
DEFAULT_SEED = 101
LABEL = (
    f"final-10m-posttrained-step195248-v-75m-posttrained-step127214-s{DEFAULT_SEED}"
    if POSTTRAINING_BENCHMARK
    else f"final-10m-step122064-v-75m-step86016-s{DEFAULT_SEED}"
)
CHARACTER = "FOX"
FINAL_CHECKPOINTS = {"10m": FINAL_10M, "75m": FINAL_75M}
CHECKPOINT_CONTRACTS = {
    "10m-postrl": POST_RL_FINALS["10m"],
    "75m-postrl": POST_RL_FINALS["75m"],
    "10m-pretraining": PRETRAINING_10M,
    "75m-pretraining": PRETRAINING_75M,
    "10m-posttraining": POSTTRAINING_10M,
    "75m-posttraining": POSTTRAINING_75M,
}
DEFAULT_P1_CONTRACT = "10m-posttraining" if POSTTRAINING_BENCHMARK else "10m-pretraining"
DEFAULT_P2_CONTRACT = "75m-posttraining" if POSTTRAINING_BENCHMARK else "75m-pretraining"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/integration.toml"),
    )
    parser.add_argument("--iso-path", type=Path)
    parser.add_argument(
        "--p1-checkpoint",
        type=Path,
        default=FINAL_10M.path,
    )
    parser.add_argument(
        "--p2-checkpoint",
        type=Path,
        default=FINAL_75M.path,
    )
    parser.add_argument(
        "--p1-contract",
        choices=tuple(CHECKPOINT_CONTRACTS),
        default=DEFAULT_P1_CONTRACT,
    )
    parser.add_argument(
        "--p2-contract",
        choices=tuple(CHECKPOINT_CONTRACTS),
        default=DEFAULT_P2_CONTRACT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-label", default=LABEL)
    return parser


def _timed_step(
    session: FrissonPolicySession,
    gamestate: Any,
) -> tuple[CanonicalControllerCommand, float]:
    started = time.perf_counter()
    command = session.step(gamestate)
    elapsed = time.perf_counter() - started
    command.validate()
    return command, elapsed


def _private_policy_rng_checks(
    metadata: dict[int, dict[str, Any]],
    *,
    seed: int,
) -> tuple[bool, bool]:
    """Read the public Frisson metadata contract without assuming native internals."""
    private_generators = True
    matching_seeds = True
    for value in metadata.values():
        runtime_contract = value.get("runtime_contract")
        upstream_runtime = value.get("upstream_runtime")
        if not isinstance(runtime_contract, dict) or not isinstance(upstream_runtime, dict):
            return False, False
        execution = upstream_runtime.get("execution")
        execution = execution if isinstance(execution, dict) else {}
        declared_seed = runtime_contract.get("private_torch_generator_seed")
        private_generators = private_generators and (
            execution.get("private_torch_generator") is True
            or declared_seed == seed
        )
        matching_seeds = matching_seeds and (
            runtime_contract.get("evaluation_seed") == seed
            and declared_seed == seed
            and execution.get("generator_seed", seed) == seed
        )
    return private_generators, matching_seeds


def _scoring_role_keys(
    profiles: dict[int, str],
    checkpoint_contract_keys: dict[int, str],
) -> dict[int, str]:
    """Keep same-profile checkpoint generations distinct in score dictionaries."""

    roles = (
        dict(profiles)
        if profiles[1] != profiles[2]
        else dict(checkpoint_contract_keys)
    )
    if len(set(roles.values())) != 2:
        raise RuntimeError("Frisson checkpoint pair must have two distinct scoring roles")
    return roles


def _trace_row(
    gamestate: Any,
    commands: dict[int, CanonicalControllerCommand],
    inference_seconds: dict[int, float],
    dispatches: dict[int, dict[str, Any]],
    barrier_seconds: float,
    model_labels: dict[int, str],
) -> dict[str, Any]:
    game_frame = int(gamestate.frame)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "game_frame": game_frame,
        "barrier": {
            "source_frame": game_frame,
            "p1_current_frame_inference_complete": True,
            "p2_current_frame_inference_complete": True,
            "controller_transaction_opened_after_both": True,
            "seconds": barrier_seconds,
        },
        "slots": {
            f"p{port}": {
                "port": port,
                "model": "frisson-ai",
                "display_model": model_labels[port],
                "requested_character": CHARACTER,
                "player_state": _player_state(gamestate.players[port]),
                "command": commands[port].as_dict(),
                "inference": {
                    "called": True,
                    "source_frame": game_frame,
                    "command_age_frames": 0,
                    "seconds": inference_seconds[port],
                    "state_frame": "t",
                    "command_frame": "t+1",
                    "action_offset_frames": 1,
                    "delay_frames": 0,
                    "native_dummy_prefix": False,
                },
                "controller_dispatch": dispatches[port],
            }
            for port in (1, 2)
        },
    }


def _first_context(gamestate: Any) -> dict[str, Any]:
    stage = str(getattr(gamestate.stage, "name", gamestate.stage)).split(".")[-1]
    characters = {
        f"p{port}": str(
            getattr(gamestate.players[port].character, "name", gamestate.players[port].character)
        ).split(".")[-1]
        for port in (1, 2)
    }
    checks = {
        "first_frame_minus_123": int(gamestate.frame) == FIRST_POLICY_FRAME,
        "final_destination": stage == MATCH_STAGE,
        "both_fox": characters == {"p1": CHARACTER, "p2": CHARACTER},
    }
    if not all(checks.values()):
        raise RuntimeError(f"first Frisson head-to-head context mismatch: {checks}")
    return {
        "frame": int(gamestate.frame),
        "stage": stage,
        "characters": characters,
        "costumes": {
            f"p{port}": int(gamestate.players[port].costume) for port in (1, 2)
        },
        "checks": checks,
    }


def _run_match(
    *,
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    artifact_label: str,
    sessions: dict[int, FrissonPolicySession],
    identities: dict[int, dict[str, Any]],
    checkpoint_contracts: dict[int, FinalCheckpoint],
    role_keys: dict[int, str],
    reproducibility: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    import melee

    output_directory = (
        project_root / str(config["frisson_ai"]["output_directory"]) / artifact_label
    )
    _require_unused_artifact_label(output_directory, artifact_label)
    replay_directory = output_directory / "replays"
    trace_path = output_directory / "controller_trace.jsonl"
    summary_path = output_directory / "summary.json"
    output_directory.mkdir(parents=True, exist_ok=False)
    replay_directory.mkdir(parents=True, exist_ok=False)
    existing_replays: set[Path] = set()

    emulator_application = _emulator_application_identity(config, project_root)
    version = _attested_emulator_release(emulator_application)

    udp_port = int(config["emulator"]["slippi_port"])
    _assert_udp_port_available(udp_port)
    console = _create_attested_dolphin_console(
        config,
        project_root,
        emulator_application,
        **_console_options(replay_directory, udp_port),
    )
    raw_controllers = {
        port: melee.Controller(
            console=console,
            port=port,
            type=melee.ControllerType.STANDARD,
        )
        for port in (1, 2)
    }
    menu_helpers = {1: melee.MenuHelper(), 2: melee.MenuHelper()}
    max_game_frames = int(config["integration"]["max_game_frames"])
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    trace_stream: TextIO | None = None
    transport: _ControllerPipeLockstep | None = None
    started_at = time.time()
    in_game = False
    game_end_observed = False
    sudden_death_transition_observed = False
    termination = "not-started"
    exception: BaseException | None = None
    shutdown_method = "not-started"
    processed_frames = 0
    first_game_frame: int | None = None
    last_game_frame: int | None = None
    previous_game_frame: int | None = None
    first_context: dict[str, Any] | None = None
    frame_delta_counts: Counter[int] = Counter()
    profiles = {port: str(identities[port]["profile"]) for port in (1, 2)}
    model_labels = {port: checkpoint_contracts[port].name for port in (1, 2)}
    model_ports = {role_keys[port]: port for port in (1, 2)}
    inference_counts = {role_keys[port]: 0 for port in (1, 2)}
    dispatch_counts = {role_keys[port]: 0 for port in (1, 2)}
    inference_times: dict[str, list[float]] = {
        role_keys[port]: [] for port in (1, 2)
    }
    barrier_times: list[float] = []
    stocks = {role_keys[port]: 4 for port in (1, 2)}
    menu_flushes = {1: 0, 2: 0}
    no_frame_watchdog = InGameNoFrameWatchdog()

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frisson-policy")
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not all(controller.connect() for controller in raw_controllers.values()):
            raise RuntimeError("libmelee could not connect both virtual controllers")
        transport = _ControllerPipeLockstep.install(console, raw_controllers)
        controllers = transport.controllers
        transport.prime()
        print(
            f"P1={model_labels[1].upper()} FOX | "
            f"P2={model_labels[2].upper()} FOX | "
            f"stage={MATCH_STAGE} | seed={seed} | inference=exact-concurrent",
            flush=True,
        )

        while processed_frames < max_game_frames:
            gamestate = transport.step()
            no_frame_watchdog.observe_step_result(gamestate, gameplay_started=in_game)
            if gamestate is None:
                if not in_game and time.time() - started_at > menu_timeout:
                    raise TimeoutError("Slippi did not provide a menu or game state before timeout")
                continue
            if in_game and gamestate.menu_state == melee.Menu.SUDDEN_DEATH:
                game_end_observed = True
                sudden_death_transition_observed = True
                termination = "natural-game-end"
                break
            if gamestate.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if in_game:
                    game_end_observed = True
                    termination = "natural-game-end"
                    break
                if time.time() - started_at > menu_timeout:
                    raise TimeoutError("automatic menu navigation did not start a match before timeout")
                transport.begin_boundary(reason="menu", game_frame=None)
                for port in (1, 2):
                    menu_helpers[port].menu_helper_simple(
                        gamestate,
                        controllers[port],
                        melee.Character.FOX,
                        melee.Stage.FINAL_DESTINATION,
                        cpu_level=0,
                        autostart=port == 2,
                        frozen_stadium=True,
                    )
                    controllers[port].flush()
                    menu_flushes[port] += 1
                transport.commit_boundary()
                continue

            in_game = True
            game_frame = int(gamestate.frame)
            _require_exact_player_ports(gamestate, game_frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                first_context = _first_context(gamestate)
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] += 1
                if delta != 1:
                    raise RuntimeError(
                        "rendered policy frames are not consecutive: "
                        f"{previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame

            barrier_started = time.perf_counter()
            futures = {
                port: executor.submit(_timed_step, sessions[port], gamestate)
                for port in (1, 2)
            }
            done, pending = wait(tuple(futures.values()))
            if pending or len(done) != 2:
                raise RuntimeError("both Frisson current-frame inferences did not finish")
            commands: dict[int, CanonicalControllerCommand] = {}
            frame_inference_seconds: dict[int, float] = {}
            for port in (1, 2):
                commands[port], frame_inference_seconds[port] = futures[port].result()
            barrier_elapsed = time.perf_counter() - barrier_started

            transport.begin_boundary(
                reason="frisson-vs-frisson-gameplay",
                game_frame=game_frame,
            )
            dispatches: dict[int, dict[str, Any]] = {}
            for port in (1, 2):
                dispatch = send_canonical_controller(
                    controllers[port],
                    commands[port],
                    flush=False,
                ).as_dict()
                dispatch.update(
                    {
                        "called": True,
                        "queued_after_both_current_frame_inferences": True,
                    }
                )
                dispatches[port] = dispatch
                transport.schedule_next_boundary(
                    port,
                    reason="frisson-zero-delay-next-frame-command",
                    game_frame=game_frame,
                )
            transport.commit_boundary()

            trace_stream.write(
                json.dumps(
                    _trace_row(
                        gamestate,
                        commands,
                        frame_inference_seconds,
                        dispatches,
                        barrier_elapsed,
                        model_labels,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            trace_stream.flush()
            for port in (1, 2):
                role = role_keys[port]
                inference_counts[role] += 1
                dispatch_counts[role] += 1
                inference_times[role].append(frame_inference_seconds[port])
                stocks[role] = int(gamestate.players[port].stock)
            barrier_times.append(barrier_elapsed)
            processed_frames += 1
            if processed_frames % 60 == 0:
                print(
                    f"frame {game_frame}: P1 {stocks[role_keys[1]]} stocks, "
                    f"P2 {stocks[role_keys[2]]} stocks",
                    flush=True,
                )
            if has_decisive_zero_stock(gamestate, (1, 2)):
                game_end_observed = True
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= max_game_frames:
            termination = "frame-limit-before-natural-end"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        executor.shutdown(wait=True, cancel_futures=True)
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(
            console,
            float(config["integration"]["replay_finalize_timeout_seconds"]),
        )

    replay_paths, replay_records = _collect_replays(
        replay_directory,
        existing_replays,
        project_root,
        first_game_frame,
        last_game_frame,
        sudden_death_transition_observed,
    )
    replay_checks = _legacy_replay_gate_checks(
        replay_records,
        sudden_death_transition_observed=sudden_death_transition_observed,
    )
    if transport is None:
        transport_record = {"installed": False}
        transport_checks = {"controller_pipe_lockstep_installed": False}
    else:
        transport_record, transport_checks = transport.seal_benchmark_audit()
    game_start_transport_proof = _game_start_transport_proof(transport_record)
    controller_audit: dict[str, Any]
    selected = [
        index
        for index, record in enumerate(replay_records)
        if record.get("tournament_result_replay") is True
    ]
    try:
        if len(selected) != 1:
            raise RuntimeError("controller audit requires one trace-covering result replay")
        rows = _read_trace_rows(trace_path)
        controller_audit = _audit_controller_boundary_records(
            rows,
            _read_replay_controller_states(replay_paths[selected[0]]),
            lag_frames=CONTROLLER_REPLAY_LAG_FRAMES,
            game_start_transport_proof=game_start_transport_proof,
        )
        controller_audit["classification"] = (
            "both Frisson decoded commands verified at the shared controller boundary"
        )
        controller_audit["trace"] = {
            **_file_identity(trace_path, project_root),
            "rows": len(rows),
        }
        controller_audit["replay"] = _file_identity(
            replay_paths[selected[0]], project_root
        )
    except Exception as audit_error:
        controller_audit = {
            "gate": {"decision": "fail", "checks": {"available": False}},
            "error": f"{type(audit_error).__name__}: {audit_error}",
        }
    controller_checks = cast(
        dict[str, bool],
        cast(dict[str, Any], controller_audit["gate"])["checks"],
    )
    diagnostics = {port: sessions[port].diagnostics() for port in (1, 2)}
    metadata = {port: sessions[port].metadata() for port in (1, 2)}
    private_policy_rngs, runtime_seeds_match = _private_policy_rng_checks(
        metadata,
        seed=seed,
    )
    gate_checks: dict[str, bool] = {
        "exact_requested_checkpoint_pair": all(
            identities[port].get("sha256") == checkpoint_contracts[port].sha256
            for port in (1, 2)
        ),
        "final_destination": True,
        "evaluation_seed_valid": _validate_evaluation_seed(seed) == seed,
        "natural_game_end_observed": game_end_observed,
        "entered_gameplay": in_game,
        "processed_at_least_one_frame": processed_frames > 0,
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "strict_consecutive_policy_frames": set(frame_delta_counts) <= {1},
        "one_inference_per_model_per_frame": all(
            count == processed_frames for count in inference_counts.values()
        ),
        "one_dispatch_per_model_per_frame": all(
            count == processed_frames for count in dispatch_counts.values()
        ),
        "both_inferences_barrier_per_frame": len(barrier_times) == processed_frames,
        "both_session_frame_counts_exact": all(
            value.get("frames_total") == processed_frames for value in diagnostics.values()
        ),
        "both_action_offsets_one": all(
            value["action_contract"]["action_offset_frames"] == 1
            for value in metadata.values()
        ),
        "both_delays_zero": all(
            value["runtime_contract"]["delay_frames"] == 0
            for value in metadata.values()
        ),
        "both_temperatures_one": all(
            value["runtime_contract"]["sample_temperature"] == 1.0
            for value in metadata.values()
        ),
        "both_private_policy_rngs": private_policy_rngs,
        "both_runtime_seeds_match_game_seed": runtime_seeds_match,
        **replay_checks,
        **{
            f"controller_audit.{name}": passed
            for name, passed in controller_checks.items()
        },
        **transport_checks,
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "Frisson head-to-head gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )
    result = "complete" if exception is None and all(gate_checks.values()) else "failed"
    winner = (
        checkpoint_contracts[1].name
        if stocks[role_keys[1]] > stocks[role_keys[2]]
        else checkpoint_contracts[2].name
        if stocks[role_keys[2]] > stocks[role_keys[1]]
        else None
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "frame-exact frozen Frisson checkpoint pair",
        "result": result,
        "error": None if exception is None else f"{type(exception).__name__}: {exception}",
        "gate": {
            "decision": "pass" if result == "complete" else "fail",
            "checks": gate_checks,
        },
        "configuration": {
            "player_1": {"model": model_labels[1], "profile": profiles[1], "character": CHARACTER, "port": 1},
            "player_2": {"model": model_labels[2], "profile": profiles[2], "character": CHARACTER, "port": 2},
            "model_to_port": model_ports,
            "policy_evaluation_seeds": {
                role_keys[port]: seed for port in (1, 2)
            },
            "stage": MATCH_STAGE,
            "seed": seed,
            "require_natural_end": True,
            "blocking_input": True,
            "online_delay_frames": 0,
            "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "inference_mode": "synchronous-concurrent",
        },
        "checkpoints": {
            f"p{port}": {
                **_selected_checkpoint_contract(identities[port]),
                "display_name": model_labels[port],
                "validation_nll": (
                    {
                        122_064: 0.7968172955299399,
                        86_016: 0.7648214009464961,
                        195_248: 0.7574608703491693,
                        127_214: 0.7248941140799456,
                    }.get(checkpoint_contracts[port].step)
                ),
            }
            for port in (1, 2)
        },
        "policies": {
            f"p{port}": {"metadata": metadata[port], "diagnostics": diagnostics[port]}
            for port in (1, 2)
        },
        "reproducibility": reproducibility,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "libmelee_module": str(Path(melee.__file__).resolve()),
            "slippi_version": version,
        },
        "emulator_application": emulator_application,
        "game_image": _game_image_identity(iso_path),
        "execution": {
            "processed_policy_frames": processed_frames,
            "first_game_frame": first_game_frame,
            "last_game_frame": last_game_frame,
            "first_game_context": first_context,
            "frame_delta_counts": {
                str(delta): count for delta, count in sorted(frame_delta_counts.items())
            },
            "inference_counts": inference_counts,
            "dispatch_counts": dispatch_counts,
            "mean_inference_seconds": {
                name: sum(values) / len(values) if values else None
                for name, values in inference_times.items()
            },
            "maximum_inference_seconds": {
                name: max(values) if values else None
                for name, values in inference_times.items()
            },
            "mean_barrier_seconds": (
                sum(barrier_times) / len(barrier_times) if barrier_times else None
            ),
            "last_stocks": stocks,
            "winner": winner,
            "game_end_observed": game_end_observed,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "wall_seconds": time.time() - started_at,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {f"p{port}": menu_flushes[port] for port in (1, 2)},
            },
        },
        "controller_boundary_audit": controller_audit,
        "artifacts": {
            "trace": {
                **_file_identity(trace_path, project_root),
                "rows": processed_frames,
            },
            "replays": replay_records,
            "summary": _display_path(summary_path, project_root),
        },
    }
    _write_json(summary_path, summary)
    if exception is not None:
        raise RuntimeError(cast(str, summary["error"]))
    return summary


def main() -> None:
    arguments = _parser().parse_args()
    config, project_root = _load_config(arguments.config)
    seed = _validate_evaluation_seed(arguments.seed)
    checkpoints = {
        1: arguments.p1_checkpoint.expanduser().resolve(),
        2: arguments.p2_checkpoint.expanduser().resolve(),
    }
    checkpoint_contract_keys = {
        1: arguments.p1_contract,
        2: arguments.p2_contract,
    }
    checkpoint_contracts = {
        port: CHECKPOINT_CONTRACTS[key]
        for port, key in checkpoint_contract_keys.items()
    }
    configured = cast(dict[str, Any], config["frisson_ai"])
    for checkpoint in checkpoints.values():
        _resolve_checkpoint(configured, project_root, checkpoint)
    identities = {
        port: inspect_frisson_checkpoint(checkpoint)
        for port, checkpoint in checkpoints.items()
    }
    profiles = {port: str(identities[port]["profile"]) for port in (1, 2)}
    for port in (1, 2):
        expected = checkpoint_contracts[port]
        required = {
            "path": expected.path.resolve(),
            "sha256": expected.sha256,
            "byte_length": expected.byte_length,
            "step": expected.step,
            "processed_target_frames": expected.processed_target_frames,
            "parameter_count": expected.parameter_count,
            "all_state_tensors_finite": True,
        }
        observed = {**identities[port], "path": checkpoints[port]}
        mismatches = {
            name: {"observed": observed.get(name), "required": value}
            for name, value in required.items()
            if observed.get(name) != value
        }
        if mismatches:
            raise RuntimeError(f"P{port} final checkpoint identity mismatch: {mismatches}")
    role_keys = _scoring_role_keys(profiles, checkpoint_contract_keys)
    policy_configs = {}
    for port in (1, 2):
        base_request = FrissonMatchRequest(player_1_checkpoint=checkpoints[port], seed=seed)
        base = _frisson_policy_config(config, project_root, base_request)
        policy_configs[port] = replace(base, port=port, opponent_port=3 - port)
        policy_configs[port].validate()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    image_path = _resolve_game_image_path(config, project_root, arguments.iso_path)
    reproducibility = _runtime_reproducibility_record(
        project_root,
        arguments.config.resolve(),
        "requirements-e010.lock",
        (
            "scripts/run_frisson_dual_oneoff.py",
            "src/melee_policy/integration/frisson_policy.py",
            "src/melee_policy/integration/frisson_match.py",
            "src/melee_policy/integration/game_bundle.py",
            "src/melee_policy/integration/match_runtime.py",
            "src/melee_policy/integration/slippi_match.py",
            "src/melee_policy/integration/native/macos_mux_replay_audio.m",
            "src/melee_policy/integration/native/macos_replay_recorder.m",
            "src/melee_policy/integration/replay_video.py",
            "patches/slippi-dolphin-two-pipe-frame-sync.patch",
        ),
    )
    sessions = {port: FrissonPolicySession(policy_configs[port]) for port in (1, 2)}
    try:
        for port in (1, 2):
            print(f"Loading {checkpoint_contracts[port].name} on P{port}...", flush=True)
            sessions[port].start()
        summary = _run_match(
            config=config,
            project_root=project_root,
            iso_path=image_path,
            artifact_label=arguments.artifact_label,
            sessions=sessions,
            identities=identities,
            checkpoint_contracts=checkpoint_contracts,
            role_keys=role_keys,
            reproducibility=reproducibility,
            seed=seed,
        )
    finally:
        for port in (2, 1):
            sessions[port].close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
