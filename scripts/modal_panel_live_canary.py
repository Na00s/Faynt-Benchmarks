#!/usr/bin/env python3
"""Preparation and explicit owned-context boundary for a native tape gate.

Importing this module performs no asset reads, policy imports or process work.
There is no execution CLI or default owned context. The separately reviewed
context must verify every admission before the explicit launch boundary. All
outputs remain research-only, with zero policy inference or benchmark writes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import modal_panel_linux_host as host

SCHEMA = "e011.modal-panel-live-tape.plan.v1"
TAPE_SCHEMA = "e011.modal-panel-controller-tape.v1"
FIRST_FRAME = -123
FRAMES = 320
LAG = 1
MAX_PLAN_BYTES = 512 * 1024
SOURCE_PATHS = (
    "scripts/modal_panel_live_canary.py",
    "scripts/modal_panel_live_context.py",
    "scripts/modal_panel_linux_host.py",
    "scripts/modal_panel_package_audit.py",
    "scripts/modal_panel_linux_build.py",
    "src/melee_policy/integration/match_runtime.py",
    "src/melee_policy/integration/slippi_ai_policy.py",
    "src/melee_policy/integration/slippi_match.py",
    "src/melee_policy/integration/frisson_match.py",
    "src/melee_policy/integration/frisson_slippi_match.py",
    "configs/integration.toml",
    "patches/slippi-dolphin-two-pipe-frame-sync.patch",
    "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/manifest.json",
)
ADMISSION_KEYS = {"policy_parity", "linux_runtime", "game_image_access", "live_execution"}
CONTROLLER_CHECKS = {
    "full_causal_overlap", "trace_frames_consecutive", "replay_covers_every_observed_trace_state",
    "terminal_unobservable_tail_bounded_by_lag", "both_slots_aligned", "both_slots_have_controller_evidence",
    "both_slots_physical_buttons_exact", "both_slots_processed_upstream_buttons_exact",
    "both_slots_intended_raw_main_stick_exact", "both_slots_processed_c_stick_within_tolerance",
    "both_slots_physical_analog_shoulders_within_tolerance", "both_slots_first_gameplay_latch_rule_exact",
    "both_slots_first_gameplay_command_carryover_rule_exact", "both_slots_game_start_controller_latch_rule_exact",
    "first_policy_frame_mapping_evidence_exact", "game_start_transport_proof_exact", "both_slots_no_dispatch_rule_exact",
}
TRANSPORT_CHECKS = {
    "controller_pipe_lockstep_installed", "controller_pipe_registration_exact", "controller_pipe_primed_exactly_once",
    "controller_pipe_device_order_complete", "controller_pipe_no_pending_boundaries_after_shutdown",
    "controller_pipe_every_boundary_committed_exactly_once", "controller_pipe_every_scheduled_boundary_consumed_exactly_once",
    "controller_pipe_every_group_flushes_both_ports", "controller_pipe_one_preamble_decision_per_step",
}
IMAGE = {"byte_length": 1459978240,
         "sha256": "0de05981a34156b9cedcef73c73d4244ac05cf6149ab3c9cfed917698819e464",
         "disc_game_id": "GALE01", "disc_revision": 2}


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def identity(value):
    body = canonical_bytes(value)
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _command(*, main=(0.5, 0.5), c=(0.5, 0.5), left=0.0, right=0.0, buttons=()):
    return {"main_stick": list(main), "c_stick": list(c), "analog_l": left,
            "analog_r": right, "buttons": list(buttons)}


def canonical_tape():
    """A deterministic two-channel tape, repeated after startup for coverage.

Each 16-frame block holds a command for eight frames, then releases it for
eight. Ten distinct blocks repeat twice. Every value is canonical [0,1]; the
unchanged native sender applies libmelee's analog correction exactly once.
"""
    neutral = _command()
    pairs = (
        (neutral, neutral),
        (_command(buttons=("A",)), _command(buttons=("B",))),
        (_command(buttons=("X",)), _command(buttons=("Y",))),
        (_command(main=(0.0, 0.5)), _command(main=(1.0, 0.5))),
        (_command(main=(0.5, 1.0)), _command(main=(0.5, 0.0))),
        (_command(main=(0.0, 1.0)), _command(main=(1.0, 0.0))),
        (_command(c=(0.0, 1.0)), _command(c=(1.0, 0.0))),
        (_command(left=0.5), _command(right=1.0)),
        (_command(buttons=("Z", "L")), _command(buttons=("Z", "R"))),
        (_command(main=(0.25, 0.75), c=(0.75, 0.25), left=1.0,
                  buttons=("A", "X", "D_UP")),
         _command(main=(0.75, 0.25), c=(0.25, 0.75), right=0.5,
                  buttons=("B", "Y", "D_UP"))),
    )
    rows = []
    for index in range(FRAMES):
        a, b = pairs[(index // 16) % len(pairs)] if index % 16 < 8 else (neutral, neutral)
        rows.append({"index": index, "source_frame": FIRST_FRAME + index,
                     "a": copy.deepcopy(a), "b": copy.deepcopy(b)})
    return {"schema": TAPE_SCHEMA, "frames": FRAMES, "first_source_frame": FIRST_FRAME,
            "encoding": "canonical-complete-controller-before-native-analog-correction",
            "policy_inference_calls": 0, "rows": rows}


def _file_identity(value):
    if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int or not 0 < value["bytes"] <= 128 * 1024**2
            or not isinstance(value["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", value["sha256"])):
        raise ValueError("bounded file identity required")


def make_plan(*, label, created_at, expires_at, source_identities, admissions,
              runtime_image, game_image, udp_ports):
    """Bind caller-supplied identities without opening any referenced assets.

    Admission hashes are requirements for execution, not admission decisions.
    The future runner must inspect those exact successful/authorized receipts.
    """
    if not isinstance(label, str) or not re.fullmatch("[a-z0-9][a-z0-9-]{0,63}", label):
        raise ValueError("bounded artifact label required")
    if (type(created_at) is not int or type(expires_at) is not int
            or created_at <= 0 or not 0 < expires_at - created_at <= 900):
        raise ValueError("absolute execution window must be positive and at most 900 seconds")
    if not isinstance(source_identities, dict) or set(source_identities) != set(SOURCE_PATHS):
        raise ValueError("exact native source identity inventory required")
    if not isinstance(admissions, dict) or set(admissions) != ADMISSION_KEYS:
        raise ValueError("explicit policy/runtime/image/resource admission identities required")
    for value in (*source_identities.values(), *admissions.values()):
        _file_identity(value)
    if (not isinstance(runtime_image, dict)
            or set(runtime_image) != {"image_id", "manifest_sha256", "platform"}
            or runtime_image["platform"] != "linux_x86_64"
            or not isinstance(runtime_image["image_id"], str)
            or not re.fullmatch("im-[A-Za-z0-9]{1,100}", runtime_image["image_id"])
            or not re.fullmatch("[0-9a-f]{64}", str(runtime_image["manifest_sha256"]))):
        raise ValueError("explicit pinned Linux runtime image required")
    if canonical_bytes(game_image) != canonical_bytes(IMAGE):
        raise ValueError("frozen NTSC 1.02 image identity required")
    if (type(udp_ports) is not list or len(udp_ports) != 2 or len(set(udp_ports)) != 2
            or any(type(p) is not int or not 1024 <= p <= 65535 for p in udp_ports)):
        raise ValueError("two distinct nonprivileged physical-case UDP ports required")
    tape = canonical_tape()
    plan = {
        "schema": SCHEMA, "label": label, "created_at": created_at, "expires_at": expires_at,
        "source_identities": copy.deepcopy(source_identities), "required_admissions": copy.deepcopy(admissions),
        "source_identity_scope": "entry-hooks-and-owner; complete-import-closure-required-in-linux-runtime-admission",
        "runtime_image": copy.deepcopy(runtime_image), "game_image": copy.deepcopy(game_image),
        "package": {"archive": {"bytes": host.ARCHIVE[0], "sha256": host.ARCHIVE[1]},
                    "manifest": {"bytes": host.MANIFEST[0], "sha256": host.MANIFEST[1]},
                    "audit_receipt": {"bytes": host.AUDIT_RECEIPT[0], "sha256": host.AUDIT_RECEIPT[1]}},
        "tape": tape, "tape_identity": identity(tape),
        "cases": [{"index": index, "a_port": a_port, "b_port": 3 - a_port,
                   "slippi_port": udp_ports[index]} for index, a_port in enumerate((1, 2))],
        "scientific": {"characters": {"p1": "FOX", "p2": "FOX"}, "stage": "FINAL_DESTINATION",
                       "cpu_level": 0, "menu_autostart_port": 2, "frozen_stadium": True,
                       "controller_replay_lag_frames": LAG, "final_source_frame": FIRST_FRAME + FRAMES - 1,
                       "required_observed_and_replay_frame": FIRST_FRAME + FRAMES,
                       "save_slp": True, "save_video": False, "policy_inference_calls": 0,
                       "controller_dispatch": "native-send_canonical_controller-flush-false",
                       "transport": "native-_ControllerPipeLockstep-v7",
                       "menu_timeout_seconds": 120.0, "replay_finalize_timeout_seconds": 30.0,
                       "shutdown": "native-_stop_console-first-SIGINT-input-drain"},
        "execution": {"sequential": True, "parallel_cases": 1, "retry": False,
                      "external_absolute_deadline_required": True, "fresh_owned_host_required": True,
                      "process_ownership": "reviewed-Runner-parent-single-child-session-descendants-inherit",
                      "leader_pid_pgid_sid_equal": True, "parent_waitid_wnowait_required": True,
                      "max_trace_bytes_per_case": 16 * 1024**2, "max_total_output_bytes": 128 * 1024**2,
                      "benchmark_ledger_writes": False, "partial_evidence": "retain-unadmitted"},
        "claims": {"competitive_games": 0, "policy_inference_calls": 0,
                   "gameplay_admitted": False, "cross_host_trajectory_equivalence": "requires-matched-start-and-emulator-RNG"},
    }
    if len(canonical_bytes(plan)) > MAX_PLAN_BYTES:
        raise ValueError("plan byte bound exceeded")
    return plan


def validate_plan(plan, *, expected_sha256, now):
    if not isinstance(plan, dict) or identity(plan)["sha256"] != expected_sha256:
        raise ValueError("frozen plan identity differs")
    rebuilt = make_plan(label=plan["label"], created_at=plan["created_at"], expires_at=plan["expires_at"],
                        source_identities=plan["source_identities"], admissions=plan["required_admissions"],
                        runtime_image=plan["runtime_image"], game_image=plan["game_image"],
                        udp_ports=[case["slippi_port"] for case in plan["cases"]])
    if canonical_bytes(rebuilt) != canonical_bytes(plan):
        raise ValueError("plan contains changed fixed semantics or extra fields")
    if type(now) not in (int, float) or not math.isfinite(now) or not plan["created_at"] <= now < plan["expires_at"]:
        raise TimeoutError("frozen absolute plan window is not active")
    return copy.deepcopy(plan)


def write_new_plan(path, plan):
    """Explicit exclusive plan write only; no parent directories are created."""
    body = canonical_bytes(plan)
    validate_plan(plan, expected_sha256=hashlib.sha256(body).hexdigest(), now=plan["created_at"])
    path = Path(path).absolute()
    if path.resolve(strict=False) != path or path.parent.resolve(strict=True) != path.parent:
        raise ValueError("canonical new plan path required")
    with path.open("xb") as stream:
        if stream.write(body) != len(body): raise OSError("short plan write")
        stream.flush(); os.fsync(stream.fileno())
    return identity(plan)


def physical_commands(plan, case_index, tape_index):
    if type(case_index) is not int or case_index not in (0, 1):
        raise ValueError("exact physical case index required")
    if type(tape_index) is not int or not 0 <= tape_index < FRAMES:
        raise ValueError("exact tape index required")
    case, row = plan["cases"][case_index], plan["tape"]["rows"][tape_index]
    return {case["a_port"]: copy.deepcopy(row["a"]), case["b_port"]: copy.deepcopy(row["b"])}


def tape_receipt_checks(*, observed_frames, controller_audit, transport_checks, policy_calls, shutdown_method):
    """Tighten the native lag checker only for this short tape's observable tail."""
    gate = controller_audit.get("gate", {})
    native_checks = gate.get("checks", {})
    alignment = controller_audit.get("alignment", {}) or {}
    return {
        "320_tape_frames_plus_one_observation": type(observed_frames) is list and all(type(f) is int for f in observed_frames) and observed_frames == list(range(FIRST_FRAME, FIRST_FRAME + FRAMES + 1)),
        "zero_policy_inference": type(policy_calls) is int and policy_calls == 0,
        "native_controller_gate": gate.get("decision") == "pass" and set(native_checks) == CONTROLLER_CHECKS and all(v is True for v in native_checks.values()),
        "native_lag_one": type(alignment.get("steady_state_trace_to_replay_lag_frames")) is int and alignment["steady_state_trace_to_replay_lag_frames"] == LAG,
        "no_unobservable_terminal_command": alignment.get("permitted_terminal_unobservable_trace_frames") == [],
        "final_command_observed_in_replay": canonical_bytes(alignment.get("last_aligned_pair")) == canonical_bytes([FIRST_FRAME + FRAMES - 1, FIRST_FRAME + FRAMES]),
        "native_transport_gate": set(transport_checks) == TRANSPORT_CHECKS and all(v is True for v in transport_checks.values()),
        "first_sigint_finalized": shutdown_method == "child-sigint-input-drain",
    }


def drive_tape(plan, case_index, *, transport, menu_helpers, native, write_trace, check_deadline):
    """Use the original transport/sender against an already owned Console.

    The launch coordinator supplies source-verified native modules and a hard
    external deadline. This function never starts a process. It reads one final
    state after command196; that state197 receives no new tape command.
    """
    case = plan["cases"][case_index]
    m, d, f, melee = native.mimic, native.dispatch, native.frisson, native.melee
    controllers = transport.controllers
    observed, started, first_context = [], time.monotonic(), None
    transport.prime()
    while len(observed) <= FRAMES:
        check_deadline()
        if time.time() >= plan["expires_at"]:
            raise TimeoutError("immutable tape deadline exhausted")
        state = transport.step()
        check_deadline()
        if state is None:
            if not observed and time.monotonic() - started > 120.0:
                raise TimeoutError("native menu state timeout")
            continue
        if state.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if observed:
                raise RuntimeError("game ended before complete controller tape and lag drain")
            if time.monotonic() - started > 120.0:
                raise TimeoutError("native menu navigation timeout")
            transport.begin_boundary(reason="menu", game_frame=None)
            for port in (1, 2):
                menu_helpers[port].menu_helper_simple(state, controllers[port], melee.Character.FOX,
                    melee.Stage.FINAL_DESTINATION, cpu_level=0, autostart=port == 2, frozen_stadium=True)
                controllers[port].flush()
            transport.commit_boundary()
            continue
        if state.menu_state == melee.Menu.SUDDEN_DEATH:
            raise RuntimeError("unexpected sudden death in controller tape")
        frame = int(state.frame)
        m._require_exact_player_ports(state, frame)
        if frame != FIRST_FRAME + len(observed):
            raise RuntimeError("controller tape frames are not exactly consecutive from -123")
        if first_context is None:
            if (state.stage != melee.Stage.FINAL_DESTINATION
                    or any(state.players[p].character != melee.Character.FOX for p in (1, 2))):
                raise RuntimeError("controller tape first-state stage/characters differ")
            first_context = {"frame": frame, "stage": "FINAL_DESTINATION",
                             "players": {f"p{p}": f._player_state(state.players[p]) for p in (1, 2)}}
        observed.append(frame)
        if len(observed) == FRAMES + 1:
            break
        index = len(observed) - 1
        commands = {port: d.CanonicalControllerCommand(**value)
                    for port, value in physical_commands(plan, case_index, index).items()}
        transport.begin_boundary(reason="controller-tape-gameplay", game_frame=frame)
        slots = {}
        for port in (1, 2):
            dispatch = d.send_canonical_controller(controllers[port], commands[port], flush=False).as_dict()
            dispatch.update(called=True, queued_after_both_tape_commands_ready=True)
            slots[f"p{port}"] = {"port": port, "model": "controller-tape", "agent_kind": "controller-tape", "checkpoint": None,
                "tape_channel": "a" if port == case["a_port"] else "b",
                "command": commands[port].as_dict(), "controller_dispatch": dispatch,
                "player_state": f._player_state(state.players[port]),
                "inference": {"called": False, "policy_inference_calls": 0}}
            transport.schedule_next_boundary(port, reason="canonical-tape-next-frame-command", game_frame=frame)
        transport.commit_boundary()
        write_trace({"schema_version": "e011.controller-tape.trace.v1", "game_frame": frame,
                     "tape_index": index, "policy_inference_calls": 0, "slots": slots})
    return {"observed_frames": observed, "first_context": first_context,
            "policy_calls": 0, "tape_frames": FRAMES, "lag_observation_frames": 1}


def audit_projection(row):
    """Select the native canonical schema while preserving true tape identity."""
    result = copy.deepcopy(row)
    result["schema_version"] = "e011.controller-tape.canonical-schema-audit-projection.v1"
    for port in (1, 2):
        slot = result["slots"][f"p{port}"]
        if slot.get("agent_kind") != "controller-tape" or slot.get("inference", {}).get("called") is not False:
            raise ValueError("projection requires explicit zero-inference controller tape")
        slot.update(model="frisson-ai", checkpoint=None,
                    model_field_role="canonical-command-schema-selector-only")
    return result


def validate_tape_row(plan, case_index, index, row):
    if (row.get("game_frame") != FIRST_FRAME + index or type(row.get("game_frame")) is not int
            or type(row.get("tape_index")) is not int or row.get("tape_index") != index
            or type(row.get("policy_inference_calls")) is not int or row.get("policy_inference_calls") != 0
            or set(row.get("slots", {})) != {"p1", "p2"}):
        raise ValueError("trace index/frame/physical slots differ from frozen tape")
    expected = physical_commands(plan, case_index, index)
    for port in (1, 2):
        slot = row["slots"][f"p{port}"]
        if (type(slot.get("port")) is not int or slot["port"] != port
                or slot.get("model") != "controller-tape" or slot.get("agent_kind") != "controller-tape"
                or slot.get("checkpoint", "missing") is not None
                or slot.get("inference", {}).get("called") is not False
                or type(slot.get("inference", {}).get("policy_inference_calls")) is not int
                or slot.get("inference", {}).get("policy_inference_calls") != 0
                or slot.get("controller_dispatch", {}).get("called") is not True
                or canonical_bytes(slot.get("command")) != canonical_bytes(expected[port])):
            raise ValueError("trace command or identity differs from frozen tape")


def _new_receipt(path, value, cap=8 * 1024**2):
    body = canonical_bytes(value)
    if len(body) > cap:
        raise ValueError("receipt output bound exceeded")
    with path.open("xb") as stream:
        if stream.write(body) != len(body):
            raise OSError("short receipt write")
        stream.flush(); os.fsync(stream.fileno())
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _output_bound(directory):
    total = 0
    for index, path in enumerate(directory.rglob("*")):
        if index >= 2000 or path.is_symlink():
            raise RuntimeError("owned output entry or symlink bound")
        if path.is_file(): total += path.stat().st_size
        elif not path.is_dir(): raise RuntimeError("nonregular owned output")
    if total > 128 * 1024**2:
        raise RuntimeError("owned output byte bound exceeded")


def _run_prepared_case(plan, index, context, directory):
    """Internal case lifecycle; only run_prepared_plan may admit this call.

    Context owns the already admitted display/runtime and the reviewed Runner
    parent deadline. It checks native aliases/source identities and descendant
    session membership. The actual context implementation is a separate gate.
    """
    n = context.native
    m, f, s, melee = n.mimic, n.frisson, n.replay, n.melee
    replay_dir = directory / "replays"; replay_dir.mkdir(mode=0o700)
    primary = directory / "controller_tape_trace.jsonl"
    projection = directory / "controller_tape_canonical_schema_audit_projection.jsonl"
    case = plan["cases"][index]
    console = None
    transport = None
    loop = {"observed_frames": [], "policy_calls": 0}
    shutdown = "not-started"
    error = None
    trace_rows = 0
    receipt = {"schema": "e011.controller-tape.case.v1", "case": case,
               "plan_sha256": identity(plan)["sha256"], "agent_kind": "controller-tape",
               "policy_inference_calls": 0, "competitive_games": 0, "gameplay_admitted": False}
    def check():
        context.check_deadline(plan["expires_at"])
        if time.time() >= plan["expires_at"]: raise TimeoutError("immutable tape deadline exhausted")
        _output_bound(directory)
    try:
        check()
        m._assert_udp_port_available(case["slippi_port"])
        options = f._console_options(replay_dir, case["slippi_port"])
        console = host.create_console(context.package_root, context.package_manifest, context.package_audit,
            replay_directory=replay_dir, slippi_port=case["slippi_port"], console_module=n.console_module,
            version_lock=context.version_lock, console_options=options)
        receipt["host"] = dict(console._modal_panel_linux_host)
        # Exact original launch method and options. Parent Runner owns the group.
        m._disable_attested_dolphin_stop_hotkey(console)
        raw = {port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD) for port in (1, 2)}
        menus = {port: melee.MenuHelper() for port in (1, 2)}
        check()
        console.run(iso_path=str(context.game_image_path))
        receipt["dolphin_process"] = context.verify_inherited_process(console._process, plan)
        if not console.connect() or not all(raw[p].connect() for p in (1, 2)):
            raise RuntimeError("native Console/controller connection failed")
        check()
        transport = m._ControllerPipeLockstep.install(console, raw)
        with primary.open("xb") as first, projection.open("xb") as projected:
            counts = [0, 0]
            def write_trace(row):
                nonlocal trace_rows
                validate_tape_row(plan, index, trace_rows, row)
                for which, stream, value in ((0, first, row), (1, projected, audit_projection(row))):
                    body = canonical_bytes(value)
                    if counts[which] + len(body) > 16 * 1024**2: raise RuntimeError("per-trace output bound exceeded")
                    if stream.write(body) != len(body): raise OSError("short trace write")
                    counts[which] += len(body); stream.flush()
                trace_rows += 1
            loop = drive_tape(plan, index, transport=transport, menu_helpers=menus, native=n,
                              write_trace=write_trace, check_deadline=check)
            for stream in (first, projected): stream.flush(); os.fsync(stream.fileno())
    except BaseException as caught:
        error = f"{type(caught).__name__}: {str(caught)[:2048]}"
    finally:
        if console is not None:
            try:
                shutdown = m._stop_console(console, 30.0)
            except BaseException as caught:
                error = (error or "") + f"; shutdown {type(caught).__name__}: {str(caught)[:1024]}"
    receipt.update(loop=loop, trace_rows=trace_rows, shutdown_method=shutdown, error=error)
    try:
        check()
        if transport is not None:
            record, checks = transport.seal_benchmark_audit()
            receipt.update(transport=record, transport_checks=checks)
        if primary.exists() and projection.exists():
            receipt["trace_identity"] = {name: {"bytes": p.stat().st_size, "sha256": host.audit.digest(p, host.audit.Budget(120), 16 * 1024**2)[1]}
                                         for name, p in (("primary", primary), ("schema_projection", projection))}
        paths, replay_records = f._collect_replays(replay_dir, set(), context.project_root,
                                                  FIRST_FRAME, FIRST_FRAME + FRAMES, False)
        receipt["replays"] = replay_records
        replay_checks = m._legacy_replay_gate_checks(replay_records, sudden_death_transition_observed=False)
        receipt["native_replay_checks"] = replay_checks
        selected = [i for i, value in enumerate(replay_records) if value.get("tournament_result_replay") is True]
        if len(selected) != 1:
            raise RuntimeError("exactly one trace-covering tape replay required")
        proof = s._game_start_transport_proof(receipt["transport"])
        candidate = s._audit_controller_boundary_candidate(projection, paths[selected[0]], context.project_root,
                                                            lag_frames=1, game_start_transport_proof=proof)
        receipt.update(native_controller_audit=candidate,
                       native_audit_model_field_role="canonical-command-schema-selector-only",
                       game_start_transport_proof=proof,
                       game_start_rng=context.read_game_start_rng(paths[selected[0]]))
        receipt["checks"] = tape_receipt_checks(observed_frames=loop["observed_frames"],
            controller_audit=candidate, transport_checks=receipt["transport_checks"],
            policy_calls=0, shutdown_method=shutdown)
        receipt["checks"]["no_runtime_error"] = error is None
        receipt["checks"]["exact_frozen_trace_rows"] = trace_rows == FRAMES
        receipt["checks"]["native_replay_gate"] = bool(replay_checks) and all(v is True for v in replay_checks.values())
    except BaseException as caught:
        receipt["audit_error"] = f"{type(caught).__name__}: {str(caught)[:2048]}"
    receipt["status"] = "controller-tape-case-passed" if receipt.get("checks") and all(receipt["checks"].values()) else "failed"
    receipt["cross_host_trajectory_equivalence"] = "unassessed-requires-matched-initial-state-and-emulator-RNG"
    _new_receipt(directory / "case.json", receipt)
    return receipt


def run_prepared_plan(plan, *, expected_sha256, context):
    """Explicit fresh-plan boundary for a future reviewed owned-host context.

    No default context or CLI exists. Context.verify_and_prepare must verify
    admission bodies, source files and pristine native aliases, native image
    identity via _resolve_game_image_path, loaded shared libraries and display,
    actual PID=PGID=SID ownership, and the Runner parent's WNOWAIT/deadline.
    Xvfb and Dolphin inherit that one owned session. Context code itself must
    be approved before this interface can be used outside synthetic tests.
    """
    plan = validate_plan(plan, expected_sha256=expected_sha256, now=time.time())
    root = Path(context.research_output_root).absolute()
    if root.resolve(strict=True) != root or "research" not in root.parts:
        raise ValueError("canonical separate research-only output root required")
    directory = root / plan["label"]
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("tape plan already has intent or partial output")
    directory.mkdir(mode=0o700)
    _new_receipt(directory / "intent.json", {"plan_sha256": expected_sha256, "cases": 2,
        "retry": False, "expires_at": plan["expires_at"], "status": "intent-before-admission"})
    result = {"schema": "e011.controller-tape.run.v1", "plan_sha256": expected_sha256,
              "agent_kind": "controller-tape", "policy_inference_calls": 0,
              "competitive_games": 0, "gameplay_admitted": False, "cases": [], "status": "failed"}
    try:
        owner = context.verify_and_prepare(plan, directory)
        if (owner.get("status") != "owned-linux-tape-context-ready" or owner.get("plan_sha256") != expected_sha256
                or owner.get("admissions") != plan["required_admissions"]
                or owner.get("runtime_image") != plan["runtime_image"]):
            raise RuntimeError("owned context admission does not bind the exact plan")
        result["owner"] = owner
        for index in (0, 1):
            context.check_deadline(plan["expires_at"])
            case_dir = directory / f"case-{index}"; case_dir.mkdir(mode=0o700)
            value = _run_prepared_case(plan, index, context, case_dir)
            result["cases"].append({"index": index, "status": value["status"],
                                    "receipt": identity(value)})
            if value["status"] != "controller-tape-case-passed": break
        if len(result["cases"]) == 2 and all(v["status"] == "controller-tape-case-passed" for v in result["cases"]):
            result["status"] = "controller-tape-local-transport-passed"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:2048]}"
    finally:
        # The external Runner performs authoritative group cleanup even if this
        # child hangs or cannot write this final receipt.
        result["parent_group_cleanup_required"] = True
    _output_bound(directory)
    _new_receipt(directory / "result.json", result)
    return result
