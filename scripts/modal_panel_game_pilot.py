#!/usr/bin/env python3
"""Two explicit Linux full-policy compatibility games, outside the panel ledger.

Preparation requires a successful controller-tape pilot and exact authorization.
Each game runs the unchanged native public-panel entry point in a fresh owned
process. This module composes the reviewed runtime, host and transport helpers.
"""
from __future__ import annotations

import prepared_runtime

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import shutil
import time
import types
import uuid

import modal_panel_build_pilot as wire
import modal_panel_linux_build as guard
import modal_panel_linux_host as host
import modal_panel_live_canary as tape
import modal_panel_live_context as context
import modal_panel_live_pilot as live
import modal_panel_match_host as match_host
import modal_panel_match_source as match_source
import modal_panel_policy_pilot as policy
import modal_panel_runtime_pilot as runtime

SCHEMA = "e011.modal-full-game-pilot.v1"
APPROVAL_SCHEMA = "e011.modal-full-game-pilot-authorization.v1"
ROOT, COMPAT, REMOTE = runtime.ROOT, runtime.COMPAT, runtime.REMOTE
INPUT = Path("/opt/full-game-input")
PLAN = Path("/opt/full-game-plan.json")
WORK = Path("/work/research/full-game-pilot")
PARENT = WORK / "source"
PROJECT = PARENT / "melee-policy"
ENTRY = REMOTE / "scripts/modal_panel_game_pilot.py"
FUNCTION_SECONDS, SESSION_SECONDS, OWNER_SECONDS, GAME_SECONDS = 3600, 3900, 3510, 900
FILE_CAP, RETURN_CAP, FILE_COUNT = 256 * 1024**2, 512 * 1024**2, 256
UPLOAD_CAP = tape.IMAGE["byte_length"] + 900 * 1024**2
APP_NAME = "frisson-linux-full-policy-pair-v1"
PANEL = "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/manifest.json"
PANEL_SHA = "f80624560cc51283dc894787984de6089a42fd1e9f451ed2f1e845f4d33a6115"
SNAPSHOT_SHA = "83d261fb7988ee20bcf49dbd648b0fc2f75d41ddca1a86643996f19443ccc755"
CHECKPOINTS = {
    ".e010-cache/post-rl-p21/frisson-melee-75m-rl-step222.pt": {
        "bytes": 644283507, "sha256": "860b0d06789a5e52fcf44bf497c9f2838e387d7c20689bb455434048732883e0"},
    ".e001-cache/slippi-ai/models/fox_d21_ditto_v4": {
        "bytes": 42332034, "sha256": "f4854fc783246b42d4bfc9309d6964c174a388f3425bb21f3f79355a4562e727"},
}
FAYNT_BUNDLE_PREFIX = "sources/faynt/268031e7bddebb4e8c7a40026cd0b95f824d9d0d"
FAYNT_BUNDLE_PATHS = tuple(f"{FAYNT_BUNDLE_PREFIX}/{name}" for name in (
    "model.py", "controller_codec.py", "tensor_batch.py", "source-manifest.json"))
SOURCE_PATHS = tuple(dict.fromkeys((*live.SOURCE_PATHS, *FAYNT_BUNDLE_PATHS,
    *(f"src/melee_policy/integration/{n}.py" for n in (
        "frisson_slippi_match", "game_bundle", "slippi_compatibility", "replay_result", "play")),
    "src/melee_policy/e000/__init__.py", "src/melee_policy/e000/enums.py",
    *(f"scripts/{n}.py" for n in (
        "slippi_panel_runtime", "slippi_panel_audit", "run_slippi_public_panel",
        "run_final_75m_vs_slippi_ai_character_sweep", "modal_panel_match_host",
        "modal_panel_match_source", "modal_panel_game_pilot")),
    "requirements-e010.lock")))
LABELS = ("modal-compat-75m-d21-fd-p2", "modal-compat-75m-d21-fd-p1")
SOURCE_REVISION = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
SOURCE_URL = "https://github.com/vladfi1/slippi-ai"


def fixed_scope():
    return {"compatibility_games": 2, "competitive_games": 0, "benchmark_ledger_writes": False,
            "profile": "75m", "checkpoint_step": 222, "opponent": "fox_d21_ditto_v4",
            "opponent_name": "Cody", "characters": ["FOX", "FOX"], "stage": "FINAL_DESTINATION",
            "seed": 1971985417, "frisson_physical_ports": [2, 1], "native_max_game_frames": 30000,
            "policy_inference": True, "save_slp": True, "save_video": False,
            "numeric_classification": context.LIMITED_CLASSIFICATION,
            "cross_host_trajectory_equivalence": "unassessed"}


def limits():
    return {"function_seconds": FUNCTION_SECONDS, "session_seconds": SESSION_SECONDS,
            "owner_seconds": OWNER_SECONDS, "game_seconds": GAME_SECONDS,
            "cpu": [4, 4], "memory_mib": [16384, 16384], "max_containers": 1,
            "nonpreemptible": True, "client_resubmissions": 0,
            "platform_restarts": "possible; same absolute expiry retained",
            "return_bytes": RETURN_CAP, "return_file_bytes": FILE_CAP, "upload_bytes": UPLOAD_CAP}


def selected_games(panel):
    result = []
    for port in (2, 1):
        label = f"sp21-75m-fox_d21_ditto_v4-supported-mirror-fox-final_destination-p{port}"
        rows = [g for g in panel["games"] if g["label"] == label]
        if len(rows) != 1: raise ValueError("exact original Fox pair required")
        game = rows[0]
        expected = {"profile": "75m", "release": "fox_d21_ditto_v4", "block": "supported-mirror",
                    "frisson_character": "FOX", "slippi_character": "FOX", "player_name": "Cody",
                    "stage": "FINAL_DESTINATION", "seed": 1971985417, "frisson_port": port,
                    "slippi_port": 3-port, "save_slp": True, "save_video": False,
                    "pair_id": "11c2a0db80872079", "slippi_character_in_deployed_roster": True}
        if any(game.get(k) != v for k, v in expected.items()): raise ValueError("original pair specification differs")
        result.append(copy.deepcopy(game))
    return result


def definition(rows, output, prior, prerequisite, authorization_sha, games):
    return {"schema": SCHEMA, "uploads": rows, "output": str(output), "scope": fixed_scope(),
            "limits": limits(), "source_paths": list(SOURCE_PATHS), "games": games,
            "artifact_labels": list(LABELS), "prior_tape_plan_sha256": prior,
            "tape_prerequisite": prerequisite, "authorization_sha256": authorization_sha,
            "policy_plan_sha256": context.LIMITED_EVIDENCE_SHA256["policy_plan"],
            "checkpoints": CHECKPOINTS, "source_snapshot_sha256": SNAPSHOT_SHA,
            "panel_manifest_sha256": PANEL_SHA, "runtime_spec": context.runtime_spec(),
            "public_source": {"url": SOURCE_URL, "revision": SOURCE_REVISION, "branch": "main"}}


def destinations(prerequisite):
    return (live.expected_destinations() | {str(REMOTE / p) for p in SOURCE_PATHS}
            | {str(REMOTE / p) for p in (*CHECKPOINTS, PANEL)}
            | {str(live.PLAN), str(INPUT / "source.json"), str(INPUT / "authorization.json")}
            | {str(INPUT / "tape" / p) for p in ("result.json", "call.json")}
            | {str(INPUT / "tape/returned" / p) for p in prerequisite["files"]})


def check_layout(plan):
    rows, proof = plan.get("uploads", []), plan.get("tape_prerequisite", {})
    if (set(proof) != {"result_sha256", "files"} or not wire.SHA.fullmatch(str(proof["result_sha256"]))
            or not isinstance(proof["files"], dict) or not 1 <= len(proof["files"]) <= live.FILE_COUNT
            or any(not live.allowed_output(n) for n in proof["files"])):
        raise ValueError("bounded exact successful tape inventory required")
    for value in proof["files"].values():
        if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
                or type(value["bytes"]) is not int or not 0 <= value["bytes"] <= live.FILE_CAP
                or not wire.SHA.fullmatch(str(value["sha256"]))):
            raise ValueError("bounded prior returned-file identity required")
    allowed = destinations(proof)
    if (not isinstance(rows, list) or len(rows) != len(allowed) or {r.get("remote") for r in rows} != allowed
            or plan != definition(rows, plan.get("output"), plan.get("prior_tape_plan_sha256"), proof,
                                  plan.get("authorization_sha256"), plan.get("games"))):
        raise ValueError("full-game input layout or immutable scope differs")
    if selected_games({"games": plan["games"]}) != plan["games"]: raise ValueError("original pair order differs")
    total = len(wire.canonical(plan))
    prior_returns = {str(INPUT / "tape/returned" / n) for n in proof["files"]}
    for row in rows:
        if (set(row) != {"local", "remote", "bytes", "sha256"} or type(row["bytes"]) is not int
                or not (0 if row["remote"] in prior_returns else 1) <= row["bytes"] <= UPLOAD_CAP
                or not wire.SHA.fullmatch(str(row["sha256"]))):
            raise ValueError("bounded regular upload identity required")
        total += row["bytes"]
    if total > UPLOAD_CAP: raise ValueError("full-game upload bound")
    pinned = {str(REMOTE / p): v for p, v in CHECKPOINTS.items()}
    pinned.update({str(INPUT / "source.json"): {"bytes": 198071, "sha256": SNAPSHOT_SHA}})
    for row in rows:
        if row["remote"] in pinned and {k: row[k] for k in ("bytes", "sha256")} != pinned[row["remote"]]:
            raise ValueError("exact model or authentic source snapshot differs")


def authorization(value, plan):
    expected = {"schema": APPROVAL_SCHEMA, "decision": "authorized", "scope": fixed_scope(), "limits": limits(),
                "bindings": {"tape_plan_sha256": plan["prior_tape_plan_sha256"],
                    "tape_result_sha256": plan["tape_prerequisite"]["result_sha256"],
                    "policy_plan_sha256": plan["policy_plan_sha256"], "checkpoints": CHECKPOINTS,
                    "source_snapshot_sha256": SNAPSHOT_SHA, "panel_manifest_sha256": PANEL_SHA,
                    "game_image_sha256": tape.IMAGE["sha256"], "package_archive_sha256": host.ARCHIVE[1]}}
    if value != expected: raise ValueError("explicit bounded full-game authorization required")


def tape_prerequisite(directory):
    prior_path = directory / "plan.json"; prior = live.verify_plan(prior_path, wire.digest(prior_path))
    result, call = (policy.bounded_json(directory / p) for p in ("result.json", "call.json"))
    rows = result.get("files", [])
    if (result.get("status") != "controller-tape-qualified" or len(rows) > live.FILE_COUNT
            or wire.checked_execution(result.get("execution")) != wire.checked_execution(call.get("execution"))):
        raise ValueError("successful controller-tape prerequisite required")
    files, identities = {}, {}
    for row in rows:
        name = row["name"]
        if not live.allowed_output(name) or name in files: raise ValueError("exact tape output inventory required")
        path = directory / "returned" / name
        if wire.regular(path, live.FILE_CAP) != row["bytes"] or wire.digest(path) != row["sha256"]:
            raise ValueError("tape evidence changed")
        files[name] = path; identities[name] = {k: row[k] for k in ("bytes", "sha256")}
    live.validate_terminal(result, files, wire.digest(prior_path), prior)
    return prior, {"result_sha256": wire.digest(directory / "result.json"), "files": identities}


def validate_tape_terminal(result, files, sha, prior, *, remote=False):
    """Run frozen tape acceptance with an isolated prior-runtime path view.

    The original uploaded plans and function code remain unchanged. Only the
    specific read of the prior runtime plan receives a view whose local input
    paths resolve to the already hash-verified Linux mounts.
    """
    if not remote: return live.validate_terminal(result, files, sha, prior)
    rows = [r for r in prior["uploads"] if r["remote"] == str(runtime.PLAN)]
    if len(rows) != 1: raise ValueError("exact prior runtime-plan path required")
    record = rows[0]
    original = runtime.verify_plan(runtime.PLAN, prior["prior_runtime_plan_sha256"], remote=True)
    view = {**original, "uploads": [{**r, "local": r["remote"]} for r in original["uploads"]]}
    def host_json(path):
        return copy.deepcopy(view) if Path(path) == Path(record["local"]) else policy.bounded_json(path)
    proxy = types.SimpleNamespace(**{**vars(policy), "bounded_json": host_json})
    function = types.FunctionType(live.validate_terminal.__code__,
        {**live.validate_terminal.__globals__, "policy": proxy}, live.validate_terminal.__name__,
        live.validate_terminal.__defaults__, live.validate_terminal.__closure__)
    return function(result, files, sha, prior)


def make_plan(output, tape_directory, authorization_path):
    """Explicit preparation reads already approved prerequisite and model files."""
    output = Path(output).absolute(); directory = context.canonical(tape_directory)
    authorization_path = Path(authorization_path).absolute()
    if output.resolve(strict=False) != output or not output.is_relative_to(COMPAT): raise ValueError("isolated canonical research output required")
    prior, proof = tape_prerequisite(directory)
    rows = list(prior["uploads"]); known = {r["remote"] for r in rows}
    for name in (*SOURCE_PATHS, *CHECKPOINTS, PANEL):
        if str(REMOTE / name) not in known:
            rows.append(policy.file_record(ROOT / name, REMOTE / name, UPLOAD_CAP)); known.add(str(REMOTE/name))
    rows.append(policy.file_record(directory / "plan.json", live.PLAN, context.MAX_METADATA))
    for name in ("result.json", "call.json", *("returned/" + n for n in proof["files"])):
        rows.append(policy.file_record(directory / name, INPUT / "tape" / name, FILE_CAP))
    rows.append(policy.file_record(COMPAT / "frisson-full-match-source.json", INPUT / "source.json", match_source.MAX_BYTES))
    rows.append(policy.file_record(Path(authorization_path), INPUT / "authorization.json", context.MAX_METADATA))
    if wire.digest(ROOT / PANEL) != PANEL_SHA: raise ValueError("frozen panel manifest differs")
    plan = definition(rows, output, wire.digest(directory / "plan.json"), proof,
                      wire.digest(authorization_path), selected_games(policy.bounded_json(ROOT / PANEL)))
    check_layout(plan); authorization(policy.bounded_json(Path(authorization_path)), plan)
    return plan


def verify_plan(path, sha, *, remote=False):
    if wire.digest(path) != sha: raise ValueError("frozen full-game plan differs")
    plan = policy.bounded_json(path); check_layout(plan)
    lookup = {}
    for row in plan["uploads"]:
        path = Path(row["remote"] if remote else row["local"])
        if wire.regular(path, row["bytes"]) != row["bytes"] or wire.digest(path) != row["sha256"]: raise ValueError("full-game upload changed")
        lookup[row["remote"]] = path
    prior = live.verify_plan(lookup[str(live.PLAN)], plan["prior_tape_plan_sha256"], remote=remote)
    result = policy.bounded_json(lookup[str(INPUT / "tape/result.json")])
    call = policy.bounded_json(lookup[str(INPUT / "tape/call.json")])
    if (wire.digest(lookup[str(INPUT / "tape/result.json")]) != plan["tape_prerequisite"]["result_sha256"]
            or result.get("status") != "controller-tape-qualified"
            or wire.checked_execution(result.get("execution")) != wire.checked_execution(call.get("execution"))):
        raise ValueError("passed bound controller-tape result required")
    files = {n: lookup[str(INPUT / "tape/returned" / n)] for n in plan["tape_prerequisite"]["files"]}
    actual = {r["name"]: {k: r[k] for k in ("bytes", "sha256")} for r in result["files"]}
    if len(actual) != len(result["files"]) or actual != plan["tape_prerequisite"]["files"]: raise ValueError("tape file mapping differs")
    for name, identity in actual.items():
        if {"bytes": wire.regular(files[name], live.FILE_CAP), "sha256": wire.digest(files[name])} != identity:
            raise ValueError("tape file proof differs")
    validate_tape_terminal(result, files, plan["prior_tape_plan_sha256"], prior, remote=remote)
    if wire.digest(lookup[str(REMOTE / PANEL)]) != PANEL_SHA or selected_games(policy.bounded_json(lookup[str(REMOTE / PANEL)])) != plan["games"]:
        raise ValueError("native pair does not match frozen manifest")
    if wire.digest(lookup[str(INPUT / "authorization.json")]) != plan["authorization_sha256"]: raise ValueError("authorization changed")
    authorization(policy.bounded_json(lookup[str(INPUT / "authorization.json")]), plan)
    match_source.validate(policy.bounded_json(lookup[str(INPUT / "source.json")]))
    return plan


def source_manifest(plan):
    allowed = {str(REMOTE / n) for n in SOURCE_PATHS if n.endswith(".py")}
    return {str(Path(r["remote"]).relative_to(REMOTE)): {k: r[k] for k in ("bytes", "sha256")}
            for r in plan["uploads"] if r["remote"] in allowed}


def assemble_admissions(plan, fresh, directory, *, index, created_at, expires_at, parent_pid):
    if fresh.get("status") != "runtime-qualified-gameplay-unadmitted": raise ValueError("fresh exact runtime qualification required")
    prior = policy.bounded_json(runtime.PLAN)
    view = {**prior, "uploads": [{**r, "local": r["remote"]} for r in prior["uploads"]]}
    runtime.validate_qualified_evidence(fresh["qualification"], view)
    if type(index) is not int or index not in (0, 1) or not 0 < expires_at-created_at <= GAME_SECONDS:
        raise ValueError("one bounded game index required")
    image = {"image_id": plan["runtime_image_id"], "manifest_sha256": plan["outer_plan_sha256"], "platform": "linux_x86_64"}
    bindings = {"policy_plan_sha256": plan["policy_plan_sha256"], "package_archive_sha256": host.ARCHIVE[1],
                "game_image_sha256": tape.IMAGE["sha256"], "runtime_image": image}
    paths = {k: {} for k in context.KINDS}
    paths["policy_parity"] = {n: live.INPUT / "policy" / (n + ".json") for n in context.LIMITED_EVIDENCE_SHA256}
    paths["game_image_access"] = {"authorization": INPUT / "authorization.json"}
    paths["live_execution"] = {"authorization": INPUT / "authorization.json", "owner_entrypoint": ENTRY}
    bodies = {"dependency_lock": (runtime.INPUT / "lock.json").read_bytes(),
              "dependency_pins": (runtime.INPUT / "pins.lock").read_bytes(),
              "source_manifest": wire.canonical(source_manifest(plan))}
    bodies.update({n: wire.canonical(fresh["qualification"]["evidence"][n]) for n in ("installed_runtime", "ubjson_extension", "os_packages")})
    directory.mkdir(exist_ok=False)
    for name, body in bodies.items():
        path = directory / (name + ".json")
        with path.open("xb") as stream:
            if stream.write(body) != len(body): raise OSError("short evidence write")
        paths["linux_runtime"][name] = path
    admissions, admission_paths = {}, {}
    for kind, evidence in paths.items():
        value = {"schema": context.SCHEMA, "kind": kind,
                 "decision": context.LIMITED_DECISION if kind == "policy_parity" else context.KINDS[kind],
                 "bindings": bindings, "evidence": {n: context.read_bound(p)[1] for n, p in evidence.items()}}
        context.validate_admission(value, kind, bindings)
        path = directory / (kind + "-admission.json"); wire.create_once(path, value)
        admission_paths[kind] = str(path); admissions[kind] = context.read_bound(path)[1]
    sources = {str(Path(r["remote"]).relative_to(REMOTE)): {k: r[k] for k in ("bytes", "sha256")}
               for r in plan["uploads"] if r["remote"] in {str(REMOTE / n) for n in SOURCE_PATHS}}
    prepared = {"schema": SCHEMA + ".execution", "agent_kind": "native-frisson-versus-slippi-policy",
                "created_at": created_at, "expires_at": expires_at, "scope": fixed_scope(), "index": index,
                "source_identities": sources, "required_admissions": admissions, "runtime_image": image,
                "game_image": tape.IMAGE, "cases": [{"slippi_port": 51442}],
                "game": plan["games"][index], "artifact_label": LABELS[index], "benchmark_ledger_writes": False}
    wire.create_once(directory / "execution-plan.json", prepared)
    owner = {"schema": "e011.modal-live-parent-declaration.v1", "parent_pid": parent_pid,
             "plan_sha256": tape.identity(prepared)["sha256"], "expires_at": expires_at,
             "entrypoint": {"path": str(ENTRY), "identity": context.read_bound(ENTRY)[1]}, "runner_sha256": context.RUNNER_SHA}
    descriptor = {"admission_paths": admission_paths, "evidence_paths": {k: {n: str(p) for n,p in v.items()} for k,v in paths.items()},
                  "owner_declaration": owner, "policy_plan_sha256": plan["policy_plan_sha256"]}
    wire.create_once(directory / "context.json", descriptor)
    return prepared, descriptor


def checked_copy(source, target, row, expires):
    wire.remaining(expires)
    if wire.regular(source, row["bytes"]) != row["bytes"] or wire.digest(source) != row["sha256"]: raise ValueError("staged input changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.resolve() != target.parent or target.is_symlink(): raise ValueError("unsafe staged target")
    count, digest = 0, hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        while chunk := src.read(1024**2):
            wire.remaining(expires); count += len(chunk)
            if count > row["bytes"] or dst.write(chunk) != len(chunk): raise OSError("bounded staged copy differs")
            digest.update(chunk)
    if count != row["bytes"] or digest.hexdigest() != row["sha256"]: raise ValueError("staged copy hash differs")


def stage_project(plan, expires):
    commands = WORK / "source-command"; commands.mkdir(); (commands / "logs").mkdir()
    runner = guard.Runner(commands, min(240, wire.remaining(expires)), 1)
    try:
        source = match_source.materialize(policy.bounded_json(INPUT / "source.json"), PARENT)
        PROJECT.mkdir()
        names = {str(REMOTE / n): n for n in (*SOURCE_PATHS, *runtime.DATA_ASSETS, *CHECKPOINTS, PANEL)}
        for row in plan["uploads"]:
            if row["remote"] in names: checked_copy(Path(row["remote"]), PROJECT / names[row["remote"]], row, expires)
        public = PROJECT / ".e001-cache/slippi-ai-source"
        public.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prepared_runtime.slippi_source(SOURCE_REVISION), public)
        wire.create_once(WORK / "source.json", {"authentic_frisson": source, "slippi": {"url": SOURCE_URL, "revision": SOURCE_REVISION},
                         "project_root": str(PROJECT), "uploaded_sources": source_manifest(plan)})
    finally: wire.create_once(commands / "commands.json", runner.commands)


def native_validation(game, label, project, manifest_path, validator_path):
    """Unchanged acceptance function in a fresh module, with explicit host roots."""
    spec = importlib.util.spec_from_file_location("isolated_panel_acceptance", validator_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT, module.ARTIFACTS, module.OUTPUT = project, project / "artifacts/integration/frisson_ai", manifest_path.parent
    return module.validate_result(game, {"label": label})


def game_child(index):
    directory = WORK / f"game-{index}"; prepared = policy.bounded_json(directory / "execution-plan.json")
    descriptor = policy.bounded_json(directory / "context.json")
    if prepared.get("schema") != SCHEMA + ".execution" or prepared.get("index") != index or prepared.get("scope") != fixed_scope():
        raise ValueError("exact full-policy child plan required")
    result = {"schema": SCHEMA + ".game", "status": "failed", "index": index,
              "plan_sha256": tape.identity(prepared)["sha256"], "artifact_label": LABELS[index], "game": prepared["game"]}
    adapter = None
    try:
        ctx = context.OwnedLinuxContext(project_root=PROJECT, research_output_root=WORK,
            package_root=runtime.WORK / "package", package_manifest=runtime.INPUT / "manifest.json",
            package_audit=runtime.INPUT / "audit.json", game_image_path=live.ROM,
            admission_paths=descriptor["admission_paths"], evidence_paths=descriptor["evidence_paths"],
            policy_plan_sha256=descriptor["policy_plan_sha256"], owner_declaration=descriptor["owner_declaration"])
        preparation = ctx.verify_and_prepare(prepared, directory); result["owner"] = preparation
        sys.path.insert(0, str(PROJECT / "scripts"))
        from melee_policy.integration import frisson_slippi_match, runtime_identity
        import slippi_panel_runtime
        adapter = match_host.MatchHostAdapter(match=frisson_slippi_match, owned_context=ctx,
            preparation=preparation, execution_plan=prepared, linux_pins_path=runtime.INPUT / "pins.lock", runtime_identity=runtime_identity)
        result["host_adapter"] = adapter.install()
        os.environ["MELEE_ISO_PATH"] = str(live.ROM)
        slippi_panel_runtime.run_game(prepared["game"], LABELS[index])
        result["native_acceptance"] = native_validation(prepared["game"], LABELS[index], PROJECT, PROJECT / PANEL,
                                                       PROJECT / "scripts/run_slippi_public_panel.py")
        result["status"] = "native-full-policy-game-passed"
    except BaseException as error: result["error"] = f"{type(error).__name__}: {error}"[:2048]
    finally:
        if adapter is not None:
            result["launch_evidence"] = adapter.launch_evidence
            try: adapter.restore()
            except BaseException as error: result.update(status="failed", restoration_error=f"{type(error).__name__}: {error}"[:2048])
        wire.create_once(directory / "result.json", result)
    return result


def owner(plan, sha, expires, image_id):
    wire.remaining(expires); WORK.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "status": "failed", "plan_sha256": sha, "scope": fixed_scope(),
              "started_unix": time.time(), "expires_unix": expires, "runtime_image_id": image_id, "games": []}
    with live.terminate_cleanup():
        try:
            fresh = runtime.perform(policy.bounded_json(runtime.PLAN), expires - 2100)
            result["fresh_runtime_status"] = fresh["status"]
            if fresh["status"] != "runtime-qualified-gameplay-unadmitted": raise ValueError("fresh runtime failed")
            stage_project(plan, expires - 1830)
            for index in (0, 1):
                now = int(time.time()); until = min(int(expires - 15), now + GAME_SECONDS)
                if until-now < 60: raise TimeoutError("insufficient full-game budget")
                prepared, _ = assemble_admissions({**plan, "runtime_image_id": image_id, "outer_plan_sha256": sha}, fresh,
                    WORK / f"game-{index}", index=index, created_at=now, expires_at=until, parent_pid=os.getpid())
                commands = WORK / f"game-command-{index}"; commands.mkdir(); (commands / "logs").mkdir()
                runner = guard.Runner(commands, until-time.time(), 1)
                runner.env.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                                   "PYTHONPATH": str(REMOTE / "scripts"), **live.native_user_environment()})
                try: runner.run([str(runtime.WORK / "venv/bin/python"), str(ENTRY), "--game-child", str(index)], WORK, build=True)
                finally: wire.create_once(commands / "commands.json", runner.commands)
                child = policy.bounded_json(WORK / f"game-{index}/result.json")
                cleanup = context.validate_parent_terminal(child["owner"], runner.commands[-1])
                result["games"].append({"index": index, "child": child, "cleanup": cleanup, "commands": runner.commands})
                if child.get("status") != "native-full-policy-game-passed": raise ValueError("native game acceptance failed")
            result["status"] = "full-policy-compatibility-pair-passed"
        except BaseException as error: result["error"] = f"{type(error).__name__}: {error}"[:2048]
        finally:
            result["ended_unix"] = time.time(); wire.create_once(WORK / "receipt.json", result)
    return result


def allowed_output(name):
    if not isinstance(name, str) or len(name) > 280 or ".." in Path(name).parts: return False
    return bool(re.fullmatch(r"(?:receipt\.json|source\.json|supervisor/(?:receipt\.json|owner\.log)|"
        r"runtime/(?:receipt\.json|qualification\.json|logs/[0-9]{3}\.log)|"
        r"(?:source-command|game-command-[01])/(?:commands\.json|logs/[0-9]{3}\.log)|"
        r"game-[01]/(?:[a-z_-]+\.json|xvfb\.log|host-[0-9]{2}\.log)|"
        r"project/artifacts/integration/frisson_ai/modal-compat-75m-d21-fd-p[12]/"
        r"(?:summary\.json|controller_trace\.jsonl|replays/[A-Za-z0-9_. -]+\.slp|game/(?:manifest\.json|game\.slp)))", name))


def collect_outputs(supervisor):
    files, total = [], 0
    candidates = []
    for base, prefix in ((WORK, ""), (WORK.parent / "full-game-supervisor", "supervisor/")):
        if not base.exists(): continue
        # Traverse only the small explicit evidence roots. Model/source trees
        # are never scanned or returned.
        roots = ([base / n for n in ("receipt.json", "source.json", "source-command", "game-command-0", "game-command-1", "game-0", "game-1")]
                 if not prefix else [base])
        for root in roots:
            paths = [root] if root.is_file() else root.rglob("*") if root.exists() else []
            for count, path in enumerate(paths):
                if count > 2000: raise ValueError("bounded game evidence inventory exceeded")
                name = prefix + path.relative_to(base).as_posix()
                if allowed_output(name): candidates.append((name, path))
    for label in LABELS:
        base = PROJECT / "artifacts/integration/frisson_ai" / label
        if base.exists():
            for count, path in enumerate(base.rglob("*")):
                if count > 1000: raise ValueError("bounded game artifacts exceeded")
                name = "project/" + path.relative_to(PROJECT).as_posix()
                if allowed_output(name): candidates.append((name, path))
    if (runtime.WORK / "receipt.json").exists():
        for row in runtime.collect_outputs(policy.bounded_json(runtime.WORK / "receipt.json"))["files"]:
            files.append({**row, "name": "runtime/" + row["name"]}); total += len(row["body"])
    for name, path in sorted(candidates):
        size = wire.regular(path, FILE_CAP); cap = 65536 if name.endswith(".log") else FILE_CAP
        with path.open("rb") as stream: stream.seek(max(0, size-cap)); body = stream.read(cap+1)
        total += len(body)
        if len(body) > cap or total > RETURN_CAP or len(files) >= FILE_COUNT: raise ValueError("bounded pair output exceeded")
        files.append({"name": name, "body": body, "sha256": hashlib.sha256(body).hexdigest(),
                      "original_bytes": size, "original_sha256": wire.digest(path), "truncated_tail": size > cap})
    receipt = policy.bounded_json(WORK / "receipt.json") if (WORK / "receipt.json").exists() else {}
    status = receipt.get("status", "failed") if supervisor.get("exit_code") == 0 and not supervisor.get("error") else "failed"
    return {"schema": SCHEMA, "status": status, "plan_sha256": receipt.get("plan_sha256"), "scope": fixed_scope(), "files": files}


def validate_terminal(terminal, files, sha, plan):
    if (terminal.get("schema") != SCHEMA or terminal.get("plan_sha256") != sha or terminal.get("scope") != fixed_scope()
            or terminal.get("status") not in {"failed", "full-policy-compatibility-pair-passed"}): raise ValueError("bound game terminal required")
    if terminal["status"] == "failed": return
    result = policy.bounded_json(files["receipt.json"]); supervisor = policy.bounded_json(files["supervisor/receipt.json"])
    if (result.get("status") != terminal["status"] or result.get("plan_sha256") != sha or result.get("scope") != fixed_scope()
            or result.get("fresh_runtime_status") != "runtime-qualified-gameplay-unadmitted" or len(result.get("games", [])) != 2
            or supervisor.get("exit_code") != 0 or supervisor.get("error") is not None
            or supervisor.get("unreaped_leader_group_guard") is not True or supervisor.get("terminated_owned_process_group") is not True):
        raise ValueError("complete owned native pair evidence required")
    lookup = {r["remote"]: Path(r["local"]) for r in plan["uploads"]}
    tape_plan = policy.bounded_json(lookup[str(live.PLAN)])
    runtime_plan = policy.bounded_json(lookup[str(runtime.PLAN)])
    fresh = policy.bounded_json(files["runtime/receipt.json"])
    runtime.validate_terminal({**fresh, "execution": terminal["execution"]}, tape_plan["prior_runtime_plan_sha256"], runtime_plan,
        {"receipt.json": files["runtime/receipt.json"].read_bytes(), "qualification.json": files["runtime/qualification.json"].read_bytes()})
    if fresh.get("status") != "runtime-qualified-gameplay-unadmitted": raise ValueError("fresh returned runtime qualification required")
    expected_sources = {str(Path(r["remote"]).relative_to(REMOTE)): {k:r[k] for k in ("bytes", "sha256")}
                        for r in plan["uploads"] if r["remote"] in {str(REMOTE / n) for n in SOURCE_PATHS}}
    policy_evidence = {n: context.read_bound(lookup[str(live.INPUT / "policy" / (n+".json"))])[0]
                       for n in context.LIMITED_EVIDENCE_SHA256}
    qualification = context.bind_policy_evidence(policy_evidence, plan["policy_plan_sha256"], decision=context.LIMITED_DECISION)
    for index, entry in enumerate(result["games"]):
        child = policy.bounded_json(files[f"game-{index}/result.json"])
        prepared = policy.bounded_json(files[f"game-{index}/execution-plan.json"])
        commands = policy.bounded_json(files[f"game-command-{index}/commands.json"])
        if (entry.get("index") != index or entry.get("child") != child or entry.get("commands") != commands
                or len(commands) != 1 or child.get("status") != "native-full-policy-game-passed"
                or child.get("game") != plan["games"][index] or child.get("artifact_label") != LABELS[index]
                or child.get("plan_sha256") != tape.identity(prepared)["sha256"]
                or prepared.get("game") != plan["games"][index] or prepared.get("scope") != fixed_scope()
                or prepared.get("schema") != SCHEMA + ".execution" or prepared.get("index") != index
                or prepared.get("artifact_label") != LABELS[index] or prepared.get("cases") != [{"slippi_port": 51442}]
                or prepared.get("agent_kind") != "native-frisson-versus-slippi-policy"
                or prepared.get("source_identities") != expected_sources
                or prepared["runtime_image"]["manifest_sha256"] != sha
                or prepared["runtime_image"]["image_id"] != result["runtime_image_id"]
                or commands[0].get("command") != [str(runtime.WORK / "venv/bin/python"), str(ENTRY), "--game-child", str(index)]
                or child["owner"]["process"]["parent_pid"] != supervisor["pid"]
                or child["owner"].get("plan_sha256") != tape.identity(prepared)["sha256"]
                or child["owner"].get("admissions") != prepared["required_admissions"]
                or child["owner"].get("runtime_image") != prepared["runtime_image"]
                or child["owner"].get("policy_qualification") != qualification
                or entry.get("cleanup") != context.validate_parent_terminal(child["owner"], commands[0])):
            raise ValueError("exact paired native child and cleanup binding required")
        bindings = {"policy_plan_sha256": plan["policy_plan_sha256"], "package_archive_sha256": host.ARCHIVE[1],
                    "game_image_sha256": tape.IMAGE["sha256"], "runtime_image": prepared["runtime_image"]}
        evidence = {"policy_parity": {n: context.read_bound(lookup[str(live.INPUT / "policy" / (n+".json"))])[1]
                                      for n in context.LIMITED_EVIDENCE_SHA256},
                    "game_image_access": {"authorization": context.read_bound(lookup[str(INPUT / "authorization.json")])[1]},
                    "live_execution": {"authorization": context.read_bound(lookup[str(INPUT / "authorization.json")])[1],
                                       "owner_entrypoint": context.read_bound(lookup[str(ENTRY)])[1]},
                    "linux_runtime": {}}
        expected_runtime = {"dependency_lock": lookup[str(runtime.INPUT / "lock.json")].read_bytes(),
                            "dependency_pins": lookup[str(runtime.INPUT / "pins.lock")].read_bytes(),
                            "source_manifest": wire.canonical(source_manifest(plan))}
        expected_runtime.update({n: wire.canonical(fresh["qualification"]["evidence"][n])
                                 for n in ("installed_runtime", "ubjson_extension", "os_packages")})
        for name, expected_body in expected_runtime.items():
            body, identity = context.read_bound(files[f"game-{index}/{name}.json"])
            if body != expected_body: raise ValueError("returned admitted runtime evidence differs")
            evidence["linux_runtime"][name] = identity
        for kind in context.KINDS:
            body, _ = context.read_bound(files[f"game-{index}/{kind}-admission.json"], prepared["required_admissions"][kind])
            admission = json.loads(body); context.validate_admission(admission, kind, bindings)
            if admission["evidence"] != evidence[kind]: raise ValueError("admission must bind the retained actual evidence")
        if policy.bounded_json(files[f"game-{index}/source_manifest.json"]) != source_manifest(plan):
            raise ValueError("full native source closure differs")
        launch = child.get("launch_evidence", {})
        if (not isinstance(launch, dict) or launch.get("adjacent_rust_loaded") is not True
                or launch.get("pgid") != child["owner"]["process"]["pgid"] or launch.get("sid") != child["owner"]["process"]["sid"]
                or launch.get("executable") != {"bytes": host.EXECUTABLE[0], "sha256": host.EXECUTABLE[1]}
                or len(launch.get("udp", [])) != 1 or launch["udp"][0].get("port") != 51442):
            raise ValueError("actual owned Linux executable and UDP evidence required")
        expected_adapter = {"schema": match_host.SCHEMA,
            "source_identities": {n: expected_sources[n] for n in match_host.SOURCE_HASHES},
            "hooks": list(match_host.HOOKS), "policy_loop_changes": False,
            "parent_terminal_cleanup_verified": False, "policy_qualification": qualification}
        if child.get("host_adapter") != expected_adapter: raise ValueError("exact five-hook host adapter receipt required")
        base = files[f"project/artifacts/integration/frisson_ai/{LABELS[index]}/summary.json"]
        project = base.parents[4]
        accepted = native_validation(plan["games"][index], LABELS[index], project, lookup[str(REMOTE / PANEL)],
                                     lookup[str(REMOTE / "scripts/run_slippi_public_panel.py")])
        old = child.get("native_acceptance", {})
        if {k:v for k,v in accepted.items() if k not in {"summary", "replay"}} != {k:v for k,v in old.items() if k not in {"summary", "replay"}}:
            raise ValueError("unchanged native acceptance does not reproduce on returned bytes")


def remote_game(sha, expires, image_id):
    plan = verify_plan(PLAN, sha, remote=True)
    if not 0 < wire.remaining(expires) <= SESSION_SECONDS+1 or not re.fullmatch("im-[A-Za-z0-9]{1,100}", image_id): raise ValueError("bounded exact game image identity required")
    import modal
    call = modal.current_function_call_id()
    if not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(call)): raise ValueError("actual function call required")
    execution = wire.checked_execution({"task_id": os.environ.get("MODAL_TASK_ID"), "execution_id": uuid.uuid4().hex})
    yield wire.checked_frame({"kind": "hello", "seq": 0, "plan_sha256": sha, "call_id": call, "execution": execution})
    until = min(expires-30, time.time()+OWNER_SECONDS)
    markers = {f"game-{i}-trace": PROJECT / "artifacts/integration/frisson_ai" / LABELS[i] / "controller_trace.jsonl" for i in (0,1)}
    supervisor = live.supervise_owner([sys.executable, str(ENTRY), "--owner", "--sha", sha, "--expires", str(until), "--image-id", image_id],
                                     until, WORK.parent / "full-game-supervisor", markers=markers)
    result = collect_outputs(supervisor); result["plan_sha256"] = sha
    yield from wire.output_frames({**result, "execution": execution})


def configure_app(modal, path, plan):
    app = modal.App(APP_NAME, include_source=False)
    image = modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), "PYTHONPATH": str(REMOTE / "scripts")})
    for row in plan["uploads"]: image = image.add_local_file(row["local"], row["remote"], copy=False)
    image = image.add_local_file(str(path), str(PLAN), copy=False)
    fn = app.function(image=image, include_source=False, serialized=False, is_generator=True,
        cpu=(4,4), memory=(16384,16384), max_containers=1, min_containers=0, buffer_containers=0,
        timeout=FUNCTION_SECONDS, startup_timeout=300, scaledown_window=2, nonpreemptible=True,
        single_use_containers=True, restrict_modal_access=True)(remote_game)
    return app, fn, image


def execute(path, sha):
    plan = verify_plan(path, sha); output = Path(plan["output"])
    if path != output / "plan.json" or output.resolve() != output: raise ValueError("canonical prepared game plan required")
    with (output / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "intent.json").exists(): return {"status": "intent-retained-no-resubmission"}
        expires = time.time()+SESSION_SECONDS
        wire.create_once(output / "intent.json", {"plan_sha256": sha, "expires_unix": expires, "logical_submissions": 1})
        try:
            if os.environ.get("MODAL_PROFILE", "frisson") != "frisson" or any(k.startswith("MODAL_") and k != "MODAL_PROFILE" for k in os.environ): raise ValueError("unexpected Modal environment")
            os.environ["MODAL_PROFILE"] = "frisson"
            if importlib.metadata.version("modal") != wire.SDK_VERSION: raise ValueError("pinned Modal SDK required")
            import modal
            app, fn, image = configure_app(modal, path, plan)
            with wire.local_deadline(wire.remaining(expires)), modal.enable_output(), app.run(detach=False, environment_name="main"):
                wire.create_once(output / "app.json", {"app_id": app.app_id, "image_id": image.object_id})
                terminal, files, records = live.receive_files(output, fn.remote_gen(sha, expires, image.object_id), sha, app.app_id, expires,
                    name_check=allowed_output, total_cap=RETURN_CAP, file_cap=FILE_CAP, file_count=FILE_COUNT)
                validate_terminal(terminal, files, sha, plan)
                wire.create_once(output / "result.json", {**terminal, "files": records, "benchmark_ledger_writes": False})
                return terminal
        except BaseException as error:
            wire.create_once(output / "failure.json", {"error": f"{type(error).__name__}: {error}"[:2048], "intent_retained": True})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path); mode.add_argument("--run", type=Path)
    mode.add_argument("--owner", action="store_true"); mode.add_argument("--game-child", type=int, choices=(0,1))
    parser.add_argument("--sha"); parser.add_argument("--expires", type=float); parser.add_argument("--image-id")
    parser.add_argument("--input-paths", type=Path)
    args = parser.parse_args()
    if args.prepare:
        spec = policy.bounded_json(args.input_paths)
        if set(spec) != {"tape_directory", "authorization_path"}: raise ValueError("exact preparation paths required")
        plan = make_plan(args.prepare, **spec); args.prepare.mkdir(exist_ok=False)
        wire.create_once(args.prepare / "plan.json", plan)
        print(json.dumps({"plan": str(args.prepare / "plan.json"), "sha256": wire.digest(args.prepare / "plan.json")}))
    elif args.owner:
        if not 0 < wire.remaining(args.expires) <= OWNER_SECONDS+1: raise ValueError("bounded owner expiry required")
        owner(verify_plan(PLAN, args.sha, remote=True), args.sha, args.expires, args.image_id)
    elif args.game_child is not None: game_child(args.game_child)
    else: print(json.dumps(execute(args.run, args.sha)))
    return 0


if __name__ == "__main__": raise SystemExit(main())
