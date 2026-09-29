#!/usr/bin/env python3
"""Explicit bounded execution of one claimed original benchmark port pair."""
from __future__ import annotations

import prepared_runtime

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import modal_panel_game_pilot as base
import modal_panel_pair_binding as binding

ENTRY_RELATIVE = "scripts/modal_panel_batch_pilot.py"
APP_NAME = "frisson-linux-panel-pair-v1"


def local_binding(labels, attempt_labels):
    panel = base.policy.bounded_json(base.ROOT / base.PANEL)
    return binding.bind(panel, labels, attempt_labels, project_root=base.ROOT, entry_relative=ENTRY_RELATIVE)


def plan_binding(path, sha, *, remote=False):
    if base.wire.digest(path) != sha: raise ValueError("exact pair plan SHA required")
    plan = base.policy.bounded_json(path)
    labels = [g["label"] for g in plan["games"]]
    root = base.REMOTE if remote else base.ROOT
    panel = base.policy.bounded_json(root / base.PANEL)
    bound = binding.bind(panel, labels, plan["artifact_labels"], project_root=root, entry_relative=ENTRY_RELATIVE)
    bound.verify_plan(path, sha, remote=remote)
    return bound, plan


def configure_app(modal, path, plan, bound):
    app = modal.App(APP_NAME, include_source=False)
    image = modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), "PYTHONPATH":str(base.REMOTE / "scripts")})
    for row in plan["uploads"]: image = image.add_local_file(row["local"], row["remote"], copy=False)
    image = image.add_local_file(str(path), str(base.PLAN), copy=False)
    fn = app.function(image=image, include_source=False, serialized=False, is_generator=True,
        cpu=(4,4), memory=(16384,16384), max_containers=1, min_containers=0, buffer_containers=0,
        timeout=bound.FUNCTION_SECONDS, startup_timeout=300, scaledown_window=2,
        nonpreemptible=True, single_use_containers=True, restrict_modal_access=True)(remote_pair)
    return app, fn, image


def remote_pair(sha, expires, image_id):
    bound, _ = plan_binding(base.PLAN, sha, remote=True)
    yield from bound.remote_game(sha, expires, image_id)


def execute(path, sha):
    bound, plan = plan_binding(path, sha)
    output = Path(plan["output"])
    if path != output / "plan.json" or output.resolve() != output: raise ValueError("canonical plan output required")
    with (output / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "intent.json").exists(): return {"status":"intent-retained-no-resubmission"}
        expires = time.time()+bound.SESSION_SECONDS
        base.wire.create_once(output / "intent.json", {"plan_sha256":sha,"expires_unix":expires,"logical_submissions":1})
        try:
            if os.environ.get("MODAL_PROFILE","frisson") != "frisson" or any(k.startswith("MODAL_") and k!="MODAL_PROFILE" for k in os.environ):
                raise ValueError("unexpected Modal environment")
            os.environ["MODAL_PROFILE"]="frisson"
            if importlib.metadata.version("modal") != base.wire.SDK_VERSION: raise ValueError("pinned Modal SDK required")
            import modal
            app, fn, image = configure_app(modal,path,plan,bound)
            with base.wire.local_deadline(base.wire.remaining(expires)), modal.enable_output(), app.run(detach=False,environment_name="main"):
                base.wire.create_once(output / "app.json",{"app_id":app.app_id,"image_id":image.object_id})
                terminal,files,records=base.live.receive_files(output,fn.remote_gen(sha,expires,image.object_id),sha,app.app_id,expires,
                    name_check=bound.allowed_output,total_cap=bound.RETURN_CAP,file_cap=bound.FILE_CAP,file_count=bound.FILE_COUNT)
                bound.validate_terminal(terminal,files,sha,plan)
                base.wire.create_once(output / "result.json",{**terminal,"files":records,"benchmark_ledger_writes":False})
                return terminal
        except BaseException as error:
            base.wire.create_once(output / "failure.json",{"error":f"{type(error).__name__}: {error}"[:2048],"intent_retained":True})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",type=Path); group.add_argument("--run",type=Path)
    group.add_argument("--owner",action="store_true"); group.add_argument("--game-child",type=int,choices=(0,1))
    parser.add_argument("--input-paths",type=Path); parser.add_argument("--sha")
    parser.add_argument("--expires",type=float); parser.add_argument("--image-id")
    args=parser.parse_args()
    if args.prepare:
        request=base.policy.bounded_json(args.input_paths)
        if set(request)!={"labels","attempt_labels","tape_directory","authorization_path"}: raise ValueError("exact pair request keys required")
        bound=local_binding(request["labels"],request["attempt_labels"])
        plan=bound.make_plan(args.prepare,request["tape_directory"],request["authorization_path"])
        args.prepare.mkdir(exist_ok=False)
        base.wire.create_once(args.prepare / "plan.json",plan)
        base.wire.create_once(args.prepare / "binding.json",bound.binding)
        print(json.dumps({"plan":str(args.prepare / "plan.json"),"sha256":base.wire.digest(args.prepare / "plan.json")}))
    elif args.owner:
        bound,plan=plan_binding(base.PLAN,args.sha,remote=True)
        if not 0<base.wire.remaining(args.expires)<=bound.OWNER_SECONDS+1: raise ValueError("bounded owner required")
        bound.owner(plan,args.sha,args.expires,args.image_id)
    elif args.game_child is not None:
        bound,_=plan_binding(base.PLAN,base.wire.digest(base.PLAN),remote=True)
        bound.game_child(args.game_child)
    else: print(json.dumps(execute(args.run,args.sha)))
    return 0


if __name__=="__main__": raise SystemExit(main())
