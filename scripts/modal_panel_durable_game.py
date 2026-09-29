"""One native game per call, with committed cloud artifacts and bounded restarts.

Existing pair workers and native policy/transport code are untouched. A fixed
logical plan may use at most three physical executions, including prior native
attempt numbers. Only the selected original game is ever executed by this call.
"""
from __future__ import annotations

import prepared_runtime
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import textwrap
import time
import types
import uuid
import modal_panel_batch_pilot as batch

ENTRY = "scripts/modal_panel_durable_game.py"
SUCCESS = "full-policy-durable-game-passed"
VOLUME = "frisson-benchmark-game-results-v1"
INDEX = "frisson-benchmark-game-index-v1"
MOUNT = Path("/persist-benchmark")
MAX_PHYSICAL = 3


def selected_function(function, namespace, index, *, owner=False):
    if type(index) is not int or index not in (0, 1):
        raise ValueError("exact original game index required")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    class Selection(ast.NodeTransformer):
        loops = lengths = statuses = 0
        def visit_For(self, node):
            self.generic_visit(node)
            if ast.unparse(node.iter) == ("(0, 1)" if owner else "enumerate(result['games'])"):
                expression = f"({index},)" if owner else f"(({index}, result['games'][0]),)"
                node.iter = ast.parse(expression, mode="eval").body
                self.loops += 1
            return node
        def visit_Compare(self, node):
            self.generic_visit(node)
            if not owner and ast.unparse(node) == "len(result.get('games', [])) != 2":
                node.comparators[0] = ast.Constant(1); self.lengths += 1
            return node
        def visit_Constant(self, node):
            if node.value == "full-policy-compatibility-pair-passed":
                self.statuses += 1
                return ast.copy_location(ast.Constant(SUCCESS), node)
            return node
    change = Selection(); tree = change.visit(tree)
    if (change.loops, change.lengths, change.statuses) != (1, 0 if owner else 1, 1):
        raise ValueError("proven per-game orchestration shape changed")
    exec(compile(ast.fix_missing_locations(tree), "<durable-single-game>", "exec"), namespace)


def bind(labels, attempt_labels, index, *, remote=False):
    root = batch.base.REMOTE if remote else batch.base.ROOT
    panel = batch.base.policy.bounded_json(root / batch.base.PANEL)
    old = batch.binding.bind(panel, labels, attempt_labels, project_root=root,
                             entry_relative="scripts/modal_panel_batch_pilot.py")
    ns = old.owner.__globals__; scope = old.fixed_scope()
    scope.update(benchmark_games=1, selected_game_indices=[index], durable_per_game=True,
                 durable_volume=VOLUME, durable_index=INDEX, max_physical_executions=MAX_PHYSICAL)
    ns.update(ENTRY=batch.base.REMOTE / ENTRY, SOURCE_PATHS=(*old.SOURCE_PATHS, ENTRY),
              fixed_scope=lambda: dict(scope))
    selected_function(old.owner, ns, index, owner=True)
    selected_function(old.validate_terminal, ns, index)
    return types.SimpleNamespace(**ns)


def plan_binding(path, sha, *, remote=False):
    if batch.base.wire.digest(path) != sha:
        raise ValueError("exact frozen durable plan required")
    plan = batch.base.policy.bounded_json(path)
    indices = plan.get("scope", {}).get("selected_game_indices")
    if indices not in ([0], [1]) or type(indices[0]) is not int:
        raise ValueError("one original game index required")
    bound = bind([g["label"] for g in plan["games"]], plan["artifact_labels"], indices[0], remote=remote)
    bound.verify_plan(path, sha, remote=remote)
    return bound, plan


def create_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = batch.base.wire.canonical(value)
    with path.open("xb") as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())


def check_manifest(manifest, sha, root, bound):
    if (manifest.get("plan_sha256") != sha or manifest.get("volume") != VOLUME
            or manifest.get("schema") != "e011.durable-game-result.v1"
            or not isinstance(manifest.get("files"), list) or len(manifest["files"]) > bound.FILE_COUNT):
        raise ValueError("durable result binding differs")
    batch.base.wire.checked_execution(manifest["execution"])
    seen = set(); total = 0
    for row in manifest["files"]:
        name = row["name"]
        if name in seen or not bound.allowed_output(name):
            raise ValueError("unsafe or duplicate durable result path")
        seen.add(name); path = root / "files" / name
        # Modal's mount root can itself be a platform-created symlink. Require
        # exact traversal beneath that resolved root, including every child.
        if path.is_symlink() or path.resolve() != root.resolve() / "files" / name:
            raise ValueError("canonical durable file required")
        if batch.base.wire.regular(path.resolve(), bound.FILE_CAP) != row["bytes"] or batch.base.wire.digest(path.resolve()) != row["sha256"]:
            raise ValueError("durable result bytes differ")
        total += row["bytes"]
    if total > bound.RETURN_CAP:
        raise ValueError("durable result total exceeds bound")
    return manifest


def persist_result(volume, root, result, execution, sha, call_id, physical_number, bound):
    rows = []
    for row in result["files"]:
        name = row["name"]; body = row["body"]
        if not bound.allowed_output(name) or len(body) > bound.FILE_CAP or hashlib.sha256(body).hexdigest() != row["sha256"]:
            raise ValueError("invalid artifact before cloud commit")
        path = root / "files" / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        rows.append({k: v for k, v in row.items() if k != "body"} | {"bytes": len(body)})
    manifest = {k: v for k, v in result.items() if k != "files"}
    manifest.update(schema="e011.durable-game-result.v1", native_schema=result["schema"],
                    files=rows, execution=execution, plan_sha256=sha, volume=VOLUME,
                    call_id=call_id, physical_number=physical_number,
                    storage_path=root.relative_to(MOUNT).as_posix())
    check_manifest(manifest, sha, root, bound)
    create_json(root / "manifest.json", manifest)
    # The pointer is published only AFTER the explicit cloud commit succeeds.
    volume.commit()
    return manifest


def retryable_startup(result, index, plan):
    """Retained native evidence must independently prove zero gameplay."""
    files = {r["name"]: r["body"] for r in result["files"]}
    prefix = "project/artifacts/integration/frisson_ai/" + plan["artifact_labels"][index] + "/"
    try:
        s = json.loads(files[prefix + "summary.json"]); e = s["execution"]
        return (s.get("error") == "RuntimeError: Slippi port belongs to another process"
            and s.get("result") == "failed" and e.get("termination") == "not-started"
            and e.get("processed_policy_frames") == 0
            and e.get("inference_counts") == {"frisson-ai": 0, "slippi-ai": 0}
            and e.get("dispatch_counts") == {"frisson-ai": 0, "slippi-ai": 0}
            and all(e.get(k) is None for k in ("first_game_frame", "last_game_frame", "first_game_context", "winner"))
            and e.get("natural_game_end") is False and e.get("game_end_observed") is False
            and e.get("controller_transport", {}).get("installed") is False
            and s.get("gate", {}).get("checks", {}).get("entered_gameplay") is False
            and files[prefix + "controller_trace.jsonl"] == b""
            and not any(n.endswith(".slp") for n in files))
    except (KeyError, ValueError, TypeError):
        return False


def run_durable(bound, plan, sha, expires, image_id, volume, index_store, call_id,
                execution, *, run_owner):
    """Testable orchestration, independently persisted from emulator state."""
    batch.base.wire.checked_execution(execution)
    if not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(call_id)):
        raise ValueError("actual bound call identity required")
    call_binding = {"call_id": call_id, "expires_unix": expires, "image_id": image_id}
    index_store.put(f"{sha}/call", call_binding, skip_if_exists=True)
    if index_store.get(f"{sha}/call") != call_binding:
        raise ValueError("a different call or extended deadline cannot reuse this game plan")
    index = plan["scope"]["selected_game_indices"][0]
    prior = int(plan["artifact_labels"][index].rsplit("-a", 1)[1]) - 1
    budget = MAX_PHYSICAL - prior
    volume.reload()
    # Recover a committed result even if the worker died before publishing its
    # index pointer or returning it to the caller. Incomplete files stay archived.
    for slot in range(budget):
        old = index_store.get(f"{sha}/physical/{slot}")
        if old is None: continue
        root = MOUNT / sha / old["execution"]["execution_id"]
        path = root / "manifest.json"
        if path.exists():
            saved = check_manifest(json.loads(path.read_bytes()), sha, root, bound)
            if (saved["execution"] != old["execution"] or saved["call_id"] != old["call_id"]
                    or saved["physical_number"] != slot + prior + 1):
                raise ValueError("committed physical execution binding differs")
            if saved["status"] == SUCCESS or not saved.get("retryable_startup", False):
                return saved
    if not 0 < batch.base.wire.remaining(expires) <= bound.SESSION_SECONDS + 1:
        raise ValueError("original absolute deadline required before restarting gameplay")
    slot = next((n for n in range(budget) if index_store.put(f"{sha}/physical/{n}",
        {"execution": execution, "call_id": call_id, "started_unix": time.time(), "expires_unix": expires},
        skip_if_exists=True)), None)
    if slot is None:
        raise RuntimeError("three physical executions exhausted; preserve attempts for review")
    root = MOUNT / sha / execution["execution_id"]
    result = run_owner()
    result["retryable_startup"] = retryable_startup(result, index, plan)
    manifest = persist_result(volume, root, result, execution, sha, call_id, slot + prior + 1, bound)
    if result["retryable_startup"] and slot + 1 < budget:
        raise RuntimeError("durably retained zero-gameplay startup failure; bounded fresh retry")
    return manifest


def remote_durable(sha, expires, image_id):
    import modal
    bound, plan = plan_binding(batch.base.PLAN, sha, remote=True)
    volume = modal.Volume.from_name(VOLUME)
    store = modal.Dict.from_name(INDEX)
    call_id = modal.current_function_call_id()
    if not re.fullmatch(r"fc-[A-Za-z0-9_-]{1,128}", str(call_id)):
        raise ValueError("actual Modal call identity required")
    execution = {"task_id": os.environ.get("MODAL_TASK_ID"), "execution_id": uuid.uuid4().hex}
    def run_owner():
        until = min(expires - 30, time.time() + bound.OWNER_SECONDS)
        supervisor = bound.live.supervise_owner([bound.sys.executable, str(bound.ENTRY), "--owner", "--sha", sha,
            "--expires", str(until), "--image-id", image_id], until, bound.WORK.parent / "full-game-supervisor")
        result = bound.collect_outputs(supervisor); result["plan_sha256"] = sha
        return result
    return run_durable(bound, plan, sha, expires, image_id, volume, store, call_id, execution, run_owner=run_owner)


def configure_app(modal, path, plan, bound):
    app = modal.App("frisson-durable-single-game-v1", include_source=False)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True, environment_name="main")
    store = modal.Dict.from_name(INDEX, create_if_missing=True, environment_name="main")
    image = modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), "PYTHONPATH": str(batch.base.REMOTE / "scripts")})
    for row in plan["uploads"]: image = image.add_local_file(row["local"], row["remote"], copy=False)
    image = image.add_local_file(str(path), str(batch.base.PLAN), copy=False)
    fn = app.function(image=image, include_source=False, serialized=False,
        cpu=(4, 4), memory=(16384, 16384), max_containers=1, min_containers=0, buffer_containers=0,
        timeout=bound.FUNCTION_SECONDS, startup_timeout=300, scaledown_window=2,
        nonpreemptible=True, single_use_containers=True,
        volumes={str(MOUNT): volume}, retries=modal.Retries(max_retries=2, initial_delay=1.0))(remote_durable)
    return app, fn, image, volume, store


def download_result(volume, manifest, output, sha, bound):
    expected = f"{sha}/{manifest['execution']['execution_id']}"
    if manifest.get("storage_path") != expected:
        raise ValueError("exact durable cloud path required")
    root = output / "durable"; root.mkdir(exist_ok=True)
    for row in manifest["files"]:
        if not bound.allowed_output(row["name"]): raise ValueError("unsafe returned path")
        path = root / "files" / row["name"]; path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or path.resolve() != root.resolve() / "files" / row["name"]:
            raise ValueError("canonical local artifact path required")
        if path.exists():
            if batch.base.wire.regular(path, bound.FILE_CAP) != row["bytes"] or batch.base.wire.digest(path) != row["sha256"]:
                raise ValueError("retained local artifact differs")
            continue
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            temporary.rename(path.with_name(path.name + ".interrupted-" + uuid.uuid4().hex))
        size = 0; digest = hashlib.sha256()
        with temporary.open("xb") as stream:
            for chunk in volume.read_file(expected + "/files/" + row["name"]):
                size += len(chunk)
                if size > row["bytes"] or size > bound.FILE_CAP: raise ValueError("cloud file grew beyond bound")
                stream.write(chunk); digest.update(chunk)
            stream.flush(); os.fsync(stream.fileno())
        if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("cloud artifact identity mismatch")
        temporary.rename(path)
    check_manifest(manifest, sha, root, bound)
    native = dict(manifest); native["schema"] = manifest["native_schema"]
    files = {r["name"]: root / "files" / r["name"] for r in manifest["files"]}
    return native, files


def recover_saved(volume, store, sha, bound):
    """Read a committed result after a lost return without submitting a worker."""
    binding = store.get(f"{sha}/call")
    if binding is None: return None
    for slot in range(MAX_PHYSICAL):
        start = store.get(f"{sha}/physical/{slot}")
        if start is None: continue
        batch.base.wire.checked_execution(start["execution"])
        if start["call_id"] != binding["call_id"]: raise ValueError("saved call identity differs")
        name = f"{sha}/{start['execution']['execution_id']}/manifest.json"
        try:
            body = bytearray()
            for chunk in volume.read_file(name):
                body.extend(chunk)
                if len(body) > 1024**2: raise ValueError("bounded saved manifest required")
        except FileNotFoundError:
            continue
        saved = json.loads(body)
        if saved.get("plan_sha256") != sha or saved.get("execution") != start["execution"] or saved.get("call_id") != start["call_id"]:
            raise ValueError("saved result identity differs")
        if saved["status"] == SUCCESS or not saved.get("retryable_startup", False): return saved
    return None


def execute(path, sha, *, resume=False):
    import fcntl
    import importlib.metadata
    import modal
    bound, plan = plan_binding(path, sha)
    output = Path(plan["output"])
    if path.absolute() != output / "plan.json": raise ValueError("canonical output required")
    if importlib.metadata.version("modal") != batch.base.wire.SDK_VERSION: raise ValueError("pinned Modal SDK required")
    if os.environ.get("MODAL_PROFILE", "frisson") != "frisson" or any(
            k.startswith("MODAL_") and k != "MODAL_PROFILE" for k in os.environ):
        raise ValueError("unexpected Modal environment")
    os.environ["MODAL_PROFILE"] = "frisson"
    with (output / "durable-coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        app, fn, image, volume, store = configure_app(modal, path, plan, bound)
        if resume:
            if not (output / "submission.json").exists():
                binding = store.get(f"{sha}/call")
                if binding is None: raise ValueError("submission identity unresolved; no resubmission permitted")
                app_record = json.loads((output / "app.json").read_bytes())
                create_json(output / "submission.json", {"plan_sha256": sha, "app_id": app_record["app_id"], "call_id": binding["call_id"]})
            submitted = json.loads((output / "submission.json").read_bytes())
            if submitted["plan_sha256"] != sha: raise ValueError("retained call plan differs")
            manifest = recover_saved(volume, store, sha, bound)
            if manifest is None:
                call = modal.FunctionCall.from_id(submitted["call_id"])
                manifest = call.get()
        else:
            if (output / "intent.json").exists(): raise ValueError("existing intent requires resume or reconciliation")
            expires = time.time() + bound.SESSION_SECONDS
            create_json(output / "intent.json", {"plan_sha256": sha, "expires_unix": expires, "max_physical": 3})
            with modal.enable_output(), app.run(detach=True, environment_name="main"):
                # Hydrate the names before the remote function looks them up.
                store.hydrate()
                create_json(output / "app.json", {"app_id": app.app_id, "image_id": image.object_id})
                call = fn.spawn(sha, expires, image.object_id)
                create_json(output / "submission.json", {"plan_sha256": sha, "app_id": app.app_id, "call_id": call.object_id})
                try: manifest = call.get()
                except Exception:
                    manifest = recover_saved(volume, store, sha, bound)
                    if manifest is None: raise
        native, files = download_result(volume, manifest, output, sha, bound)
        bound.validate_terminal(native, files, sha, plan)
        # Record every physical start, including interrupted or failed executions.
        history = [store.get(f"{sha}/physical/{n}") for n in range(MAX_PHYSICAL)]
        record = {"manifest": manifest, "physical_history": [r for r in history if r is not None]}
        destination = output / "durable-result.json"
        if destination.exists():
            if json.loads(destination.read_bytes()) != record: raise ValueError("durable result publication changed")
        else: create_json(destination, record)
        return {"status": manifest["status"], "physical_executions": len(record["physical_history"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path); g.add_argument("--owner", action="store_true")
    g.add_argument("--game-child", type=int, choices=(0, 1))
    p.add_argument("--sha"); p.add_argument("--expires", type=float); p.add_argument("--image-id")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.owner or a.game_child is not None:
        sha = a.sha or batch.base.wire.digest(batch.base.PLAN)
        bound, plan = plan_binding(batch.base.PLAN, sha, remote=True)
        if a.owner:
            if not 0 < batch.base.wire.remaining(a.expires) <= bound.OWNER_SECONDS + 1: raise ValueError("bounded owner required")
            bound.owner(plan, sha, a.expires, a.image_id)
        else:
            if a.game_child != plan["scope"]["selected_game_indices"][0]: raise ValueError("unselected game refused")
            bound.game_child(a.game_child)
    else:
        print(json.dumps(execute(a.run.absolute(), a.sha, resume=a.resume)))


if __name__ == "__main__": main()
