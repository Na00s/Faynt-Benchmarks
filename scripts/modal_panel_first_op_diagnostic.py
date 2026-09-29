#!/usr/bin/env python3
"""One native Fox d24 P2 observation, with a separately gated TF debug observer.

Preparation and synthetic tests execute no policies. --run requires a frozen plan
hash, starts one owned subprocess, preserves partials, and never retries. The
unchanged policy gate remains separate. Debug instrumentation may alter fusion:
same-host first-row bitwise equality is mandatory for an unperturbed diagnosis.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import struct
import subprocess
import sys
import time

SCHEMA = "e011.slippi-first-op-diagnostic.v1"
HELPER = "scripts/modal_panel_first_op_diagnostic.py"
COMPAT = "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/research/modal-compat-v1"
REFERENCES = {
    "arm64": (COMPAT + "/native-canary-slippi-fox_d24_ditto_v4-p2-v3.json",
              "3d3855a62a69c0b59a8e23c4b003da541e33648ffdf256300f71b06c07ef44f3"),
    "x86_64": (COMPAT + "/policy-pilot-v4/returned/case-07.json",
               "c2000afb93055cf12570cc14d3d9a3fef1b57919d086538ae6936c9a5f67fc97"),
}
OP_REGEX = r"^(MatMul|BatchMatMulV2|BiasAdd|AddV2|Add|Sub|Mul|RealDiv|Mean|Square|Sqrt|Rsqrt|Sigmoid|Tanh|Erf|Relu|Gelu|SelectV2|Select|ConcatV2|Split|Unpack|Pack|GatherV2|OneHot|Cast)$"
LIMITS = {"wall_seconds": 180, "dump_bytes_sampled": 128 * 1024**2,
          "file_bytes_hard": 64 * 1024**2, "files": 64,
          "buffer_events": 4096, "decoded_events": 8192,
          "decoded_float_elements": 2_000_000, "poll_seconds": 0.1}
TF_VERSION = "2.21.0-dev20260203"
TF_GIT = "v1.12.1-136077-g7330196a7a7"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("identity requires a regular nonsymlink file")
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def create_once(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > LIMITS["file_bytes_hard"]:
        raise ValueError("JSON output exceeds the individual file cap")
    with Path(path).open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short output write")
        stream.flush()
        os.fsync(stream.fileno())


def inspect_tree(path):
    rows = []
    for directory, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode) or not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
                raise ValueError("unexpected dump object")
            if stat.S_ISREG(st.st_mode):
                rows.append((p, st.st_size))
    if len(rows) > LIMITS["files"] or sum(n for _, n in rows) > LIMITS["dump_bytes_sampled"]:
        raise ValueError("sampled aggregate dump cap exceeded")
    if any(n > LIMITS["file_bytes_hard"] for _, n in rows):
        raise ValueError("individual dump cap exceeded")
    return rows


def bitwise_equal(a, b):
    """Compare numeric leaf bytes, including signed zero, and exact structure."""
    import numpy as np
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        if set(a) == {"dtype", "shape", "values"}:
            if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                return False
            x = np.asarray(a["values"], dtype=a["dtype"])
            y = np.asarray(b["values"], dtype=b["dtype"])
            return (list(x.shape) == a["shape"] and list(y.shape) == b["shape"]
                    and x.dtype.kind in "bifu" and np.isfinite(x).all()
                    and np.isfinite(y).all() and x.tobytes() == y.tobytes())
        return all(bitwise_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(bitwise_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return struct.pack('!d', a) == struct.pack('!d', b)
    return a == b


def make_plan(root, machine):
    import modal_panel_policy_canary as native
    root = Path(root).resolve()
    reference, expected = REFERENCES[machine]
    path = root / reference
    ident = identity(path)
    if ident["sha256"] != expected or ident["bytes"] > native.RECEIPT_BYTES:
        raise ValueError("original same-host reference changed")
    ref = json.loads(path.read_text())
    if (ref["passed"] is not True or ref["environment"]["machine"] != machine
            or len(ref["rows"]) != 384 or not all(ref["checks"].values())):
        raise ValueError("requires the complete original native receipt")
    native.validate_plan(ref["plan"], root)
    p = ref["plan"]
    if (p["kind"], p["key"], p["port"], p["seed"]) != ("slippi", "fox_d24_ditto_v4", 2, 20260910):
        raise ValueError("diagnostic scope changed")
    return {"schema": SCHEMA, "machine": machine, "reference": {"path": reference, **ident},
            "helper": identity(root / HELPER), "native_plan": p, "limits": LIMITS,
            "op_regex": OP_REGEX, "tensor_debug_mode": "FULL_TENSOR",
            "tensor_dtypes": ["float32"], "warmup_calls": 1, "real_observations": 1,
            "baseline": {"row": ref["rows"][0], "environment": ref["environment"],
                         "runtime_contract": ref["runtime_contract"]},
            "notes": ["DebugIdentityV2 may inhibit fusion. Bitwise same-host outer output is required.",
                      "The aggregate disk bound is sampled every0.1s and may overshoot before termination.",
                      "RLIMIT_FSIZE is the hard per-file bound. Circular buffers bound event count, not RAM.",
                      "No new tolerance, checkpoint, seed, delay, context, optimizer or threading setting."]}


def verify_plan(plan, root):
    if plan != make_plan(root, plan["machine"]):
        raise ValueError("frozen diagnostic inputs or limits changed")
    if platform.machine() != plan["machine"]:
        raise ValueError("same-host architecture required")


@contextlib.contextmanager
def warmup_observer(tf, basic_class, dump_root, state):
    original = basic_class.warmup

    def wrapped(self):
        state["warmup_calls"] += 1
        if state["warmup_calls"] != 1:
            raise RuntimeError("more than one native warmup")
        state["writer"] = tf.debugging.experimental.enable_dump_debug_info(
            str(dump_root), tensor_debug_mode="FULL_TENSOR",
            circular_buffer_size=LIMITS["buffer_events"],
            op_regex=OP_REGEX, tensor_dtypes=["float32"])
        result = original(self)
        state["warmup_finished_unix"] = time.time()
        return result

    basic_class.warmup = wrapped
    try:
        yield
    finally:
        basic_class.warmup = original


def read_dump(dump_root, observation_started, reader_factory=None):
    """Use the exact installed TensorFlow DebugDataReader; no graph rerun."""
    import numpy as np
    if reader_factory is None:
        from tensorflow.python.debug.lib.debug_events_reader import DebugDataReader
        reader_factory = DebugDataReader
    inspect_tree(dump_root)
    values, count = [], 0
    with reader_factory(str(dump_root)) as reader:
        reader.update()
        traces = reader.graph_execution_traces(digest=True)
        if not traces or len(traces) > LIMITS["decoded_events"]:
            raise ValueError("missing or excessive graph execution traces")
        for index, trace in enumerate(traces):
            if re.fullmatch(OP_REGEX, trace.op_type) is None:
                raise ValueError("unexpected traced op type")
            x = reader.graph_execution_trace_to_tensor_value(trace)
            if x is None or x.dtype != np.dtype("float32") or not np.isfinite(x).all():
                raise ValueError("invalid FULL_TENSOR float payload")
            count += x.size
            if count > LIMITS["decoded_float_elements"]:
                raise ValueError("decoded tensor count cap")
            graph = reader.graph_by_id(trace.graph_id)
            op = graph.get_op_creation_digest(trace.op_name)
            values.append({"sequence": index, "wall_time": trace.wall_time,
                           "phase": "observation" if trace.wall_time >= observation_started else "warmup",
                           "graph_id": trace.graph_id, "graph_name": graph.name,
                           "op_name": trace.op_name, "op_type": trace.op_type,
                           "input_names": list(op.input_names or ()),
                           "output_slot": trace.output_slot, "dtype": x.dtype.str,
                           "shape": list(x.shape), "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                           "values": x.tolist()})
    if not any(x["phase"] == "observation" for x in values):
        raise ValueError("no real-observation tensor traces")
    return {"schema": SCHEMA + ".tensors", "float_elements": count, "traces": values,
            "limitations": ["Execution order alone cannot identify the first causal divergence.",
                           "Pair graph nodes and input edges before attributing an operation.",
                           "The filter excludes constants, resource reads and random sampling nodes."]}


def worker(plan, root, output):
    import resource
    import modal_panel_policy_canary as native
    verify_plan(plan, root)
    intent = json.loads((output / "intent.json").read_text())
    if (intent["coordinator_pid"] != os.getppid()
            or os.getpgid(0) != os.getpid() or os.getsid(0) != os.getpid()
            or intent["plan_sha256"] != digest(output / "plan.json")):
        raise ValueError("worker requires its one-shot owned coordinator")
    create_once(output / "worker-started.json", {"pid": os.getpid(), "started_unix": time.time()})
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS["file_bytes_hard"], LIMITS["file_bytes_hard"]))
    sys.path.insert(0, str(root / "src"))
    from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow
    expected_flags = plan["baseline"]["environment"]["environment_flags"]
    if any(os.environ.get(k) != v for k, v in expected_flags.items()):
        raise ValueError("original native environment flags required")
    setup = configure_cpu_tensorflow(plan["native_plan"]["seed"])
    import tensorflow as tf
    import torch
    from tensorflow.python.util import _pywrap_util_port
    if tf.__version__ != TF_VERSION or tf.version.GIT_VERSION != TF_GIT or _pywrap_util_port.IsMklEnabled():
        raise ValueError("original TensorFlow commit and non-oneDNN CPU required")
    torch.manual_seed(plan["native_plan"]["seed"])
    torch.set_num_threads(1)
    states = native._read_states(root / plan["native_plan"]["replay"]["path"], 1)
    from melee_policy.integration import slippi_ai_policy as boundary
    boundary._activate_pinned_source(root / plan["native_plan"]["slippi_source"])
    from slippi_ai.tf.agents import BasicAgent
    manifest = native.validate_plan(plan["native_plan"], root)
    state = {"warmup_calls": 0}
    events = {"sample": []}
    policy = None
    restore = lambda: None
    try:
        with warmup_observer(tf, BasicAgent, output / "dump", state):
            policy = native._build_session(plan["native_plan"], root, manifest)
            policy.start()
        if state["warmup_calls"] != 1:
            raise RuntimeError("native warmup boundary missing")
        restore = native.attach_slippi_observers(policy, events)
        observation_started = time.time()
        command = policy.step(states[0])
        observation_finished = time.time()
        if len(events["sample"]) != 1:
            raise RuntimeError("one observation must cause one native inference")
        row = {"frame": int(states[0].frame), "generation": 0, "command": command.as_dict(),
               "numeric": {"sample": events["sample"][0]}}
        diagnostics, metadata = policy.diagnostics(), policy.metadata()
    finally:
        restore()
        if policy is not None:
            policy.close()
        if "writer" in state:
            state["writer"].FlushNonExecutionFiles()
            state["writer"].FlushExecutionFiles()
            tf.debugging.experimental.disable_dump_debug_info()
    verify_plan(plan, root)
    boundary.verify_runtime_assets(policy.config)
    checks = {
        "same_host_bitwise_row": bool(bitwise_equal(row, plan["baseline"]["row"])),
        "native_runtime_contract": metadata["runtime_contract"] == plan["baseline"]["runtime_contract"],
        "one_inference": diagnostics["frames_total"] == 1,
        "one_barrier": diagnostics["current_frame_inference_barriers"] == 1,
        "native_decoder": diagnostics["capture_decoder_assertions"] == 1
                          and diagnostics["capture_decoder_mismatches"] == 0,
        "no_reset": diagnostics["resets"] == 0 and diagnostics["generation"] == 0,
        "no_fault": diagnostics["fault"] is None,
        "frame_minus123": row["frame"] == -123,
    }
    receipt = {"schema": SCHEMA + ".result", "checks": checks, "passed": all(checks.values()),
               "status": "unperturbed-first-observation" if all(checks.values()) else "observer-perturbed-or-native-failure",
               "row": row, "diagnostics": diagnostics, "setup": setup,
               "warmup_calls": state["warmup_calls"], "warmup_finished_unix": state["warmup_finished_unix"],
               "observation_started_unix": observation_started, "observation_finished_unix": observation_finished,
               "tensorflow": {"version": tf.__version__, "git": tf.version.GIT_VERSION,
                              "compiler": tf.version.COMPILER_VERSION,
                              "build_info": tf.sysconfig.get_build_info()},
               "policy_steps": 1, "live_games": 0, "strict_policy_gate_changed": False}
    create_once(output / "observation.json", receipt)
    tensors = read_dump(output / "dump", observation_started)
    create_once(output / "tensors.json", tensors)
    files = [{"path": str(p.relative_to(output)), **identity(p)} for p, _ in inspect_tree(output)]
    create_once(output / "files.json", files)


def run(plan_path, approved):
    root = Path.cwd().resolve()
    plan_path = Path(plan_path).absolute()
    if plan_path != plan_path.resolve() or digest(plan_path) != approved:
        raise ValueError("exact canonical plan and approved hash required")
    plan = json.loads(plan_path.read_text())
    verify_plan(plan, root)
    output = plan_path.parent
    create_once(output / "intent.json", {"plan_sha256": approved, "started_unix": time.time(),
                                       "coordinator_pid": os.getpid(),
                                       "no_retry": True, "deadline_seconds": LIMITS["wall_seconds"]})
    process = None
    failure = None
    started = time.monotonic()
    try:
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, str(root / HELPER), "--worker", str(plan_path),
                                        "--approved-plan-sha", approved],
                                       cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            create_once(output / "worker.json", {"pid": process.pid})
            while process.poll() is None:
                inspect_tree(output)
                if time.monotonic() - started > LIMITS["wall_seconds"]:
                    raise TimeoutError("absolute worker deadline")
                time.sleep(LIMITS["poll_seconds"])
            if process.returncode != 0:
                raise RuntimeError(f"worker exit {process.returncode}")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    create_once(output / "execution.json", {"plan_sha256": approved, "failure": failure,
                                         "wall_seconds": time.monotonic() - started,
                                         "partials_retained": True,
                                         "note": "Parent completion alone is not an observer or policy pass."})
    if failure:
        raise RuntimeError(failure["message"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--run", type=Path)
    group.add_argument("--worker", type=Path)
    parser.add_argument("--approved-plan-sha")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.prepare:
        output = args.prepare.absolute()
        if output != output.resolve() or output.exists() or not output.parent.is_dir():
            raise ValueError("prepare requires a fresh canonical output directory")
        plan = make_plan(root, platform.machine())
        output.mkdir()
        create_once(output / "plan.json", plan)
        print(json.dumps({"plan": str(output / "plan.json"), **identity(output / "plan.json")}))
    elif args.run:
        run(args.run, args.approved_plan_sha)
    else:
        if digest(args.worker) != args.approved_plan_sha:
            raise ValueError("worker approved plan hash mismatch")
        worker(json.loads(args.worker.read_text()), root, args.worker.parent)


if __name__ == "__main__":
    main()
