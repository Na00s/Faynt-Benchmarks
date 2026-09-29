#!/usr/bin/env python3
"""Fixed-observation CPU policy portability canary. No emulator or cloud calls.

Each fresh process observes the existing native policy for 320 replay frames,
then an explicit game reset and the first 64 frames again. Sampling, tokenizers,
context, FIFO, source verifiers and controller decoding remain native. Numeric
observers copy returned values without another policy or random-number call.
The 256-entry Frisson ring wraps before reset; actor-context metadata is 128.

This is a policy compatibility gate. Emulator and closed-loop parity require
separate gates. A failed comparison never relaxes its fixed tolerances.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import sys
import time
from collections.abc import Mapping
from itertools import chain
from pathlib import Path

SCHEMA = "e011.modal_policy_canary.plan.v1"
RESULT_SCHEMA = "e011.modal_policy_canary.result.v1"
MANIFEST_SHA = "f80624560cc51283dc894787984de6089a42fd1e9f451ed2f1e845f4d33a6115"
MANIFEST_PATH = "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/manifest.json"
REPLAY_PATH = ".e000-cache/raw/039d70ae59bc37fc1fe5b72c1dadcd95d2b2877b9ad1964bf978aaa6662962ad.slp"
MODEL_SOURCE = ".e010-cache/frisson-ai-source/268031e7bddebb4e8c7a40026cd0b95f824d9d0d"
SLIPPI_SOURCE = ".e001-cache/slippi-ai-source"
FRAMES = 320
RESET_FRAMES = 64
TIMEOUT_SECONDS = 240
RECEIPT_BYTES = 128 * 1024**2
ERROR_RECEIPT_BYTES = 64 * 1024
ATOL = RTOL = 1e-5
CASES = (
    ("frisson", "75m"),
    ("frisson", "10m"),
    ("slippi", "fox_d21_ditto_v4"),
    ("slippi", "fox_d24_ditto_v4"),
    ("slippi", "gm"),
)
SOURCE_FILES = (
    "scripts/modal_panel_policy_canary.py",
    "scripts/slippi_panel_runtime.py",
    "scripts/slippi_panel_audit.py",
    "src/melee_policy/__init__.py",
    "src/melee_policy/integration/__init__.py",
    "src/melee_policy/integration/frisson_policy.py",
    "src/melee_policy/integration/slippi_ai_policy.py",
    "src/melee_policy/integration/slippi_compatibility.py",
    "src/melee_policy/integration/post_rl_checkpoints.py",
)
DEPENDENCIES = (
    "numpy",
    "torch",
    "tf_nightly",
    "tfp-nightly",
    "tf_keras-nightly",
    "dm-sonnet",
    "dm-tree",
    "melee",
    "pyenet-vladfi",
    "py-ubjson",
    "pyarrow",
    "fsspec",
    "tqdm",
    "py7zr",
    "fancyflags",
    "portpicker",
    "wandb",
    "absl-py",
    "PyYAML",
    "pandas",
    "pyvers",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(root, path):
    resolved = (Path(root) / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": path, "sha256": digest(resolved), "bytes": resolved.stat().st_size}


def safe_relative(path):
    if type(path) is not str or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("asset paths must be nonempty relative paths without traversal")
    return path


def check_identity(root, record):
    if set(record) != {"path", "sha256", "bytes"}:
        raise ValueError("asset identity requires path, sha256 and bytes")
    safe_relative(record["path"])
    if type(record["bytes"]) is not int or record["bytes"] < 0:
        raise ValueError("asset byte count must be nonnegative integer")
    if identity(root, record["path"]) != record:
        raise ValueError(f"asset changed: {record['path']}")


def make_plan(
    root,
    kind,
    key,
    *,
    port=1,
    seed=20260910,
    frisson_repository="..",
    frisson_revision="268031e7bddebb4e8c7a40026cd0b95f824d9d0d",
    frisson_subdirectory="melee-policy-frisson-ai",
    frisson_source=MODEL_SOURCE,
):
    """Return a plan for root to freeze; this function writes no files.

    Paths name existing assets. The explicit Frisson provenance repository may
    be outside root locally. A Linux package may use an authentic shallow Git
    object pack and its verified commit here. Native Git checks still run.
    """
    root = Path(root).resolve()
    if (kind, key) not in CASES:
        raise ValueError("case is outside the bounded initial compatibility panel")
    manifest = json.loads((root / MANIFEST_PATH).read_text())
    record = manifest["frisson_checkpoints" if kind == "frisson" else "releases"][key]
    checkpoint_path = (
        record["relative_path"] if kind == "frisson" else str(Path(".e001-cache/slippi-ai/models") / key)
    )
    files = list(SOURCE_FILES) + [
        f"{frisson_source}/{name}" for name in ("model.py", "controller_codec.py", "tensor_batch.py")
    ]
    plan = {
        "schema": SCHEMA,
        "kind": kind,
        "key": key,
        "seed": seed,
        "port": port,
        "frames": FRAMES,
        "reset_frames": RESET_FRAMES,
        "manifest": identity(root, MANIFEST_PATH),
        "replay": identity(root, REPLAY_PATH),
        "checkpoint": identity(root, checkpoint_path),
        "sources": [identity(root, path) for path in files],
        "slippi_source": SLIPPI_SOURCE,
        "frisson_source": frisson_source,
        "frisson_git": (
            None
            if frisson_repository is None
            else {
                "repository": frisson_repository,
                "revision": frisson_revision,
                "subdirectory": frisson_subdirectory,
            }
        ),
    }
    validate_plan(plan, root)
    return plan


def validate_plan(plan, root):
    expected = {
        "schema",
        "kind",
        "key",
        "seed",
        "port",
        "frames",
        "reset_frames",
        "manifest",
        "replay",
        "checkpoint",
        "sources",
        "slippi_source",
        "frisson_source",
        "frisson_git",
    }
    if set(plan) != expected or plan["schema"] != SCHEMA:
        raise ValueError("invalid canary plan schema")
    if (plan["kind"], plan["key"]) not in CASES:
        raise ValueError("case is outside frozen initial panel")
    if type(plan["port"]) is not int or plan["port"] not in (1, 2):
        raise ValueError("port must be one or two")
    if type(plan["seed"]) is not int or not 0 <= plan["seed"] < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (plan["frames"], plan["reset_frames"]) != (FRAMES, RESET_FRAMES):
        raise ValueError("fixed 320 plus 64 frame scope differs")
    for field in ("manifest", "replay", "checkpoint"):
        check_identity(root, plan[field])
    if plan["manifest"]["sha256"] != MANIFEST_SHA:
        raise ValueError("frozen panel manifest differs")
    if plan["replay"]["sha256"] != Path(REPLAY_PATH).stem:
        raise ValueError("fixed replay identity differs")
    for field in ("slippi_source", "frisson_source"):
        safe_relative(plan[field])
    provenance = plan["frisson_git"]
    if provenance is not None:
        if set(provenance) != {"repository", "revision", "subdirectory"}:
            raise ValueError("invalid Frisson Git provenance")
        if type(provenance["repository"]) is not str or not provenance["repository"]:
            raise ValueError("explicit Frisson Git repository required")
        revision = provenance["revision"]
        if (
            type(revision) is not str
            or len(revision) != 40
            or any(x not in "0123456789abcdef" for x in revision)
        ):
            raise ValueError("exact Frisson Git revision required")
        safe_relative(provenance["subdirectory"])
    wanted = set(SOURCE_FILES) | {
        f"{plan['frisson_source']}/{name}" for name in ("model.py", "controller_codec.py", "tensor_batch.py")
    }
    paths = [record["path"] for record in plan["sources"]]
    if len(paths) != len(set(paths)) or set(paths) != wanted:
        raise ValueError("source allowlist differs")
    for record in plan["sources"]:
        check_identity(root, record)
    manifest = json.loads((Path(root) / plan["manifest"]["path"]).read_text())
    record = manifest["frisson_checkpoints" if plan["kind"] == "frisson" else "releases"][plan["key"]]
    length = record["byte_length" if plan["kind"] == "frisson" else "bytes"]
    if (plan["checkpoint"]["sha256"], plan["checkpoint"]["bytes"]) != (record["sha256"], length):
        raise ValueError("checkpoint is outside frozen panel identity")
    return manifest


def numeric_tree(value):
    """Copy nested native numeric values. No RNG, tensor mutation or inference."""
    import numpy as np

    if dataclasses.is_dataclass(value):
        return {field.name: numeric_tree(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("numeric mappings require string keys")
        return {key: numeric_tree(child) for key, child in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {name: numeric_tree(getattr(value, name)) for name in value._fields}
    if isinstance(value, (tuple, list)):
        return [numeric_tree(child) for child in value]
    if value is None or type(value) is str:
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    array = np.array(value, copy=True)
    if array.dtype.kind not in "bifu" or not bool(np.isfinite(array).all()):
        raise ValueError("numeric evidence must be finite real, integer or boolean")
    return {"dtype": array.dtype.str, "shape": list(array.shape), "values": array.tolist()}


def observed_call(function, sink, *, include_inputs=True):
    """Transparent wrapper: invoke function once, return its identical object."""

    def call(*args, **kwargs):
        inputs = numeric_tree({"args": args, "kwargs": kwargs}) if include_inputs else None
        output = function(*args, **kwargs)
        sink.append({"inputs": inputs, "outputs": numeric_tree(output)})
        return output

    return call


def cache_status(cache):
    return {
        "capacity": int(cache.capacity),
        **{
            name: numeric_tree(getattr(cache, name))
            for name in ("valid_length", "write_position", "next_position")
        },
    }


def attach_frisson_observers(policy_session, events):
    runtime = policy_session._runtime
    policy = runtime._policy
    original_forward = policy.encoder.forward
    original_step = policy.backbone.step
    original_sample = policy.controller_head.sample
    policy.encoder.forward = observed_call(original_forward, events["encoder"])

    def backbone(encoded, cache, **kwargs):
        before = cache_status(cache)
        hidden, updated = original_step(encoded, cache, **kwargs)
        events["backbone"].append(
            {
                "before": before,
                "after": cache_status(updated),
                "reset_mask": numeric_tree(kwargs["reset_mask"]),
                "hidden": numeric_tree(hidden),
            }
        )
        return hidden, updated

    def sample(*args, **kwargs):
        # The private torch.Generator is deliberately excluded from serialization.
        output = original_sample(*args, **kwargs)
        events["sample"].append(numeric_tree(output))
        return output

    policy.backbone.step = backbone
    policy.controller_head.sample = sample

    def restore():
        policy.encoder.forward = original_forward
        policy.backbone.step = original_step
        policy.controller_head.sample = original_sample

    return restore


def attach_slippi_observers(policy_session, events):
    basic = policy_session._runtime._policy._agent._agent
    original = basic._sample
    basic._sample = observed_call(original, events["sample"])
    return lambda: setattr(basic, "_sample", original)


def _read_states(replay, count):
    from melee_policy.integration.slippi_compatibility import _default_console_factory

    console = _default_console_factory(replay)
    try:
        if not console.connect():
            raise RuntimeError("fixed replay connection failed")
        states = []
        for index in range(count):
            state = console.step()
            if state is None or state.frame != -123 + index or set(state.players) != {1, 2}:
                raise ValueError(f"replay frame or physical port mismatch at {index}")
            states.append(copy.deepcopy(state))
        return states
    finally:
        console.stop()


def _build_session(plan, root, manifest):
    from melee_policy.integration import frisson_policy as frisson
    from melee_policy.integration import slippi_ai_policy as slippi

    if plan["kind"] == "frisson":
        git = plan["frisson_git"]
        provenance = (
            {}
            if git is None
            else {
                "model_source_repository": (root / git["repository"]).resolve(),
                "model_source_revision": git["revision"],
                "model_source_subdirectory": git["subdirectory"],
            }
        )
        config = frisson.FrissonPolicyConfig(
            model_source_directory=root / plan["frisson_source"],
            slippi_ai_source_directory=root / plan["slippi_source"],
            checkpoint_path=root / plan["checkpoint"]["path"],
            port=plan["port"],
            opponent_port=3 - plan["port"],
            evaluation_seed=plan["seed"],
            **provenance,
        )
        return frisson.FrissonPolicySession(config)
    from slippi_panel_runtime import register_release, session

    record = manifest["releases"][plan["key"]]
    register_release(record)
    config = slippi.SlippiAIPolicyConfig(
        source_directory=root / plan["slippi_source"],
        checkpoint_path=root / plan["checkpoint"]["path"],
        port=plan["port"],
        opponent_port=3 - plan["port"],
        release=plan["key"],
        name=record["selected_name"],
        policy_delay_frames=record["delay"],
        console_delay_frames=0,
    )
    return session(config, record=record)


def observation_gate(plan, rows, diagnostics):
    total = FRAMES + RESET_FRAMES
    checks = {
        "384_observations": len(rows) == total,
        "replay_frame_sequence": [row["frame"] for row in rows]
        == list(range(-123, 197)) + list(range(-123, -59)),
        "generation_sequence": [row["generation"] for row in rows] == [0] * FRAMES + [1] * RESET_FRAMES,
        "all_frames_inferred": diagnostics.get("frames_total") == total,
        "barrier_every_frame": diagnostics.get("current_frame_inference_barriers") == total,
        "one_explicit_reset": diagnostics.get("resets") == 1,
        "reset_generation": diagnostics.get("generation") == 1,
        "64_frames_after_reset": diagnostics.get("frames_in_generation") == RESET_FRAMES,
        "no_fault": diagnostics.get("fault") is None,
    }
    if plan["kind"] == "frisson":
        import numpy as np

        checks["native_cache_resets"] = diagnostics["upstream"]["cache_resets"] == 2
        checks["native_parser_resets"] = diagnostics["upstream"]["parser_resets"] == 2
        checks["256_cache_ring_counter_sequence"] = all(
            row["numeric"]["backbone"]["after"]
            == {
                "capacity": 256,
                "valid_length": numeric_tree(np.asarray([min(index % FRAMES + 1, 256)], dtype=np.int64)),
                "write_position": numeric_tree(np.asarray([(index % FRAMES + 1) % 256], dtype=np.int64)),
                "next_position": numeric_tree(np.asarray([index % FRAMES + 1], dtype=np.int64)),
            }
            for index, row in enumerate(rows)
        )
    else:
        checks["native_decoder_exact"] = diagnostics.get("capture_decoder_mismatches") == 0
        checks["native_decoder_every_frame"] = diagnostics.get("capture_decoder_assertions") == total
    return checks


def run_observations(policy, states, kind, events):
    """Drive the frozen sequence; only the existing session advances policy state."""
    rows = []
    for generation, sequence in enumerate((states, states[:RESET_FRAMES])):
        if generation:
            policy.reset("fixed-observation-portability-canary")
        for state in sequence:
            counts = {name: len(values) for name, values in events.items()}
            command = policy.step(copy.deepcopy(state))
            needed = ("encoder", "backbone", "sample") if kind == "frisson" else ("sample",)
            if any(len(events[name]) != counts[name] + 1 for name in needed):
                raise RuntimeError("observer count differs from exactly one native inference")
            rows.append(
                {
                    "frame": int(state.frame),
                    "generation": generation,
                    "command": command.as_dict(),
                    "numeric": {name: events[name][-1] for name in needed},
                }
            )
    return rows


def run_canary(plan, root):
    root = Path(root).resolve()
    manifest = validate_plan(plan, root)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow

    started = time.monotonic()
    setup = configure_cpu_tensorflow(plan["seed"])
    import torch

    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    states = _read_states(root / plan["replay"]["path"], FRAMES)
    policy = _build_session(plan, root, manifest)
    events = {name: [] for name in ("encoder", "backbone", "sample")}

    def restore():
        return None

    try:
        policy.start()
        restore = (attach_frisson_observers if plan["kind"] == "frisson" else attach_slippi_observers)(
            policy, events
        )
        rows = run_observations(policy, states, plan["kind"], events)
        diagnostics, metadata = policy.diagnostics(), policy.metadata()
    finally:
        restore()
        policy.close()
    validate_plan(plan, root)
    checks = observation_gate(plan, rows, diagnostics)
    runtime_contract = metadata["runtime_contract"]
    if plan["kind"] == "slippi":
        checks["native_policy_delay"] = (
            runtime_contract["effective_policy_delay_frames"] == manifest["releases"][plan["key"]]["delay"]
        )
        checks["native_name"] = (
            runtime_contract["effective_player_name"] == manifest["releases"][plan["key"]]["selected_name"]
        )
        from melee_policy.integration.slippi_ai_policy import verify_runtime_assets

        verify_runtime_assets(policy.config)
    else:
        checks["frisson_ring_contract"] = (
            runtime_contract["context_mode"] == "ring"
            and runtime_contract["actor_trajectory_context_frames"] == 128
            and runtime_contract["kv_cache_capacity_frames"] == 256
            and runtime_contract["delay_frames"] == 0
        )
        from melee_policy.integration.frisson_policy import verify_source_assets

        verify_source_assets(policy.config)
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema": RESULT_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "plan": plan,
        "rows": rows,
        "diagnostics": diagnostics,
        "metadata": metadata,
        "runtime_contract": runtime_contract,
        "tensorflow_setup": setup,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "versions": versions,
            "torch_threads": torch.get_num_threads(),
            "environment_flags": {
                key: os.environ.get(key)
                for key in (
                    "TF_ENABLE_ONEDNN_OPTS",
                    "OMP_NUM_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                    "TF_NUM_INTRAOP_THREADS",
                )
            },
        },
        "policy_steps": len(rows),
        "live_games": 0,
        "source_assets_unchanged": True,
        "wall_seconds": time.monotonic() - started,
        "limitations": [
            "Fixed replay observations do not establish emulator or closed-loop parity.",
            "Numeric observers add wall-time overhead; this is not an isolated latency benchmark.",
            "Frisson uses its existing 256-entry ring, with actor-context metadata 128; "
            "native RL prefix differs.",
        ],
    }


def compare_numeric(left, right, *, atol=ATOL, rtol=RTOL):
    """Require structure/dtype/discrete equality; report fixed-bound float errors."""
    import numpy as np

    errors, totals = [], {"float_elements": 0, "discrete_elements": 0, "max_abs": 0.0, "max_rel": 0.0}

    def walk(a, b, path):
        if type(a) is not type(b):
            errors.append({"path": path, "reason": "structure type differs"})
            return
        if isinstance(a, dict) and set(a) == {"dtype", "shape", "values"}:
            if set(b) != set(a) or a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                errors.append({"path": path, "reason": "numeric dtype or shape differs"})
                return
            x, y = np.asarray(a["values"], dtype=a["dtype"]), np.asarray(b["values"], dtype=b["dtype"])
            if list(x.shape) != a["shape"] or list(y.shape) != b["shape"] or x.dtype.kind not in "bifu":
                errors.append({"path": path, "reason": "invalid numeric representation"})
                return
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                errors.append({"path": path, "reason": "nonfinite numeric values"})
                return
            if x.dtype.kind == "f":
                totals["float_elements"] += x.size
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                relative = delta / np.maximum(np.abs(x.astype(np.float64)), 1e-12)
                absolute_max = float(np.max(delta, initial=0))
                totals["max_abs"] = max(totals["max_abs"], absolute_max)
                totals["max_rel"] = max(totals["max_rel"], float(np.max(relative, initial=0)))
                if not np.all(delta <= atol + rtol * np.abs(x)):
                    errors.append(
                        {"path": path, "reason": "float tolerance exceeded", "max_abs": absolute_max}
                    )
            else:
                totals["discrete_elements"] += x.size
                if not np.array_equal(x, y):
                    errors.append({"path": path, "reason": "discrete values differ"})
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                errors.append({"path": path, "reason": "mapping keys differ"})
                return
            for key in a:
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                errors.append({"path": path, "reason": "sequence length differs"})
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]")
        elif a != b:
            errors.append({"path": path, "reason": "scalar differs"})

    walk(left, right, "numeric")
    return {
        "passed": not errors,
        "atol": atol,
        "rtol": rtol,
        "relative_denominator_floor": 1e-12,
        **totals,
        "mismatch_count": len(errors),
        "first_mismatches": errors[:50],
    }


def compare_results(left, right):
    for result in (left, right):
        if result.get("schema") != RESULT_SCHEMA or result.get("passed") is not True:
            raise ValueError("comparison requires two passing complete native canary receipts")
        if len(result.get("rows", [])) != FRAMES + RESET_FRAMES:
            raise ValueError("comparison requires all 384 steps")
        if not result.get("checks") or not all(value is True for value in result["checks"].values()):
            raise ValueError("comparison requires all native checks")
        if not all(observation_gate(result["plan"], result["rows"], result["diagnostics"]).values()):
            raise ValueError("receipt observation gates do not reproduce")
    a, b = left["plan"], right["plan"]
    fields = ("kind", "key", "seed", "port", "frames", "reset_frames")
    bindings_equal = all(a[name] == b[name] for name in fields)
    for name in ("manifest", "checkpoint", "replay"):
        bindings_equal &= (a[name]["sha256"], a[name]["bytes"]) == (b[name]["sha256"], b[name]["bytes"])

    def source_roles(plan):
        return {
            record["path"].replace(plan["frisson_source"] + "/", "frisson-source/"): (
                record["sha256"],
                record["bytes"],
            )
            for record in plan["sources"]
        }

    bindings_equal &= source_roles(a) == source_roles(b)
    commands_equal = all(
        x["command"] == y["command"] for x, y in zip(left["rows"], right["rows"], strict=True)
    )
    frame_context_equal = [(x["frame"], x["generation"]) for x in left["rows"]] == [
        (x["frame"], x["generation"]) for x in right["rows"]
    ]
    numeric = compare_numeric(
        [row["numeric"] for row in left["rows"]], [row["numeric"] for row in right["rows"]]
    )
    diagnostic_fields = (
        "frames_total",
        "frames_in_generation",
        "current_frame_inference_barriers",
        "resets",
        "generation",
        "fault",
    )
    diagnostics_equal = all(
        left["diagnostics"].get(name) == right["diagnostics"].get(name) for name in diagnostic_fields
    )
    checks = {
        "identical_input_bindings": bool(bindings_equal),
        "exact_commands": commands_equal,
        "frame_context_equal": frame_context_equal,
        "reset_barrier_counts_equal": diagnostics_equal,
        "numeric_parity": numeric["passed"],
        "native_checks_match": left["checks"] == right["checks"],
        "runtime_contract_equal": left["runtime_contract"] == right["runtime_contract"],
    }
    return {
        "schema": "e011.modal_policy_canary.comparison.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "numeric": numeric,
        "command_mismatch_indices": [
            i
            for i, (x, y) in enumerate(zip(left["rows"], right["rows"], strict=True))
            if x["command"] != y["command"]
        ],
        "left_environment": left["environment"],
        "right_environment": right["environment"],
    }


def write_new(path, value, *, max_bytes=RECEIPT_BYTES, deadline_monotonic=None):
    """Stream bounded UTF-8 JSON exclusively, retaining any partial on failure."""
    if type(max_bytes) is not int or not 0 < max_bytes <= RECEIPT_BYTES:
        raise ValueError("receipt byte limit must be a positive integer up to 128 MiB")
    encoder = json.JSONEncoder(separators=(",", ":"), sort_keys=True, allow_nan=False)
    pending = bytearray()
    count = 0
    output_hash = hashlib.sha256()

    def check_deadline():
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("canary deadline exhausted during receipt serialization")

    def flush_block(stream):
        if pending:
            check_deadline()
            if stream.write(pending) != len(pending):
                raise OSError("short receipt write; partial output retained")
            output_hash.update(pending)
            pending.clear()

    check_deadline()
    with Path(path).open("xb") as stream:
        for chunk in chain(encoder.iterencode(value), ("\n",)):
            check_deadline()
            encoded = chunk.encode("utf-8")
            count += len(encoded)
            if count > max_bytes:
                raise ValueError(f"receipt exceeds {max_bytes} bytes; partial output retained")
            pending.extend(encoded)
            if len(pending) >= 64 * 1024:
                flush_block(stream)
        flush_block(stream)
        check_deadline()
        stream.flush()
        os.fsync(stream.fileno())
        check_deadline()
    return {"bytes": count, "sha256": output_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", type=Path)
    group.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; every original receipt must be retained")
    plan_bytes = args.plan.read_bytes() if args.plan else None
    if args.plan and (
        not args.approved_plan_sha256 or hashlib.sha256(plan_bytes).hexdigest() != args.approved_plan_sha256
    ):
        parser.error("exact approved plan SHA-256 is required before policy execution")

    def deadline(*_):
        signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
        signal.alarm(10)
        raise TimeoutError("240-second canary ceiling exceeded; cleanup has a 10-second ceiling")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(TIMEOUT_SECONDS)
    started = time.time()
    deadline_monotonic = time.monotonic() + TIMEOUT_SECONDS
    try:
        try:
            if args.compare:
                result = compare_results(*(json.loads(path.read_text()) for path in args.compare))
                result["source_receipts"] = [
                    {"path": str(path), "sha256": digest(path)} for path in args.compare
                ]
            else:
                result = run_canary(json.loads(plan_bytes), args.root)
                result["approved_plan_sha256"] = args.approved_plan_sha256
        except BaseException as error:
            result = {"schema": RESULT_SCHEMA, "passed": False, "error": f"{type(error).__name__}: {error}"}
        result["started_unix"], result["finished_unix"] = started, time.time()
        result["receipt_byte_limit"] = RECEIPT_BYTES
        written = write_new(args.output, result, deadline_monotonic=deadline_monotonic)
        if digest(args.output) != written["sha256"]:
            raise OSError("written receipt SHA-256 verification failed")
        print(json.dumps({"passed": result["passed"], "output": str(args.output), **written}))
        return 0 if result["passed"] else 1
    except BaseException as error:
        # The active alarm also bounds hashing and this best-effort small error
        # receipt. A fired deadline keeps the existing ten-second cleanup alarm.
        failure_path = args.output.with_name(args.output.name + ".error.json")
        failure = {
            "schema": RESULT_SCHEMA,
            "passed": False,
            "error": f"{type(error).__name__}: {error}"[:2048],
            "started_unix": started,
            "finished_unix": time.time(),
            "primary_output": str(args.output),
            "primary_output_retained": args.output.exists(),
            "primary_output_bytes": args.output.stat().st_size if args.output.exists() else None,
            "primary_output_admitted": False,
            "receipt_byte_limit": RECEIPT_BYTES,
        }
        try:
            written = write_new(failure_path, failure, max_bytes=ERROR_RECEIPT_BYTES)
            print(json.dumps({"passed": False, "output": str(failure_path), **written}))
        except BaseException as receipt_error:
            print(
                json.dumps({"passed": False, "error_receipt_failure": str(receipt_error)[:2048]}),
                file=sys.stderr,
            )
        return 1
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
