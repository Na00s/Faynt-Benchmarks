from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import modal_panel_policy_canary as canary
import numpy as np
import pytest


def test_fixed_initial_scope_covers_both_profiles_delays_and_ports():
    assert canary.CASES == (
        ("frisson", "75m"),
        ("frisson", "10m"),
        ("slippi", "fox_d21_ditto_v4"),
        ("slippi", "fox_d24_ditto_v4"),
        ("slippi", "gm"),
    )
    assert canary.FRAMES > 256
    assert canary.RESET_FRAMES == 64
    assert canary.ATOL == canary.RTOL == 1e-5


@pytest.fixture
def plan(tmp_path, monkeypatch):
    manifest = {
        "frisson_checkpoints": {
            "75m": {
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "byte_length": 10,
                "relative_path": "checkpoint.pt",
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    replay = tmp_path / "replay.slp"
    replay.write_bytes(b"fixed replay")
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    sources = list(canary.SOURCE_FILES) + [
        f"model/{name}" for name in ("model.py", "controller_codec.py", "tensor_batch.py")
    ]
    for source in sources:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(canary, "MANIFEST_SHA", canary.digest(manifest_path))
    monkeypatch.setattr(canary, "REPLAY_PATH", canary.digest(replay) + ".slp")
    value = {
        "schema": canary.SCHEMA,
        "kind": "frisson",
        "key": "75m",
        "seed": 20260910,
        "port": 1,
        "frames": 320,
        "reset_frames": 64,
        "manifest": canary.identity(tmp_path, "manifest.json"),
        "replay": canary.identity(tmp_path, "replay.slp"),
        "checkpoint": canary.identity(tmp_path, "checkpoint.pt"),
        "sources": [canary.identity(tmp_path, source) for source in sources],
        "slippi_source": "slippi",
        "frisson_source": "model",
        "frisson_git": None,
    }
    return value, tmp_path


def test_plan_accepts_authentic_native_git_verifier_route(plan):
    value, root = plan
    assert canary.validate_plan(value, root)["frisson_checkpoints"]["75m"]["byte_length"] == 10


def test_plan_accepts_explicit_native_git_object_snapshot_route(plan):
    value, root = plan
    value["frisson_git"] = {
        "repository": "..",
        "revision": "a" * 40,
        "subdirectory": "melee-policy-frisson-ai",
    }
    canary.validate_plan(value, root)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frames", 128),
        ("reset_frames", 0),
        ("kind", "hal"),
        ("key", "master"),
        ("port", 0),
        ("port", True),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**32),
    ],
)
def test_plan_rejects_scope_changes(plan, key, value):
    document, root = plan
    document[key] = value
    with pytest.raises(ValueError):
        canary.validate_plan(document, root)


@pytest.mark.parametrize("path", ["/tmp/file", "../file", "a/../../file", ""])
def test_asset_paths_reject_escape(path):
    with pytest.raises(ValueError):
        canary.safe_relative(path)


def test_changed_checkpoint_or_source_fails_before_import(plan):
    document, root = plan
    (root / document["checkpoint"]["path"]).write_bytes(b"bad weights")
    with pytest.raises(ValueError, match="asset changed"):
        canary.validate_plan(document, root)


def test_missing_or_duplicate_source_binding_fails(plan):
    document, root = plan
    document["sources"][-1] = document["sources"][0]
    with pytest.raises(ValueError, match="allowlist"):
        canary.validate_plan(document, root)


def test_bound_checkpoint_still_must_match_frozen_manifest(plan):
    document, root = plan
    (root / "checkpoint.pt").write_bytes(b"different weights")
    document["checkpoint"] = canary.identity(root, "checkpoint.pt")
    with pytest.raises(ValueError, match="outside frozen panel"):
        canary.validate_plan(document, root)


def test_numeric_copy_is_independent_and_preserves_dtype_shape():
    source = np.array([[1.25, 2.5]], dtype=np.float32)
    copied = canary.numeric_tree(source)
    source[0, 0] = 7
    assert copied == {"dtype": "<f4", "shape": [1, 2], "values": [[1.25, 2.5]]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.array([object()]), np.array([1j])])
def test_numeric_evidence_rejects_nonfinite_and_nonnumeric(value):
    with pytest.raises(ValueError):
        canary.numeric_tree(value)


def test_numeric_copy_handles_dataclass_and_namedtuple_without_mutation():
    from collections import namedtuple

    @dataclasses.dataclass
    class Value:
        field: object

    Pair = namedtuple("Pair", ["x", "y"])
    source = Value(Pair(np.int32(3), np.bool_(True)))
    copied = canary.numeric_tree(source)
    assert copied["field"]["x"] == {"dtype": "<i4", "shape": [], "values": 3}
    assert copied["field"]["y"] == {"dtype": "|b1", "shape": [], "values": True}


def test_observer_invokes_native_once_returns_same_object_and_does_not_consume_rng():
    rng = np.random.RandomState(11)
    native_calls, records = [], []
    output = np.array([3.0], dtype=np.float32)

    def native(value, *, mask):
        native_calls.append((value, mask))
        return output

    rng_before = rng.get_state()
    argument, mask = np.array([1]), np.array([True])
    observed = canary.observed_call(native, records)(argument, mask=mask)
    assert observed is output
    assert len(native_calls) == len(records) == 1
    assert native_calls[0][0] is argument and native_calls[0][1] is mask
    assert np.array_equal(rng.get_state()[1], rng_before[1])
    assert rng.get_state()[2:] == rng_before[2:]
    output[0] = 7.0
    assert records[0]["outputs"]["values"] == [3.0]


def test_observer_preserves_native_exception_and_does_not_retry():
    calls, records = [], []

    def broken(value):
        calls.append(value)
        raise RuntimeError("native fault")

    with pytest.raises(RuntimeError, match="native fault"):
        canary.observed_call(broken, records)(np.int32(1))
    assert len(calls) == 1 and not records


def test_frisson_observers_preserve_generator_and_restore_native_methods():
    calls = []
    cache = SimpleNamespace(
        capacity=256, valid_length=np.array([0]), write_position=np.array([0]), next_position=np.array([0])
    )
    generator = object()
    output = np.array([3.0], dtype=np.float32)

    def encode(game, controller):
        calls.append(("encode", game, controller))
        return output

    def step(encoded, seen_cache, *, reset_mask):
        calls.append(("step", encoded, seen_cache, reset_mask))
        return output, seen_cache

    def sample(hidden, controller, *, generator, temperature):
        calls.append(("sample", hidden, controller, generator, temperature))
        return output

    policy = SimpleNamespace(
        encoder=SimpleNamespace(forward=encode),
        backbone=SimpleNamespace(step=step),
        controller_head=SimpleNamespace(sample=sample),
    )
    session = SimpleNamespace(_runtime=SimpleNamespace(_policy=policy))
    events = {name: [] for name in ("encoder", "backbone", "sample")}
    restore = canary.attach_frisson_observers(session, events)
    assert policy.encoder.forward(np.array([1]), np.array([2])) is output
    assert policy.backbone.step(output, cache, reset_mask=np.array([True]))[1] is cache
    assert (
        policy.controller_head.sample(output, np.array([2]), generator=generator, temperature=1.0) is output
    )
    assert [call[0] for call in calls] == ["encode", "step", "sample"]
    assert calls[-1][3] is generator
    assert all(len(values) == 1 for values in events.values())
    restore()
    assert (
        policy.encoder.forward is encode
        and policy.backbone.step is step
        and policy.controller_head.sample is sample
    )


def test_slippi_observer_keeps_native_compiled_function_and_restores():
    def native(value):
        return value

    basic = SimpleNamespace(_sample=native)
    session = SimpleNamespace(
        _runtime=SimpleNamespace(_policy=SimpleNamespace(_agent=SimpleNamespace(_agent=basic)))
    )
    events = {"sample": []}
    restore = canary.attach_slippi_observers(session, events)
    value = np.array([1.0], dtype=np.float32)
    assert basic._sample(value) is value
    assert len(events["sample"]) == 1
    restore()
    assert basic._sample is native


def _rows():
    rows = []
    for generation, count in enumerate((320, 64)):
        for index in range(count):
            after = canary.cache_status(
                SimpleNamespace(
                    capacity=256,
                    valid_length=np.asarray([min(index + 1, 256)], dtype=np.int64),
                    write_position=np.asarray([(index + 1) % 256], dtype=np.int64),
                    next_position=np.asarray([index + 1], dtype=np.int64),
                )
            )
            rows.append(
                {
                    "frame": -123 + index,
                    "generation": generation,
                    "command": {"buttons": []},
                    "numeric": {
                        "backbone": {"after": after},
                        "sample": canary.numeric_tree(np.array([1.0], dtype=np.float32)),
                    },
                }
            )
    return rows


def _diagnostics():
    return {
        "frames_total": 384,
        "frames_in_generation": 64,
        "current_frame_inference_barriers": 384,
        "resets": 1,
        "generation": 1,
        "fault": None,
        "upstream": {"cache_resets": 2, "parser_resets": 2},
    }


def test_ring_wrap_and_explicit_reset_gate():
    rows = _rows()
    assert all(canary.observation_gate({"kind": "frisson"}, rows, _diagnostics()).values())
    rows[128]["numeric"]["backbone"]["after"]["next_position"] = canary.numeric_tree(
        np.asarray([1], dtype=np.int64)
    )
    assert not canary.observation_gate({"kind": "frisson"}, rows, _diagnostics())[
        "256_cache_ring_counter_sequence"
    ]


def test_native_counter_shapes_and_exact_wrap_values():
    rows = _rows()
    for frame, valid, write, position in (
        (0, 1, 1, 1),
        (127, 128, 128, 128),
        (255, 256, 0, 256),
        (256, 256, 1, 257),
        (319, 256, 64, 320),
        (320, 1, 1, 1),
    ):
        assert rows[frame]["numeric"]["backbone"]["after"] == {
            "capacity": 256,
            "valid_length": {"dtype": "<i8", "shape": [1], "values": [valid]},
            "write_position": {"dtype": "<i8", "shape": [1], "values": [write]},
            "next_position": {"dtype": "<i8", "shape": [1], "values": [position]},
        }


@pytest.mark.parametrize("mutation", ["dtype", "shape", "list", "reset"])
def test_counter_gate_rejects_native_shape_dtype_and_reset_mutations(mutation):
    rows = _rows()
    counter = rows[320]["numeric"]["backbone"]["after"]["next_position"]
    if mutation == "list":
        rows[320]["numeric"]["backbone"]["after"]["next_position"] = canary.numeric_tree([1])
    elif mutation == "reset":
        counter["values"] = [321]
    elif mutation == "shape":
        counter["shape"] = []
    else:
        counter["dtype"] = "<i4"
    assert not canary.observation_gate({"kind": "frisson"}, rows, _diagnostics())[
        "256_cache_ring_counter_sequence"
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("frames_total", 383),
        ("resets", 0),
        ("current_frame_inference_barriers", 383),
        ("frames_in_generation", 320),
        ("fault", "failure"),
    ],
)
def test_missing_inference_or_reset_is_a_failure(field, value):
    diagnostics = _diagnostics()
    diagnostics[field] = value
    assert not all(canary.observation_gate({"kind": "frisson"}, _rows(), diagnostics).values())


@pytest.mark.parametrize(
    "left,right,passed",
    [
        (np.array([1], np.int32), np.array([1], np.int64), False),
        (np.array([1], np.int32), np.array([2], np.int32), False),
        (np.array([1.0], np.float32), np.array([1.000005], np.float32), True),
        (np.array([1.0], np.float32), np.array([1.001], np.float32), False),
        (np.array([1.0], np.float32), np.array([[1.0]], np.float32), False),
    ],
)
def test_numeric_comparison_has_fixed_bounds_and_exact_discrete_fields(left, right, passed):
    result = canary.compare_numeric(canary.numeric_tree(left), canary.numeric_tree(right))
    assert result["passed"] is passed
    assert result["atol"] == result["rtol"] == 1e-5


def test_numeric_comparison_rejects_nonfinite_forged_values():
    value = canary.numeric_tree(np.array([1.0], np.float32))
    corrupted = copy.deepcopy(value)
    corrupted["values"] = [float("nan")]
    assert not canary.compare_numeric(value, corrupted)["passed"]


@pytest.fixture
def receipts():
    plan = {
        "kind": "frisson",
        "key": "75m",
        "seed": 20260910,
        "port": 1,
        "frames": 320,
        "reset_frames": 64,
        "sources": [],
        "frisson_source": "model",
        **{name: {"sha256": name, "bytes": 1} for name in ("manifest", "checkpoint", "replay")},
    }
    result = {
        "schema": canary.RESULT_SCHEMA,
        "passed": True,
        "rows": _rows(),
        "plan": plan,
        "diagnostics": _diagnostics(),
        "checks": {"native": True},
        "environment": {"machine": "arm64"},
        "runtime_contract": {"context": "ring"},
    }
    return result, copy.deepcopy(result)


def test_same_receipt_comparison_passes_across_named_machines(receipts):
    left, right = receipts
    right["environment"]["machine"] = "x86_64"
    assert canary.compare_results(left, right)["passed"]


@pytest.mark.parametrize("mutation", ["command", "weights", "delay", "numeric", "frame", "source"])
def test_cross_platform_mismatch_fails_without_resampling_or_relaxation(receipts, mutation):
    left, right = receipts
    if mutation == "command":
        right["rows"][100]["command"]["buttons"] = ["A"]
    elif mutation == "weights":
        right["plan"]["checkpoint"]["sha256"] = "different"
    elif mutation == "delay":
        right["runtime_contract"]["delay"] = 2
    elif mutation == "numeric":
        right["rows"][150]["numeric"]["sample"]["values"] = [2.0]
    elif mutation == "frame":
        right["rows"][1]["generation"] = 3
    else:
        right["plan"]["sources"] = [{"path": "model/model.py", "sha256": "different", "bytes": 9}]
    if mutation == "frame":
        with pytest.raises(ValueError, match="observation gates"):
            canary.compare_results(left, right)
    else:
        assert not canary.compare_results(left, right)["passed"]


def test_truncated_or_failed_receipt_is_rejected(receipts):
    left, right = receipts
    right["rows"].pop()
    with pytest.raises(ValueError, match="384"):
        canary.compare_results(left, right)
    right["passed"] = False
    with pytest.raises(ValueError, match="passing"):
        canary.compare_results(left, right)


def test_output_is_exclusive_and_preserves_original(tmp_path):
    target = tmp_path / "result.json"
    canary.write_new(target, {"first": True})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        canary.write_new(target, {"second": True})
    assert target.read_bytes() == original


def test_streaming_output_preserves_every_numeric_value_and_exact_hash(tmp_path):
    target = tmp_path / "result.json"
    value = {"rows": [{"values": list(range(512))} for _ in range(384)], "label": "café"}
    record = canary.write_new(target, value)
    assert json.loads(target.read_bytes()) == value
    assert record == {"bytes": target.stat().st_size, "sha256": canary.digest(target)}
    assert record["bytes"] <= canary.RECEIPT_BYTES == 128 * 1024**2


def test_compact_slippi_trace_density_preserves_all_native_values(tmp_path):
    target = tmp_path / "gm-trace.json"
    # Representative native observation shape: nested typed tensors with long
    # float32 hidden-state/logit vectors, repeated for every original/reset frame.
    tensor = canary.numeric_tree(np.linspace(-3, 3, 2048, dtype=np.float32))
    value = {"rows": [{"sample": {"hidden": tensor, "logits": tensor}} for _ in range(384)]}
    record = canary.write_new(target, value)
    compact = target.read_bytes()
    assert json.loads(compact) == value
    assert (
        compact == (json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
    )
    pretty_bytes = len((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    assert record["bytes"] < pretty_bytes * 0.75
    assert len(json.loads(compact)["rows"]) == 384


def test_receipt_limit_is_enforced_and_partial_file_is_retained(tmp_path):
    target = tmp_path / "result.json"
    with pytest.raises(ValueError, match="partial output retained"):
        canary.write_new(target, {"values": list(range(100_000))}, max_bytes=100_000)
    assert 0 < target.stat().st_size <= 100_000
    partial = target.read_bytes()
    with pytest.raises(FileExistsError):
        canary.write_new(target, {"retry": True})
    assert target.read_bytes() == partial


@pytest.mark.parametrize("cap", [0, -1, True, 128 * 1024**2 + 1])
def test_receipt_rejects_invalid_or_enlarged_limit(tmp_path, cap):
    with pytest.raises(ValueError, match="byte limit"):
        canary.write_new(tmp_path / "result.json", {}, max_bytes=cap)


def test_receipt_deadline_is_checked_before_open(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    monkeypatch.setattr(canary.time, "monotonic", lambda: 10)
    with pytest.raises(TimeoutError, match="serialization"):
        canary.write_new(target, {}, deadline_monotonic=10)
    assert not target.exists()


def test_receipt_deadline_interrupts_streaming_and_retains_partial(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    count = 0

    def elapsed():
        nonlocal count
        count += 1
        return 11 if count > 30_000 else 0

    monkeypatch.setattr(canary.time, "monotonic", elapsed)
    with pytest.raises(TimeoutError, match="serialization"):
        canary.write_new(target, list(range(100_000)), deadline_monotonic=10)
    assert 0 < target.stat().st_size <= canary.RECEIPT_BYTES


def test_receipt_short_write_fails_without_retry(tmp_path, monkeypatch):
    calls = []

    class ShortWriter(io.BytesIO):
        def write(self, data):
            calls.append(len(data))
            return super().write(data[:1])

    monkeypatch.setattr(Path, "open", lambda *args: ShortWriter())
    with pytest.raises(OSError, match="short receipt write"):
        canary.write_new(tmp_path / "result.json", {"value": 1})
    assert len(calls) == 1


@pytest.fixture
def main_cli(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    target = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "canary",
            "--plan",
            str(plan),
            "--approved-plan-sha256",
            canary.digest(plan),
            "--output",
            str(target),
        ],
    )
    alarms = []
    monkeypatch.setattr(canary.signal, "signal", lambda *args: None)
    monkeypatch.setattr(canary.signal, "alarm", alarms.append)
    monkeypatch.setattr(canary, "run_canary", lambda *args: {"passed": True, "rows": list(range(384))})
    return target, alarms


def test_main_keeps_deadline_through_receipt_write_and_hash(main_cli, monkeypatch):
    target, alarms = main_cli
    original_write, original_digest = canary.write_new, canary.digest
    operations = []

    def write(*args, **kwargs):
        assert alarms[-1] == canary.TIMEOUT_SECONDS
        assert kwargs["deadline_monotonic"] > canary.time.monotonic()
        operations.append("write")
        return original_write(*args, **kwargs)

    def digest(path):
        assert alarms[-1] == canary.TIMEOUT_SECONDS
        operations.append("hash")
        return original_digest(path)

    monkeypatch.setattr(canary, "write_new", write)
    monkeypatch.setattr(canary, "digest", digest)
    assert canary.main() == 0
    assert operations == ["write", "hash"]
    assert alarms == [canary.TIMEOUT_SECONDS, 0]
    assert len(json.loads(target.read_text())["rows"]) == 384


def test_main_retains_failed_primary_and_creates_small_error_receipt(main_cli, monkeypatch):
    target, alarms = main_cli
    original_write = canary.write_new

    def write(path, value, **kwargs):
        assert alarms[-1] == canary.TIMEOUT_SECONDS
        if path == target:
            kwargs["max_bytes"] = 100_000
            value = {"values": list(range(100_000))}
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(canary, "write_new", write)
    assert canary.main() == 1
    assert 0 < target.stat().st_size <= 100_000
    error_path = target.with_name("result.json.error.json")
    error = json.loads(error_path.read_text())
    assert error["passed"] is error["primary_output_admitted"] is False
    assert error["primary_output_retained"] is True
    assert error["primary_output_bytes"] == target.stat().st_size
    assert error_path.stat().st_size < canary.ERROR_RECEIPT_BYTES
    assert alarms[-1] == 0


def test_source_does_not_call_emulator_cloud_network_or_asset_verifier_replacement():
    source = Path(canary.__file__).read_text()
    assert "run_game(" not in source and "modal.App" not in source
    assert "verify_source_assets =" not in source and "verify_runtime_assets =" not in source
    assert "requests.get(" not in source and "urlopen(" not in source


def test_native_numpy_frame_numbers_export_as_json_and_reset_once():
    events, resets = {"sample": []}, []
    command = SimpleNamespace(as_dict=lambda: {"buttons": []})

    def step(state):
        events["sample"].append(canary.numeric_tree(np.array([state.frame], np.int32)))
        return command

    policy = SimpleNamespace(step=step, reset=resets.append)
    states = [SimpleNamespace(frame=np.int32(i)) for i in range(-123, 197)]
    rows = canary.run_observations(policy, states, "slippi", events)
    assert len(rows) == 384 and len(resets) == 1
    assert all(type(row["frame"]) is int for row in rows)
    assert json.loads(json.dumps(rows)) == rows
    assert rows[320]["frame"] == -123


def test_missing_observer_event_stops_immediately_without_retry():
    count = []
    command = SimpleNamespace(as_dict=lambda: {})

    def step(state):
        count.append(state.frame)
        return command

    with pytest.raises(RuntimeError, match="exactly one native"):
        canary.run_observations(
            SimpleNamespace(step=step), [SimpleNamespace(frame=-123)], "slippi", {"sample": []}
        )
    assert count == [-123]
