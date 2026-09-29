"""Synthetic tests only. No TensorFlow import, policy, replay, or cloud call."""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/modal_panel_first_op_diagnostic.py"
spec = importlib.util.spec_from_file_location("first_op_diagnostic", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def leaf(values=(1.0,), dtype="<f4"):
    return {"dtype": dtype, "shape": [len(values)], "values": list(values)}


@pytest.mark.parametrize("left,right,expected", [
    (leaf(), leaf(), True),
    (leaf((0.,)), leaf((-0.,)), False),
    (leaf(), leaf((1.0000001192092896,)), False),
    (leaf(), leaf(dtype="<f8"), False),
    (leaf(), {**leaf(), "shape": [2]}, False),
    ({"x": [leaf()]}, {"x": [leaf()]}, True),
    ({"x": [leaf()]}, {"x": []}, False),
    ({"x": 1}, {"y": 1}, False),
    (1, True, False), (0., -0., False),
    (leaf((float("nan"),)), leaf((float("nan"),)), False),
])
def test_bitwise_oracle(left, right, expected):
    assert bool(m.bitwise_equal(left, right)) is expected


def test_create_once_and_bound(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    m.create_once(p, {"a": 1})
    with pytest.raises(FileExistsError):
        m.create_once(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}
    monkeypatch.setitem(m.LIMITS, "file_bytes_hard", 2)
    with pytest.raises(ValueError):
        m.create_once(tmp_path / "too-big", {"a": 1})


@pytest.mark.parametrize("kind", ["symlink", "file-count", "individual", "aggregate"])
def test_tree_caps(tmp_path, monkeypatch, kind):
    (tmp_path / "a").write_bytes(b"abcd")
    if kind == "symlink":
        (tmp_path / "alias").symlink_to(tmp_path / "a")
    elif kind == "file-count":
        monkeypatch.setitem(m.LIMITS, "files", 0)
    elif kind == "individual":
        monkeypatch.setitem(m.LIMITS, "file_bytes_hard", 3)
    else:
        monkeypatch.setitem(m.LIMITS, "dump_bytes_sampled", 3)
    with pytest.raises(ValueError):
        m.inspect_tree(tmp_path)


@pytest.mark.parametrize("repeat,raises", [(False, False), (False, True), (True, False)])
def test_warmup_original_once_and_restore(tmp_path, repeat, raises):
    calls = []
    class Basic:
        def warmup(self):
            calls.append("original")
            if raises:
                raise RuntimeError("fixture")
            return "unchanged-object"
    original = Basic.warmup
    def enable(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"tensor_debug_mode": "FULL_TENSOR",
                          "circular_buffer_size": 4096,
                          "op_regex": m.OP_REGEX, "tensor_dtypes": ["float32"]}
        calls.append("enable")
        return object()
    tf = SimpleNamespace(debugging=SimpleNamespace(experimental=SimpleNamespace(
        enable_dump_debug_info=enable)))
    state = {"warmup_calls": 0}
    with contextlib.ExitStack() as stack:
        if raises or repeat:
            stack.enter_context(pytest.raises(RuntimeError))
        with m.warmup_observer(tf, Basic, tmp_path, state):
            assert Basic().warmup() == "unchanged-object"
            if repeat:
                Basic().warmup()
    assert Basic.warmup is original
    assert calls == ["enable", "original"]


class Reader:
    def __init__(self, values=None, traces=None):
        self.values = values if values is not None else [np.asarray([1.], np.float32)] * 2
        self.traces = traces if traces is not None else [
            SimpleNamespace(wall_time=1., op_type="MatMul", op_name="encoder/MatMul",
                            graph_id="g", output_slot=0),
            SimpleNamespace(wall_time=3., op_type="MatMul", op_name="encoder/MatMul",
                            graph_id="g", output_slot=0)]
        self.closed = False
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.closed = True
    def update(self):
        pass
    def graph_execution_traces(self, digest):
        assert digest is True
        return self.traces
    def graph_execution_trace_to_tensor_value(self, trace):
        return self.values[self.traces.index(trace)]
    def graph_by_id(self, graph_id):
        assert graph_id == "g"
        return SimpleNamespace(name="packed_fn", get_op_creation_digest=lambda name:
                               SimpleNamespace(input_names=("embedding:0", "weight:0")))


def test_supported_reader_records_graph_edges_and_values(tmp_path):
    r = Reader()
    result = m.read_dump(tmp_path, 2., lambda _: r)
    assert r.closed
    assert [v["phase"] for v in result["traces"]] == ["warmup", "observation"]
    assert result["float_elements"] == 2
    assert result["traces"][1]["input_names"] == ["embedding:0", "weight:0"]
    assert result["traces"][1]["values"] == [1.]


@pytest.mark.parametrize("kind", ["empty", "dtype", "nonfinite", "op", "count", "elements", "warmup-only"])
def test_reader_fail_closed(tmp_path, monkeypatch, kind):
    r = Reader()
    start = 2.
    if kind == "empty":
        r.traces = []
    elif kind == "dtype":
        r.values[0] = np.asarray([1.], np.float64)
    elif kind == "nonfinite":
        r.values[0] = np.asarray([np.inf], np.float32)
    elif kind == "op":
        r.traces[0].op_type = "ReadVariableOp"
    elif kind == "count":
        monkeypatch.setitem(m.LIMITS, "decoded_events", 1)
    elif kind == "elements":
        monkeypatch.setitem(m.LIMITS, "decoded_float_elements", 1)
    else:
        start = 4.
    with pytest.raises(ValueError):
        m.read_dump(tmp_path, start, lambda _: r)
    assert r.closed


def plan_fixture(tmp_path, monkeypatch):
    helper = tmp_path / m.HELPER
    helper.parent.mkdir()
    helper.write_text("frozen fixture helper")
    reference = tmp_path / "reference.json"
    native_plan = {"kind": "slippi", "key": "fox_d24_ditto_v4", "port": 2, "seed": 20260910}
    ref = {"passed": True, "environment": {"machine": "arm64"}, "rows": [leaf()] * 384,
           "checks": {"native": True}, "plan": native_plan, "runtime_contract": {"delay": 24}}
    reference.write_text(json.dumps(ref))
    monkeypatch.setattr(m, "REFERENCES", {"arm64": ("reference.json", m.digest(reference))})
    monkeypatch.setattr(m.platform, "machine", lambda: "arm64")
    native = SimpleNamespace(RECEIPT_BYTES=128 * 1024**2, validate_plan=lambda p, r: {})
    monkeypatch.setitem(sys.modules, "modal_panel_policy_canary", native)
    return m.make_plan(tmp_path, "arm64")


def test_plan_exact_source_bindings_and_no_changes(tmp_path, monkeypatch):
    p = plan_fixture(tmp_path, monkeypatch)
    m.verify_plan(p, tmp_path)
    (tmp_path / m.HELPER).write_text("changed")
    with pytest.raises(ValueError):
        m.verify_plan(p, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("real_observations", 2), ("warmup_calls", 2), ("tensor_dtypes", ["float64"]),
    ("op_regex", ".*"), ("tensor_debug_mode", "NO_TENSOR"), ("baseline", {}),
])
def test_plan_changes_rejected(tmp_path, monkeypatch, field, value):
    p = plan_fixture(tmp_path, monkeypatch)
    p[field] = value
    with pytest.raises(ValueError):
        m.verify_plan(p, tmp_path)


def test_plan_wrong_host_rejected(tmp_path, monkeypatch):
    p = plan_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError):
        m.verify_plan(p, tmp_path)


def test_reference_mutation_rejected(tmp_path, monkeypatch):
    plan_fixture(tmp_path, monkeypatch)
    with (tmp_path / "reference.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError):
        m.make_plan(tmp_path, "arm64")


class Process:
    pid = 321
    def __init__(self, running=False):
        self.running, self.returncode, self.terminated, self.killed = running, 0, False, False
    def poll(self):
        return None if self.running else self.returncode
    def terminate(self):
        self.terminated = True
    def wait(self, timeout=None):
        if timeout is not None and self.running:
            raise subprocess.TimeoutExpired("fixture", timeout)
        return self.returncode
    def kill(self):
        self.killed, self.running = True, False


@pytest.mark.parametrize("fail", [False, True])
def test_coordinator_one_attempt_partial_preservation(tmp_path, monkeypatch, fail):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    monkeypatch.setattr(m, "verify_plan", lambda *_: None)
    child = Process(running=fail)
    invocations = []
    def popen(args, **kwargs):
        invocations.append((args, kwargs))
        assert kwargs["start_new_session"] is True
        return child
    monkeypatch.setattr(m.subprocess, "Popen", popen)
    if fail:
        monkeypatch.setattr(m, "inspect_tree", lambda _: (_ for _ in ()).throw(ValueError("cap")))
        with pytest.raises(RuntimeError):
            m.run(plan_path, m.digest(plan_path))
        assert child.terminated and child.killed
    else:
        m.run(plan_path, m.digest(plan_path))
        assert not child.terminated and not child.killed
    assert (tmp_path / "worker.log").exists()
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert bool(execution["failure"]) is fail
    with pytest.raises(FileExistsError):
        m.run(plan_path, m.digest(plan_path))
    assert len(invocations) == 1


def test_wrong_approved_hash_prevents_launch(tmp_path, monkeypatch):
    p = tmp_path / "plan.json"
    p.write_text("{}")
    monkeypatch.setattr(m.subprocess, "Popen", lambda *_a, **_k: pytest.fail("unexpected launch"))
    with pytest.raises(ValueError):
        m.run(p, "0" * 64)
    assert not (tmp_path / "intent.json").exists()


def test_fixed_filter_omits_weight_reads_and_randomness():
    for name in ("ReadVariableOp", "Const", "RandomUniform", "Multinomial"):
        assert m.re.fullmatch(m.OP_REGEX, name) is None
    for name in ("MatMul", "AddV2", "Mean", "Sigmoid", "Tanh", "Erf", "ConcatV2"):
        assert m.re.fullmatch(m.OP_REGEX, name) is not None

