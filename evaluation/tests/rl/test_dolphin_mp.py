"""Worker-process Dolphin stepping (P7-PLAN.md 2): wire codec, worker loop, parent env, dispatch.

The in-process ``DolphinEnv`` stays the default and behaviourally untouched
(``tests/rl/test_dolphin_env.py``); these tests cover the additive worker mode: the config knobs, the
numpy wire codec, the behaviour-preserving ``step_rows`` refactor, the parent's slice / assemble
logic, the worker main loop driven inline over a real pipe (no processes), and two real
multiprocessing tests.  The mp tests use ``forkserver`` with module preload: the fork server pays the
~20 s torch import once per test process on the slow local mount (seconds on Modal), and every worker
forks from it instantly; the ``spawn`` default is exercised by the Modal canary run (P7-PLAN.md 2.4).
"""

from __future__ import annotations

import multiprocessing
import threading
from typing import Any

import numpy as np
import pytest
import torch

melee = pytest.importorskip("melee")  # the ``dolphin`` extra (melee==0.47.3)

from controller_codec import ButtonState, ControllerState, StickState  # noqa: E402
from melee_rl.config import (  # noqa: E402
    EnvConfig,
    EvalConfig,
    RLConfig,
    TeacherConfig,
    finalize_config,
    resolve_model_config,
)
from melee_rl.env import dolphin_mp  # noqa: E402
from melee_rl.env.async_env import AsyncEnvProtocol  # noqa: E402
from melee_rl.env.dolphin import (  # noqa: E402
    MP_START_METHODS,
    DolphinEnv,
    DolphinEnvConfig,
    build_env_output,
    controller_rows,
)
from melee_rl.env.dolphin_mp import (  # noqa: E402
    EnvWorkerError,
    WorkerDolphinEnv,
    assemble_states,
    build_dolphin_env,
    controller_arrays,
    rows_from_arrays,
    slice_arrays,
    worker_configs,
)
from melee_rl.env.protocol import INITIAL_FRAME_INDEX, EnvOutput, EnvProtocol  # noqa: E402
from melee_rl.frames import tree_leaves  # noqa: E402
from melee_rl.learner import LearnerConfig  # noqa: E402
from tensor_batch import BUTTON_ORDER  # noqa: E402
from tests.rl import _mp_backend  # noqa: E402

START = INITIAL_FRAME_INDEX


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _control(
    batch: int,
    *,
    main: tuple[float, float] = (0.5, 0.5),
    c: tuple[float, float] = (0.5, 0.5),
    shoulder: float = 0.0,
    pressed: tuple[str, ...] = (),
) -> ControllerState:
    def full(value: float) -> torch.Tensor:
        return torch.full((batch,), value, dtype=torch.float32)

    return ControllerState(
        main_stick=StickState(x=full(main[0]), y=full(main[1])),
        c_stick=StickState(x=full(c[0]), y=full(c[1])),
        shoulder=full(shoulder),
        buttons=ButtonState(**{name: torch.full((batch,), name in pressed) for name in BUTTON_ORDER}),
    )


def _plain_env(
    scripts: dict[int, list[_mp_backend.Script]], **overrides: Any
) -> tuple[DolphinEnv, _mp_backend.FakeBackendMP]:
    backend = _mp_backend.FakeBackendMP(scripts)
    settings: dict[str, Any] = {"num_envs": len(scripts), "check_iso": False}
    settings.update(overrides)
    return DolphinEnv(DolphinEnvConfig(**settings), cpu_level=9, backend=backend), backend


def _assert_outputs_equal(a: EnvOutput, b: EnvOutput) -> None:
    assert a.frame_index.tolist() == b.frame_index.tolist()
    assert a.needs_reset.tolist() == b.needs_reset.tolist()
    assert tuple(a.frames) == tuple(b.frames) and tuple(a.rewards) == tuple(b.rewards)
    for port in a.frames:
        pairs = zip(tree_leaves(a.frames[port]), tree_leaves(b.frames[port]), strict=True)
        for (path, leaf), (other_path, other) in pairs:
            assert path == other_path
            assert torch.equal(leaf, other), (port, path)
        assert torch.equal(a.rewards[port], b.rewards[port])


def _recv(conn: Any, timeout: float = 60.0) -> Any:
    assert conn.poll(timeout), "no message from the worker loop"
    return conn.recv()


# ---------------------------------------------------------------------------
# 1-2: config knobs and the wire codec (no environments)
# ---------------------------------------------------------------------------


def test_worker_config_knobs_validation_and_slicing() -> None:
    config = DolphinEnvConfig()
    assert config.worker_processes == 0 and config.mp_start_method == "spawn"
    assert config.envs_per_worker == config.num_envs == 1
    assert MP_START_METHODS == ("spawn", "forkserver")
    grouped = DolphinEnvConfig(num_envs=8, worker_processes=4, slippi_port=51500)
    assert grouped.envs_per_worker == 2
    with pytest.raises(ValueError, match="worker_processes"):
        DolphinEnvConfig(worker_processes=-1)
    with pytest.raises(ValueError, match="worker_processes"):
        DolphinEnvConfig(num_envs=8, worker_processes=3)
    with pytest.raises(ValueError, match="keepalive"):
        DolphinEnvConfig(num_envs=4, worker_processes=2, keepalive_seconds=1.0)
    with pytest.raises(ValueError, match="mp_start_method"):
        DolphinEnvConfig(mp_start_method="fork")
    sliced = worker_configs(grouped)
    assert [c.num_envs for c in sliced] == [2, 2, 2, 2]
    assert [c.slippi_port for c in sliced] == [51500, 51502, 51504, 51506]
    assert all(c.worker_processes == 0 for c in sliced)
    assert all(c.players == grouped.players and c.boot == grouped.boot for c in sliced)
    with pytest.raises(ValueError, match="worker_processes"):
        worker_configs(config)  # 0 workers: the in-process path, nothing to slice


def test_controller_array_codec_round_trips_rows() -> None:
    control = _control(4, main=(0.1, 0.9), c=(0.3, 0.7), shoulder=1.0, pressed=("X", "D_UP"))
    arrays = controller_arrays(control, 4)
    assert set(arrays) == {"main_x", "main_y", "c_x", "c_y", "shoulder", *BUTTON_ORDER}
    assert arrays["main_x"].dtype == np.float32 and arrays["shoulder"].dtype == np.float32
    assert arrays["A"].dtype == np.bool_ and arrays["D_UP"].dtype == np.bool_
    assert all(array.shape == (4,) for array in arrays.values())
    assert rows_from_arrays(arrays, 4) == controller_rows(control, 4)
    part = slice_arrays(arrays, 2, 4)
    assert all(array.shape == (2,) for array in part.values())
    assert rows_from_arrays(part, 2) == controller_rows(control, 4)[2:4]
    with pytest.raises(ValueError, match="batch"):
        controller_arrays(control, 3)
    with pytest.raises(ValueError, match="batch"):
        rows_from_arrays(arrays, 3)


# ---------------------------------------------------------------------------
# 3-4: the step_rows refactor and the parent assembly (in-process, no mp)
# ---------------------------------------------------------------------------


def test_step_rows_refactor_keeps_dolphin_env_outputs_identical() -> None:
    scripts = {0: [_mp_backend.Script(_mp_backend._boot())], 1: [_mp_backend.Script(_mp_backend._boot())]}
    twin = {0: [_mp_backend.Script(_mp_backend._boot())], 1: [_mp_backend.Script(_mp_backend._boot())]}
    env_a, _ = _plain_env(scripts)
    env_b, _ = _plain_env(twin)
    try:
        control = _control(2, main=(0.25, 0.75), pressed=("A",))
        a_out = env_a.step({0: control})
        batch, frame_index = env_b.step_rows({0: controller_rows(control, 2)})
        b_out = build_env_output(batch, frame_index, env_b.controlled_ports)
        _assert_outputs_equal(a_out, b_out)
        # step_rows keeps the pending state in sync (current() reflects the last frame).
        _assert_outputs_equal(env_b.current(), b_out)
        # collect_rows reproduces the pending state from the instances (what the worker's replies use).
        again = build_env_output(*env_b.collect_rows(), env_b.controlled_ports)
        _assert_outputs_equal(env_b.current(), again)
        with pytest.raises(ValueError, match="needs controllers for ports"):
            env_b.step_rows({1: controller_rows(control, 2)})
    finally:
        env_a.close()
        env_b.close()
    with pytest.raises(RuntimeError, match="closed"):
        env_b.step_rows({0: controller_rows(control, 2)})


def test_parent_assembly_matches_a_single_wide_env() -> None:
    # The wide reference: 4 envs in one process, carrying the projectile ids two workers would produce.
    wide_ids = [10, 11, 20, 21]
    wide_scripts = {
        i: [_mp_backend.Script(_mp_backend._boot(spawn_id=spawn))] for i, spawn in enumerate(wide_ids)
    }
    wide, _ = _plain_env(wide_scripts, players=("policy", "policy"), slippi_port=51520)
    # The sliced world: worker_configs of the same config, one in-process env per slice.
    grouped = DolphinEnvConfig(
        num_envs=4,
        worker_processes=2,
        players=("policy", "policy"),
        slippi_port=51520,
        check_iso=False,
    )
    parts = [
        DolphinEnv(config, cpu_level=9, backend=_mp_backend.numbered_backend(config, w))
        for w, config in enumerate(worker_configs(grouped))
    ]
    try:
        boot_states = []
        for part in parts:
            batch, frame_index = part.collect_rows()
            boot_states.append((batch.leaves, frame_index))
        boot = assemble_states(boot_states, grouped.controlled_ports)
        boot.validate(4)
        _assert_outputs_equal(wide.current(), boot)
        left = _control(4, main=(0.0, 0.5))
        right = _control(4, main=(1.0, 0.5), pressed=("Y",))
        wide_out = wide.step({0: left, 1: right})
        arrays = {0: controller_arrays(left, 4), 1: controller_arrays(right, 4)}
        states = []
        for w, part in enumerate(parts):
            rows = {
                port: rows_from_arrays(slice_arrays(arrays[port], w * 2, (w + 1) * 2), 2) for port in (0, 1)
            }
            batch, frame_index = part.step_rows(rows)
            states.append((batch.leaves, frame_index))
        merged = assemble_states(states, grouped.controlled_ports)
        merged.validate(4)
        _assert_outputs_equal(wide_out, merged)
        assert merged.frames[0]["items"]["x"][:, 0].tolist() == [10.0, 11.0, 20.0, 21.0]
        # build_env_output invariants survive the assembly: perspective swap and rewards share storage.
        pairs = zip(tree_leaves(merged.frames[1]["p0"]), tree_leaves(merged.frames[0]["p1"]), strict=True)
        for (path, leaf), (other_path, other) in pairs:
            assert path == other_path and leaf is other
        assert merged.rewards[0] is merged.rewards[1]
    finally:
        wide.close()
        for part in parts:
            part.close()


# ---------------------------------------------------------------------------
# 5-6: the worker loop, driven inline over a real pipe (no processes)
# ---------------------------------------------------------------------------


def _inline_worker(
    config: DolphinEnvConfig, factory: Any
) -> tuple[Any, threading.Thread, list[_mp_backend.FakeBackendMP]]:
    made: list[_mp_backend.FakeBackendMP] = []

    def recording(cfg: DolphinEnvConfig, worker_index: int) -> _mp_backend.FakeBackendMP:
        backend = factory(cfg, worker_index)
        made.append(backend)
        return backend

    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(
        target=dolphin_mp._worker_main, args=(child, config, 9, 0, recording), daemon=True
    )
    thread.start()
    return parent, thread, made


def test_worker_loop_inline_serves_ready_step_stats_reset_and_shutdown() -> None:
    config = DolphinEnvConfig(num_envs=2, check_iso=False, slippi_port=51540)
    parent, thread, made = _inline_worker(config, _mp_backend.plain_backend)
    tag, payload = _recv(parent)
    assert tag == "ready"
    leaves, frame_index, version = payload
    assert version == dolphin_mp._PROTOCOL_VERSION
    assert frame_index == [START, START]
    assert leaves["stage"].tolist() == [25, 25]
    assert isinstance(leaves["p0.x"], np.ndarray) and leaves["p0.x"].shape == (2,)
    parent.send(("step", {0: controller_arrays(_control(2, pressed=("A",)), 2)}))
    tag, (stepped, indices) = _recv(parent)
    assert tag == "state" and indices == [START + 1, START + 1]
    assert stepped["p0.controller.buttons.A"].tolist() == [True, True]
    assert stepped["p1.controller.buttons.A"].tolist() == [False, False]
    parent.send(("stats", None))
    tag, rows = _recv(parent)
    assert tag == "stats" and [row["index"] for row in rows] == [0, 1]
    assert [row["launches"] for row in rows] == [1, 1]
    parent.send(("reset", None))
    tag, (after_reset, reset_indices) = _recv(parent)
    assert tag == "state" and reset_indices == [START, START]
    assert after_reset["p0.x"].tolist() == [-20.0 + START] * 2
    parent.send(None)
    assert _recv(parent) is None
    thread.join(30.0)
    assert not thread.is_alive()
    assert all(console.stopped >= 1 for console in made[0].consoles)


def test_worker_loop_inline_reports_launch_failure_as_error_then_none() -> None:
    config = DolphinEnvConfig(num_envs=1, check_iso=False, launch_retries=0, slippi_port=51550)
    parent, thread, made = _inline_worker(config, _mp_backend.failing_backend)
    tag, (name, message, trace) = _recv(parent)
    assert tag == "error" and name == "DolphinLaunchError"
    assert "1 launch attempt" in message and "DolphinLaunchError" in trace
    assert _recv(parent) is None
    thread.join(30.0)
    assert not thread.is_alive()
    assert all(console.stopped >= 1 for console in made[0].consoles)
    # An unknown tag on a healthy loop takes the same envelope path and stops the worker.
    parent, thread, made = _inline_worker(
        DolphinEnvConfig(num_envs=1, check_iso=False, slippi_port=51551), _mp_backend.plain_backend
    )
    tag, _ = _recv(parent)
    assert tag == "ready"
    parent.send(("bogus", None))
    tag, (name, message, trace) = _recv(parent)
    assert tag == "error" and name == "RuntimeError" and "bogus" in message
    assert _recv(parent) is None
    thread.join(30.0)
    assert not thread.is_alive()
    assert all(console.stopped >= 1 for console in made[0].consoles)


# ---------------------------------------------------------------------------
# 7-8: real worker processes (forkserver + preload; ~20 s import once per test process locally)
# ---------------------------------------------------------------------------


def test_worker_env_real_processes_step_fault_relaunch_stats_and_close() -> None:
    config = DolphinEnvConfig(
        num_envs=4,
        worker_processes=2,
        mp_start_method="forkserver",
        console_timeout_s=0.05,
        launch_retries=1,
        slippi_port=51600,
        check_iso=False,
    )
    env = WorkerDolphinEnv(config, cpu_level=9, backend_factory=_mp_backend.flaky_backend)
    try:
        assert isinstance(env, EnvProtocol)
        assert env.num_envs == 4 and env.controlled_ports == (0,)
        assert env.worker_count == 2 and env.envs_per_worker == 2
        boot = env.current()
        boot.validate(4)
        assert boot.frame_index.tolist() == [START] * 4
        neutral = _control(4)
        for step in (1, 2):
            out = env.step({0: neutral})
            out.validate(4)
            assert out.frame_index.tolist() == [START + step] * 4
        # Worker 1's local env 0 (= global env 2) times out in game: the worker relaunches it in place.
        out = env.step({0: neutral})
        out.validate(4)
        assert out.frame_index.tolist() == [START + 3, START + 3, START, START + 3]
        assert out.needs_reset.tolist() == [False, False, True, False]
        stats = env.stats()
        assert [row["index"] for row in stats] == [0, 1, 2, 3]
        assert [row["worker"] for row in stats] == [0, 0, 1, 1]
        assert [row["relaunches"] for row in stats] == [0, 0, 1, 0]
        assert [row["slippi_port"] for row in stats] == [51600, 51601, 51602, 51603]
        assert "ConsoleTimeout" in stats[2]["faults"][0]
        after = env.step({0: neutral})
        assert after.frame_index.tolist() == [START + 4, START + 4, START + 1, START + 4]
        with pytest.raises(ValueError, match="needs controllers for ports"):
            env.step({0: neutral, 1: neutral})
    finally:
        env.close()
    assert env.closed
    env.close()  # idempotent
    assert [process.exitcode for process in env._procs] == [0, 0]
    with pytest.raises(RuntimeError, match="closed"):
        env.step({0: _control(4)})
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_worker_env_boot_failure_raises_env_worker_error_and_reaps_the_process() -> None:
    config = DolphinEnvConfig(
        num_envs=1,
        worker_processes=1,
        mp_start_method="forkserver",
        launch_retries=0,
        slippi_port=51700,
        check_iso=False,
    )
    with pytest.raises(EnvWorkerError, match="DolphinLaunchError") as caught:
        WorkerDolphinEnv(config, cpu_level=9, backend_factory=_mp_backend.failing_backend)
    assert "1 launch attempt" in str(caught.value)
    leftovers = [
        process
        for process in multiprocessing.active_children()
        if process.name.startswith("melee-rl-env-worker")
    ]
    assert leftovers == []


# ---------------------------------------------------------------------------
# 9: dispatch and the load-time eval divisibility rule
# ---------------------------------------------------------------------------


def test_build_dolphin_env_selects_worker_mode_and_eval_divisibility_is_checked_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inline = DolphinEnvConfig(num_envs=1, check_iso=False, slippi_port=51560)
    env = build_dolphin_env(inline, cpu_level=9, backend=_mp_backend.plain_backend(inline, 0))
    try:
        assert isinstance(env, DolphinEnv)
    finally:
        env.close()

    calls: dict[str, Any] = {}

    class Stub:
        def __init__(self, config: DolphinEnvConfig, *, cpu_level: int, backend_factory: Any = None) -> None:
            calls.update(config=config, cpu_level=cpu_level, backend_factory=backend_factory)

    monkeypatch.setattr(dolphin_mp, "WorkerDolphinEnv", Stub)
    grouped = DolphinEnvConfig(num_envs=4, worker_processes=2, check_iso=False)
    made = build_dolphin_env(grouped, cpu_level=3, backend_factory=_mp_backend.plain_backend)
    assert isinstance(made, Stub)
    assert calls == {
        "config": grouped,
        "cpu_level": 3,
        "backend_factory": _mp_backend.plain_backend,
    }
    with pytest.raises(ValueError, match="backend_factory"):
        build_dolphin_env(grouped, cpu_level=9, backend=_mp_backend.plain_backend(inline, 0))
    with pytest.raises(ValueError, match="backend_factory"):
        build_dolphin_env(inline, cpu_level=9, backend_factory=_mp_backend.plain_backend)

    def _rl_config(eval_envs: int) -> RLConfig:
        return RLConfig(
            env=EnvConfig(
                type="dolphin",
                dolphin=DolphinEnvConfig(num_envs=8, worker_processes=4, check_iso=False),
            ),
            teacher=TeacherConfig(source="none"),
            learner=LearnerConfig(kl_teacher_weight=0.0),
            eval=EvalConfig(enabled=True, num_envs=eval_envs),
        )

    good = _rl_config(8)
    model = resolve_model_config(good)
    assert finalize_config(good, model).eval.num_envs == 8
    with pytest.raises(ValueError, match="worker_processes"):
        finalize_config(_rl_config(6), model)


# ---------------------------------------------------------------------------
# 10-12: the P8 chunked wire (protocol v2) and the native async surface
# ---------------------------------------------------------------------------


def test_chunk_wire_inline_step_chunk_and_deadline() -> None:
    """One ``step_chunk`` message steps k frames and replies with k states, in order."""

    config = DolphinEnvConfig(num_envs=2, check_iso=False, slippi_port=51560, chunk_frames=2)
    parent, thread, _made = _inline_worker(config, _mp_backend.plain_backend)
    tag, payload = _recv(parent)
    assert tag == "ready" and payload[2] == dolphin_mp._PROTOCOL_VERSION == 3
    chunk = [
        {0: controller_arrays(_control(2, pressed=("A",)), 2)},
        {0: controller_arrays(_control(2), 2)},
    ]
    parent.send(("step_chunk", chunk))
    tag, states = _recv(parent)
    assert tag == "states" and len(states) == 2
    (first_leaves, first_indices), (second_leaves, second_indices) = states
    assert first_indices == [START + 1, START + 1] and second_indices == [START + 2, START + 2]
    assert first_leaves["p0.controller.buttons.A"].tolist() == [True, True]
    assert second_leaves["p0.controller.buttons.A"].tolist() == [False, False]
    parent.send(None)
    assert _recv(parent) is None
    thread.join(30.0)
    assert not thread.is_alive()

    deadline = dolphin_mp._chunk_deadline_s(
        DolphinEnvConfig(chunk_frames=3, console_timeout_s=30.0, boot_timeout_s=120.0, launch_retries=2)
    )
    assert deadline == 30.0 * 3 + 120.0 * 3 + 60.0
    single = DolphinEnvConfig(console_timeout_s=30.0, boot_timeout_s=120.0, launch_retries=2)
    assert dolphin_mp._chunk_deadline_s(single) == dolphin_mp._step_deadline_s(single)


def test_chunk_wire_inline_mid_chunk_hard_failure_reports_error() -> None:
    """A hard fault inside a chunk takes the same error envelope as a single step (fail-fast)."""

    config = DolphinEnvConfig(
        num_envs=1,
        check_iso=False,
        launch_retries=0,
        console_timeout_s=0.05,
        slippi_port=51561,
        chunk_frames=3,
    )

    def half_dead(cfg: DolphinEnvConfig, worker_index: int) -> _mp_backend.FakeBackendMP:
        # 2 game frames then a timeout; the relaunch fails to connect -> DolphinLaunchError mid-chunk.
        scripts = [
            _mp_backend.Script([*_mp_backend._boot(game_frames=2), None]),
            _mp_backend.Script([], connect_ok=False),
        ]
        return _mp_backend.FakeBackendMP({0: scripts})

    parent, thread, _made = _inline_worker(config, half_dead)
    tag, _ = _recv(parent)
    assert tag == "ready"
    chunk = [{0: controller_arrays(_control(1), 1)} for _ in range(3)]
    parent.send(("step_chunk", chunk))
    tag, (name, message, trace) = _recv(parent)
    assert tag == "error" and name == "DolphinLaunchError"
    assert "1 launch attempt" in message and "DolphinLaunchError" in trace
    assert _recv(parent) is None
    thread.join(30.0)
    assert not thread.is_alive()


def test_worker_env_native_async_push_pop_peek_reset_and_sync_parity() -> None:
    """The native push/pop surface: chunked sends, FIFO pops across workers, the seed, the guards."""

    config = DolphinEnvConfig(
        num_envs=2,
        worker_processes=2,
        mp_start_method="forkserver",
        chunk_frames=2,
        slippi_port=51800,
        check_iso=False,
    )
    env = WorkerDolphinEnv(config, cpu_level=9, backend_factory=_mp_backend.plain_backend)
    twin, _ = _plain_env(
        {0: [_mp_backend.Script(_mp_backend._boot())], 1: [_mp_backend.Script(_mp_backend._boot())]},
        slippi_port=51820,
    )
    try:
        assert isinstance(env, AsyncEnvProtocol) and isinstance(env, EnvProtocol)
        assert env.chunk_frames == 2 and env.in_flight == 1
        boot = env.pop()
        _assert_outputs_equal(boot, twin.current())
        assert env.in_flight == 0 and env.current() is boot
        actions = [
            _control(2, main=(0.25, 0.5)),
            _control(2, pressed=("A",)),
            _control(2, main=(0.9, 0.1)),
        ]
        env.push({0: actions[0]})
        assert env.in_flight == 1  # buffered, nothing sent yet
        env.push({0: actions[1]})
        assert env.in_flight == 2  # one chunk of two frames in transit
        expected = [twin.step({0: action}) for action in actions[:2]]
        peeked = env.peek()
        _assert_outputs_equal(peeked, expected[0])
        first = env.pop()
        assert first is peeked
        second = env.pop()
        _assert_outputs_equal(second, expected[1])
        assert env.in_flight == 0
        with pytest.raises(RuntimeError, match="no state in flight"):
            env.pop()
        env.push({0: actions[2]})
        with pytest.raises(RuntimeError, match="1 of 2"):
            env.pop()
        with pytest.raises(RuntimeError, match="asynchronous"):
            env.step({0: actions[2]})
        after_reset = env.reset()  # drops the buffered action, reseeds the boot state
        assert env.in_flight == 1 and after_reset.frame_index.tolist() == [START, START]
        assert env.pop() is after_reset
        env.push({0: actions[0]})
        env.push({0: actions[1]})
        assert env.in_flight == 2
        _assert_outputs_equal(env.pop(), env.current())
    finally:
        env.close()
        twin.close()
    assert [process.exitcode for process in env._procs] == [0, 0]
