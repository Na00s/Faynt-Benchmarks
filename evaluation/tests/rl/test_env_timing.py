"""The frame-step timing decomposition (5 Sep 2026, ``/rl-dolphin-speedup`` lever 0).

``DolphinEnv.timing()`` counts steps, the fan-out, the per-Dolphin advance times (and their histogram);
the worker process counts its idle / unpack / step / reply time and answers a ``timing`` message; the
parent counts sends, the wait until the first and the last reply, the receives and the assembly; the
tracker turns cumulative snapshots into per-rollout metrics.  Fake consoles throughout: the numbers
are only checked for shape, monotonicity and arithmetic, never for absolute values.
"""

from __future__ import annotations

import multiprocessing
import threading
from typing import Any

import pytest
import torch

melee = pytest.importorskip("melee")  # the ``dolphin`` extra (melee==0.47.3)

from controller_codec import ControllerState, CustomV1Codec  # noqa: E402
from melee_rl.env import dolphin_mp  # noqa: E402
from melee_rl.env.dolphin import DolphinEnv, DolphinEnvConfig  # noqa: E402
from melee_rl.env.dolphin_mp import WorkerDolphinEnv  # noqa: E402
from melee_rl.env.timing import (  # noqa: E402
    ADVANCE_BINS_MS,
    BIN_LABELS,
    EnvTimingTracker,
    bin_index,
    new_env_timing,
    new_parent_timing,
    new_worker_timing,
    subtract,
    summarize,
    summarize_env,
)
from melee_rl.frames import neutral_labels  # noqa: E402
from tests.rl import _mp_backend  # noqa: E402
from tests.rl._mp_backend import Script, _boot  # noqa: E402


def _neutral(batch: int) -> ControllerState:
    return CustomV1Codec().decode(neutral_labels((batch,)))


def _recv(conn: Any, timeout: float = 60.0) -> Any:
    assert conn.poll(timeout), "no reply from the worker"
    return conn.recv()


# ---------------------------------------------------------------------------
# the counters and their arithmetic
# ---------------------------------------------------------------------------


def test_bins_and_subtract_and_summaries() -> None:
    assert len(BIN_LABELS) == len(ADVANCE_BINS_MS) + 1
    assert bin_index(0.0) == 0 and bin_index(0.0019) == 0 and bin_index(0.002) == 0
    assert bin_index(0.0021) == 1 and bin_index(0.5) == len(ADVANCE_BINS_MS)
    before = new_env_timing()
    now = new_env_timing()
    now.update(
        steps=2,
        wall_s=0.10,
        fanout_s=0.08,
        advance_s=0.12,
        advance_max_s=0.07,
        convert_s=0.004,
        deferred_s=0.001,
        output_s=0.002,
        instance_frames=4,
    )
    now["advance_hist"][3] = 3
    now["advance_hist"][-1] = 1
    delta = subtract(now, before)
    assert delta == now
    summary = summarize_env(delta)
    assert summary["env/steps"] == 2.0
    assert summary["env/step_ms"] == pytest.approx(50.0)
    assert summary["env/fanout_ms"] == pytest.approx(40.0)
    assert summary["env/advance_mean_ms"] == pytest.approx(30.0)
    assert summary["env/advance_max_ms"] == pytest.approx(35.0)
    assert summary["env/tail_ms"] == pytest.approx(10.0)
    assert summary["env/convert_ms"] == pytest.approx(1.0)
    assert summary["env/advance_share/le8ms"] == pytest.approx(0.75)
    assert summary["env/advance_share/gt100ms"] == pytest.approx(0.25)
    assert sum(value for key, value in summary.items() if "/advance_share/" in key) == pytest.approx(1.0)
    # An empty delta summarises to zeros, never a division error.
    assert all(value == 0.0 for value in summarize_env(subtract(before, before)).values())

    parent = new_parent_timing(2)
    parent.update(steps=4, send_s=0.004, wait_first_s=0.02, wait_last_s=0.04, recv_s=0.008, assemble_s=0.002)
    parent["arrival_s"] = [0.02, 0.04]
    worker_a = new_worker_timing()
    worker_a.update(messages=4, recv_wait_s=0.1, unpack_s=0.002, step_s=0.06, reply_s=0.004)
    worker_b = dict(worker_a, step_s=0.1)
    env_a = new_env_timing()
    env_a.update(steps=4, wall_s=0.06, fanout_s=0.05, advance_s=0.08, advance_max_s=0.045, instance_frames=8)
    env_a["advance_hist"][2] = 8
    env_b = dict(env_a, wall_s=0.1, fanout_s=0.09, advance_s=0.16, advance_max_s=0.085)
    env_b["advance_hist"] = list(env_a["advance_hist"])
    combined = summarize(
        {
            "parent": parent,
            "workers": [{"worker": worker_a, "env": env_a}, {"worker": worker_b, "env": env_b}],
        }
    )
    assert combined["parent/steps"] == 4.0
    assert combined["parent/wait_first_ms"] == pytest.approx(5.0)
    assert combined["parent/wait_last_ms"] == pytest.approx(10.0)
    assert combined["parent/arrival_min_ms"] == pytest.approx(5.0)
    assert combined["parent/arrival_max_ms"] == pytest.approx(10.0)
    assert combined["worker/step_ms"] == pytest.approx(20.0)  # the mean over the two workers
    assert combined["worker/step_ms_max"] == pytest.approx(25.0)
    assert combined["worker/recv_wait_ms"] == pytest.approx(25.0)
    assert combined["env/steps"] == 4.0
    assert combined["env/step_ms"] == pytest.approx(20.0)  # (0.06 + 0.1) / 8 worker-steps
    assert combined["env/advance_mean_ms"] == pytest.approx(15.0)  # 0.24 s / 16 Dolphin frames
    assert combined["env/advance_max_ms"] == pytest.approx(16.25)
    assert combined["env/advance_share/le6ms"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the in-process environment
# ---------------------------------------------------------------------------


def test_dolphin_env_timing_counts_steps_dolphin_frames_and_bins() -> None:
    backend = _mp_backend.FakeBackendMP({i: [Script(_boot())] for i in range(2)})
    config = DolphinEnvConfig(num_envs=2, check_iso=False, slippi_port=51700)
    env = DolphinEnv(config, cpu_level=9, backend=backend)
    try:
        zero = env.timing()
        assert zero == new_env_timing()
        for _ in range(3):
            env.step({0: _neutral(2)})
        timing = env.timing()
        assert timing["steps"] == 3 and timing["instance_frames"] == 6
        assert sum(timing["advance_hist"]) == 6 and len(timing["advance_hist"]) == len(BIN_LABELS)
        assert timing["wall_s"] >= timing["fanout_s"] > 0.0
        assert timing["advance_s"] > 0.0 and timing["convert_s"] > 0.0
        # The per-step maximum is at least the per-step mean over the two Dolphins.
        assert timing["advance_max_s"] >= timing["advance_s"] / 2 - 1e-9
        assert timing["output_s"] > 0.0 and timing["deferred_s"] >= 0.0
        # A copy, not the live dict: stepping again does not mutate what was returned.
        env.step({0: _neutral(2)})
        assert timing["steps"] == 3 and env.timing()["steps"] == 4
        summary = summarize(subtract(env.timing(), zero))
        assert summary["env/steps"] == 4.0 and summary["env/step_ms"] > 0.0
        assert sum(value for key, value in summary.items() if "/advance_share/" in key) == pytest.approx(1.0)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# the worker process and the parent
# ---------------------------------------------------------------------------


def test_worker_loop_answers_the_timing_message() -> None:
    config = DolphinEnvConfig(num_envs=2, check_iso=False, slippi_port=51710)
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(
        target=dolphin_mp._worker_main, args=(child, config, 9, 0, _mp_backend.plain_backend), daemon=True
    )
    thread.start()
    try:
        tag, payload = _recv(parent)
        assert tag == "ready" and payload[2] == dolphin_mp._PROTOCOL_VERSION == 3
        arrays = dolphin_mp.controller_arrays(_neutral(2), 2)
        for _ in range(2):
            parent.send(("step", {0: arrays}))
            tag, _state = _recv(parent)
            assert tag == "state"
        parent.send(("timing", None))
        tag, timing = _recv(parent)
        assert tag == "timing" and set(timing) == {"worker", "env"}
        worker = timing["worker"]
        assert worker["messages"] == 2 and worker["recv_wait_s"] > 0.0 and worker["step_s"] > 0.0
        assert worker["unpack_s"] >= 0.0 and worker["reply_s"] >= 0.0
        assert timing["env"]["steps"] == 2 and timing["env"]["instance_frames"] == 4
        # stats and timing requests are not counted as served steps.
        parent.send(("stats", None))
        assert _recv(parent)[0] == "stats"
        parent.send(("timing", None))
        assert _recv(parent)[1]["worker"]["messages"] == 2
    finally:
        parent.send(None)
        assert _recv(parent) is None
        thread.join(30.0)


def test_worker_env_timing_round_trip_and_summary() -> None:
    config = DolphinEnvConfig(
        num_envs=4,
        worker_processes=2,
        mp_start_method="forkserver",
        slippi_port=51720,
        check_iso=False,
    )
    env = WorkerDolphinEnv(config, cpu_level=9, backend_factory=_mp_backend.plain_backend)
    try:
        first = env.timing()
        assert first["parent"] == new_parent_timing(2)
        assert [row["worker"]["messages"] for row in first["workers"]] == [0, 0]
        tracker = EnvTimingTracker()
        assert tracker.update([("arm", env)]) == {}  # the baseline call
        for _ in range(3):
            env.step({0: _neutral(4)})
        timing = env.timing()
        parent = timing["parent"]
        assert parent["steps"] == 3 and len(parent["arrival_s"]) == 2
        assert parent["wait_last_s"] >= parent["wait_first_s"] > 0.0
        assert parent["send_s"] > 0.0 and parent["recv_s"] > 0.0 and parent["assemble_s"] > 0.0
        assert all(0.0 < value <= parent["wait_last_s"] + 1e-9 for value in parent["arrival_s"])
        assert [row["worker"]["messages"] for row in timing["workers"]] == [3, 3]
        assert [row["env"]["steps"] for row in timing["workers"]] == [3, 3]
        assert [row["env"]["instance_frames"] for row in timing["workers"]] == [6, 6]
        metrics = tracker.update([("arm", env)])
        assert metrics["env_timing/arm/parent/steps"] == 3.0
        assert metrics["env_timing/arm/worker/messages"] == 3.0
        assert metrics["env_timing/arm/env/steps"] == 3.0
        assert metrics["env_timing/arm/env/step_ms"] > 0.0
        parent_metrics = {key.rsplit("/", 1)[1]: value for key, value in metrics.items() if "/parent/" in key}
        assert parent_metrics["wait_last_ms"] >= parent_metrics["wait_first_ms"]
        assert parent_metrics["arrival_max_ms"] >= parent_metrics["arrival_min_ms"]
        # The next delta covers only the steps since the previous update.
        env.step({0: _neutral(4)})
        again = tracker.update([("arm", env)])
        assert again["env_timing/arm/parent/steps"] == 1.0 and again["env_timing/arm/env/steps"] == 1.0
        # The chunked (P8) path counts its frames too: k = 2 pushes make one round trip of two states.
        env.reset()
        chunked = WorkerDolphinEnv(
            DolphinEnvConfig(
                num_envs=4,
                worker_processes=2,
                mp_start_method="forkserver",
                slippi_port=51730,
                check_iso=False,
                chunk_frames=2,
            ),
            cpu_level=9,
            backend_factory=_mp_backend.plain_backend,
        )
        try:
            chunked.pop()  # the boot state
            chunked.push({0: _neutral(4)})
            chunked.push({0: _neutral(4)})
            chunked.pop()
            chunked.pop()
            chunk_timing = chunked.timing()
            assert chunk_timing["parent"]["steps"] == 1  # one round trip
            assert [row["worker"]["messages"] for row in chunk_timing["workers"]] == [1, 1]
            assert [row["env"]["steps"] for row in chunk_timing["workers"]] == [2, 2]
        finally:
            chunked.close()
    finally:
        env.close()


def test_tracker_skips_environments_without_timing() -> None:
    class Plain:
        pass

    class Timed:
        def __init__(self) -> None:
            self.calls = 0

        def timing(self) -> dict[str, Any]:
            self.calls += 1
            env = new_env_timing()
            env.update(steps=self.calls, wall_s=0.01 * self.calls, instance_frames=2 * self.calls)
            env["advance_hist"][0] = 2 * self.calls
            return env

    tracker = EnvTimingTracker()
    timed = Timed()
    assert tracker.update([("a", Plain()), ("b", timed)]) == {}
    metrics = tracker.update([("a", Plain()), ("b", timed)])
    assert set(key.split("/")[1] for key in metrics) == {"b"}
    assert metrics["env_timing/b/env/steps"] == 1.0
    assert metrics["env_timing/b/env/step_ms"] == pytest.approx(10.0)
    assert isinstance(metrics["env_timing/b/env/advance_share/le2ms"], float)
    assert torch.tensor(metrics["env_timing/b/env/advance_share/le2ms"]).item() == pytest.approx(1.0)
