"""Where a Dolphin frame-step's wall time goes (5 Sep 2026, ``/rl-dolphin-speedup`` lever 0).

Three cumulative counter dicts, one per layer of the lockstep step, all read by :func:`subtract` and
:func:`summarize`:

* the **environment** (``DolphinEnv.step_rows``): the step's wall, the thread fan-out's wall, the sum
  and the per-step maximum of the per-Dolphin advance times (send + ``console.step``), the conversion
  time, the deferred (parked / relaunch) handling, the output build, and a histogram of the per-Dolphin
  advance times in :data:`ADVANCE_BINS_MS`;
* the **worker process** (``dolphin_mp._worker_main``): time blocked in ``recv`` (idle: the parent's
  turnaround), unpacking the controller arrays, stepping its ``DolphinEnv``, and sending the reply;
* the **parent** (``WorkerDolphinEnv``): the sends, the wait until the first and the last worker reply
  is available, the receives (unpickling), the assembly, and the per-worker arrival times.

The counters are cumulative since construction; a reader keeps a snapshot and takes deltas, so a
rollout's numbers are one ``timing()`` round trip per worker (``EnvTimingTracker``).  Nothing here
touches what the environments do -- it is bookkeeping around the calls that already happen.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, runtime_checkable

ADVANCE_BINS_MS: Final[tuple[float, ...]] = (2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
"""Upper edges (ms) of the per-Dolphin advance-time histogram; the last bin is everything above."""

BIN_LABELS: Final[tuple[str, ...]] = (
    *(f"le{int(edge)}ms" for edge in ADVANCE_BINS_MS),
    f"gt{int(ADVANCE_BINS_MS[-1])}ms",
)


def bin_index(seconds: float) -> int:
    """The histogram bin of one advance time."""

    return bisect.bisect_left(ADVANCE_BINS_MS, seconds * 1000.0)


def new_env_timing() -> dict[str, Any]:
    return {
        "steps": 0,
        "wall_s": 0.0,
        "fanout_s": 0.0,
        "advance_s": 0.0,
        "advance_max_s": 0.0,
        "convert_s": 0.0,
        "deferred_s": 0.0,
        "output_s": 0.0,
        "instance_frames": 0,
        "advance_hist": [0] * (len(ADVANCE_BINS_MS) + 1),
    }


def new_worker_timing() -> dict[str, Any]:
    return {"messages": 0, "recv_wait_s": 0.0, "unpack_s": 0.0, "step_s": 0.0, "reply_s": 0.0}


def new_parent_timing(workers: int) -> dict[str, Any]:
    return {
        "steps": 0,
        "send_s": 0.0,
        "wait_first_s": 0.0,
        "wait_last_s": 0.0,
        "recv_s": 0.0,
        "assemble_s": 0.0,
        "arrival_s": [0.0] * workers,
    }


def subtract(now: Any, before: Any) -> Any:
    """``now - before`` for the nested counter dicts (numbers, lists of numbers, dicts, lists of dicts)."""

    if isinstance(now, Mapping):
        assert isinstance(before, Mapping)
        return {key: subtract(value, before[key]) for key, value in now.items()}
    if isinstance(now, list):
        assert isinstance(before, list) and len(now) == len(before)
        return [subtract(a, b) for a, b in zip(now, before, strict=True)]
    if isinstance(now, bool):
        return now
    if isinstance(now, int | float):
        return now - before
    return now


def _per(value: float, count: float) -> float:
    return value / count if count else 0.0


def summarize_env(delta: Mapping[str, Any], prefix: str = "env") -> dict[str, float]:
    """Flat per-step (ms) and per-Dolphin-frame (ms) numbers of one environment's counter delta."""

    steps = float(delta["steps"])
    frames = float(delta["instance_frames"])
    hist = list(delta["advance_hist"])
    total = float(sum(hist))
    advance_mean = _per(delta["advance_s"], frames) * 1000.0
    fanout = _per(delta["fanout_s"], steps) * 1000.0
    out = {
        f"{prefix}/steps": steps,
        f"{prefix}/step_ms": _per(delta["wall_s"], steps) * 1000.0,
        f"{prefix}/fanout_ms": fanout,
        f"{prefix}/advance_mean_ms": advance_mean,
        f"{prefix}/advance_max_ms": _per(delta["advance_max_s"], steps) * 1000.0,
        f"{prefix}/tail_ms": fanout - advance_mean,
        f"{prefix}/convert_ms": _per(delta["convert_s"], frames) * 1000.0,
        f"{prefix}/deferred_ms": _per(delta["deferred_s"], steps) * 1000.0,
        f"{prefix}/output_ms": _per(delta["output_s"], steps) * 1000.0,
    }
    for label, count in zip(BIN_LABELS, hist, strict=True):
        out[f"{prefix}/advance_share/{label}"] = _per(float(count), total)
    return out


def summarize_worker(delta: Mapping[str, Any], prefix: str = "worker") -> dict[str, float]:
    messages = float(delta["messages"])
    return {
        f"{prefix}/messages": messages,
        f"{prefix}/recv_wait_ms": _per(delta["recv_wait_s"], messages) * 1000.0,
        f"{prefix}/unpack_ms": _per(delta["unpack_s"], messages) * 1000.0,
        f"{prefix}/step_ms": _per(delta["step_s"], messages) * 1000.0,
        f"{prefix}/reply_ms": _per(delta["reply_s"], messages) * 1000.0,
    }


def summarize_parent(delta: Mapping[str, Any], prefix: str = "parent") -> dict[str, float]:
    steps = float(delta["steps"])
    arrivals = [_per(value, steps) * 1000.0 for value in delta["arrival_s"]]
    out = {
        f"{prefix}/steps": steps,
        f"{prefix}/send_ms": _per(delta["send_s"], steps) * 1000.0,
        f"{prefix}/wait_first_ms": _per(delta["wait_first_s"], steps) * 1000.0,
        f"{prefix}/wait_last_ms": _per(delta["wait_last_s"], steps) * 1000.0,
        f"{prefix}/recv_ms": _per(delta["recv_s"], steps) * 1000.0,
        f"{prefix}/assemble_ms": _per(delta["assemble_s"], steps) * 1000.0,
        f"{prefix}/arrival_min_ms": min(arrivals) if arrivals else 0.0,
        f"{prefix}/arrival_max_ms": max(arrivals) if arrivals else 0.0,
        f"{prefix}/arrival_mean_ms": _per(sum(arrivals), len(arrivals)),
    }
    return out


def _mean_max(rows: Sequence[Mapping[str, float]], key: str) -> tuple[float, float]:
    values = [row[key] for row in rows]
    return (_per(sum(values), len(values)), max(values)) if values else (0.0, 0.0)


def summarize(delta: Mapping[str, Any]) -> dict[str, float]:
    """The flat summary of a ``timing()`` delta: an in-process environment's (``env/*``) or a worker
    environment's (``parent/*``, ``worker/*`` mean and max over workers, ``env/*`` frame-weighted)."""

    if "parent" not in delta:
        return summarize_env(delta)
    out = summarize_parent(delta["parent"])
    workers = list(delta["workers"])
    worker_rows = [summarize_worker(row["worker"]) for row in workers]
    for key in ("recv_wait_ms", "unpack_ms", "step_ms", "reply_ms"):
        mean, worst = _mean_max(worker_rows, f"worker/{key}")
        out[f"worker/{key}"] = mean
        out[f"worker/{key}_max"] = worst
    out["worker/messages"] = _mean_max(worker_rows, "worker/messages")[0]
    # The environment counters summed over the workers: the per-step means become means over
    # (worker, step) pairs, the per-frame means stay per Dolphin frame, and ``env/steps`` is the
    # parent's step count (every worker steps once per parent step).
    merged = new_env_timing()
    count = 0
    for row in workers:
        env = row["env"]
        for key, value in env.items():
            if key == "advance_hist":
                merged[key] = [a + b for a, b in zip(merged[key], value, strict=True)]
            else:
                merged[key] += value
        count += 1
    env_summary = summarize_env(merged)
    env_summary["env/steps"] = _per(float(merged["steps"]), count)
    out.update(env_summary)
    return out


@runtime_checkable
class TimedEnv(Protocol):
    def timing(self) -> Mapping[str, Any]: ...


class EnvTimingTracker:
    """Snapshots per named environment; :meth:`update` returns the flat metrics of the delta since
    the previous call, under ``env_timing/<name>/``.  Environments without ``timing()`` are skipped."""

    def __init__(self) -> None:
        self._previous: dict[str, Mapping[str, Any]] = {}

    def update(self, named: Sequence[tuple[str, object]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, env in named:
            method = getattr(env, "timing", None)
            if not callable(method):
                continue
            now = method()
            before = self._previous.get(name)
            self._previous[name] = now
            if before is None:
                continue  # the first call establishes the baseline (boot and warm-up excluded)
            for key, value in summarize(subtract(now, before)).items():
                metrics[f"env_timing/{name}/{key}"] = float(value)
        return metrics


__all__ = [
    "ADVANCE_BINS_MS",
    "BIN_LABELS",
    "EnvTimingTracker",
    "TimedEnv",
    "bin_index",
    "new_env_timing",
    "new_parent_timing",
    "new_worker_timing",
    "subtract",
    "summarize",
    "summarize_env",
    "summarize_parent",
    "summarize_worker",
]
