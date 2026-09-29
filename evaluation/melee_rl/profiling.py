"""Step profiling for the RL loop (5 Sep 2026, the throughput investigation, brief
``.claude/commands/rl-speedup.md``).

``[profiling]`` names the learner steps to profile.  During such a step the trainer, the rollout workers and
the learner mark *sections* -- wall-clock accumulators, plus ``torch.profiler.record_function`` labels while a
trace is being recorded -- and *windows*: bounded occurrences of a repeated unit of work (the first
``actor_frames`` frames of a worker's rollout, the first ``learner_chunks`` microbatch passes of a learner
phase) or one-shot regions (the prefix snapshot, the optimizer step, the checkpoint write), recorded with
``torch.profiler`` (CPU + CUDA activities) into one chrome trace per window.  When the step ends the traces
are gzipped into ``<dir>/step_<n>/`` and summarised into ``<dir>/summary.json`` and ``summary.md``: per window
the wall time, the device time by kernel family (matmul / attention / reduction / elementwise / copy /
other), the share of the wall with a kernel running, launches, host syncs (``cudaStreamSynchronize`` and its
siblings, ``aten::_local_scalar_dense``, device-to-host copies), copy bytes per direction, ATen dispatches,
the device memory peak, and the same numbers per section inside the window (kernels are attributed to the
section that launched them through Kineto's correlation ids).  Nothing here reads or writes a tensor's
values or the RNG: a profiled run is bit-identical to an unprofiled one (``tests/rl/test_profiling.py``).

Window names follow ``<phase>/<kind>/<who>``; the occurrence limit comes from the name: ``actor/frames/*``
takes ``actor_frames``, ``learner/chunks/*`` takes ``learner_chunks``, everything else is one-shot.  Only
one window records at a time: opening a second one closes the first (so a phase with fewer occurrences than
its limit still gets summarised), and ``end_step`` closes whatever is left open.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import gzip
import json
import os
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import torch
from torch.profiler import ProfilerActivity, profile, record_function

SUMMARY_JSON: Final[str] = "summary.json"
SUMMARY_MD: Final[str] = "summary.md"
FAMILIES: Final[tuple[str, ...]] = ("matmul", "attention", "reduction", "elementwise", "copy", "other")
_FAMILY_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("copy", re.compile(r"Memcpy|Memset|copy_kernel|cudaMemcpy", re.IGNORECASE)),
    ("attention", re.compile(r"attention|fmha|flash|sdpa|mem_efficient", re.IGNORECASE)),
    ("matmul", re.compile(r"gemm|gemv|cutlass|cublas|xmma|wgrad|dgrad|matmul|bmm|dot_", re.IGNORECASE)),
    (
        "reduction",
        re.compile(
            r"reduce|softmax|norm|sum_|mean|argmax|cumsum|multinomial|scan|logsumexp|welford", re.IGNORECASE
        ),
    ),
    (
        "elementwise",
        re.compile(
            r"elementwise|vectorized|fill|where|index|gather|scatter|cat_|CatArray|arange|one_hot|embedding|"
            r"unrolled|clamp|masked|triu|tril|nonzero|copy_|compare|bitwise|activation|silu|sigmoid|"
            r"remainder|binary|unary|distribution|philox|random",
            re.IGNORECASE,
        ),
    ),
)
_DEVICE_CATEGORIES: Final[frozenset[str]] = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_RUNTIME_CATEGORIES: Final[frozenset[str]] = frozenset({"cuda_runtime", "cuda_driver"})
_LOCAL_SCALAR: Final[str] = "aten::_local_scalar_dense"


def kernel_family(name: str) -> str:
    """The family of a device event by its name (first matching rule; ``"other"`` when none does)."""

    for family, pattern in _FAMILY_RULES:
        if pattern.search(name):
            return family
    return "other"


@dataclass(frozen=True)
class ProfilingConfig:
    """``[profiling]``: which learner steps to profile and how much of each to trace.

    ``steps`` are learner-step indices (``Trainer.step`` before the step runs; the burn-in ladder counts);
    ``actor_frames`` / ``learner_chunks`` bound the traced occurrences of a worker's frames / a learner
    phase's microbatch passes; ``dir`` is where the files go (relative to the run's checkpoint directory,
    or absolute); ``memory`` records the device memory peaks; ``with_stack`` adds Python stacks to the
    traces (large).
    """

    enabled: bool = False
    steps: tuple[int, ...] = ()
    actor_frames: int = 32
    learner_chunks: int = 4
    dir: str = "profile"
    memory: bool = True
    with_stack: bool = False

    def __post_init__(self) -> None:
        if self.enabled and not self.steps:
            raise ValueError("[profiling] steps must list at least one learner step when enabled")
        if any(step < 0 for step in self.steps):
            raise ValueError("[profiling] steps must be >= 0")
        if self.actor_frames < 1:
            raise ValueError("[profiling] actor_frames must be >= 1")
        if self.learner_chunks < 1:
            raise ValueError("[profiling] learner_chunks must be >= 1")
        if not self.dir:
            raise ValueError("[profiling] dir must be a directory name")

    def limit(self, window: str) -> int:
        """Occurrences of ``window`` to trace: by the name's prefix (frames, chunks) else one."""

        if window.startswith("actor/frames"):
            return self.actor_frames
        if window.startswith("learner/chunks"):
            return self.learner_chunks
        return 1


class SectionProfiler(Protocol):
    """What the trainer, the workers and the learner call (a no-op outside profiled steps)."""

    @property
    def active(self) -> bool: ...

    def begin_step(self, step: int) -> None: ...

    def end_step(self, metrics: Mapping[str, float] | None = None) -> bool: ...

    def section(self, name: str) -> contextlib.AbstractContextManager[None]: ...

    def window(self, name: str) -> contextlib.AbstractContextManager[None]: ...


class NullProfiler:
    """The inert profiler: every call is a no-op and every context manager is ``nullcontext``."""

    @property
    def active(self) -> bool:
        return False

    def begin_step(self, step: int) -> None:
        return None

    def end_step(self, metrics: Mapping[str, float] | None = None) -> bool:
        return False

    def section(self, name: str) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def window(self, name: str) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


NULL_PROFILER: Final[NullProfiler] = NullProfiler()


# ---------------------------------------------------------------------------
# trace summary
# ---------------------------------------------------------------------------


def _memcpy_direction(name: str) -> str | None:
    if "HtoD" in name:
        return "h2d"
    if "DtoH" in name:
        return "d2h"
    if "DtoD" in name:
        return "d2d"
    return None


class _Intervals:
    """Membership lookup over the ``(start, end)`` ranges of one annotation name (sorted by start; a
    range that contains ``timestamp`` starts no earlier than ``timestamp - longest``, which bounds the
    backward scan when ranges of the same name nest)."""

    def __init__(self, ranges: Sequence[tuple[float, float]]) -> None:
        ordered = sorted(ranges)
        self._starts = [start for start, _ in ordered]
        self._ends = [end for _, end in ordered]
        self._longest = max((end - start for start, end in ordered), default=0.0)

    def contains(self, timestamp: float) -> bool:
        index = bisect.bisect_right(self._starts, timestamp) - 1
        floor = timestamp - self._longest
        while index >= 0 and self._starts[index] >= floor:
            if self._ends[index] >= timestamp:
                return True
            index -= 1
        return False


def summarize_events(
    events: Sequence[Mapping[str, Any]], *, occurrences: int, wall_s: float
) -> dict[str, Any]:
    """Aggregate the ``traceEvents`` of one chrome trace (times in microseconds) into the window summary."""

    device: list[tuple[float, float, str, str, Any, int]] = []  # start, end, cat, name, correlation, bytes
    launches: dict[Any, float] = {}  # correlation id -> the CPU timestamp of the runtime call
    sync_timestamps: list[float] = []
    aten_timestamps: list[float] = []
    annotations: dict[str, list[tuple[float, float]]] = defaultdict(list)
    syncs = {"stream": 0, "device": 0, "event": 0, "local_scalar": 0, "d2h": 0}
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        args = event.get("args") or {}
        if cat in _DEVICE_CATEGORIES:
            size = args.get("bytes")
            device.append((start, start + duration, cat, name, args.get("correlation"), int(size or 0)))
            if _memcpy_direction(name) == "d2h":
                syncs["d2h"] += 1
        elif cat in _RUNTIME_CATEGORIES:
            correlation = args.get("correlation")
            if correlation is not None:
                launches[correlation] = start
            if "Synchronize" in name:
                if "Stream" in name:
                    syncs["stream"] += 1
                elif "Device" in name:
                    syncs["device"] += 1
                else:
                    syncs["event"] += 1
                sync_timestamps.append(start)
        elif cat == "cpu_op":
            if name.startswith("aten::"):
                aten_timestamps.append(start)
                if name == _LOCAL_SCALAR:
                    syncs["local_scalar"] += 1
                    sync_timestamps.append(start)
        elif cat == "user_annotation":
            annotations[name].append((start, start + duration))
    device.sort()
    busy_us = 0.0
    current_start: float | None = None
    current_end = 0.0
    kernel_us = 0.0
    device_us = 0.0
    kernels = 0
    by_family: dict[str, float] = defaultdict(float)
    memcpy = {
        direction + suffix: 0.0 if suffix == "_s" else 0
        for direction in ("h2d", "d2h", "d2d")
        for suffix in ("_count", "_bytes", "_s")
    }
    for start, end, cat, name, _, size in device:
        duration = end - start
        device_us += duration
        if cat == "kernel":
            kernels += 1
            kernel_us += duration
            by_family[kernel_family(name)] += duration
        else:
            by_family["copy"] += duration
            direction = _memcpy_direction(name) if cat == "gpu_memcpy" else None
            if direction is not None:
                memcpy[f"{direction}_count"] += 1
                memcpy[f"{direction}_bytes"] += size
                memcpy[f"{direction}_s"] += duration * 1e-6
        if current_start is None or start > current_end:
            if current_start is not None:
                busy_us += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        busy_us += current_end - current_start
    total_syncs = syncs["stream"] + syncs["device"] + syncs["event"] + syncs["local_scalar"]
    per_occurrence = max(1, occurrences)
    sync_sorted = sorted(sync_timestamps)
    aten_sorted = sorted(aten_timestamps)
    annotation_summary: dict[str, dict[str, Any]] = {}
    for name, ranges in annotations.items():
        lookup = _Intervals(ranges)
        wall_us = sum(end - start for start, end in ranges)
        inside_device_us = 0.0
        inside_kernels = 0
        for start, end, cat, _, correlation, _ in device:
            launched = launches.get(correlation)
            if launched is None or not lookup.contains(launched):
                continue
            inside_device_us += end - start
            if cat == "kernel":
                inside_kernels += 1
        inside_syncs = sum(1 for stamp in sync_sorted if lookup.contains(stamp))
        inside_aten = sum(1 for stamp in aten_sorted if lookup.contains(stamp))
        count = len(ranges)
        annotation_summary[name] = {
            "count": count,
            "wall_s": wall_us * 1e-6,
            "device_s": inside_device_us * 1e-6,
            "kernels": inside_kernels,
            "syncs": inside_syncs,
            "aten_ops": inside_aten,
            "per_call": {
                "wall_ms": wall_us * 1e-3 / count,
                "device_ms": inside_device_us * 1e-3 / count,
                "kernels": inside_kernels / count,
                "syncs": inside_syncs / count,
                "aten_ops": inside_aten / count,
            },
        }
    return {
        "occurrences": occurrences,
        "wall_s": wall_s,
        "kernels": kernels,
        "device_events": len(device),
        "kernel_time_s": kernel_us * 1e-6,
        "device_time_s": device_us * 1e-6,
        "busy_s": busy_us * 1e-6,
        "busy_share": (busy_us * 1e-6 / wall_s) if wall_s > 0.0 else 0.0,
        "by_family": {family: seconds * 1e-6 for family, seconds in by_family.items() if seconds > 0.0},
        "aten_ops": len(aten_timestamps),
        "syncs": syncs,
        "memcpy": memcpy,
        "per_occurrence": {
            "wall_ms": wall_s * 1e3 / per_occurrence,
            "device_ms": device_us * 1e-3 / per_occurrence,
            "kernels": kernels / per_occurrence,
            "aten_ops": len(aten_timestamps) / per_occurrence,
            "syncs": total_syncs / per_occurrence,
            "h2d_mb": memcpy["h2d_bytes"] / 1e6 / per_occurrence,
            "d2h_mb": memcpy["d2h_bytes"] / 1e6 / per_occurrence,
        },
        "annotations": annotation_summary,
    }


# ---------------------------------------------------------------------------
# the step profiler
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    calls: int = 0
    wall_s: float = 0.0


@dataclass
class _Window:
    name: str
    limit: int
    calls: int = 0
    occurrences: int = 0
    started: float = 0.0
    wall_s: float = 0.0
    profiler: Any = None
    allocated_before: int = 0
    summary: dict[str, Any] | None = None


@dataclass
class _Step:
    step: int
    started: float
    directory: Path
    allocated_before: int
    sections: dict[str, _Section] = field(default_factory=dict)
    windows: dict[str, _Window] = field(default_factory=dict)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class StepProfiler:
    """Records the windows and sections of the configured learner steps; see the module docstring."""

    def __init__(self, config: ProfilingConfig, output_dir: str | Path, device: torch.device | str) -> None:
        if not config.enabled:
            raise ValueError("StepProfiler needs an enabled ProfilingConfig (use build_profiler)")
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.records: list[dict[str, Any]] = []
        self._current: _Step | None = None
        self._open: str | None = None

    # -- facts ----------------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._current is not None

    @property
    def open_window(self) -> str | None:
        return self._open

    def environment(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "torch": torch.__version__,
            "device": self.device.type,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "threads": torch.get_num_threads(),
        }
        if self._cuda:
            info["device_name"] = torch.cuda.get_device_name(self.device)
            info["cuda"] = torch.version.cuda
        return info

    # -- steps ---------------------------------------------------------------------------------

    def begin_step(self, step: int) -> None:
        if self._current is not None:
            self.end_step()
        if step not in self.config.steps:
            return
        directory = self.output_dir / f"step_{step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        allocated = 0
        if self._cuda:
            torch.cuda.synchronize(self.device)
            if self.config.memory:
                torch.cuda.reset_peak_memory_stats(self.device)
                allocated = torch.cuda.memory_allocated(self.device)
        self._current = _Step(
            step=step, started=time.perf_counter(), directory=directory, allocated_before=allocated
        )
        self._open = None

    def end_step(self, metrics: Mapping[str, float] | None = None) -> bool:
        current = self._current
        if current is None:
            return False
        if self._open is not None:
            self._close_window(current.windows[self._open])
        wall_s = time.perf_counter() - current.started
        memory: dict[str, Any] = {
            "device": self.device.type,
            "allocated_before_bytes": current.allocated_before,
        }
        if self._cuda and self.config.memory:
            torch.cuda.synchronize(self.device)
            memory["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            memory["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            memory["allocated_after_bytes"] = torch.cuda.memory_allocated(self.device)
        timing = {
            key: float(value)
            for key, value in (metrics or {}).items()
            if key.startswith("timing/") or key in ("step", "env/fps", "frames_seen")
        }
        windows: dict[str, Any] = {}
        for name, window in current.windows.items():
            summary = window.summary if window.summary is not None else {"occurrences": window.occurrences}
            windows[name] = {**summary, "limit": window.limit, "calls": window.calls}
        sections = {
            name: {
                "calls": section.calls,
                "wall_s": section.wall_s,
                "wall_ms_per_call": section.wall_s * 1e3 / section.calls if section.calls else 0.0,
            }
            for name, section in current.sections.items()
        }
        record = {
            "step": current.step,
            "wall_s": wall_s,
            "timing": timing,
            "metrics": {key: float(value) for key, value in (metrics or {}).items()},
            "windows": windows,
            "sections": sections,
            "memory": memory,
        }
        self.records.append(record)
        self._current = None
        self._open = None
        self._write_summaries()
        return True

    # -- sections and windows ---------------------------------------------------------------------

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        current = self._current
        if current is None:
            yield
            return
        state = current.sections.get(name)
        if state is None:
            state = current.sections[name] = _Section()
        state.calls += 1
        started = time.perf_counter()
        try:
            if self._open is not None:
                with record_function(name):
                    yield
            else:
                yield
        finally:
            state.wall_s += time.perf_counter() - started

    @contextlib.contextmanager
    def window(self, name: str) -> Iterator[None]:
        current = self._current
        if current is None:
            yield
            return
        state = current.windows.get(name)
        if state is None:
            state = current.windows[name] = _Window(name=name, limit=self.config.limit(name))
        state.calls += 1
        if state.summary is not None or state.occurrences >= state.limit:
            yield
            return
        if self._open is not None and self._open != name:
            self._close_window(current.windows[self._open])
        if self._open is None:
            self._open_window(state)
        state.occurrences += 1
        try:
            with record_function(f"window/{name}"):
                yield
        finally:
            if state.occurrences >= state.limit and self._open == name:
                self._close_window(state)

    def _open_window(self, state: _Window) -> None:
        if self._cuda:
            torch.cuda.synchronize(self.device)
            if self.config.memory:
                torch.cuda.reset_peak_memory_stats(self.device)
                state.allocated_before = torch.cuda.memory_allocated(self.device)
        activities = [ProfilerActivity.CPU]
        if self._cuda:
            activities.append(ProfilerActivity.CUDA)
        state.profiler = profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=self.config.with_stack,
        )
        state.started = time.perf_counter()
        state.profiler.start()
        self._open = state.name

    def _close_window(self, state: _Window) -> None:
        if state.profiler is None:
            self._open = None
            return
        if self._cuda:
            torch.cuda.synchronize(self.device)
        state.profiler.stop()
        state.wall_s = time.perf_counter() - state.started
        memory: dict[str, Any] = {}
        if self._cuda and self.config.memory:
            memory = {
                "allocated_before_bytes": state.allocated_before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
        current = self._current
        assert current is not None
        processing_started = time.perf_counter()
        raw = current.directory / f"{_slug(state.name)}.json"
        packed = raw.with_name(raw.name + ".gz")
        state.profiler.export_chrome_trace(str(raw))
        with raw.open("r", encoding="utf-8") as stream:
            trace = json.load(stream)
        events = trace.get("traceEvents", []) if isinstance(trace, Mapping) else []
        summary = summarize_events(events, occurrences=state.occurrences, wall_s=state.wall_s)
        del trace, events
        with raw.open("rb") as source, gzip.open(packed, "wb") as target:
            shutil.copyfileobj(source, target)
        raw.unlink()
        summary["memory"] = memory
        summary["trace"] = str(packed.relative_to(self.output_dir))
        # The export / parse / gzip of the trace runs inside the step: report it so the step's own timers
        # can be corrected (the bench of 5 Sep 2026 carried ~125 s of it per profiled step).
        summary["processing_s"] = time.perf_counter() - processing_started
        state.summary = summary
        state.profiler = None
        self._open = None

    # -- files -----------------------------------------------------------------------------------

    def _write_summaries(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": dataclasses.asdict(self.config),
            "environment": self.environment(),
            "steps": self.records,
        }
        _write_text_atomic(self.output_dir / SUMMARY_JSON, json.dumps(payload, indent=1))
        _write_text_atomic(self.output_dir / SUMMARY_MD, render_summary(payload))


def build_profiler(
    config: ProfilingConfig, checkpoint_dir: Path | None, device: torch.device | str
) -> SectionProfiler:
    """:data:`NULL_PROFILER` when profiling is off; else a :class:`StepProfiler` writing under
    ``checkpoint_dir / config.dir`` (or ``config.dir`` itself when absolute)."""

    if not config.enabled:
        return NULL_PROFILER
    directory = Path(config.dir)
    if not directory.is_absolute():
        if checkpoint_dir is None:
            raise ValueError(
                "[profiling] needs a checkpoint directory for its relative dir "
                f"{config.dir!r}: set runtime.checkpoint_dir / --checkpoint-dir, or give an absolute dir"
            )
        directory = Path(checkpoint_dir) / directory
    return StepProfiler(config, directory, device)


# ---------------------------------------------------------------------------
# the markdown summary
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _merged_annotations(windows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for window in windows.values():
        for name, item in (window.get("annotations") or {}).items():
            total = merged.setdefault(
                name, {"count": 0, "wall_s": 0.0, "device_s": 0.0, "kernels": 0, "syncs": 0, "aten_ops": 0}
            )
            for key in total:
                total[key] += item[key]
    return merged


def render_summary(payload: Mapping[str, Any]) -> str:
    """``summary.md``: one block per profiled step (windows, sections, kernel families)."""

    lines = ["# Step profile", ""]
    environment = payload.get("environment") or {}
    if environment:
        lines.append(
            "Environment: " + ", ".join(f"{key} = {value}" for key, value in sorted(environment.items()))
        )
        lines.append("")
    for record in payload.get("steps", []):
        timing = record.get("timing") or {}
        step_s = timing.get("timing/step_s")
        head = f"## Step {record['step']} (wall {_fmt(record['wall_s'], 1)} s"
        if step_s is not None:
            head += f", timing/step_s {_fmt(step_s, 1)}"
        processing = sum(float(w.get("processing_s", 0.0)) for w in (record.get("windows") or {}).values())
        lines.append(head + f", trace processing {_fmt(processing, 1)} s inside the step)")
        lines.append("")
        if timing:
            lines.append(
                "Timing keys: "
                + ", ".join(f"{key} = {_fmt(value, 3)}" for key, value in sorted(timing.items()))
            )
            lines.append("")
        memory = record.get("memory") or {}
        if "peak_allocated_bytes" in memory:
            lines.append(
                f"Device memory: peak allocated {memory['peak_allocated_bytes'] / 2**30:.2f} GiB, "
                f"peak reserved {memory['peak_reserved_bytes'] / 2**30:.2f} GiB"
            )
            lines.append("")
        windows = record.get("windows") or {}
        lines.append(
            "| window | traced / calls | wall ms/occ | device ms/occ | busy % | kernels/occ | aten/occ | "
            "syncs/occ | H2D MB/occ | D2H MB/occ | peak alloc GiB |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for name in sorted(windows):
            window = windows[name]
            per = window.get("per_occurrence")
            if not per:
                lines.append(f"| `{name}` | 0 / {window.get('calls', 0)} | | | | | | | | | |")
                continue
            peak = (window.get("memory") or {}).get("peak_allocated_bytes")
            lines.append(
                f"| `{name}` | {window['occurrences']} / {window.get('calls', window['occurrences'])} | "
                f"{_fmt(per['wall_ms'])} | {_fmt(per['device_ms'])} | "
                f"{_fmt(100 * window['busy_share'], 1)} | "
                f"{_fmt(per['kernels'], 1)} | {_fmt(per['aten_ops'], 1)} | {_fmt(per['syncs'], 1)} | "
                f"{_fmt(per['h2d_mb'])} | {_fmt(per['d2h_mb'])} | "
                f"{'' if peak is None else _fmt(peak / 2**30)} |"
            )
        lines.append("")
        families = [(name, window) for name, window in sorted(windows.items()) if window.get("by_family")]
        if families:
            lines.append("| window | " + " | ".join(f"{family} ms/occ" for family in FAMILIES) + " |")
            lines.append("|---|" + "---|" * len(FAMILIES))
            for name, window in families:
                occurrences = max(1, int(window.get("occurrences", 1)))
                cells = [
                    _fmt(window["by_family"].get(family, 0.0) * 1e3 / occurrences) for family in FAMILIES
                ]
                lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
            lines.append("")
        sections = record.get("sections") or {}
        traced = _merged_annotations(windows)
        lines.append(
            "| section | calls | wall ms/call | traced calls | traced wall ms/call | device ms/call | "
            "kernels/call | syncs/call | aten/call |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name in sorted(sections):
            section = sections[name]
            item = traced.get(name)
            if item and item["count"]:
                count = item["count"]
                extra = (
                    f"{count} | {_fmt(item['wall_s'] * 1e3 / count)} | "
                    f"{_fmt(item['device_s'] * 1e3 / count)} | "
                    f"{_fmt(item['kernels'] / count, 1)} | {_fmt(item['syncs'] / count, 1)} | "
                    f"{_fmt(item['aten_ops'] / count, 1)}"
                )
            else:
                extra = "0 | | | | | "
            lines.append(f"| `{name}` | {section['calls']} | {_fmt(section['wall_ms_per_call'])} | {extra} |")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "FAMILIES",
    "NULL_PROFILER",
    "SUMMARY_JSON",
    "SUMMARY_MD",
    "NullProfiler",
    "ProfilingConfig",
    "SectionProfiler",
    "StepProfiler",
    "build_profiler",
    "kernel_family",
    "render_summary",
    "summarize_events",
]
