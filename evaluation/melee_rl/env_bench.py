"""The Dolphin-only bench (5 Sep 2026, ``/rl-dolphin-speedup`` lever 0): the emulators' own pace.

No policy, no learner: a pool of Dolphins is stepped in lockstep with random (or neutral) controllers
and every layer of the frame-step is timed (:mod:`melee_rl.env.timing`).  What it answers:

* how long one lockstep frame-step of ``num_envs`` Dolphins takes without any actor-side work, and
  how that scales with the pool size (24 / 48 / 72 / 96) and the worker count;
* where the time goes -- the per-Dolphin advance time (send + ``console.step``), the in-worker
  lockstep tail (the slowest of a worker's Dolphins), the conversion, the pipe round trip and the
  cross-worker tail (the parent's wait until the last reply minus the wait until the first);
* what ``chunk_frames = k`` buys when the k frames are emulated back to back with one round trip and
  the controller held for the k frames (lever 8's Dolphin-side number).

``run_env_bench`` measures one specification; ``main`` runs a sweep from the command line and writes
a JSON report.  The Modal side is ``melee_rl.modal_app::bench_env`` (the wide Dolphin box).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from controller_codec import ControllerLabels, ControllerState, CustomV1Codec
from melee_rl.env.dolphin import DEFAULT_DOLPHIN_PATH, DolphinBackend, DolphinEnv, DolphinEnvConfig
from melee_rl.env.dolphin_mp import BackendFactory, WorkerDolphinEnv
from melee_rl.env.timing import subtract, summarize
from melee_rl.frames import neutral_labels

CONTROLLER_MODES = ("random", "neutral")
PLAYER_LAYOUTS: dict[str, tuple[str, str]] = {"self": ("policy", "policy"), "cpu": ("policy", "cpu")}
DEFAULT_STAGES = (25, 24, 18, 26, 8, 6)  # FD, BF, PS, DL, FoD, YS: the six-stage recipes' cycle


@dataclass(frozen=True)
class EnvBenchSpec:
    """One measurement: a pool of Dolphins, how it is stepped, and for how long."""

    num_envs: int = 96
    worker_processes: int = 16
    chunk_frames: int = 1
    frames: int = 600
    warmup: int = 60
    players: tuple[str, str] = ("policy", "policy")
    cpu_level: int = 9
    controllers: str = "random"
    hold: int = 1
    seed: int = 0
    stages: tuple[int, ...] = DEFAULT_STAGES
    step_threads: int = 0
    mp_start_method: str = "spawn"
    slippi_port: int = 51441
    dolphin_path: str = DEFAULT_DOLPHIN_PATH
    iso_path: str = "/iso/melee.iso"
    console_timeout_s: float = 30.0
    boot_timeout_s: float = 300.0
    check_iso: bool = True
    shared_memory: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if self.frames < 1 or self.warmup < 0:
            raise ValueError("frames must be >= 1 and warmup >= 0")
        if self.controllers not in CONTROLLER_MODES:
            raise ValueError(f"controllers must be one of {CONTROLLER_MODES}, got {self.controllers!r}")
        if self.hold < 1:
            raise ValueError("hold must be >= 1")
        if self.chunk_frames > 1 and self.worker_processes < 1:
            raise ValueError("chunk_frames > 1 needs worker processes (the push/pop surface)")
        if self.chunk_frames > 1 and (self.frames % self.chunk_frames or self.warmup % self.chunk_frames):
            raise ValueError("frames and warmup must be multiples of chunk_frames")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        layout = "self" if self.players == ("policy", "policy") else "cpu"
        shm = "_shm" if self.shared_memory else ""
        return (
            f"e{self.num_envs}_w{self.worker_processes}_k{self.chunk_frames}_{layout}_{self.controllers}{shm}"
        )


def env_config(spec: EnvBenchSpec) -> DolphinEnvConfig:
    return DolphinEnvConfig(
        num_envs=spec.num_envs,
        players=spec.players,
        dolphin_path=spec.dolphin_path,
        iso_path=spec.iso_path,
        stage=spec.stages[0],
        stages=spec.stages,
        slippi_port=spec.slippi_port,
        console_timeout_s=spec.console_timeout_s,
        boot_timeout_s=spec.boot_timeout_s,
        keepalive_seconds=0.0,
        check_iso=spec.check_iso,
        step_threads=spec.step_threads,
        worker_processes=spec.worker_processes,
        mp_start_method=spec.mp_start_method,
        chunk_frames=spec.chunk_frames,
        shared_memory=spec.shared_memory,
    )


class ControllerSource:
    """Random (uniform over the codec's vocabularies) or neutral controllers, held ``hold`` frames."""

    def __init__(self, spec: EnvBenchSpec, ports: Sequence[int]) -> None:
        self._codec = CustomV1Codec()
        self._ports = tuple(ports)
        self._batch = spec.num_envs
        self._hold = spec.hold
        self._random = spec.controllers == "random"
        self._generator = torch.Generator().manual_seed(spec.seed)
        self._frame = 0
        self._current: dict[int, ControllerState] = {}
        buttons, sticks = self._codec.axis_sizes
        self._sizes = (buttons, sticks)

    def _draw(self) -> dict[int, ControllerState]:
        if not self._random:
            neutral = self._codec.decode(neutral_labels((self._batch,)))
            return dict.fromkeys(self._ports, neutral)
        buttons, sticks = self._sizes
        out = {}
        for port in self._ports:
            labels = ControllerLabels(
                buttons=torch.randint(0, buttons, (self._batch,), generator=self._generator),
                main_stick=torch.randint(0, sticks, (self._batch,), generator=self._generator),
            )
            out[port] = self._codec.decode(labels)
        return out

    def next(self) -> dict[int, ControllerState]:
        if self._frame % self._hold == 0:
            self._current = self._draw()
        self._frame += 1
        return self._current


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        return ordered[min(count - 1, round(fraction * (count - 1)))]

    return {
        "mean": sum(ordered) / count,
        "p50": at(0.5),
        "p90": at(0.9),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _machine() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def _stats_totals(env: DolphinEnv | WorkerDolphinEnv) -> dict[str, int]:
    rows = env.stats()
    return {
        "relaunches": int(sum(int(row["relaunches"]) for row in rows)),
        "faults": int(sum(len(row["faults"]) for row in rows)),
        "menu_frames": int(sum(int(row["menu_frames"]) for row in rows)),
    }


def run_env_bench(
    spec: EnvBenchSpec,
    *,
    backend: DolphinBackend | None = None,
    backend_factory: BackendFactory | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Boot ``spec``'s pool, warm it up, step ``spec.frames`` lockstep frames, and report."""

    say = log if log is not None else (lambda _message: None)
    config = env_config(spec)
    report: dict[str, Any] = {
        "name": spec.name,
        "spec": asdict(spec),
        "machine": _machine(),
        "error": None,
        "ok": False,
    }
    env: DolphinEnv | WorkerDolphinEnv | None = None
    started = time.perf_counter()
    try:
        boot_started = time.perf_counter()
        if spec.worker_processes == 0:
            env = DolphinEnv(config, cpu_level=spec.cpu_level, backend=backend)
        else:
            env = WorkerDolphinEnv(config, cpu_level=spec.cpu_level, backend_factory=backend_factory)
        report["boot_s"] = time.perf_counter() - boot_started
        say(f"{spec.name}: {spec.num_envs} Dolphins up in {report['boot_s']:.1f} s")
        source = ControllerSource(spec, env.controlled_ports)
        resets = 0
        k = spec.chunk_frames

        def lockstep(count: int) -> list[float]:
            nonlocal resets
            walls: list[float] = []
            assert env is not None
            for _ in range(count):
                actions = source.next()
                tick = time.perf_counter()
                output = env.step(actions)
                walls.append(time.perf_counter() - tick)
                resets += int(output.needs_reset.sum())
            return walls

        def chunked(count: int) -> list[float]:
            nonlocal resets
            walls: list[float] = []
            assert isinstance(env, WorkerDolphinEnv)
            for _ in range(count // k):
                tick = time.perf_counter()
                for _frame in range(k):
                    env.push(source.next())
                for _frame in range(k):
                    resets += int(env.pop().needs_reset.sum())
                walls.append(time.perf_counter() - tick)
            return walls

        if k > 1:
            assert isinstance(env, WorkerDolphinEnv)
            env.pop()  # the boot state seeded into the asynchronous queue
        stepper = chunked if k > 1 else lockstep
        warm_started = time.perf_counter()
        stepper(spec.warmup)
        report["warmup_s"] = time.perf_counter() - warm_started
        before = env.timing()
        resets = 0
        measured_started = time.perf_counter()
        walls = stepper(spec.frames)
        wall_s = time.perf_counter() - measured_started
        delta = subtract(env.timing(), before)
        env_frames = spec.frames * spec.num_envs
        per_step = [wall / k for wall in walls]
        report.update(
            {
                "frames": spec.frames,
                "env_frames": env_frames,
                "wall_s": wall_s,
                "env_fps": env_frames / wall_s if wall_s > 0 else 0.0,
                "dolphin_fps": spec.frames / wall_s if wall_s > 0 else 0.0,
                "step_ms": {key: value * 1000.0 for key, value in _percentiles(per_step).items()},
                "chunk_ms": {key: value * 1000.0 for key, value in _percentiles(walls).items()},
                "resets": resets,
                "timing": summarize(delta),
                **_stats_totals(env),
                "ok": True,
            }
        )
        say(format_report(report))
    except Exception as error:  # the report is the deliverable
        report["error"] = repr(error)
        import traceback

        report["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:  # best effort on the way down
                report["close_error"] = repr(error)
    report["duration_s"] = time.perf_counter() - started
    return report


def format_report(report: dict[str, Any]) -> str:
    if not report.get("ok"):
        return f"{report['name']}: FAILED {report.get('error')}"
    step = report["step_ms"]
    timing = report["timing"]
    parts = [
        f"{report['name']}: {report['env_fps']:.0f} env fps ({report['dolphin_fps']:.1f} per Dolphin), "
        f"step {step['mean']:.1f} ms mean / {step['p50']:.1f} p50 / {step['p90']:.1f} p90 / "
        f"{step['max']:.1f} max; resets {report['resets']}, relaunches {report['relaunches']}",
        f"  Dolphin advance {timing.get('env/advance_mean_ms', 0.0):.1f} ms mean, "
        f"{timing.get('env/advance_max_ms', 0.0):.1f} ms per-step max, in-worker tail "
        f"{timing.get('env/tail_ms', 0.0):.1f} ms, convert {timing.get('env/convert_ms', 0.0):.2f} ms per "
        f"Dolphin frame, env step {timing.get('env/step_ms', 0.0):.1f} ms",
    ]
    if "parent/steps" in timing:
        first, last = timing["parent/wait_first_ms"], timing["parent/wait_last_ms"]
        parent = (
            f"  parent send {timing['parent/send_ms']:.2f} ms, first reply {first:.1f} ms, last reply "
            f"{last:.1f} ms (cross-worker tail {last - first:.1f} ms), recv {timing['parent/recv_ms']:.2f} "
            f"ms, assemble {timing['parent/assemble_ms']:.2f} ms"
        )
        worker = (
            f"  worker idle {timing['worker/recv_wait_ms']:.1f} ms, unpack {timing['worker/unpack_ms']:.2f} "
            f"ms, step {timing['worker/step_ms']:.1f} ms (max {timing['worker/step_ms_max']:.1f}), reply "
            f"{timing['worker/reply_ms']:.2f} ms"
        )
        parts.extend([parent, worker])
    return "\n".join(parts)


def run_env_bench_sweep(
    specs: Sequence[EnvBenchSpec],
    *,
    backend_factory: BackendFactory | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """One report per specification, in order.  Each specification gets its own Slippi-port band
    (``slippi_port + 256 * index``): the previous pool's Dolphins may still hold their UDP ports for a
    moment after ``close()`` (the 5 Sep 2026 sweep lost four measurements to "no free UDP port")."""

    reports = []
    for index, spec in enumerate(specs):
        banded = replace(spec, slippi_port=spec.slippi_port + 256 * index)
        reports.append(run_env_bench(banded, backend_factory=backend_factory, log=log))
    return reports


def parse_specs(argv: Sequence[str] | None = None) -> tuple[list[EnvBenchSpec], argparse.Namespace]:
    parser = argparse.ArgumentParser(description="The Dolphin-only frame-step bench (lever 0).")
    parser.add_argument("--sizes", default="96", help="comma-separated pool sizes, one measurement each")
    parser.add_argument(
        "--workers", default="16", help="worker processes per size (one value or one per size)"
    )
    parser.add_argument("--chunks", default="1", help="chunk_frames k per size (one value or one per size)")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--players", default="self", help="self | cpu, one value or one per size")
    parser.add_argument("--cpu-level", type=int, default=9)
    parser.add_argument("--controllers", choices=CONTROLLER_MODES, default="random")
    parser.add_argument("--hold", type=int, default=1, help="frames a random controller is held")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", default=",".join(str(stage) for stage in DEFAULT_STAGES))
    parser.add_argument("--step-threads", type=int, default=0)
    parser.add_argument("--mp-start-method", default="spawn")
    parser.add_argument("--slippi-port", type=int, default=51441)
    parser.add_argument("--dolphin-path", default=DEFAULT_DOLPHIN_PATH)
    parser.add_argument("--iso-path", default="/iso/melee.iso")
    parser.add_argument("--no-check-iso", action="store_true")
    parser.add_argument(
        "--shm", default="0", help="shared-memory frame exchange per size: 0 / 1, one value or one per size"
    )
    parser.add_argument("--json", default=None, help="write the list of reports here")
    args = parser.parse_args(argv)
    sizes = [int(value) for value in args.sizes.split(",") if value]
    workers = [int(value) for value in args.workers.split(",") if value]
    chunks = [int(value) for value in args.chunks.split(",") if value]
    layouts = [value for value in args.players.split(",") if value]
    for layout in layouts:
        if layout not in PLAYER_LAYOUTS:
            raise SystemExit(f"--players entries must be one of {sorted(PLAYER_LAYOUTS)}, got {layout!r}")
    shared = [value.strip() in ("1", "true", "yes") for value in args.shm.split(",") if value]
    if len(workers) == 1:
        workers = workers * len(sizes)
    if len(chunks) == 1:
        chunks = chunks * len(sizes)
    if len(layouts) == 1:
        layouts = layouts * len(sizes)
    if len(shared) == 1:
        shared = shared * len(sizes)
    if any(len(values) != len(sizes) for values in (workers, chunks, layouts, shared)):
        raise SystemExit(
            "--workers, --chunks, --players and --shm need one value each, or one per --sizes entry"
        )
    stages = tuple(int(value) for value in args.stages.split(",") if value)
    specs = [
        EnvBenchSpec(
            num_envs=size,
            worker_processes=count,
            chunk_frames=chunk,
            frames=args.frames,
            warmup=args.warmup,
            players=PLAYER_LAYOUTS[layout],
            cpu_level=args.cpu_level,
            controllers=args.controllers,
            hold=args.hold,
            seed=args.seed,
            stages=stages,
            step_threads=args.step_threads,
            mp_start_method=args.mp_start_method,
            slippi_port=args.slippi_port,
            dolphin_path=args.dolphin_path,
            iso_path=args.iso_path,
            check_iso=not args.no_check_iso,
            shared_memory=shm,
        )
        for size, count, chunk, layout, shm in zip(sizes, workers, chunks, layouts, shared, strict=True)
    ]
    return specs, args


def main(argv: Sequence[str] | None = None) -> int:
    specs, args = parse_specs(argv)
    reports = run_env_bench_sweep(specs, log=lambda message: print(message, flush=True))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=1))
        print(f"wrote {path}")
    return 0 if all(report["ok"] for report in reports) else 1


def with_defaults(spec: EnvBenchSpec, **changes: Any) -> EnvBenchSpec:
    return replace(spec, **changes)


__all__ = [
    "CONTROLLER_MODES",
    "DEFAULT_STAGES",
    "PLAYER_LAYOUTS",
    "ControllerSource",
    "EnvBenchSpec",
    "env_config",
    "format_report",
    "main",
    "parse_specs",
    "run_env_bench",
    "run_env_bench_sweep",
    "with_defaults",
]


if __name__ == "__main__":
    sys.exit(main())
