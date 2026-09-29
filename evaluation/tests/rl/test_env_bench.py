"""The Dolphin-only bench (5 Sep 2026, lever 0) on the fake consoles: shape, arithmetic, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

melee = pytest.importorskip("melee")  # the ``dolphin`` extra (melee==0.47.3)

from melee_rl import env_bench  # noqa: E402
from melee_rl.env_bench import (  # noqa: E402
    ControllerSource,
    EnvBenchSpec,
    env_config,
    format_report,
    parse_specs,
    run_env_bench,
)
from tests.rl import _mp_backend  # noqa: E402


def test_spec_validation_and_name() -> None:
    spec = EnvBenchSpec(num_envs=4, worker_processes=2)
    assert spec.name == "e4_w2_k1_self_random"
    assert (
        EnvBenchSpec(players=("policy", "cpu"), controllers="neutral", label="").name
        == "e96_w16_k1_cpu_neutral"
    )
    assert EnvBenchSpec(label="mine").name == "mine"
    with pytest.raises(ValueError, match="controllers"):
        EnvBenchSpec(controllers="chaos")
    with pytest.raises(ValueError, match="hold"):
        EnvBenchSpec(hold=0)
    with pytest.raises(ValueError, match="worker processes"):
        EnvBenchSpec(worker_processes=0, chunk_frames=2)
    with pytest.raises(ValueError, match="multiples"):
        EnvBenchSpec(worker_processes=2, chunk_frames=2, frames=5)
    config = env_config(EnvBenchSpec(num_envs=12, worker_processes=2, stages=(25, 24), check_iso=False))
    assert config.num_envs == 12 and config.worker_processes == 2 and config.stages == (25, 24)
    assert config.stage == 25 and config.keepalive_seconds == 0.0 and config.players == ("policy", "policy")


def test_controller_source_is_seeded_and_holds() -> None:
    spec = EnvBenchSpec(num_envs=3, worker_processes=1, seed=7, hold=2, check_iso=False)
    a = ControllerSource(spec, (0, 1))
    b = ControllerSource(spec, (0, 1))
    first = a.next()
    assert set(first) == {0, 1}
    assert torch.equal(first[0].shoulder, b.next()[0].shoulder)
    assert torch.equal(first[0].main_stick.x, a.next()[0].main_stick.x)  # held for hold = 2 frames
    third = a.next()
    assert not torch.equal(first[0].as_packed_tensor(), third[0].as_packed_tensor())
    neutral = ControllerSource(
        EnvBenchSpec(num_envs=2, worker_processes=1, controllers="neutral"), (0,)
    ).next()
    assert torch.equal(neutral[0].main_stick.x, torch.full((2,), 0.5))


def test_bench_in_worker_mode_reports_the_measured_frames_only() -> None:
    spec = EnvBenchSpec(
        num_envs=4,
        worker_processes=2,
        frames=6,
        warmup=2,
        players=("policy", "cpu"),
        stages=(25,),
        slippi_port=51800,
        mp_start_method="forkserver",
        check_iso=False,
    )
    report = run_env_bench(spec, backend_factory=_mp_backend.plain_backend, log=lambda _m: None)
    assert report["ok"], report.get("traceback")
    assert report["frames"] == 6 and report["env_frames"] == 24 and report["env_fps"] > 0.0
    assert report["dolphin_fps"] == pytest.approx(report["env_fps"] / 4)
    assert set(report["step_ms"]) == {"mean", "p50", "p90", "p99", "max"}
    assert report["step_ms"]["max"] >= report["step_ms"]["p50"] > 0.0
    timing = report["timing"]
    assert timing["parent/steps"] == 6.0 and timing["env/steps"] == 6.0  # the warm-up is excluded
    assert timing["worker/messages"] == 6.0
    assert timing["parent/wait_last_ms"] >= timing["parent/wait_first_ms"] > 0.0
    assert report["resets"] == 0 and report["relaunches"] == 0 and report["faults"] == 0
    assert report["boot_s"] > 0.0 and report["warmup_s"] > 0.0 and report["duration_s"] > 0.0
    text = format_report(report)
    assert "env fps" in text and "cross-worker tail" in text and report["name"] in text
    assert json.dumps(report)  # JSON-able


def test_bench_chunked_and_in_process() -> None:
    chunked = EnvBenchSpec(
        num_envs=4,
        worker_processes=2,
        chunk_frames=2,
        hold=2,
        frames=4,
        warmup=2,
        players=("policy", "cpu"),
        stages=(25,),
        slippi_port=51810,
        mp_start_method="forkserver",
        check_iso=False,
    )
    report = run_env_bench(chunked, backend_factory=_mp_backend.plain_backend)
    assert report["ok"], report.get("traceback")
    assert report["frames"] == 4 and report["timing"]["parent/steps"] == 2.0  # two round trips
    assert report["timing"]["env/steps"] == 4.0 and report["timing"]["worker/messages"] == 2.0
    assert report["chunk_ms"]["mean"] == pytest.approx(2 * report["step_ms"]["mean"])
    in_process = EnvBenchSpec(
        num_envs=2,
        worker_processes=0,
        frames=3,
        warmup=1,
        controllers="neutral",
        players=("policy", "cpu"),
        stages=(25,),
        slippi_port=51820,
        check_iso=False,
    )
    backend = _mp_backend.plain_backend(env_config(in_process), 0)
    report = run_env_bench(in_process, backend=backend)
    assert report["ok"], report.get("traceback")
    assert report["timing"]["env/steps"] == 3.0 and "parent/steps" not in report["timing"]
    assert report["timing"]["env/advance_mean_ms"] > 0.0
    assert "cross-worker tail" not in format_report(report)
    failed = run_env_bench(
        EnvBenchSpec(num_envs=1, worker_processes=0, frames=1, check_iso=False, dolphin_path="/nonexistent")
    )
    assert not failed["ok"] and failed["error"] and "FAILED" in format_report(failed)


def test_parse_specs_builds_one_spec_per_size(tmp_path: Path) -> None:
    specs, args = parse_specs(
        [
            "--sizes",
            "24,48",
            "--workers",
            "4,8",
            "--chunks",
            "3",
            "--frames",
            "9",
            "--warmup",
            "3",
            "--players",
            "cpu",
        ]
    )
    assert [spec.num_envs for spec in specs] == [24, 48]
    assert [spec.worker_processes for spec in specs] == [4, 8]
    assert all(spec.chunk_frames == 3 and spec.players == ("policy", "cpu") for spec in specs)
    assert args.json is None
    specs, _ = parse_specs(["--sizes", "96", "--workers", "16", "--json", str(tmp_path / "out.json")])
    assert len(specs) == 1 and specs[0].name == "e96_w16_k1_self_random"
    with pytest.raises(SystemExit):
        parse_specs(["--sizes", "24,48", "--workers", "4,8,16"])
    mixed, _ = parse_specs(["--sizes", "24,48", "--workers", "4", "--shm", "0,1"])
    assert [spec.shared_memory for spec in mixed] == [False, True] and mixed[1].name.endswith("_shm")
    assert env_bench.DEFAULT_STAGES == (25, 24, 18, 26, 8, 6)
