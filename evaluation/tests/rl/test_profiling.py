"""``melee_rl.profiling`` (5 Sep 2026, the throughput investigation): the ``[profiling]`` table, the
trace summary, and the step profiler's windows and sections inside a real trainer run."""

from __future__ import annotations

import gzip
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch

from melee_rl.config import (
    CONFIG_DIR,
    RLConfig,
    config_from_mapping,
    finalize_config,
    resolve_model_config,
    resolve_paths,
)
from melee_rl.profiling import (
    NULL_PROFILER,
    NullProfiler,
    ProfilingConfig,
    StepProfiler,
    build_profiler,
    kernel_family,
    summarize_events,
)
from melee_rl.train import RunOptions, Trainer

SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
IGNORED_ON_COMPARE = ("timing/", "env/fps")


def _comparable(row: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in row.items() if not key.startswith(IGNORED_ON_COMPARE)}


def _tiny_config(**profiling: Any) -> RLConfig:
    """``smoke_tiny.toml`` with a ``[profiling]`` table spliced in (the run file itself stays unprofiled)."""

    with SMOKE_TINY.open("rb") as stream:
        raw = tomllib.load(stream)
    if profiling:
        raw["profiling"] = profiling
    config = resolve_paths(config_from_mapping(raw), SMOKE_TINY.parent)
    return finalize_config(config, resolve_model_config(config))


# -- the config table ----------------------------------------------------------------------------


def test_profiling_config_defaults_and_validation() -> None:
    default = ProfilingConfig()
    assert not default.enabled and default.steps == () and default.dir == "profile"
    assert default.actor_frames == 32 and default.learner_chunks == 4 and default.memory is True
    enabled = ProfilingConfig(enabled=True, steps=(2, 3))
    assert enabled.limit("actor/frames/self") == 32
    assert enabled.limit("learner/chunks/train") == 4
    assert enabled.limit("learner/optimizer") == 1 and enabled.limit("loop/save") == 1
    with pytest.raises(ValueError, match="steps"):
        ProfilingConfig(enabled=True)
    with pytest.raises(ValueError, match="steps"):
        ProfilingConfig(steps=(-1,))
    with pytest.raises(ValueError, match="actor_frames"):
        ProfilingConfig(actor_frames=0)
    with pytest.raises(ValueError, match="learner_chunks"):
        ProfilingConfig(learner_chunks=0)
    with pytest.raises(ValueError, match="dir"):
        ProfilingConfig(dir="")


def test_profiling_table_is_parsed_and_defaults_to_off() -> None:
    plain = _tiny_config()
    assert plain.profiling == ProfilingConfig()
    config = _tiny_config(enabled=True, steps=[1], actor_frames=3, learner_chunks=2, dir="prof")
    assert config.profiling == ProfilingConfig(
        enabled=True, steps=(1,), actor_frames=3, learner_chunks=2, dir="prof"
    )
    with pytest.raises(ValueError, match="unknown key"):
        _tiny_config(enabled=True, steps=[1], frames=3)


def test_build_profiler_is_inert_when_disabled_and_needs_a_directory_when_enabled(tmp_path: Path) -> None:
    device = torch.device("cpu")
    assert build_profiler(ProfilingConfig(), None, device) is NULL_PROFILER
    assert isinstance(NULL_PROFILER, NullProfiler)
    NULL_PROFILER.begin_step(3)
    with NULL_PROFILER.section("anything"), NULL_PROFILER.window("actor/frames/x"):
        pass
    assert NULL_PROFILER.end_step({"timing/step_s": 1.0}) is False and not NULL_PROFILER.active
    enabled = ProfilingConfig(enabled=True, steps=(0,))
    with pytest.raises(ValueError, match="checkpoint"):
        build_profiler(enabled, None, device)
    profiler = build_profiler(enabled, tmp_path / "run", device)
    assert isinstance(profiler, StepProfiler) and profiler.output_dir == tmp_path / "run" / "profile"
    absolute = build_profiler(
        ProfilingConfig(enabled=True, steps=(0,), dir=str(tmp_path / "elsewhere")), None, device
    )
    assert isinstance(absolute, StepProfiler) and absolute.output_dir == tmp_path / "elsewhere"


# -- the trace summary ----------------------------------------------------------------------------------


def test_kernel_family_classifies_by_name() -> None:
    assert kernel_family("void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>") == "matmul"
    assert kernel_family("ampere_sgemm_128x64_nn") == "matmul"
    assert kernel_family("void gemv2N_kernel<...>") == "matmul"
    assert kernel_family("fmha_cutlassF_f32_aligned_64x64_rf_sm80") == "attention"
    assert kernel_family("void at::native::(anonymous namespace)::softmax_warp_forward<...>") == "reduction"
    assert kernel_family("void at::native::reduce_kernel<512, 1, ...>") == "reduction"
    assert kernel_family("void at::native::vectorized_elementwise_kernel<4, ...>") == "elementwise"
    assert kernel_family("void at::native::index_put_kernel_impl<...>") == "elementwise"
    assert kernel_family("Memcpy HtoD (Pageable -> Device)") == "copy"
    assert kernel_family("Memset (Device)") == "copy"
    assert kernel_family("void something_unusual<...>") == "other"


def _event(
    cat: str, name: str, ts: float, dur: float, **args: Any
) -> dict[str, Any]:  # one chrome-trace "X" event
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": args}


def test_summarize_events_counts_launches_syncs_copies_and_busy_time() -> None:
    events = [
        _event("user_annotation", "actor/step_sample", 0.0, 100.0),
        _event("user_annotation", "actor/step_sample", 200.0, 100.0),
        _event("cpu_op", "aten::mm", 10.0, 5.0),
        _event("cpu_op", "aten::_local_scalar_dense", 20.0, 1.0),
        _event("cpu_op", "aten::mm", 210.0, 5.0),
        _event("cuda_runtime", "cudaLaunchKernel", 12.0, 2.0, correlation=1),
        _event("cuda_runtime", "cudaStreamSynchronize", 21.0, 30.0, correlation=2),
        _event("cuda_runtime", "cudaLaunchKernel", 212.0, 2.0, correlation=3),
        _event("cuda_runtime", "cudaMemcpyAsync", 230.0, 2.0, correlation=4),
        _event("cuda_runtime", "cudaMemcpyAsync", 240.0, 2.0, correlation=5),
        # two kernels overlapping in time count once for the busy share; one is outside any annotation
        _event("kernel", "ampere_sgemm_128x64_nn", 50.0, 40.0, correlation=1),
        _event("kernel", "void at::native::vectorized_elementwise_kernel<4>", 70.0, 40.0, correlation=99),
        _event("kernel", "ampere_sgemm_128x64_nn", 250.0, 10.0, correlation=3),
        _event("gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 260.0, 5.0, correlation=4, bytes=1024),
        _event("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", 270.0, 5.0, correlation=5, bytes=512),
        {"ph": "M", "name": "process_name", "args": {"name": "python"}},  # metadata rows are ignored
    ]
    summary = summarize_events(events, occurrences=2, wall_s=0.5)
    assert summary["occurrences"] == 2 and summary["wall_s"] == 0.5
    assert summary["kernels"] == 3 and summary["device_events"] == 5
    assert summary["kernel_time_s"] == pytest.approx((40 + 40 + 10) * 1e-6)
    assert summary["device_time_s"] == pytest.approx((40 + 40 + 10 + 5 + 5) * 1e-6)
    # busy time is the union of device intervals: [50,110) + [250,260) + [260,265) + [270,275) = 80 us
    assert summary["busy_s"] == pytest.approx(80e-6)
    assert summary["busy_share"] == pytest.approx(80e-6 / 0.5)
    assert summary["by_family"] == {
        "matmul": pytest.approx(50e-6),
        "elementwise": pytest.approx(40e-6),
        "copy": pytest.approx(10e-6),
    }
    assert summary["aten_ops"] == 3
    assert summary["syncs"] == {"stream": 1, "device": 0, "event": 0, "local_scalar": 1, "d2h": 1}
    assert summary["memcpy"] == {
        "h2d_count": 1,
        "h2d_bytes": 1024,
        "h2d_s": pytest.approx(5e-6),
        "d2h_count": 1,
        "d2h_bytes": 512,
        "d2h_s": pytest.approx(5e-6),
        "d2d_count": 0,
        "d2d_bytes": 0,
        "d2d_s": 0.0,
    }
    per = summary["per_occurrence"]
    assert per["kernels"] == 1.5 and per["aten_ops"] == 1.5 and per["syncs"] == 1.0
    assert per["wall_ms"] == pytest.approx(250.0) and per["device_ms"] == pytest.approx(50e-3)
    # Annotations: kernels are attributed through their launch's correlation id; the sgemm launched at
    # 12 us belongs to the first section, the one at 212 us to the second; the elementwise kernel has
    # no launch record and belongs to none.
    section = summary["annotations"]["actor/step_sample"]
    assert section["count"] == 2 and section["wall_s"] == pytest.approx(200e-6)
    assert section["device_s"] == pytest.approx((40 + 10 + 5 + 5) * 1e-6)
    assert section["kernels"] == 2 and section["aten_ops"] == 3 and section["syncs"] == 2
    assert section["per_call"]["wall_ms"] == pytest.approx(0.1)
    assert section["per_call"]["kernels"] == 1.0


def test_summarize_events_without_device_events_reports_zero_busy_time() -> None:
    events = [
        _event("user_annotation", "learner/epoch/forward", 0.0, 10.0),
        _event("cpu_op", "aten::add", 1.0, 1.0),
    ]
    summary = summarize_events(events, occurrences=1, wall_s=1e-5)
    assert summary["kernels"] == 0 and summary["busy_s"] == 0.0 and summary["busy_share"] == 0.0
    assert summary["by_family"] == {} and summary["aten_ops"] == 1
    assert summary["annotations"]["learner/epoch/forward"]["device_s"] == 0.0


# -- the step profiler in a real run -------------------------------------------------------------------


def test_step_profiler_traces_the_configured_step_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = _tiny_config(enabled=True, steps=[1], actor_frames=3, learner_chunks=2)
    trainer = Trainer(config, RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled"))
    assert isinstance(trainer.profiler, StepProfiler)
    assert trainer.worker._profiler is trainer.profiler and trainer.learner._profiler is trainer.profiler
    try:
        trainer.run(3)
    finally:
        trainer.close()
    profile_dir = run_dir / "profile"
    summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    assert [record["step"] for record in summary["steps"]] == [1]
    record = summary["steps"][0]
    assert record["timing"]["timing/step_s"] > 0.0 and record["timing"]["step"] == 2.0
    assert record["metrics"]["loss/total"] == pytest.approx(record["metrics"]["loss/total"])
    assert {"loss/actor_kl_pre", "pre/teacher_kl", "timing/learner_s"} <= set(record["metrics"])
    windows = record["windows"]
    batches = config.learner.ppo.num_batches
    frames = windows["actor/frames/worker"]
    assert frames["occurrences"] == 3 and frames["limit"] == 3 and frames["wall_s"] > 0.0
    assert (
        frames["processing_s"] >= 0.0
    )  # the export / parse / gzip time, reported so timers can be corrected
    assert frames["aten_ops"] > 0 and frames["kernels"] == 0  # CPU run: no device events
    assert set(frames["annotations"]) >= {
        "actor/observe",
        "actor/write",
        "actor/step_sample",
        "actor/actions",
        "actor/env_step",
        "actor/reward",
    }
    assert frames["annotations"]["actor/step_sample"]["count"] == 3
    # One chunk per trajectory here (microbatch_envs None), so the prepare window is suspended by the
    # first trajectory's tail window after one occurrence; its later chunks are counted, not traced.
    prepare = windows["learner/chunks/prepare"]
    assert prepare["occurrences"] == 1 and prepare["calls"] == batches and prepare["limit"] == 2
    assert windows["learner/chunks/train"]["occurrences"] == 2  # both trajectories' chunks of epoch 0
    assert windows["learner/chunks/post"]["occurrences"] == 2
    train = windows["learner/chunks/train"]["annotations"]
    assert {"learner/epoch/forward", "learner/epoch/loss", "learner/epoch/backward"} <= set(train)
    assert "learner/epoch/accumulate" in train and train["learner/epoch/forward"]["count"] == 2
    for name in ("actor/snapshot/worker", "actor/finish/worker", "learner/optimizer", "learner/snapshot"):
        assert windows[name]["occurrences"] == 1, name
    assert windows["learner/prepare/tail"]["occurrences"] == 1
    assert windows["learner/drift"]["occurrences"] == 1 and windows["loop/metrics"]["occurrences"] == 1
    assert windows["loop/save"]["occurrences"] == 1  # save_interval_steps = 1 in smoke_tiny
    sections = record["sections"]
    assert sections["actor/step_sample"]["calls"] == batches * config.actor.rollout_length
    assert sections["loop/rollouts"]["calls"] == 1 and sections["loop/learner"]["calls"] == 1
    assert sections["learner/prepare/validate"]["calls"] == batches
    assert all(item["wall_s"] >= 0.0 for item in sections.values())
    assert record["memory"]["device"] == "cpu"
    # One gzipped chrome trace per window, readable and carrying the annotations.
    step_dir = profile_dir / "step_00000001"
    traces = sorted(step_dir.glob("*.json.gz"))
    assert {path.name for path in traces} >= {"actor_frames_worker.json.gz", "learner_chunks_train.json.gz"}
    with gzip.open(step_dir / "actor_frames_worker.json.gz", "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    names = {event.get("name") for event in trace["traceEvents"]}
    assert "actor/step_sample" in names and not (step_dir / "actor_frames_worker.json").exists()
    markdown = (profile_dir / "summary.md").read_text(encoding="utf-8")
    assert "actor/frames/worker" in markdown and "learner/chunks/train" in markdown
    assert "actor/step_sample" in markdown


def test_profiling_changes_nothing_but_the_files(tmp_path: Path) -> None:
    """The profiler consumes no randomness and touches no tensor: the same run with and without it."""

    def run(config: RLConfig, run_dir: Path) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
        trainer = Trainer(config, RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled"))
        try:
            result = trainer.run(2)
            return result.history, trainer.policy.get_state()
        finally:
            trainer.close()

    plain_history, plain_state = run(_tiny_config(), tmp_path / "plain")
    profiled_history, profiled_state = run(
        _tiny_config(enabled=True, steps=[0, 1], actor_frames=2, learner_chunks=1), tmp_path / "profiled"
    )
    assert [_comparable(row) for row in plain_history] == [_comparable(row) for row in profiled_history]
    assert plain_state.keys() == profiled_state.keys()
    assert all(torch.equal(plain_state[key], profiled_state[key]) for key in plain_state)
    assert not (tmp_path / "plain" / "profile").exists()
    summary = json.loads((tmp_path / "profiled" / "profile" / "summary.json").read_text(encoding="utf-8"))
    assert [record["step"] for record in summary["steps"]] == [0, 1]


def test_windows_close_themselves_at_the_end_of_a_step_and_never_nest(tmp_path: Path) -> None:
    profiler = StepProfiler(
        ProfilingConfig(enabled=True, steps=(0,), actor_frames=5), tmp_path / "prof", torch.device("cpu")
    )
    assert not profiler.active
    profiler.begin_step(0)
    assert profiler.active
    # Occurrence 1 of 5: the window stays open afterwards.
    with profiler.window("actor/frames/a"), profiler.section("actor/step_sample"):
        torch.ones(2) + 1
    assert profiler.open_window == "actor/frames/a"
    with profiler.window("learner/optimizer"):  # another window closes the open one first
        torch.ones(2) * 2
    assert profiler.open_window is None
    for _ in range(6):
        with profiler.window("learner/chunks/train"):
            torch.ones(2) * 3
    assert profiler.open_window is None  # closed after the 4th occurrence (the default limit)
    assert profiler.end_step({"timing/step_s": 0.1}) is True and not profiler.active
    record = profiler.records[-1]
    assert record["windows"]["actor/frames/a"]["occurrences"] == 1
    assert record["windows"]["learner/chunks/train"]["occurrences"] == 4
    assert record["windows"]["learner/chunks/train"]["calls"] == 6
    assert record["windows"]["learner/optimizer"]["occurrences"] == 1
    profiler.begin_step(1)  # not a profiled step
    assert not profiler.active
    with profiler.window("actor/frames/a"), profiler.section("x"):
        pass
    assert profiler.end_step({}) is False and len(profiler.records) == 1
