"""Metrics and W&B wrapper (PLAN.md §8, §6 P3): fixed names, disabled mode, summary line, metric keys."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from melee_rl.config import CONFIG_DIR, load_config
from melee_rl.logging import (
    AUTO_TAGS,
    EXTRA_METRIC_NAMES,
    LEARNER_METRIC_NAMES,
    LOOP_METRIC_NAMES,
    METRIC_NAMES,
    WANDB_MODES,
    LoggingConfig,
    MetricsLogger,
    format_summary,
    missing_metrics,
)
from melee_rl.train import RunOptions, Trainer

# PLAN.md §8, verbatim.
PLAN_SECTION_8 = (
    "loss/total",
    "loss/ppo_objective",
    "loss/teacher_kl",
    "loss/reverse_teacher_kl",
    "loss/actor_kl_mean",
    "loss/actor_kl_max",
    "loss/actor_kl_pre",
    "loss/entropy",
    "ppo/log_rho_mean",
    "ppo/log_rho_abs_max",
    "ppo/clip_fraction",
    "ppo/reverted",
    "ppo/epochs",
    "value/loss",
    "value/uev",
    "value/return_mean",
    "value/value_mean",
    "reward/per_frame",
    "reward/ko_diff_per_minute",
    "reward/damage_dealt_per_minute",
    "reward/damage_taken_per_minute",
    "reward/deaths_per_minute",
    "env/fps",
    "env/frames_total",
    "env/resets",
    "timing/rollout_s",
    "timing/reprime_s",
    "timing/learner_s",
    "opt/lr_policy",
    "opt/lr_value",
    "weights/drift_step",
    "weights/drift_total",
    "step",
    "frames_seen",
)


def test_metric_names_match_plan_section_8() -> None:
    assert set(METRIC_NAMES) == set(PLAN_SECTION_8)
    assert len(set(METRIC_NAMES)) == len(METRIC_NAMES)
    assert set(LEARNER_METRIC_NAMES) | set(LOOP_METRIC_NAMES) == set(METRIC_NAMES)
    assert not set(LEARNER_METRIC_NAMES) & set(LOOP_METRIC_NAMES)
    learner_names = {"loss/total", "ppo/reverted", "value/uev", "opt/lr_policy", "timing/learner_s"}
    assert learner_names <= set(LEARNER_METRIC_NAMES)
    assert {"reward/per_frame", "env/fps", "env/resets", "timing/rollout_s", "step", "frames_seen"} <= set(
        LOOP_METRIC_NAMES
    )
    assert WANDB_MODES == ("online", "offline", "disabled")
    # P6a extras: the remaining slippi-ai reward statistics and the number of trained ports.
    for name in (
        "reward/ledge_grabs_per_minute",
        "reward/stalling_fraction",
        "reward/approaching_factor",
        "env/trained_ports",
    ):
        assert name in EXTRA_METRIC_NAMES and name not in METRIC_NAMES
    assert len(set(EXTRA_METRIC_NAMES)) == len(EXTRA_METRIC_NAMES) and not set(EXTRA_METRIC_NAMES) & set(
        METRIC_NAMES
    )


def test_disabled_logger_history_summary_and_mode_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"):
        monkeypatch.delenv(name, raising=False)
    config = LoggingConfig()
    assert config.resolved_mode() == "disabled" and config.resolved_project() == "melee-rl"
    assert config.resolved_entity() is None and config.resolved_dir() is None
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    monkeypatch.setenv("WANDB_ENTITY", "someone")
    monkeypatch.setenv("WANDB_DIR", "/somewhere")
    assert config.resolved_mode() == "offline" and config.resolved_project() == "from-env"
    assert config.resolved_entity() == "someone" and config.resolved_dir() == "/somewhere"
    # The TOML beats the environment, an explicit override beats both.
    explicit = LoggingConfig(mode="online", project="p", entity="e", dir="d")
    assert explicit.resolved_mode() == "online" and explicit.resolved_mode("disabled") == "disabled"
    assert explicit.resolved_project() == "p" and explicit.resolved_entity() == "e"
    assert explicit.resolved_dir() == "d"
    assert config.resolved_mode("disabled") == "disabled"
    with pytest.raises(ValueError, match="mode"):
        LoggingConfig(mode="sometimes")
    with pytest.raises(ValueError, match="mode"):
        config.resolved_mode("bogus")
    with pytest.raises(ValueError, match="print_interval_steps"):
        LoggingConfig(print_interval_steps=-1)

    logger = MetricsLogger(config, mode="disabled", run_config={"a": 1})
    assert logger.mode == "disabled" and logger.backend is None and logger.run_id is None
    logger.log({"loss/total": 1.0, "step": 1.0}, step=1)
    logger.log({"loss/total": 0.5, "step": 2.0}, step=2)
    assert [row["step"] for row in logger.history] == [1.0, 2.0]
    assert logger.history[-1]["loss/total"] == 0.5 and logger.history[-1]["step"] == 2.0
    logger.finish()
    logger.finish()  # idempotent
    with pytest.raises(RuntimeError, match="finished"):
        logger.log({"loss/total": 0.0}, step=3)
    forgetful = MetricsLogger(config, mode="disabled", keep_history=False)
    forgetful.log({"loss/total": 1.0}, step=1)
    assert forgetful.history == []

    metrics = {
        "step": 3.0,
        "env/fps": 1234.5,
        "reward/per_frame": -0.0012,
        "loss/total": 0.1234,
        "loss/actor_kl_mean": 1.2e-5,
        "loss/teacher_kl": 0.001,
        "loss/entropy": 10.934,
        "value/uev": 0.98,
        "ppo/clip_fraction": 0.01,
        "ppo/reverted": 0.0,
        "ppo/stage": 2.0,
        "timing/step_s": 1.5,
    }
    summary = format_summary(metrics)
    assert summary.startswith("step 3 ") and "fps 1234" in summary and "ppo" in summary
    assert "reverted 0" in summary and "uev 0.98" in summary and "1.5" in summary
    # Entropy on the console line (BC-INIT-PLAN.md 2.3): the from-scratch run's collapse from 10.94 to
    # 3.5 nats was only visible in W&B; the maximum is ln 728 + ln 85 = 11.03.
    assert "entropy 10.93" in summary and summary.index("teacher_kl") < summary.index("entropy")
    assert "entropy -" in format_summary({"step": 1.0})
    assert format_summary({}).startswith("step -")  # missing values print as "-"

    assert missing_metrics(dict.fromkeys(METRIC_NAMES, 0.0)) == []
    missing = missing_metrics({"loss/total": 1.0})
    assert "loss/total" not in missing and "env/fps" in missing and len(missing) == len(METRIC_NAMES) - 1


def test_run_identity_tags_name_and_job_type() -> None:
    """Our W&B runs are separable from others in the shared project: automatic tags, group, job type."""

    assert AUTO_TAGS == ("rl", "melee_rl")
    assert LoggingConfig().job_type == "rl" and LoggingConfig().group is None
    config = LoggingConfig(
        entity="frisson-labs",
        project="Faynt",
        group="rl-post-training",
        tags=("ppo", "tiny"),
        job_type="rl",
    )
    logger = MetricsLogger(config, mode="disabled", name="run-1", tags=("extra", "ppo"))
    assert logger.name == "run-1"
    assert logger.tags == ("rl", "melee_rl", "ppo", "tiny", "extra")  # automatic first, then TOML, then extra
    assert logger.job_type == "rl" and logger.group == "rl-post-training"
    plain = MetricsLogger(config, mode="disabled")
    assert plain.name is None and plain.tags == ("rl", "melee_rl", "ppo", "tiny")
    named = MetricsLogger(LoggingConfig(name="from-toml"), mode="disabled")
    assert named.name == "from-toml" and named.tags == AUTO_TAGS
    with pytest.raises(ValueError, match="tags"):
        LoggingConfig(tags=("ppo", ""))
    with pytest.raises(ValueError, match="tags"):
        MetricsLogger(config, mode="disabled", tags=(" ",))


def test_metrics_keys_present() -> None:
    """One training step of the tiny smoke config logs every PLAN.md §8 name (+ the guard flag and fps)."""

    config = load_config(CONFIG_DIR / "smoke_tiny.toml")
    trainer = Trainer(config, RunOptions(wandb_mode="disabled"))
    try:
        metrics = trainer.train_step()
    finally:
        trainer.close()
    assert missing_metrics(metrics) == [], missing_metrics(metrics)
    batch = config.env.num_envs * config.actor.rollout_length * config.learner.ppo.num_batches
    assert metrics["step"] == 1.0 and metrics["frames_seen"] == float(batch)
    assert metrics["env/frames_total"] == float(batch) and metrics["env/fps"] > 0.0
    assert metrics["ppo/reverted"] in (0.0, 1.0) and metrics["ppo/epochs"] == 2.0
    assert metrics["timing/rollout_s"] > 0.0 and metrics["timing/reprime_s"] >= 0.0
    assert metrics["timing/learner_s"] > 0.0 and metrics["timing/step_s"] >= metrics["timing/learner_s"]
    for name in ("reward/per_frame", "reward/ko_diff_per_minute", "reward/damage_dealt_per_minute"):
        assert isinstance(metrics[name], float)
    assert metrics["reward/env_per_frame"] == 0.0  # no learnable signal in the smoke config
    assert metrics["env/resets"] >= 0.0 and metrics["env/resets_total"] >= metrics["env/resets"]
    assert set(EXTRA_METRIC_NAMES) <= set(metrics)
    assert metrics["env/trained_ports"] == 1.0
    # The toy never grabs a ledge and never leaves the stage box (unknown stage id -> edge 100, y >= 0).
    assert metrics["reward/ledge_grabs_per_minute"] == 0.0 and metrics["reward/stalling_fraction"] == 0.0
    assert isinstance(metrics["reward/approaching_factor"], float)
    assert trainer.logger.history == [metrics]
    assert isinstance(format_summary(metrics), str)


@pytest.mark.skipif(
    not os.environ.get("MELEE_RL_TEST_WANDB"),
    reason="offline W&B run: set MELEE_RL_TEST_WANDB=1 (imports wandb, about 15 s on this mount)",
)
def test_offline_wandb_run_writes_files(tmp_path: Path) -> None:
    config = LoggingConfig(project="melee-rl-test", dir=str(tmp_path), tags=("test",), name="offline-smoke")
    logger = MetricsLogger(config, mode="offline", run_config={"a": 1, "nested": {"b": 2.0}})
    try:
        assert logger.mode == "offline" and logger.backend is not None and logger.run_id
        logger.log({"loss/total": 1.0, "step": 1.0}, step=1)
        logger.log({"loss/total": 0.5, "step": 2.0}, step=2)
    finally:
        logger.finish()
    assert any(tmp_path.rglob("*.wandb")), sorted(str(path) for path in tmp_path.rglob("*"))[:20]
    assert logger.history[-1]["loss/total"] == 0.5


def test_jsonl_file_records_every_row(tmp_path: Path) -> None:
    """5 Sep 2026 (/rl-speedup): ``[logging] jsonl = true`` appends every logged row to ``metrics.jsonl``."""

    import json

    from melee_rl.logging import METRICS_JSONL

    assert METRICS_JSONL == "metrics.jsonl" and not LoggingConfig().jsonl
    path = tmp_path / "run" / METRICS_JSONL
    logger = MetricsLogger(LoggingConfig(mode="disabled", jsonl=True), jsonl_path=path)
    logger.log({"timing/step_s": 1.5, "loss/actor_kl_pre": 1e-9}, step=1)
    logger.log({"timing/step_s": 1.25, "eval/cpu9/damage_dealt_per_minute": 100}, step=2)
    logger.finish()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"step": 1, "timing/step_s": 1.5, "loss/actor_kl_pre": 1e-9},
        {"step": 2, "timing/step_s": 1.25, "eval/cpu9/damage_dealt_per_minute": 100.0},
    ]
    assert MetricsLogger(LoggingConfig(mode="disabled")).jsonl_path is None
    config = load_config(CONFIG_DIR / "smoke_tiny.toml")
    trainer = Trainer(
        replace(config, logging=replace(config.logging, jsonl=True)),
        RunOptions(checkpoint_dir=tmp_path / "trainer", wandb_mode="disabled"),
    )
    try:
        trainer.run(2)
    finally:
        trainer.close()
    lines = (tmp_path / "trainer" / METRICS_JSONL).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    assert "timing/step_s" in json.loads(lines[0]) and "loss/actor_kl_pre" in json.loads(lines[0])
