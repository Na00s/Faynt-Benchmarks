"""Metric names and the Weights & Biases wrapper (PLAN.md §8, ENV.md §7; R14).

The metric names are fixed here: :data:`LEARNER_METRIC_NAMES` are emitted by
``PPOLearner.step`` (P2), :data:`LOOP_METRIC_NAMES` by the training loop (``melee_rl.train``,
P3); together they are PLAN.md §8 (:data:`METRIC_NAMES`).  The loop adds a few documented
extras (:data:`EXTRA_METRIC_NAMES`) -- the environment's own reward signal, KOs per minute,
cumulative counters, the remaining timings and the burn-in stage -- and the learner's
``pre/*`` / ``ppo_step/{i}/*`` diagnostics pass through unchanged.

:class:`MetricsLogger` wraps ``wandb`` behind the three modes ``online`` / ``offline`` /
``disabled``; the mode is resolved as *explicit override > TOML > ``WANDB_MODE`` > disabled*
(project, entity and directory follow the same order with ``WANDB_PROJECT``, ``WANDB_ENTITY``,
``WANDB_DIR``; the API key is read by ``wandb`` itself and never by this package).  In
``disabled`` mode ``wandb`` is not even imported (``import wandb`` costs about 15 s on the
dev container's Windows mount, ENV.md §4) and the logger only keeps its in-memory
:attr:`MetricsLogger.history`, which the tests and :class:`melee_rl.train.TrainResult` read.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from melee_rl.learner import STAGES

WandbMode = Literal["online", "offline", "disabled"]
WANDB_MODES: Final[tuple[str, ...]] = ("online", "offline", "disabled")
DEFAULT_PROJECT: Final[str] = "melee-rl"
AUTO_TAGS: Final[tuple[str, ...]] = ("rl", "melee_rl")
"""Tags every run carries, so RL runs are separable from other runs in a shared W&B project."""

LEARNER_METRIC_NAMES: Final[tuple[str, ...]] = (
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
    "timing/learner_s",
    "opt/lr_policy",
    "opt/lr_value",
    "weights/drift_step",
    "weights/drift_total",
)
"""PLAN.md §8 names produced by ``PPOLearner.step``."""

LOOP_METRIC_NAMES: Final[tuple[str, ...]] = (
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
    "step",
    "frames_seen",
)
"""PLAN.md §8 names produced by the training loop."""

METRIC_NAMES: Final[tuple[str, ...]] = LEARNER_METRIC_NAMES + LOOP_METRIC_NAMES
"""Every name of PLAN.md §8 (``missing_metrics`` checks a logged row against it)."""

EXTRA_METRIC_NAMES: Final[tuple[str, ...]] = (
    "reward/env_per_frame",
    "reward/kos_per_minute",
    "reward/ledge_grabs_per_minute",
    "reward/stalling_fraction",
    "reward/approaching_factor",
    "env/resets_total",
    "env/hard_resets",
    "env/trained_ports",
    "timing/policy_s",
    "timing/env_s",
    "timing/step_s",
    "ppo/stage",
    "opt/entropy_weight",
    "teacher/ema_tau",
    "loss/reference_kl",
    "loss/reverse_reference_kl",
)
"""Loop extras beyond PLAN.md §8 (``ppo/stage``, ``opt/entropy_weight``, ``teacher/ema_tau`` and the two
``loss/*reference_kl`` names come from the learner -- the adaptive levers of 15 Sep 2026, zero when off; the
three ``reward/*`` P6a names
are slippi-ai's remaining ``player_stats`` -- bad ledge grabs per minute, the fraction of frames stalling
offstage, the mean approaching factor -- of the trained player; ``env/trained_ports`` is trained rows per
environment: 1, 2, or fractional for a mixed run).  A mixed run (P10) additionally logs per-arm copies of
the reward statistics under ``arm/<group>/*`` -- variable names, so they are not registered here (they
pass through like the learner's ``pre/*`` / ``ppo_step/{i}/*`` diagnostics)."""


def _clean_tags(tags: Sequence[str], where: str) -> tuple[str, ...]:
    cleaned: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"{where} tags must be non-empty strings, got {tag!r}")
        if tag not in cleaned:
            cleaned.append(tag)
    return tuple(cleaned)


@dataclass(frozen=True)
class LoggingConfig:
    """``[logging]``: W&B settings (``None`` defers to the environment variable, then the default).

    ``group`` / ``tags`` / ``job_type`` separate our runs from other people's in a shared project
    (the shipped configs use ``group = "rl-post-training"``, ``job_type = "rl"`` and per-config tags;
    :data:`AUTO_TAGS` are always added).
    """

    mode: str | None = None
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    job_type: str | None = "rl"
    dir: str | None = None
    print_interval_steps: int = 1
    jsonl: bool = False
    """Append every logged row to ``metrics.jsonl`` in the run directory (5 Sep 2026: the record of a run
    without W&B -- the bench A/Bs read ``timing/*``, ``loss/actor_kl_pre`` and ``pre/teacher_kl`` from it)."""

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in WANDB_MODES:
            raise ValueError(f"logging.mode must be one of {WANDB_MODES}, got {self.mode!r}")
        if self.print_interval_steps < 0:
            raise ValueError("logging.print_interval_steps must be >= 0 (0 disables the console summary)")
        _clean_tags(self.tags, "logging.")

    def resolved_mode(self, override: str | None = None) -> str:
        """``override`` > ``mode`` > ``$WANDB_MODE`` > ``"disabled"``."""

        if override is not None:
            mode = override
        elif self.mode is not None:
            mode = self.mode
        else:
            mode = os.environ.get("WANDB_MODE", "disabled")
        if mode not in WANDB_MODES:
            raise ValueError(f"W&B mode must be one of {WANDB_MODES}, got {mode!r}")
        return mode

    def resolved_project(self) -> str:
        return self.project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT

    def resolved_entity(self) -> str | None:
        return self.entity or os.environ.get("WANDB_ENTITY") or None

    def resolved_dir(self) -> str | None:
        return self.dir or os.environ.get("WANDB_DIR") or None


METRICS_JSONL: Final[str] = "metrics.jsonl"


class MetricsLogger:
    """Per-step metric rows to W&B (``online`` / ``offline``) and to :attr:`history`."""

    def __init__(
        self,
        config: LoggingConfig,
        *,
        mode: str | None = None,
        run_config: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        resume: bool = False,
        keep_history: bool = True,
        name: str | None = None,
        tags: Sequence[str] = (),
        jsonl_path: str | Path | None = None,
    ) -> None:
        """``name`` overrides ``config.name``; ``tags`` follow :data:`AUTO_TAGS` and ``config.tags``;
        ``jsonl_path`` appends every row as one JSON line (``{"step": n, ...row}``)."""

        self.config = config
        self.jsonl_path = None if jsonl_path is None else Path(jsonl_path)
        self._mode = config.resolved_mode(mode)
        self._keep_history = keep_history
        self.name = config.name if name is None else name
        self.tags = _clean_tags((*AUTO_TAGS, *config.tags, *tags), "run")
        self.group = config.group
        self.job_type = config.job_type
        self.history: list[dict[str, float]] = []
        self._finished = False
        self._run: Any = None
        if self._mode != "disabled":
            import wandb  # imported only when a W&B run is wanted (ENV.md §4: ~15 s on this mount)

            self._run = wandb.init(
                project=config.resolved_project(),
                entity=config.resolved_entity(),
                name=self.name,
                group=self.group,
                job_type=self.job_type,
                tags=list(self.tags),
                dir=config.resolved_dir(),
                mode=cast(WandbMode, self._mode),
                config=None if run_config is None else dict(run_config),
                id=run_id,
                resume="allow" if (run_id and resume) else None,
            )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def backend(self) -> Any:
        """The ``wandb`` run object, or ``None`` in ``disabled`` mode."""

        return self._run

    @property
    def run_id(self) -> str | None:
        return None if self._run is None else str(self._run.id)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Record one row (values coerced to ``float``) at W&B step ``step``."""

        if self._finished:
            raise RuntimeError("the metrics logger is finished")
        row = {name: float(value) for name, value in metrics.items()}
        if self._keep_history:
            self.history.append(row)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": int(step), **row}) + "\n")
        if self._run is not None:
            self._run.log(row, step=int(step))

    def finish(self) -> None:
        """Close the W&B run (idempotent)."""

        if self._run is not None and not self._finished:
            self._run.finish()
        self._finished = True


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def format_summary(metrics: Mapping[str, float]) -> str:
    """One console line per learner step (the loop prints it every ``print_interval_steps``).

    A mixed run's per-arm rewards (``arm/<name>/per_frame``, P10) are appended after the pooled
    ``reward/frame`` -- the pooled number dilutes the CPU arm with the zero-sum self-play rows -- and a
    past-self arm's snapshot (``arm/<name>/snapshot_step``, ``melee_rl.pool``; -1 = the current self).
    """

    step = metrics.get("step")
    reverted = metrics.get("ppo/reverted")
    stage = metrics.get("ppo/stage")
    stage_name = "-" if stage is None else STAGES[int(stage)]
    parts = [
        f"step {'-' if step is None else int(step)}",
        f"fps {_fmt(metrics.get('env/fps'), '.0f')}",
        f"reward/frame {_fmt(metrics.get('reward/per_frame'), '.4f')}",
    ]
    for key in sorted(metrics):
        # arm/<name>/per_frame only: an arm's per-character rows (arm/<name>/<character>/per_frame) stay in
        # the metrics and out of the console line.
        if key.startswith("arm/") and key.endswith("/per_frame") and key.count("/") == 2:
            parts.append(f"{key.split('/', 2)[1]} r/f {format(metrics[key], '.4f')}")
        elif key.startswith("arm/") and key.endswith("/snapshot_step"):
            played = int(metrics[key])
            parts.append(f"{key.split('/', 2)[1]} vs {'self' if played < 0 else f'step {played}'}")
    parts += [
        f"ko/min {_fmt(metrics.get('reward/ko_diff_per_minute'), '.2f')}",
        f"loss {_fmt(metrics.get('loss/total'), '.4f')}",
        f"actor_kl {_fmt(metrics.get('loss/actor_kl_mean'), '.2e')}",
        f"teacher_kl {_fmt(metrics.get('loss/teacher_kl'), '.2e')}",
        f"entropy {_fmt(metrics.get('loss/entropy'), '.2f')}",  # of ln 728 + ln 85 = 11.03 (BC-INIT-PLAN.md)
        f"uev {_fmt(metrics.get('value/uev'), '.2f')}",
        f"clip {_fmt(metrics.get('ppo/clip_fraction'), '.3f')}",
        f"drift {_fmt(metrics.get('weights/drift_total'), '.2e')}",
        f"reverted {'-' if reverted is None else int(reverted)}",
        f"stage {stage_name}",
        f"{_fmt(metrics.get('timing/step_s'), '.2f')} s",
    ]
    return " | ".join(parts)


def missing_metrics(metrics: Mapping[str, float]) -> list[str]:
    """PLAN.md §8 names absent from ``metrics`` (empty for a complete row)."""

    return [name for name in METRIC_NAMES if name not in metrics]


__all__ = [
    "AUTO_TAGS",
    "DEFAULT_PROJECT",
    "EXTRA_METRIC_NAMES",
    "LEARNER_METRIC_NAMES",
    "LOOP_METRIC_NAMES",
    "METRIC_NAMES",
    "WANDB_MODES",
    "LoggingConfig",
    "MetricsLogger",
    "WandbMode",
    "format_summary",
    "missing_metrics",
]
