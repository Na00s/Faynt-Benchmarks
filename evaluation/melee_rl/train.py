"""Training loop and launcher (PLAN.md §6 P3; slippi-ai ``rl/run_lib.py:677-728`` loop shape).

``python -m melee_rl.train --config configs/rl/<name>.toml [--steps N] [--checkpoint-dir DIR]
[--wandb-mode online|offline|disabled] [--device cpu|cuda] [--no-resume]``

:class:`Trainer` builds everything from an :class:`RLConfig`: the policy (random init seeded with
``policy.seed``, or a BC / RL / Ali training checkpoint), the teacher (a frozen copy of the initial
policy, a checkpoint, or none) with its checkpoint record, the value net, the ``PPOLearner``, the
environment (``[env]``: the dummy toy or ``DolphinEnv``, P5), the opponent (``cpu`` / ``self`` /
``other``; ``None`` for two-port self-play, P6a) and the ``RolloutWorker`` on
``opponent.trained_ports``.  ``[opponent] type = "mix"`` (P10) builds one environment and one worker
per group instead (``self`` / ``cpu<n>`` / ``other:<path>`` arms, and ``pool`` -- the run's own
``step_<n>.pt`` snapshots drawn by PFSP, :mod:`melee_rl.pool`, 11 Sep 2026; every group's trajectories feed
the same ``learner.step``, per-arm statistics land under ``arm/<name>/*``, and ``policy.init_value``
warm-starts the value net from the RL checkpoint the policy starts from).  A checkpoint init is
frozen as ``init.pt`` in the run directory at first launch and read from there ever after
(resumes included), so a mutable source path -- another run's ``best.pt`` -- cannot change what
this run started from even if that run continues or its files vanish.
One :meth:`Trainer.train_step` is slippi-ai's ``LearnerManager.step`` +
``run`` body: optional environment reset every ``reset_every_n_steps`` learner steps with
``burnin_steps_after_reset`` discarded rollouts, self-play refresh every ``opponent.update_interval``
steps, ``ppo.num_batches`` rollouts, ``learner.step`` (the burn-in ladder lives inside it), the PLAN.md
§8 loop metrics (``reward/*`` from the rollout frames, ``env/*``, ``timing/*``, ``step``,
``frames_seen`` = training frames, i.e. both perspectives in two-port mode while ``env/*`` counts
environment frames), logging, and ``latest.pt`` every ``save_interval_steps`` (+ ``step_<n>.pt``
every ``snapshot_interval_steps``).  Resume (``runtime.resume``): an existing ``latest.pt`` in the
checkpoint directory restores the learner (weights, both Adams, ladder position), the loop
counters, the RNG states (incl. the ``self`` / ``other`` opponent's generator) and rebuilds the
teacher from the checkpoint's ``teacher`` record; the environment restarts (nothing of Dolphin
could be saved anyway).  ``Trainer(..., on_save=callback)`` runs ``callback()`` after every
:meth:`Trainer.save` (the Modal functions hang their Volume commit on it, so a container restart
loses at most one save interval; BC-INIT-PLAN.md step 2.2, 24 Aug 2026).

Deterministic mode (``runtime.deterministic``) seeds torch, sets the thread count and enables
``torch.use_deterministic_algorithms`` for the duration of :meth:`Trainer.run` (restored after).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging as std_logging
import os
import shutil
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from melee_rl import checkpoint as checkpoint_lib
from melee_rl.actor import RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.config import (
    RLConfig,
    config_to_mapping,
    default_config_yaml,
    load_config,
    pipeline_enabled,
    resolve_model_config,
)
from melee_rl.env.async_env import AsyncEnvAdapter, AsyncEnvProtocol
from melee_rl.env.dolphin import DolphinEnvConfig
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.env.protocol import EnvProtocol
from melee_rl.env.timing import EnvTimingTracker
from melee_rl.evaluate import (
    BEST_NAME,
    HISTORY_NAME,
    EvalResult,
    Evaluator,
    format_eval_summary,
    trajectory_metrics,
)
from melee_rl.learner import PPOLearner
from melee_rl.logging import METRICS_JSONL, WANDB_MODES, MetricsLogger, format_summary
from melee_rl.opponents import (
    Opponent,
    SelfOpponent,
    arm_characters,
    make_group_opponent,
    make_opponent,
)
from melee_rl.pipeline import PipelinedRolloutWorker
from melee_rl.pool import PoolOpponent, pool_counts
from melee_rl.profiling import SectionProfiler, build_profiler
from melee_rl.protocol import PolicyProtocol
from melee_rl.slippi_ai_agent import CHARACTER_IDS
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueNet
from model import ModelConfig

LOG = std_logging.getLogger("melee_rl.train")

CHARACTER_NAMES: dict[int, str] = {character: name for name, character in CHARACTER_IDS.items()}
"""libmelee character id -> its name in the ``arm/<arm>/<name>/*`` metrics (17 Sep 2026)."""


def opponent_generator(opponent: object) -> torch.Generator | None:
    """The sampling generator an opponent keeps (a ``PolicyOpponent``'s, the slippi-ai port's), or ``None``:
    what RL checkpoints save and a resume restores, per mix arm."""

    generator = getattr(opponent, "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


RESUMABLE_MODEL_FIELDS: frozenset[str] = frozenset({"gradient_checkpointing"})
"""Model-config fields a resume may change (5 Sep 2026): gradient checkpointing trades memory for compute and
leaves the forward, the gradients and every logged number bit-identical (measured twice, sessions 45 and 47),
so a run may switch it off to take ``learner.compile`` mid-way; every other field still refuses the resume."""


@dataclass(frozen=True)
class RunOptions:
    """Command-line overrides of an :class:`RLConfig` (``None`` keeps the TOML value)."""

    num_steps: int | None = None
    checkpoint_dir: str | Path | None = None
    wandb_mode: str | None = None
    device: str | None = None
    resume: bool | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()


@dataclass
class TrainResult:
    """What :meth:`Trainer.run` returns (the tests and the launcher read it)."""

    config: RLConfig
    step: int
    steps_run: int
    frames_seen: int
    history: list[dict[str, float]]
    checkpoint: Path | None
    wandb_run_id: str | None
    run_name: str | None = None
    tags: tuple[str, ...] = ()
    evals: list[EvalResult] = field(default_factory=list)

    @property
    def last_metrics(self) -> dict[str, float] | None:
        return self.history[-1] if self.history else None


def build_policy(
    config: RLConfig,
    model_config: ModelConfig,
    device: torch.device,
    *,
    checkpoint: str | Path | None = None,
) -> MeleePolicyAdapter:
    """The student: a seeded random init, or the weights of a BC / RL checkpoint (``policy.init``).

    ``checkpoint`` overrides where the weights are read from (the trainer passes the frozen
    ``init.pt`` copy so a mutated or deleted source path cannot change what a run starts from, P10).
    """

    policy = config.policy
    if policy.init == "checkpoint":
        source = policy.checkpoint if checkpoint is None else checkpoint
        assert source is not None
        loaded = checkpoint_lib.load_bc_checkpoint(source)
        adapter = MeleePolicyAdapter(model_config)
        try:
            adapter.load_state_dict(loaded.state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(f"policy.checkpoint {source} does not fit the model config: {error}") from error
    else:
        torch.manual_seed(policy.seed)
        adapter = MeleePolicyAdapter(model_config)
    adapter.fast_step = policy.fast_step
    adapter.fast_attention = policy.fast_attention
    adapter.fast_encoder = policy.fast_encoder
    adapter.graph_step = policy.graph_step
    return adapter.to(device)


def _freeze(policy: MeleePolicyAdapter) -> MeleePolicyAdapter:
    policy.eval()
    policy.requires_grad_(False)
    return policy


def build_teacher(
    config: RLConfig,
    model_config: ModelConfig,
    policy: PolicyProtocol,
    device: torch.device,
    *,
    init_checkpoint: str | Path | None = None,
) -> tuple[PolicyProtocol | None, dict[str, Any] | None]:
    """``(teacher, record)`` per ``[teacher]``; the record is what the RL checkpoint stores
    (INTERFACE.md §8.2).

    ``init_checkpoint`` is the file the policy actually initialised from (the frozen ``init.pt``
    when a run directory exists, P10): a ``source = "policy"`` record pins that immutable file, so
    resumes survive the original ``policy.checkpoint`` path being overwritten or deleted.
    """

    teacher = config.teacher
    if teacher.source == "none":
        return None, None
    if teacher.source in ("policy", "ema"):
        if config.policy.init == "random":
            record: dict[str, Any] = {"random_init": True, "seed": config.policy.seed}
        else:
            source = config.policy.checkpoint if init_checkpoint is None else init_checkpoint
            assert source is not None
            record = {
                "source": str(source),
                "sha256": checkpoint_lib.sha256_file(source),
            }
        if teacher.source == "ema":
            # The anchor starts as the init and then tracks the policy (15 Sep 2026): its weights are learner
            # state, and tau is part of its identity (a resume under another tau is refused like another
            # file).
            assert teacher.ema_tau is not None
            record["ema_tau"] = float(teacher.ema_tau)
        return policy.clone_frozen(), _teacher_record(config, record)
    assert teacher.path is not None
    # A frozen teacher never backpropagates: gradient checkpointing would only cost recompute, and the
    # compiled lean forward (learner.compile) refuses it -- the file's own flag (on in Ali's training
    # files) is dropped.
    loaded = _freeze(checkpoint_lib.load_policy(teacher.path, device=device, gradient_checkpointing=False))
    if loaded.context_length < model_config.context_length:
        raise ValueError(
            f"teacher context_length {loaded.context_length} is shorter than the policy's "
            f"{model_config.context_length}"
        )
    record = {"source": str(teacher.path), "sha256": checkpoint_lib.sha256_file(teacher.path)}
    return loaded, _teacher_record(config, record)


def _teacher_record(config: RLConfig, record: dict[str, Any]) -> dict[str, Any]:
    """``record`` plus the teacher's own delay when the run sets one (``[teacher] delay_frames``, 11 Sep
    2026): the alignment is part of the teacher's identity, so a resume under another one is refused like
    another file; runs without the setting keep the record they always had."""

    if config.teacher.delay_frames is not None:
        record["delay_frames"] = int(config.teacher.delay_frames)
    return record


def build_reference(
    config: RLConfig, model_config: ModelConfig, device: torch.device
) -> PolicyProtocol | None:
    """The frozen ``[teacher] reference_path`` policy whose KL the learner logs at weight 0 (15 Sep 2026);
    ``None`` without one.  Not part of the teacher's record: a resume may add, drop or change it."""

    path = config.teacher.reference_path
    if path is None:
        return None
    loaded = _freeze(checkpoint_lib.load_policy(path, device=device, gradient_checkpointing=False))
    if loaded.context_length < model_config.context_length:
        raise ValueError(
            f"reference context_length {loaded.context_length} is shorter than the policy's "
            f"{model_config.context_length}"
        )
    return loaded


def teacher_from_record(
    record: Mapping[str, Any] | None,
    model_config: ModelConfig,
    device: torch.device,
) -> PolicyProtocol | None:
    """Rebuild the teacher of an RL checkpoint from its ``teacher`` record (the teacher is not learner
    state)."""

    if record is None:
        return None
    if record.get("random_init"):
        torch.manual_seed(int(record["seed"]))
        return _freeze(MeleePolicyAdapter(model_config).to(device))
    source = str(record["source"])
    digest = checkpoint_lib.sha256_file(source)
    if digest != record.get("sha256"):
        raise ValueError(
            f"teacher checkpoint {source} changed since the RL checkpoint was written (sha256 mismatch)"
        )
    return _freeze(checkpoint_lib.load_policy(source, device=device, gradient_checkpointing=False))


def build_env(config: RLConfig) -> EnvProtocol:
    """The environment of ``[env]``: the dummy toy, or ``DolphinEnv`` (libmelee is imported only then)."""

    if config.env.type == "dummy":
        return DummyMeleeEnv(config.env.dummy)
    if config.env.type == "dolphin":
        from melee_rl.env.dolphin_mp import build_dolphin_env

        return build_dolphin_env(config.env.dolphin, cpu_level=config.opponent.cpu_level)
    raise ValueError(f"env.type {config.env.type!r} is not supported")


def interleaved_rollouts(workers: Sequence[RolloutWorker]) -> list[Trajectory]:
    """One rollout per worker, the workers stepped frame by frame in alternation (lever 1, 5 Sep 2026).

    Each worker's generator runs up to the point where its frame's actions are on the way to its
    Dolphins; the next worker's frame then runs while those Dolphins emulate.  Every worker still sees
    exactly the frames it would see alone and draws from its own generator in the same order, so the
    trajectories are the sequential ones bit for bit.  The workers share the policy module, so one
    inference-mode context covers them all.
    """

    if not workers:
        return []
    for worker in workers:
        worker.split_steps = True
    with workers[0].inference_mode():
        active = [worker.rollout_steps() for worker in workers]
        while active:
            remaining = []
            for steps in active:
                try:
                    next(steps)
                except StopIteration:
                    continue
                remaining.append(steps)
            active = remaining
    return [worker.take_trajectory() for worker in workers]


def mix_env_configs(config: RLConfig) -> list[DummyEnvConfig | DolphinEnvConfig]:
    """One environment configuration per mix group (P10).

    Each group gets the active env table with its own ``num_envs`` and ``players``; Dolphin groups
    also get their ``mix_workers`` share and Slippi-port / replay-dir offsets by cumulative
    environment index, so the groups never collide.  Dummy groups offset the seed so equal-shaped
    arms play different games.
    """

    groups = config.opponent.groups
    if not groups:
        raise ValueError('mix_env_configs needs opponent.type = "mix"')
    configs: list[DummyEnvConfig | DolphinEnvConfig] = []
    start = 0
    if config.env.type == "dummy":
        for index, group in enumerate(groups):
            configs.append(
                replace(
                    config.env.dummy,
                    num_envs=group.num_envs,
                    players=group.players,
                    seed=config.env.dummy.seed + index,
                )
            )
            start += group.num_envs
        return configs
    workers = config.opponent.mix_workers or tuple(0 for _ in groups)
    dolphin = config.env.dolphin
    for group, group_workers in zip(groups, workers, strict=True):
        offset = dolphin.env_index_offset + start
        # 17 Sep 2026 (the league run): an arm whose opponent plays other characters carries player 1's
        # character per Dolphin, in the global-index table the worker slices keep (``opponent_characters``).
        characters = arm_characters(group, config.opponent.arm(group))
        table: tuple[int, ...] = ()
        if characters is not None and set(characters) != {dolphin.characters[1]}:
            table = (dolphin.characters[1],) * offset + characters
        configs.append(
            replace(
                dolphin,
                num_envs=group.num_envs,
                players=group.players,
                worker_processes=group_workers,
                slippi_port=dolphin.slippi_port + start,
                env_index_offset=offset,
                opponent_characters=table,
            )
        )
        start += group.num_envs
    return configs


def build_mix_envs(config: RLConfig) -> list[EnvProtocol]:
    """The environments of a mixed run, one per group (:func:`mix_env_configs` builds their configs)."""

    env_configs = mix_env_configs(config)
    envs: list[EnvProtocol] = []
    if config.env.type == "dummy":
        for sub in env_configs:
            assert isinstance(sub, DummyEnvConfig)
            envs.append(DummyMeleeEnv(sub))
        return envs
    from melee_rl.env.dolphin_mp import build_dolphin_env

    for group, sub in zip(config.opponent.groups, env_configs, strict=True):
        assert isinstance(sub, DolphinEnvConfig)
        envs.append(build_dolphin_env(sub, cpu_level=group.cpu_level))
    return envs


@contextlib.contextmanager
def runtime_settings(deterministic: bool, threads: int | None, device: torch.device) -> Iterator[None]:
    """Thread count and ``torch.use_deterministic_algorithms`` for the duration of a run (restored after)."""

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if threads is not None:
        torch.set_num_threads(threads)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=device.type != "cpu")
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


class Trainer:
    """Everything an RL run needs, built from an :class:`RLConfig`; see the module docstring."""

    def __init__(
        self,
        config: RLConfig,
        options: RunOptions | None = None,
        *,
        on_save: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.options = RunOptions() if options is None else options
        self.on_save = on_save
        runtime = config.runtime
        self.device = torch.device(self.options.device or runtime.device)
        self.model_config = resolve_model_config(config)
        self.checkpoint_dir = _optional_path(self.options.checkpoint_dir or runtime.checkpoint_dir)
        self.resume = runtime.resume if self.options.resume is None else self.options.resume
        # 5 Sep 2026: the step profiler (inert unless [profiling] names this run's steps).
        self.profiler: SectionProfiler = build_profiler(config.profiling, self.checkpoint_dir, self.device)

        self.init_checkpoint = self._freeze_init_checkpoint()
        self.policy = build_policy(config, self.model_config, self.device, checkpoint=self.init_checkpoint)
        self.teacher, self.teacher_record = build_teacher(
            config, self.model_config, self.policy, self.device, init_checkpoint=self.init_checkpoint
        )
        self.reference = build_reference(config, self.model_config, self.device)
        torch.manual_seed(runtime.seed)
        self.value_net = ValueNet(self.model_config, config.learner.value).to(self.device)
        if config.policy.init_value:
            assert self.init_checkpoint is not None
            checkpoint_lib.load_value_net(self.init_checkpoint, self.value_net)
        self.learner = PPOLearner(
            self.policy,
            self.teacher,
            self.value_net,
            config.learner,
            profiler=self.profiler,
            teacher_delay_frames=config.teacher.delay_frames,
            teacher_ema_tau=config.teacher.ema_tau if config.teacher.source == "ema" else None,
            reference=self.reference,
        )
        self.env: EnvProtocol
        self.opponent: Opponent | None
        self.worker: RolloutWorker
        if config.opponent.type == "mix":
            # P10: one environment and one rollout worker per mix group; every group's trajectories
            # feed the same learner step.  Lockstep only (finalize_config refuses mix + pipeline).
            self.group_names: tuple[str, ...] = tuple(group.name for group in config.opponent.groups)
            self.envs: list[EnvProtocol] = build_mix_envs(config)
            self.opponents: list[Opponent | None] = []
            self.workers: list[RolloutWorker] = []
            for index, (group, env) in enumerate(zip(config.opponent.groups, self.envs, strict=True)):
                opponent = make_group_opponent(
                    group,
                    config.opponent,
                    self.policy,
                    delay_frames=config.actor.delay_frames,
                    seed=config.opponent.seed + index,
                    device=self.device,
                    snapshot_dir=self.checkpoint_dir,
                    slippi=config.slippi_ai,
                )
                if isinstance(opponent, PoolOpponent) and self.checkpoint_dir is None:
                    LOG.warning(
                        "mix arm %s has no checkpoint directory: it plays the current self", group.name
                    )
                self.opponents.append(opponent)
                self.workers.append(
                    RolloutWorker(
                        self.policy,
                        env,
                        opponent,
                        replace(config.actor, seed=config.actor.seed + index),
                        ports=group.trained_ports,
                        profiler=self.profiler,
                        name=group.name,
                    )
                )
            self.env = self.envs[0]
            self.opponent = self.opponents[0]
            self.worker = self.workers[0]
            # 17 Sep 2026 (the league run): an arm whose Dolphins play several opponent characters gets its
            # statistics per character as well (arm/<arm>/<character>/*).
            self.arm_characters: list[tuple[int, ...] | None] = [
                arm_characters(group, config.opponent.arm(group)) for group in config.opponent.groups
            ]
        else:
            self.group_names = ()
            self.arm_characters = [None]
            self.env = build_env(config)
            self.opponent = make_opponent(
                config.opponent,
                self.policy,
                batch_size=self.env.num_envs,
                delay_frames=config.actor.delay_frames,
                device=self.device,
            )
            if pipeline_enabled(config):
                # P8: the delayed pipeline needs the asynchronous push/pop surface; WorkerDolphinEnv
                # is native, everything else (dummy, in-process Dolphin) is wrapped in the adapter.
                chunk = config.env.dolphin.chunk_frames if config.env.type == "dolphin" else 1
                if isinstance(self.env, AsyncEnvProtocol):
                    worker_env: EnvProtocol = self.env
                else:
                    worker_env = AsyncEnvAdapter(self.env, chunk)
                self.worker = PipelinedRolloutWorker(
                    self.policy,
                    worker_env,
                    self.opponent,
                    config.actor,
                    ports=config.opponent.trained_ports,
                    profiler=self.profiler,
                )
            else:
                self.worker = RolloutWorker(
                    self.policy,
                    self.env,
                    self.opponent,
                    config.actor,
                    ports=config.opponent.trained_ports,
                    profiler=self.profiler,
                )
            self.envs = [self.env]
            self.opponents = [self.opponent]
            self.workers = [self.worker]
        self.evaluator: Evaluator | None = (
            Evaluator(config.eval, config, self.model_config, self.device) if config.eval.enabled else None
        )
        self.best_eval: dict[str, Any] | None = None
        self.evals: list[EvalResult] = []
        self.evals_done = 0
        self._last_eval_step: int | None = None
        self._last_eval_time = time.perf_counter()

        self.step = 0
        self.frames_seen = 0
        self.rollouts_trained = 0
        self.resets = 0
        self.game_starts = 0
        self.opponent_refreshes = 0
        self.runtime_s_before = 0.0  # wall time of earlier launches (loop_state["runtime_s"])
        self._run_started: float | None = None
        self.resumed = False
        self._wandb_run_id: str | None = None
        self._latest: Path | None = None
        self._saved_step: int | None = None
        self._closed = False
        self._env_timing = EnvTimingTracker()  # lever 0: env_timing/<arm>/* per learner step
        if self.checkpoint_dir is not None and self.resume:
            latest = self.checkpoint_dir / checkpoint_lib.LATEST_NAME
            if latest.exists():
                self._restore(latest)
        jsonl_path = (
            self.checkpoint_dir / METRICS_JSONL
            if config.logging.jsonl and self.checkpoint_dir is not None
            else None
        )
        self.logger = MetricsLogger(
            config.logging,
            mode=self.options.wandb_mode,
            run_config=config_to_mapping(config),
            run_id=self._wandb_run_id,
            resume=self.resumed,
            name=self.options.run_name,
            tags=self.options.tags,
            jsonl_path=jsonl_path,
        )

    # -- facts --------------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, float]]:
        return self.logger.history

    @property
    def latest_checkpoint(self) -> Path | None:
        return self._latest

    # -- resume ----------------------------------------------------------------------------

    def _freeze_init_checkpoint(self) -> Path | None:
        """The file the policy (and value net) initialise from; ``None`` for a random init (P10).

        ``policy.checkpoint`` paths like ``best.pt`` / ``latest.pt`` are mutable -- resuming the
        source run overwrites them -- so the first launch copies the exact bytes into the run
        directory as ``init.pt`` and every later construction (resumes included) reads the frozen
        copy; the ``source = "policy"`` teacher record pins the frozen file too.  Without a
        checkpoint directory there is nowhere to freeze onto and the source is read directly.  A
        pre-``init.pt`` run that resumes never freezes late (by then the source may have moved on).
        """

        policy = self.config.policy
        if policy.init != "checkpoint":
            return None
        assert policy.checkpoint is not None
        source = Path(policy.checkpoint)
        if self.checkpoint_dir is None:
            return source
        frozen = self.checkpoint_dir / checkpoint_lib.INIT_NAME
        if frozen.exists():
            return frozen
        latest = self.checkpoint_dir / checkpoint_lib.LATEST_NAME
        if self.resume and latest.exists():
            return source
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temporary = frozen.with_name(frozen.name + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, frozen)
        temporary.unlink(missing_ok=True)  # Modal Volume v1 can keep a 0-byte ghost after os.replace
        LOG.info("froze policy.checkpoint %s as %s", source, frozen)
        return frozen

    def _restore(self, latest: Path) -> None:
        payload = checkpoint_lib.load_checkpoint(latest)
        saved_model = dict(payload["model_config"])
        current_model = asdict(self.model_config)
        if saved_model != current_model:
            keys = set(saved_model) | set(current_model)
            differing = sorted(
                key
                for key in keys
                if saved_model.get(key) != current_model.get(key) and key not in RESUMABLE_MODEL_FIELDS
            )
            if differing:
                raise ValueError(
                    f"{latest}: the checkpoint's model_config differs from this config's model on "
                    f"{differing}; use another checkpoint_dir or --no-resume"
                )
        saved_teacher = payload.get("teacher")
        if saved_teacher != self.teacher_record:
            raise ValueError(
                f"{latest}: the checkpoint's teacher record {saved_teacher!r} differs from this config's "
                f"{self.teacher_record!r}; use another checkpoint_dir or --no-resume"
            )
        self.step = checkpoint_lib.restore_learner(payload, self.learner)
        if saved_teacher is not None and saved_teacher.get("ema_tau") is not None:
            # An EMA anchor is learner state: restore_learner just put the checkpoint's anchor weights into
            # the teacher built at init (the same object keeps the lean-forward binding and the EMA pairs).
            assert self.teacher is not None and self.learner.teacher is self.teacher
        else:
            self.teacher = teacher_from_record(saved_teacher, self.model_config, self.device)
            self.learner.set_teacher(self.teacher)
        loop_state = payload.get("loop_state") or {}
        self.frames_seen = int(loop_state.get("frames_seen", 0))
        self.rollouts_trained = int(loop_state.get("rollouts_trained", 0))
        self.resets = int(loop_state.get("hard_resets", 0))
        self.game_starts = int(loop_state.get("game_starts", 0))
        self.opponent_refreshes = int(loop_state.get("opponent_refreshes", 0))
        self.runtime_s_before = float(loop_state.get("runtime_s", 0.0))
        best_eval = loop_state.get("best_eval")
        self.best_eval = dict(best_eval) if isinstance(best_eval, Mapping) else None
        self.evals_done = int(loop_state.get("evals_done", 0))
        last_eval_step = loop_state.get("last_eval_step")
        self._last_eval_step = None if last_eval_step is None else int(last_eval_step)
        rng = payload.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        actors = rng.get("actors")
        if actors is not None:
            # P10: a mixed run stores one actor generator per rollout worker, in group order.
            if len(actors) != len(self.workers):
                raise ValueError(
                    f"{latest}: the checkpoint has {len(actors)} actor generators but this config builds "
                    f"{len(self.workers)} rollout workers; use another checkpoint_dir or --no-resume"
                )
            for worker, state in zip(self.workers, actors, strict=True):
                if state is not None:
                    worker.generator.set_state(state)
        elif rng.get("actor") is not None:
            self.worker.generator.set_state(rng["actor"])
        opponent_states = rng.get("opponents")
        if opponent_states is not None:
            if len(opponent_states) != len(self.opponents):
                raise ValueError(
                    f"{latest}: the checkpoint has {len(opponent_states)} opponent generators but this "
                    f"config builds {len(self.opponents)}; use another checkpoint_dir or --no-resume"
                )
            for opponent, state in zip(self.opponents, opponent_states, strict=True):
                generator = opponent_generator(opponent)
                if state is not None and generator is not None:
                    generator.set_state(state)
        elif rng.get("opponent") is not None:
            generator = opponent_generator(self.opponent)
            if generator is not None:
                generator.set_state(rng["opponent"])
        pools = loop_state.get("pools")
        if pools is not None:
            if len(pools) != len(self.opponents):
                raise ValueError(
                    f"{latest}: the checkpoint has {len(pools)} pool states but this config builds "
                    f"{len(self.opponents)} opponents; use another checkpoint_dir or --no-resume"
                )
            for opponent, state in zip(self.opponents, pools, strict=True):
                if isinstance(opponent, PoolOpponent) and state is not None:
                    opponent.restore(state)  # the refresh below then reloads the snapshot it was playing
        for opponent in self.opponents:
            if opponent is not None:
                opponent.refresh(self.policy)  # the self-play clone was built from the pre-restore weights
        self._wandb_run_id = payload.get("wandb_run_id")
        self._latest = latest
        self._saved_step = self.step
        self.resumed = True
        LOG.info("resumed from %s at step %d (%d frames seen)", latest, self.step, self.frames_seen)

    # -- one learner step --------------------------------------------------------------------

    def _reset_env(self) -> None:
        for worker in self.workers:
            worker.reset_env()
        self.resets += 1
        for _ in range(self.config.runtime.burnin_steps_after_reset):
            for worker in self.workers:
                worker.rollout()  # discarded: refills the history ring, as slippi-ai's burn-in unrolls

    @property
    def interleaved(self) -> bool:
        """Lever 1: the mix arms roll out frame by frame in alternation (``[actor] interleave_arms``)."""

        return bool(self.config.actor.interleave_arms) and len(self.workers) > 1

    def _env_timing_metrics(self) -> dict[str, float]:
        """``env_timing/<arm>/*`` from the Dolphin environments' cumulative timing counters (lever 0,
        5 Sep 2026): the delta since the previous learner step; the first call is the baseline.  Only in
        lockstep mode -- the pipelined worker keeps chunks in flight between rollouts."""

        if pipeline_enabled(self.config):
            return {}
        if self.group_names:
            named: list[tuple[str, object]] = list(zip(self.group_names, self.envs, strict=True))
        else:
            named = [("env", self.env)]
        return self._env_timing.update(named)

    def train_step(self) -> dict[str, float]:
        """Rollouts, ``learner.step``, loop metrics, logging and checkpointing for one learner step."""

        if self._closed:
            raise RuntimeError("the trainer is closed")
        runtime, ppo = self.config.runtime, self.config.learner.ppo
        profiler = self.profiler
        profiler.begin_step(self.step)
        started = time.perf_counter()
        interval = runtime.reset_every_n_steps
        if interval is not None and self.step > 0 and self.step % interval == 0:
            self._reset_env()
        if self.step % self.config.opponent.update_interval == 0:
            for opponent in self.opponents:
                if isinstance(opponent, SelfOpponent):
                    opponent.refresh(self.policy)
                    self.opponent_refreshes += 1
        rollout_step = self.step
        for opponent in self.opponents:
            if isinstance(opponent, PoolOpponent):
                opponent.maybe_draw(rollout_step, self.policy)  # a past self every [opponent.pool] interval
        trajectories: list[Trajectory] = []
        learner_batches: list[Trajectory] = []  # what the learner steps on: the arms, or each rollout joined
        group_trajectories: list[list[Trajectory]] = [[] for _ in self.workers]
        timings = {"rollout_s": 0.0, "reprime_s": 0.0, "policy_s": 0.0, "env_s": 0.0, "opponent_s": 0.0}
        arm_timings = [dict(timings) for _ in self.workers]
        with profiler.section("loop/rollouts"):
            for _ in range(ppo.num_batches):
                if self.interleaved:
                    batch = interleaved_rollouts(self.workers)  # lever 1: the arms frame by frame
                else:
                    batch = [worker.rollout() for worker in self.workers]
                if self.config.learner.concat_arms and len(batch) > 1:
                    # 17 Sep 2026: one learner trajectory per rollout; the arms' copies feed only the per-arm
                    # metrics, so they drop their prefix caches (the joined one holds them).
                    learner_batches.append(Trajectory.batch(batch))
                    batch = [replace(trajectory, initial_cache=None) for trajectory in batch]
                else:
                    learner_batches.extend(batch)
                for index, (worker, trajectory) in enumerate(zip(self.workers, batch, strict=True)):
                    trajectories.append(trajectory)
                    group_trajectories[index].append(trajectory)
                    for name in timings:
                        seconds = worker.timings.get(name, 0.0)
                        timings[name] += seconds
                        arm_timings[index][name] += seconds
        with profiler.section("loop/env_timing"):
            env_timing = self._env_timing_metrics()
        with profiler.section("loop/learner"):
            metrics = self.learner.step(learner_batches)
        self.step += 1
        stalling = self.config.actor.reward.stalling_threshold
        # Training frames: every trained perspective counts (two-port self-play doubles them);
        # env/fps and env/frames_total count environment frames.
        frames = sum(trajectory.batch_size * trajectory.rollout_length for trajectory in trajectories)
        total_envs = sum(env.num_envs for env in self.envs)
        total_rows = sum(worker.batch_size for worker in self.workers)
        env_frames = ppo.num_batches * self.config.actor.rollout_length * total_envs
        self.frames_seen += frames
        self.rollouts_trained += len(trajectories)
        with profiler.window("loop/metrics"):
            if not self.group_names:
                metrics.update(
                    trajectory_metrics(
                        trajectories, ports=len(self.worker.ports), stalling_threshold=stalling
                    )
                )
            else:
                # P10: pooled reward/* over every trained row, per-arm copies under arm/<name>/*, and
                # env/resets counted per environment inside each group (the ports division is per group).
                combined = trajectory_metrics(trajectories, ports=1, stalling_threshold=stalling)
                resets = 0.0
                for name, worker, group, characters in zip(
                    self.group_names, self.workers, group_trajectories, self.arm_characters, strict=True
                ):
                    arm = trajectory_metrics(group, ports=len(worker.ports), stalling_threshold=stalling)
                    resets += arm["env/resets"]
                    for key, value in arm.items():
                        metrics[f"arm/{name}/{key.split('/', 1)[1]}"] = value
                    if characters is not None and len(set(characters)) > 1 and len(worker.ports) == 1:
                        for character in sorted(set(characters)):
                            rows = [index for index, value in enumerate(characters) if value == character]
                            split = trajectory_metrics(group, ports=1, stalling_threshold=stalling, rows=rows)
                            label = CHARACTER_NAMES.get(character, f"char{character}")
                            for key, value in split.items():
                                metrics[f"arm/{name}/{label}/{key.split('/', 1)[1]}"] = value
                combined["env/resets"] = resets
                metrics.update(combined)
                # Lever 0 (5 Sep 2026): the arms roll out one after the other, so each arm's own
                # rollout / policy / Dolphin-wait seconds are what a per-arm lever would change.
                for name, arm_timing in zip(self.group_names, arm_timings, strict=True):
                    for key, value in arm_timing.items():
                        metrics[f"timing/{name}/{key}"] = value
                # The past-self arms: this step's stocks into the KO-share table, then the pool keys
                # (the snapshot played, its age, the KO shares -- the minimum is the forgetting measure).
                for name, opponent, group in zip(
                    self.group_names, self.opponents, group_trajectories, strict=True
                ):
                    if isinstance(opponent, PoolOpponent):
                        opponent.record(*pool_counts(group))
                        for key, value in opponent.metrics(rollout_step).items():
                            metrics[f"arm/{name}/{key}"] = value
        metrics.update(env_timing)
        self.game_starts += int(metrics["env/resets"])
        elapsed = time.perf_counter() - started
        metrics.update(
            {
                "timing/rollout_s": timings["rollout_s"],
                "timing/reprime_s": timings["reprime_s"],
                "timing/policy_s": timings["policy_s"],
                "timing/env_s": timings["env_s"],
                "timing/opponent_s": timings["opponent_s"],
                "timing/step_s": elapsed,
                "timing/runtime_total_s": self.runtime_s,
                "env/fps": env_frames / elapsed if elapsed > 0.0 else 0.0,
                "env/frames_total": float(sum(worker.frames_seen for worker in self.workers)),
                "env/resets_total": float(self.game_starts),
                "env/hard_resets": float(self.resets),
                "env/trained_ports": float(total_rows / total_envs),
                "step": float(self.step),
                "frames_seen": float(self.frames_seen),
            }
        )
        with profiler.section("loop/log"):
            self.logger.log(metrics, step=self.step)
            every = self.config.logging.print_interval_steps
            if every and self.step % every == 0:
                print(format_summary(metrics), flush=True)
        if self.checkpoint_dir is not None and self.step % runtime.save_interval_steps == 0:
            with profiler.window("loop/save"):
                self.save()
        if profiler.end_step(metrics) and self.on_save is not None:
            self.on_save()  # the profile files of this step become durable with the checkpoints
        return metrics

    # -- evaluation (P6b item 0) --------------------------------------------------------------------

    def _eval_due(self) -> bool:
        """After a learner step: is an evaluation due by ``interval_steps`` or ``interval_seconds``?"""

        if self.evaluator is None:
            return False
        settings = self.config.eval
        if settings.interval_steps is not None and self.step % settings.interval_steps == 0:
            return True
        if settings.interval_seconds is not None:
            return time.perf_counter() - self._last_eval_time >= settings.interval_seconds
        return False

    def evaluate_now(self) -> EvalResult:
        """Evaluate the current policy: log ``eval/*`` at this step, update ``best.pt`` and the history."""

        if self.evaluator is None:
            raise RuntimeError("evaluation is not enabled ([eval] enabled = true)")
        best_path = None if self.checkpoint_dir is None else self.checkpoint_dir / BEST_NAME
        result = self.evaluator.evaluate(self.policy, step=self.step, best_path=best_path)
        self.evals.append(result)
        self.evals_done += 1
        self._last_eval_step = self.step
        self._last_eval_time = time.perf_counter()
        value = result.selector
        improved = value is not None and (self.best_eval is None or value > float(self.best_eval["value"]))
        if improved:
            assert value is not None
            self.best_eval = {
                "step": int(self.step),
                "metric": result.metric,
                "metric_opponent": result.metric_opponent,
                "value": float(value),
            }
        metrics = result.metrics()
        if self.best_eval is not None:
            metrics["eval/best_value"] = float(self.best_eval["value"])
            metrics["eval/best_step"] = float(self.best_eval["step"])
        metrics["eval/improved"] = float(improved)
        self.logger.log(metrics, step=self.step)
        if self.config.logging.print_interval_steps:
            print(format_eval_summary(result), flush=True)
        if self.checkpoint_dir is not None:
            self._append_eval_history({**result.to_record(), "improved": improved, "best": self.best_eval})
            if improved:
                self._write_checkpoint(self.checkpoint_dir / BEST_NAME)
            self.save()  # latest.pt carries the eval bookkeeping (best_eval, evals_done) for a resume
        return result

    def _append_eval_history(self, record: Mapping[str, Any]) -> None:
        assert self.checkpoint_dir is not None
        path = self.checkpoint_dir / HISTORY_NAME
        records: list[Any] = []
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            records = list(loaded) if isinstance(loaded, list) else []
        records.append(dict(record))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(records, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    def _maybe_evaluate_at_start(self) -> None:
        if (
            self.evaluator is not None
            and self.config.eval.at_start
            and self.step == 0
            and self.evals_done == 0
        ):
            self.evaluate_now()

    def _maybe_evaluate_at_end(self) -> None:
        if self.evaluator is not None and self.config.eval.at_end and self._last_eval_step != self.step:
            self.evaluate_now()

    # -- checkpoints ---------------------------------------------------------------------------

    def _write_checkpoint(self, path: Path) -> Path:
        """Write the full RL checkpoint of the current state to ``path``."""

        rng: dict[str, Any] = {"torch": torch.get_rng_state(), "actor": self.worker.generator.get_state()}
        single = opponent_generator(self.opponent)
        if single is not None:
            rng["opponent"] = single.get_state()
        if len(self.workers) > 1:
            # P10: one actor generator per rollout worker (group order); every arm whose opponent samples
            # keeps its generator (PolicyOpponent arms, and the slippi-ai port's since 17 Sep 2026).
            rng["actors"] = [worker.generator.get_state() for worker in self.workers]
            generators = [opponent_generator(opponent) for opponent in self.opponents]
            rng["opponents"] = [
                None if generator is None else generator.get_state() for generator in generators
            ]
        loop_state: dict[str, Any] = {
            "frames_seen": self.frames_seen,
            "rollouts_trained": self.rollouts_trained,
            "hard_resets": self.resets,
            "game_starts": self.game_starts,
            "opponent_refreshes": self.opponent_refreshes,
            "runtime_s": self.runtime_s,
            "best_eval": self.best_eval,
            "evals_done": self.evals_done,
            "last_eval_step": self._last_eval_step,
        }
        pools = [
            opponent.pool_state() if isinstance(opponent, PoolOpponent) else None
            for opponent in self.opponents
        ]
        if any(state is not None for state in pools):
            # The past-self arms: their KO-share tables, draw generators and the snapshots they play.
            loop_state["pools"] = pools
        config_yaml = self.config.policy.config_yaml or default_config_yaml()
        checkpoint_lib.save_rl_checkpoint(
            path,
            self.learner,
            self.model_config,
            step=self.step,
            rl_config=config_to_mapping(self.config),
            teacher=self.teacher_record,
            delay_frames=self.config.actor.delay_frames,
            context_mode=self.config.actor.context_mode,
            rng=rng,
            wandb_run_id=self.logger.run_id,
            config_yaml=config_yaml,
            loop_state=loop_state,
        )
        return path

    def save(self, *, snapshot: bool | None = None) -> Path:
        """Write ``latest.pt`` (and ``step_<n>.pt`` when ``snapshot``; default: the snapshot interval),
        then run ``on_save`` once (after every file of this save is on disk)."""

        if self.checkpoint_dir is None:
            raise RuntimeError("no checkpoint_dir is configured")
        runtime = self.config.runtime
        latest, periodic = checkpoint_lib.checkpoint_paths(self.checkpoint_dir, self.step)
        self._write_checkpoint(latest)
        if snapshot is None:
            interval = runtime.snapshot_interval_steps
            snapshot = bool(interval) and self.step % interval == 0
        if snapshot:
            shutil.copyfile(latest, periodic)
        self._latest = latest
        self._saved_step = self.step
        if self.on_save is not None:
            self.on_save()
        return latest

    # -- the run ----------------------------------------------------------------------------------

    def run(self, num_steps: int | None = None) -> TrainResult:
        """``num_steps`` learner steps (default: the CLI ``--steps``, then ``runtime.num_steps``)."""

        if self._closed:
            raise RuntimeError("the trainer is closed")
        runtime = self.config.runtime
        steps = self.options.num_steps if num_steps is None else num_steps
        steps = runtime.num_steps if steps is None else steps
        if steps < 0:
            raise ValueError("num_steps must be >= 0")
        started = time.perf_counter()
        self._run_started = started
        ran = 0
        evals_before = len(self.evals)
        with runtime_settings(runtime.deterministic, runtime.threads, self.device):
            self._last_eval_time = time.perf_counter()
            self._maybe_evaluate_at_start()
            for _ in range(steps):
                limit = runtime.max_runtime_s
                if limit is not None and time.perf_counter() - started > limit:
                    LOG.info("max_runtime_s %.0f reached after %d steps", runtime.max_runtime_s, ran)
                    break
                total = runtime.max_total_runtime_s
                if total is not None and self.runtime_s > total:
                    LOG.info(
                        "max_total_runtime_s %.0f reached after %d steps (%.0f s across launches)",
                        total,
                        ran,
                        self.runtime_s,
                    )
                    break
                self.train_step()
                ran += 1
                if self._eval_due():
                    self.evaluate_now()
            self._maybe_evaluate_at_end()
            if self.checkpoint_dir is not None and ran and self._saved_step != self.step:
                self.save()
        self.runtime_s_before += time.perf_counter() - started
        self._run_started = None
        return TrainResult(
            config=self.config,
            step=self.step,
            steps_run=ran,
            frames_seen=self.frames_seen,
            history=list(self.logger.history),
            checkpoint=self._latest,
            wandb_run_id=self.logger.run_id,
            run_name=self.logger.name,
            tags=self.logger.tags,
            evals=list(self.evals[evals_before:]),
        )

    @property
    def runtime_s(self) -> float:
        """Wall time spent in :meth:`run` across every launch of this run directory (earlier launches
        through ``loop_state["runtime_s"]``, this one live) -- what ``max_total_runtime_s`` caps."""

        current = 0.0 if self._run_started is None else time.perf_counter() - self._run_started
        return self.runtime_s_before + current

    def close(self) -> None:
        """Finish the W&B run and close the environment (idempotent)."""

        if self._closed:
            return
        self.logger.finish()
        for env in self.envs:
            env.close()
        self._closed = True


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m melee_rl.train", description="RL post-training for MeleePolicy"
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML run file (configs/rl/*.toml)")
    parser.add_argument(
        "--steps", type=int, default=None, help="learner steps to run (default: [runtime] num_steps)"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="where latest.pt / step_<n>.pt go")
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default=None, help="override [logging] mode")
    parser.add_argument("--device", default=None, help="override [runtime] device (cpu, cuda, cuda:0)")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="ignore an existing latest.pt"
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="W&B run name (default: [logging] name, else rl-<config>-<UTC time>)",
    )
    parser.add_argument(
        "--wandb-tag",
        dest="wandb_tags",
        action="append",
        default=[],
        help="extra W&B tag (repeatable; the config file's stem and the automatic tags are always added)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="TABLE.KEY=VALUE",
        help="override one run-file value (repeatable; TOML literal or bare string), e.g. "
        "--set policy.fast_attention=lean --set learner.ppo.num_batches=4",
    )
    return parser.parse_args(argv)


def default_run_name(config_path: Path, when: datetime | None = None) -> str:
    """``rl-<config stem>-<UTC yyyymmdd-HHMMSS>``."""

    stamp = (datetime.now(UTC) if when is None else when).strftime("%Y%m%d-%H%M%S")
    return f"rl-{config_path.stem}-{stamp}"


def main(argv: Sequence[str] | None = None, *, on_save: Callable[[], None] | None = None) -> TrainResult:
    """The launcher: parse ``argv``, build the :class:`Trainer` (``on_save`` passed through) and run it."""

    args = parse_args(argv)
    std_logging.basicConfig(level=std_logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config(args.config, overrides=args.overrides)
    run_name = args.wandb_name or config.logging.name or default_run_name(Path(args.config))
    options = RunOptions(
        num_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        wandb_mode=args.wandb_mode,
        device=args.device,
        resume=None if args.resume else False,
        run_name=run_name,
        tags=(Path(args.config).stem, *args.wandb_tags),
    )
    trainer = Trainer(config, options, on_save=on_save)
    try:
        return trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()


__all__ = [
    "RunOptions",
    "TrainResult",
    "Trainer",
    "build_env",
    "build_mix_envs",
    "build_policy",
    "build_reference",
    "build_teacher",
    "default_run_name",
    "main",
    "mix_env_configs",
    "opponent_generator",
    "parse_args",
    "runtime_settings",
    "teacher_from_record",
    "trajectory_metrics",
]
