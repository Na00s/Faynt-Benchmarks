"""TOML -> frozen dataclasses for an RL run (PLAN.md §3.4, §3.5, §6 P3; R13).

A run file is parsed with the stdlib ``tomllib`` into :class:`RLConfig`, a tree of the frozen
dataclasses the package already has (``ActorConfig``, ``OpponentConfig``, ``DummyEnvConfig``,
``LearnerConfig`` with ``ReturnsConfig`` / ``ValueConfig`` / ``PPOConfig``, ``RewardConfig``,
``LoggingConfig``) plus the P3 tables ``[runtime]``, ``[policy]``, ``[teacher]``, ``[env]`` and
``[delay]``::

    format = "melee_rl.rl_config.v1"
    [runtime]   num_steps, seed, deterministic, device, threads, checkpoint_dir, resume,
                save_interval_steps, snapshot_interval_steps, reset_every_n_steps,
                burnin_steps_after_reset, max_runtime_s
    [policy]    config_yaml (default: Ali's config.yaml), profile, init = "random" | "checkpoint",
                checkpoint, seed, compute_dtype, cache_dtype, [policy.model_overrides]
    [teacher]   source = "policy" | "checkpoint" | "ema" | "none", path, delay_frames (the teacher's own
                delay, 11 Sep 2026: unset = the run's actor.delay_frames, 0 = the hindsight teacher; never
                above the run's), ema_tau (source = "ema", 15 Sep 2026: the anchor is an exponential moving
                average of the policy, updated after every accepted PPO step), reference_path (a frozen file
                whose KL is logged at weight 0: the distance-from-anchor read once the anchor itself moves)
    [env]       type = "dummy" | "dolphin", [env.dummy] (DummyEnvConfig), [env.dolphin] (DolphinEnvConfig,
                P5); the active table's ``players`` defaults from the opponent
    [actor]     ActorConfig, [actor.reward] RewardConfig
    [opponent]  OpponentConfig
    [learner]   LearnerConfig, [learner.returns] [learner.value] [learner.ppo] [learner.reward]
    [delay]     allow_mismatch
    [logging]   LoggingConfig
    [eval]      EvalConfig (P6b: the in-loop evaluator; off by default)
    [mimic]     MimicConfig (P12: the released MIMIC policy as an evaluation opponent)
    [smashbot]  SmashBotConfig (P25: altf4/SmashBot, the expert system, as an evaluation opponent)
    [slippi_ai] SlippiAIConfig (P27: vladfi1/slippi-ai's released agents as evaluation opponents)
    [phillip]   PhillipConfig (P32: vladfi1/phillip's released agents as evaluation opponents)
    [profiling] ProfilingConfig (5 Sep 2026: torch.profiler windows of the named learner steps)
    [video]     VideoConfig, [video.render] RenderConfig (P9: the on-demand clip recorder; its
                ``rollout_length`` is checked against the *recorded checkpoint*, not this run file's
                profile, so ``finalize_config`` leaves it alone)

The model config stays Ali's ``config.yaml`` (PLAN.md §3.4): :func:`resolve_model_config` loads
it with ``ModelConfig.from_yaml(path, profile)``, applies the RL precision and the
``model_overrides`` (``dataclasses.replace``).  Keys are checked exactly (unknown keys, wrong
types and wrong-length arrays are errors), relative paths resolve against the TOML's directory,
and :func:`finalize_config` applies the cross-field rules: ``delay_frames ==
action_offset_frames - 1`` unless ``[delay] allow_mismatch`` (PLAN.md §3.5), ``C + T <=
context_length``, ``microbatch_envs`` divides the learner batch ``num_envs * len(trained ports)``
(two-port self-play doubles the rows, P6a), bf16 forces ``ppo.behaviour_logits = "learner"``
(PLAN.md §3.9), a random-init teacher needs zero teacher-KL weights (PLAN.md §5, §9.4) and the
active environment table's ``players`` must match ``opponent.type`` (``cpu`` -> ``("policy",
"cpu")``; ``self`` / ``other`` -> ``("policy", "policy")``).  ``[opponent]`` carries the P6a keys
``train`` (self-play on both ports) and ``checkpoint`` (the ``other`` opponent's file, TOML-relative),
and the P10 keys ``mix`` / ``mix_workers`` (the mixed curriculum: per-group environments whose counts
must sum to ``num_envs``, per-group worker splits, lockstep only; a ``pool*<n>`` arm plays the run's own
``step_<n>.pt`` ladder, so it needs ``runtime.snapshot_interval_steps > 0``, 11 Sep 2026).
``[policy] init_value`` warm-starts the value net from the same RL checkpoint as the policy (P10).
"""

from __future__ import annotations

import copy
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Final, TypeVar

import torch

import model as model_module
from melee_rl.actor import ActorConfig
from melee_rl.adapter import rl_model_config
from melee_rl.checkpoint import model_config_from_yaml
from melee_rl.env.dolphin import DolphinEnvConfig
from melee_rl.env.dummy import DummyEnvConfig
from melee_rl.evaluate import EvalConfig
from melee_rl.fast_step import FAST_ATTENTIONS
from melee_rl.learner import LearnerConfig
from melee_rl.logging import LoggingConfig
from melee_rl.mimic import MimicConfig
from melee_rl.opponents import OpponentConfig, split_rival_delay
from melee_rl.phillip import PhillipConfig
from melee_rl.profiling import ProfilingConfig
from melee_rl.slippi_ai_agent import SlippiAIConfig
from melee_rl.smashbot import SmashBotConfig
from melee_rl.video import VideoConfig
from model import ModelConfig

CONFIG_FORMAT: Final[str] = "melee_rl.rl_config.v1"
CONFIG_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "configs" / "rl"
POLICY_INITS: Final[tuple[str, ...]] = ("random", "checkpoint")
TEACHER_SOURCES: Final[tuple[str, ...]] = ("policy", "checkpoint", "ema", "none")
ENV_TYPES: Final[tuple[str, ...]] = ("dummy", "dolphin")
TABLE_ORDER: Final[tuple[str, ...]] = (
    "runtime",
    "policy",
    "teacher",
    "env",
    "actor",
    "opponent",
    "learner",
    "delay",
    "logging",
    "eval",
    "video",
    "mimic",
    "smashbot",
    "slippi_ai",
    "phillip",
    "profiling",
)


def default_config_yaml() -> Path:
    """Ali's ``config.yaml`` next to ``model.py``."""

    return Path(model_module.__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class RuntimeConfig:
    """``[runtime]`` (slippi-ai ``RuntimeConfig``, ``rl/run_lib.py:30-46``; steps instead of seconds).

    ``max_runtime_s`` caps one launch (measured from ``Trainer.run``); ``max_total_runtime_s`` (2 Sep
    2026) caps the run across launches -- the wall time of every launch accumulates in the checkpoint's
    ``loop_state["runtime_s"]`` -- which is how a run longer than Modal's 24-hour container limit
    (the 36-hour 10m run) is expressed: launch by the same command until the total is spent.
    """

    num_steps: int = 10
    seed: int = 0
    deterministic: bool = True
    device: str = "cpu"
    threads: int | None = None
    checkpoint_dir: str | None = None
    resume: bool = True
    save_interval_steps: int = 1
    snapshot_interval_steps: int = 0
    reset_every_n_steps: int | None = None
    burnin_steps_after_reset: int = 0
    max_runtime_s: float | None = None
    max_total_runtime_s: float | None = None

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("[runtime] num_steps must be >= 0")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError, TypeError) as error:
            raise ValueError(f"[runtime] device {self.device!r} is not a torch device: {error}") from error
        if self.threads is not None and self.threads < 1:
            raise ValueError("[runtime] threads must be >= 1 or omitted")
        if self.save_interval_steps < 1:
            raise ValueError("[runtime] save_interval_steps must be >= 1")
        if self.snapshot_interval_steps < 0:
            raise ValueError("[runtime] snapshot_interval_steps must be >= 0 (0 disables step_<n>.pt copies)")
        if self.reset_every_n_steps is not None and self.reset_every_n_steps < 1:
            raise ValueError("[runtime] reset_every_n_steps must be >= 1 or omitted")
        if self.burnin_steps_after_reset < 0:
            raise ValueError("[runtime] burnin_steps_after_reset must be >= 0")
        if self.max_runtime_s is not None and not self.max_runtime_s > 0.0:
            raise ValueError("[runtime] max_runtime_s must be positive or omitted")
        if self.max_total_runtime_s is not None and not self.max_total_runtime_s > 0.0:
            raise ValueError("[runtime] max_total_runtime_s must be positive or omitted")


@dataclass(frozen=True)
class PolicyConfig:
    """``[policy]``: which ``ModelConfig`` to build and how to initialise the weights.

    ``config_yaml`` ``None`` means Ali's ``config.yaml`` next to ``model.py``; ``profile`` ``None``
    means the YAML's active profile; ``model_overrides`` are ``ModelConfig`` fields applied with
    ``dataclasses.replace`` (the tiny test models); ``compute_dtype`` / ``cache_dtype`` are the RL
    precision (PLAN.md §3.9) and win over the same keys in ``model_overrides``.  ``fast_step``
    routes the actor's per-frame step through the vectorised ring path (``melee_rl.fast_step``,
    P7-PLAN.md 3; parity-tested, off by default).  ``fast_attention`` (5 Sep 2026) picks that path's
    attention: ``"sdpa"`` (Ali's) or ``"lean"`` (copy-free per-kv-head matmuls over the ring, needs
    ``fast_step``); ``fast_encoder`` (5 Sep 2026) encodes the actor's frames without host round trips
    (``melee_rl.fast_encode``, bit-identical); ``graph_step`` (5 Sep 2026) replays the frame step from a CUDA
    graph (``melee_rl.graph_step``; needs both, ``actor.packed_frames`` and a cuda device).
    """

    config_yaml: str | None = None
    profile: str | None = None
    init: str = "random"
    checkpoint: str | None = None
    init_value: bool = False
    seed: int = 0
    compute_dtype: str = "float32"
    cache_dtype: str = "float32"
    fast_step: bool = False
    fast_attention: str = "sdpa"
    fast_encoder: bool = False
    graph_step: bool = False
    model_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fast_attention not in FAST_ATTENTIONS:
            raise ValueError(
                f"policy.fast_attention must be one of {FAST_ATTENTIONS}, got {self.fast_attention!r}"
            )
        if self.fast_attention != "sdpa" and not self.fast_step:
            raise ValueError(f'policy.fast_attention = "{self.fast_attention}" needs policy.fast_step = true')
        if self.graph_step and not (self.fast_step and self.fast_encoder):
            raise ValueError(
                "policy.graph_step = true needs policy.fast_step = true and policy.fast_encoder = true "
                "(a sync-free step is what a CUDA graph can capture)"
            )
        if self.init not in POLICY_INITS:
            raise ValueError(f"policy.init must be one of {POLICY_INITS}, got {self.init!r}")
        if self.init == "checkpoint" and not self.checkpoint:
            raise ValueError('policy.checkpoint is required when policy.init = "checkpoint"')
        if self.init_value and self.init != "checkpoint":
            raise ValueError(
                'policy.init_value = true needs policy.init = "checkpoint" (the value net is warm-started '
                "from the same RL checkpoint file, P10)"
            )


@dataclass(frozen=True)
class TeacherConfig:
    """``[teacher]``: ``policy`` (a frozen copy of the initial policy), ``checkpoint`` (``path``), ``ema``
    (a copy of the initial policy that tracks the trained one, ``ema_tau``) or ``none``.

    ``delay_frames`` (11 Sep 2026, ``/delay-finetune``) is the teacher's own delay ``D_T``: the learner reads
    the teacher at row ``k + D - D_T`` with the pairing of a ``D_T``-delayed policy, whose decision executes
    at the same frame as the student's (``Trajectory.teacher_window``).  Unset means the run's
    ``actor.delay_frames`` (today's rows, bit for bit); ``0`` is the hindsight teacher -- the frozen delay-0
    expert read where the decision lands; it can never exceed the run's delay, and ``learner.teacher_prefix``
    needs it equal to the run's (``finalize_config``).

    ``ema_tau`` (15 Sep 2026, ``STRENGTH-LEVERS.md`` §2.2): with ``source = "ema"`` the anchor starts as the
    initial policy and, after every *accepted* PPO step, moves ``theta_T <- (1 - tau) theta_T + tau theta``
    -- a ``1 / tau``-step moving average of the policy, so the teacher KL measures the policy's speed rather
    than its distance from a fixed file (the manual "re-anchor hop" of ``DELAY-PLAN.md`` §15, continuous).
    The anchor is learner state (saved in the RL checkpoint, restored on resume) and ``tau`` is part of the
    teacher's identity (a resume under another ``tau`` is refused).  ``reference_path`` names a frozen file
    whose forward and reverse KL are logged at weight 0 (``loss/reference_kl``, ``loss/reverse_reference_kl``)
    -- the distance-from-anchor read that ``loss/teacher_kl`` stops being once the anchor moves; any source
    but ``"none"`` may carry one.
    """

    source: str = "policy"
    path: str | None = None
    delay_frames: int | None = None
    ema_tau: float | None = None
    reference_path: str | None = None

    def __post_init__(self) -> None:
        if self.source not in TEACHER_SOURCES:
            raise ValueError(f"teacher.source must be one of {TEACHER_SOURCES}, got {self.source!r}")
        if self.source == "checkpoint" and not self.path:
            raise ValueError('teacher.path is required when teacher.source = "checkpoint"')
        if self.delay_frames is not None:
            if self.delay_frames < 0:
                raise ValueError("teacher.delay_frames must be >= 0 or omitted (the run's delay)")
            if self.source == "none":
                raise ValueError('teacher.delay_frames has no meaning with teacher.source = "none"')
        if self.source == "ema":
            if self.ema_tau is None or not (0.0 < self.ema_tau <= 1.0) or self.ema_tau != self.ema_tau:
                raise ValueError(
                    f'teacher.source = "ema" needs teacher.ema_tau in (0, 1], got {self.ema_tau!r}'
                )
        elif self.ema_tau is not None:
            raise ValueError(f'teacher.ema_tau is only used by teacher.source = "ema", got {self.source!r}')
        if self.reference_path is not None and self.source == "none":
            raise ValueError('teacher.reference_path has no meaning with teacher.source = "none"')


@dataclass(frozen=True)
class EnvConfig:
    """``[env]``: ``type`` (``dummy`` | ``dolphin``) plus the per-type tables ``[env.dummy]`` /
    ``[env.dolphin]``.

    Both sub-tables always exist (defaults otherwise); :attr:`active` is the one ``type`` selects and
    ``num_envs`` / ``players`` read from it.
    """

    type: str = "dummy"
    dummy: DummyEnvConfig = field(default_factory=DummyEnvConfig)
    dolphin: DolphinEnvConfig = field(default_factory=DolphinEnvConfig)

    def __post_init__(self) -> None:
        if self.type not in ENV_TYPES:
            raise ValueError(f"env.type {self.type!r} is not supported; expected one of {ENV_TYPES}")

    @property
    def active(self) -> DummyEnvConfig | DolphinEnvConfig:
        return self.dolphin if self.type == "dolphin" else self.dummy

    @property
    def num_envs(self) -> int:
        return self.active.num_envs

    @property
    def players(self) -> tuple[str, str]:
        return self.active.players


@dataclass(frozen=True)
class DelayConfig:
    """``[delay]``: ``allow_mismatch`` permits ``delay_frames != action_offset_frames - 1`` (PLAN.md §3.5)."""

    allow_mismatch: bool = False


@dataclass(frozen=True)
class RLConfig:
    """One RL run: the tables above in TOML order."""

    format: str = CONFIG_FORMAT
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    delay: DelayConfig = field(default_factory=DelayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    mimic: MimicConfig = field(default_factory=MimicConfig)
    smashbot: SmashBotConfig = field(default_factory=SmashBotConfig)
    slippi_ai: SlippiAIConfig = field(default_factory=SlippiAIConfig)
    phillip: PhillipConfig = field(default_factory=PhillipConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def __post_init__(self) -> None:
        if self.format != CONFIG_FORMAT:
            raise ValueError(f"format must be {CONFIG_FORMAT!r}, got {self.format!r}")


_TABLE_TYPES: Final[dict[str, type]] = {
    "runtime": RuntimeConfig,
    "policy": PolicyConfig,
    "teacher": TeacherConfig,
    "env": EnvConfig,
    "actor": ActorConfig,
    "opponent": OpponentConfig,
    "learner": LearnerConfig,
    "delay": DelayConfig,
    "logging": LoggingConfig,
    "eval": EvalConfig,
    "video": VideoConfig,
    "mimic": MimicConfig,
    "smashbot": SmashBotConfig,
    "slippi_ai": SlippiAIConfig,
    "phillip": PhillipConfig,
    "profiling": ProfilingConfig,
}

T = TypeVar("T")


def _is_optional(hint: Any) -> tuple[bool, Any]:
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(args) == 1 and len(typing.get_args(hint)) == 2:
            return True, args[0]
    return False, hint


def _type_name(hint: Any) -> str:
    return getattr(hint, "__name__", str(hint))


def _coerce(hint: Any, value: Any, table: str, name: str) -> Any:
    """``value`` (a TOML scalar, array or table) as the field type ``hint``; exact checks."""

    label = f"[{table}] {name}"
    optional, inner = _is_optional(hint)
    if value is None:
        if optional:
            return None
        raise ValueError(f"{label} cannot be null")
    hint = inner
    if hint is Any:
        return value
    if isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a table")
        return _build(hint, value, f"{table}.{name}")
    origin = typing.get_origin(hint)
    if origin is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, list | tuple):
            raise ValueError(f"{label} must be a list")
        items = list(value)
        if len(args) == 2 and args[1] is Ellipsis:
            element_types = [args[0]] * len(items)
        else:
            element_types = list(args)
            if len(items) != len(element_types):
                kinds = ", ".join(_type_name(arg) for arg in element_types)
                raise ValueError(
                    f"{label} must be a list of {len(element_types)} ({kinds}), got {len(items)}"
                )
        return tuple(
            _coerce(element, item, table, f"{name}[{index}]")
            for index, (element, item) in enumerate(zip(element_types, items, strict=True))
        )
    if origin in (dict, Mapping) or hint in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a table")
        args = typing.get_args(hint)
        if len(args) == 2 and isinstance(args[1], type) and is_dataclass(args[1]):
            # A table of tables (``[opponent.arms.<alias>]``, 17 Sep 2026): every value is built and checked.
            return {str(key): _coerce(args[1], item, table, f"{name}.{key}") for key, item in value.items()}
        return dict(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be bool, got {type(value).__name__}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be int, got {type(value).__name__}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label} must be float, got {type(value).__name__}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be str, got {type(value).__name__}")
        return value
    raise TypeError(f"{label}: unsupported field type {hint!r}")


def _build(cls: type[T], mapping: Mapping[str, Any], table: str) -> T:
    """Instantiate the dataclass ``cls`` from ``mapping`` (the TOML table ``table``), keys checked exactly."""

    if not isinstance(mapping, Mapping):
        raise ValueError(f"[{table}] must be a table")
    hints = typing.get_type_hints(cls)
    known = {item.name: item for item in fields(cls)}  # type: ignore[arg-type]
    for key in mapping:
        if key not in known:
            raise ValueError(f"[{table}] unknown key {key!r}; expected one of {sorted(known)}")
    kwargs = {key: _coerce(hints[key], value, table, key) for key, value in mapping.items()}
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise ValueError(f"[{table}] {error}") from error


def config_from_mapping(raw: Mapping[str, Any]) -> RLConfig:
    """Build an :class:`RLConfig` from a parsed TOML mapping (no path resolution, no model checks)."""

    if not isinstance(raw, Mapping):
        raise TypeError(f"an RL config must be a mapping, got {type(raw).__name__}")
    declared = raw.get("format")
    if declared != CONFIG_FORMAT:
        raise ValueError(f"format must be {CONFIG_FORMAT!r} (top-level key), got {declared!r}")
    for key in raw:
        if key != "format" and key not in TABLE_ORDER:
            raise ValueError(f"unknown table {key!r}; expected {TABLE_ORDER}")
    tables: dict[str, Any] = {name: raw.get(name, {}) for name in TABLE_ORDER}
    opponent = _build(OpponentConfig, tables["opponent"], "opponent")
    env_raw = tables["env"]
    if not isinstance(env_raw, Mapping):
        raise ValueError("[env] must be a table")
    env_raw = dict(env_raw)
    for table_name in ("dummy", "dolphin"):
        sub_raw = env_raw.get(table_name, {})
        if not isinstance(sub_raw, Mapping):
            raise ValueError(f"[env.{table_name}] must be a table")
        if "players" not in sub_raw:
            env_raw[table_name] = {**sub_raw, "players": list(_expected_players(opponent.type))}
    built: dict[str, Any] = {
        name: _build(_TABLE_TYPES[name], table, name) for name, table in tables.items() if name != "env"
    }
    built["env"] = _build(EnvConfig, env_raw, "env")
    return RLConfig(format=CONFIG_FORMAT, **built)


def config_to_mapping(config: RLConfig) -> dict[str, Any]:
    """The plain-container form of ``config`` (``dataclasses.asdict``; checkpoints, W&B config, round
    trips)."""

    return asdict(config)


def _expected_players(opponent_type: str) -> tuple[str, str]:
    return ("policy", "cpu") if opponent_type == "cpu" else ("policy", "policy")


def _resolve(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return value if path.is_absolute() else str(base_dir / path)


def resolve_paths(config: RLConfig, base_dir: str | Path) -> RLConfig:
    """Relative paths (``policy.config_yaml``, ``policy.checkpoint``, ``teacher.path``,
    ``runtime.checkpoint_dir``, ``opponent.checkpoint``, ``env.dolphin.dolphin_path`` / ``iso_path`` /
    ``replay_dir``, ``video.bc`` / ``output_dir`` / ``video.render.dolphin_path`` / ``iso_path``)
    resolved against ``base_dir``; a missing ``config_yaml`` becomes Ali's."""

    base = Path(base_dir)
    policy = replace(
        config.policy,
        config_yaml=(
            str(default_config_yaml())
            if config.policy.config_yaml is None
            else _resolve(config.policy.config_yaml, base)
        ),
        checkpoint=_resolve(config.policy.checkpoint, base),
    )
    teacher = replace(
        config.teacher,
        path=_resolve(config.teacher.path, base),
        reference_path=_resolve(config.teacher.reference_path, base),
    )
    runtime = replace(config.runtime, checkpoint_dir=_resolve(config.runtime.checkpoint_dir, base))
    opponent = replace(
        config.opponent,
        checkpoint=_resolve(config.opponent.checkpoint, base),
        mix=tuple(_resolve_mix_spec(spec, base) for spec in config.opponent.mix),
    )
    dolphin = config.env.dolphin
    dolphin = replace(
        dolphin,
        dolphin_path=_resolve(dolphin.dolphin_path, base) or dolphin.dolphin_path,
        iso_path=_resolve(dolphin.iso_path, base) or dolphin.iso_path,
        replay_dir=_resolve(dolphin.replay_dir, base),
    )
    env = replace(config.env, dolphin=dolphin)
    evaluation = replace(
        config.eval,
        opponents=tuple(_resolve_opponent_spec(spec, base) for spec in config.eval.opponents),
    )
    render = config.video.render
    video = replace(
        config.video,
        bc=_resolve(config.video.bc, base),
        output_dir=_resolve(config.video.output_dir, base) or config.video.output_dir,
        render=replace(
            render,
            dolphin_path=_resolve(render.dolphin_path, base) or render.dolphin_path,
            iso_path=_resolve(render.iso_path, base) or render.iso_path,
        ),
    )
    return replace(
        config,
        policy=policy,
        teacher=teacher,
        runtime=runtime,
        opponent=opponent,
        env=env,
        eval=evaluation,
        video=video,
    )


def _resolve_opponent_spec(spec: str, base_dir: Path) -> str:
    """``other:<relative path>[@d<n>]`` -> ``other:<base_dir / path>[@d<n>]``; every other spec unchanged."""

    prefix = "other:"
    if spec.startswith(prefix):
        path, delay = split_rival_delay(spec[len(prefix) :])
        suffix = "" if delay is None else f"@d{delay}"
        return prefix + (_resolve(path, base_dir) or "") + suffix
    return spec


def _resolve_mix_spec(spec: str, base_dir: Path) -> str:
    """``other:<relative path>[@d<n>]*<n>`` -> the resolved path; every other mix spec unchanged."""

    head, star, count = spec.rpartition("*")
    prefix = "other:"
    if star and head.startswith(prefix):
        path, delay = split_rival_delay(head[len(prefix) :])
        resolved = _resolve(path, base_dir) or ""
        suffix = "" if delay is None else f"@d{delay}"
        return f"{prefix}{resolved}{suffix}*{count}"
    return spec


def resolve_model_config(config: RLConfig) -> ModelConfig:
    """``ModelConfig.from_yaml(config_yaml, profile)`` + RL precision + ``model_overrides``; a profile the
    YAML lacks comes from ``checkpoint.EXTRA_PROFILES`` (Ali's 75m), as it does for his checkpoints."""

    policy = config.policy
    path = default_config_yaml() if policy.config_yaml is None else Path(policy.config_yaml)
    if not path.is_file():
        raise ValueError(f"policy.config_yaml {path} does not exist")
    base = model_config_from_yaml(path, policy.profile)  # knows checkpoint.EXTRA_PROFILES (Ali's 75m)
    changes = {
        **policy.model_overrides,
        "compute_dtype": policy.compute_dtype,
        "cache_dtype": policy.cache_dtype,
    }
    try:
        return rl_model_config(base, **changes)
    except TypeError as error:
        raise ValueError(f"policy.model_overrides: {error}") from error


def pipeline_enabled(config: RLConfig) -> bool:
    """True when the run uses the delayed rollout pipeline (P8): ``batch_steps > 1`` or a Dolphin
    ``chunk_frames > 1``.  The trainer then wraps a synchronous environment in ``AsyncEnvAdapter``
    and builds ``PipelinedRolloutWorker``; both knobs at 1 keep today's lockstep loop untouched."""

    chunk = config.env.dolphin.chunk_frames if config.env.type == "dolphin" else 1
    return config.actor.batch_steps > 1 or chunk > 1


def finalize_config(config: RLConfig, model_config: ModelConfig) -> RLConfig:
    """Cross-field validation against the resolved model; returns the config with forced settings applied."""

    actor, learner, ppo = config.actor, config.learner, config.learner.ppo
    expected_delay = model_config.action_offset_frames - 1
    if actor.delay_frames != expected_delay and not config.delay.allow_mismatch:
        raise ValueError(
            f"delay_frames {actor.delay_frames} does not match the model's action_offset_frames "
            f"{model_config.action_offset_frames} - 1 = {expected_delay}; set [delay] allow_mismatch = true "
            f"for a deliberate mismatch (PLAN.md §3.5)"
        )
    teacher_delay = config.teacher.delay_frames
    if teacher_delay is not None:
        if teacher_delay > actor.delay_frames:
            raise ValueError(
                f"teacher.delay_frames {teacher_delay} exceeds actor.delay_frames {actor.delay_frames}: a "
                "teacher cannot see less than the student (its row k + D - D_T would lie before the decision)"
            )
        if teacher_delay != actor.delay_frames and learner.teacher_prefix:
            raise ValueError(
                f"learner.teacher_prefix = true needs teacher.delay_frames = actor.delay_frames "
                f"({actor.delay_frames}), got {teacher_delay}: the stored prefix keys hold the student's "
                "pairing; a teacher at its own delay forwards its own window"
            )
        if config.teacher.source == "ema" and teacher_delay != actor.delay_frames:
            raise ValueError(
                f'teacher.source = "ema" plays at the run\'s delay: teacher.delay_frames must be unset or '
                f"{actor.delay_frames}, got {teacher_delay} (a hindsight EMA of the student has no meaning)"
            )
    if actor.decision_stride > 1 and config.runtime.num_steps > 0:
        raise ValueError(
            f"actor.decision_stride {actor.decision_stride} is a play-time flag (the match recorder, lever 8 "
            "check (ii), 5 Sep 2026): a training run has no learner contract for held rows yet "
            "(PLAN.md 2.3 / 3.5 first)"
        )
    context = actor.resolve_context_frames(model_config.context_length)
    window = context + actor.rollout_length + 1
    value_context = learner.value.context_length
    if value_context is not None and value_context < window:
        raise ValueError(
            f"value.context_length {value_context} is smaller than the value window C + T + 1 = {window}"
        )
    num_envs = config.env.num_envs
    microbatch = learner.microbatch_envs
    if config.opponent.type == "mix":
        groups = config.opponent.groups
        total = sum(group.num_envs for group in groups)
        if total != num_envs:
            raise ValueError(
                f"the mix groups cover {total} environments but env.{config.env.type} num_envs is {num_envs}"
            )
        if any(group.kind == "pool" for group in groups) and config.runtime.snapshot_interval_steps == 0:
            raise ValueError(
                "a pool*<n> arm plays the run's own step_<n>.pt snapshots: set [runtime] "
                "snapshot_interval_steps > 0"
            )
        if microbatch is not None and learner.concat_arms:
            joined = sum(group.rows for group in groups)
            if microbatch > joined or joined % microbatch:
                raise ValueError(
                    f"microbatch_envs {microbatch} must divide the joined rows of the mix ({joined}: "
                    "learner.concat_arms joins every arm's rows per rollout)"
                )
        elif microbatch is not None:
            for group in groups:
                if microbatch > group.rows or group.rows % microbatch:
                    raise ValueError(
                        f"microbatch_envs {microbatch} must divide every mix group's rows; group "
                        f"{group.spec!r} has {group.rows} rows ({group.num_envs} environments x "
                        f"{len(group.trained_ports)} trained ports)"
                    )
    else:
        if learner.concat_arms:
            raise ValueError('learner.concat_arms joins a mix\'s arms: it needs [opponent] type = "mix"')
        ports = len(config.opponent.trained_ports)
        rows = num_envs * ports
        if microbatch is not None and (microbatch > rows or rows % microbatch):
            detail = f" x trained ports {ports} = {rows}" if ports > 1 else ""
            raise ValueError(f"microbatch_envs {microbatch} must divide num_envs {num_envs}{detail}")
    if model_config.compute_dtype != "float32":
        if ppo.behaviour_logits == "actor":
            raise ValueError(
                'ppo.behaviour_logits = "actor" needs exact fp32 behaviour logits; under compute_dtype '
                f'{model_config.compute_dtype} use "learner" (or "auto", PLAN.md §3.6 / §3.9)'
            )
        if ppo.behaviour_logits == "auto":
            config = replace(config, learner=replace(learner, ppo=replace(ppo, behaviour_logits="learner")))
            learner = config.learner
    teacher = config.teacher
    anchored = learner.kl_teacher_weight != 0.0 or learner.reverse_kl_teacher_weight != 0.0
    if teacher.source == "none" and anchored:
        raise ValueError(
            'teacher.source = "none" requires learner.kl_teacher_weight = 0 and '
            "learner.reverse_kl_teacher_weight = 0"
        )
    if teacher.source == "policy" and config.policy.init == "random":
        if learner.kl_teacher_weight != 0.0:
            raise ValueError(
                "a random-init teacher anchors the policy to noise (PLAN.md §9.4): set "
                f"learner.kl_teacher_weight = 0 (got {learner.kl_teacher_weight}) or initialise the policy "
                "from a checkpoint"
            )
        if learner.reverse_kl_teacher_weight != 0.0:
            raise ValueError(
                "a random-init teacher anchors the policy to noise (PLAN.md §9.4): set "
                f"learner.reverse_kl_teacher_weight = 0 (got {learner.reverse_kl_teacher_weight})"
            )
    if config.policy.graph_step:
        if not actor.packed_frames:
            raise ValueError(
                "policy.graph_step = true needs actor.packed_frames = true (the captured graph reads the "
                "packer's static frame buffers)"
            )
        if not config.runtime.device.startswith("cuda"):
            raise ValueError(
                f"policy.graph_step = true needs a cuda runtime.device, got {config.runtime.device!r}"
            )
    if learner.teacher_prefix and config.actor.context_mode != "prefix":
        raise ValueError(
            'learner.teacher_prefix = true needs actor.context_mode = "prefix" (the teacher attends to the '
            f"actor's stored cache; got context_mode {config.actor.context_mode!r})"
        )
    if learner.fast_forward and config.actor.context_mode != "prefix":
        raise ValueError(
            'learner.fast_forward = true needs actor.context_mode = "prefix" (the lean forward covers the '
            f"prefix window only; got context_mode {config.actor.context_mode!r})"
        )
    if learner.compile and model_config.gradient_checkpointing:
        raise ValueError(
            "learner.compile = true needs the model's gradient_checkpointing off: set "
            "[policy] model_overrides = {gradient_checkpointing = false} (the compiled lean pass fits "
            "without it)"
        )
    if config.opponent.type == "mix":
        if config.env.players != ("policy", "policy"):
            raise ValueError(
                f'with opponent.type = "mix" leave env.{config.env.type}.players at ("policy", "policy") '
                f"(each group derives its own players from its spec), got {config.env.players}"
            )
        mix_workers = config.opponent.mix_workers
        if config.env.type == "dolphin" and config.env.dolphin.worker_processes > 0:
            if not mix_workers:
                raise ValueError(
                    f"env.dolphin.worker_processes = {config.env.dolphin.worker_processes} needs "
                    "opponent.mix_workers (one worker count per group, summing to worker_processes)"
                )
            if sum(mix_workers) != config.env.dolphin.worker_processes:
                raise ValueError(
                    f"opponent.mix_workers {mix_workers} must sum to env.dolphin.worker_processes "
                    f"{config.env.dolphin.worker_processes}"
                )
        elif mix_workers:
            raise ValueError(
                f"opponent.mix_workers needs env.dolphin.worker_processes > 0; the {config.env.type} "
                "environment runs without worker processes"
            )
    else:
        expected_players = _expected_players(config.opponent.type)
        if config.env.players != expected_players:
            raise ValueError(
                f"env.{config.env.type}.players {config.env.players} does not match opponent.type "
                f"{config.opponent.type!r} (expected {expected_players})"
            )
    if pipeline_enabled(config):
        if actor.interleave_arms:
            raise ValueError(
                "actor.interleave_arms is the lockstep loop's arm overlap (lever 1, 5 Sep 2026): the delay "
                "pipeline has no frame generator to interleave"
            )
        batch_steps = actor.batch_steps
        chunk = config.env.dolphin.chunk_frames if config.env.type == "dolphin" else 1
        slack = (batch_steps - 1) + (chunk - 1)
        if slack > actor.delay_frames:
            raise ValueError(
                "the delay pipeline needs (batch_steps - 1) + (chunk_frames - 1) <= delay_frames: "
                f"({batch_steps} - 1) + ({chunk} - 1) = {slack} > {actor.delay_frames} (P8-PLAN.md 2)"
            )
        pipelined_opponent = config.opponent.type == "cpu" or (
            config.opponent.type == "self" and config.opponent.train
        )
        if not pipelined_opponent:
            raise ValueError(
                f"pipelined mode (batch_steps {batch_steps}, chunk_frames {chunk}) supports "
                'opponent.type "cpu" or "self" with train = true: a port-controlling opponent would '
                "act zero-delay on states already in flight; giving PolicyOpponent its own delay "
                "queue is the planned extension (P8-PLAN.md 2)"
            )
    if config.eval.enabled and config.eval.rollout_length > model_config.context_length:
        raise ValueError(
            f"eval.rollout_length {config.eval.rollout_length} must fit in the model's context_length "
            f"{model_config.context_length} (the evaluator runs in ring mode without a history window)"
        )
    workers = config.env.dolphin.worker_processes
    if (
        config.env.type == "dolphin"
        and workers > 0
        and config.eval.enabled
        and config.eval.worker_processes is None
        and config.eval.num_envs % workers
    ):
        raise ValueError(
            f"eval.num_envs {config.eval.num_envs} must be divisible by env.dolphin.worker_processes "
            f"{workers} (the evaluation environments inherit the worker layout unless eval.worker_processes "
            "sets their own)"
        )
    return config


def parse_override(text: str) -> tuple[list[str], Any]:
    """``table.key=value`` -> (path, value); the value is a TOML literal, or a bare string when it is not one.

    ``learner.ppo.num_batches=4`` gives an int, ``policy.fast_encoder=true`` a bool,
    ``learner.learning_rate=4e-5`` a float, ``policy.fast_attention=lean`` (or ``="lean"``) the string.
    """

    key, separator, value = text.partition("=")
    key = key.strip()
    if not separator or not key or any(not part for part in key.split(".")):
        raise ValueError(f"override {text!r} must look like table.key=value")
    literal = value.strip()
    try:
        parsed = tomllib.loads(f"value = {literal}")["value"]
    except tomllib.TOMLDecodeError:
        parsed = literal
    return key.split("."), parsed


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """A deep copy of the parsed TOML ``raw`` with every ``table.key=value`` override applied (5 Sep 2026).

    Intermediate tables are created when missing; a key whose current value is a table cannot be replaced by a
    scalar.  Unknown keys and wrong types are reported by :func:`config_from_mapping` as for the file itself.
    """

    result = copy.deepcopy(dict(raw))
    for text in overrides:
        path, value = parse_override(text)
        node: dict[str, Any] = result
        for part in path[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise ValueError(f"override {text!r}: {part!r} is a value, not a table")
            node = child
        if isinstance(node.get(path[-1]), dict) and not isinstance(value, dict):
            raise ValueError(f"override {text!r}: {path[-1]!r} is a table")
        node[path[-1]] = value
    return result


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> RLConfig:
    """Parse, apply ``table.key=value`` overrides, resolve paths against the TOML's directory, load the model
    config and finalize."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    if overrides:
        raw = apply_overrides(raw, overrides)
    config = resolve_paths(config_from_mapping(raw), source.resolve().parent)
    return finalize_config(config, resolve_model_config(config))


__all__ = [
    "CONFIG_DIR",
    "CONFIG_FORMAT",
    "ENV_TYPES",
    "POLICY_INITS",
    "TABLE_ORDER",
    "TEACHER_SOURCES",
    "DelayConfig",
    "EnvConfig",
    "PolicyConfig",
    "RLConfig",
    "RuntimeConfig",
    "TeacherConfig",
    "apply_overrides",
    "config_from_mapping",
    "config_to_mapping",
    "default_config_yaml",
    "finalize_config",
    "load_config",
    "parse_override",
    "pipeline_enabled",
    "resolve_model_config",
    "resolve_paths",
]
