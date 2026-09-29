"""RL checkpoints, BC export and Ali's training checkpoints (PLAN.md §3.3, INTERFACE.md §8).

The one module that knows the on-disk formats, so that a different answer from Ali is a local
change (user decision 21 Aug 2026).  Every format is a plain container of tensors, numbers,
strings and ``None``, written with ``torch.save`` and read back with
``torch.load(path, weights_only=True)``:

* ``melee_policy.bc_checkpoint.v1`` (INTERFACE.md §8.1, proposed for Ali): ``format``,
  ``created``, ``slippi_ai_commit``, ``model_config`` (``dataclasses.asdict(ModelConfig)``),
  ``config_yaml_sha256``, ``codec``, ``state_dict`` (``MeleePolicy`` keys -- the adapter adds
  none) and an optional ``training`` record.  Load: ``MeleePolicy(ModelConfig(**model_config))``
  + ``load_state_dict(state_dict, strict=True)``.
* ``melee_rl.rl_checkpoint.v1`` (INTERFACE.md §8.2): the BC keys plus ``step``, ``rl_config``,
  ``learner_config``, ``value_config``, ``value_state_dict``, ``policy_optimizer``,
  ``value_optimizer``, ``learner_step`` (the burn-in ladder position), ``teacher``,
  ``delay_frames``, ``context_mode``, ``rng``, ``wandb_run_id`` and ``loop_state`` (the training
  loop's counters, P3).  An RL checkpoint is a valid BC checkpoint (slippi-ai: "this state is
  valid as an imitation state", ``run_lib.py:659``).
* Ali's training checkpoint (``training.py:732-758``, merged 22 Aug 2026; our label
  :data:`ALI_TRAINING_FORMAT`): ``{"format_version": 1, "model": policy.state_dict(),
  "optimizer_bundle", "training_state", "frame_cursor", "rng_state", "resolved_config": {...,
  "model": asdict(ModelConfig)}, "metadata"}`` -- no ``format`` string.  :func:`detect_format`
  recognises it; :func:`load_bc_checkpoint` maps ``resolved_config["model"]`` to the
  ``ModelConfig`` and ``model`` to the state dict, so ``[policy] init = "checkpoint"``,
  ``[teacher] source = "checkpoint"`` and the ``other`` opponent accept his files directly (P6a).
  His own reader (``training.load_checkpoint``) needs a live ``OptimizerBundle``; we only read the
  payload.  :func:`export_bc` keeps writing §8.1 (INTERFACE.md §10, question 8).

Files are written atomically (temporary file + ``os.replace``); ``latest.pt`` plus periodic
``step_<n>.pt`` copies are the training loop's naming (``checkpoint_paths``, slippi-ai
``run_lib.py:659-672``).  :func:`restore_learner` resumes a :class:`PPOLearner` exactly (same
next loss -- tested).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch

import model as model_module
from controller_codec import CustomV1Codec
from melee_rl.adapter import MeleePolicyAdapter, rl_model_config
from melee_rl.learner import PPOLearner
from melee_rl.value import ValueConfig, ValueNet
from model import ModelConfig

BC_FORMAT: Final[str] = "melee_policy.bc_checkpoint.v1"
RL_FORMAT: Final[str] = "melee_rl.rl_checkpoint.v1"
ALI_TRAINING_FORMAT: Final[str] = "melee_policy.training_checkpoint.v1"
"""Our label for Ali's ``format_version == 1`` training checkpoints (``training.py:732-758``)."""
ALI_FORMAT_VERSION: Final[int] = 1
"""``training.CHECKPOINT_FORMAT_VERSION`` as merged on 22 Aug 2026."""
RUNTIME_SCHEMA_VERSION: Final[str] = "melee-policy.final-pretraining-runtime.v1"
"""``metadata["schema_version"]`` of Ali's family-pretraining checkpoints (first seen 1 Sep 2026, the
10m ``step-106806.pt``): still ``format_version 1`` with the ``model`` state dict, but
``resolved_config["model"]`` is a 16-field summary -- ``profile``, the six dimensions, parameter counts
-- rather than the ``ModelConfig`` fields.  :func:`ali_model_config` rebuilds the config from
``config.yaml``'s profile and checks the summary against it."""
PROFILE_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "d_ff",
)
"""The ``ModelConfig`` fields the runtime summary repeats; each must agree with the profile."""
EXTRA_PROFILES: Final[Mapping[str, Mapping[str, int]]] = {
    "75m": {"d_model": 768, "n_layers": 11, "n_heads": 12, "n_kv_heads": 3, "head_dim": 64, "d_ff": 1920},
}
"""Scaling profiles Ali's checkpoints name that ``config.yaml`` on this branch does not define.

``75m`` is the family-pretraining profile of Ali's ``pretraining`` and ``benchmark-launcher`` branches
(``config.yaml:91`` at ``268031e`` / ``faab939``; first needed 4 Sep 2026 for ``75m-muon-low`` step
86016).  :func:`model_config_from_yaml` injects an entry only when the YAML has no profile of that name,
so a merge of Ali's file simply takes over; the parameter-count check of :func:`ali_model_config` still
runs against the state dict either way."""
EXTRA_PROFILE_PARAMETERS: Final[Mapping[str, int]] = {"75m": 75_305_709}
"""Ali's ``expected_parameter_counts[...].total`` for :data:`EXTRA_PROFILES` (checked by the tests)."""
SAFE_GLOBALS: Final[tuple[type, ...]] = (frozenset,)
"""Builtins ``torch.load(weights_only=True)`` must be told to accept beyond its own allow-list.

Ali's curriculum checkpoints (``melee-policy.curriculum.v1`` blocks inside the runtime schema; first seen
5 Sep 2026 -- ``cur-6-kd/10m/B-mix10/steps/step-195248.pt`` and ``cur-3/75m/B-mix10/steps/step-127214.pt``)
pickle the secondary stage's ``ranks`` as a ``frozenset`` under ``resolved_config["curriculum"]`` and
``metadata["curriculum"]``, and the weights-only unpickler rejects every global it was not given
(``Unsupported global: GLOBAL frozenset``).  A ``frozenset`` carries no code, so allow-listing it keeps the
no-code-execution guarantee; :func:`load_checkpoint` applies it to every file it reads."""
FORMATS: Final[tuple[str, ...]] = (BC_FORMAT, RL_FORMAT, ALI_TRAINING_FORMAT)
LATEST_NAME: Final[str] = "latest.pt"
INIT_NAME: Final[str] = "init.pt"
"""The frozen copy of ``policy.checkpoint`` a run writes into its own directory at first launch (P10):
``best.pt`` / ``latest.pt`` source paths are mutable, so the run keeps the exact bytes it started from."""


def utc_now() -> str:
    """ISO-8601 UTC timestamp with a ``Z`` suffix (``2026-08-21T00:00:00Z``)."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def codec_info() -> dict[str, Any]:
    codec = CustomV1Codec()
    return {
        "name": "custom_v1",
        "vocab_sizes": dict(zip(codec.component_order, codec.vocabulary_sizes, strict=True)),
    }


def _plain(value: Any) -> Any:
    """Containers of tensors / numbers / strings only (``Path`` -> ``str``, tensors -> CPU copies)."""

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key) if isinstance(key, Path) else key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_plain(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"checkpoint payloads accept tensors, numbers, strings and containers, not {type(value)}")


def atomic_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """``torch.save`` to a temporary file in the same directory, then ``os.replace``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)
    # Modal Volumes (v1) were seen keeping a 0-byte ghost entry under the renamed-away name (P4,
    # 21 Aug 2026); a no-op on ordinary filesystems.
    temporary.unlink(missing_ok=True)
    return target


# ---------------------------------------------------------------------------
# BC format (INTERFACE.md §8.1)
# ---------------------------------------------------------------------------


def bc_payload(
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    *,
    config_yaml: str | Path | None = None,
    training: Mapping[str, Any] | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    """The ``melee_policy.bc_checkpoint.v1`` dictionary for ``state_dict`` (CPU copies)."""

    return {
        "format": BC_FORMAT,
        "created": utc_now() if created is None else created,
        "slippi_ai_commit": model_config.slippi_ai_commit,
        "model_config": asdict(model_config),
        "config_yaml_sha256": None if config_yaml is None else sha256_file(config_yaml),
        "codec": codec_info(),
        "state_dict": _plain(dict(state_dict)),
        "training": {} if training is None else _plain(dict(training)),
    }


def save_bc_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    **options: Any,
) -> dict[str, Any]:
    """Write a BC checkpoint (``bc_payload`` options) and return the payload."""

    payload = bc_payload(state_dict, model_config, **options)
    atomic_save(payload, path)
    return payload


def export_bc(
    path: str | Path,
    learner: PPOLearner,
    model_config: ModelConfig,
    *,
    step: int,
    config_yaml: str | Path | None = None,
    wandb_run_id: str | None = None,
    delay_frames: int = 0,
) -> dict[str, Any]:
    """Export the learner's policy as a BC checkpoint (PLAN.md §3.3 ``export_bc``).

    ``delay_frames`` is the delay the policy was trained to play at (``training.delay_frames``, read back by
    :func:`trained_delay`).
    """

    training = {
        "step": int(step),
        "dataset": "rl_post_training",
        "wandb_run": wandb_run_id,
        "delay_frames": int(delay_frames),
    }
    return save_bc_checkpoint(
        path,
        learner.policy.get_state(),
        model_config,
        config_yaml=config_yaml,
        training=training,
    )


@dataclass(frozen=True)
class BCCheckpoint:
    """A loaded checkpoint's policy part: ``ModelConfig`` + ``state_dict`` (+ the raw payload).

    ``metadata`` is the format's own training record: our ``training`` dict, or Ali's ``metadata``
    (``run_name`` / ``run_id`` / ...) plus his ``training_state`` and its ``processed_target_frames``.
    """

    format: str
    model_config: ModelConfig
    state_dict: dict[str, torch.Tensor]
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def default_config_yaml() -> Path:
    """Ali's ``config.yaml`` next to ``model.py`` -- the profiles a runtime-schema checkpoint names."""

    return Path(model_module.__file__).resolve().parent / "config.yaml"


def model_config_from_yaml(path: str | Path, profile: str | None = None) -> ModelConfig:
    """``ModelConfig.from_yaml(path, profile)`` that also knows :data:`EXTRA_PROFILES`.

    The YAML's own ``scaling.profiles`` entry wins whenever it exists; a profile the YAML lacks is filled
    in from the table (the six scaling dimensions, everything else from the YAML as for any profile).
    """

    import yaml

    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(root, Mapping):
        raise TypeError("config root must be a mapping")
    scaling = root.get("scaling")
    profiles = scaling.get("profiles") if isinstance(scaling, Mapping) else None
    if profile in EXTRA_PROFILES and isinstance(profiles, Mapping) and profile not in profiles:
        assert profile is not None and scaling is not None
        patched = {**scaling, "profiles": {**profiles, profile: dict(EXTRA_PROFILES[profile])}}
        root = {**root, "scaling": patched}
    return ModelConfig.from_mapping(root, profile=profile)


def ali_model_config(payload: Mapping[str, Any], *, config_yaml: str | Path | None = None) -> ModelConfig:
    """The ``ModelConfig`` of an Ali training checkpoint.

    ``training.py``'s format stores every ``ModelConfig`` field under ``resolved_config["model"]``; the
    runtime schema (:data:`RUNTIME_SCHEMA_VERSION`) stores a summary with a ``profile`` name instead, so
    the config comes from ``config_yaml`` (Ali's ``config.yaml`` by default) and the summary's dimensions
    and ``total_trainable_parameters`` are checked against the profile and the state dict -- a wrong
    profile or a foreign state dict is an error, never a silent mis-load.
    """

    block = payload["resolved_config"]["model"]
    names = {item.name for item in fields(ModelConfig)}
    if names <= set(block):
        return ModelConfig(**block)
    profile = block.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError(
            "the checkpoint's resolved_config['model'] is neither a ModelConfig nor a profile summary "
            f"(no 'profile'); keys {sorted(block)}"
        )
    path = default_config_yaml() if config_yaml is None else Path(config_yaml)
    try:
        config = model_config_from_yaml(path, profile)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"checkpoint profile {profile!r} is not a profile of {path}: {error}") from error
    for name in PROFILE_SUMMARY_FIELDS:
        if name in block and block[name] != getattr(config, name):
            raise ValueError(
                f"checkpoint profile {profile!r}: {name} {block[name]!r} differs from {path}'s "
                f"{getattr(config, name)!r}"
            )
    expected = block.get("total_trainable_parameters")
    if expected is not None:
        actual = sum(int(value.numel()) for value in payload["model"].values())
        if int(expected) != actual:
            raise ValueError(
                f"checkpoint profile {profile!r}: total_trainable_parameters {expected} but the state dict "
                f"holds {actual} parameters"
            )
    return config


def detect_format(payload: Mapping[str, Any]) -> str:
    """Which of :data:`FORMATS` a loaded payload is in (``ValueError`` for anything else).

    Ours carry a ``format`` string; Ali's carry ``format_version == 1`` with ``model`` and
    ``resolved_config["model"]``.
    """

    declared = payload.get("format")
    if declared is not None:
        if declared in (BC_FORMAT, RL_FORMAT):
            return str(declared)
        raise ValueError(f"unknown checkpoint format {declared!r}, expected one of {FORMATS}")
    if "format_version" in payload:
        version = payload["format_version"]
        if version != ALI_FORMAT_VERSION:
            raise ValueError(
                f"unsupported training checkpoint format_version {version!r} (expected {ALI_FORMAT_VERSION}, "
                "melee-policy-frisson-ai training.py)"
            )
        resolved = payload.get("resolved_config")
        if (
            not isinstance(payload.get("model"), Mapping)
            or not isinstance(resolved, Mapping)
            or not isinstance(resolved.get("model"), Mapping)
        ):
            raise ValueError(
                "a format_version 1 training checkpoint needs a 'model' state dict and "
                "'resolved_config' with a 'model' table (asdict(ModelConfig))"
            )
        return ALI_TRAINING_FORMAT
    raise ValueError(
        f"unknown checkpoint format: neither a 'format' nor a 'format_version' key; expected one of {FORMATS}"
    )


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """``torch.load(weights_only=True)`` plus a format check (:func:`detect_format`); the raw payload."""

    with torch.serialization.safe_globals(list(SAFE_GLOBALS)):
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: a checkpoint must be a mapping, got {type(payload).__name__}")
    try:
        detect_format(payload)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return dict(payload)


def load_bc_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> BCCheckpoint:
    """Read the policy part of a BC / RL checkpoint or of Ali's training checkpoint."""

    payload = load_checkpoint(path, map_location=map_location)
    kind = detect_format(payload)
    if kind == ALI_TRAINING_FORMAT:
        model_config = ali_model_config(payload)
        raw_state = payload["model"]
        metadata: dict[str, Any] = dict(payload.get("metadata") or {})
        training_state = payload.get("training_state")
        if isinstance(training_state, Mapping):
            metadata.setdefault("training_state", dict(training_state))
            if "processed_target_frames" in training_state:
                metadata.setdefault("processed_target_frames", training_state["processed_target_frames"])
    else:
        model_config = ModelConfig(**payload["model_config"])
        raw_state = payload["state_dict"]
        metadata = dict(payload.get("training") or {})
    state_dict = {str(key): value for key, value in raw_state.items()}
    return BCCheckpoint(
        format=kind,
        model_config=model_config,
        state_dict=state_dict,
        payload=payload,
        metadata=metadata,
    )


def load_policy(
    path: str | Path,
    *,
    rl_precision: bool = True,
    device: str | torch.device | None = None,
    gradient_checkpointing: bool | None = None,
) -> MeleePolicyAdapter:
    """A ``MeleePolicyAdapter`` with the checkpoint's config (fp32 RL precision by default) and weights.

    ``gradient_checkpointing`` (10 Sep 2026) overrides the file's flag when not ``None``: it is memory only
    (the weights and the forward are the file's), and a frozen teacher, which never backpropagates, reads its
    file with it off -- Ali's training files carry it on, and the compiled lean forward refuses it.
    """

    checkpoint = load_bc_checkpoint(path)
    config = rl_model_config(checkpoint.model_config) if rl_precision else checkpoint.model_config
    if gradient_checkpointing is not None:
        config = replace(config, gradient_checkpointing=gradient_checkpointing)
    adapter = MeleePolicyAdapter(config)
    adapter.load_state_dict(checkpoint.state_dict, strict=True)
    if device is not None:
        adapter.to(device)
    return adapter


# ---------------------------------------------------------------------------
# RL format (INTERFACE.md §8.2)
# ---------------------------------------------------------------------------


def rl_payload(
    learner: PPOLearner,
    model_config: ModelConfig,
    *,
    step: int,
    rl_config: Mapping[str, Any] | None = None,
    teacher: Mapping[str, Any] | None = None,
    delay_frames: int | None = None,
    context_mode: str | None = None,
    rng: Mapping[str, Any] | None = None,
    wandb_run_id: str | None = None,
    config_yaml: str | Path | None = None,
    created: str | None = None,
    loop_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``melee_rl.rl_checkpoint.v1`` dictionary: BC keys plus the learner state.

    ``loop_state`` (P3) holds the training loop's counters (``frames_seen``, ``rollouts_trained``,
    ``hard_resets``, ``game_starts``, ``opponent_refreshes``); ``None`` when a caller has none.
    """

    state = learner.state_dict()
    payload = bc_payload(
        state["policy"],
        model_config,
        config_yaml=config_yaml,
        training={
            "step": int(step),
            "dataset": "rl_post_training",
            "wandb_run": wandb_run_id,
            "delay_frames": 0 if delay_frames is None else int(delay_frames),
        },
        created=created,
    )
    payload.update(
        {
            "format": RL_FORMAT,
            "step": int(step),
            "rl_config": None if rl_config is None else _plain(dict(rl_config)),
            "learner_config": asdict(learner.config),
            "value_config": asdict(learner.value_net.config),
            "value_state_dict": state["value"],
            "policy_optimizer": state["policy_optimizer"],
            "value_optimizer": state["value_optimizer"],
            "learner_step": int(state["step"]),
            # The adaptive levers of 15 Sep 2026: the live entropy bonus and learning rate, and the EMA
            # anchor's weights when the teacher is one (``None`` for a fixed teacher).
            "entropy_weight": float(state["entropy_weight"]),
            "policy_lr": float(state["policy_lr"]),
            "teacher_state_dict": state.get("teacher"),
            "teacher": None if teacher is None else _plain(dict(teacher)),
            "delay_frames": delay_frames,
            "context_mode": context_mode,
            "rng": None if rng is None else _plain(dict(rng)),
            "wandb_run_id": wandb_run_id,
            "loop_state": None if loop_state is None else _plain(dict(loop_state)),
        }
    )
    return payload


def save_rl_checkpoint(
    path: str | Path,
    learner: PPOLearner,
    model_config: ModelConfig,
    **options: Any,
) -> dict[str, Any]:
    """Write an RL checkpoint (``rl_payload`` options) and return the payload."""

    payload = rl_payload(learner, model_config, **options)
    atomic_save(payload, path)
    return payload


def restore_learner(payload: Mapping[str, Any], learner: PPOLearner) -> int:
    """Load an RL payload into ``learner`` (weights, optimizers, ladder position); returns ``step``."""

    if payload.get("format") != RL_FORMAT:
        raise ValueError(f"restore_learner needs a {RL_FORMAT} payload, got {payload.get('format')!r}")
    saved_value = ValueConfig(**payload["value_config"])
    if saved_value != learner.value_net.config:
        message = f"value config mismatch: checkpoint {saved_value}, learner {learner.value_net.config}"
        raise ValueError(message)
    learner.load_state_dict(
        {
            "policy": payload["state_dict"],
            "value": payload["value_state_dict"],
            "policy_optimizer": payload["policy_optimizer"],
            "value_optimizer": payload["value_optimizer"],
            "step": payload["learner_step"],
            # Absent from files written before 15 Sep 2026: the learner then keeps its config's values.
            "entropy_weight": payload.get("entropy_weight"),
            "policy_lr": payload.get("policy_lr"),
            "teacher": payload.get("teacher_state_dict"),
        }
    )
    return int(payload["step"])


def load_value_net(path: str | Path, value_net: ValueNet) -> None:
    """Warm-start ``value_net`` from an RL checkpoint's value head (``policy.init_value``, P10).

    Only RL checkpoints carry a value net (a BC file or Ali's training checkpoint has none), and the
    checkpoint's ``ValueConfig`` must match the learner's exactly (as in :func:`restore_learner`).
    """

    payload = load_checkpoint(path)
    kind = detect_format(payload)
    if kind != RL_FORMAT:
        raise ValueError(
            f"{path}: policy.init_value needs a {RL_FORMAT} checkpoint (only RL checkpoints carry the "
            f"value net), got {kind}"
        )
    saved = ValueConfig(**payload["value_config"])
    if saved != value_net.config:
        raise ValueError(f"{path}: value config mismatch: checkpoint {saved}, learner {value_net.config}")
    value_net.load_state_dict(payload["value_state_dict"])


def checkpoint_paths(directory: str | Path, step: int) -> tuple[Path, Path]:
    """``(latest.pt, step_<n>.pt)`` inside ``directory``."""

    base = Path(directory)
    return base / LATEST_NAME, base / f"step_{int(step):08d}.pt"


def trained_delay(source: str | Path | Mapping[str, Any]) -> int:
    """The delay a checkpoint was trained to play at (11 Sep 2026, ``/delay-finetune``).

    One checkpoint means one total delay: a policy trained at ``D`` decides ``D + 1`` frames ahead and plays
    under a console delay ``k <= D`` by queueing the rest itself, never below ``D``.  An RL payload has stored
    the run's ``actor.delay_frames`` since P3 (nothing read it before); the BC ``training`` record carries it
    since this date; anything without the field -- a BC file from before, Ali's training checkpoints -- reads
    the model's ``action_offset_frames - 1`` (0 for every file so far).  ``source`` is a path or a loaded
    payload.
    """

    payload = source if isinstance(source, Mapping) else load_checkpoint(source)
    kind = detect_format(payload)
    value: Any = None
    if kind == RL_FORMAT:
        value = payload.get("delay_frames")
    if value is None and kind in (BC_FORMAT, RL_FORMAT):
        training = payload.get("training")
        if isinstance(training, Mapping):
            value = training.get("delay_frames")
    if value is None:
        if kind == ALI_TRAINING_FORMAT:
            value = ali_model_config(payload).action_offset_frames - 1
        else:
            value = int(payload["model_config"]["action_offset_frames"]) - 1
    delay = int(value)
    if delay < 0:
        raise ValueError(f"checkpoint delay_frames must be >= 0, got {delay}")
    return delay


__all__ = [
    "ALI_FORMAT_VERSION",
    "ALI_TRAINING_FORMAT",
    "BC_FORMAT",
    "EXTRA_PROFILES",
    "EXTRA_PROFILE_PARAMETERS",
    "FORMATS",
    "INIT_NAME",
    "LATEST_NAME",
    "PROFILE_SUMMARY_FIELDS",
    "RL_FORMAT",
    "RUNTIME_SCHEMA_VERSION",
    "SAFE_GLOBALS",
    "BCCheckpoint",
    "ali_model_config",
    "atomic_save",
    "bc_payload",
    "checkpoint_paths",
    "codec_info",
    "default_config_yaml",
    "detect_format",
    "export_bc",
    "load_bc_checkpoint",
    "load_checkpoint",
    "load_policy",
    "load_value_net",
    "model_config_from_yaml",
    "restore_learner",
    "rl_payload",
    "save_bc_checkpoint",
    "save_rl_checkpoint",
    "sha256_file",
    "trained_delay",
    "utc_now",
]
