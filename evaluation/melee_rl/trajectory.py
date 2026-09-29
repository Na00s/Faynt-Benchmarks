"""``Trajectory``: the batch-major rollout record shared by actor and learner (PLAN.md §2.3).

Window ``W = C + T + 1`` rows ``w``: ``0..C-1`` history, ``C..C+T-1`` the rollout decisions,
``C+T`` the bootstrap state; ``t = w - C`` is the rollout index; ``D`` is the delay.

* ``frames`` -- tree with leaves ``[B, W]`` (``items`` ``[B, W, 15]``): raw canonical fields
  (``GameStateBatch`` leaves), not encoder outputs;
* ``controller_exec`` -- labels ``[B, W]``: executed on the transition into row ``w``;
* ``controller_prev_sample`` -- labels ``[B, W]``: the policy's previous-action input at ``w`` =
  the sample made at ``w-1``; neutral at reset rows and padded rows;
* ``sampled_labels`` -- labels ``[B, W]``: the sample made *at* ``w``, so
  ``sampled_labels[w] == controller_exec[w+1+D]``; row ``C+T`` holds neutral labels (the policy
  samples for ``s_T`` at the start of the next rollout);
* ``sampled_logits`` -- ``{buttons: [B, T, 728], main_stick: [B, T, 85]}``: the raw actor logits
  of rows ``C..C+T-1`` (conditioned on the sampled prefix, exact behaviour log-probs);
* ``rewards`` -- ``[B, T]``: ``r(s_t -> s_{t+1})`` = frame reward (``melee_rl.reward``) plus
  ``env_rewards``, zero on transitions into reset rows;
* ``env_rewards`` -- ``[B, T]``: the environment-emitted part (zero for Melee; the dummy env's
  learnable signal);
* ``is_resetting`` -- ``[B, W]`` bool: ``frame == -123`` rows (game starts);
* ``padding`` -- ``[B, W]`` bool: true for real frames, false for history an env does not have;
* ``frame_index`` -- ``[B, W]`` long: game frame number (diagnostics);
* ``delayed_actions`` -- view ``[B, D]``: ``sampled_labels[:, C+T-D : C+T]``, the decisions not
  yet executed at ``s_T``;
* ``initial_cache`` -- ``context_mode = "prefix"`` only: the actor's KV cache as it stood right before
  the rollout's first sample (``melee_rl.kv_cache.snapshot_cache``), the history the learner attends to
  through ``unroll_window(..., prefix=)`` instead of re-forwarding rows ``0..C-1`` (slippi-ai's
  ``trajectory.initial_state``; INTERFACE.md patch #1).  The history rows stay in the window for the
  teacher and the value net.

Learner slices (INTERFACE.md §9; slippi-ai ``rl/learner.py:217-259``, verified): policy and teacher
rows ``[0, C+T-D)`` with inputs ``(frame, controller_prev_sample)`` and teacher-forcing targets
``sampled_labels``; the decisions entering the loss are rows ``[C, C+T-D)`` = rollout ``[0, T-D)``
with actor log-probs from ``sampled_logits[:, :T-D]``; advantages pair decision ``t`` with value row
``t + D`` (``advantage_rows = [D, T)``); the value net reads rows ``[0, C+T]`` with the undelayed
pairing ``(frame, controller_exec)`` and bootstraps at ``C+T``.  With ``D = 0`` the shifts vanish.

A teacher of its own delay ``D_T <= D`` (``[teacher] delay_frames``, 11 Sep 2026) reads
``teacher_window(D_T)``: rows ``[0, C+T-D_T)`` with the pairing of a ``D_T``-delayed policy,
``(frame j, controller_exec[j + D_T]) -> controller_exec[j + D_T + 1]``; decision ``k`` is paired with teacher
row ``k + D - D_T`` (``teacher_decision_rows``), where a ``D_T``-delayed decision executes at the same frame,
and ``teacher_keep`` drops the decisions whose teacher row lies in another game.  ``D_T = D`` is
``policy_window`` bit for bit, ``D_T = 0`` the hindsight teacher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import torch

from controller_codec import COMPONENT_ORDER, ControllerLabels
from melee_rl import reward as reward_lib
from melee_rl.frames import (
    FrameTree,
    labels_cat,
    labels_index,
    labels_to,
    tree_cat,
    tree_index,
    tree_leaves,
    tree_to,
    validate_frame_tree,
)
from melee_rl.kv_cache import cache_cat, cache_index, cache_to
from model import KVCache

CONTEXT_MODES: Final[tuple[str, ...]] = ("reprime", "ring", "prefix")
"""``reprime``: the actor re-primes C rows per rollout (exact); ``ring``: the rolling cache, the learner
re-forwards C + T rows (inexact); ``prefix``: the rolling cache plus its snapshot on the trajectory, the
learner forwards the T rows against it (exact, PLAN.md §3.2 option b)."""
VOCAB_SIZES: Final[dict[str, int]] = {"buttons": 728, "main_stick": 85}


@dataclass(frozen=True)
class Window:
    """Inputs of ``PolicyProtocol.unroll_window`` / the value net over ``[B, L]`` rows."""

    frames: FrameTree
    controller_prev: ControllerLabels
    targets: ControllerLabels | None
    reset_mask: torch.Tensor
    padding_mask: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.reset_mask.shape[1])


def _rows(tree_or_labels: Any, rows: slice) -> Any:
    index = (slice(None), rows)
    if isinstance(tree_or_labels, ControllerLabels):
        return labels_index(tree_or_labels, index)
    if isinstance(tree_or_labels, torch.Tensor):
        return tree_or_labels[index]
    return tree_index(tree_or_labels, index)


def _mask_labels(labels: ControllerLabels, keep: torch.Tensor) -> ControllerLabels:
    """Neutral labels where ``keep`` (``[B, L]`` bool) is false."""

    return ControllerLabels(
        buttons=torch.where(keep, labels.buttons, torch.zeros_like(labels.buttons)),
        main_stick=torch.where(keep, labels.main_stick, torch.zeros_like(labels.main_stick)),
    )


@dataclass(frozen=True)
class Trajectory:
    """One rollout of ``B`` environments; see the module docstring for every field."""

    frames: FrameTree
    controller_exec: ControllerLabels
    controller_prev_sample: ControllerLabels
    sampled_labels: ControllerLabels
    sampled_logits: dict[str, torch.Tensor]
    rewards: torch.Tensor
    env_rewards: torch.Tensor
    is_resetting: torch.Tensor
    padding: torch.Tensor
    frame_index: torch.Tensor
    context_frames: int
    rollout_length: int
    delay_frames: int
    context_mode: str = "reprime"
    initial_cache: KVCache | None = None

    # -- shapes -------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(self.is_resetting.shape[0])

    @property
    def window_length(self) -> int:
        return self.context_frames + self.rollout_length + 1

    @property
    def device(self) -> torch.device:
        return self.is_resetting.device

    # -- slices (INTERFACE.md §9) -------------------------------------------------------

    @property
    def policy_rows(self) -> slice:
        """Window rows forwarded through the policy and the teacher: ``[0, C+T-D)``."""

        return slice(0, self.context_frames + self.rollout_length - self.delay_frames)

    @property
    def decision_rows(self) -> slice:
        """Window rows whose decisions enter the PPO loss: ``[C, C+T-D)``."""

        return slice(self.context_frames, self.context_frames + self.rollout_length - self.delay_frames)

    @property
    def actor_rows(self) -> slice:
        """Rollout rows of ``sampled_logits`` entering the loss: ``[0, T-D)``."""

        return slice(0, self.rollout_length - self.delay_frames)

    @property
    def advantage_rows(self) -> slice:
        """Value rows paired with the decisions: ``[D, T)`` (decision ``t`` <-> row ``t + D``)."""

        return slice(self.delay_frames, self.rollout_length)

    @property
    def value_rows(self) -> slice:
        """Window rows read by the value net (bootstrap at ``C+T``): ``[0, C+T]``."""

        return slice(0, self.window_length)

    @property
    def delayed_actions(self) -> ControllerLabels:
        """The ``D`` decisions not yet executed at ``s_T``: ``sampled_labels[:, C+T-D : C+T]``."""

        end = self.context_frames + self.rollout_length
        return labels_index(self.sampled_labels, (slice(None), slice(end - self.delay_frames, end)))

    def policy_window(self) -> Window:
        """``(frame, controller_prev_sample) -> sampled_labels`` over rows ``[0, C+T-D)``."""

        rows = self.policy_rows
        return Window(
            frames=_rows(self.frames, rows),
            controller_prev=_rows(self.controller_prev_sample, rows),
            targets=_rows(self.sampled_labels, rows),
            reset_mask=self.is_resetting[:, rows],
            padding_mask=self.padding[:, rows],
        )

    def value_window(self) -> Window:
        """``(frame, controller_exec)`` over rows ``[0, C+T]`` (the undelayed pairing)."""

        rows = self.value_rows
        return Window(
            frames=_rows(self.frames, rows),
            controller_prev=_rows(self.controller_exec, rows),
            targets=None,
            reset_mask=self.is_resetting[:, rows],
            padding_mask=self.padding[:, rows],
        )

    def actor_logits(self) -> dict[str, torch.Tensor]:
        """Behaviour logits of the decisions entering the loss: ``sampled_logits[:, :T-D]``."""

        return {name: value[:, self.actor_rows] for name, value in self.sampled_logits.items()}

    def actor_labels(self) -> ControllerLabels:
        """The labels those logits produced: ``sampled_labels[:, C : C+T-D]``."""

        return labels_index(self.sampled_labels, (slice(None), self.decision_rows))

    def rollout_frames(self) -> FrameTree:
        """Rows ``[C, C+T]`` (``T + 1`` frames) -- the reward's input."""

        return _rows(self.frames, slice(self.context_frames, self.window_length))

    def prefix_window(self) -> Window:
        """The decision rows ``[C, C+T-D)`` alone: the policy inputs of ``context_mode = "prefix"``, where
        ``initial_cache`` supplies the history (``unroll_window(..., prefix=initial_cache)``)."""

        rows = self.decision_rows
        return Window(
            frames=_rows(self.frames, rows),
            controller_prev=_rows(self.controller_prev_sample, rows),
            targets=_rows(self.sampled_labels, rows),
            reset_mask=self.is_resetting[:, rows],
            padding_mask=self.padding[:, rows],
        )

    # -- the teacher's own delay (``[teacher] delay_frames``, 11 Sep 2026, /delay-finetune) ------------------

    def _check_teacher_delay(self, teacher_delay: int) -> None:
        if teacher_delay < 0 or teacher_delay > self.delay_frames:
            raise ValueError(
                f"teacher delay_frames {teacher_delay} must lie in [0, delay_frames {self.delay_frames}]: a "
                "teacher cannot see less than the student"
            )

    def teacher_rows(self, teacher_delay: int) -> slice:
        """The rows a teacher of delay ``D_T`` forwards: ``[0, C+T-D_T)`` (``policy_rows`` at ``D_T = D``)."""

        self._check_teacher_delay(teacher_delay)
        return slice(0, self.context_frames + self.rollout_length - teacher_delay)

    def teacher_decision_rows(self, teacher_delay: int) -> slice:
        """The teacher rows paired with the trained decisions: ``[C + D - D_T, C+T-D_T)``.

        Decision ``k`` executes at ``k + D + 1``; a ``D_T``-delayed decision made at row ``j = k + D - D_T``
        executes at the same frame, so that is the row whose distribution is about the same controller.
        """

        self._check_teacher_delay(teacher_delay)
        shift = self.delay_frames - teacher_delay
        return slice(self.context_frames + shift, self.context_frames + self.rollout_length - teacher_delay)

    def teacher_window(self, teacher_delay: int) -> Window:
        """``(frame j, controller_exec[j + D_T]) -> controller_exec[j + D_T + 1]`` over :meth:`teacher_rows`.

        A teacher trained at delay ``D_T`` read with its own pairing: its previous-action input is the sample
        it made one frame earlier, which executes on the transition into ``j + D_T``, and its target the
        sample executing at ``j + D_T + 1`` -- at the decision rows exactly ``sampled_labels[k]``, the
        student's own label, so ``distributions.ppo_terms`` works unchanged.  Neutral at reset rows and
        padded rows, as the actor's inputs are.  At ``D_T = D`` this is :meth:`policy_window` bit for bit; at
        ``D_T = 0`` the hindsight teacher (DIDA, Liotet et al. 2022): the undelayed pairing of
        :meth:`value_window` read where each decision lands.
        """

        rows = self.teacher_rows(teacher_delay)
        end = self.context_frames + self.rollout_length
        prev = _rows(self.controller_exec, slice(teacher_delay, end))
        targets = _rows(self.controller_exec, slice(teacher_delay + 1, end + 1))
        padding = self.padding[:, rows]
        reset = self.is_resetting[:, rows]
        return Window(
            frames=_rows(self.frames, rows),
            controller_prev=_mask_labels(prev, padding & ~reset),
            targets=_mask_labels(targets, padding),
            reset_mask=reset,
            padding_mask=padding,
        )

    def teacher_keep(self, teacher_delay: int) -> torch.Tensor:
        """``[B, T-D]`` bool: decision ``k`` keeps its teacher terms unless a reset falls in
        ``(k, k + D - D_T]``, which puts the teacher's frame in the next game.  All true at ``D_T = D``."""

        self._check_teacher_delay(teacher_delay)
        shift = self.delay_frames - teacher_delay
        start = self.context_frames
        stop = self.context_frames + self.rollout_length - self.delay_frames
        if shift == 0:
            return torch.ones((self.batch_size, stop - start), dtype=torch.bool, device=self.device)
        counts = torch.cumsum(self.is_resetting.long(), dim=1)
        rows = torch.arange(start, stop, device=self.device)
        between = counts[:, rows + shift] - counts[:, rows]
        return between == 0

    # -- rewards ------------------------------------------------------------------

    def recompute_rewards(
        self,
        config: reward_lib.RewardConfig | None = None,
        *,
        check_bounds: bool = True,
    ) -> Trajectory:
        """The trajectory with ``rewards`` recomputed from its frames under ``config`` (+ ``env_rewards``)."""

        resets = self.is_resetting[:, self.context_frames :]
        frame_rewards = reward_lib.compute_rewards(self.rollout_frames(), config, check_bounds=check_bounds)
        rewards = reward_lib.zero_at_resets(frame_rewards, resets) + self.env_rewards
        return replace(self, rewards=rewards)

    # -- validation -------------------------------------------------------------------

    def validate(self) -> None:
        """Check every shape, dtype and the static invariants of the contract."""

        batch, window = self.batch_size, self.window_length
        context, length, delay = self.context_frames, self.rollout_length, self.delay_frames
        if context < 0 or length < 1 or delay < 0:
            raise ValueError("context_frames >= 0, rollout_length >= 1 and delay_frames >= 0 are required")
        if delay >= length:
            raise ValueError(f"delay_frames {delay} must be smaller than rollout_length {length}")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}, got {self.context_mode!r}")
        cache = self.initial_cache
        if self.context_mode == "prefix" and cache is None:
            raise ValueError("context_mode 'prefix' requires initial_cache")
        if self.context_mode != "prefix" and cache is not None:
            raise ValueError(f"context_mode {self.context_mode!r} does not carry an initial_cache")
        if cache is not None:
            if cache.keys.ndim != 5 or cache.keys.shape[1] != batch or cache.values.shape != cache.keys.shape:
                shape = tuple(cache.keys.shape)
                raise ValueError(f"initial_cache K/V must both be [L, {batch}, C, K, Dh], got keys {shape}")
            for name in ("valid_length", "write_position", "next_position"):
                value = getattr(cache, name)
                if tuple(value.shape) != (batch,) or value.dtype != torch.long:
                    raise ValueError(f"initial_cache.{name} must be long [{batch}], got {value.shape}")
            if bool(((cache.valid_length < 0) | (cache.valid_length > cache.capacity)).any()):
                raise ValueError("initial_cache.valid_length is outside [0, capacity]")
        validate_frame_tree(self.frames, (batch, window))
        for name, labels in (
            ("controller_exec", self.controller_exec),
            ("controller_prev_sample", self.controller_prev_sample),
            ("sampled_labels", self.sampled_labels),
        ):
            for component, vocabulary in VOCAB_SIZES.items():
                value = getattr(labels, component)
                if tuple(value.shape) != (batch, window) or value.dtype != torch.long:
                    raise ValueError(
                        f"{name}.{component} must be long [{batch}, {window}], got {value.shape}"
                    )
                if bool(((value < 0) | (value >= vocabulary)).any()):
                    raise ValueError(f"{name}.{component} must be in [0, {vocabulary})")
        for component in COMPONENT_ORDER:
            logits = self.sampled_logits.get(component)
            expected = (batch, length, VOCAB_SIZES[component])
            if logits is None or tuple(logits.shape) != expected or not logits.is_floating_point():
                raise ValueError(f"sampled_logits[{component!r}] must be float {expected}")
        for name, value, dtype in (
            ("rewards", self.rewards, torch.float32),
            ("env_rewards", self.env_rewards, torch.float32),
        ):
            if tuple(value.shape) != (batch, length) or value.dtype != dtype:
                raise ValueError(f"{name} must be {dtype} [{batch}, {length}], got {value.shape}")
        for name, value, dtype in (
            ("is_resetting", self.is_resetting, torch.bool),
            ("padding", self.padding, torch.bool),
            ("frame_index", self.frame_index, torch.long),
        ):
            if tuple(value.shape) != (batch, window) or value.dtype != dtype:
                raise ValueError(f"{name} must be {dtype} [{batch}, {window}], got {value.shape}")
        if bool((self.is_resetting & ~self.padding).any()):
            raise ValueError("padded rows cannot be resetting")

    # -- conversions ------------------------------------------------------------------

    def to(self, device: torch.device | str, *, cache: bool = True) -> Trajectory:
        """Every tensor on ``device``; ``cache=False`` leaves ``initial_cache`` where it is (the learner's
        ``prefix_on_host``: the stored prefixes stay in host memory and ride to the device per microbatch)."""

        return replace(
            self,
            frames=tree_to(self.frames, device),
            controller_exec=labels_to(self.controller_exec, device),
            controller_prev_sample=labels_to(self.controller_prev_sample, device),
            sampled_labels=labels_to(self.sampled_labels, device),
            sampled_logits={name: value.to(device) for name, value in self.sampled_logits.items()},
            rewards=self.rewards.to(device),
            env_rewards=self.env_rewards.to(device),
            is_resetting=self.is_resetting.to(device),
            padding=self.padding.to(device),
            frame_index=self.frame_index.to(device),
            initial_cache=(
                cache_to(self.initial_cache, device)
                if cache and self.initial_cache is not None
                else self.initial_cache
            ),
        )

    def index_envs(self, index: Any) -> Trajectory:
        """The sub-batch ``index`` (slice, list or long tensor along ``B``) -- learner microbatches."""

        return replace(
            self,
            frames=tree_index(self.frames, index),
            controller_exec=labels_index(self.controller_exec, index),
            controller_prev_sample=labels_index(self.controller_prev_sample, index),
            sampled_labels=labels_index(self.sampled_labels, index),
            sampled_logits={name: value[index] for name, value in self.sampled_logits.items()},
            rewards=self.rewards[index],
            env_rewards=self.env_rewards[index],
            is_resetting=self.is_resetting[index],
            padding=self.padding[index],
            frame_index=self.frame_index[index],
            initial_cache=None if self.initial_cache is None else cache_index(self.initial_cache, index),
        )

    @staticmethod
    def batch(trajectories: Sequence[Trajectory]) -> Trajectory:
        """Concatenate along ``B`` (same ``C``, ``T``, ``D`` and mode) -- two ports, several workers."""

        if not trajectories:
            raise ValueError("batch needs at least one trajectory")
        first = trajectories[0]
        for other in trajectories[1:]:
            same = (
                other.context_frames == first.context_frames
                and other.rollout_length == first.rollout_length
                and other.delay_frames == first.delay_frames
                and other.context_mode == first.context_mode
            )
            if not same:
                raise ValueError(
                    "trajectories must share context_frames, rollout_length, delay_frames and mode"
                )
        caches = [item.initial_cache for item in trajectories if item.initial_cache is not None]
        if caches and len(caches) != len(trajectories):
            raise ValueError("trajectories must all carry an initial_cache or none")
        return replace(
            first,
            frames=tree_cat([item.frames for item in trajectories]),
            controller_exec=labels_cat([item.controller_exec for item in trajectories]),
            controller_prev_sample=labels_cat([item.controller_prev_sample for item in trajectories]),
            sampled_labels=labels_cat([item.sampled_labels for item in trajectories]),
            sampled_logits={
                name: torch.cat([item.sampled_logits[name] for item in trajectories])
                for name in first.sampled_logits
            },
            rewards=torch.cat([item.rewards for item in trajectories]),
            env_rewards=torch.cat([item.env_rewards for item in trajectories]),
            is_resetting=torch.cat([item.is_resetting for item in trajectories]),
            padding=torch.cat([item.padding for item in trajectories]),
            frame_index=torch.cat([item.frame_index for item in trajectories]),
            initial_cache=cache_cat(caches) if caches else None,
        )

    def memory_report(self) -> dict[str, int]:
        """Bytes held per field group and in total (PLAN.md §2.4)."""

        def size(*tensors: torch.Tensor) -> int:
            return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

        frames = size(*(leaf for _, leaf in tree_leaves(self.frames)))
        labels = size(
            *self.controller_exec.values(),
            *self.controller_prev_sample.values(),
            *self.sampled_labels.values(),
        )
        logits = size(*self.sampled_logits.values())
        masks = size(self.is_resetting, self.padding, self.frame_index)
        rewards = size(self.rewards, self.env_rewards)
        cache = 0 if self.initial_cache is None else int(self.initial_cache.storage_report()["bytes"])
        return {
            "frames": frames,
            "labels": labels,
            "logits": logits,
            "masks": masks,
            "rewards": rewards,
            "cache": cache,
            "total": frames + labels + logits + masks + rewards + cache,
        }


__all__ = ["CONTEXT_MODES", "VOCAB_SIZES", "Trajectory", "Window"]
