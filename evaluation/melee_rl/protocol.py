"""``PolicyProtocol``: everything the RL package needs from a policy (INTERFACE.md §4).

The rest of ``melee_rl`` (actor, learner, opponents, checkpointing) depends on this
protocol only; ``melee_rl.adapter.MeleePolicyAdapter`` is the one implementation and
the only module that knows ``MeleePolicy`` internals.

Conventions (INTERFACE.md §1): batch-major ``[B]`` for one step and ``[B, L]`` for a
window; labels are ``controller_codec.ControllerLabels`` with ``torch.long`` leaves;
logits are ``{component: float32 [..., V_component]}`` in ``CustomV1Codec`` order
(``("buttons", "main_stick")``, vocabularies ``(728, 85)``); masks are ``torch.bool``.
A row with ``reset_mask`` true starts a new segment (no attention across it, RoPE
positions restart at 0); ``padding_mask`` false marks rows that carry no frame (left
padding for envs with a short history) -- they are neither attended nor written to the
cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch

from controller_codec import ControllerLabels
from model import KVCache
from tensor_batch import GameStateBatch

Frames = GameStateBatch | Mapping[str, Any]
"""A frame tree (``melee_rl.frames``) or its typed ``GameStateBatch`` view."""


class ActionSpec(NamedTuple):
    """Component order and vocabulary sizes of the policy's action space."""

    component_order: tuple[str, ...]
    vocab_sizes: tuple[int, ...]


@dataclass(frozen=True)
class StepOutput:
    """One sampled step.

    ``labels`` are the sampled ``[B]`` labels; ``logits`` are the raw (un-tempered)
    per-component logits that produced them, ``{name: float32 [B, V_name]}``, with the
    autoregressive conditioning on the *sampled* prefix -- so
    ``distributions.log_prob(logits, labels)`` is the exact behaviour log-probability
    when ``temperature == 1.0`` (the RL default); at other temperatures the behaviour
    distribution is ``softmax(logits / temperature)``.
    """

    labels: ControllerLabels
    logits: dict[str, torch.Tensor]


@dataclass(frozen=True)
class WindowOutput:
    """Teacher-forced logits over a window.

    ``logits`` are ``{name: float32 [B, L, V_name]}`` conditioned on the *given* targets
    (``model.py:1777-1788``); ``hidden`` is the backbone output ``[B, L, d_model]``
    (value-net reuse, diagnostics).  Padded rows hold zeros.
    """

    logits: dict[str, torch.Tensor]
    hidden: torch.Tensor


@runtime_checkable
class PolicyProtocol(Protocol):
    """Policy interface used by the actor (cache path) and the learner (window path)."""

    @property
    def action_spec(self) -> ActionSpec: ...

    @property
    def delay_offset(self) -> int:
        """The model's ``action_offset_frames`` (1 today; INTERFACE.md §7)."""
        ...

    @property
    def context_length(self) -> int:
        """Maximum ``L`` accepted by :meth:`unroll_window` and :meth:`prime`."""
        ...

    def encode_observation(self, obs: Mapping[str, Any]) -> GameStateBatch:
        """Validate a frame tree and return its typed view on the policy device."""
        ...

    def init_cache(self, batch_size: int) -> KVCache:
        """A fresh, empty KV cache for ``batch_size`` environments."""
        ...

    def reset_slots(self, cache: KVCache, reset_mask: torch.Tensor) -> None:
        """Clear the cache slots where ``reset_mask`` (``[B]`` bool) is true."""
        ...

    def prime(
        self,
        frames: Frames,
        controller_prev: ControllerLabels,
        reset_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        cache: KVCache,
    ) -> None:
        """Reset every slot, then fill ``cache`` with the ``[B, C]`` history (no sampling)."""
        ...

    def step_sample(
        self,
        frame: Frames,
        controller_prev: ControllerLabels,
        reset_mask: torch.Tensor,
        cache: KVCache,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> StepOutput:
        """Advance the cache by one ``[B]`` frame and sample an action."""
        ...

    def step_sample_multi(
        self,
        frames: Frames,
        controller_prev: ControllerLabels,
        reset_mask: torch.Tensor,
        cache: KVCache,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[StepOutput]:
        """``b`` consecutive ``[B]`` frames in one call == ``b`` :meth:`step_sample` calls (P8).

        ``frames`` is ``[B, b]``; ``controller_prev`` the raw labels before frame 0 (the
        method masks the chain at reset rows); ``reset_mask`` is ``[B, b]``.  Advances
        ``cache`` and consumes ``generator`` exactly like the sequential calls.
        """
        ...

    def unroll_window(
        self,
        frames: Frames,
        controller_prev: ControllerLabels,
        targets: ControllerLabels,
        reset_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        prefix: KVCache | None = None,
    ) -> WindowOutput:
        """Teacher-forced logits over a ``[B, L]`` window, ``L <= context_length``.

        ``prefix`` (a detached snapshot of the actor's cache, ``melee_rl.kv_cache.snapshot_cache``)
        makes the rows continue that history exactly as the cached steps did -- the learner path of
        ``context_mode = "prefix"``; positions, segments and the sliding band follow the cache.
        """
        ...

    def get_state(self) -> dict[str, torch.Tensor]:
        """A detached CPU copy of the parameters and persistent buffers."""
        ...

    def set_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load a state produced by :meth:`get_state` (strict)."""
        ...

    def clone_frozen(self) -> PolicyProtocol:
        """A deep copy in eval mode with gradients disabled (teacher / self-opponent)."""
        ...


__all__ = [
    "ActionSpec",
    "Frames",
    "PolicyProtocol",
    "StepOutput",
    "WindowOutput",
]
