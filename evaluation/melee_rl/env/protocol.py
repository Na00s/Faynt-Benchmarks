"""``EnvProtocol``: what the rollout worker needs from an environment (PLAN.md §2.2, §6 P1).

Semantics follow slippi-ai's ``Environment`` (``slippi_ai/envs.py``):

* the environment always has a *pending* state -- the initial state before any action,
  or the state produced by the last ``step`` (``envs.py:70-101`` ``current_state``);
* ``step(actions)`` sends one controller per controlled port and returns the next state
  (``envs.py:105-118``); a port that is not controlled is driven by the environment itself
  (Dolphin's in-game CPU, the dummy env's script);
* ``needs_reset[b]`` is true exactly when ``frames[b]`` is the first frame of a new game,
  i.e. ``frame_index[b] == -123`` (``envs.py:25-26`` ``is_initial_frame``); the policy
  resets its state *at* that row (``is_resetting`` in the trajectory), nothing is
  discarded;
* every controlled port receives the same game from its own perspective: ``p0`` is that
  port's player (``tensor_batch.py:163-165``), so port 1's tree is port 0's with ``p0``
  and ``p1`` swapped (:func:`swap_perspective`).

Everything is batched over ``B`` environments: frame-tree leaves are ``[B]`` (``items``
``[B, 15]``), masks ``[B]`` bool, indices ``[B]`` long, controllers
``controller_codec.ControllerState`` with ``[B]`` leaves.  Environments produce CPU
tensors; the worker moves them to the policy device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import torch

from controller_codec import ControllerState
from melee_rl.frames import FRAME_SPEC, FrameTree, validate_frame_tree

INITIAL_FRAME_INDEX: Final[int] = -123
"""Frame index of the first frame of a game (slippi-ai ``envs.py:25-26``)."""


def swap_perspective(tree: Mapping[str, Any]) -> FrameTree:
    """The same frame tree seen from the other port (``p0`` <-> ``p1``); shares the leaves."""

    swapped: FrameTree = dict(tree)
    swapped["p0"], swapped["p1"] = tree["p1"], tree["p0"]
    return swapped


def canonical_frame_tree(tree: Mapping[str, Any]) -> FrameTree:
    """``tree`` restricted to :data:`FRAME_SPEC` in spec order (shares the leaves).

    Environments may add diagnostics to their trees; the worker's buffers and the
    trajectory carry exactly the canonical leaves, in a fixed order.
    """

    def select(spec: Mapping[str, Any], node: Mapping[str, Any]) -> FrameTree:
        result: FrameTree = {}
        for key, kind in spec.items():
            value = node[key]
            result[key] = select(kind, value) if isinstance(kind, Mapping) else value
        return result

    return select(FRAME_SPEC, tree)


@dataclass(frozen=True)
class EnvOutput:
    """One batched environment state.

    ``frames`` maps every controlled port to its perspective of the state (leaves
    ``[B]``); ``needs_reset`` (``[B]`` bool) marks first frames of new games;
    ``frame_index`` (``[B]`` long) is the game frame number (``-123`` at a game start);
    ``rewards`` maps every controlled port to an environment-emitted reward (``[B]``
    float32) for the transition *into* this state -- zero for Melee, where the reward is
    computed from the frames (``melee_rl.reward``); the dummy environment's
    learnable-signal mode uses it.  Rewards are zero on reset rows.
    """

    frames: Mapping[int, FrameTree]
    needs_reset: torch.Tensor
    frame_index: torch.Tensor
    rewards: Mapping[int, torch.Tensor]

    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(self.frames)

    @property
    def batch_size(self) -> int:
        return int(self.needs_reset.shape[0])

    def validate(self, num_envs: int | None = None) -> None:
        """Check shapes, dtypes and the ``needs_reset <-> frame_index == -123`` invariant."""

        if self.needs_reset.ndim != 1 or self.needs_reset.dtype != torch.bool:
            raise TypeError(
                f"needs_reset must be a [B] bool tensor, got {tuple(self.needs_reset.shape)}"
            )
        batch = self.batch_size
        if num_envs is not None and batch != num_envs:
            raise ValueError(f"EnvOutput batch {batch} does not match num_envs {num_envs}")
        if self.frame_index.shape != (batch,) or self.frame_index.is_floating_point():
            raise TypeError(f"frame_index must be a [{batch}] integer tensor, got {self.frame_index.shape}")
        if bool((self.needs_reset != (self.frame_index == INITIAL_FRAME_INDEX)).any()):
            raise ValueError("needs_reset must be true exactly where frame_index == -123")
        if not self.frames:
            raise ValueError("EnvOutput has no frames")
        for port, tree in self.frames.items():
            validate_frame_tree(tree, (batch,))
            if port not in self.rewards:
                raise ValueError(f"EnvOutput has frames but no rewards for port {port}")
        for port, reward in self.rewards.items():
            if port not in self.frames:
                raise ValueError(f"EnvOutput has rewards but no frames for port {port}")
            if reward.shape != (batch,) or not reward.is_floating_point():
                raise TypeError(
                    f"rewards[{port}] must be a [{batch}] float tensor, got {reward.shape} {reward.dtype}"
                )


@runtime_checkable
class EnvProtocol(Protocol):
    """A batch of synchronous Melee environments."""

    @property
    def num_envs(self) -> int: ...

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        """Ports that must receive a controller in every :meth:`step` (``(0,)`` or ``(0, 1)``)."""
        ...

    def current(self) -> EnvOutput:
        """The pending state (the initial state, or the result of the last :meth:`step`)."""
        ...

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        """Apply one controller per controlled port and return the next state."""
        ...

    def reset(self) -> EnvOutput:
        """Hard reset: every environment starts a new game; returns the new pending state."""
        ...

    def close(self) -> None: ...


__all__ = [
    "INITIAL_FRAME_INDEX",
    "EnvOutput",
    "EnvProtocol",
    "canonical_frame_tree",
    "swap_perspective",
]
