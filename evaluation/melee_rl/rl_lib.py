# Ported from slippi-ai slippi_ai/jax/rl_lib.py@577965a7731dc53e3472ea63d9e9853a4e9d65fa (MIT)
"""Discounts, returns and value-target helpers (PLAN.md §3.7, R3).

Re-implementation in PyTorch of the small pieces the value function needs:

* ``discount_from_halflife`` -- ``0.5 ** (1 / (halflife * fps))`` (``jax/rl_lib.py:9-11``,
  ``rl/learner.py:133``): 4 s -> 0.99712, 8 s -> 0.99856 per frame;
* ``discounted_returns`` -- the reverse scan ``G_t = r_t + gamma_t * G_{t+1}`` with ``G_T`` =
  bootstrap (``tf/rl_lib.py:7-22``, ``jax/rl_lib.py:13-37``), here with time on the *last*
  axis (batch-major ``[B, T]``);
* ``player_respawn`` / ``respawn_happened`` -- the rising edge of ``ON_HALO_DESCENT`` for
  either player (``tf/value_function.py:19-24``), the ``discount_on_death`` mask;
* :class:`ReturnsConfig` and :func:`value_discounts` -- the per-transition discounts:
  ``discount`` everywhere, ``discount_on_death`` on respawn transitions and, by default, zero
  on transitions into a reset row (PLAN.md §3.6 deviation 1: slippi-ai's separate value
  function does not zero them, ``tf/value_function.py:74``; its built-in policy head does,
  ``tf/policies.py:104``);
* ``mean_and_variance`` / ``unexplained_variance`` -- the ``uev`` metric
  (``tf/tf_utils.py:19-22``, ``tf/value_function.py:92-93``).

lambda = 1 throughout (no GAE in any reference).  Everything is float32.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch

ACTION_ON_HALO_DESCENT: Final[int] = 0x0C
"""libmelee ``melee.enums.Action.ON_HALO_DESCENT`` (the respawn platform, ``tf/value_function.py:23``)."""

VARIANCE_EPSILON: Final[float] = 1.0e-8


def discount_from_halflife(halflife_seconds: float, fps: float = 60.0) -> float:
    """Per-frame discount with the given reward half-life in seconds (``jax/rl_lib.py:9-11``)."""

    if not halflife_seconds > 0.0 or not fps > 0.0:
        raise ValueError("halflife_seconds and fps must be positive")
    return float(0.5 ** (1.0 / (halflife_seconds * fps)))


@dataclass(frozen=True)
class ReturnsConfig:
    """``[learner.returns]``: how value targets are built (PLAN.md §3.7).

    ``reward_halflife`` in seconds gives the discount; ``discount_on_death`` replaces it on
    transitions where either player respawns (``None`` disables, slippi-ai's default);
    ``zero_discount_on_reset`` zeroes the discount of transitions into a reset row so that no
    value leaks across games (deviation 1 of PLAN.md §3.6).
    """

    reward_halflife: float = 4.0
    discount_on_death: float | None = None
    zero_discount_on_reset: bool = True

    def __post_init__(self) -> None:
        if not self.reward_halflife > 0.0:
            raise ValueError("reward_halflife must be positive (seconds)")
        if self.discount_on_death is not None and not 0.0 <= self.discount_on_death <= 1.0:
            raise ValueError("discount_on_death must lie in [0, 1]")

    @property
    def discount(self) -> float:
        return discount_from_halflife(self.reward_halflife)


def discounted_returns(
    rewards: torch.Tensor,
    discounts: torch.Tensor | float,
    bootstrap: torch.Tensor | float,
) -> torch.Tensor:
    """``G_t = r_t + gamma_t * G_{t+1}`` for ``t = T-1 .. 0`` with ``G_T = bootstrap``.

    ``rewards`` is ``[..., T]`` (time on the last axis), ``discounts`` a float or a tensor
    broadcastable to it, ``bootstrap`` a float or a tensor broadcastable to ``[...]``.  Returns
    float32 ``[..., T]``; gradients flow through ``bootstrap`` if it carries them (callers
    detach it, as ``tf/value_function.py:83`` does with ``stop_gradient``).
    """

    values = rewards.to(torch.float32)
    gammas = torch.as_tensor(discounts, dtype=torch.float32, device=values.device)
    gammas = torch.broadcast_to(gammas, values.shape)
    acc = torch.as_tensor(bootstrap, dtype=torch.float32, device=values.device)
    acc = torch.broadcast_to(acc, values.shape[:-1])
    steps = values.shape[-1]
    if steps == 0:
        return values.clone()
    outputs: list[torch.Tensor] = []
    for index in reversed(range(steps)):
        acc = values[..., index] + gammas[..., index] * acc
        outputs.append(acc)
    outputs.reverse()
    return torch.stack(outputs, dim=-1)


def player_respawn(action: torch.Tensor) -> torch.Tensor:
    """Rising edge of ``ON_HALO_DESCENT`` along the last axis: ``[..., L] -> [..., L-1]`` bool."""

    return (action[..., :-1] != ACTION_ON_HALO_DESCENT) & (action[..., 1:] == ACTION_ON_HALO_DESCENT)


def respawn_happened(frames: Mapping[str, Any]) -> torch.Tensor:
    """Either player respawns on the transition (``tf/value_function.py:76-78``), ``[..., L-1]`` bool."""

    return player_respawn(frames["p0"]["action"]) | player_respawn(frames["p1"]["action"])


def value_discounts(
    config: ReturnsConfig,
    rewards: torch.Tensor,
    *,
    is_resetting: torch.Tensor | None = None,
    frames: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Per-transition discounts ``[..., T]`` for ``rewards [..., T]``.

    ``is_resetting`` holds the reset flags of the ``T + 1`` window rows (``[..., T+1]``); the
    transition ``t -> t+1`` gets discount 0 where ``is_resetting[..., t+1]`` (when
    ``config.zero_discount_on_reset``).  ``frames`` are the same ``T + 1`` rows, needed only
    for ``discount_on_death``.
    """

    discounts = torch.full_like(rewards, config.discount, dtype=torch.float32)
    if config.discount_on_death is not None:
        if frames is None:
            raise ValueError("discount_on_death needs the rollout frames (T + 1 rows)")
        respawn = respawn_happened(frames)
        if respawn.shape != discounts.shape:
            raise ValueError(f"frames must cover {discounts.shape[-1] + 1} rows, got {respawn.shape[-1] + 1}")
        discounts = torch.where(respawn, config.discount_on_death, discounts)
    if config.zero_discount_on_reset:
        if is_resetting is None:
            raise ValueError("zero_discount_on_reset needs the is_resetting flags (T + 1 rows)")
        expected = (*discounts.shape[:-1], discounts.shape[-1] + 1)
        if tuple(is_resetting.shape) != expected:
            raise ValueError(f"is_resetting must have shape {expected}, got {tuple(is_resetting.shape)}")
        discounts = torch.where(is_resetting[..., 1:].to(torch.bool), 0.0, discounts)
    return discounts


def mean_and_variance(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and population variance over every element (``tf/tf_utils.py:19-22``)."""

    values = values.to(torch.float32)
    mean = values.mean()
    variance = ((values - mean) ** 2).mean()
    return mean, variance


def unexplained_variance(squared_errors: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``mean(squared_errors) / (var(targets) + 1e-8)``, the ``uev`` metric (``tf/value_function.py:92``)."""

    _, variance = mean_and_variance(targets)
    return squared_errors.to(torch.float32).mean() / (variance + VARIANCE_EPSILON)


__all__ = [
    "ACTION_ON_HALO_DESCENT",
    "VARIANCE_EPSILON",
    "ReturnsConfig",
    "discount_from_halflife",
    "discounted_returns",
    "mean_and_variance",
    "player_respawn",
    "respawn_happened",
    "unexplained_variance",
    "value_discounts",
]
