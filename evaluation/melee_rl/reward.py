# Ported from slippi-ai slippi_ai/reward.py@577965a7731dc53e3472ea63d9e9853a4e9d65fa (MIT)
"""slippi-ai reward semantics over frame trees (PLAN.md §3.8/§5, R9; P6a: every term).

Re-implementation in PyTorch of ``slippi_ai/reward.py``:

* a death is the rising edge of the dying action states ``action <= 0xA``
  (``is_dying`` / ``process_deaths``, ``reward.py:11-19``);
* damage is the positive percent increment ``max(percent[t+1] - percent[t], 0)``
  (``process_damages``, ``reward.py:21-23``);
* a player's reward is ``-(deaths + damage_ratio * damages)`` plus ``nana_ratio`` times the
  same for Nana where ``nana.exists[t+1]`` (``reward.py:144-155``);
* a *bad ledge grab* is the rising edge of ``action == EDGE_CATCHING`` unless, at the source
  frame, the opponent is on the centre side of the player (``(player.x < opponent.x) == (player.x
  < 0)``) or invulnerable; it costs ``ledge_grab_penalty`` (``reward.py:24-41, 155-156``);
* *stalling* is being further than ``stalling_threshold`` outside the box ``[-edge_x, edge_x] x
  [0, MAX_STALLING_Y]`` (``edge_x`` from the stage id, unknown stages 100) at the destination
  frame; it costs ``stalling_penalty / 60`` per frame (``reward.py:65-92, 158-159``);
* the *approaching factor* is the player's displacement projected on the unit direction to the
  opponent at the source frame, zero on the respawn edge (``dying[t] & ~dying[t+1]``), weighted by
  ``approaching_factor`` (``reward.py:43-63, 161``);
* the reward is zero-sum, ``p0 - p1`` (``reward.py:166``; every term enters for both players),
  and must lie in ``(-2, 2)`` (``reward.py:169-170``);
* the learner zeroes the reward of transitions into a reset row (``rl/learner.py:103-105``),
  :func:`zero_at_resets`.

Conventions: batch-major; the time axis is the *last* dimension of the scalar leaves, so
``frames`` with leaves ``[B, L]`` give rewards ``[B, L-1]`` (slippi-ai's arrays are
time-major ``[T]``; only the axis differs).  Everything is computed in float32 from
integer percents (``uint16`` upstream, ``int64`` here).  One documented deviation: the stage
is read per frame from the tree's ``stage`` leaf (slippi-ai uses ``stage[0]`` of a chunk); in a
single game the two agree.  Terms whose weight is 0 are not evaluated (their leaves --
``x`` / ``y`` / ``invulnerable`` / ``stage`` -- need not exist in hand-built trees).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import torch

DYING_ACTION_MAX: Final[int] = 0xA
"""Action ids ``<= 0xA`` are the dying states (``reward.py:9-12``)."""

EDGE_CATCHING: Final[int] = 0xFC
"""``melee.Action.EDGE_CATCHING`` -- the initial ledge-grab state (``reward.py:25``)."""

FRAMES_PER_MINUTE: Final[int] = 60 * 60

MAX_STALLING_Y: Final[float] = 60.0
"""Above this height counts as offstage for stalling; the highest top platform is 54.4 on Battlefield
(``reward.py:70-72``)."""

DEFAULT_STALLING_THRESHOLD: Final[float] = 20.0
"""``reward.py:85``."""

UNKNOWN_STAGE_EDGE_X: Final[float] = 100.0
"""``get_edge_x`` default for stage ids without an entry (``reward.py:68``)."""

EDGE_X: Final[Mapping[int, float]] = MappingProxyType(
    {
        24: 71.3078536987,  # Battlefield
        25: 88.4735488892,  # Final Destination
        26: 80.1791534424,  # Dreamland
        8: 66.2554016113,  # Fountain of Dreams
        18: 90.657852,  # Pokemon Stadium
        6: 58.907848,  # Yoshi's Story
    }
)
"""libmelee ``stages.EDGE_POSITION`` keyed by the internal stage id (``melee/stages.py:22-29``,
``melee/enums.py:8-13``; ``reward.py:65-68``) -- the x coordinate of the ledge edge."""


@dataclass(frozen=True)
class RewardConfig:
    """``slippi_ai.reward.RewardConfig`` (``reward.py:110-117``); ``stalling_penalty`` is per second."""

    damage_ratio: float = 0.01
    nana_ratio: float = 0.5
    ledge_grab_penalty: float = 0.0
    approaching_factor: float = 0.0
    stalling_penalty: float = 0.0
    stalling_threshold: float = DEFAULT_STALLING_THRESHOLD

    def __post_init__(self) -> None:
        for name in (
            "damage_ratio",
            "nana_ratio",
            "ledge_grab_penalty",
            "approaching_factor",
            "stalling_penalty",
            "stalling_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        for name in ("damage_ratio", "nana_ratio", "ledge_grab_penalty", "stalling_penalty"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not self.stalling_threshold > 0.0:
            raise ValueError("stalling_threshold must be positive")


# ---------------------------------------------------------------------------
# per-frame and per-transition signals
# ---------------------------------------------------------------------------


def is_dying(action: torch.Tensor) -> torch.Tensor:
    """``action <= 0xA`` (``reward.py:9-12``)."""

    return action <= DYING_ACTION_MAX


def deaths(action: torch.Tensor) -> torch.Tensor:
    """Rising edges of :func:`is_dying` along the last dimension: ``[..., L] -> [..., L-1]`` bool."""

    dying = is_dying(action)
    return ~dying[..., :-1] & dying[..., 1:]


def damages(percent: torch.Tensor) -> torch.Tensor:
    """``max(percent[t+1] - percent[t], 0)`` in float32: ``[..., L] -> [..., L-1]``."""

    values = percent.to(torch.float32)
    return torch.clamp(values[..., 1:] - values[..., :-1], min=0.0)


def grabbed_ledge(action: torch.Tensor) -> torch.Tensor:
    """Rising edges of ``action == EDGE_CATCHING``: ``[..., L] -> [..., L-1]`` bool (``reward.py:24-26``)."""

    grabbing = action == EDGE_CATCHING
    return ~grabbing[..., :-1] & grabbing[..., 1:]


def bad_ledge_grabs(player: Mapping[str, Any], opponent: Mapping[str, Any]) -> torch.Tensor:
    """Ledge grabs that are penalised (``reward.py:28-41``): ``[..., L-1]`` bool.

    A grab is fine when the opponent is on the centre side of the player (it is being edge-guarded
    back) or invulnerable (just respawned); both read at the source frame of the transition.
    """

    grabs = grabbed_ledge(player["action"])
    player_x = player["x"]
    opponent_right = player_x < opponent["x"]
    centre_right = player_x < 0.0
    opponent_towards_centre = opponent_right == centre_right
    invulnerable = opponent["invulnerable"].to(torch.bool)
    return grabs & opponent_towards_centre[..., :-1] & ~invulnerable[..., :-1]


def normalize(xys: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Unit vectors along the last axis, ``xys / (|xys| + epsilon)`` (``reward.py:43-45``)."""

    radius = torch.sqrt(torch.sum(xys * xys, dim=-1, keepdim=True))
    return xys / (radius + epsilon)


def approaching_factor(player: Mapping[str, Any], opponent: Mapping[str, Any]) -> torch.Tensor:
    """How much the player moved towards the opponent on each transition (``reward.py:47-63``).

    ``(xy[t+1] - xy[t]) . normalize(opp_xy[t] - xy[t])``, zero where the player respawns
    (``dying[t] & ~dying[t+1]``: the teleport to the respawn platform).  ``[..., L-1]`` float32.
    """

    xy = torch.stack([player["x"].to(torch.float32), player["y"].to(torch.float32)], dim=-1)
    opponent_xy = torch.stack([opponent["x"].to(torch.float32), opponent["y"].to(torch.float32)], dim=-1)
    velocity = xy[..., 1:, :] - xy[..., :-1, :]
    direction = normalize(opponent_xy - xy)
    factor = torch.sum(velocity * direction[..., :-1, :], dim=-1)
    dying = is_dying(player["action"])
    respawning = dying[..., :-1] & ~dying[..., 1:]
    return torch.where(respawning, torch.zeros_like(factor), factor)


def stage_edge_x(stage: torch.Tensor) -> torch.Tensor:
    """:data:`EDGE_X` of every stage id in ``stage`` (``UNKNOWN_STAGE_EDGE_X`` elsewhere), float32."""

    result = torch.full(stage.shape, UNKNOWN_STAGE_EDGE_X, dtype=torch.float32, device=stage.device)
    for stage_id, edge in EDGE_X.items():
        result = torch.where(stage == stage_id, edge, result)
    return result


def amount_offstage(player: Mapping[str, Any], stage: torch.Tensor) -> torch.Tensor:
    """Distance outside the box ``[-edge_x, edge_x] x [0, MAX_STALLING_Y]`` per frame (``reward.py:74-82``).

    ``stage`` broadcasts against the player's ``x`` / ``y`` (a per-frame leaf, or one id).
    """

    x = player["x"].to(torch.float32)
    y = player["y"].to(torch.float32)
    dx = torch.clamp(x.abs() - stage_edge_x(stage), min=0.0)
    below = -torch.clamp(y, max=0.0)
    above = torch.clamp(y - MAX_STALLING_Y, min=0.0)
    dy = torch.maximum(below, above)
    return torch.sqrt(dx * dx + dy * dy)


def is_stalling_offstage(
    player: Mapping[str, Any],
    stage: torch.Tensor,
    threshold: float = DEFAULT_STALLING_THRESHOLD,
) -> torch.Tensor:
    """``amount_offstage > threshold`` per frame, ``[..., L]`` bool (``reward.py:87-92``)."""

    return amount_offstage(player, stage) > threshold


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------


def player_base_reward(player: Mapping[str, Any], damage_ratio: float) -> torch.Tensor:
    """``-(deaths + damage_ratio * damages)`` for one player (or Nana) tree (``reward.py:142-145``)."""

    return -(deaths(player["action"]).to(torch.float32) + damage_ratio * damages(player["percent"]))


def player_reward(
    player: Mapping[str, Any],
    opponent: Mapping[str, Any] | None,
    stage: torch.Tensor | None,
    config: RewardConfig,
) -> torch.Tensor:
    """One player's reward, every term (``reward.py:147-163``), ``[..., L-1]`` float32.

    ``opponent`` is needed by the ledge and approach terms, ``stage`` by the stalling term; a term with
    weight 0 is skipped (and its inputs may be ``None``).
    """

    reward = player_base_reward(player, config.damage_ratio)
    nana = player.get("nana")
    if config.nana_ratio != 0.0 and nana is not None:
        nana_reward = config.nana_ratio * player_base_reward(nana, config.damage_ratio)
        exists = nana["exists"][..., 1:].to(torch.bool)
        reward = reward + torch.where(exists, nana_reward, torch.zeros_like(nana_reward))
    if config.ledge_grab_penalty != 0.0:
        if opponent is None:
            raise ValueError("ledge_grab_penalty needs the opponent tree")
        reward = reward - config.ledge_grab_penalty * bad_ledge_grabs(player, opponent).to(torch.float32)
    if config.stalling_penalty != 0.0:
        if stage is None:
            raise ValueError("stalling_penalty needs the frame tree's stage leaf")
        stalling = is_stalling_offstage(player, stage, config.stalling_threshold)[..., 1:]
        reward = reward - (config.stalling_penalty / 60.0) * stalling.to(torch.float32)
    if config.approaching_factor != 0.0:
        if opponent is None:
            raise ValueError("approaching_factor needs the opponent tree")
        reward = reward + config.approaching_factor * approaching_factor(player, opponent)
    return reward


def compute_rewards(
    frames: Mapping[str, Any],
    config: RewardConfig | None = None,
    *,
    check_bounds: bool = True,
) -> torch.Tensor:
    """Zero-sum reward ``p0 - p1`` for every transition: leaves ``[..., L]`` -> ``[..., L-1]`` float32.

    ``check_bounds`` enforces slippi-ai's sanity check ``-2 < reward < 2``
    (``reward.py:169-170``) with a ``ValueError``.
    """

    config = RewardConfig() if config is None else config
    p0, p1 = frames["p0"], frames["p1"]
    stage = frames.get("stage")
    rewards = player_reward(p0, p1, stage, config) - player_reward(p1, p0, stage, config)
    if check_bounds and rewards.numel() and not bool(((rewards > -2.0) & (rewards < 2.0)).all()):
        raise ValueError(
            f"rewards must lie in (-2, 2), got range [{float(rewards.min())}, {float(rewards.max())}]"
        )
    return rewards


def zero_at_resets(rewards: torch.Tensor, is_resetting: torch.Tensor) -> torch.Tensor:
    """Zero the reward of every transition into a reset row (``rl/learner.py:103-105``).

    ``rewards`` is ``[..., L-1]`` for the transitions of a ``[..., L]`` window whose reset flags
    are ``is_resetting``; the reward of ``s_t -> s_{t+1}`` is dropped where ``is_resetting[t+1]``.
    """

    if is_resetting.shape[:-1] != rewards.shape[:-1] or is_resetting.shape[-1] != rewards.shape[-1] + 1:
        raise ValueError(
            f"is_resetting must have shape {(*rewards.shape[:-1], rewards.shape[-1] + 1)}, "
            f"got {tuple(is_resetting.shape)}"
        )
    return torch.where(is_resetting[..., 1:].to(torch.bool), torch.zeros_like(rewards), rewards)


def ko_diff(frames: Mapping[str, Any]) -> torch.Tensor:
    """``p1 deaths - p0 deaths`` per transition, ``[..., L-1]`` long (``reward.py:118-123``)."""

    return deaths(frames["p1"]["action"]).long() - deaths(frames["p0"]["action"]).long()


def player_stats(
    player: Mapping[str, Any],
    opponent: Mapping[str, Any] | None = None,
    stage: torch.Tensor | None = None,
    *,
    stalling_threshold: float = DEFAULT_STALLING_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Per-minute rates and fractions of one player tree (``reward.py:175-203``).

    Always ``deaths`` and ``damages`` (per minute; plus ``nana_deaths`` / ``nana_damages`` where the
    tree has Nana); with ``opponent`` also ``ledge_grabs`` (bad grabs per minute) and
    ``approaching_factor`` (mean); with ``stage`` also ``stalling`` (the fraction of frames spent
    stalling offstage).  Scalars, averaged over every transition / frame of the tree.
    """

    stats = {
        "deaths": deaths(player["action"]).to(torch.float32).mean() * FRAMES_PER_MINUTE,
        "damages": damages(player["percent"]).mean() * FRAMES_PER_MINUTE,
    }
    if opponent is not None:
        stats["ledge_grabs"] = bad_ledge_grabs(player, opponent).to(torch.float32).mean() * FRAMES_PER_MINUTE
        stats["approaching_factor"] = approaching_factor(player, opponent).mean()
    if stage is not None:
        stats["stalling"] = is_stalling_offstage(player, stage, stalling_threshold).to(torch.float32).mean()
    nana = player.get("nana")
    if nana is not None:
        exists = nana["exists"][..., 1:].to(torch.bool)
        zero = torch.zeros((), dtype=torch.float32, device=exists.device)
        nana_deaths = torch.where(exists, deaths(nana["action"]).to(torch.float32), zero)
        nana_damages = torch.where(exists, damages(nana["percent"]), zero)
        stats["nana_deaths"] = nana_deaths.mean() * FRAMES_PER_MINUTE
        stats["nana_damages"] = nana_damages.mean() * FRAMES_PER_MINUTE
    return stats


__all__ = [
    "DEFAULT_STALLING_THRESHOLD",
    "DYING_ACTION_MAX",
    "EDGE_CATCHING",
    "EDGE_X",
    "FRAMES_PER_MINUTE",
    "MAX_STALLING_Y",
    "UNKNOWN_STAGE_EDGE_X",
    "RewardConfig",
    "amount_offstage",
    "approaching_factor",
    "bad_ledge_grabs",
    "compute_rewards",
    "damages",
    "deaths",
    "grabbed_ledge",
    "is_dying",
    "is_stalling_offstage",
    "ko_diff",
    "normalize",
    "player_base_reward",
    "player_reward",
    "player_stats",
    "stage_edge_x",
    "zero_at_resets",
]
