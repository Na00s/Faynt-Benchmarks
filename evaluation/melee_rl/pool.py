"""The past-self pool (STRENGTH-PLAN.md §6; 11 Sep 2026, built during strength gate 1).

A ``pool*<n>`` mix arm (:func:`melee_rl.opponents.parse_mix_spec`) trains port 0 against one of the run's
own ``step_<n>.pt`` snapshots on port 1.  The whole lockstep group plays the same snapshot, so a group
stays one opponent.  Every ``[opponent.pool] interval`` learner steps :meth:`PoolOpponent.maybe_draw`
picks a snapshot at least ``min_age`` steps old and loads its policy weights *in place* into the
opponent's frozen policy (``load_state_dict`` copies into the same parameter tensors, as
``SelfOpponent.refresh`` does): no worker or environment is rebuilt, and the opponent's cache and delay
queue carry on across the swap -- the opponent is not trained, so exactness does not matter.  With no
snapshot old enough (a fresh run, no checkpoint directory) the arm plays the current self, refreshed at
every draw.

The draw is prioritised fictitious self-play (AlphaStar's PFSP): snapshot ``j`` is drawn with weight
``f_hard(x_j) = (1 - x_j)^2``, where ``x_j`` is the arm's KO share against it -- the fraction of the
stocks lost in those games that the student took, Laplace-smoothed (:func:`ko_share`) -- floored at
``floor`` so no snapshot starves; a snapshot never played joins at the highest weight present (OpenAI
Five's rule for a new past self).  Each time a snapshot is drawn again its counts are multiplied by
``decay``, so the estimate follows the moving student.  ``max_candidates`` thins a long ladder to evenly
spaced snapshots, the oldest and the newest kept.

The smallest KO share over the scored snapshots is AlphaStar's forgetting measure: logged every learner
step as ``arm/<name>/min_ko_share``, it is the strength plan's in-loop cycling detector, beside
``snapshot_step``, ``snapshot_age`` and ``ko_share`` of the snapshot being played
(:meth:`PoolOpponent.metrics`).
"""

from __future__ import annotations

import logging as std_logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from melee_rl import checkpoint as checkpoint_lib
from melee_rl import reward as reward_lib
from melee_rl.opponents import PolicyOpponent, PoolConfig
from melee_rl.protocol import PolicyProtocol
from melee_rl.trajectory import Trajectory

LOG = std_logging.getLogger("melee_rl.pool")
SNAPSHOT_NAME: Final[re.Pattern[str]] = re.compile(r"^step_(\d+)\.pt$")
DRAW_SEED_OFFSET: Final[int] = 1_000_003
"""The draw generator is seeded with the opponent's sampling seed plus this: two streams from one seed."""


def snapshot_step(name: str) -> int:
    """The learner step in a ``step_<n>.pt`` file name."""

    match = SNAPSHOT_NAME.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not a step_<n>.pt snapshot name")
    return int(match.group(1))


def list_snapshots(directory: str | Path | None, step: int, *, min_age: int = 0) -> list[tuple[int, Path]]:
    """``(step, path)`` of every ``step_<n>.pt`` in ``directory`` from step ``step - min_age`` or
    earlier, oldest first; nothing for a missing directory (a fresh run) or ``None`` (no checkpoint
    directory)."""

    if directory is None:
        return []
    base = Path(directory)
    if not base.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in base.iterdir():
        match = SNAPSHOT_NAME.match(path.name)
        if match is not None and int(match.group(1)) <= step - min_age:
            found.append((int(match.group(1)), path))
    return sorted(found)


def thin_snapshots(snapshots: Sequence[tuple[int, Path]], limit: int) -> list[tuple[int, Path]]:
    """At most ``limit`` evenly spaced entries of ``snapshots`` -- index ``i * (n - 1) / (limit - 1)``
    rounded half up, so the oldest and the newest are always kept; ``limit = 0`` keeps them all."""

    if limit < 0 or limit == 1:
        raise ValueError(f"limit must be 0 (keep all) or >= 2, got {limit}")
    count = len(snapshots)
    if limit == 0 or count <= limit:
        return list(snapshots)
    span = limit - 1
    indices = sorted({(2 * index * (count - 1) + span) // (2 * span) for index in range(limit)})
    return [snapshots[index] for index in indices]


def ko_share(kos: float, deaths: float) -> float:
    """The student's share of the stocks that fell, Laplace-smoothed: ``(kos + 1) / (kos + deaths + 2)``."""

    return (kos + 1.0) / (kos + deaths + 2.0)


def pfsp_probabilities(shares: Sequence[float | None], *, floor: float) -> list[float]:
    """PFSP draw probabilities: weight ``max((1 - x)^2, floor)`` per KO share ``x`` (AlphaStar's ``f_hard``);
    ``None`` (never played) takes the highest weight present, 1 when nothing has been played yet."""

    if not shares:
        raise ValueError("pfsp_probabilities needs at least one snapshot")
    weights = [None if share is None else max((1.0 - share) ** 2, floor) for share in shares]
    known = [weight for weight in weights if weight is not None]
    top = max(known) if known else 1.0
    filled = [top if weight is None else weight for weight in weights]
    total = sum(filled)
    return [weight / total for weight in filled]


def pool_counts(trajectories: Sequence[Trajectory]) -> tuple[float, float]:
    """``(kos, deaths)``: the stocks the student took and lost over the rollout rows of one-port
    ``trajectories`` (port 0 is the student's perspective; a death is a rising edge of ``reward.deaths``)."""

    kos = deaths = 0
    for trajectory in trajectories:
        frames = trajectory.rollout_frames()
        kos += int(reward_lib.deaths(frames["p1"]["action"]).sum())
        deaths += int(reward_lib.deaths(frames["p0"]["action"]).sum())
    return float(kos), float(deaths)


class PoolOpponent(PolicyOpponent):
    """Past selves on ``port``: the run's ``step_<n>.pt`` snapshots, drawn by PFSP and loaded in place.

    Starts as a frozen copy of ``student`` (``clone_frozen``, so the fast-path flags come along) and stays
    that until :meth:`maybe_draw` finds a snapshot old enough.  :attr:`table` maps a snapshot's file name
    to ``[kos, deaths, draws]`` (the counts decayed per draw); :attr:`current` is the file name being
    played, ``None`` for the current self.  ``directory`` is the run's checkpoint directory (``None``: the
    current self forever).
    """

    def __init__(
        self,
        student: PolicyProtocol,
        batch_size: int,
        *,
        directory: str | Path | None,
        config: PoolConfig,
        port: int = 1,
        delay_frames: int = 0,
        temperature: float = 1.0,
        seed: int = 1,
    ) -> None:
        super().__init__(
            student.clone_frozen(),
            batch_size,
            port=port,
            delay_frames=delay_frames,
            temperature=temperature,
            seed=seed,
        )
        self.directory = None if directory is None else Path(directory)
        self.config = config
        self.current: str | None = None
        self.table: dict[str, list[float]] = {}
        self.draws = 0
        self.reloads = 0
        self.candidates = 0
        self.probability = 1.0
        self._last_counts = (0.0, 0.0)
        self._draw_generator = torch.Generator().manual_seed(seed + DRAW_SEED_OFFSET)

    @property
    def snapshot_step(self) -> int | None:
        """The learner step of the snapshot being played (``None``: the current self)."""

        return None if self.current is None else snapshot_step(self.current)

    def maybe_draw(self, step: int, student: PolicyProtocol) -> bool:
        """At every ``interval``-th learner step, draw what the arm plays next and load it (True: drew)."""

        if step % self.config.interval:
            return False
        snapshots = thin_snapshots(
            list_snapshots(self.directory, step, min_age=self.config.min_age), self.config.max_candidates
        )
        self.candidates = len(snapshots)
        self.draws += 1
        if not snapshots:
            self.current = None
            self.probability = 1.0
            self.policy.set_state(student.get_state())  # the current self, as a self-opponent refresh
            self.reloads += 1
            return True
        names = [path.name for _, path in snapshots]
        probabilities = pfsp_probabilities([self._share(name) for name in names], floor=self.config.floor)
        weights = torch.tensor(probabilities, dtype=torch.float64)
        index = int(torch.multinomial(weights, 1, generator=self._draw_generator))
        name = names[index]
        self.probability = probabilities[index]
        entry = self.table.setdefault(name, [0.0, 0.0, 0.0])
        entry[0] *= self.config.decay
        entry[1] *= self.config.decay
        entry[2] += 1.0
        if name != self.current:
            self._load(snapshots[index][1])
            self.current = name
        return True

    def _share(self, name: str) -> float | None:
        entry = self.table.get(name)
        return None if entry is None else ko_share(entry[0], entry[1])

    def _load(self, path: Path) -> None:
        """Copy a snapshot's policy weights into the frozen policy (the same parameter tensors)."""

        self.policy.set_state(checkpoint_lib.load_bc_checkpoint(path).state_dict)
        self.reloads += 1

    def refresh(self, student: PolicyProtocol) -> None:
        """Reload what the arm is playing -- the resume path (``Trainer._restore``): the current snapshot's
        weights, or the student's when it plays the current self or the snapshot file is gone."""

        path = None if self.current is None or self.directory is None else self.directory / self.current
        if path is not None and path.is_file():
            self._load(path)
            return
        if self.current is not None:
            LOG.warning("pool snapshot %s is gone: the current self until the next draw", self.current)
            self.current = None
        self.policy.set_state(student.get_state())

    def record(self, kos: float, deaths: float) -> None:
        """One learner step of the arm: the stocks the student took (``kos``) and lost (``deaths``)."""

        self._last_counts = (float(kos), float(deaths))
        if self.current is None:
            return  # the current self is not a past self: nothing to score
        entry = self.table.setdefault(self.current, [0.0, 0.0, 1.0])
        entry[0] += float(kos)
        entry[1] += float(deaths)

    def metrics(self, step: int) -> dict[str, float]:
        """The arm's pool keys (``arm/<name>/<key>``) for the learner step that rolled out at ``step``."""

        snapshot = self.snapshot_step
        kos, deaths = self._last_counts
        metrics = {
            "snapshot_step": -1.0 if snapshot is None else float(snapshot),
            "snapshot_age": 0.0 if snapshot is None else float(step - snapshot),
            "ko_share": ko_share(kos, deaths),
            "probability": self.probability,
            "candidates": float(self.candidates),
            "scored": float(len(self.table)),
            "reloads": float(self.reloads),
        }
        shares = [ko_share(entry[0], entry[1]) for entry in self.table.values()]
        if shares:
            metrics["min_ko_share"] = min(shares)
            metrics["mean_ko_share"] = sum(shares) / len(shares)
        return metrics

    def pool_state(self) -> dict[str, Any]:
        """The bookkeeping the trainer stores in ``loop_state["pools"]`` (plain containers and one tensor)."""

        return {
            "current": self.current,
            "table": {name: list(entry) for name, entry in self.table.items()},
            "draws": self.draws,
            "reloads": self.reloads,
            "candidates": self.candidates,
            "probability": self.probability,
            "draw_generator": self._draw_generator.get_state(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Take back :meth:`pool_state`; :meth:`refresh` then loads the weights of what was being played."""

        current = state.get("current")
        self.current = None if current is None else str(current)
        table = state.get("table") or {}
        self.table = {str(name): [float(value) for value in entry] for name, entry in dict(table).items()}
        self.draws = int(state.get("draws", 0))
        self.reloads = int(state.get("reloads", 0))
        self.candidates = int(state.get("candidates", 0))
        self.probability = float(state.get("probability", 1.0))
        generator = state.get("draw_generator")
        if isinstance(generator, torch.Tensor):
            self._draw_generator.set_state(generator)


__all__ = [
    "DRAW_SEED_OFFSET",
    "SNAPSHOT_NAME",
    "PoolOpponent",
    "ko_share",
    "list_snapshots",
    "pfsp_probabilities",
    "pool_counts",
    "snapshot_step",
    "thin_snapshots",
]
