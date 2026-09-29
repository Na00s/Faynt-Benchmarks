"""In-loop evaluation of the student against fixed opponents (PLAN.md §6 P6b item 0; decided 22 Aug 2026).

Self-play metrics cannot show progress (the reward is zero-sum, so every mean is 0 by construction), so the
training loop periodically plays a *frozen copy* of the current policy for ``frames`` environment frames
against each configured opponent in fresh evaluation environments and logs slippi-ai's ``player_stats``
(``reward.py:175-203``) as ``eval/<opponent>/<stat>`` to the same W&B run at the current learner step:

* triggers: ``at_start`` (step 0 of a fresh run), every ``interval_seconds`` of wall time and / or every
  ``interval_steps`` learner steps (checked after a step completes), ``at_end`` (unless the last step was just
  evaluated);
* opponents: ``cpu<level>`` (the environment's own CPU, ``("policy", "cpu")``), ``best`` (the run's own
  ``best.pt`` as an ``OtherOpponent`` on port 1 -- skipped until it exists) and ``other:<path>`` (a fixed
  checkpoint, ours or Ali's format) and ``mimic`` (2 Sep 2026: the released MIMIC Fox of P12 on port 2,
  through :mod:`melee_rl.mimic` -- the outside yardstick, since CPU 9 mis-ranked our runs; it reads the
  raw gamestates, so its environment is built in-process) and ``slippi:<release>`` (17 Sep 2026: a released
  slippi-ai agent through the PyTorch port, :mod:`melee_rl.slippi_ai_torch`, on worker envs booted with the
  release's own character, logged as ``eval/<release>/*``); one evaluation environment per opponent,
  booted on demand and closed afterwards (8 Dolphins boot in ≈ 30 s, ENV.md §5.3), ports offset from
  the training Dolphins;
* ``metric`` of ``metric_opponent`` (default ``damage_dealt_per_minute`` vs ``cpu9``) selects ``best.pt``
  (a full RL checkpoint next to ``latest.pt``, so it resumes and loads as an ``other`` opponent);
  ``eval_history.json`` in the run directory keeps every record; there is no stop rule (user decision);
* the evaluation worker runs in ``ring`` context mode (the model's native rolling cache, no re-prime -- the
  deployment path; exactness is irrelevant here) with the training reward config and delay, the configured
  ``temperature`` (1.0 = the training distribution) and a fixed seed per evaluation, so evaluations of the
  same weights are comparable; an ``other`` / ``best`` rival plays at the delay its file was trained at,
  ``other:<path>@d<n>`` at ``n`` (11 Sep 2026: the naive fork point of a delayed run is the base at the run's
  delay).

slippi-ai evaluates offline (``scripts/run_evaluator.py``, ``scripts/eval_two.py``) and logs self-play
``ko_diff`` only for non-mirror character matchups (``rl/run_lib.py:512-525``); W&B can have a run open in
one process at a time and needs monotonic steps, hence the in-loop design (an offline entrypoint can reuse
:class:`Evaluator` later).
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from melee_rl import reward as reward_lib
from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.env.dummy import DummyMeleeEnv
from melee_rl.env.protocol import EnvProtocol
from melee_rl.frames import tree_cat, tree_index
from melee_rl.mimic import MimicPlayer, MimicRuntime, load_runtime
from melee_rl.opponents import (
    CpuOpponent,
    MimicOpponent,
    Opponent,
    OtherOpponent,
    rival_delay,
    split_rival_delay,
)
from melee_rl.protocol import PolicyProtocol
from melee_rl.slippi_ai_agent import RELEASED_MODELS
from melee_rl.trajectory import Trajectory

if TYPE_CHECKING:
    from melee_rl.config import RLConfig
    from model import ModelConfig

EVAL_PREFIX: Final[str] = "eval/"
BEST_NAME: Final[str] = "best.pt"
HISTORY_NAME: Final[str] = "eval_history.json"
OPPONENT_KINDS: Final[tuple[str, ...]] = ("cpu", "best", "other", "mimic", "slippi")
EVAL_STATS: Final[tuple[str, ...]] = (
    "reward_per_frame",
    "ko_diff_per_minute",
    "kos_per_minute",
    "damage_dealt_per_minute",
    "damage_taken_per_minute",
    "deaths_per_minute",
    "ledge_grabs_per_minute",
    "stalling_fraction",
    "approaching_factor",
    "game_starts",
    "frames",
    "seconds",
    "fps",
)
"""Per-opponent statistics of one evaluation (``eval/<opponent>/<stat>``); the first nine are slippi-ai's
``player_stats`` of the student over the evaluation rollouts, ``game_starts`` counts reset rows (boots,
restarts, relaunches)."""
SELECTABLE_STATS: Final[tuple[str, ...]] = EVAL_STATS[:9]
"""Statistics ``EvalConfig.metric`` may select ``best.pt`` by."""
EVAL_PORT_GAP: Final[int] = 8
"""Slippi ports of the evaluation Dolphins start this many ports after the training Dolphins' range."""
_CPU_SPEC: Final[re.Pattern[str]] = re.compile(r"^cpu([1-9])$")


@dataclass(frozen=True)
class EvalOpponent:
    """A parsed ``EvalConfig.opponents`` entry: ``cpu<level>`` / ``best`` / ``other:<path>[@d<n>]`` /
    ``mimic`` / ``slippi:<release>``; ``delay_frames`` is an ``other`` file's ``@d<n>`` (``None`` = its own
    trained delay), ``release`` a ``slippi`` opponent's registry name (also its ``name``)."""

    name: str
    kind: str
    cpu_level: int | None = None
    path: str | None = None
    delay_frames: int | None = None
    release: str | None = None


def parse_opponent(spec: str) -> EvalOpponent:
    """``"cpu9"`` -> the level-9 CPU, ``"best"`` -> the run's best checkpoint, ``"other:<path>[@d<n>]"`` -> a
    file (at its own trained delay, or ``n``; 11 Sep 2026), ``"mimic"`` -> the released MIMIC policy
    (``[mimic]``)."""

    usage = (
        "eval opponent spec must be 'cpu<1-9>', 'best', 'other:<path>[@d<n>]', 'mimic' or "
        f"'slippi:<release>', got {spec!r}"
    )
    if not isinstance(spec, str) or not spec:
        raise ValueError(usage)
    match = _CPU_SPEC.match(spec)
    if match is not None:
        return EvalOpponent(name=spec, kind="cpu", cpu_level=int(match.group(1)))
    if spec == "best":
        return EvalOpponent(name="best", kind="best")
    if spec == "mimic":
        return EvalOpponent(name="mimic", kind="mimic")
    if spec.startswith("other:"):
        path, delay = split_rival_delay(spec[len("other:") :])
        if not path:
            raise ValueError(usage)
        return EvalOpponent(name="other", kind="other", path=path, delay_frames=delay)
    if spec.startswith("slippi:"):
        release = spec[len("slippi:") :]
        if release not in RELEASED_MODELS:
            raise ValueError(f"eval opponent {spec!r}: {release!r} is not a known slippi-ai release")
        if len(RELEASED_MODELS[release].character_ids) != 1:
            raise ValueError(
                f"eval opponent {spec!r}: {release} plays {len(RELEASED_MODELS[release].characters)} "
                "characters; an evaluation plays single-character releases (a mix arm's table picks "
                "characters)"
            )
        return EvalOpponent(name=release, kind="slippi", release=release)
    raise ValueError(usage)


@dataclass(frozen=True)
class EvalConfig:
    """``[eval]``: the in-loop evaluator (off by default).

    ``interval_seconds`` / ``interval_steps`` are checked after every learner step (either may be omitted);
    ``at_start`` evaluates step 0 of a fresh run (the baseline), ``at_end`` the final weights; ``frames`` is
    per opponent (at least ``num_envs * rollout_length``); ``opponents`` are ``cpu<level>`` / ``best`` /
    ``other:<path>`` / ``mimic`` (distinct names, one ``other`` per config; relative ``other`` paths resolve
    against the run file; ``mimic`` needs a Dolphin environment and the ``[mimic]`` table); ``metric`` of
    ``metric_opponent`` selects ``best.pt`` (``best`` itself cannot be the selector); ``temperature`` /
    ``seed`` drive the frozen student's sampling (``mimic`` samples with ``[mimic] seed + seed``);
    ``worker_processes`` gives the evaluation Dolphins their own process layout (``None`` inherits the
    training one, which then must divide ``num_envs``; ``mimic`` always runs in-process).
    """

    enabled: bool = False
    interval_seconds: float | None = 3600.0
    interval_steps: int | None = None
    at_start: bool = True
    at_end: bool = True
    frames: int = 57_600
    num_envs: int = 8
    rollout_length: int = 128
    opponents: tuple[str, ...] = ("cpu9", "best")
    temperature: float = 1.0
    seed: int = 0
    metric: str = "damage_dealt_per_minute"
    metric_opponent: str = "cpu9"
    worker_processes: int | None = None

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("eval.frames must be >= 1")
        if self.worker_processes is not None and self.worker_processes < 0:
            raise ValueError("eval.worker_processes must be >= 0 (0 = in-process) or omitted (inherit)")
        if self.worker_processes and self.num_envs % self.worker_processes:
            raise ValueError(
                f"eval.worker_processes {self.worker_processes} must divide eval.num_envs {self.num_envs}"
            )
        if self.num_envs < 1:
            raise ValueError("eval.num_envs must be >= 1")
        if self.rollout_length < 1:
            raise ValueError("eval.rollout_length must be >= 1")
        minimum = self.num_envs * self.rollout_length
        if self.frames < minimum:
            raise ValueError(
                f"eval.frames {self.frames} must cover at least one rollout: num_envs {self.num_envs} x "
                f"rollout_length {self.rollout_length} = {minimum} frames"
            )
        if not self.opponents:
            raise ValueError("eval.opponents must name at least one opponent")
        names = [item.name for item in self.parsed_opponents]
        if len(set(names)) != len(names):
            raise ValueError(f"eval.opponents must have distinct names, got {self.opponents!r}")
        if self.metric not in SELECTABLE_STATS:
            raise ValueError(f"eval.metric must be one of {SELECTABLE_STATS}, got {self.metric!r}")
        if self.metric_opponent not in names:
            raise ValueError(
                f"eval.metric_opponent {self.metric_opponent!r} must be one of the configured opponents "
                f"{names}"
            )
        if self.metric_opponent == "best":
            raise ValueError(
                "eval.metric_opponent cannot be 'best' (the best checkpoint would select itself)"
            )
        if self.interval_seconds is not None and not self.interval_seconds > 0.0:
            raise ValueError("eval.interval_seconds must be positive or omitted")
        if self.interval_steps is not None and self.interval_steps < 1:
            raise ValueError("eval.interval_steps must be >= 1 or omitted")
        no_trigger = self.interval_seconds is None and self.interval_steps is None
        if self.enabled and no_trigger and not (self.at_start or self.at_end):
            raise ValueError(
                "eval.enabled needs an interval (interval_seconds / interval_steps) or at_start / at_end"
            )
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("eval.temperature must be finite and >= 0")

    @property
    def parsed_opponents(self) -> tuple[EvalOpponent, ...]:
        return tuple(parse_opponent(spec) for spec in self.opponents)


@dataclass
class EvalResult:
    """One evaluation: per-opponent statistics (``EVAL_STATS``), the opponents skipped, timing."""

    step: int
    stats: dict[str, dict[str, float]]
    skipped: tuple[str, ...]
    seconds: float
    metric: str
    metric_opponent: str

    @property
    def selector(self) -> float | None:
        """The value that selects ``best.pt`` (``None`` when the selector opponent was skipped)."""

        values = self.stats.get(self.metric_opponent)
        return None if values is None else float(values[self.metric])

    def metrics(self) -> dict[str, float]:
        """The flat ``eval/*`` row logged to W&B: ``eval/step``, ``eval/seconds``, ``eval/skipped`` and
        ``eval/<opponent>/<stat>``."""

        row = {
            f"{EVAL_PREFIX}step": float(self.step),
            f"{EVAL_PREFIX}seconds": float(self.seconds),
            f"{EVAL_PREFIX}skipped": float(len(self.skipped)),
        }
        for name, values in self.stats.items():
            for stat, value in values.items():
                row[f"{EVAL_PREFIX}{name}/{stat}"] = float(value)
        return row

    def to_record(self) -> dict[str, Any]:
        """A JSON-able record (``eval_history.json``, checkpoints)."""

        return {
            "step": int(self.step),
            "seconds": float(self.seconds),
            "metric": self.metric,
            "metric_opponent": self.metric_opponent,
            "selector": self.selector,
            "stats": {
                name: {stat: float(value) for stat, value in values.items()}
                for name, values in self.stats.items()
            },
            "skipped": list(self.skipped),
        }


def format_eval_summary(result: EvalResult) -> str:
    """One console line per evaluation."""

    parts = [f"eval step {result.step}"]
    for name, values in result.stats.items():
        parts.append(
            f"{name}: dealt {values['damage_dealt_per_minute']:.1f}/min "
            f"taken {values['damage_taken_per_minute']:.1f}/min "
            f"ko_diff {values['ko_diff_per_minute']:+.2f}/min deaths {values['deaths_per_minute']:.2f}/min "
            f"({values['frames']:.0f} frames, {values['fps']:.0f} fps)"
        )
    for name in result.skipped:
        parts.append(f"{name}: skipped")
    parts.append(f"{result.seconds:.1f} s")
    return " | ".join(parts)


def trajectory_metrics(
    trajectories: Sequence[Trajectory],
    *,
    ports: int = 1,
    stalling_threshold: float = reward_lib.DEFAULT_STALLING_THRESHOLD,
    rows: Sequence[int] | None = None,
) -> dict[str, float]:
    """``reward/*`` and ``env/resets`` over the rollout rows of ``trajectories`` (slippi-ai ``player_stats``).

    ``ports`` is the number of trained perspectives per environment (two-port self-play stacks both
    perspectives of a game, so a game start appears in ``ports`` rows: ``env/resets`` counts it once);
    ``stalling_threshold`` is the reward's (``reward/stalling_fraction``).  With two trained ports the
    ``p0`` / ``p1`` statistics average over both perspectives (each is the student), so "dealt" and
    "taken" coincide and the zero-sum ``reward/per_frame`` is 0 by construction.  Shared by the training
    loop (``melee_rl.train``) and the evaluator.  ``rows`` (17 Sep 2026) keeps only those rows of every
    trajectory (the per-character statistics of a league arm).
    """

    if not trajectories:
        raise ValueError("trajectory_metrics needs at least one trajectory")
    if ports < 1:
        raise ValueError("ports must be >= 1")
    if rows is not None:
        index = torch.tensor(list(rows), dtype=torch.long)

        def pick(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[index.to(tensor.device)]

        frames = tree_cat(
            [
                tree_index(trajectory.rollout_frames(), index.to(trajectory.device))
                for trajectory in trajectories
            ]
        )
        rewards = torch.cat([pick(trajectory.rewards) for trajectory in trajectories])
        env_rewards = torch.cat([pick(trajectory.env_rewards) for trajectory in trajectories])
        resetting = [pick(trajectory.is_resetting) for trajectory in trajectories]
    else:
        frames = tree_cat([trajectory.rollout_frames() for trajectory in trajectories])
        rewards = torch.cat([trajectory.rewards for trajectory in trajectories])
        env_rewards = torch.cat([trajectory.env_rewards for trajectory in trajectories])
        resetting = [trajectory.is_resetting for trajectory in trajectories]
    stage = frames["stage"]
    p0 = reward_lib.player_stats(frames["p0"], frames["p1"], stage, stalling_threshold=stalling_threshold)
    p1 = reward_lib.player_stats(frames["p1"])
    ko_diff = float(reward_lib.ko_diff(frames).to(torch.float32).mean()) * reward_lib.FRAMES_PER_MINUTE
    resets = sum(
        int(flags[:, trajectory.context_frames :].sum())
        for trajectory, flags in zip(trajectories, resetting, strict=True)
    )
    return {
        "reward/per_frame": float(rewards.mean()),
        "reward/env_per_frame": float(env_rewards.mean()),
        "reward/ko_diff_per_minute": ko_diff,
        "reward/kos_per_minute": float(p1["deaths"]),
        "reward/damage_dealt_per_minute": float(p1["damages"]),
        "reward/damage_taken_per_minute": float(p0["damages"]),
        "reward/deaths_per_minute": float(p0["deaths"]),
        "reward/ledge_grabs_per_minute": float(p0["ledge_grabs"]),
        "reward/stalling_fraction": float(p0["stalling"]),
        "reward/approaching_factor": float(p0["approaching_factor"]),
        "env/resets": float(resets / ports),
    }


def stats_from_metrics(metrics: Mapping[str, float], *, frames: int, seconds: float) -> dict[str, float]:
    """:data:`EVAL_STATS` from one :func:`trajectory_metrics` row plus the frames and wall time.

    Shared by the in-loop evaluator (one opponent) and the video recorder (one clip, P9)."""

    return {
        "reward_per_frame": metrics["reward/per_frame"],
        "ko_diff_per_minute": metrics["reward/ko_diff_per_minute"],
        "kos_per_minute": metrics["reward/kos_per_minute"],
        "damage_dealt_per_minute": metrics["reward/damage_dealt_per_minute"],
        "damage_taken_per_minute": metrics["reward/damage_taken_per_minute"],
        "deaths_per_minute": metrics["reward/deaths_per_minute"],
        "ledge_grabs_per_minute": metrics["reward/ledge_grabs_per_minute"],
        "stalling_fraction": metrics["reward/stalling_fraction"],
        "approaching_factor": metrics["reward/approaching_factor"],
        "game_starts": metrics["env/resets"],
        "frames": float(frames),
        "seconds": float(seconds),
        "fps": frames / seconds if seconds > 0.0 else 0.0,
    }


def build_eval_env(
    config: RLConfig,
    *,
    num_envs: int,
    players: tuple[str, str],
    cpu_level: int,
    port_offset: int,
    save_replays: bool | None = None,
    replay_dir: str | Path | None = None,
    in_process: bool = False,
    characters: tuple[int, int] | None = None,
) -> EnvProtocol:
    """A fresh evaluation environment of the run's ``[env] type`` with ``num_envs`` envs and ``players``.

    The dummy toy gets a different seed than the training env; Dolphins get Slippi ports after the training
    range (``slippi_port + port_offset``; busy ports are re-picked anyway).  ``save_replays`` /
    ``replay_dir`` override the run file for this environment only (P9: the video recorder keeps the
    ``.slp`` of every clip; the training Dolphins are unaffected); ``[eval] worker_processes`` replaces
    the training worker layout when set; ``in_process`` builds the Dolphins in this process
    (``worker_processes = 0``) for an opponent that reads the raw gamestates (``mimic``) and wins over
    both; ``characters`` boots every Dolphin with that pair instead of the run's (a ``slippi:<release>``
    opponent's own character, 17 Sep 2026).  All of these are ignored by the dummy toy.
    """

    env = config.env
    if env.type == "dummy":
        dummy = replace(env.dummy, num_envs=num_envs, players=players, seed=env.dummy.seed + 10_000)
        return DummyMeleeEnv(dummy)
    if env.type == "dolphin":
        from melee_rl.env.dolphin_mp import build_dolphin_env

        overrides: dict[str, Any] = {
            "num_envs": num_envs,
            "players": players,
            "slippi_port": env.dolphin.slippi_port + port_offset,
        }
        if save_replays is not None:
            overrides["save_replays"] = save_replays
        if replay_dir is not None:
            overrides["replay_dir"] = str(replay_dir)
        if config.eval.worker_processes is not None:
            overrides["worker_processes"] = config.eval.worker_processes
        if in_process:
            overrides["worker_processes"] = 0
        if characters is not None:
            overrides["characters"] = tuple(characters)
            overrides["opponent_characters"] = ()
        if overrides.get("worker_processes", env.dolphin.worker_processes) == 0:
            overrides["shared_memory"] = False  # the exchange is the worker mode's; none in-process
        return build_dolphin_env(replace(env.dolphin, **overrides), cpu_level=cpu_level)
    raise ValueError(f"env.type {env.type!r} is not supported")


class Evaluator:
    """Plays a frozen copy of the student against the configured opponents; see the module docstring."""

    def __init__(
        self,
        config: EvalConfig,
        rl_config: RLConfig,
        model_config: ModelConfig,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self.rl_config = rl_config
        self.model_config = model_config
        self.device = torch.device(device)
        actor = rl_config.actor
        self._actor = ActorConfig(
            rollout_length=config.rollout_length,
            context_frames=0,
            delay_frames=actor.delay_frames,
            context_mode="ring",
            temperature=config.temperature,
            seed=config.seed,
            reward=actor.reward,
            check_reward_bounds=actor.check_reward_bounds,
        )
        self._actor.resolve_context_frames(model_config.context_length)  # T must fit the model's context
        self._port_offset = rl_config.env.num_envs + EVAL_PORT_GAP
        self._mimic_runtime: MimicRuntime | None = None
        self._slippi_networks: dict[str, Any] = {}

    def _mimic(self) -> MimicRuntime:
        """The released MIMIC model, loaded once per evaluator (``[mimic]``); players are per evaluation."""

        if self._mimic_runtime is None:
            self._mimic_runtime = load_runtime(self.rl_config.mimic)
        return self._mimic_runtime

    def _slippi(self, release: str, num_envs: int) -> Opponent:
        """A fresh port agent of ``release`` for one evaluation (the network is loaded once per evaluator)."""

        from melee_rl.slippi_ai_torch import SlippiAITorchOpponent, load_network

        settings = self.rl_config.slippi_ai
        network = self._slippi_networks.get(release)
        if network is None:
            released = RELEASED_MODELS[release]
            network = load_network(
                Path(settings.model_dir) / release,
                name=release,
                sha256=released.sha256 if settings.verify_sha256 else None,
            )
            self._slippi_networks[release] = network
        default = RELEASED_MODELS[release].default_name
        (character,) = RELEASED_MODELS[release].character_ids
        return SlippiAITorchOpponent(
            network,
            num_envs,
            names=(default,) * num_envs,
            device=self.device,
            seed=self.config.seed,
            temperature=settings.sample_temperature,
            graph=settings.torch_graph,
            characters=(character,) * num_envs,
            release=release,
        )

    @property
    def rollouts_per_opponent(self) -> int:
        return math.ceil(self.config.frames / (self.config.num_envs * self.config.rollout_length))

    def evaluate(
        self, policy: PolicyProtocol, *, step: int, best_path: str | Path | None = None
    ) -> EvalResult:
        """Evaluate ``policy`` (a frozen copy is played; the student is untouched) at learner ``step``."""

        started = time.perf_counter()
        frozen = policy.clone_frozen()
        stats: dict[str, dict[str, float]] = {}
        skipped: list[str] = []
        for opponent in self.config.parsed_opponents:
            if opponent.kind == "best" and (best_path is None or not Path(best_path).is_file()):
                skipped.append(opponent.name)
                continue
            stats[opponent.name] = self._play(frozen, opponent, best_path)
        return EvalResult(
            step=step,
            stats=stats,
            skipped=tuple(skipped),
            seconds=time.perf_counter() - started,
            metric=self.config.metric,
            metric_opponent=self.config.metric_opponent,
        )

    def _play(
        self, frozen: PolicyProtocol, opponent: EvalOpponent, best_path: str | Path | None
    ) -> dict[str, float]:
        """``frames`` environment frames of ``frozen`` against one opponent in a fresh environment."""

        config = self.config
        rival: Opponent | None = None
        players: tuple[str, str] = ("policy", "policy")
        cpu_level = 9
        extra: dict[str, Any] = {}
        if opponent.kind == "cpu":
            assert opponent.cpu_level is not None
            rival = CpuOpponent(cpu_level=opponent.cpu_level)
            players = ("policy", "cpu")
            cpu_level = opponent.cpu_level
        elif opponent.kind == "mimic":
            extra["in_process"] = True  # MIMIC reads the raw gamestates, which live in this process only
        elif opponent.kind == "slippi":
            assert opponent.release is not None
            (character,) = RELEASED_MODELS[opponent.release].character_ids
            extra["characters"] = (self.rl_config.env.dolphin.characters[0], character)
            rival = self._slippi(opponent.release, config.num_envs)
        else:
            path = best_path if opponent.kind == "best" else opponent.path
            assert path is not None
            # A file plays at the delay it was trained at unless the spec says otherwise (11 Sep 2026); until
            # then the rival inherited the run's delay, and a delay-0 anchor in a delayed run played late.
            delay = rival_delay(path, opponent.delay_frames)
            rival = OtherOpponent(
                path, config.num_envs, delay_frames=delay, seed=config.seed, device=self.device
            )
        env = build_eval_env(
            self.rl_config,
            num_envs=config.num_envs,
            players=players,
            cpu_level=cpu_level,
            port_offset=self._port_offset,
            **extra,
        )
        if opponent.kind == "mimic":
            gamestates = getattr(env, "gamestates", None)
            if gamestates is None:
                env.close()
                raise ValueError(
                    "the mimic eval opponent needs the raw gamestates of an in-process Dolphin environment "
                    '([env] type = "dolphin")'
                )
            mimic = self.rl_config.mimic
            rival = MimicOpponent(
                MimicPlayer(
                    self._mimic(),
                    config.num_envs,
                    player=1,
                    temperature=mimic.temperature,
                    top_k=mimic.top_k,
                    top_p=mimic.top_p,
                    seed=mimic.seed + config.seed,
                ),
                gamestates,
                port=1,
            )
        assert rival is not None
        try:
            worker = RolloutWorker(frozen, env, rival, self._actor)
            started = time.perf_counter()
            trajectories = [worker.rollout() for _ in range(self.rollouts_per_opponent)]
            seconds = time.perf_counter() - started
        finally:
            env.close()
        metrics = trajectory_metrics(
            trajectories, ports=1, stalling_threshold=self.rl_config.actor.reward.stalling_threshold
        )
        return stats_from_metrics(metrics, frames=worker.frames_seen, seconds=seconds)


__all__ = [
    "BEST_NAME",
    "EVAL_PORT_GAP",
    "EVAL_PREFIX",
    "EVAL_STATS",
    "HISTORY_NAME",
    "OPPONENT_KINDS",
    "SELECTABLE_STATS",
    "EvalConfig",
    "EvalOpponent",
    "EvalResult",
    "Evaluator",
    "build_eval_env",
    "format_eval_summary",
    "parse_opponent",
    "stats_from_metrics",
    "trajectory_metrics",
]
