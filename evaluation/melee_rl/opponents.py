"""Opponents for the rollout worker (PLAN.md §3.8, R10; P6a): ``cpu``, ``self`` and ``other``.

* ``cpu`` -- the environment drives the other port itself (the dummy env's script; Dolphin's
  in-game CPU at ``cpu_level``, slippi-ai default 9, ``dolphin.py:47-48``).  No actions.
* ``self`` -- a frozen copy of the student (``PolicyProtocol.clone_frozen``) plays port 1 on the
  swapped perspective the environment emits for that port, with its own KV cache (the model's
  native rolling cache, never re-primed: the opponent is not trained, exactness does not
  matter), its own delay queue, its own generator and temperature 1.0.  The training loop
  refreshes it every ``update_interval`` learner steps with :meth:`SelfOpponent.refresh`
  (``set_state(student.get_state())``, slippi-ai ``run_lib.py:117-137, 225-231``).  With
  ``train = true`` (slippi-ai ``OpponentConfig.train``, ``run_lib.py:112-137, 186-219``) there is
  no opponent object at all: the student plays both ports and *both* perspectives are trained
  (:func:`make_opponent` returns ``None``, ``RolloutWorker(ports=(0, 1))``).
* ``other`` -- a frozen policy loaded from a checkpoint (ours or Ali's format,
  ``melee_rl.checkpoint.load_policy``) plays port 1 the same way; ``refresh`` is a no-op and the
  file's path and sha256 are recorded (:attr:`OtherOpponent.source` / :attr:`OtherOpponent.sha256`).
* ``mix`` (P10) -- a curriculum over per-group environments: ``[opponent] mix`` lists specs like
  ``["self*32", "cpu9*64"]`` (:func:`parse_mix_spec` -> :class:`MixGroup`), the trainer builds one
  environment and one rollout worker per group (``self`` = two-port training, ``cpu<1-9>`` = the
  in-game CPU at that level, ``other:<path>`` = a frozen checkpoint sized to the group,
  :func:`make_group_opponent`) and every group's trajectories feed the same learner step.
  ``mix_workers`` splits ``env.dolphin.worker_processes`` across the groups.
* ``pool`` (11 Sep 2026, STRENGTH-PLAN.md §6) -- a mix arm ``pool*<n>`` whose port-1 opponent is one of the
  run's own ``step_<n>.pt`` snapshots, drawn by prioritised fictitious self-play every ``[opponent.pool]
  interval`` learner steps and loaded in place (:class:`melee_rl.pool.PoolOpponent`, :class:`PoolConfig`); the
  current self until a snapshot is ``min_age`` steps old.
* ``slippi`` (17 Sep 2026, LEAGUE-PLAN.md §2) -- a mix arm ``slippi:<release>*<n>`` whose port-1 opponent is
  one of slippi-ai's released agents run as a PyTorch port on the GPU (:mod:`melee_rl.slippi_ai_torch`); the
  arm's ``[opponent.arms.<alias>]`` table (:class:`ArmConfig`) gives each of its Dolphins a character and a
  name tag.
* ``mimic`` (P12) -- the released MIMIC policy (:mod:`melee_rl.mimic`) on port 1, driven from the raw
  libmelee gamestates and returning a ``ControllerState`` rather than labels
  (:class:`MimicOpponent`).  Evaluation only: it is never a training opponent.

:class:`PolicyOpponent` is the shared body (cache, delay queue, generator, ``act`` /
``reset_state``); :class:`SelfOpponent` and :class:`OtherOpponent` only differ in where the policy
comes from and what ``refresh`` does.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import torch

from controller_codec import ControllerLabels, ControllerState
from melee_rl import checkpoint as checkpoint_lib
from melee_rl.env.dolphin import ControllerRow, neutral_row
from melee_rl.frames import labels_map, neutral_labels
from melee_rl.mimic import MimicPlayer, rows_to_state
from melee_rl.phillip import PhillipPlayer
from melee_rl.protocol import PolicyProtocol
from melee_rl.slippi_ai_agent import RELEASED_MODELS, SlippiAIConfig, SlippiAIPlayer
from melee_rl.smashbot import SmashBotPlayer

OPPONENT_TYPES: Final[tuple[str, ...]] = ("cpu", "self", "other", "mix")
CHARACTER_TABLE_KINDS: Final[frozenset[str]] = frozenset({"cpu", "other", "slippi"})
"""Mix arms whose port-1 character an ``[opponent.arms.<alias>]`` table may set (not ``self`` / ``pool``:
their port 1 is the student itself, and our bot plays Fox only)."""
_MIX_CPU: Final[re.Pattern[str]] = re.compile(r"^cpu([1-9])$")
_RIVAL_DELAY: Final[re.Pattern[str]] = re.compile(r"^(?P<path>.*)@d(?P<delay>\d+)$")


def split_rival_delay(text: str) -> tuple[str, int | None]:
    """``"<path>@d<n>"`` -> ``(path, n)``; anything else -> ``(text, None)``.

    The suffix names the delay a frozen checkpoint plays at (``other:<path>@d<n>`` in eval and mix specs,
    11 Sep 2026); without it the file plays at the delay it was trained at (:func:`rival_delay`).
    """

    match = _RIVAL_DELAY.match(text)
    if match is None:
        return text, None
    return match.group("path"), int(match.group("delay"))


def rival_delay(path: str | Path, override: int | None = None) -> int:
    """The delay a frozen checkpoint plays at: ``override`` when given (``@d<n>`` in a spec,
    ``opponent.delay_frames``), else the delay the file was trained at (``checkpoint.trained_delay``).

    11 Sep 2026: until then every rival inherited the run's delay, so a delay-0 anchor in a D = 18 run would
    have played 18 frames late.
    """

    return checkpoint_lib.trained_delay(path) if override is None else int(override)


@dataclass(frozen=True)
class MixGroup:
    """One arm of a mixed run: ``self*<n>`` (two-port training), ``cpu<1-9>*<n>``,
    ``other:<path>[@d<n>]*<n>``, ``pool*<n>`` (the run's own past selves, :mod:`melee_rl.pool`, 11 Sep 2026)
    or ``slippi:<release>*<n>`` (a released slippi-ai agent, 17 Sep 2026); ``delay_frames`` is an ``other``
    arm's ``@d<n>`` (``None`` = the file's own trained delay), ``release`` a ``slippi`` arm's
    :data:`~melee_rl.slippi_ai_agent.RELEASED_MODELS` name."""

    spec: str
    kind: str
    num_envs: int
    cpu_level: int = 9
    checkpoint: str | None = None
    alias: str = ""
    delay_frames: int | None = None
    release: str | None = None

    @property
    def name(self) -> str:
        """The arm's metric name (``arm/<name>/*``): ``self``, ``cpu<level>`` or ``other``, with ``_<alias>``
        when the spec carries one (``self*24@a``: several arms of one kind, lever 1's smaller lockstep
        groups, 5 Sep 2026)."""

        base = f"cpu{self.cpu_level}" if self.kind == "cpu" else self.kind
        return f"{base}_{self.alias}" if self.alias else base

    @property
    def players(self) -> tuple[str, str]:
        """The group's environment players: the CPU arm boots Dolphin's CPU on port 1."""

        return ("policy", "cpu") if self.kind == "cpu" else ("policy", "policy")

    @property
    def trained_ports(self) -> tuple[int, ...]:
        """``(0, 1)`` for the two-port ``self`` arm, ``(0,)`` otherwise."""

        return (0, 1) if self.kind == "self" else (0,)

    @property
    def rows(self) -> int:
        """Training rows this arm contributes per rollout (its gradient share of the mix)."""

        return self.num_envs * len(self.trained_ports)


def parse_mix_spec(spec: str) -> MixGroup:
    """``self*32`` | ``cpu9*64`` | ``other:/path*4`` | ``pool*24`` | ``slippi:gm*24`` -> :class:`MixGroup`
    (the P10 mixed curriculum; ``pool`` arms since 11 Sep 2026, ``slippi`` arms since 17 Sep 2026)."""

    usage = (
        f'a mix spec is "self*<count>", "cpu<1-9>*<count>", "other:<path>[@d<delay>]*<count>", '
        f'"pool*<count>" or "slippi:<release>*<count>", optionally with a "@<alias>" suffix, got {spec!r}'
    )
    head, star, count_text = spec.rpartition("*")
    if not star or not head:
        raise ValueError(usage)
    count_text, at, alias = count_text.partition("@")
    if at and not alias.isidentifier():
        raise ValueError(usage)
    try:
        count = int(count_text)
    except ValueError as error:
        raise ValueError(usage) from error
    if count < 1:
        raise ValueError(f"mix group {spec!r} needs a positive environment count, got {count}")
    if head == "self":
        return MixGroup(spec=spec, kind="self", num_envs=count, alias=alias)
    if head == "pool":
        return MixGroup(spec=spec, kind="pool", num_envs=count, alias=alias)
    match = _MIX_CPU.match(head)
    if match:
        return MixGroup(spec=spec, kind="cpu", num_envs=count, cpu_level=int(match.group(1)), alias=alias)
    prefix = "other:"
    if head.startswith(prefix):
        path, delay = split_rival_delay(head[len(prefix) :])
        if not path:
            raise ValueError(usage)
        return MixGroup(
            spec=spec, kind="other", num_envs=count, checkpoint=path, alias=alias, delay_frames=delay
        )
    if head.startswith("slippi:"):
        release = head[len("slippi:") :]
        if not release:
            raise ValueError(usage)
        if release not in RELEASED_MODELS:
            raise ValueError(
                f"mix group {spec!r}: {release!r} is not a known slippi-ai release; "
                f"have {sorted(RELEASED_MODELS)}"
            )
        return MixGroup(spec=spec, kind="slippi", num_envs=count, alias=alias, release=release)
    raise ValueError(usage)


@dataclass(frozen=True)
class ArmConfig:
    """``[opponent.arms.<alias>]``: what each Dolphin of one aliased mix arm plays (17 Sep 2026, the league
    run).

    ``characters`` (libmelee ids) is the opponent's character on each of the arm's Dolphins, cycled over the
    arm's own Dolphin index in blocks of ``character_repeat`` (24 Dolphins, twelve characters, repeat 2:
    Dolphins ``2c`` and ``2c + 1`` play character ``c``, which puts each character on two different stages of
    a six-stage cycle). ``names`` are a ``slippi`` arm's player tags (a one-hot input of the release), cycled
    the same way in blocks of ``name_repeat``. Empty means the default: the release's only character (or the
    run's ``characters[1]``) and the release's first tag."""

    characters: tuple[int, ...] = ()
    character_repeat: int = 1
    names: tuple[str, ...] = ()
    name_repeat: int = 1

    def __post_init__(self) -> None:
        if self.character_repeat < 1:
            raise ValueError("opponent.arms character_repeat must be >= 1")
        if self.name_repeat < 1:
            raise ValueError("opponent.arms name_repeat must be >= 1")
        if any(not 0 <= character < 33 for character in self.characters):
            raise ValueError(
                f"opponent.arms characters must be libmelee ids in [0, 33), got {self.characters!r}"
            )

    def character_for(self, index: int) -> int | None:
        """The character of the arm's Dolphin ``index`` (``None`` without a table)."""

        if not self.characters:
            return None
        return self.characters[(index // self.character_repeat) % len(self.characters)]

    def name_for(self, index: int) -> str | None:
        """The name tag of the arm's Dolphin ``index`` (``None`` without a table)."""

        if not self.names:
            return None
        return self.names[(index // self.name_repeat) % len(self.names)]


def arm_characters(group: MixGroup, arm: ArmConfig) -> tuple[int, ...] | None:
    """Player 1's character on each of ``group``'s Dolphins, or ``None`` when the run's ``characters`` stand.

    A ``slippi`` arm without a character table plays its release's only character (``OpponentConfig`` refuses
    a multi-character release without one)."""

    if arm.characters:
        return tuple(int(arm.character_for(index) or 0) for index in range(group.num_envs))
    if group.kind == "slippi" and group.release is not None:
        (only,) = RELEASED_MODELS[group.release].character_ids
        return (only,) * group.num_envs
    return None


def arm_names(group: MixGroup, arm: ArmConfig) -> tuple[str, ...]:
    """A ``slippi`` arm's name tag on each of its Dolphins (the release's first tag without a table)."""

    if group.kind != "slippi" or group.release is None:
        raise ValueError(f"only slippi arms carry name tags, got {group.spec!r}")
    default = RELEASED_MODELS[group.release].default_name
    return tuple(arm.name_for(index) or default for index in range(group.num_envs))


@dataclass(frozen=True)
class PoolConfig:
    """``[opponent.pool]``: the past-self arms (``pool*<n>`` mix groups, :mod:`melee_rl.pool`; 11 Sep 2026).

    Every ``interval`` learner steps a pool arm draws one of the run's ``step_<n>.pt`` snapshots at
    least ``min_age`` steps old (``max_candidates`` > 0 thins the ladder to that many evenly spaced
    ones, the oldest and the newest kept) by PFSP: weight ``max((1 - x)^2, floor)`` of the arm's KO
    share ``x`` against each, a snapshot never played joining at the highest weight present.  ``decay``
    multiplies a snapshot's counts each time it is drawn again, so the estimate follows the student.
    """

    interval: int = 8
    min_age: int = 32
    max_candidates: int = 0
    floor: float = 0.05
    decay: float = 0.5

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("opponent.pool.interval must be >= 1")
        if self.min_age < 0:
            raise ValueError("opponent.pool.min_age must be >= 0")
        if self.max_candidates < 0 or self.max_candidates == 1:
            raise ValueError("opponent.pool.max_candidates must be 0 (every snapshot) or >= 2")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("opponent.pool.floor must be in (0, 1]")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("opponent.pool.decay must be in [0, 1]")


@dataclass(frozen=True)
class OpponentConfig:
    """``[opponent]`` settings.

    ``type`` is ``cpu`` | ``self`` | ``other`` | ``mix``; ``cpu_level`` is Dolphin-only (P5); ``train``
    (self-play only) trains both ports of the self-play game -- the student plays port 1 itself and
    there is no opponent object; ``checkpoint`` is the ``other`` opponent's file (ours or Ali's format;
    relative paths resolve against the run file); ``update_interval`` / ``temperature`` / ``seed``
    apply to the ``self`` and ``other`` policies.  ``mix`` (P10) lists the curriculum's arms as
    ``self*<n>`` / ``cpu<1-9>*<n>`` / ``other:<path>*<n>`` specs (:func:`parse_mix_spec`); the arms'
    environment counts must sum to the env table's ``num_envs`` and ``mix_workers`` (Dolphin only)
    splits ``worker_processes`` across the arms, one entry per spec; ``pool`` (the ``[opponent.pool]``
    table, :class:`PoolConfig`) sets up the ``pool*<n>`` arms.  ``delay_frames`` (``other`` only, 11 Sep 2026)
    is the delay the frozen checkpoint plays at; unset, it plays at the delay its file was trained at.
    """

    type: str = "cpu"
    cpu_level: int = 9
    update_interval: int = 1
    temperature: float = 1.0
    seed: int = 1
    train: bool = False
    checkpoint: str | None = None
    delay_frames: int | None = None
    mix: tuple[str, ...] = ()
    mix_workers: tuple[int, ...] = ()
    pool: PoolConfig = field(default_factory=PoolConfig)
    arms: dict[str, ArmConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in OPPONENT_TYPES:
            raise ValueError(f"opponent type must be one of {OPPONENT_TYPES}, got {self.type!r}")
        if self.delay_frames is not None:
            if self.type != "other":
                raise ValueError(
                    'opponent.delay_frames is only used by type = "other" (a frozen checkpoint plays at its '
                    f"own delay), got type {self.type!r}"
                )
            if self.delay_frames < 0:
                raise ValueError("opponent.delay_frames must be >= 0 or omitted (the file's trained delay)")
        if not 1 <= self.cpu_level <= 9:
            raise ValueError("cpu_level must be in [1, 9]")
        if self.update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.train and self.type != "self":
            raise ValueError(
                f'opponent.train = true is a self-play switch (type = "self"), got type {self.type!r}'
            )
        if self.type == "other" and not self.checkpoint:
            raise ValueError('opponent.checkpoint is required when opponent.type = "other"')
        if self.type != "other" and self.checkpoint is not None:
            raise ValueError(f'opponent.checkpoint is only used by type = "other", got type {self.type!r}')
        if self.type == "mix":
            if not self.mix:
                raise ValueError('opponent.mix must list at least one group when opponent.type = "mix"')
            groups = tuple(parse_mix_spec(spec) for spec in self.mix)
            names = [group.name for group in groups]
            if len(set(names)) != len(names):
                raise ValueError(f"mix group names must be distinct, got {names}")
        elif self.mix:
            raise ValueError(f'opponent.mix is only used by type = "mix", got type {self.type!r}')
        if self.mix_workers:
            if self.type != "mix":
                raise ValueError(f'opponent.mix_workers is only used by type = "mix", got type {self.type!r}')
            if len(self.mix_workers) != len(self.mix):
                raise ValueError(
                    f"opponent.mix_workers needs one worker count per mix group ({len(self.mix)}), "
                    f"got {len(self.mix_workers)}"
                )
            for spec, workers in zip(self.mix, self.mix_workers, strict=True):
                if workers < 1:
                    raise ValueError(f"opponent.mix_workers entries must be >= 1, got {workers} for {spec!r}")
                group = parse_mix_spec(spec)
                if group.num_envs % workers:
                    raise ValueError(
                        f"mix group {spec!r} has {group.num_envs} environments, not divisible by its "
                        f"{workers} worker processes"
                    )
        self._check_arms()

    def _check_arms(self) -> None:
        """``[opponent.arms.<alias>]`` names an aliased arm of this mix that can take its settings."""

        if self.arms and self.type != "mix":
            raise ValueError(f'opponent.arms is only used by type = "mix", got type {self.type!r}')
        groups = {group.alias: group for group in self.groups if group.alias}
        for alias, arm in self.arms.items():
            group = groups.get(alias)
            if group is None:
                raise ValueError(
                    f"opponent.arms.{alias} does not name an aliased arm of opponent.mix "
                    f"(aliases {sorted(groups)})"
                )
            if arm.characters and group.kind not in CHARACTER_TABLE_KINDS:
                raise ValueError(
                    f"opponent.arms.{alias}: a {group.kind} arm's port 1 is the student itself, which plays "
                    f"Fox; characters are for {sorted(CHARACTER_TABLE_KINDS)} arms"
                )
            if arm.names and group.kind != "slippi":
                raise ValueError(
                    f"opponent.arms.{alias}: names are a slippi arm's input, not a {group.kind} arm's"
                )
        for group in self.groups:
            if group.kind != "slippi" or group.release is None:
                continue
            released = RELEASED_MODELS[group.release]
            arm = self.arm(group)
            if not arm.characters and len(released.character_ids) != 1:
                raise ValueError(
                    f"mix arm {group.spec!r}: {group.release} plays {len(released.characters)} characters; "
                    "list the ones its Dolphins play in the characters of "
                    f"[opponent.arms.{group.alias or '<alias>'}]"
                )
            wrong = sorted(set(arm.characters) - set(released.character_ids))
            if wrong:
                raise ValueError(
                    f"mix arm {group.spec!r}: {group.release} cannot play libmelee characters {wrong}; "
                    f"it plays {released.characters} ({released.character_ids})"
                )
            if released.names:
                untrained = sorted(set(arm.names) - set(released.names))
                if untrained:
                    raise ValueError(
                        f"mix arm {group.spec!r}: {group.release} was trained with the tags "
                        f"{released.names}, got {untrained}"
                    )

    def arm(self, group: MixGroup) -> ArmConfig:
        """``group``'s ``[opponent.arms.<alias>]`` table (the empty default when it has none)."""

        return self.arms.get(group.alias, ArmConfig()) if group.alias else ArmConfig()

    @property
    def trained_ports(self) -> tuple[int, ...]:
        """The environment ports the student is trained on: ``(0, 1)`` for ``self + train``, else ``(0,)``."""

        if self.type == "mix":
            raise ValueError('a "mix" opponent has per-group trained ports (OpponentConfig.groups)')
        return (0, 1) if self.type == "self" and self.train else (0,)

    @property
    def groups(self) -> tuple[MixGroup, ...]:
        """The parsed ``mix`` arms (empty for the plain single-opponent types)."""

        if self.type != "mix":
            return ()
        return tuple(parse_mix_spec(spec) for spec in self.mix)


@runtime_checkable
class Opponent(Protocol):
    """What the rollout worker needs from an opponent."""

    @property
    def port(self) -> int: ...

    @property
    def controls_port(self) -> bool:
        """True when :meth:`act` returns labels the worker must send to the environment."""
        ...

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels | None:
        """Labels to execute this step given the opponent's perspective ``[B]`` tree (``None`` for cpu)."""
        ...

    def refresh(self, student: PolicyProtocol) -> None:
        """Copy the student's parameters (self-play); no-op otherwise."""
        ...

    def reset_state(self) -> None:
        """Forget everything (cache, delay queue, previous action) after a hard environment reset."""
        ...


class CpuOpponent:
    """The environment's own opponent; emits no actions."""

    def __init__(self, port: int = 1, cpu_level: int = 9) -> None:
        self._port = port
        self.cpu_level = cpu_level

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return False

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels | None:
        return None

    def refresh(self, student: PolicyProtocol) -> None:
        return None

    def reset_state(self) -> None:
        return None


class PolicyOpponent:
    """A frozen policy on ``port`` with its own cache, delay queue and generator (``self`` / ``other`` body).

    ``policy`` must already be frozen (eval mode, no gradients); its parameters are never trained here.
    """

    def __init__(
        self,
        policy: PolicyProtocol,
        batch_size: int,
        *,
        port: int = 1,
        delay_frames: int = 0,
        temperature: float = 1.0,
        seed: int = 1,
    ) -> None:
        if delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._port = port
        self.delay_frames = delay_frames
        self.temperature = temperature
        self.policy = policy
        self._batch_size = batch_size
        self._cache = self.policy.init_cache(batch_size)
        self._device = self._cache.keys.device
        self._generator = torch.Generator(device=self._device).manual_seed(seed)
        self._neutral = neutral_labels((batch_size,), device=self._device)
        self._queue: deque[ControllerLabels] = deque()
        self._last = self._neutral
        self.reset_state()

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def generator(self) -> torch.Generator:
        """The opponent's sampling generator (its state goes into RL checkpoints, P3)."""

        return self._generator

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels:
        reset = needs_reset.to(device=self._device, dtype=torch.bool)
        prev = labels_map(lambda c: torch.where(reset, torch.zeros_like(c), c), self._last)
        output = self.policy.step_sample(
            frames,
            prev,
            reset,
            self._cache,
            temperature=self.temperature,
            generator=self._generator,
        )
        self._last = output.labels
        self._queue.append(output.labels)
        return self._queue.popleft()

    def refresh(self, student: PolicyProtocol) -> None:
        """No-op: a fixed opponent does not follow the student (``SelfOpponent`` overrides this)."""

        return None

    def reset_state(self) -> None:
        everything = torch.ones(self._batch_size, dtype=torch.bool, device=self._device)
        self.policy.reset_slots(self._cache, everything)
        self._queue = deque(self._neutral for _ in range(self.delay_frames))
        self._last = self._neutral


class SelfOpponent(PolicyOpponent):
    """A frozen copy of the student; ``refresh`` copies the student's current parameters."""

    def __init__(
        self,
        student: PolicyProtocol,
        batch_size: int,
        *,
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

    def refresh(self, student: PolicyProtocol) -> None:
        self.policy.set_state(student.get_state())


class OtherOpponent(PolicyOpponent):
    """A frozen policy from a checkpoint file (ours or Ali's format); never refreshed."""

    def __init__(
        self,
        path: str | Path,
        batch_size: int,
        *,
        port: int = 1,
        delay_frames: int = 0,
        temperature: float = 1.0,
        seed: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"opponent checkpoint {source} does not exist")
        policy = checkpoint_lib.load_policy(source, device=device)
        policy.eval()
        policy.requires_grad_(False)
        self.source = str(source)
        self.sha256 = checkpoint_lib.sha256_file(source)
        super().__init__(
            policy,
            batch_size,
            port=port,
            delay_frames=delay_frames,
            temperature=temperature,
            seed=seed,
        )


class MimicOpponent:
    """MIMIC on ``port``, driven from the raw libmelee gamestates (P12).

    Unlike every other opponent this one is not a :class:`PolicyProtocol`: MIMIC has its own
    observation (built from the gamestate, not from our frame tree) and its own action space, so it
    reads ``gamestates()`` -- the environment's raw per-instance state -- and returns a
    ``ControllerState`` through :meth:`act_state` instead of labels through :meth:`act`.  Sending it
    through Ali's codec would quantise MIMIC's 37 stick clusters onto an 85-value vocabulary, so the
    rollout worker prefers ``act_state`` whenever an opponent defines it.
    """

    def __init__(
        self,
        player: MimicPlayer,
        gamestates: Callable[[], Sequence[Any]],
        *,
        port: int = 1,
    ) -> None:
        self.player = player
        self._gamestates = gamestates
        self._port = port

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels:
        raise NotImplementedError("MIMIC emits a ControllerState; the worker must call act_state")

    def act_state(self, frames: Any, needs_reset: torch.Tensor) -> ControllerState:
        """One frame of MIMIC for every environment (``frames`` is unused: MIMIC builds its own)."""

        rows = self.player.rows(list(self._gamestates()), needs_reset)
        return rows_to_state(rows)

    def refresh(self, student: PolicyProtocol) -> None:
        return None

    def reset_state(self) -> None:
        self.player.reset_all()


class SmashBotOpponent:
    """SmashBot on ``port``, driven from the raw libmelee gamestates (P25).

    Like :class:`MimicOpponent` this is not a :class:`PolicyProtocol`: SmashBot is an expert system with
    its own observation (the gamestate itself) and its own action space (a controller), so it reads
    ``gamestates()`` and returns a ``ControllerState`` through :meth:`act_state`.  Routing it through
    Ali's codec would quantise its stick values onto an 85-value vocabulary and lose the difference
    between a digital ``L`` press and an analog shoulder, both of which SmashBot's chains rely on.

    ``delay_frames`` holds SmashBot's input back by ``k`` frames **on its own port only**, which is the
    instrument of the calibration sweep.  Our environment runs ``use_exi_inputs`` + ``blocking_input``
    + ``online_delay = 0``, so by default an opponent decides on frame N and its input lands on frame N
    -- tighter than the named-pipe path SmashBot was written against, where a chain that presses a
    frame early to absorb the latency would land on time.  With ``k > 0`` the queue is primed with
    ``k`` neutral frames, and an environment whose ``needs_reset`` is set has its own queued frames
    neutralised, so a new game never inherits the previous one's in-flight buttons.
    """

    def __init__(
        self,
        player: SmashBotPlayer,
        gamestates: Callable[[], Sequence[Any]],
        *,
        port: int = 1,
        delay_frames: int = 0,
    ) -> None:
        if delay_frames < 0:
            raise ValueError(f"delay_frames must be >= 0, got {delay_frames}")
        self.player = player
        self._gamestates = gamestates
        self._port = port
        self.delay_frames = int(delay_frames)
        self._queue: deque[list[ControllerRow]] = deque()
        self._prime()

    def _prime(self) -> None:
        self._queue.clear()
        for _ in range(self.delay_frames):
            self._queue.append([neutral_row()] * self.player.num_envs)

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels:
        raise NotImplementedError("SmashBot emits a ControllerState; the worker must call act_state")

    def act_state(self, frames: Any, needs_reset: torch.Tensor) -> ControllerState:
        """One frame of SmashBot for every environment (``frames`` is unused: it reads the gamestate)."""

        rows = self.player.rows(list(self._gamestates()), needs_reset)
        if self.delay_frames:
            for index, reset in enumerate(needs_reset.to(dtype=torch.bool, device="cpu").tolist()):
                if not reset:
                    continue
                for queued in self._queue:
                    queued[index] = neutral_row()
            self._queue.append(rows)
            rows = self._queue.popleft()
        return rows_to_state(rows)

    def refresh(self, student: PolicyProtocol) -> None:
        return None

    def reset_state(self) -> None:
        self.player.reset_all()
        self._prime()


class PhillipOpponent:
    """vladfi1/phillip on ``port``, driven from the raw libmelee gamestates (P32, 18 Sep 2026).

    Like :class:`MimicOpponent` and :class:`SmashBotOpponent` this is not a :class:`PolicyProtocol`:
    Phillip embeds its own observation (upstream's ``GameMemory`` fields, converted from the gamestate
    by :mod:`melee_rl.phillip`) and picks one of its own discrete controller actions, which come back
    as a controller through :meth:`act_state`.  Upstream's ``Agent`` runs in a sidecar interpreter
    (TensorFlow 2.13 needs Python 3.11); the player owns that process, so :meth:`close` ends it.

    There is no delay queue here: Phillip's delay, where an agent has one, is a queue of whole decisions
    *inside* the agent (``agent.py:43``), so every agent plays at its own trained delay by construction.
    """

    def __init__(
        self,
        player: PhillipPlayer,
        gamestates: Callable[[], Sequence[Any]],
        *,
        port: int = 1,
    ) -> None:
        self.player = player
        self._gamestates = gamestates
        self._port = port

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels:
        raise NotImplementedError("Phillip emits a ControllerState; the worker must call act_state")

    def act_state(self, frames: Any, needs_reset: torch.Tensor) -> ControllerState:
        """One frame of Phillip for every environment (``frames`` is unused: it reads the gamestate)."""

        return rows_to_state(self.player.rows(list(self._gamestates()), needs_reset))

    def refresh(self, student: PolicyProtocol) -> None:
        return None

    def reset_state(self) -> None:
        self.player.reset_all()

    def close(self) -> None:
        self.player.close()


class SlippiAIOpponent:
    """A released slippi-ai agent on ``port``, driven from the raw libmelee gamestates (P27).

    Like :class:`MimicOpponent` and :class:`SmashBotOpponent` this is not a :class:`PolicyProtocol`:
    the agent owns its observation (upstream's parser and observation filter), its recurrent state, its
    delay queue, its sampler and its own action space, so it reads ``gamestates()`` and returns a
    ``ControllerState`` through :meth:`act_state`.  Routing it through Ali's codec would quantise its
    161-point stick grid onto an 85-value vocabulary.

    There is deliberately no ``delay_frames`` here, unlike SmashBot's: every release carries 18-24
    frames of *policy* delay inside itself (``[slippi_ai] console_delay`` is the only knob, and it pays
    for a console lag our environment does not have).  Adding ours on top would be a second,
    undocumented handicap.
    """

    def __init__(
        self,
        player: SlippiAIPlayer,
        gamestates: Callable[[], Sequence[Any]],
        *,
        port: int = 1,
    ) -> None:
        self.player = player
        self._gamestates = gamestates
        self._port = port

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels:
        raise NotImplementedError("slippi-ai emits a ControllerState; the worker must call act_state")

    def act_state(self, frames: Any, needs_reset: torch.Tensor) -> ControllerState:
        """One frame of the agent for every environment (``frames`` is unused: it builds its own)."""

        return rows_to_state(self.player.rows(list(self._gamestates()), needs_reset))

    def refresh(self, student: PolicyProtocol) -> None:
        return None

    def reset_state(self) -> None:
        self.player.reset_all()


def make_opponent(
    config: OpponentConfig,
    student: PolicyProtocol,
    *,
    batch_size: int,
    delay_frames: int,
    port: int = 1,
    device: torch.device | str | None = None,
) -> Opponent | None:
    """Build the opponent described by ``config`` for ``batch_size`` environments.

    Returns ``None`` for ``self + train``: the student plays every port itself (``RolloutWorker(...,
    opponent=None, ports=(0, 1))``).  ``device`` is where an ``other`` policy is loaded (the student's).
    ``delay_frames`` is the run's delay -- the self-play clone's; an ``other`` checkpoint plays at its own
    trained delay unless ``config.delay_frames`` says otherwise (:func:`rival_delay`, 11 Sep 2026).
    """

    if config.type == "cpu":
        return CpuOpponent(port=port, cpu_level=config.cpu_level)
    if config.type == "other":
        assert config.checkpoint is not None
        opponent = OtherOpponent(
            config.checkpoint,
            batch_size,
            port=port,
            delay_frames=rival_delay(config.checkpoint, config.delay_frames),
            temperature=config.temperature,
            seed=config.seed,
            device=device,
        )
        # The loaded policy is a fresh adapter: inherit the student's fast-step setting (P7-PLAN.md 3.2).
        from melee_rl.adapter import MeleePolicyAdapter

        if isinstance(opponent.policy, MeleePolicyAdapter):
            opponent.policy.fast_step = bool(getattr(student, "fast_step", False))
            opponent.policy.fast_attention = str(getattr(student, "fast_attention", "sdpa"))
            opponent.policy.fast_encoder = bool(getattr(student, "fast_encoder", False))
            opponent.policy.graph_step = bool(getattr(student, "graph_step", False))
        return opponent
    if config.train:
        return None
    return SelfOpponent(
        student,
        batch_size,
        port=port,
        delay_frames=delay_frames,
        temperature=config.temperature,
        seed=config.seed,
    )


def slippi_torch_opponent(
    group: MixGroup,
    arm: ArmConfig,
    settings: SlippiAIConfig,
    *,
    seed: int,
    device: torch.device | str | None,
) -> Opponent:
    """A ``slippi:<release>*<n>`` arm's opponent: the release file from ``settings.model_dir`` (its registry
    sha256 checked unless ``verify_sha256 = false``) in the PyTorch port, one name tag and one character per
    Dolphin from the arm's table, the port's step captured as a CUDA graph when ``settings.torch_graph``."""

    from melee_rl.slippi_ai_torch import SlippiAITorchOpponent, load_network  # torch-only, no TensorFlow

    assert group.release is not None
    released = RELEASED_MODELS[group.release]
    network = load_network(
        Path(settings.model_dir) / group.release,
        name=group.release,
        sha256=released.sha256 if settings.verify_sha256 else None,
    )
    return SlippiAITorchOpponent(
        network,
        group.num_envs,
        names=arm_names(group, arm),
        device="cpu" if device is None else device,
        seed=seed,
        temperature=settings.sample_temperature,
        graph=settings.torch_graph,
        characters=arm_characters(group, arm) or (),
        release=group.release,
    )


def make_group_opponent(
    group: MixGroup,
    config: OpponentConfig,
    student: PolicyProtocol,
    *,
    delay_frames: int,
    seed: int,
    device: torch.device | str | None = None,
    snapshot_dir: str | Path | None = None,
    slippi: SlippiAIConfig | None = None,
) -> Opponent | None:
    """The opponent of one mix arm (P10).

    ``self`` trains both ports itself (``None``, exactly the P6a two-port path); ``cpu`` is the
    in-game CPU at the group's level; ``other`` a frozen checkpoint sized to the group's
    environments; ``pool`` the run's own past selves, drawn from the ``step_<n>.pt`` files in
    ``snapshot_dir`` (the run's checkpoint directory; :class:`melee_rl.pool.PoolOpponent`, 11 Sep 2026).
    ``seed`` should differ per group (the trainer passes ``opponent.seed + index``).  ``delay_frames`` is the
    run's delay -- the pool arms' -- while an ``other`` arm plays at its file's own trained delay unless its
    spec carries ``@d<n>`` (:func:`rival_delay`).  ``slippi`` arms (17 Sep 2026) load ``[slippi_ai]
    model_dir`` / the release into the PyTorch port with the arm's per-Dolphin name tags
    (:func:`slippi_torch_opponent`).
    """

    if group.kind == "self":
        return None
    if group.kind == "slippi":
        return slippi_torch_opponent(
            group, config.arm(group), SlippiAIConfig() if slippi is None else slippi, seed=seed, device=device
        )
    if group.kind == "cpu":
        return CpuOpponent(port=1, cpu_level=group.cpu_level)
    if group.kind == "pool":
        from melee_rl.pool import PoolOpponent  # melee_rl.pool builds on this module

        return PoolOpponent(
            student,
            group.num_envs,
            directory=snapshot_dir,
            config=config.pool,
            port=1,
            delay_frames=delay_frames,
            temperature=config.temperature,
            seed=seed,
        )
    assert group.checkpoint is not None
    opponent = OtherOpponent(
        group.checkpoint,
        group.num_envs,
        port=1,
        delay_frames=rival_delay(group.checkpoint, group.delay_frames),
        temperature=config.temperature,
        seed=seed,
        device=device,
    )
    # The loaded policy is a fresh adapter: inherit the student's fast-step setting (P7-PLAN.md 3.2).
    from melee_rl.adapter import MeleePolicyAdapter

    if isinstance(opponent.policy, MeleePolicyAdapter):
        opponent.policy.fast_step = bool(getattr(student, "fast_step", False))
        opponent.policy.fast_attention = str(getattr(student, "fast_attention", "sdpa"))
        opponent.policy.fast_encoder = bool(getattr(student, "fast_encoder", False))
        opponent.policy.graph_step = bool(getattr(student, "graph_step", False))
    return opponent


__all__ = [
    "CHARACTER_TABLE_KINDS",
    "OPPONENT_TYPES",
    "ArmConfig",
    "CpuOpponent",
    "MimicOpponent",
    "MixGroup",
    "Opponent",
    "OpponentConfig",
    "OtherOpponent",
    "PhillipOpponent",
    "PolicyOpponent",
    "PoolConfig",
    "SelfOpponent",
    "SmashBotOpponent",
    "arm_characters",
    "arm_names",
    "make_group_opponent",
    "make_opponent",
    "parse_mix_spec",
    "rival_delay",
    "slippi_torch_opponent",
    "split_rival_delay",
]
