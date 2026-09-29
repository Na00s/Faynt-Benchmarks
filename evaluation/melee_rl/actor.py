"""``RolloutWorker``: delayed, batched rollouts of a ``PolicyProtocol`` in an ``EnvProtocol`` (PLAN.md §6 P1,
P6a).

Per env step (slippi-ai ``evaluators.py:296-316`` ``RolloutWorker.rollout``, ``eval_lib.py:127-132``
``DelayedAgent``): record the pending state ``s_k`` in the history ring with the labels executed
on the transition into it; sample ``a_k`` from the policy with ``controller_prev`` = its own previous
sample (neutral at a game start, ``needs_reset`` resets the cache slot); push ``a_k`` into a queue of
depth ``D`` pre-filled with neutral labels and execute the label that comes out (the sample made
``D`` steps earlier); step the environment.  After ``T`` steps the pending state becomes the
bootstrap row and the last ``W = C + T + 1`` rows of the ring form the :class:`Trajectory`.

Context handling (PLAN.md §3.2): in ``context_mode = "reprime"`` the worker clears and re-primes
the KV cache with the ``C`` rows preceding the rollout at every boundary
(``PolicyProtocol.prime``), so the actor logits equal the learner's re-forward of the same
window exactly (fp32); in ``"ring"`` mode it never touches the cache (the model's rolling cache
keeps up to ``context_length`` frames) and the learner's window differs where a segment began
before the window; in ``"prefix"`` mode it keeps the rolling cache too and stores a detached
snapshot of it, taken right before the rollout's first sample, as ``Trajectory.initial_cache``
(``melee_rl.kv_cache``), so the learner reproduces the actor exactly by forwarding the ``T`` rollout
rows against that prefix (INTERFACE.md patch #1).  Rows an environment does not have yet (before
the first rollout, after :meth:`RolloutWorker.reset_history`) are left-padded with the neutral frame
and ``padding = False``.

Trained ports (P6a, slippi-ai ``OpponentConfig.train = True``, ``rl/run_lib.py:112-137, 186-219``):
the worker trains the policy on every port in ``ports`` of the same ``E`` environments.  Each
trained port is one *perspective* of the game (the environment emits the swapped tree for port 1),
and every perspective is a row of the internal batch ``B' = P * E`` laid out **port-major**: row
``p * E + i`` is port ``ports[p]`` of environment ``i``.  The rings, the KV cache, the delay queue
and the generator are sized ``B'``; one ``step_sample`` covers every row; ``actions[port]`` is the
decoded slice ``rows[p * E : (p + 1) * E]``; each row's reward is computed from its own perspective
(the zero-sum terms negate between the two rows of an env).  Row ``p * E + i`` is exactly what a
one-port worker on that port would record, so the actor / learner exactness is unchanged per row.
With ``opponent=None`` every controlled port must be trained (``ports == env.controlled_ports``);
otherwise the opponent plays the remaining controlled port.  :attr:`RolloutWorker.frames_seen`
counts **environment** frames (``T * E`` per rollout); the trajectory batch is ``B'``.

The worker stores the raw logits of every sample (behaviour log-probs are exact at temperature
1.0), one ``torch.Generator`` per worker, rewards computed with ``melee_rl.reward`` and zeroed on
transitions into reset rows, plus the environment's own reward signal.  Timings of the last
rollout are in :attr:`RolloutWorker.timings` (``rollout_s``, ``reprime_s``, ``policy_s``, ``env_s``,
``opponent_s`` -- the opponent's own call per frame, 17 Sep 2026).
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from controller_codec import ControllerLabels, ControllerState, CustomV1Codec
from melee_rl import reward as reward_lib
from melee_rl.env.protocol import EnvOutput, EnvProtocol, canonical_frame_tree
from melee_rl.frames import (
    FrameTree,
    labels_index,
    labels_map,
    labels_to,
    neutral_frame,
    neutral_labels,
    tree_cat,
    tree_index,
    tree_map,
    tree_to,
)
from melee_rl.kv_cache import snapshot_cache
from melee_rl.opponents import Opponent
from melee_rl.packed import FramePacker, PackedRing
from melee_rl.profiling import NULL_PROFILER, SectionProfiler
from melee_rl.protocol import PolicyProtocol
from melee_rl.trajectory import CONTEXT_MODES, Trajectory
from model import KVCache


@dataclass(frozen=True)
class ActorConfig:
    """``[actor]`` settings of a rollout worker.

    ``rollout_length`` is ``T``; ``context_frames`` is ``C`` (``None`` -> ``context_length - T``);
    ``delay_frames`` is ``D`` (P3's config validates ``D == action_offset_frames - 1`` unless the
    mismatch is allowed); ``context_mode`` is ``"reprime"`` (exact, re-primes ``C`` rows per rollout),
    ``"ring"`` (the rolling cache; the learner's re-forward is inexact) or ``"prefix"`` (the rolling
    cache plus its snapshot on the trajectory; exact, INTERFACE.md patch #1).
    ``packed_frames`` (5 Sep 2026, ``/rl-speedup`` lever 3) moves every frame to the policy device in one
    pinned copy per dtype into static buffers and keeps the history ring packed the same way
    (``melee_rl.packed``): three copies and three ring writes per frame instead of 78 and 88, bit-identical.
    ``batch_steps`` is ``b``, the frames per policy call of the delay pipeline (P8): 1 is today's
    lockstep loop; ``b > 1`` selects :class:`melee_rl.pipeline.PipelinedRolloutWorker` and needs
    the slack ``(b - 1) + (chunk_frames - 1) <= D`` (enforced by ``finalize_config`` and the
    pipelined worker); ``b`` must divide ``T`` so every rollout sample lands in the ring.
    ``decision_stride`` (5 Sep 2026, ``/rl-dolphin-speedup`` lever 8, variant (a)) is ``k``: the policy
    still samples every frame, but only every k-th frame's sample (counted per worker since construction
    or the last ``reset_env``) is executed, and it is held for the k - 1 frames between; a reset row forces a
    decision.  The previous-action input the model sees is the executed (held) controller, as in BC data.
    1 is today's loop, bit for bit.  A play-time flag (the match recorder, lever 8's check (ii)): the
    learner has no contract for held rows yet, so ``finalize_config`` refuses ``k > 1`` for a training run.
    ``interleave_arms`` (5 Sep 2026, lever 1) makes a mixed run step its arms frame by frame in alternation
    instead of one whole rollout after the other (``melee_rl.train.interleaved_rollouts``): while one arm's
    policy call and bookkeeping run, the other arm's Dolphins emulate (``WorkerDolphinEnv.begin_step`` /
    ``end_step``).  Exact per environment -- every arm sees the frames it would see alone and draws from its
    own generator in the same order; the trajectories and metrics are bit-identical.  Lockstep only.
    """

    rollout_length: int = 128
    context_frames: int | None = None
    delay_frames: int = 0
    context_mode: str = "reprime"
    batch_steps: int = 1
    packed_frames: bool = False
    decision_stride: int = 1
    interleave_arms: bool = False
    temperature: float = 1.0
    seed: int = 0
    reward: reward_lib.RewardConfig = field(default_factory=reward_lib.RewardConfig)
    check_reward_bounds: bool = True

    def __post_init__(self) -> None:
        if self.decision_stride < 1:
            raise ValueError("decision_stride must be >= 1")
        if self.rollout_length < 1:
            raise ValueError("rollout_length must be >= 1")
        if self.context_frames is not None and self.context_frames < 0:
            raise ValueError("context_frames must be >= 0")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if self.delay_frames >= self.rollout_length:
            raise ValueError("delay_frames must be smaller than rollout_length")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}, got {self.context_mode!r}")
        if self.batch_steps < 1:
            raise ValueError("batch_steps must be >= 1")
        if self.rollout_length % self.batch_steps:
            raise ValueError(
                f"batch_steps {self.batch_steps} must divide rollout_length {self.rollout_length} "
                "(every one of the T samples must land in the ring at the rollout tail, P8-PLAN.md 2)"
            )
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and >= 0")

    def resolve_context_frames(self, context_length: int) -> int:
        """``C`` for a policy with ``context_length``; checks ``C + T <= context_length``."""

        context = context_length - self.rollout_length if self.context_frames is None else self.context_frames
        if context < 0 or context + self.rollout_length > context_length:
            raise ValueError(
                f"context_frames {context} + rollout_length {self.rollout_length} must fit in "
                f"context_length {context_length}"
            )
        return context


@dataclass(frozen=True)
class _Rows:
    frames: FrameTree
    controller_exec: ControllerLabels
    controller_prev: ControllerLabels
    sampled: ControllerLabels
    is_resetting: torch.Tensor
    padding: torch.Tensor
    frame_index: torch.Tensor


def _mask_labels(labels: ControllerLabels, keep: torch.Tensor) -> ControllerLabels:
    """Neutral labels where ``keep`` is false."""

    def blank(component: torch.Tensor) -> torch.Tensor:
        return torch.where(keep, component, torch.zeros_like(component))

    return labels_map(blank, labels)


def _select_labels(take: torch.Tensor, new: ControllerLabels, held: ControllerLabels) -> ControllerLabels:
    """``new`` where ``take`` is true, ``held`` elsewhere (the decision-stride hold, per row)."""

    return ControllerLabels(
        buttons=torch.where(take, new.buttons, held.buttons),
        main_stick=torch.where(take, new.main_stick, held.main_stick),
    )


def _write_tree(ring: Mapping[str, Any], row: Mapping[str, Any], slot: int) -> None:
    for key, value in ring.items():
        if isinstance(value, Mapping):
            _write_tree(value, row[key], slot)
        else:
            value[:, slot] = row[key]


def _check_ports(env: EnvProtocol, opponent: Opponent | None, ports: tuple[int, ...]) -> None:
    controlled = tuple(env.controlled_ports)
    if not ports or len(set(ports)) != len(ports):
        raise ValueError(f"ports must be a non-empty tuple of distinct ports, got {ports!r}")
    for port in ports:
        if port not in controlled:
            raise ValueError(f"port {port} is not controlled by the environment ({controlled})")
    if opponent is None:
        if set(ports) != set(controlled):
            raise ValueError(
                f"opponent is None but the environment controls ports {controlled} and only {ports} are "
                "trained; pass an opponent for the remaining port or train every controlled port"
            )
        return
    if opponent.port in ports:
        raise ValueError(
            f"the opponent must play a different port: port {opponent.port} is a trained port ({ports})"
        )
    if opponent.controls_port != (opponent.port in controlled):
        raise ValueError(
            f"opponent on port {opponent.port} controls_port={opponent.controls_port} but the env "
            f"controls ports {controlled}"
        )
    covered = set(ports) | ({opponent.port} if opponent.controls_port else set())
    if covered != set(controlled):
        raise ValueError(
            f"ports {ports} plus the opponent's port {opponent.port} do not cover the controlled ports "
            f"{controlled}"
        )


class RolloutWorker:
    """Rolls ``adapter`` out in ``env`` on ``ports`` against ``opponent``; see the module docstring."""

    def __init__(
        self,
        adapter: PolicyProtocol,
        env: EnvProtocol,
        opponent: Opponent | None,
        config: ActorConfig,
        *,
        ports: tuple[int, ...] = (0,),
        profiler: SectionProfiler | None = None,
        name: str = "worker",
    ) -> None:
        ports = tuple(ports)
        _check_ports(env, opponent, ports)
        self.adapter = adapter
        self.name = name
        self._profiler: SectionProfiler = NULL_PROFILER if profiler is None else profiler
        self.env = env
        self.opponent = opponent
        self.config = config
        self.ports = ports
        self._envs = env.num_envs
        self._num_ports = len(ports)
        self._context = config.resolve_context_frames(adapter.context_length)
        self._length = config.rollout_length
        self._delay = config.delay_frames
        self._batch = self._envs * self._num_ports
        self._capacity = self._context + self._length + 1
        self._cache = adapter.init_cache(self._batch)
        self._device = self._cache.keys.device
        self._generator = torch.Generator(device=self._device).manual_seed(config.seed)
        self._codec = CustomV1Codec()
        batch, capacity, device = self._batch, self._capacity, self._device
        self._packer: FramePacker | None = None
        self._ring: PackedRing | None = None
        self._opponent_packer: FramePacker | None = None
        if config.packed_frames:
            self._packer = FramePacker(batch, device, ports=self._num_ports)
            self._ring = PackedRing(batch, capacity, device)
            self._frames = self._ring.tree
            if (
                opponent is not None
                and opponent.controls_port
                and not getattr(opponent, "host_frames", False)
                and getattr(opponent, "act_state", None) is None
            ):
                # 17 Sep 2026: a policy opponent (other / pool / a self clone) reads its own packer's static
                # tree, the only input its adapter captures a CUDA graph for; the leaf-wise tree kept it eager
                # (the league smoke's delay-0 arm: 44 s of a 137-s rollout).
                self._opponent_packer = FramePacker(self._envs, device)
        else:
            self._frames = neutral_frame((batch, capacity), device=device)
        self._exec = neutral_labels((batch, capacity), device=device)
        self._prev = neutral_labels((batch, capacity), device=device)
        self._sampled = neutral_labels((batch, capacity), device=device)
        self._is_resetting = torch.zeros((batch, capacity), dtype=torch.bool, device=device)
        self._frame_index = torch.zeros((batch, capacity), dtype=torch.long, device=device)
        self._rows = 0
        self._history_start = torch.zeros(batch, dtype=torch.long, device=device)
        self._neutral = neutral_labels((batch,), device=device)
        self._queue: deque[ControllerLabels] = deque(self._neutral for _ in range(self._delay))
        self._last_sample = self._neutral
        self._exec_labels = self._neutral
        self._stride = config.decision_stride  # lever 8 (a): every k-th sample executes and is held
        self._held = self._neutral
        self._frame_count = 0
        # Lever 1: split the env step around the generator's yield when the env can (set by the trainer's
        # interleaved driver); the finished trajectory of a generator-driven rollout waits here.
        self.split_steps = False
        self._result: Trajectory | None = None
        self._pending = env.current()
        self._pending.validate(self._envs)
        self.timings: dict[str, float] = {}
        self.frames_seen = 0
        self.rollouts_done = 0

    # -- facts --------------------------------------------------------------------------

    @property
    def port(self) -> int:
        """The first trained port (``ports[0]``)."""

        return self.ports[0]

    @property
    def envs(self) -> int:
        """``E``, the number of environments."""

        return self._envs

    @property
    def batch_size(self) -> int:
        """``B' = len(ports) * E``, the rows of every trajectory."""

        return self._batch

    @property
    def context_frames(self) -> int:
        return self._context

    @property
    def rollout_length(self) -> int:
        return self._length

    @property
    def delay_frames(self) -> int:
        return self._delay

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def window_length(self) -> int:
        return self._capacity

    @property
    def generator(self) -> torch.Generator:
        """The sampling generator (its state goes into RL checkpoints, P3)."""

        return self._generator

    # -- resets ---------------------------------------------------------------------------

    def reset_history(self, mask: torch.Tensor | None = None) -> None:
        """Forget the history of the envs in ``mask`` (all by default): earlier rows become padding.

        ``mask`` is ``[E]`` (per environment: every trained perspective of that env) or ``[B']`` (per row).
        """

        if mask is None:
            keep = torch.ones(self._batch, dtype=torch.bool, device=self._device)
        else:
            keep = mask.to(device=self._device, dtype=torch.bool)
            if tuple(keep.shape) == (self._envs,):
                keep = self._tile(keep)
            elif tuple(keep.shape) != (self._batch,):
                raise ValueError(
                    f"mask must have shape ({self._envs},) per environment or ({self._batch},) per row, "
                    f"got {tuple(keep.shape)}"
                )
        start = torch.full_like(self._history_start, self._rows)
        self._history_start = torch.where(keep, start, self._history_start)
        self.adapter.reset_slots(self._cache, keep)

    def reset_env(self) -> None:
        """Hard reset: new games everywhere, empty history, neutral delay queue, opponent reset."""

        self._pending = self.env.reset()
        self._pending.validate(self._envs)
        self.reset_history()
        self._queue = deque(self._neutral for _ in range(self._delay))
        self._last_sample = self._neutral
        self._exec_labels = self._neutral
        self._held = self._neutral
        self._frame_count = 0
        if self.opponent is not None:
            self.opponent.reset_state()

    # -- the ring ---------------------------------------------------------------------------

    def _write_row(
        self,
        row: int,
        frames: FrameTree,
        *,
        controller_exec: ControllerLabels,
        controller_prev: ControllerLabels,
        sampled: ControllerLabels,
        is_resetting: torch.Tensor,
        frame_index: torch.Tensor,
    ) -> None:
        slot = row % self._capacity
        if self._ring is not None and self._packer is not None and frames is self._packer.tree:
            self._ring.write(slot, self._packer.buffers)  # the frame is the packer's static buffers
        else:
            _write_tree(self._frames, frames, slot)
        rings = ((self._exec, controller_exec), (self._prev, controller_prev), (self._sampled, sampled))
        for ring, value in rings:
            ring.buttons[:, slot] = value.buttons
            ring.main_stick[:, slot] = value.main_stick
        self._is_resetting[:, slot] = is_resetting
        self._frame_index[:, slot] = frame_index

    def _write_sampled(self, row: int, sampled: ControllerLabels) -> None:
        slot = row % self._capacity
        self._sampled.buttons[:, slot] = sampled.buttons
        self._sampled.main_stick[:, slot] = sampled.main_stick

    def _gather(self, start: int, length: int) -> _Rows:
        """Global rows ``[start, start + length)`` as left-padded ``[B', length]`` copies."""

        rows = torch.arange(start, start + length, device=self._device)
        slots = torch.remainder(rows, self._capacity)
        valid = (rows.unsqueeze(0) >= self._history_start.unsqueeze(1)) & (rows.unsqueeze(0) >= 0)
        neutral = neutral_frame((self._batch, length), device=self._device)

        def pad(selected: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
            keep = valid.reshape(self._batch, length, *([1] * (selected.ndim - 2)))
            return torch.where(keep, selected, blank)

        frames = tree_map(pad, tree_index(self._frames, (slice(None), slots)), neutral)
        index = (slice(None), slots)
        return _Rows(
            frames=frames,
            controller_exec=_mask_labels(labels_map(lambda c: c[index], self._exec), valid),
            controller_prev=_mask_labels(labels_map(lambda c: c[index], self._prev), valid),
            sampled=_mask_labels(labels_map(lambda c: c[index], self._sampled), valid),
            is_resetting=self._is_resetting[index] & valid,
            padding=valid,
            frame_index=torch.where(valid, self._frame_index[index], 0),
        )

    def _reprime(self) -> None:
        history = self._gather(self._rows - self._context, self._context)
        self.adapter.prime(
            history.frames,
            history.controller_prev,
            history.is_resetting,
            history.padding,
            self._cache,
        )

    # -- rollout ------------------------------------------------------------------------------

    def _tile(self, per_env: torch.Tensor) -> torch.Tensor:
        """``[E] -> [B']`` port-major: the env's value repeated for every trained port."""

        return per_env if self._num_ports == 1 else per_env.repeat(self._num_ports)

    def _observe(self, output: EnvOutput, port: int) -> FrameTree:
        return tree_to(canonical_frame_tree(output.frames[port]), self._device)

    def _observe_rows_cpu(self, output: EnvOutput) -> FrameTree:
        """The ``[B']`` canonical tree of every trained perspective, port-major, still on the CPU."""

        if self._num_ports == 1:
            return canonical_frame_tree(output.frames[self.ports[0]])
        return tree_cat([canonical_frame_tree(output.frames[port]) for port in self.ports])

    def _observe_rows(self, output: EnvOutput) -> FrameTree:
        """The ``[B']`` tree of every trained perspective, port-major, on the policy device."""

        if self._packer is not None:
            return self._packer.upload([canonical_frame_tree(output.frames[port]) for port in self.ports])
        return tree_to(self._observe_rows_cpu(output), self._device)

    def _env_reward(self, output: EnvOutput) -> torch.Tensor:
        """The ``[B']`` environment reward of every trained perspective, on the policy device."""

        reward = torch.cat([output.rewards[port] for port in self.ports])
        return reward.to(device=self._device, dtype=torch.float32)

    def _decode(self, labels: ControllerLabels) -> ControllerState:
        return self._codec.decode(labels_to(labels, "cpu"))

    def _actions(self, labels: ControllerLabels) -> dict[int, ControllerState]:
        """One controller per trained port from the ``[B']`` labels (rows ``p * E : (p + 1) * E``)."""

        envs = self._envs
        return {
            port: self._decode(labels_index(labels, slice(p * envs, (p + 1) * envs)))
            for p, port in enumerate(self.ports)
        }

    def rollout(self) -> Trajectory:
        """Run ``T`` environment steps and return the trajectory of the last ``W`` rows (batch ``B'``).

        The adapter's inference mode is hoisted around the whole rollout (one ``eval()`` /
        ``train()`` pair and one ``no_grad`` instead of one per frame -- the adapter's own
        ``_inference`` is idempotent, P7-PLAN.md 3.2); the mode is restored afterwards.
        """

        with self.inference_mode():
            return self._rollout_frames()

    def _rollout_frames(self) -> Trajectory:
        """The whole rollout, driven to its end (the pipelined worker overrides this with its own loop)."""

        for _ in self.rollout_steps():
            pass
        return self.take_trajectory()

    @contextlib.contextmanager
    def inference_mode(self) -> Iterator[None]:
        """The adapter in ``eval()`` under ``no_grad`` for the duration; the training mode restored after."""

        module = self.adapter if isinstance(self.adapter, torch.nn.Module) else None
        was_training = module is not None and module.training
        if was_training and module is not None:
            module.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if was_training and module is not None:
                module.train(True)

    def take_trajectory(self) -> Trajectory:
        """The trajectory of the last generator-driven rollout (:meth:`rollout_steps`), once."""

        result = self._result
        if result is None:
            raise RuntimeError("no finished rollout to take: drive rollout_steps() to its end first")
        self._result = None
        return result

    def rollout_steps(self) -> Iterator[None]:
        """The rollout as a generator: one yield per frame, right after the frame's actions were sent.

        Run it to its end inside :meth:`inference_mode` and take the trajectory with
        :meth:`take_trajectory`.  With ``split_steps`` and an environment that offers ``begin_step`` /
        ``end_step`` (``WorkerDolphinEnv``), the Dolphins emulate the frame across the yield, so a driver
        that alternates two workers' generators (lever 1, ``melee_rl.train.interleaved_rollouts``) overlaps
        one arm's policy call and bookkeeping with the other arm's emulation.  Driven alone this is
        :meth:`rollout`, frame for frame.
        """

        started = time.perf_counter()
        length = self._length
        reprime_s = policy_s = env_s = opponent_s = 0.0
        if self.config.context_mode == "reprime":
            tick = time.perf_counter()
            self._reprime()
            reprime_s = time.perf_counter() - tick
        profiler = self._profiler
        with profiler.window(f"actor/snapshot/{self.name}"):
            initial_cache = self._snapshot_cache()
        opponent = self.opponent
        opponent_port = opponent.port if opponent is not None and opponent.controls_port else None
        logits_rows: list[dict[str, torch.Tensor]] = []
        env_reward_rows: list[torch.Tensor] = []
        frame_window = f"actor/frames/{self.name}"
        env = self.env
        begin_step = getattr(env, "begin_step", None) if self.split_steps else None
        end_step = getattr(env, "end_step", None) if self.split_steps else None
        split = callable(begin_step) and callable(end_step)
        for _ in range(length):
            with profiler.window(frame_window):
                with profiler.section("actor/observe"):
                    pending = self._pending
                    env_reset = pending.needs_reset.to(self._device)
                    needs_reset = self._tile(env_reset)
                    frames = self._observe_rows(pending)
                    prev = _mask_labels(self._last_sample, ~needs_reset)
                    row = self._rows
                with profiler.section("actor/write"):
                    self._write_row(
                        row,
                        frames,
                        controller_exec=self._exec_labels,
                        controller_prev=prev,
                        sampled=self._neutral,
                        is_resetting=needs_reset,
                        frame_index=self._tile(pending.frame_index.to(self._device)),
                    )
                tick = time.perf_counter()
                with profiler.section("actor/step_sample"):
                    step = self.adapter.step_sample(
                        frames,
                        prev,
                        needs_reset,
                        self._cache,
                        temperature=self.config.temperature,
                        generator=self._generator,
                    )
                policy_s += time.perf_counter() - tick
                with profiler.section("actor/actions"):
                    self._write_sampled(row, step.labels)
                    logits_rows.append(step.logits)
                    executed = step.labels
                    if self._stride > 1:
                        # Lever 8 (a): a decision every k frames (or at a reset row), held in between.
                        decide = needs_reset | (self._frame_count % self._stride == 0)
                        executed = _select_labels(decide, step.labels, self._held)
                        self._held = executed
                    self._frame_count += 1
                    self._last_sample = executed
                    self._queue.append(executed)
                    to_execute = self._queue.popleft()
                    actions = self._actions(to_execute)
                    if opponent is not None and opponent_port is not None:
                        opponent_tick = time.perf_counter()
                        if getattr(opponent, "host_frames", False):
                            # 17 Sep 2026: the slippi-ai port packs the leaves it reads on the host and
                            # uploads three tensors, instead of receiving ~78 per-leaf uploads.
                            observed = canonical_frame_tree(pending.frames[opponent_port])
                        elif self._opponent_packer is not None:
                            tree = canonical_frame_tree(pending.frames[opponent_port])
                            observed = self._opponent_packer.upload([tree])
                        else:
                            observed = self._observe(pending, opponent_port)
                        # An opponent with its own action space (P12: MIMIC) emits a ControllerState
                        # directly; routing it through the codec would quantise values our
                        # vocabularies cannot express.
                        act_state = getattr(opponent, "act_state", None)
                        if act_state is not None:
                            actions[opponent_port] = act_state(observed, env_reset)
                        else:
                            opponent_labels = opponent.act(observed, env_reset)
                            if opponent_labels is None:
                                raise RuntimeError("an opponent that controls a port must return labels")
                            actions[opponent_port] = self._decode(opponent_labels)
                        opponent_s += time.perf_counter() - opponent_tick
                tick = time.perf_counter()
                with profiler.section("actor/env_step"):
                    if split:
                        assert begin_step is not None
                        begin_step(actions)  # the Dolphins emulate across the yield
                    else:
                        self._pending = env.step(actions)
                        self._pending.validate(self._envs)
                env_s += time.perf_counter() - tick
                # One frame window per frame; an interleaving driver's other arm may open its own window
                # across this yield (the profiler keeps one open at a time: profile without interleaving).
                yield
                if split:
                    assert end_step is not None
                    tick = time.perf_counter()
                    with profiler.section("actor/env_step"):
                        self._pending = end_step()
                        self._pending.validate(self._envs)
                    env_s += time.perf_counter() - tick
                with profiler.section("actor/reward"):
                    self._exec_labels = to_execute
                    env_reward_rows.append(self._env_reward(self._pending))
                    self._rows += 1

        # The bootstrap row: the pending state, recorded without sampling (the next rollout samples it).
        with profiler.window(f"actor/finish/{self.name}"):
            pending = self._pending
            needs_reset = self._tile(pending.needs_reset.to(self._device))
            self._result = self._finish_rollout(
                bootstrap_frames=self._observe_rows(pending),
                bootstrap_exec=self._exec_labels,
                needs_reset=needs_reset,
                frame_index=self._tile(pending.frame_index.to(self._device)),
                logits_rows=logits_rows,
                env_reward_rows=env_reward_rows,
                started=started,
                reprime_s=reprime_s,
                policy_s=policy_s,
                env_s=env_s,
                initial_cache=initial_cache,
                opponent_s=opponent_s,
            )

    def _snapshot_cache(self) -> KVCache | None:
        """In ``"prefix"`` mode, the ring as it stands before the rollout's first sample (host copy)."""

        if self.config.context_mode != "prefix":
            return None
        return snapshot_cache(self._cache)

    def _finish_rollout(
        self,
        *,
        bootstrap_frames: FrameTree,
        bootstrap_exec: ControllerLabels,
        needs_reset: torch.Tensor,
        frame_index: torch.Tensor,
        logits_rows: list[dict[str, torch.Tensor]],
        env_reward_rows: list[torch.Tensor],
        started: float,
        reprime_s: float,
        policy_s: float,
        env_s: float,
        initial_cache: KVCache | None = None,
        opponent_s: float = 0.0,
    ) -> Trajectory:
        """The shared rollout tail: write the bootstrap row, gather the window, build the trajectory.

        Both loops call this with ``self._rows`` advanced by exactly ``T`` since the rollout began:
        the lockstep loop from ``self._pending`` / ``self._exec_labels``, the pipelined loop from the
        peeked (not popped) environment state paired with the peeked head of its execution deque.
        """

        context, length, delay = self._context, self._length, self._delay
        self._write_row(
            self._rows,
            bootstrap_frames,
            controller_exec=bootstrap_exec,
            controller_prev=_mask_labels(self._last_sample, ~needs_reset),
            sampled=self._neutral,
            is_resetting=needs_reset,
            frame_index=frame_index,
        )
        window = self._gather(self._rows - (context + length), self._capacity)
        sampled_logits = {
            name: torch.stack([logits[name] for logits in logits_rows], dim=1) for name in logits_rows[0]
        }
        rollout_resets = window.is_resetting[:, context:]
        rollout_frames = tree_index(window.frames, (slice(None), slice(context, None)))
        frame_rewards = reward_lib.compute_rewards(
            rollout_frames,
            self.config.reward,
            check_bounds=self.config.check_reward_bounds,
        )
        frame_rewards = reward_lib.zero_at_resets(frame_rewards, rollout_resets)
        env_rewards = reward_lib.zero_at_resets(torch.stack(env_reward_rows, dim=1), rollout_resets)
        trajectory = Trajectory(
            frames=window.frames,
            controller_exec=window.controller_exec,
            controller_prev_sample=window.controller_prev,
            sampled_labels=window.sampled,
            sampled_logits=sampled_logits,
            rewards=frame_rewards + env_rewards,
            env_rewards=env_rewards,
            is_resetting=window.is_resetting,
            padding=window.padding,
            frame_index=window.frame_index,
            context_frames=context,
            rollout_length=length,
            delay_frames=delay,
            context_mode=self.config.context_mode,
            initial_cache=initial_cache,
        )
        self.frames_seen += length * self._envs
        self.rollouts_done += 1
        self.timings = {
            "rollout_s": time.perf_counter() - started,
            "reprime_s": reprime_s,
            "policy_s": policy_s,
            "env_s": env_s,
            "opponent_s": opponent_s,
        }
        return trajectory


__all__ = ["ActorConfig", "RolloutWorker"]
