"""``PipelinedRolloutWorker``: the delay-funded rollout pipeline (P8-PLAN.md 3.6).

# Ported from slippi-ai slippi_ai/evaluators.py@577965a7731dc53e3472ea63d9e9853a4e9d65fa and
# slippi_ai/eval_lib.py@577965a (MIT) -- the slack equation, the runahead priming, the
# execution deque and the peek-based rollout tail; no code copied.

With a delay ``D``, the action executed into a frame was sampled ``D`` frames earlier, so up to
``D`` future environment frames are already determined: the worker pushes those actions ahead and
the environment emulates while the policy computes.  Buffering ``b = batch_steps`` observations
per policy call and ``k = chunk_frames`` actions per environment message amortises the per-call
and per-message overheads; the slack inequality ``(b - 1) + (k - 1) <= D`` (slippi-ai
``evaluators.py:157-182``) guarantees neither side ever starves, and ``runahead r = D - (b - 1)``
actions are primed before the first rollout.

**The pipeline reorders computation, not data** (P8-PLAN.md 2): with FIFO action consumption the
sample made at state ``s_u`` still executes into ``s_{u + D + 1}``, exactly like the lockstep
delay queue, and the generator draws keep their per-frame order inside a batch -- so on the dummy
environment the produced trajectories are **bit-identical** to :class:`RolloutWorker`'s for every
legal ``(b, k)`` (the golden guard, ``tests/rl/test_pipeline.py``).  The invariants that make the
tail exact, kept from slippi-ai:

* the execution deque starts ``[neutral]``, appends on every push and pops on every state pop, so
  it always pairs a popped state with the action executed into it; its length is ``1 + r`` at
  every rollout boundary (their ``evaluators.py:314-332`` assertion);
* the delay queue holds ``b - 1`` samples at every boundary (prefill ``D``, ``r`` priming pops,
  ``+T`` / ``-T`` per rollout), so a flush always has actions to push;
* the bootstrap row comes from ``env.peek()`` -- **not** popped: the same state is the first pop
  of the next rollout -- paired with the peeked deque head; ``b | T`` puts all ``T`` samples in
  the ring before the tail;
* a popped state's ``rewards`` belong to the transition *into* it: the first pop of a rollout
  re-observes the previous bootstrap state (already accounted), so rewards are buffered from the
  second pop on, and the peek supplies the ``T``-th.

Restrictions (P8-PLAN.md 2): a port-controlling opponent cannot act zero-delay on states already
in flight, so the pipeline runs only against the in-game CPU (``cpu``) or with every controlled
port trained (``self`` + ``train``); ``other`` -- and ``self`` without ``train`` -- need a
delayed-opponent queue that arrives with the delayed opponent-pool campaign.  Evaluation always
uses the lockstep loop.  ``env_s`` here means time *blocked on* the environment (pop waits + push
sends), not emulation time -- the overlap is the point.
"""

from __future__ import annotations

import time
from collections import deque

import torch

from controller_codec import ControllerLabels
from melee_rl.actor import ActorConfig, RolloutWorker, _mask_labels
from melee_rl.env.async_env import AsyncEnvProtocol
from melee_rl.env.protocol import EnvProtocol
from melee_rl.frames import FrameTree, tree_index, tree_stack, tree_to_packed
from melee_rl.opponents import Opponent
from melee_rl.profiling import SectionProfiler
from melee_rl.protocol import PolicyProtocol
from melee_rl.trajectory import Trajectory


class PipelinedRolloutWorker(RolloutWorker):
    """A :class:`RolloutWorker` that overlaps policy and environment time through the delay queue."""

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
        if not isinstance(env, AsyncEnvProtocol):
            raise TypeError(
                "PipelinedRolloutWorker needs an environment with the asynchronous push/pop surface "
                "(AsyncEnvProtocol); wrap a synchronous environment in AsyncEnvAdapter"
            )
        chunk = env.chunk_frames
        slack = (config.batch_steps - 1) + (chunk - 1)
        if slack > config.delay_frames:
            raise ValueError(
                f"the pipeline slack (batch_steps - 1) + (chunk_frames - 1) <= delay_frames is violated: "
                f"({config.batch_steps} - 1) + ({chunk} - 1) = {slack} > {config.delay_frames}"
            )
        if opponent is not None and opponent.controls_port:
            raise ValueError(
                "a port-controlling opponent cannot act zero-delay inside the pipeline (its states are "
                "already in flight); pipelined mode supports the in-game CPU and two-port self-play -- "
                "a delayed opponent needs its own D-length queue (the delayed opponent-pool extension)"
            )
        super().__init__(adapter, env, opponent, config, ports=ports, profiler=profiler, name=name)
        self._async_env: AsyncEnvProtocol = env
        self._batch_steps = config.batch_steps
        self._runahead = config.delay_frames - (config.batch_steps - 1)
        self._exec_deque: deque[ControllerLabels] = deque([self._neutral])
        for _ in range(self._runahead):
            self._push_action()

    @property
    def batch_steps(self) -> int:
        return self._batch_steps

    @property
    def runahead(self) -> int:
        """``r = D - (b - 1)``: actions kept ahead of the environment at every rollout boundary."""

        return self._runahead

    # -- pushing --------------------------------------------------------------------------

    def _push_action(self) -> float:
        """Pop the delay queue, remember the action in the execution deque, push it to the env.

        Returns the seconds spent inside ``env.push`` (the send / compute half of ``env_s``).
        """

        to_execute = self._queue.popleft()
        self._exec_deque.append(to_execute)
        actions = self._actions(to_execute)
        tick = time.perf_counter()
        self._async_env.push(actions)
        return time.perf_counter() - tick

    # -- the pipelined rollout --------------------------------------------------------------

    def _flush(
        self,
        frames_cpu: list[FrameTree],
        resets: list[torch.Tensor],
        indices: list[torch.Tensor],
        execs: list[ControllerLabels],
        logits_rows: list[dict[str, torch.Tensor]],
    ) -> float:
        """One policy call over the ``b`` buffered frames; writes their ring rows and delay-queue
        entries.  Returns the seconds of the ``step_sample_multi`` call (``policy_s``)."""

        packed = tree_to_packed(tree_stack(frames_cpu, dim=1), self._device)
        reset = torch.stack(resets, dim=1).to(self._device)
        frame_index = torch.stack(indices, dim=1).to(self._device)
        tick = time.perf_counter()
        steps = self.adapter.step_sample_multi(
            packed,
            self._last_sample,
            reset,
            self._cache,
            temperature=self.config.temperature,
            generator=self._generator,
        )
        elapsed = time.perf_counter() - tick
        chain = self._last_sample
        for index, step in enumerate(steps):
            frame_reset = reset[:, index]
            self._write_row(
                self._rows,
                tree_index(packed, (slice(None), index)),
                controller_exec=execs[index],
                controller_prev=_mask_labels(chain, ~frame_reset),
                sampled=step.labels,
                is_resetting=frame_reset,
                frame_index=frame_index[:, index],
            )
            logits_rows.append(step.logits)
            self._queue.append(step.labels)
            chain = step.labels
            self._rows += 1
        self._last_sample = chain
        return elapsed

    def _rollout_frames(self) -> Trajectory:
        started = time.perf_counter()
        length, batch_steps = self._length, self._batch_steps
        reprime_s = policy_s = env_s = 0.0
        if self.config.context_mode == "reprime":
            tick = time.perf_counter()
            self._reprime()
            reprime_s = time.perf_counter() - tick
        # Every buffered frame is flushed inside the loop (b | T), so the ring is complete at each boundary
        # and the snapshot precedes this rollout's first sample exactly as in the lockstep loop.
        initial_cache = self._snapshot_cache()
        logits_rows: list[dict[str, torch.Tensor]] = []
        env_reward_rows: list[torch.Tensor] = []
        frames_cpu: list[FrameTree] = []
        resets: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        execs: list[ControllerLabels] = []
        for step_index in range(length):
            tick = time.perf_counter()
            state = self._async_env.pop()
            env_s += time.perf_counter() - tick
            state.validate(self._envs)
            if step_index > 0:
                # A popped state's rewards belong to the transition into it; the first pop
                # re-observes the previous rollout's bootstrap state, already accounted there.
                env_reward_rows.append(self._env_reward(state))
            frames_cpu.append(self._observe_rows_cpu(state))
            resets.append(self._tile(state.needs_reset))
            indices.append(self._tile(state.frame_index))
            execs.append(self._exec_deque.popleft())
            if len(frames_cpu) == batch_steps:
                policy_s += self._flush(frames_cpu, resets, indices, execs, logits_rows)
                frames_cpu, resets, indices, execs = [], [], [], []
            env_s += self._push_action()

        # The tail: the bootstrap state is peeked, never popped -- it is the next rollout's first pop.
        tick = time.perf_counter()
        peeked = self._async_env.peek()
        env_s += time.perf_counter() - tick
        peeked.validate(self._envs)
        env_reward_rows.append(self._env_reward(peeked))
        return self._finish_rollout(
            bootstrap_frames=self._observe_rows(peeked),
            bootstrap_exec=self._exec_deque[0],
            needs_reset=self._tile(peeked.needs_reset.to(self._device)),
            frame_index=self._tile(peeked.frame_index.to(self._device)),
            logits_rows=logits_rows,
            env_reward_rows=env_reward_rows,
            started=started,
            reprime_s=reprime_s,
            policy_s=policy_s,
            env_s=env_s,
            initial_cache=initial_cache,
        )

    # -- resets --------------------------------------------------------------------------------

    def reset_env(self) -> None:
        """Hard reset, then rebuild the pipeline: fresh execution deque and ``r`` primed pushes."""

        super().reset_env()
        self._exec_deque = deque([self._neutral])
        for _ in range(self._runahead):
            self._push_action()

    def _boundary_invariants(self) -> tuple[int, int, int]:
        """(exec-deque length, delay-queue length, env in_flight) -- test hook for the boundary state."""

        return len(self._exec_deque), len(self._queue), self._async_env.in_flight


__all__ = ["PipelinedRolloutWorker"]
