"""Asynchronous environment surface for the delay pipeline (P8-PLAN.md 3.3).

# Ported from slippi-ai slippi_ai/envs.py@577965a7731dc53e3472ea63d9e9853a4e9d65fa (MIT)
# -- the push / pop / peek semantics and the "initial state counts as in transit"
# bookkeeping; no code copied.

``AsyncEnvProtocol`` is what :class:`melee_rl.pipeline.PipelinedRolloutWorker` needs beside the
untouched synchronous ``EnvProtocol``: actions are *pushed* (buffered into chunks of
``chunk_frames``) and states are *popped* in FIFO order, so the environment can hold frames in
flight while the policy computes.  ``in_flight`` counts the states still owed to ``pop`` -- the
seeded initial state plus pushes minus pops since the last reset; the pipeline keeps it at
``1 + runahead`` at every rollout boundary.

``AsyncEnvAdapter`` puts any synchronous environment (the dummy env, the in-process
``DolphinEnv``, test fakes) behind both protocols: a push buffers the actions and the
``chunk_frames``-th push computes the whole chunk with ``env.step`` (compute-on-flush -- no real
overlap; it exists for correctness, tests and the dummy path, exactly the role of slippi-ai's
synchronous ``BatchedEnvironment``).  ``WorkerDolphinEnv`` implements the protocol natively
(``melee_rl.env.dolphin_mp``): there a chunk is one pipe message per worker and the overlap is
real.

``pop`` / ``peek`` on an empty queue **raise** instead of blocking or flushing a partial chunk:
under the pipeline slack ``(batch_steps - 1) + (chunk_frames - 1) <= delay_frames`` a computed
state is always available (P8-PLAN.md 3.3 derives availability ``>= 1``), so an empty pop is an
invariant violation, never something to wait out.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from controller_codec import ControllerState
from melee_rl.env.protocol import EnvOutput, EnvProtocol


@runtime_checkable
class AsyncEnvProtocol(Protocol):
    """Push / pop surface of an environment that can hold frames in flight."""

    @property
    def num_envs(self) -> int: ...

    @property
    def controlled_ports(self) -> tuple[int, ...]: ...

    @property
    def chunk_frames(self) -> int:
        """Actions per chunk: pushes are buffered and computed / sent in groups of this size."""
        ...

    @property
    def in_flight(self) -> int:
        """States still owed to :meth:`pop`: the seeded initial state + pushes - pops since reset."""
        ...

    def current(self) -> EnvOutput:
        """The latest observed state (the boot state, or the last :meth:`pop`)."""
        ...

    def push(self, actions: Mapping[int, ControllerState]) -> None:
        """Queue one controller per controlled port (the ``chunk_frames``-th push flushes)."""
        ...

    def pop(self) -> EnvOutput:
        """The next state in FIFO order; raises when nothing is in flight (never blocks forever)."""
        ...

    def peek(self) -> EnvOutput:
        """The state the next :meth:`pop` will return, without consuming it."""
        ...

    def reset(self) -> EnvOutput:
        """Hard reset; the returned boot state is also seeded as the next :meth:`pop`."""
        ...

    def close(self) -> None: ...


class AsyncEnvAdapter:
    """A synchronous ``EnvProtocol`` environment behind both protocols (compute-on-flush)."""

    def __init__(self, env: EnvProtocol, chunk_frames: int = 1) -> None:
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        if isinstance(env, AsyncEnvProtocol):
            raise ValueError("the environment is already asynchronous; use its own push/pop surface")
        self._env = env
        self._chunk = chunk_frames
        self._buffer: list[Mapping[int, ControllerState]] = []
        self._states: deque[EnvOutput] = deque([env.current()])
        self._async_used = False

    # -- facts --------------------------------------------------------------------------

    @property
    def env(self) -> EnvProtocol:
        """The wrapped synchronous environment."""

        return self._env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return self._env.controlled_ports

    @property
    def chunk_frames(self) -> int:
        return self._chunk

    @property
    def in_flight(self) -> int:
        return len(self._states) + len(self._buffer)

    def current(self) -> EnvOutput:
        return self._env.current()

    # -- the asynchronous surface ---------------------------------------------------------

    def push(self, actions: Mapping[int, ControllerState]) -> None:
        self._async_used = True
        self._buffer.append(actions)
        if len(self._buffer) == self._chunk:
            buffered, self._buffer = self._buffer, []
            for queued in buffered:
                self._states.append(self._env.step(queued))

    def _require_state(self, action: str) -> None:
        if not self._states:
            raise RuntimeError(
                f"{action} with no state in flight ({len(self._buffer)} of {self._chunk} actions "
                "buffered): the pipeline slack (batch_steps - 1) + (chunk_frames - 1) <= delay_frames "
                "guarantees availability, so this is an invariant violation"
            )

    def pop(self) -> EnvOutput:
        self._async_used = True
        self._require_state("pop")
        return self._states.popleft()

    def peek(self) -> EnvOutput:
        self._require_state("peek")
        return self._states[0]

    # -- the synchronous surface ----------------------------------------------------------

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        """Synchronous stepping, legal until the first push / pop; keeps the seed current."""

        if self._async_used or self._buffer:
            raise RuntimeError(
                "step() after asynchronous use: the adapter holds pipeline state (use push/pop, "
                "or reset() first)"
            )
        output = self._env.step(actions)
        self._states.clear()
        self._states.append(output)
        return output

    def reset(self) -> EnvOutput:
        """Hard reset: drop buffered actions and queued states, reseed with the boot state."""

        self._buffer.clear()
        self._states.clear()
        output = self._env.reset()
        self._states.append(output)
        self._async_used = False
        return output

    def close(self) -> None:
        self._env.close()


__all__ = ["AsyncEnvAdapter", "AsyncEnvProtocol"]
