"""``AsyncEnvProtocol`` / ``AsyncEnvAdapter`` (P8-PLAN.md 3.3): chunk buffering, FIFO pops, the
peek-without-consume discipline, the deadlock guard, reset reseeding and the sync-mixing guard."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from controller_codec import ButtonState, ControllerState, StickState
from melee_rl.env.async_env import AsyncEnvAdapter, AsyncEnvProtocol
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.env.protocol import EnvOutput, EnvProtocol
from tensor_batch import BUTTON_ORDER

B = 2


def _control(main_x: float) -> ControllerState:
    def full(value: float) -> torch.Tensor:
        return torch.full((B,), value, dtype=torch.float32)

    return ControllerState(
        main_stick=StickState(x=full(main_x), y=full(0.5)),
        c_stick=StickState(x=full(0.5), y=full(0.5)),
        shoulder=full(0.0),
        buttons=ButtonState(**{name: torch.zeros(B, dtype=torch.bool) for name in BUTTON_ORDER}),
    )


class _CountingEnv:
    """A ``DummyMeleeEnv`` that records every step / reset / close (the adapter's inner env)."""

    def __init__(self, num_envs: int = B, game_frames: int = 10_000) -> None:
        self.inner = DummyMeleeEnv(DummyEnvConfig(num_envs=num_envs, seed=5, game_frames=game_frames))
        self.stepped: list[Mapping[int, ControllerState]] = []
        self.resets = 0
        self.closes = 0

    @property
    def num_envs(self) -> int:
        return self.inner.num_envs

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return self.inner.controlled_ports

    def current(self) -> EnvOutput:
        return self.inner.current()

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        self.stepped.append(actions)
        return self.inner.step(actions)

    def reset(self) -> EnvOutput:
        self.resets += 1
        return self.inner.reset()

    def close(self) -> None:
        self.closes += 1
        self.inner.close()


def test_adapter_satisfies_both_protocols_and_seeds_the_boot_state() -> None:
    env = _CountingEnv()
    adapter = AsyncEnvAdapter(env, chunk_frames=2)
    assert isinstance(adapter, EnvProtocol) and isinstance(adapter, AsyncEnvProtocol)
    assert not isinstance(env.inner, AsyncEnvProtocol)  # the sync env has no push/pop surface
    assert adapter.num_envs == B and adapter.controlled_ports == (0,)
    assert adapter.chunk_frames == 2 and adapter.env is env
    boot = env.current()
    assert adapter.in_flight == 1
    assert adapter.pop() is boot  # the initial state is in transit (slippi-ai envs.py:504-584)
    assert adapter.in_flight == 0 and env.stepped == []


def test_push_buffers_a_chunk_then_flushes_in_order() -> None:
    env = _CountingEnv()
    adapter = AsyncEnvAdapter(env, chunk_frames=3)
    adapter.pop()  # consume the seed
    actions = [{0: _control(x)} for x in (0.1, 0.6, 0.9)]
    adapter.push(actions[0])
    adapter.push(actions[1])
    assert env.stepped == [] and adapter.in_flight == 2  # buffered, not computed
    adapter.push(actions[2])
    assert env.stepped == actions and adapter.in_flight == 3  # the 3rd push flushed the whole chunk
    # FIFO pops; the dummy env echoes the executed controller, so states identify their actions.
    for expected in (0.1, 0.6, 0.9):
        state = adapter.pop()
        echoed = state.frames[0]["p0"]["controller"]["main_stick"]["x"]
        assert echoed.tolist() == pytest.approx([expected] * B, abs=0.05)
    assert adapter.in_flight == 0


def test_peek_returns_the_next_pop_without_consuming() -> None:
    adapter = AsyncEnvAdapter(_CountingEnv(), chunk_frames=1)
    seed = adapter.peek()
    assert adapter.peek() is seed and adapter.in_flight == 1
    assert adapter.pop() is seed
    adapter.push({0: _control(0.3)})
    first = adapter.peek()
    assert adapter.peek() is first and adapter.pop() is first


def test_pop_and_peek_raise_instead_of_blocking() -> None:
    adapter = AsyncEnvAdapter(_CountingEnv(), chunk_frames=2)
    adapter.pop()
    with pytest.raises(RuntimeError, match="no state in flight"):
        adapter.pop()
    adapter.push({0: _control(0.4)})  # 1 of 2 buffered: still nothing computed
    with pytest.raises(RuntimeError, match="1 of 2"):
        adapter.pop()
    with pytest.raises(RuntimeError, match="peek"):
        adapter.peek()


def test_reset_drops_buffers_and_reseeds() -> None:
    env = _CountingEnv()
    adapter = AsyncEnvAdapter(env, chunk_frames=2)
    adapter.pop()
    adapter.push({0: _control(0.2)})  # buffered, never computed
    out = adapter.reset()
    assert env.resets == 1 and env.stepped == []  # the buffered action was dropped, not stepped
    assert adapter.in_flight == 1
    assert adapter.pop() is out
    assert bool(out.needs_reset.all())
    # After the reset the chunk cycle starts fresh: two pushes flush one whole chunk.
    adapter.push({0: _control(0.7)})
    adapter.push({0: _control(0.8)})
    assert len(env.stepped) == 2 and adapter.in_flight == 2


def test_sync_step_delegates_until_async_use_then_raises() -> None:
    env = _CountingEnv()
    adapter = AsyncEnvAdapter(env, chunk_frames=2)
    stepped = adapter.step({0: _control(0.5)})
    assert env.stepped and adapter.current() is stepped
    assert adapter.in_flight == 1  # the seed follows the sync step (a warm pipeline handoff)
    assert adapter.pop() is stepped
    with pytest.raises(RuntimeError, match="asynchronous"):
        adapter.step({0: _control(0.5)})
    # reset() clears the mixing guard.
    adapter.reset()
    adapter.step({0: _control(0.5)})
    adapter.push({0: _control(0.5)})
    with pytest.raises(RuntimeError, match="asynchronous"):
        adapter.step({0: _control(0.5)})


def test_close_delegates_and_construction_validates() -> None:
    env = _CountingEnv()
    adapter = AsyncEnvAdapter(env)
    assert adapter.chunk_frames == 1
    adapter.close()
    assert env.closes == 1
    with pytest.raises(ValueError, match="chunk_frames"):
        AsyncEnvAdapter(_CountingEnv(), chunk_frames=0)
    with pytest.raises(ValueError, match="already asynchronous"):
        AsyncEnvAdapter(AsyncEnvAdapter(_CountingEnv()))
