"""Fixtures for the RL tests (``tests/rl`` only; Ali's tests are untouched).

All RL tests are CPU-only, deterministic, fp32 and network-free (PLAN.md §7).  They run
single-threaded (ENV.md §4): the tiny models are dominated by per-op overhead, and with the
container's 16 OpenMP threads the cached step path is ~30x slower (measured 21 Aug 2026:
parity test 9 s vs 0.3 s).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter, rl_model_config
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.opponents import CpuOpponent
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueConfig
from model import ModelConfig

CONFIG_YAML = Path(__file__).resolve().parents[2] / "config.yaml"


def _tiny_model_config() -> ModelConfig:
    """The 3m profile shrunk for tests: d_model 32, 2 layers, context_length 24, fp32, no checkpointing."""

    base = ModelConfig.from_yaml(CONFIG_YAML, profile="3m")
    return rl_model_config(
        base,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        context_length=24,
        encoder_hidden_size=8,
        item_mlp_layers=1,
        gradient_checkpointing=False,
    )


def _randomize_autoregressive_decoders(policy: MeleePolicyAdapter, generator: torch.Generator) -> None:
    """Give the per-component decoders weight.

    Ali zero-initialises them (``model.py:1676-1677``), so at random init the main-stick
    logits do not depend on the taken buttons; with random decoders, conditioning and
    target-alignment bugs become visible in the parity tests.
    """

    with torch.no_grad():
        for component in policy.controller_head.components.values():
            decoder = cast(nn.Linear, component.decoder)
            decoder.weight.normal_(0.0, 0.5, generator=generator)
            cast(torch.Tensor, decoder.bias).normal_(0.0, 0.5, generator=generator)


@pytest.fixture(autouse=True)
def single_threaded_torch() -> Iterator[None]:
    """Run every RL test with one intra-op thread; restore the caller's setting afterwards."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def tiny_config() -> ModelConfig:
    return _tiny_model_config()


@pytest.fixture
def tiny_adapter(tiny_config: ModelConfig) -> MeleePolicyAdapter:
    torch.manual_seed(1234)
    adapter = MeleePolicyAdapter(tiny_config)
    _randomize_autoregressive_decoders(adapter, torch.Generator().manual_seed(99))
    return adapter


@pytest.fixture
def three_m_config() -> ModelConfig:
    """The real 3m profile (``context_length 256``, gradient checkpointing on) at RL precision."""

    return rl_model_config(ModelConfig.from_yaml(CONFIG_YAML, profile="3m"))


@pytest.fixture
def tiny_value_config() -> ValueConfig:
    """A value backbone matching the tiny policy: d_model 32, 2 layers, 4 heads of 8, d_ff 64."""

    return ValueConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8, d_ff=64)


RolloutFactory = Callable[..., list[Trajectory]]


@pytest.fixture
def rollout_trajectories() -> RolloutFactory:
    """Factory: ``count`` consecutive dummy-env rollouts of ``adapter`` (P2+ learner tests).

    Defaults match ``tests/rl/test_actor.py`` (B = 3, T = 8, C = 16, cpu opponent); ``game_frames``
    21 makes games restart inside the windows.  Pass ``delay=`` for ``D > 0``.  The toy games score
    only after ~60 frames (the cpu has to walk over), so learner tests that need dense rewards pass
    ``synthetic_rewards=s``: ``rewards`` becomes ``s * N(0, 1)`` per transition (frames, labels,
    logits, resets and padding stay real; ``env_rewards`` untouched).
    """

    def make(
        adapter: MeleePolicyAdapter,
        count: int,
        *,
        num_envs: int = 3,
        rollout_length: int = 8,
        context_frames: int = 16,
        delay: int = 0,
        game_frames: int = 21,
        seed: int = 7,
        env_seed: int = 5,
        learnable_signal: bool = False,
        context_mode: str = "reprime",
        synthetic_rewards: float | None = None,
    ) -> list[Trajectory]:
        env = DummyMeleeEnv(
            DummyEnvConfig(
                num_envs=num_envs,
                seed=env_seed,
                game_frames=game_frames,
                learnable_signal=learnable_signal,
            )
        )
        config = ActorConfig(
            rollout_length=rollout_length,
            context_frames=context_frames,
            delay_frames=delay,
            context_mode=context_mode,
            seed=seed,
        )
        worker = RolloutWorker(adapter, env, CpuOpponent(), config)
        trajectories = [worker.rollout() for _ in range(count)]
        if synthetic_rewards is not None:
            generator = torch.Generator().manual_seed(seed + 1000)
            scale = synthetic_rewards
            trajectories = [
                replace(item, rewards=scale * torch.randn(item.rewards.shape, generator=generator))
                for item in trajectories
            ]
        return trajectories

    return make
