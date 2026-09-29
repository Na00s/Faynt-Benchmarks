"""The CUDA-graph frame step (5 Sep 2026, ``/rl-speedup`` lever 4): bit-identical replay, cache restore after
capture, the adapter's routing and the config rules.  The capture tests need a CUDA device (the local RTX runs
them in the gate); the flag and config tests run everywhere."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from controller_codec import ControllerLabels
from melee_rl.actor import ActorConfig
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.config import (
    PolicyConfig,
    RLConfig,
    RuntimeConfig,
    config_from_mapping,
    config_to_mapping,
    finalize_config,
)
from melee_rl.frames import random_valid_frames
from melee_rl.learner import LearnerConfig
from melee_rl.packed import FramePacker, StaticFrameTree
from melee_rl.train import build_policy
from model import KVCache, ModelConfig

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")


def _labels(batch: int, seed: int, device: torch.device) -> ControllerLabels:
    generator = torch.Generator().manual_seed(seed)
    return ControllerLabels(
        buttons=torch.randint(0, 728, (batch,), generator=generator).to(device),
        main_stick=torch.randint(0, 85, (batch,), generator=generator).to(device),
    )


def _cuda_adapter(tiny_config: ModelConfig) -> MeleePolicyAdapter:
    torch.manual_seed(1234)
    adapter = MeleePolicyAdapter(tiny_config).to(torch.device("cuda"))
    adapter.eval()
    adapter.fast_step = True
    adapter.fast_encoder = True
    adapter.fast_attention = "lean"
    return adapter


def _drive(
    adapter: MeleePolicyAdapter,
    cache: KVCache,
    packer: FramePacker,
    steps: int,
    *,
    resets: dict[int, list[bool]],
    seed: int,
) -> list[tuple[dict[str, torch.Tensor], ControllerLabels]]:
    device = cache.keys.device
    batch = cache.batch_size
    generator = torch.Generator(device=device).manual_seed(seed)
    outputs = []
    for step in range(steps):
        frame = packer.upload([random_valid_frames((batch,), torch.Generator().manual_seed(1000 + step))])
        prev = _labels(batch, 2000 + step, device)
        reset = torch.tensor(resets.get(step, [False] * batch), dtype=torch.bool, device=device)
        out = adapter.step_sample(frame, prev, reset, cache, temperature=1.0, generator=generator)
        outputs.append(({name: value.clone() for name, value in out.logits.items()}, out.labels))
    return outputs


@cuda_only
def test_graph_replay_is_bit_identical_to_the_eager_fast_path(tiny_config: ModelConfig) -> None:
    from melee_rl.graph_step import GraphedStep

    device = torch.device("cuda")
    batch = 3
    adapter = _cuda_adapter(tiny_config)
    resets = {3: [True, False, False], 9: [False, True, True], 30: [True, True, True]}
    steps = tiny_config.context_length * 2 + 7  # the ring overflows on the way
    packer = FramePacker(batch, device)
    eager_cache = adapter.init_cache(batch)
    adapter.graph_step = False
    eager = _drive(adapter, eager_cache, packer, steps, resets=resets, seed=7)
    graph_cache = adapter.init_cache(batch)
    adapter.graph_step = True
    graphed = _drive(adapter, graph_cache, packer, steps, resets=resets, seed=7)
    adapter.graph_step = False
    assert len(adapter._graphs) == 1
    graph = next(iter(adapter._graphs.values()))
    assert (
        isinstance(graph, GraphedStep) and graph.replays == steps and graph.matches(graph_cache, packer.tree)
    )
    for (logits_a, labels_a), (logits_b, labels_b) in zip(eager, graphed, strict=True):
        for name in ("buttons", "main_stick"):
            assert torch.equal(logits_a[name], logits_b[name])
            assert torch.equal(labels_a.as_dict()[name], labels_b.as_dict()[name])
    for name in ("valid_length", "write_position", "next_position", "keys", "values"):
        assert torch.equal(getattr(eager_cache, name), getattr(graph_cache, name)), name


@cuda_only
def test_a_policy_opponent_replays_its_own_graph_bit_identically(tiny_config: ModelConfig) -> None:
    """17 Sep 2026: with packed frames a policy opponent (an ``other`` / pool / self clone) reads its own
    packer's static tree, so its adapter captures a graph for its cache; the rollouts equal the eager
    opponent's bit for bit."""

    from melee_rl.actor import RolloutWorker
    from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
    from melee_rl.graph_step import GraphedStep
    from melee_rl.opponents import SelfOpponent

    student = _cuda_adapter(tiny_config)
    student.graph_step = True
    rollouts = {}
    for graphed in (False, True):
        env = DummyMeleeEnv(DummyEnvConfig(num_envs=3, seed=5, game_frames=21, players=("policy", "policy")))
        rival = SelfOpponent(student, 3, port=1, delay_frames=2, seed=11)
        rival.policy.graph_step = graphed  # type: ignore[attr-defined]
        config = ActorConfig(rollout_length=8, context_frames=16, packed_frames=True, seed=7)
        worker = RolloutWorker(student, env, rival, config)
        rollouts[graphed] = [worker.rollout() for _ in range(3)]
        graphs = list(rival.policy._graphs.values())  # type: ignore[attr-defined]
        if graphed:
            assert len(graphs) == 1 and isinstance(graphs[0], GraphedStep) and graphs[0].replays >= 3 * 8
        else:
            assert graphs == []
    for eager, replayed in zip(rollouts[False], rollouts[True], strict=True):
        for name in ("controller_exec", "controller_prev_sample", "sampled_labels"):
            for component in ("buttons", "main_stick"):
                assert torch.equal(
                    getattr(getattr(eager, name), component), getattr(getattr(replayed, name), component)
                )
        for name in ("buttons", "main_stick"):
            assert torch.equal(eager.sampled_logits[name], replayed.sampled_logits[name])
        assert torch.equal(eager.rewards, replayed.rewards)


@cuda_only
def test_capture_leaves_the_cache_as_it_was_and_needs_a_static_tree(tiny_config: ModelConfig) -> None:
    from melee_rl.graph_step import GraphedStep

    device = torch.device("cuda")
    batch = 2
    adapter = _cuda_adapter(tiny_config)
    packer = FramePacker(batch, device)
    cache = adapter.init_cache(batch)
    # advance the cache a little, then capture: the capture must not move it
    frame = packer.upload([random_valid_frames((batch,), torch.Generator().manual_seed(1))])
    adapter.step_sample(
        frame, _labels(batch, 2, device), torch.zeros(batch, dtype=torch.bool, device=device), cache
    )
    before = {
        name: getattr(cache, name).clone()
        for name in ("valid_length", "write_position", "next_position", "keys")
    }
    graph = GraphedStep(adapter, cache, frame)
    for name, value in before.items():
        assert torch.equal(getattr(cache, name), value), name
    assert graph.replays == 0
    # a plain (non-static) tree never captures: the adapter falls back to the eager fast path
    adapter.graph_step = True
    plain = random_valid_frames((batch,), torch.Generator().manual_seed(3), device=device)
    adapter.step_sample(
        plain, _labels(batch, 4, device), torch.zeros(batch, dtype=torch.bool, device=device), cache
    )
    assert adapter._graphs == {}
    adapter.graph_step = False
    with pytest.raises(ValueError, match="fast_encoder"):
        adapter.fast_encoder = False
        GraphedStep(adapter, cache, frame)
    adapter.fast_encoder = True


def test_static_frame_tree_marks_the_packer_s_views() -> None:
    packer = FramePacker(2, torch.device("cpu"))
    assert isinstance(packer.tree, StaticFrameTree) and isinstance(packer.tree, dict)
    assert packer.upload([random_valid_frames((2,), torch.Generator().manual_seed(5))]) is packer.tree


def test_graph_step_flag_rules(tiny_config: ModelConfig) -> None:
    assert PolicyConfig().graph_step is False
    with pytest.raises(ValueError, match="graph_step"):
        PolicyConfig(graph_step=True)
    with pytest.raises(ValueError, match="graph_step"):
        PolicyConfig(graph_step=True, fast_step=True)
    policy = PolicyConfig(graph_step=True, fast_step=True, fast_encoder=True)
    tiny_actor = ActorConfig(rollout_length=8, context_frames=16)
    config = RLConfig(
        policy=policy,
        runtime=RuntimeConfig(device="cpu"),
        actor=tiny_actor,
        learner=LearnerConfig(kl_teacher_weight=0.0),  # a random init has no teacher to anchor to
    )
    mapping = config_to_mapping(config)
    assert mapping["policy"]["graph_step"] is True and config_from_mapping(mapping).policy == policy
    with pytest.raises(ValueError, match="packed_frames"):
        finalize_config(config, tiny_config)
    packed = replace(config, actor=replace(tiny_actor, packed_frames=True))
    with pytest.raises(ValueError, match="cuda"):
        finalize_config(packed, tiny_config)
    ready = replace(packed, runtime=RuntimeConfig(device="cuda"))
    assert finalize_config(ready, tiny_config).policy.graph_step is True
    built = build_policy(config, tiny_config, torch.device("cpu"))
    assert built.graph_step is True and built.clone_frozen().graph_step is True
