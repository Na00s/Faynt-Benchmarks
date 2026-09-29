"""``step_sample_multi`` (P8-PLAN.md 3.2): b frames in one call == b sequential ``step_sample`` calls.

The contract is exact (bit-identical labels, logits, cache K/V + metadata, and generator state):
the multi call encodes the frames one by one -- the encoder consumes each frame's previous-action
labels, so nothing can be fused -- inside a single inference context, chaining ``controller_prev``
through its own samples and masking the chain at reset rows.
"""

from __future__ import annotations

import pytest
import torch

from controller_codec import ControllerLabels
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.frames import (
    FrameTree,
    frame_tree_to_game_state,
    neutral_labels,
    random_valid_frames,
    tree_stack,
    tree_to_packed,
)
from melee_rl.protocol import StepOutput
from model import KVCache

B = 3


def _frames(steps: int, seed: int) -> tuple[list[FrameTree], FrameTree]:
    generator = torch.Generator().manual_seed(seed)
    per_frame = [random_valid_frames((B,), generator) for _ in range(steps)]
    return per_frame, tree_stack(per_frame, dim=1)


def _resets(steps: int, reset_at: tuple[tuple[int, int], ...]) -> torch.Tensor:
    reset = torch.zeros((B, steps), dtype=torch.bool)
    for row, column in reset_at:
        reset[row, column] = True
    return reset


def _random_labels(seed: int) -> ControllerLabels:
    generator = torch.Generator().manual_seed(seed)
    return ControllerLabels(
        buttons=torch.randint(0, 728, (B,), generator=generator),
        main_stick=torch.randint(0, 85, (B,), generator=generator),
    )


def _mask(labels: ControllerLabels, keep: torch.Tensor) -> ControllerLabels:
    return ControllerLabels(
        buttons=torch.where(keep, labels.buttons, torch.zeros_like(labels.buttons)),
        main_stick=torch.where(keep, labels.main_stick, torch.zeros_like(labels.main_stick)),
    )


def _sequential(
    adapter: MeleePolicyAdapter,
    per_frame: list[FrameTree],
    prev: ControllerLabels,
    reset: torch.Tensor,
    cache: KVCache,
    generator: torch.Generator,
) -> list[StepOutput]:
    """The reference: b ``step_sample`` calls with the masked chain the lockstep worker feeds."""

    outputs: list[StepOutput] = []
    chain = prev
    for index, frame in enumerate(per_frame):
        masked = _mask(chain, ~reset[:, index])
        step = adapter.step_sample(frame, masked, reset[:, index], cache, generator=generator)
        outputs.append(step)
        chain = step.labels
    return outputs


def _assert_steps_equal(multi: list[StepOutput], sequential: list[StepOutput]) -> None:
    assert len(multi) == len(sequential)
    for index, (a, b) in enumerate(zip(multi, sequential, strict=True)):
        assert torch.equal(a.labels.buttons, b.labels.buttons), index
        assert torch.equal(a.labels.main_stick, b.labels.main_stick), index
        for name in ("buttons", "main_stick"):
            assert a.logits[name].dtype == torch.float32
            assert torch.equal(a.logits[name], b.logits[name]), (index, name)


def _assert_caches_equal(a: KVCache, b: KVCache) -> None:
    assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    for name in ("valid_length", "write_position", "next_position"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


@pytest.mark.parametrize("fast", [False, True], ids=["slow", "fast"])
@pytest.mark.parametrize(
    "steps,reset_at",
    [(3, ()), (4, ((0, 0), (1, 2), (2, 3)))],
    ids=["no-resets", "resets-inside"],
)
def test_multi_step_equals_sequential_steps(
    tiny_adapter: MeleePolicyAdapter,
    fast: bool,
    steps: int,
    reset_at: tuple[tuple[int, int], ...],
) -> None:
    tiny_adapter.fast_step = fast
    per_frame, stacked = _frames(steps, seed=17)
    reset = _resets(steps, reset_at)
    prev = _random_labels(23)
    cache_multi = tiny_adapter.init_cache(B)
    cache_seq = tiny_adapter.init_cache(B)
    generator_multi = torch.Generator().manual_seed(5)
    generator_seq = torch.Generator().manual_seed(5)
    multi = tiny_adapter.step_sample_multi(stacked, prev, reset, cache_multi, generator=generator_multi)
    sequential = _sequential(tiny_adapter, per_frame, prev, reset, cache_seq, generator_seq)
    _assert_steps_equal(multi, sequential)
    _assert_caches_equal(cache_multi, cache_seq)
    assert torch.equal(generator_multi.get_state(), generator_seq.get_state())


def test_multi_step_after_prime_and_packed_frames(tiny_adapter: MeleePolicyAdapter) -> None:
    """Post-prime cache positions, a packed device tree and a GameStateBatch all give the same steps."""

    context = 4
    generator = torch.Generator().manual_seed(41)
    history = random_valid_frames((B, context), generator)
    history_prev = neutral_labels((B, context))
    no_reset = torch.zeros((B, context), dtype=torch.bool)
    all_real = torch.ones((B, context), dtype=torch.bool)
    per_frame, stacked = _frames(2, seed=43)
    reset = _resets(2, ((1, 0),))
    prev = _random_labels(47)

    def primed_cache() -> KVCache:
        cache = tiny_adapter.init_cache(B)
        tiny_adapter.prime(history, history_prev, no_reset, all_real, cache)
        return cache

    cache_multi, cache_seq, cache_typed = primed_cache(), primed_cache(), primed_cache()
    generator_multi = torch.Generator().manual_seed(11)
    generator_seq = torch.Generator().manual_seed(11)
    generator_typed = torch.Generator().manual_seed(11)
    packed = tree_to_packed(stacked, "cpu")
    multi = tiny_adapter.step_sample_multi(packed, prev, reset, cache_multi, generator=generator_multi)
    sequential = _sequential(tiny_adapter, per_frame, prev, reset, cache_seq, generator_seq)
    _assert_steps_equal(multi, sequential)
    _assert_caches_equal(cache_multi, cache_seq)
    typed = tiny_adapter.step_sample_multi(
        frame_tree_to_game_state(stacked), prev, reset, cache_typed, generator=generator_typed
    )
    _assert_steps_equal(typed, sequential)


def test_multi_step_b1_degenerates_to_step_sample(tiny_adapter: MeleePolicyAdapter) -> None:
    per_frame, stacked = _frames(1, seed=29)
    reset = _resets(1, ((2, 0),))
    prev = _random_labels(31)
    cache_multi = tiny_adapter.init_cache(B)
    cache_single = tiny_adapter.init_cache(B)
    generator_multi = torch.Generator().manual_seed(3)
    generator_single = torch.Generator().manual_seed(3)
    (multi,) = tiny_adapter.step_sample_multi(stacked, prev, reset, cache_multi, generator=generator_multi)
    single = tiny_adapter.step_sample(
        per_frame[0], _mask(prev, ~reset[:, 0]), reset[:, 0], cache_single, generator=generator_single
    )
    assert torch.equal(multi.labels.buttons, single.labels.buttons)
    assert torch.equal(multi.labels.main_stick, single.labels.main_stick)
    for name in ("buttons", "main_stick"):
        assert torch.equal(multi.logits[name], single.logits[name])
    _assert_caches_equal(cache_multi, cache_single)


def test_multi_step_validation_errors(tiny_adapter: MeleePolicyAdapter) -> None:
    _, stacked = _frames(2, seed=53)
    prev = _random_labels(59)
    cache = tiny_adapter.init_cache(B)
    with pytest.raises(ValueError, match="reset_mask"):
        tiny_adapter.step_sample_multi(stacked, prev, torch.zeros(B, dtype=torch.bool), cache)
    with pytest.raises(ValueError, match="reset_mask"):
        tiny_adapter.step_sample_multi(stacked, prev, torch.zeros((B + 1, 2), dtype=torch.bool), cache)
    with pytest.raises(ValueError, match="at least one"):
        tiny_adapter.step_sample_multi(stacked, prev, torch.zeros((B, 0), dtype=torch.bool), cache)
    with pytest.raises(ValueError, match="leading shape"):
        tiny_adapter.step_sample_multi(stacked, prev, torch.zeros((B, 3), dtype=torch.bool), cache)
    with pytest.raises(ValueError, match="controller_prev"):
        tiny_adapter.step_sample_multi(
            stacked, neutral_labels((B, 2)), torch.zeros((B, 2), dtype=torch.bool), cache
        )
