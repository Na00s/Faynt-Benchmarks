"""INTERFACE.md patch #1: the parallel forward continued from a detached KV-cache prefix.

The oracle is Ali's own cached path (``forward(use_cache=True)``, the actor's ``step`` loop) continued on
a clone of the primed ring.  ``forward(new_frames, prefix=cache)`` must reproduce its hidden states for a
partial ring, a full ring (the oldest slots fall out of the sliding band as the rows advance), an overflowed
ring and an empty ring; a row that resets at its first new frame ignores the prefix; a reset inside the new
rows opens a new segment; padding rows neither attend nor are attended; gradients reach only the new rows;
``prefix=None`` is today's code path bit for bit.  Tolerances are Ali's backbone tolerances
(``tests/test_model.py``): rtol 1e-5 / atol 1e-6.  Written first (3 Sep 2026, session 39), the patch after.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from model import CausalTransformer, GroupedQueryCausalAttention, KVCache, ModelConfig, TransformerBlock

CONFIG_YAML = Path(__file__).resolve().parents[2] / "config.yaml"
INPUT_DIM = 13
B = 3
CONTEXT = 24
RTOL, ATOL = 1.0e-5, 1.0e-6


def _config(**changes: Any) -> ModelConfig:
    """Ali's ``_tiny_config`` (``tests/test_model.py``) with ``context_length`` 24, fp32, no checkpointing."""

    tiny = replace(
        ModelConfig.from_yaml(CONFIG_YAML),
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        context_length=CONTEXT,
        compute_dtype="float32",
        cache_dtype="float32",
        gradient_checkpointing=False,
    )
    return replace(tiny, **changes) if changes else tiny


def _backbone(seed: int = 5, **changes: Any) -> CausalTransformer:
    torch.manual_seed(seed)
    return CausalTransformer(_config(**changes), input_dim=INPUT_DIM)


def _frames(time: int, seed: int) -> torch.Tensor:
    return torch.randn((B, time, INPUT_DIM), generator=torch.Generator().manual_seed(seed))


def _clone(cache: KVCache) -> KVCache:
    return KVCache(
        keys=cache.keys.clone(),
        values=cache.values.clone(),
        valid_length=cache.valid_length.clone(),
        write_position=cache.write_position.clone(),
        next_position=cache.next_position.clone(),
    )


def _prime(backbone: CausalTransformer, history: int, *, seed: int = 11) -> KVCache:
    """The ring an actor holds after ``history`` cached steps (Ali's path); empty when ``history == 0``."""

    cache = backbone.init_cache(B, dtype=torch.float32)
    if history == 0:
        return cache
    backbone.eval()
    with torch.no_grad():
        _, primed = backbone(_frames(history, seed), cache=cache, use_cache=True)
    assert primed is not None
    return primed


def _oracle(
    backbone: CausalTransformer,
    cache: KVCache,
    frames: torch.Tensor,
    reset: torch.Tensor | None = None,
    padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ali's cached path continued on a clone of ``cache``: what the actor computes for ``frames``."""

    backbone.eval()
    with torch.no_grad():
        hidden, _ = backbone(
            frames, reset_mask=reset, padding_mask=padding, cache=_clone(cache), use_cache=True
        )
    return hidden


def _prefix_forward(
    backbone: CausalTransformer,
    cache: KVCache,
    frames: torch.Tensor,
    reset: torch.Tensor | None = None,
    padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """The learner's path: training mode, gradients enabled, attending to the detached prefix."""

    backbone.train()
    hidden, returned = backbone(frames, reset_mask=reset, padding_mask=padding, prefix=cache)
    assert returned is None and hidden.requires_grad
    return hidden


def _plain(backbone: CausalTransformer, frames: torch.Tensor) -> torch.Tensor:
    backbone.eval()
    with torch.no_grad():
        hidden, _ = backbone(frames)
    return hidden


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


@pytest.mark.parametrize(
    ("history", "time"),
    [(0, 8), (10, 8), (CONTEXT, 8), (40, CONTEXT)],
    ids=["empty", "partial", "full-ring", "overflowed-max-rows"],
)
def test_prefix_forward_matches_the_cached_step_loop(history: int, time: int) -> None:
    backbone = _backbone()
    cache = _prime(backbone, history)
    assert cache.valid_length.tolist() == [min(history, CONTEXT)] * B
    assert cache.next_position.tolist() == [history] * B
    frames = _frames(time, seed=23)
    expected = _oracle(backbone, cache, frames)
    actual = _prefix_forward(backbone, cache, frames)
    assert actual.shape == (B, time, backbone.config.d_model)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    # The prefix was consulted: without it the same rows read differently, unless the ring was empty.
    alone = _plain(backbone, frames)
    if history == 0:
        torch.testing.assert_close(actual, alone, rtol=0.0, atol=0.0)
    else:
        assert _max_difference(actual, alone) > 1.0e-4


def test_a_row_resetting_at_its_first_new_frame_ignores_the_prefix() -> None:
    backbone = _backbone()
    cache = _prime(backbone, 10)
    frames = _frames(8, seed=29)
    reset = torch.zeros((B, 8), dtype=torch.bool)
    reset[1, 0] = True
    expected = _oracle(backbone, cache, frames, reset)
    actual = _prefix_forward(backbone, cache, frames, reset)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    # Row 1 reads as a fresh game start; the other rows still use the ring.
    alone = _plain(backbone, frames)
    torch.testing.assert_close(actual[1], alone[1], rtol=RTOL, atol=ATOL)
    assert _max_difference(actual[0], alone[0]) > 1.0e-4


def test_a_reset_inside_the_new_rows_opens_a_segment() -> None:
    backbone = _backbone()
    cache = _prime(backbone, CONTEXT)
    frames = _frames(8, seed=31)
    reset = torch.zeros((B, 8), dtype=torch.bool)
    reset[0, 3] = True
    reset[2, 5] = True
    expected = _oracle(backbone, cache, frames, reset)
    actual = _prefix_forward(backbone, cache, frames, reset)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    # Rows after a reset see neither the prefix nor the rows before it: another ring changes nothing there.
    other = _prime(backbone, CONTEXT, seed=12)
    changed = _prefix_forward(backbone, other, frames, reset)
    torch.testing.assert_close(actual[0, 3:], changed[0, 3:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2, 5:], changed[2, 5:], rtol=0.0, atol=0.0)
    assert _max_difference(actual[1], changed[1]) > 1.0e-4


def test_padding_rows_neither_attend_nor_are_attended() -> None:
    backbone = _backbone()
    cache = _prime(backbone, 10)
    frames = _frames(8, seed=37)
    padding = torch.ones((B, 8), dtype=torch.bool)
    padding[0, 2] = False
    padding[1, 5] = False
    expected = _oracle(backbone, cache, frames, padding=padding)
    actual = _prefix_forward(backbone, cache, frames, padding=padding)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    assert torch.count_nonzero(actual[0, 2]) == 0 and torch.count_nonzero(actual[1, 5]) == 0
    perturbed = frames.clone()
    perturbed[0, 2] += 1_000.0
    perturbed[1, 5] += 1_000.0
    changed = _prefix_forward(backbone, cache, perturbed, padding=padding)
    torch.testing.assert_close(changed, actual, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("checkpointing", [False, True], ids=["plain", "checkpointed"])
def test_gradients_reach_the_new_rows_only(checkpointing: bool) -> None:
    backbone = _backbone(gradient_checkpointing=checkpointing)
    cache = _prime(backbone, 10)
    cache.keys.requires_grad_(True)
    cache.values.requires_grad_(True)
    frames = _frames(8, seed=41)
    reset = torch.zeros((B, 8), dtype=torch.bool)
    reset[2, 4] = True
    expected = _oracle(backbone, cache, frames, reset)
    hidden = _prefix_forward(backbone, cache, frames, reset)
    torch.testing.assert_close(hidden, expected, rtol=RTOL, atol=ATOL)
    hidden.square().sum().backward()
    weight = backbone.input_projection.weight
    assert weight.grad is not None and float(weight.grad.abs().sum()) > 0.0
    assert cache.keys.grad is None and cache.values.grad is None
    # Checkpointing changes neither the forward nor the gradient (same seed, opposite flag).
    other = _backbone(gradient_checkpointing=not checkpointing)
    other_hidden = _prefix_forward(other, cache, frames, reset)
    other_hidden.square().sum().backward()
    torch.testing.assert_close(other_hidden, hidden, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(other.input_projection.weight.grad, weight.grad, rtol=RTOL, atol=ATOL)


def test_no_prefix_and_an_empty_prefix_are_the_existing_path() -> None:
    backbone = _backbone()
    frames = _frames(8, seed=43)
    reset = torch.zeros((B, 8), dtype=torch.bool)
    reset[1, 2] = True
    padding = torch.ones((B, 8), dtype=torch.bool)
    padding[2, 0] = False
    backbone.train()

    def run(reset_mask: torch.Tensor | None, padding_mask: torch.Tensor | None) -> None:
        plain, _ = backbone(frames, reset_mask=reset_mask, padding_mask=padding_mask)
        explicit, _ = backbone(frames, reset_mask=reset_mask, padding_mask=padding_mask, prefix=None)
        empty = backbone.init_cache(B, dtype=torch.float32)
        emptied, _ = backbone(frames, reset_mask=reset_mask, padding_mask=padding_mask, prefix=empty)
        torch.testing.assert_close(explicit, plain, rtol=0.0, atol=0.0)
        torch.testing.assert_close(emptied, plain, rtol=0.0, atol=0.0)

    run(None, None)
    run(reset, padding)


def test_chronological_all_matches_the_per_layer_reference() -> None:
    backbone = _backbone()
    cache = _prime(backbone, 30)  # overflowed: 24 valid slots of 30 frames
    backbone.eval()
    with torch.no_grad():
        cache.reset_slots(torch.tensor([False, True, False]))
        _, continued = backbone(_frames(5, seed=47), cache=cache, use_cache=True)
    assert continued is cache
    assert cache.valid_length.tolist() == [CONTEXT, 5, CONTEXT]
    assert cache.next_position.tolist() == [35, 5, 35]
    keys, values, valid, positions = cache.chronological_all()
    assert keys.shape == (2, B, CONTEXT, 2, 8) and values.shape == keys.shape
    assert not keys.requires_grad and not values.requires_grad
    assert valid.shape == (B, CONTEXT) and positions.shape == (B, CONTEXT)
    for layer in range(cache.n_layers):
        reference_keys, reference_values, reference_valid = cache.chronological(layer)
        torch.testing.assert_close(keys[layer], reference_keys, rtol=0.0, atol=0.0)
        torch.testing.assert_close(values[layer], reference_values, rtol=0.0, atol=0.0)
        assert torch.equal(valid, reference_valid)
    for row, length in enumerate(cache.valid_length.tolist()):
        start = int(cache.next_position[row]) - length
        assert positions[row, :length].tolist() == list(range(start, start + length))
    # A partial ring is trimmed to its longest history.
    partial = _prime(backbone, 10)
    keys, values, valid, positions = partial.chronological_all()
    assert keys.shape == (2, B, 10, 2, 8) and valid.all()
    reference_keys, _, _ = partial.chronological(1)
    torch.testing.assert_close(keys[1], reference_keys[:, :10], rtol=0.0, atol=0.0)
    assert positions[0].tolist() == list(range(10))
    empty = backbone.init_cache(B, dtype=torch.float32)
    keys, _, valid, positions = empty.chronological_all()
    assert keys.shape == (2, B, 0, 2, 8) and valid.shape == (B, 0) and positions.shape == (B, 0)


def test_prefix_validation() -> None:
    backbone = _backbone()
    cache = _prime(backbone, 10)
    frames = _frames(8, seed=53)
    backbone.eval()
    with torch.no_grad(), pytest.raises(ValueError, match="prefix"):
        backbone(frames, cache=_clone(cache), use_cache=True, prefix=cache)
    small = backbone.init_cache(B - 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="cache K/V shapes"):
        backbone(frames, prefix=small)
    layer = backbone.layers[0]
    assert isinstance(layer, TransformerBlock)
    attention = layer.attention
    assert isinstance(attention, GroupedQueryCausalAttention)
    normalized = layer.attention_norm(backbone.input_projection(frames))
    positions = torch.arange(8).unsqueeze(0).expand(B, -1)
    prefix_keys = torch.zeros((B, 2, 4, 8))
    with pytest.raises(ValueError, match="attention_mask"):
        attention(normalized, positions, None, (prefix_keys, prefix_keys))
