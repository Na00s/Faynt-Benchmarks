"""The sync-free encoder forward (5 Sep 2026, ``/rl-speedup`` lever 2).

``melee_rl.fast_encode.fast_encode`` re-orchestrates ``SlippiEncoder.forward`` with Ali's modules and helpers
and drops what the actor does not need: the eleven ``bool(tensor.any())`` range checks (each a host round
trip, 11 sync points per actor frame on the bench of 5 Sep 2026) and the two one-hots per player that the
learned-embedding configuration never reads.  The live tensors and their order are unchanged, so the result is
bit-identical (``torch.equal``), and a CUDA graph can capture it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from controller_codec import ControllerLabels
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.fast_encode import fast_encode
from melee_rl.frames import frame_tree_to_game_state, random_valid_frames
from tests.rl.test_fast_step import _drive, _labels


class _Census(TorchDispatchMode):
    def __init__(self) -> None:
        self.syncs = 0
        self.calls = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        self.calls += 1
        if func._overloadpacket is torch.ops.aten._local_scalar_dense:
            self.syncs += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("shape", [(3,), (2, 5)])
def test_fast_encode_is_bit_identical_to_the_encoder(
    tiny_adapter: MeleePolicyAdapter, shape: tuple[int, ...]
) -> None:
    encoder = tiny_adapter.encoder
    frames = random_valid_frames(shape, torch.Generator().manual_seed(11))
    generator = torch.Generator().manual_seed(12)
    labels = ControllerLabels(
        buttons=torch.randint(0, 728, shape, generator=generator),
        main_stick=torch.randint(0, 85, shape, generator=generator),
    )
    with torch.no_grad():
        expected = encoder(frames, labels)
        actual = fast_encode(encoder, frames, labels)
        as_state = fast_encode(encoder, frame_tree_to_game_state(frames), labels)  # attribute access too
    assert actual.shape == expected.shape == (*shape, encoder.output_dim)
    assert torch.equal(actual, expected) and torch.equal(as_state, expected)


def test_fast_encode_never_reads_the_host(tiny_adapter: MeleePolicyAdapter) -> None:
    encoder = tiny_adapter.encoder
    frames = random_valid_frames((4,), torch.Generator().manual_seed(21))
    labels = _labels(4, 22)
    with torch.no_grad():
        slow = _Census()
        with slow:
            encoder(frames, labels)
        fast = _Census()
        with fast:
            fast_encode(encoder, frames, labels)
    # On the CPU ``F.one_hot`` validates its input with two host reads (the CUDA kernel makes none: the
    # bench of 5 Sep 2026 traced exactly 11 sync points per actor step).  Ali's forward: 11 range checks
    # + 17 one-hots (4 x [action, character, jumps] + stage + 2 items + 2 controller) = 45; the lean
    # forward drops the checks and the 8 dead one-hots: 9 one-hots = 18 reads, all of them CPU-only.
    assert slow.syncs == 45 and fast.syncs == 18 and fast.calls < slow.calls


def test_fast_encoder_flag_routes_the_step_and_keeps_the_logits(tiny_adapter: MeleePolicyAdapter) -> None:
    resets = {3: [True, False, False], 9: [False, True, True]}
    outputs = []
    for flag in (False, True):
        tiny_adapter.fast_encoder = flag
        cache = tiny_adapter.init_cache(3)
        outputs.append(_drive(tiny_adapter, True, cache, 12, batch=3, resets=resets))
    tiny_adapter.fast_encoder = False
    for a, b in zip(outputs[0], outputs[1], strict=True):
        for name in ("buttons", "main_stick"):
            assert torch.equal(a.logits[name], b.logits[name])
            assert torch.equal(a.labels.as_dict()[name], b.labels.as_dict()[name])


def test_fast_encode_refuses_configurations_it_does_not_cover(tiny_adapter: MeleePolicyAdapter) -> None:
    from dataclasses import replace

    from model import SlippiEncoder

    named = SlippiEncoder(
        replace(tiny_adapter.config, condition_on_player_name=True, player_name_vocab_size=4),
        tiny_adapter.codec,
    )
    frames = random_valid_frames((2,), torch.Generator().manual_seed(31))
    with pytest.raises(NotImplementedError, match="player"):
        fast_encode(named, frames, _labels(2, 32))
