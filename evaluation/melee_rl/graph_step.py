"""CUDA-graph replay of the actor's frame step (5 Sep 2026, ``/rl-speedup`` lever 4).

The actor's step (``MeleePolicyAdapter.step_sample`` on the fast path) dispatches about 2 200 kernels per
frame from Python; on the bench of 5 Sep 2026 that dispatch was half of the 60 ms call, the GPU idling
between launches.  :class:`GraphedStep` captures the whole step once -- the sync-free encoder
(``melee_rl.fast_encode``), the ring step (``melee_rl.fast_step``) and the head's residual and buttons
logits -- into a ``torch.cuda.CUDAGraph`` against static buffers, and replays it per frame with one
launch.  What stays outside the graph is what
must: the two ``torch.multinomial`` draws (the caller's generator is consumed exactly as before) and the
stick component, whose logits condition on the sampled buttons.  A replay runs the captured kernels on the
same tensors, so the logits, labels and cache are bit-identical to the eager fast path
(``tests/rl/test_graph_step.py``).

The static inputs are the frame tree's leaves (the :class:`melee_rl.packed.FramePacker`'s device buffers,
the same tensors every frame -- the adapter captures only for its :class:`~melee_rl.packed.StaticFrameTree`),
the previous-action labels and the reset mask (copied into device buffers per frame), and the cache, which the
ring step updates in place.  Warm-up and capture run the step on whatever the buffers hold and advance the
cache; a snapshot taken before is copied back after, so capturing is invisible to the rollout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from controller_codec import ControllerLabels
from melee_rl.frames import FrameTree, tree_leaves
from melee_rl.kv_cache import CACHE_FIELDS, snapshot_cache
from melee_rl.protocol import StepOutput
from model import KVCache, _AutoregressiveComponent

if TYPE_CHECKING:
    from melee_rl.adapter import MeleePolicyAdapter


def _sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> torch.Tensor:
    """``fast_sample``'s sample branch for one component (``model.py:1904-1911``)."""

    if temperature > 0.0:
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        return torch.multinomial(flat, 1, generator=generator).reshape(logits.shape[:-1])
    return logits.float().argmax(dim=-1)


class GraphedStep:
    """One captured frame step bound to ``(adapter, cache, frame)``; see the module docstring."""

    def __init__(
        self,
        adapter: MeleePolicyAdapter,
        cache: KVCache,
        frame: FrameTree,
        *,
        warmup: int = 2,
    ) -> None:
        if cache.keys.device.type != "cuda":
            raise ValueError("GraphedStep needs a CUDA cache")
        if not adapter.fast_step or not adapter.fast_encoder:
            raise ValueError(
                "GraphedStep needs adapter.fast_step and adapter.fast_encoder (a sync-free step)"
            )
        self.adapter = adapter
        self.cache = cache
        self.frame = frame
        self.device = cache.keys.device
        batch = cache.batch_size
        for path, leaf in tree_leaves(frame):
            if not isinstance(leaf, torch.Tensor) or leaf.device != self.device or leaf.shape[0] != batch:
                raise ValueError(f"frame leaf {path} must be a [{batch}, ...] tensor on {self.device}")
        self.prev_buttons = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.prev_stick = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.reset = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.stepper = adapter._fast(cache)
        self.head = adapter.controller_head
        self.buttons = cast(_AutoregressiveComponent, self.head.components["buttons"])
        self.stick = cast(_AutoregressiveComponent, self.head.components["main_stick"])
        self.graph = torch.cuda.CUDAGraph()
        self.residual: torch.Tensor
        self.buttons_logits: torch.Tensor
        self.replays = 0
        self._capture(warmup)

    def _body(self) -> tuple[torch.Tensor, torch.Tensor]:
        labels = ControllerLabels(self.prev_buttons, self.prev_stick)
        encoded = self.adapter._encode(self.frame, labels)
        hidden = self.stepper.step(encoded, self.reset)
        residual = self.head.to_residual(hidden)
        return residual, self.buttons.logits(residual, self.prev_buttons)

    def _capture(self, warmup: int) -> None:
        before = snapshot_cache(self.cache, device=None)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with self.adapter._inference(), torch.cuda.stream(stream):
            for _ in range(warmup):
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with self.adapter._inference(), torch.cuda.graph(self.graph):
            self.residual, self.buttons_logits = self._body()
        with torch.no_grad():
            for name in CACHE_FIELDS:
                getattr(self.cache, name).copy_(getattr(before, name))

    def matches(self, cache: KVCache, frame: FrameTree) -> bool:
        return cache is self.cache and frame is self.frame

    def step(
        self,
        prev: ControllerLabels,
        reset: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> StepOutput:
        """One frame: replay the graph, then sample as ``fast_sample`` does (buttons, condition, stick)."""

        self.prev_buttons.copy_(prev.buttons)
        self.prev_stick.copy_(prev.main_stick)
        self.reset.copy_(reset)
        self.graph.replay()
        self.replays += 1
        buttons_logits = self.buttons_logits.clone()  # the graph overwrites its outputs on the next replay
        buttons = _sample(buttons_logits, temperature, generator)
        residual = self.buttons.update(self.residual, buttons)
        stick_logits = self.stick.logits(residual, self.prev_stick)
        stick = _sample(stick_logits, temperature, generator)
        return StepOutput(
            labels=ControllerLabels(buttons=buttons, main_stick=stick),
            logits={"buttons": buttons_logits.float(), "main_stick": stick_logits.float()},
        )


__all__ = ["GraphedStep"]
