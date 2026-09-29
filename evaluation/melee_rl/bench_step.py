"""Local micro-benchmark of the actor's per-frame policy step (P7-PLAN.md 3, item 7).

Measures ``MeleePolicyAdapter.step_sample`` wall time -- exactly what the rollout's ``policy_s``
times (``actor.py``) -- for the slow (Ali's cached loops) and fast (``melee_rl.fast_step``) paths on
synthetic device-resident frames, plus a CUDA-backend parity check (the CI tests are CPU-only, i.e.
math SDPA; the GPU run covers the fused SDPA backends) and a TorchDispatchMode op / sync census.

Not a test; run it by hand::

    python -m melee_rl.bench_step --device cuda --profile 5m --batch 128 --frames 128 --parity

Go / no-go for the Modal benchmark (P7-PLAN.md 3): fast >= 2.5x slow at B = 128, fast-core host
syncs batch-independent, parity within rtol 1e-4 / atol 1e-5.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from controller_codec import ControllerLabels
from melee_rl.adapter import MeleePolicyAdapter, rl_model_config
from melee_rl.frames import random_valid_frames
from model import ModelConfig


def _build(profile: str, device: torch.device) -> MeleePolicyAdapter:
    torch.manual_seed(0)
    adapter = MeleePolicyAdapter(rl_model_config(ModelConfig.from_yaml("config.yaml", profile=profile)))
    adapter.to(device)
    adapter.eval()
    return adapter


def _inputs(batch: int, count: int, device: torch.device) -> list[Any]:
    return [
        random_valid_frames((batch,), torch.Generator().manual_seed(100 + index), device=device)
        for index in range(count)
    ]


def _prev(batch: int, seed: int, device: torch.device) -> ControllerLabels:
    generator = torch.Generator().manual_seed(seed)
    return ControllerLabels(
        buttons=torch.randint(0, 728, (batch,), generator=generator).to(device),
        main_stick=torch.randint(0, 85, (batch,), generator=generator).to(device),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(
    adapter: MeleePolicyAdapter,
    fast: bool,
    *,
    batch: int,
    frames: int,
    device: torch.device,
    warmup: int = 10,
) -> tuple[float, float]:
    adapter.fast_step = fast
    cache = adapter.init_cache(batch)
    generator = torch.Generator(device=device).manual_seed(1)
    inputs = _inputs(batch, 8, device)
    prev = _prev(batch, 2, device)
    quiet = torch.zeros(batch, dtype=torch.bool, device=device)
    reset_row = quiet.clone()
    reset_row[0] = True
    for index in range(warmup):
        adapter.step_sample(inputs[index % 8], prev, quiet, cache, generator=generator)
    _sync(device)
    per_frame: list[float] = []
    started = time.perf_counter()
    for index in range(frames):
        reset = reset_row if index and index % 40 == 0 else quiet
        tick = time.perf_counter()
        adapter.step_sample(inputs[index % 8], prev, reset, cache, generator=generator)
        per_frame.append(time.perf_counter() - tick)
    _sync(device)
    total = time.perf_counter() - started
    return total, statistics.median(per_frame)


class _Census(TorchDispatchMode):
    def __init__(self) -> None:
        self.calls = 0
        self.syncs = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        self.calls += 1
        if func._overloadpacket is torch.ops.aten._local_scalar_dense:
            self.syncs += 1
        return func(*args, **(kwargs or {}))


def _census(adapter: MeleePolicyAdapter, fast: bool, batch: int, device: torch.device) -> tuple[int, int]:
    adapter.fast_step = fast
    cache = adapter.init_cache(batch)
    generator = torch.Generator(device=device).manual_seed(3)
    frame = _inputs(batch, 1, device)[0]
    prev = _prev(batch, 4, device)
    quiet = torch.zeros(batch, dtype=torch.bool, device=device)
    adapter.step_sample(frame, prev, quiet, cache, generator=generator)
    census = _Census()
    with census:
        adapter.step_sample(frame, prev, quiet, cache, generator=generator)
    return census.calls, census.syncs


def _parity(adapter: MeleePolicyAdapter, batch: int, steps: int, device: torch.device) -> None:
    caches = [adapter.init_cache(batch), adapter.init_cache(batch)]
    worst = 0.0
    for step in range(steps):
        frame = random_valid_frames((batch,), torch.Generator().manual_seed(500 + step), device=device)
        prev = _prev(batch, 600 + step, device)
        reset = torch.zeros(batch, dtype=torch.bool, device=device)
        if step and step % 25 == 0:
            reset[step % batch] = True
        outputs = []
        for fast, cache in zip((False, True), caches, strict=True):
            adapter.fast_step = fast
            generator = torch.Generator(device=device).manual_seed(700 + step)
            outputs.append(adapter.step_sample(frame, prev, reset, cache, generator=generator))
        for name in ("buttons", "main_stick"):
            diff = (outputs[0].logits[name] - outputs[1].logits[name]).abs().max().item()
            worst = max(worst, diff)
    slow_cache, fast_cache = caches
    metadata_equal = (
        torch.equal(slow_cache.valid_length, fast_cache.valid_length)
        and torch.equal(slow_cache.write_position, fast_cache.write_position)
        and torch.equal(slow_cache.next_position, fast_cache.next_position)
    )
    print(f"parity over {steps} steps: max |logit diff| {worst:.3e}, metadata equal {metadata_equal}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default="5m")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parity", action="store_true", help="300-step fast-vs-slow parity check")
    parser.add_argument("--dispatch-count", action="store_true", help="ATen op / sync census per frame")
    arguments = parser.parse_args(argv)
    device = torch.device(arguments.device)
    adapter = _build(arguments.profile, device)
    print(
        f"profile {arguments.profile} on {device.type}, B = {arguments.batch}, "
        f"T = {arguments.frames} frames per measurement"
    )
    results = {}
    for name, fast in (("slow", False), ("fast", True)):
        total, median = _run(adapter, fast, batch=arguments.batch, frames=arguments.frames, device=device)
        results[name] = total
        print(f"{name}: {total:.3f} s per {arguments.frames}-frame rollout, {median * 1000:.2f} ms/frame")
    print(f"speedup: {results['slow'] / results['fast']:.2f}x")
    if arguments.dispatch_count:
        for name, fast in (("slow", False), ("fast", True)):
            calls, syncs = _census(adapter, fast, arguments.batch, device)
            print(f"{name}: {calls} ATen dispatches, {syncs} host syncs per frame at B = {arguments.batch}")
    if arguments.parity:
        _parity(adapter, arguments.batch, 300, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
