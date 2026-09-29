"""KV-cache snapshots for ``context_mode = "prefix"`` (INTERFACE.md patch #1; session 39, 3 Sep 2026).

The actor snapshots its ring cache right before the first sampled step of a rollout; the trajectory
carries the snapshot (``Trajectory.initial_cache``) and the learner hands it to
``PolicyProtocol.unroll_window(..., prefix=...)`` -- slippi-ai's ``trajectory.initial_state``
(``evaluators.py:40``, ``rl/learner.py:237-241``) for a KV-cache transformer.  A snapshot is a detached
copy of all five ``KVCache`` tensors (K/V ``[L, B, C, K, Dh]`` and the ``[B]`` metadata); the helpers
below move, slice and concatenate it along the environment axis (dim 1 of K/V, dim 0 of the metadata)
the way ``Trajectory.to`` / ``index_envs`` / ``batch`` treat every other field.  A fast-path cache
(``melee_rl.fast_step``) keeps stale K/V in slots it has invalidated; that is harmless here because the
prefix reader (``KVCache.chronological_all``) gathers by ``valid_length`` / ``write_position`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from model import KVCache

CACHE_FIELDS: tuple[str, ...] = ("keys", "values", "valid_length", "write_position", "next_position")


def snapshot_cache(cache: KVCache, device: torch.device | str | None = "cpu") -> KVCache:
    """A detached copy of ``cache`` on ``device`` (``None`` keeps the cache's device).

    The source keeps rolling; the copy never changes.  The default parks the snapshot in host memory
    so the rollouts accumulated for one learner step do not hold their prefixes on the GPU
    (``Trajectory.to`` moves it back with the rest of the trajectory).
    """

    def copy(tensor: torch.Tensor) -> torch.Tensor:
        target = tensor.device if device is None else torch.device(device)
        return tensor.detach().to(target, copy=True)

    return KVCache(**{name: copy(getattr(cache, name)) for name in CACHE_FIELDS})


def cache_to(cache: KVCache, device: torch.device | str) -> KVCache:
    """Every tensor of ``cache`` on ``device`` (no copy when it is already there)."""

    return KVCache(**{name: getattr(cache, name).to(device) for name in CACHE_FIELDS})


def cache_index(cache: KVCache, index: Any) -> KVCache:
    """The environments ``index`` (slice, list or long tensor): dim 1 of K/V, dim 0 of the metadata."""

    return KVCache(
        keys=cache.keys[:, index],
        values=cache.values[:, index],
        valid_length=cache.valid_length[index],
        write_position=cache.write_position[index],
        next_position=cache.next_position[index],
    )


def cache_cat(caches: Sequence[KVCache]) -> KVCache:
    """Concatenate along the environment axis (same layers, capacity, heads and dtype)."""

    if not caches:
        raise ValueError("cache_cat needs at least one cache")
    return KVCache(
        keys=torch.cat([cache.keys for cache in caches], dim=1),
        values=torch.cat([cache.values for cache in caches], dim=1),
        valid_length=torch.cat([cache.valid_length for cache in caches]),
        write_position=torch.cat([cache.write_position for cache in caches]),
        next_position=torch.cat([cache.next_position for cache in caches]),
    )


__all__ = ["CACHE_FIELDS", "cache_cat", "cache_index", "cache_to", "snapshot_cache"]
