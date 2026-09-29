"""Environments behind ``EnvProtocol`` (PLAN.md §2.1, §6 P1/P5).

``protocol`` defines the batched environment contract (frame trees, ``needs_reset`` with
slippi-ai's ``frame == -123`` semantics, per-port perspectives); ``dummy`` is the
deterministic toy Melee used by the tests and the dummy-env training configs;
``libmelee_frames`` converts libmelee gamestates into frame trees and ``dolphin`` (P5) runs
headless Dolphins through libmelee behind the same protocol (libmelee is imported only when a
``DolphinEnv`` is built).  Import the submodules you need; this package imports nothing heavy at
module level.
"""

from __future__ import annotations

__all__: list[str] = []
