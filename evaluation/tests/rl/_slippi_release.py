"""A synthetic slippi-ai release file for the port's tests (17 Sep 2026): the real files' structure -- a plain
pickle with upstream enums, a ``config`` table, a ``name_map``, the RL block and ``state['policy']`` in
TensorFlow's variable order -- at a size a test can build, without slippi-ai or TensorFlow on the path."""

from __future__ import annotations

import contextlib
import itertools
import pickle
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


class _Stubbed:
    """A class pickled under an upstream module name, as the real files pickle their enums."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.value,))


@contextlib.contextmanager
def _fake_upstream() -> Iterator[tuple[type, type]]:
    """``slippi_ai.embed.ItemsType`` and ``melee.enums.Character`` stand-ins, importable only while pickling
    (the real ``melee`` is put back afterwards: the loader under test must not need any of them)."""

    items_type = type("ItemsType", (_Stubbed,), {"__module__": "slippi_ai.embed"})
    character = type("Character", (_Stubbed,), {"__module__": "melee.enums"})
    fakes = {
        "slippi_ai": types.ModuleType("slippi_ai"),
        "slippi_ai.embed": types.ModuleType("slippi_ai.embed"),
        "melee": types.ModuleType("melee"),
        "melee.enums": types.ModuleType("melee.enums"),
    }
    fakes["slippi_ai.embed"].ItemsType = items_type  # type: ignore[attr-defined]
    fakes["melee.enums"].Character = character  # type: ignore[attr-defined]
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        yield items_type, character
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def tiny_config(
    items: Any,
    *,
    version: int = 5,
    hidden: int = 8,
    layers: int = 2,
    residual: int = 6,
    depth: int = 2,
    max_names: int = 3,
    delay: int = 3,
    mask: bool = True,
) -> dict[str, Any]:
    embed: dict[str, Any] = {
        "player": {
            "xy_scale": 0.05,
            "shield_scale": 0.01,
            "speed_scale": 0.5,
            "with_speeds": False,
            "with_controller": False,
        },
        "controller": {"axis_spacing": 32, "shoulder_spacing": 10},
    }
    if version >= 5:
        embed["player"].update(with_nana=True, legacy_jumps_left=False)
        embed.update(with_randall=True, with_fod=True, items={"type": items, "mlp_sizes": [5, 4]})
    return {
        "version": version,
        "max_names": max_names,
        "network": {
            "name": "tx_like",
            "mlp": {"depth": 2, "width": 128, "dropout_rate": 0.0},
            "tx_like": {
                "hidden_size": hidden,
                "num_layers": layers,
                "ffw_multiplier": 2,
                "recurrent_layer": "lstm",
                "activation": "gelu",
            },
        },
        "controller_head": {
            "independent": {"residual": False},
            "autoregressive": {"residual_size": residual, "component_depth": depth},
            "name": "autoregressive",
        },
        "embed": embed,
        "observation": {"animation": {"mask": mask}},
        "policy": {"train_value_head": False, "delay": delay},
    }


COMPONENT_SIZES = (1,) * 8 + (33,) * 4 + (11,)


def tf_order_arrays(
    config: dict[str, Any], rng: np.random.Generator, *, input_size: int, value_readout: bool = True
) -> list[np.ndarray]:
    """Random weights in TensorFlow's variable order, written out by hand from the Sonnet module tree."""

    tx = config["network"]["tx_like"]
    hidden, layers, ffw = tx["hidden_size"], tx["num_layers"], tx["ffw_multiplier"]
    head = config["controller_head"]["autoregressive"]
    residual, depth = head["residual_size"], head["component_depth"]

    def w(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    arrays: list[np.ndarray] = []
    for size in COMPONENT_SIZES:  # res_blocks[k]: decoder (b, w), then the encoder MLP's layers (b, w)
        arrays += [w(residual), w(size, residual)]
        widths = [residual + size, *([residual] * depth), size]
        for fan_in, fan_out in itertools.pairwise(widths):
            arrays += [w(fan_out), w(fan_in, fan_out)]
    arrays += [w(residual), w(hidden, residual)]  # to_residual
    items = config["embed"].get("items")
    if items is not None and getattr(items["type"], "value", items["type"]) == "mlp":
        widths = [254, *items["mlp_sizes"]]
        for fan_in, fan_out in itertools.pairwise(widths):
            arrays += [w(fan_out), w(fan_in, fan_out)]
    arrays += [w(hidden), w(input_size, hidden)]  # the encoder
    for _ in range(layers):
        arrays += [w(hidden, 4 * hidden), w(hidden, 4 * hidden), w(4 * hidden)]  # LSTM: w_h, w_i, b
        arrays += [w(hidden), 1.0 + w(hidden)]  # LayerNorm: bias, scale
        arrays += [w(hidden * ffw), w(hidden, hidden * ffw), w(hidden), w(hidden * ffw, hidden)]
    if value_readout:
        arrays += [w(1), w(hidden, 1)]
    return arrays


def v5_input_size(max_names: int, item_out: int) -> int:
    player = 4 + 399 + 33 + 1 + 7 + 1 + 1
    game = 2 * (player + player + 1) + 64 + 2 + 2 + 15 * item_out
    return game + (8 + 33 * 4 + 11) + max_names


def v4_input_size(max_names: int) -> int:
    player = 4 + 399 + 33 + 1 + 6 + 1 + 1
    return 2 * player + 64 + (8 + 33 * 4 + 11) + max_names


def write_release(
    path: Path,
    *,
    version: int = 5,
    names: tuple[str, ...] = ("Cody", "Hax"),
    value_readout: bool = True,
    seed: int = 7,
    **config_changes: Any,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Write a tiny release to ``path`` (hidden 8 x 2 layers, residual 6, three names by default); returns the
    raw config and the arrays in TensorFlow order."""

    with _fake_upstream() as (items_type, character):
        config = tiny_config(items_type("mlp"), version=version, **config_changes)
        size = v5_input_size(config["max_names"], 4) if version >= 5 else v4_input_size(config["max_names"])
        arrays = tf_order_arrays(
            config, np.random.default_rng(seed), input_size=size, value_readout=value_readout
        )
        state = {
            "config": config,
            "name_map": {"Master Player": 0, "Cody": 1, "cody": 1, "Hax": 2},
            "rl_config": {"agent": {"name": list(names), "char": [character(1), character(22)]}},
            "state": {"policy": tuple(arrays)},
            "step": 12,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(state, protocol=4))
    return config, arrays
