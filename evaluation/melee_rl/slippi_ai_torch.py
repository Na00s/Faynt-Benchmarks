"""slippi-ai's released policies as a PyTorch port on the GPU (17 Sep 2026, LEAGUE-PLAN.md §3.2).

The league run trains our delay-18 Fox against slippi-ai's own agents -- gm as its twelve characters, the Fox
ditto ``fox_d18_ditto_v4`` and the Falco Fox-killer ``falco_d18_vs_fox_v4`` -- inside the training loop, where
their TensorFlow agent cannot go: it would take CPU from 96 Dolphins, need a TensorFlow + torch image, and
need the raw gamestates the worker environments do not ship (``melee_rl/env/dolphin_mp.py:24-34``). This
module reads a release file without TensorFlow or slippi-ai and plays it from the frame tree the actor already
hands an opponent. The recorder keeps playing their TensorFlow agent (:mod:`melee_rl.slippi_ai_agent`), which
is the independent check; ``melee_rl.slippi_ai_parity`` measures the port against it logit by logit.

Everything below is a transcription of upstream at the pin (``third_party/slippi-ai``, ``577965a``, MIT):

* **Loading** (``slippi_ai/saving.py:61-63``, ``tf/saving.py:17-111``): a release is a plain pickle. The
  unpickler here resolves numpy's array reconstruction and nothing else -- every other global (their enums,
  ``melee.enums.Character``) becomes an inert :class:`ReleaseStub` -- so no upstream code runs and no pickle
  can call a function. ``state['policy']`` is a tuple of arrays in Sonnet's variable order (attributes sorted
  by name, depth first): the head's thirteen components, ``to_residual``, the items MLP, the encoder, then per
  layer the LSTM (``w_h``, ``w_i``, ``b``) and the ``ResBlock`` (LayerNorm ``bias``, ``scale``; two Linears),
  then the policy's unused value readout (:func:`expected_shapes`). ``w_h`` and ``w_i`` share a shape, so only
  the parity gate can catch a swapped read. The config is upgraded as ``tf/saving.py:17-78`` does: a version
  <= 4 file has no Randall / FoD / items / Nana and six jumps, a version <= 3 file no tech mask.
* **The input** (``tf/embed.py``): per player -- percent x 0.01, facing +-1, x / y x 0.05, the action as a
  one-hot of 399 (clamped), the character of 33, invulnerable, jumps of 7 (6 in legacy files, all-zero
  outside), shield x 0.01, on-ground, then the same for Nana plus ``exists`` -- the stage of 64, Randall, the
  FoD platforms, fifteen item slots through one shared relu MLP, the previous *sampled* controller (8 buttons,
  each stick axis a one-hot of ``axis_spacing + 1``, the shoulder of ``shoulder_spacing + 1``) and the name
  tag (one-hot, all-zero outside ``max_names``); every float clamped to +-10. :func:`pack_game` gathers the
  frame tree's leaves into three device tensors first.
* **The core** (``tf/networks.py:171-226, 305, 378-424``, ``tx_like``): Linear encoder, then per layer an LSTM
  added to its input (gates i, f, g, o; one bias, whose forget offset Sonnet folds in at initialisation) and a
  ``ResBlock`` -- LayerNorm **without epsilon**, Linear(``ffw_multiplier`` H), exact GELU, Linear(H),
  residual.
* **The head** (``tf/controller_heads.py:107-202``): residual = Linear(``residual_size``); per component (A,
  B, X, Y, Z, L, R, D_UP, main x, main y, c x, c y, shoulder) a relu MLP of ``component_depth`` hidden layers
  over (residual, that component's previous value), a Bernoulli / categorical sample, and residual +=
  Linear(the sample's embedding).
* **The agent** (``eval_lib.py:90-198``, ``tf/agents.py``, ``observations.py:160-200``): the delay queue is
  primed **once** with ``delay`` dummy outputs (all zero: buttons up, sticks and shoulder at class 0) and
  never cleared; ``needs_reset`` zeroes the LSTM rows only; the network's controller input is the agent's own
  previous sample; a v4+ file masks the opponent's neutral / forward / back tech as neutral tech for the first
  7 frames of the animation (``ACTION_MASKS = {}``: the default mask for every character). A sample decodes to
  Melee's analog grid (sticks k / ``axis_spacing``, shoulder k / ``shoulder_spacing``), which libmelee
  0.47.3's remap cannot move (``melee_rl/slippi_ai_agent.py:31-35``).

Numbers are float32 throughout, with TensorFlow's layouts and operation order (``x @ w + b``), so the port's
logits differ from TensorFlow's by kernel rounding only (the gate: |Δ logit| <= 1e-3 * max(1, |logit|), mean
KL <= 1e-6)."""

from __future__ import annotations

import hashlib
import itertools
import math
import pickle
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from controller_codec import ButtonState, ControllerState, StickState
from melee_rl.frames import FrameTree
from tensor_batch import BUTTON_ORDER, MAX_ITEMS

ACTION_SIZE: Final[int] = 0x18F
"""``embed_action``: a one-hot of 399, clamped (``tf/embed.py:414-416``)."""
CHARACTER_SIZE: Final[int] = 0x21
STAGE_SIZE: Final[int] = 64
ITEM_TYPE_SIZE: Final[int] = 0xEC + 1 + 1
"""``embed_item_type``: ``MAX_ITEM_TYPE + 1`` classes plus the ``EXTRA`` class for anything outside."""
ITEM_STATE_SIZE: Final[int] = 11 + 1 + 1
ITEM_WIDTH: Final[int] = 1 + ITEM_TYPE_SIZE + ITEM_STATE_SIZE + 2
"""One item slot before the MLP: exists, type, state, x, y (254)."""
FLOAT_CLAMP: Final[float] = 10.0
TECH_ACTIONS: Final[tuple[int, int, int]] = (0xC7, 0xC8, 0xC9)
"""libmelee ``Action.NEUTRAL_TECH`` / ``FORWARD_TECH`` / ``BACKWARD_TECH``."""
TECH_MASK_WINDOW: Final[int] = 7
"""``observations.DEFAULT_TECH_MASK_WINDOW``."""
COMPONENTS: Final[tuple[str, ...]] = (*BUTTON_ORDER, "main_x", "main_y", "c_x", "c_y", "shoulder")
"""The head's autoregressive order; also the columns of a packed controller ``[B, 13]`` of class indices."""
SUPPORTED_ACTIVATIONS: Final[tuple[str, ...]] = ("gelu", "relu")


class ReleaseLoadError(ValueError):
    """A file that is not a readable slippi-ai release."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


class ReleaseStub:
    """What an upstream class becomes when a release is read without slippi-ai: its constructor arguments and
    its pickled state, nothing callable (an ``ItemsType('mlp')`` is a stub with ``args == ('mlp',)``)."""

    module: str = ""
    qualname: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __repr__(self) -> str:
        return f"<{self.module}.{self.qualname} args={self.args!r}>"


_NUMPY_GLOBALS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }
)


class _ReleaseUnpickler(pickle.Unpickler):
    _stubs: ClassVar[dict[tuple[str, str], type[ReleaseStub]]] = {}

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _NUMPY_GLOBALS:
            return super().find_class(module.replace("numpy.core.", "numpy._core."), name)
        key = (module, name)
        stub = self._stubs.get(key)
        if stub is None:
            stub = type(name, (ReleaseStub,), {"module": module, "qualname": name})
            self._stubs[key] = stub
        return stub


def load_release_state(path: str | Path) -> dict[str, Any]:
    """A release file as a plain ``dict`` (``config``, ``name_map``, ``state['policy']``, ``rl_config`` /
    ``agent_config``, ``step``), read without importing slippi-ai, TensorFlow or libmelee."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            state = _ReleaseUnpickler(handle).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError) as error:
        raise ReleaseLoadError(f"{source} is not a slippi-ai release: {error}") from error
    policy = state.get("state", {}).get("policy") if isinstance(state, dict) else None
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict) or policy is None:
        raise ReleaseLoadError(f"{source} is not a slippi-ai release (no config / state['policy'])")
    platform = state["config"].get("platform", "tf")
    if platform != "tf" or isinstance(policy, Mapping):
        # 22 Sep 2026: the shared delay-0 Fox is ``platform: 'jax'`` with flax's nested parameter dict.
        # A valid release, just not a Sonnet one: the recorder's ``[slippi_ai]`` agent plays it through
        # upstream's own JAX code (``melee_rl.slippi_ai_agent``); this port reads the TensorFlow layout.
        raise ReleaseLoadError(
            f"{source} is a {str(platform).upper()} release (config platform {platform!r}, "
            f"{type(policy).__name__} parameters): this port reads TensorFlow releases only -- play it "
            "through the recorder's [slippi_ai] agent (melee_rl.slippi_ai_agent) instead"
        )
    if not all(isinstance(array, np.ndarray) for array in policy):
        raise ReleaseLoadError(f"{source}: state['policy'] must hold numpy arrays")
    return state


def enum_value(value: Any) -> Any:
    """An upstream enum read as a stub gives its value (``ItemsType('mlp')`` -> ``'mlp'``); anything else is
    returned as it is."""

    if isinstance(value, ReleaseStub):
        return value.args[0] if value.args else value.state
    return value


def release_arrays(state: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    """``state['state']['policy']`` as a tuple of float32 arrays."""

    return tuple(np.asarray(array, dtype=np.float32) for array in state["state"]["policy"])


def _table(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ReleaseSpec:
    """Everything the port reads from a release's ``config`` (after upstream's upgrade) and its RL block."""

    name: str
    version: int
    delay: int
    max_names: int
    hidden_size: int
    num_layers: int
    ffw_multiplier: int
    activation: str
    residual_size: int
    component_depth: int
    axis_spacing: int
    shoulder_spacing: int
    xy_scale: float
    shield_scale: float
    with_nana: bool
    legacy_jumps_left: bool
    with_randall: bool
    with_fod: bool
    items_type: str
    item_mlp_sizes: tuple[int, ...]
    tech_mask: bool
    name_map: Mapping[str, int]
    rl_names: tuple[str, ...]

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, name: str) -> ReleaseSpec:
        """Read ``state['config']`` the way ``tf/saving.py:upgrade_config`` + ``train_lib.Config``'s defaults
        would.

        Refused rather than approximated: speeds or the controller in the player embedding (our frames carry
        neither), any network but ``tx_like`` with an LSTM, any head but ``autoregressive``, frame
        skipping."""

        config = state["config"]
        raw_version = config.get("version")
        version = 1 if raw_version is None else int(raw_version)
        if version < 2:
            raise ValueError(f"{name}: config version {version} predates the embed config; not supported")
        embed = _table(config, "embed") if version >= 3 else {}
        player = _table(embed, "player")
        controller = _table(embed, "controller")
        if version >= 3:
            axis_spacing = int(controller.get("axis_spacing", 16))
            shoulder_spacing = int(controller.get("shoulder_spacing", 4))
        else:  # tf/saving.py:37-50: the embed config every version-2 policy was trained with
            axis_spacing, shoulder_spacing = 16, 4
        if bool(player.get("with_speeds", False)):
            raise ValueError(
                f"{name}: player.with_speeds is on; our frames carry no speeds (tf/embed.py:454-462)"
            )
        if bool(player.get("with_controller", False)):
            raise ValueError(f"{name}: player.with_controller is on; the port embeds only its own controller")
        if version >= 5:
            items = _table(embed, "items")
            items_type = str(enum_value(items.get("type", "mlp"))).lower()
            mlp_sizes = tuple(int(size) for size in items.get("mlp_sizes", (128, 32)))
            with_nana = bool(player.get("with_nana", True))
            legacy_jumps = bool(player.get("legacy_jumps_left", False))
            with_randall = bool(embed.get("with_randall", True))
            with_fod = bool(embed.get("with_fod", True))
        else:  # tf/saving.py:65-76
            items_type, mlp_sizes = "skip", (128, 32)
            with_nana, legacy_jumps, with_randall, with_fod = False, True, False, False
        if items_type not in ("skip", "flat", "mlp"):
            raise ValueError(f"{name}: unknown items type {items_type!r}")
        if version >= 4:
            observation = _table(config, "observation")
            tech_mask = bool(_table(observation, "animation").get("mask", True))
            if int(_table(observation, "frame_skip").get("skip", 0)) > 1:
                raise ValueError(f"{name}: observation frame_skip is on; not supported")
        else:  # tf/saving.py:52-58: NULL_OBSERVATION_CONFIG
            tech_mask = False
        network = _table(config, "network")
        if network.get("name") != "tx_like":
            raise ValueError(f"{name}: network {network.get('name')!r} is not tx_like")
        tx_like = _table(network, "tx_like")
        if tx_like.get("recurrent_layer", "lstm") != "lstm":
            raise ValueError(f"{name}: recurrent_layer {tx_like.get('recurrent_layer')!r} is not lstm")
        activation = str(tx_like.get("activation", "gelu"))
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"{name}: activation {activation!r} is not one of {SUPPORTED_ACTIVATIONS}")
        head = _table(config, "controller_head")
        if head.get("name", "independent") != "autoregressive":
            raise ValueError(f"{name}: controller head {head.get('name')!r} is not autoregressive")
        autoregressive = _table(head, "autoregressive")
        for spacing, native, what in ((axis_spacing, 160, "axis"), (shoulder_spacing, 140, "shoulder")):
            if spacing < 1 or native % spacing:
                raise ValueError(f"{name}: {what} spacing {spacing} must divide {native}")
        return cls(
            name=name,
            version=version,
            delay=int(_table(config, "policy").get("delay", 0)),
            max_names=int(config.get("max_names", 16)),
            hidden_size=int(tx_like.get("hidden_size", 128)),
            num_layers=int(tx_like.get("num_layers", 1)),
            ffw_multiplier=int(tx_like.get("ffw_multiplier", 4)),
            activation=activation,
            residual_size=int(autoregressive.get("residual_size", 128)),
            component_depth=int(autoregressive.get("component_depth", 0)),
            axis_spacing=axis_spacing,
            shoulder_spacing=shoulder_spacing,
            xy_scale=float(player.get("xy_scale", 0.05)),
            shield_scale=float(player.get("shield_scale", 0.01)),
            with_nana=with_nana,
            legacy_jumps_left=legacy_jumps,
            with_randall=with_randall,
            with_fod=with_fod,
            items_type=items_type,
            item_mlp_sizes=mlp_sizes,
            tech_mask=tech_mask,
            name_map={str(key): int(code) for key, code in (state.get("name_map") or {}).items()},
            rl_names=_rl_names(state),
        )

    @property
    def jumps_size(self) -> int:
        return 6 if self.legacy_jumps_left else 7

    @property
    def player_width(self) -> int:
        return 4 + ACTION_SIZE + CHARACTER_SIZE + 1 + self.jumps_size + 1 + 1

    @property
    def item_out(self) -> int:
        """One item slot's width after the embedding (0 when items are skipped)."""

        if self.items_type == "skip":
            return 0
        return self.item_mlp_sizes[-1] if self.items_type == "mlp" else ITEM_WIDTH

    @property
    def component_sizes(self) -> tuple[int, ...]:
        return (1,) * len(BUTTON_ORDER) + (self.axis_spacing + 1,) * 4 + (self.shoulder_spacing + 1,)

    @property
    def uniforms_per_row(self) -> int:
        """Uniform draws one row of one step consumes: one per button, one per class of every categorical."""

        return sum(self.component_sizes)

    @property
    def input_size(self) -> int:
        player = self.player_width + ((self.player_width + 1) if self.with_nana else 0)
        game = 2 * player + STAGE_SIZE + 2 * int(self.with_randall) + 2 * int(self.with_fod)
        game += MAX_ITEMS * self.item_out
        return game + sum(self.component_sizes) + self.max_names

    @property
    def default_name(self) -> str:
        """The tag upstream plays an RL release as (``build_delayed_agent``: the first it trained with)."""

        return self.rl_names[0] if self.rl_names else "Master Player"

    def name_code(self, name: str) -> int:
        """``eval_lib.get_name_code``: the ``name_map`` entry, no normalisation."""

        if name not in self.name_map:
            raise ValueError(f"Nametag {name!r} of {self.name} must be one of {sorted(self.name_map)}")
        return self.name_map[name]


def _rl_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    """``eval_lib.get_name_from_rl_state``: ``rl_config['agent']['name']`` or ``agent_config['name']``."""

    agent: Any = None
    if isinstance(state.get("rl_config"), Mapping):
        agent = state["rl_config"].get("agent")
    elif isinstance(state.get("agent_config"), Mapping):
        agent = state["agent_config"]
    if not isinstance(agent, Mapping) or agent.get("name") is None:
        return ()
    names = agent["name"]
    return (str(names),) if isinstance(names, str) else tuple(str(item) for item in names)


def expected_shapes(spec: ReleaseSpec) -> list[tuple[str, tuple[int, ...]]]:
    """``state['policy']``'s arrays by position, named, in TensorFlow's layouts (without the value
    readout)."""

    residual, hidden = spec.residual_size, spec.hidden_size
    shapes: list[tuple[str, tuple[int, ...]]] = []
    for component, size in zip(COMPONENTS, spec.component_sizes, strict=True):
        shapes += [
            (f"head/{component}/decoder/b", (residual,)),
            (f"head/{component}/decoder/w", (size, residual)),
        ]
        widths = [residual + size, *([residual] * spec.component_depth), size]
        for layer, (fan_in, fan_out) in enumerate(itertools.pairwise(widths)):
            shapes += [
                (f"head/{component}/encoder{layer}/b", (fan_out,)),
                (f"head/{component}/encoder{layer}/w", (fan_in, fan_out)),
            ]
    shapes += [("head/to_residual/b", (residual,)), ("head/to_residual/w", (hidden, residual))]
    if spec.items_type == "mlp":
        widths = [ITEM_WIDTH, *spec.item_mlp_sizes]
        for layer, (fan_in, fan_out) in enumerate(itertools.pairwise(widths)):
            shapes += [(f"items/mlp{layer}/b", (fan_out,)), (f"items/mlp{layer}/w", (fan_in, fan_out))]
    shapes += [("encoder/b", (hidden,)), ("encoder/w", (spec.input_size, hidden))]
    wide = hidden * spec.ffw_multiplier
    for layer in range(spec.num_layers):
        base = f"layer{layer}"
        shapes += [
            (f"{base}/lstm/w_h", (hidden, 4 * hidden)),
            (f"{base}/lstm/w_i", (hidden, 4 * hidden)),
            (f"{base}/lstm/b", (4 * hidden,)),
            (f"{base}/layernorm/bias", (hidden,)),
            (f"{base}/layernorm/scale", (hidden,)),
            (f"{base}/linear1/b", (wide,)),
            (f"{base}/linear1/w", (hidden, wide)),
            (f"{base}/linear2/b", (hidden,)),
            (f"{base}/linear2/w", (wide, hidden)),
        ]
    return shapes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the frame tree, packed
# ---------------------------------------------------------------------------


class PackedGame(NamedTuple):
    """The leaves the input reads, gathered per dtype: ``floats [B, 46]``, ``ints [B, 47]``, ``bools [B,
    29]``."""

    floats: torch.Tensor
    ints: torch.Tensor
    bools: torch.Tensor


_PLAYER_FLOATS: Final[tuple[str, ...]] = ("x", "y", "shield_strength")
_PLAYER_INTS: Final[tuple[str, ...]] = ("percent", "action", "character", "jumps_left")
_PLAYER_BOOLS: Final[tuple[str, ...]] = ("facing", "invulnerable", "on_ground")
_WHO: Final[tuple[tuple[str, bool], ...]] = (("p0", False), ("p0", True), ("p1", False), ("p1", True))
"""(player, is Nana) in packing order: p0, p0's Nana, p1, p1's Nana."""


def _who_offset(index: int, width: int) -> int:
    return index * width


P1_ACTION_COLUMN: Final[int] = _who_offset(2, len(_PLAYER_INTS)) + _PLAYER_INTS.index("action")
"""The column of ``ints`` holding the opponent's (``p1``'s) action: what the tech mask reads and rewrites."""
_GLOBAL_FLOATS: Final[int] = 4 * len(_PLAYER_FLOATS)
_GLOBAL_INTS: Final[int] = 4 * len(_PLAYER_INTS)
_NANA_EXISTS: Final[int] = 4 * len(_PLAYER_BOOLS)
_ITEM_EXISTS: Final[int] = _NANA_EXISTS + 2


def pack_game_shapes(batch: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """The ``(floats, ints, bools)`` shapes :func:`pack_game` produces for ``batch`` rows."""

    floats = 4 * len(_PLAYER_FLOATS) + 4 + 2 * MAX_ITEMS
    ints = 4 * len(_PLAYER_INTS) + 1 + 2 * MAX_ITEMS
    bools = 4 * len(_PLAYER_BOOLS) + 2 + MAX_ITEMS
    return (batch, floats), (batch, ints), (batch, bools)


def pack_game(frames: FrameTree, *, out: PackedGame | None = None) -> PackedGame:
    """Gather a ``[B]`` frame tree's embedded leaves into three tensors (one ``cat`` per dtype, on its device;
    into ``out``'s tensors when given, e.g. pinned staging)."""

    batch = int(frames["stage"].shape[0])
    floats: list[torch.Tensor] = []
    ints: list[torch.Tensor] = []
    bools: list[torch.Tensor] = []
    for player, is_nana in _WHO:
        node = frames[player]["nana"] if is_nana else frames[player]
        floats += [node[key].reshape(batch, 1) for key in _PLAYER_FLOATS]
        ints += [node[key].reshape(batch, 1) for key in _PLAYER_INTS]
        bools += [node[key].reshape(batch, 1) for key in _PLAYER_BOOLS]
    bools += [
        frames["p0"]["nana"]["exists"].reshape(batch, 1),
        frames["p1"]["nana"]["exists"].reshape(batch, 1),
    ]
    floats += [
        frames["randall"]["x"].reshape(batch, 1),
        frames["randall"]["y"].reshape(batch, 1),
        frames["fod_platforms"]["left"].reshape(batch, 1),
        frames["fod_platforms"]["right"].reshape(batch, 1),
    ]
    items = frames["items"]
    floats += [items["x"].reshape(batch, MAX_ITEMS), items["y"].reshape(batch, MAX_ITEMS)]
    ints += [
        frames["stage"].reshape(batch, 1),
        items["type"].reshape(batch, MAX_ITEMS),
        items["state"].reshape(batch, MAX_ITEMS),
    ]
    bools.append(items["exists"].reshape(batch, MAX_ITEMS))
    if out is not None:
        torch.cat([leaf.to(torch.float32) for leaf in floats], dim=1, out=out.floats)
        torch.cat([leaf.to(torch.long) for leaf in ints], dim=1, out=out.ints)
        torch.cat([leaf.to(torch.bool) for leaf in bools], dim=1, out=out.bools)
        return out
    return PackedGame(
        floats=torch.cat([leaf.to(torch.float32) for leaf in floats], dim=1),
        ints=torch.cat([leaf.to(torch.long) for leaf in ints], dim=1),
        bools=torch.cat([leaf.to(torch.bool) for leaf in bools], dim=1),
    )


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------


def _buffer_name(name: str) -> str:
    return "w__" + name.replace("/", "__")


def _one_hot(index: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """``tf.one_hot``: ``[..., C]`` float32, all zeros for an index outside ``[0, C)``
    (``OneHotPolicy.EMPTY``)."""

    return (index.unsqueeze(-1) == classes).to(torch.float32)


class SlippiAINetwork(nn.Module):
    """A release's input, core and head with its weights as buffers (inference only)."""

    scale_percent: torch.Tensor
    scale_xy: torch.Tensor
    scale_shield: torch.Tensor
    one: torch.Tensor
    minus_one: torch.Tensor

    def __init__(self, spec: ReleaseSpec) -> None:
        super().__init__()
        self.spec = spec
        self._shapes = expected_shapes(spec)
        for name, shape in self._shapes:
            self.register_buffer(_buffer_name(name), torch.zeros(shape, dtype=torch.float32))
        for size in {
            ACTION_SIZE,
            CHARACTER_SIZE,
            spec.jumps_size,
            STAGE_SIZE,
            ITEM_TYPE_SIZE,
            ITEM_STATE_SIZE,
            spec.axis_spacing + 1,
            spec.shoulder_spacing + 1,
            spec.max_names,
        }:
            self.register_buffer(f"classes_{size}", torch.arange(size, dtype=torch.long))
        for key, value in (
            ("scale_percent", 0.01),
            ("scale_xy", spec.xy_scale),
            ("scale_shield", spec.shield_scale),
            ("one", 1.0),
            ("minus_one", -1.0),
        ):
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))
        self.loaded = False

    # -- weights -------------------------------------------------------------

    def weight(self, name: str) -> torch.Tensor:
        """The array ``name`` of :func:`expected_shapes` (TensorFlow layout)."""

        value = getattr(self, _buffer_name(name))
        assert isinstance(value, torch.Tensor)
        return value

    def load_arrays(self, arrays: Sequence[np.ndarray]) -> None:
        """Assign ``state['policy']`` by position with a shape check per array; the trailing value readout
        (``(1,)``, ``(H, 1)``) may be present or not."""

        count = len(self._shapes)
        if len(arrays) not in (count, count + 2):
            raise ValueError(
                f"{self.spec.name}: expected {count} policy arrays (+2 for the value readout), "
                f"got {len(arrays)}"
            )
        if len(arrays) == count + 2:
            tail = [tuple(np.shape(array)) for array in arrays[count:]]
            if tail != [(1,), (self.spec.hidden_size, 1)]:
                raise ValueError(
                    f"{self.spec.name}: the two trailing arrays {tail} are not the value readout"
                )
        with torch.no_grad():
            for position, ((name, shape), array) in enumerate(zip(self._shapes, arrays[:count], strict=True)):
                if tuple(np.shape(array)) != shape:
                    raise ValueError(
                        f"{self.spec.name}: array #{position} {name} has shape {tuple(np.shape(array))}, "
                        f"expected {shape}"
                    )
                self.weight(name).copy_(torch.as_tensor(np.asarray(array, dtype=np.float32)))
        self.loaded = True

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """``snt.Linear``: ``matmul(x, w) + b``."""

        return torch.matmul(x, self.weight(f"{name}/w")) + self.weight(f"{name}/b")

    def _clamp(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.clamp(value, min=-FLOAT_CLAMP), max=FLOAT_CLAMP)

    def _classes(self, size: int) -> torch.Tensor:
        value = getattr(self, f"classes_{size}")
        assert isinstance(value, torch.Tensor)
        return value

    # -- the input -----------------------------------------------------------

    def _player(self, game: PackedGame, who: int, *, nana: bool) -> list[torch.Tensor]:
        spec = self.spec
        f = _who_offset(who, len(_PLAYER_FLOATS))
        i = _who_offset(who, len(_PLAYER_INTS))
        b = _who_offset(who, len(_PLAYER_BOOLS))
        floats, ints, bools = game
        percent = ints[:, i].to(torch.float32) * self.scale_percent
        facing = torch.where(bools[:, b], self.one, self.minus_one)
        parts = [
            self._clamp(percent).unsqueeze(-1),
            facing.unsqueeze(-1),
            self._clamp(floats[:, f] * self.scale_xy).unsqueeze(-1),
            self._clamp(floats[:, f + 1] * self.scale_xy).unsqueeze(-1),
            _one_hot(torch.clamp(ints[:, i + 1], 0, ACTION_SIZE - 1), self._classes(ACTION_SIZE)),
            _one_hot(ints[:, i + 2], self._classes(CHARACTER_SIZE)),
            bools[:, b + 1].to(torch.float32).unsqueeze(-1),
            _one_hot(ints[:, i + 3], self._classes(spec.jumps_size)),
            self._clamp(floats[:, f + 2] * self.scale_shield).unsqueeze(-1),
            bools[:, b + 2].to(torch.float32).unsqueeze(-1),
        ]
        if nana:
            exists = bools[:, _NANA_EXISTS + (0 if who == 1 else 1)]
            parts.append(exists.to(torch.float32).unsqueeze(-1))
        return parts

    def item_features(self, game: PackedGame) -> torch.Tensor:
        """``[B, 15, 254]``: every slot's exists, type (``EXTRA`` class past 236), state (past 11), x, y."""

        floats, ints, bools = game
        start = _GLOBAL_FLOATS + 4
        x = floats[:, start : start + MAX_ITEMS] * self.scale_xy
        y = floats[:, start + MAX_ITEMS : start + 2 * MAX_ITEMS] * self.scale_xy
        types = ints[:, _GLOBAL_INTS + 1 : _GLOBAL_INTS + 1 + MAX_ITEMS]
        states = ints[:, _GLOBAL_INTS + 1 + MAX_ITEMS : _GLOBAL_INTS + 1 + 2 * MAX_ITEMS]
        extra_type, extra_state = ITEM_TYPE_SIZE - 1, ITEM_STATE_SIZE - 1
        types = torch.where((types < 0) | (types >= extra_type), extra_type, types)
        states = torch.where((states < 0) | (states >= extra_state), extra_state, states)
        exists = bools[:, _ITEM_EXISTS : _ITEM_EXISTS + MAX_ITEMS]
        return torch.cat(
            [
                exists.to(torch.float32).unsqueeze(-1),
                _one_hot(types, self._classes(ITEM_TYPE_SIZE)),
                _one_hot(states, self._classes(ITEM_STATE_SIZE)),
                self._clamp(x).unsqueeze(-1),
                self._clamp(y).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _component_embedding(self, component: int, value: torch.Tensor) -> torch.Tensor:
        size = self.spec.component_sizes[component]
        if size == 1:
            return value.to(torch.float32).unsqueeze(-1)
        return _one_hot(value, self._classes(size))

    def embed(self, game: PackedGame, prev: torch.Tensor, names: torch.Tensor) -> torch.Tensor:
        """``StateAction`` embedded: the game (p0, p1, stage, Randall, FoD, items), the previous controller
        (``[B, 13]`` class indices) and the name codes (``[B]``) -- ``[B, input_size]``."""

        spec = self.spec
        floats, ints, _ = game
        parts: list[torch.Tensor] = []
        for player in (0, 2):
            parts += self._player(game, player, nana=False)
            if spec.with_nana:
                parts += self._player(game, player + 1, nana=True)
        parts.append(_one_hot(ints[:, _GLOBAL_INTS], self._classes(STAGE_SIZE)))
        if spec.with_randall:
            parts += [
                self._clamp(floats[:, _GLOBAL_FLOATS] * self.scale_xy).unsqueeze(-1),
                self._clamp(floats[:, _GLOBAL_FLOATS + 1] * self.scale_xy).unsqueeze(-1),
            ]
        if spec.with_fod:
            parts += [
                self._clamp(floats[:, _GLOBAL_FLOATS + 2] * self.scale_xy).unsqueeze(-1),
                self._clamp(floats[:, _GLOBAL_FLOATS + 3] * self.scale_xy).unsqueeze(-1),
            ]
        if spec.items_type != "skip":
            items = self.item_features(game)
            if spec.items_type == "mlp":
                for layer in range(len(spec.item_mlp_sizes)):
                    items = torch.relu(self._linear(items, f"items/mlp{layer}"))
            parts.append(items.reshape(items.shape[0], -1))
        for component in range(len(COMPONENTS)):
            parts.append(self._component_embedding(component, prev[:, component]))
        parts.append(_one_hot(names, self._classes(spec.max_names)))
        return torch.cat(parts, dim=-1)

    # -- the core ------------------------------------------------------------

    def core(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of ``tx_like`` from ``[B, input_size]``: the output ``[B, H]`` and the new ``[L, B, H]``
        states."""

        activation = functional.gelu if self.spec.activation == "gelu" else torch.relu
        x = self._linear(x, "encoder")
        hiddens: list[torch.Tensor] = []
        cells: list[torch.Tensor] = []
        for layer in range(self.spec.num_layers):
            base = f"layer{layer}"
            gates = (
                torch.matmul(x, self.weight(f"{base}/lstm/w_i"))
                + torch.matmul(hidden[layer], self.weight(f"{base}/lstm/w_h"))
                + self.weight(f"{base}/lstm/b")
            )
            i, f, g, o = torch.chunk(gates, 4, dim=-1)
            new_cell = torch.sigmoid(f) * cell[layer] + torch.sigmoid(i) * torch.tanh(g)
            new_hidden = torch.sigmoid(o) * torch.tanh(new_cell)
            x = x + new_hidden
            centred = x - torch.mean(x, dim=-1, keepdim=True)
            normed = centred / torch.sqrt(torch.mean(torch.square(centred), dim=-1, keepdim=True))
            y = normed * self.weight(f"{base}/layernorm/scale") + self.weight(f"{base}/layernorm/bias")
            y = self._linear(activation(self._linear(y, f"{base}/linear1")), f"{base}/linear2")
            x = x + y
            hiddens.append(new_hidden)
            cells.append(new_cell)
        return x, torch.stack(hiddens), torch.stack(cells)

    # -- the head ------------------------------------------------------------

    def _component_logits(self, component: int, residual: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        name = COMPONENTS[component]
        h = torch.cat([residual, self._component_embedding(component, prev)], dim=-1)
        depth = self.spec.component_depth
        for layer in range(depth + 1):
            h = self._linear(h, f"head/{name}/encoder{layer}")
            if layer < depth:
                h = torch.relu(h)
        return h

    def _head(
        self,
        out: torch.Tensor,
        prev: torch.Tensor,
        *,
        forced: torch.Tensor | None,
        uniforms: torch.Tensor | None,
        temperature: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        residual = self._linear(out, "head/to_residual")
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        offset = 0
        for component, size in enumerate(self.spec.component_sizes):
            component_logits = self._component_logits(component, residual, prev[:, component])
            logits.append(component_logits)
            if forced is not None:
                value = forced[:, component]
            else:
                assert uniforms is not None
                draws = uniforms[:, offset : offset + size]
                if size == 1:
                    value = sample_bernoulli(component_logits[:, 0], draws[:, 0], temperature).to(torch.long)
                else:
                    value = sample_categorical(component_logits, draws, temperature)
            offset += size
            values.append(value)
            decoded = self._linear(
                self._component_embedding(component, value), f"head/{COMPONENTS[component]}/decoder"
            )
            residual = residual + decoded
        return torch.stack(values, dim=1), logits

    def head_sample(
        self, out: torch.Tensor, prev: torch.Tensor, uniforms: torch.Tensor, temperature: float
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample every component in order from pre-drawn ``uniforms`` (``[B, uniforms_per_row]``): the ``[B,
        13]`` class indices and the per-component logits."""

        return self._head(out, prev, forced=None, uniforms=uniforms, temperature=temperature)

    def head_forced(self, out: torch.Tensor, prev: torch.Tensor, forced: torch.Tensor) -> list[torch.Tensor]:
        """Every component's logits with the residual teacher-forced on ``forced``
        (``AutoRegressive.distance``)."""

        return self._head(out, prev, forced=forced, uniforms=None, temperature=1.0)[1]


def sample_bernoulli(logit: torch.Tensor, uniform: torch.Tensor, temperature: float) -> torch.Tensor:
    """``tfp.distributions.Bernoulli(logits / T).sample()``: true with probability ``sigmoid(logit / T)``."""

    return uniform < torch.sigmoid(logit / temperature)


def sample_categorical(logits: torch.Tensor, uniforms: torch.Tensor, temperature: float) -> torch.Tensor:
    """``Categorical(logits / T).sample()`` by Gumbel-max over one uniform per class."""

    gumbel = -torch.log(-torch.log(uniforms))
    return torch.argmax(logits / temperature + gumbel, dim=-1)


# ---------------------------------------------------------------------------
# the agent glue
# ---------------------------------------------------------------------------


def tech_mask_step(
    action: torch.Tensor, prev: torch.Tensor, count: torch.Tensor, reset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One frame of ``observations.AnimationFilter.update`` per row: ``(masked action, new prev, new count)``.

    A reset row forgets (``prev = 0``, ``count = 0``) before the update; the count restarts whenever the
    action changes, and neutral / forward / back tech read as neutral tech while it is below 7."""

    zero = torch.zeros_like(prev)
    prev = torch.where(reset, zero, prev)
    count = torch.where(reset, zero, count)
    count = torch.where(action == prev, count + 1, zero)
    is_tech = (action == TECH_ACTIONS[0]) | (action == TECH_ACTIONS[1]) | (action == TECH_ACTIONS[2])
    masked = torch.where(
        is_tech & (count < TECH_MASK_WINDOW), torch.full_like(action, TECH_ACTIONS[0]), action
    )
    return masked, action, count


def _grid(spacing: int) -> torch.Tensor:
    """``DiscreteEmbedding.decode``: ``(k / n).astype(float32)`` -- float64 division, then rounded once."""

    return torch.from_numpy((np.arange(spacing + 1) / spacing).astype(np.float32))


def decode_indices(indices: torch.Tensor, spec: ReleaseSpec) -> ControllerState:
    """``[B, 13]`` class indices -> the ``ControllerState`` upstream's ``send_controller`` would press."""

    indices = indices.to("cpu", torch.long)
    axis, shoulder = _grid(spec.axis_spacing), _grid(spec.shoulder_spacing)
    buttons = {name: indices[:, column].to(torch.bool) for column, name in enumerate(BUTTON_ORDER)}
    first = len(BUTTON_ORDER)
    return ControllerState(
        main_stick=StickState(x=axis[indices[:, first]], y=axis[indices[:, first + 1]]),
        c_stick=StickState(x=axis[indices[:, first + 2]], y=axis[indices[:, first + 3]]),
        shoulder=shoulder[indices[:, first + 4]],
        buttons=ButtonState(**buttons),
    )


def encode_controller(control: ControllerState, spec: ReleaseSpec) -> torch.Tensor:
    """The inverse of :func:`decode_indices` for values on the grid (tests and the parity harness)."""

    columns = [getattr(control.buttons, name).to(torch.long) for name in BUTTON_ORDER]
    for value in (control.main_stick.x, control.main_stick.y, control.c_stick.x, control.c_stick.y):
        columns.append(torch.round(value.to(torch.float64) * spec.axis_spacing).to(torch.long))
    columns.append(torch.round(control.shoulder.to(torch.float64) * spec.shoulder_spacing).to(torch.long))
    return torch.stack(columns, dim=1)


class SlippiAIAgent:
    """One release batched over ``num_envs`` environments: recurrent state, previous sample, tech-mask state,
    name codes and the delay queue (see the module docstring).

    The step reads its inputs from static buffers (the packed frame, the reset mask, the pre-drawn uniforms)
    and updates its state in place, so ``graph = True`` can capture it once as a ``torch.cuda.CUDAGraph`` and
    replay it every frame (one launch instead of a few hundred; LEAGUE-PLAN.md §3.2 "Speed"). Capture happens
    at the first step, on that step's inputs, with the state restored afterwards: a captured agent plays
    exactly what the eager one plays."""

    def __init__(
        self,
        network: SlippiAINetwork,
        num_envs: int,
        *,
        names: Sequence[str],
        device: torch.device | str,
        seed: int = 0,
        temperature: float = 1.0,
        tech_mask: bool | None = None,
        graph: bool = False,
        warmup: int = 3,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if len(names) != num_envs:
            raise ValueError(f"names must give one tag per environment ({num_envs}), got {len(names)}")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not network.loaded:
            raise ValueError(f"{network.spec.name}: load the release's arrays before building an agent")
        self.device = torch.device(device)
        if graph and self.device.type != "cuda":
            raise ValueError(f"graph = true replays a CUDA graph: it needs a CUDA device, got {self.device}")
        spec = network.spec
        codes = [spec.name_code(name) for name in names]
        if spec.rl_names:
            untrained = sorted(set(names) - set(spec.rl_names))
            if untrained:  # eval_lib.build_delayed_agent refuses a batch of names it was not trained with
                raise ValueError(f"{spec.name} was trained with the name(s) {spec.rl_names}, got {untrained}")
        self.network = network.to(self.device)
        self.spec = spec
        self.num_envs = num_envs
        self.temperature = float(temperature)
        self.tech_mask = spec.tech_mask if tech_mask is None else bool(tech_mask)
        self.names = torch.tensor(codes, dtype=torch.long, device=self.device)
        self.name_tags = tuple(names)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.graph = bool(graph)
        self.warmup = int(warmup)
        self.frames_seen = 0
        self.replays = 0
        self.captures = 0
        self._graph: Any = None
        batch = num_envs
        float_shape, int_shape, bool_shape = pack_game_shapes(batch)
        self._floats = torch.zeros(float_shape, dtype=torch.float32, device=self.device)
        self._ints = torch.zeros(int_shape, dtype=torch.long, device=self.device)
        self._bools = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)
        # Host frames are packed into pinned staging and uploaded in three non-blocking copies (CUDA only).
        pinned = self.device.type == "cuda"
        self._host = PackedGame(
            torch.zeros(float_shape, dtype=torch.float32, pin_memory=pinned),
            torch.zeros(int_shape, dtype=torch.long, pin_memory=pinned),
            torch.zeros(bool_shape, dtype=torch.bool, pin_memory=pinned),
        )
        self._reset = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self._uniforms = torch.full((batch, spec.uniforms_per_row), 0.5, device=self.device)
        self._sample = torch.zeros(batch, len(COMPONENTS), dtype=torch.long, device=self.device)
        self.hidden = torch.zeros(spec.num_layers, batch, spec.hidden_size, device=self.device)
        self.cell = torch.zeros_like(self.hidden)
        self.prev = torch.zeros(batch, len(COMPONENTS), dtype=torch.long, device=self.device)
        self.tech_prev = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.tech_count = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.reset_all()

    def _state(self) -> tuple[torch.Tensor, ...]:
        return (self.hidden, self.cell, self.prev, self.tech_prev, self.tech_count)

    def reset_all(self) -> None:
        """A fresh agent: zero recurrent state and previous controller, a forgotten tech mask, the queue
        primed with ``delay`` dummy outputs (``DelayedAgent.__init__``). In place, so a captured graph stays
        bound."""

        with torch.no_grad():
            for tensor in self._state():
                tensor.zero_()
        batch = self.num_envs
        self.last_sample = torch.zeros(batch, len(COMPONENTS), dtype=torch.long)
        dummy = torch.zeros(batch, len(COMPONENTS), dtype=torch.long)
        self._queue: deque[torch.Tensor] = deque(dummy for _ in range(self.spec.delay))

    def _load(self, frames: FrameTree, needs_reset: torch.Tensor) -> None:
        """The frame into the static buffers: a host frame tree through the pinned staging (three uploads), a
        device one directly."""

        if frames["stage"].device.type == "cpu" and self.device.type != "cpu":
            game = pack_game(frames, out=self._host)
            self._floats.copy_(game.floats, non_blocking=True)
            self._ints.copy_(game.ints, non_blocking=True)
            self._bools.copy_(game.bools, non_blocking=True)
        else:
            game = pack_game(frames)
            self._floats.copy_(game.floats)
            self._ints.copy_(game.ints)
            self._bools.copy_(game.bools)
        self._reset.copy_(needs_reset.to(torch.bool), non_blocking=True)

    def _prepare(self) -> tuple[PackedGame, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The tech mask (state updated in place) and the reset state: ``(game, x, hidden, cell)``."""

        reset = self._reset
        ints = self._ints
        if self.tech_mask:
            masked, prev, count = tech_mask_step(
                ints[:, P1_ACTION_COLUMN], self.tech_prev, self.tech_count, reset
            )
            self.tech_prev.copy_(prev)
            self.tech_count.copy_(count)
            ints = ints.clone()
            ints[:, P1_ACTION_COLUMN] = masked
        game = PackedGame(self._floats, ints, self._bools)
        keep = (~reset).view(1, -1, 1)
        hidden = torch.where(keep, self.hidden, torch.zeros_like(self.hidden))
        cell = torch.where(keep, self.cell, torch.zeros_like(self.cell))
        x = self.network.embed(game, self.prev, self.names)
        return game, x, hidden, cell

    def _body(self) -> list[torch.Tensor]:
        """One sampled step from the static buffers into the state and ``_sample`` (what a graph captures);
        the logits are returned for the eager callers."""

        _, x, hidden, cell = self._prepare()
        out, new_hidden, new_cell = self.network.core(x, hidden, cell)
        sample, logits = self.network.head_sample(out, self.prev, self._uniforms, self.temperature)
        self.hidden.copy_(new_hidden)
        self.cell.copy_(new_cell)
        self.prev.copy_(sample)
        self._sample.copy_(sample)
        return logits

    def _capture(self) -> None:
        saved = [tensor.clone() for tensor in self._state()]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._body()
        for tensor, value in zip(self._state(), saved, strict=True):
            tensor.copy_(value)
        self._graph = graph
        self.captures += 1

    def _queue_through(self, fresh: torch.Tensor) -> torch.Tensor:
        self.last_sample = fresh
        self._queue.append(fresh)
        self.frames_seen += self.num_envs
        return self._queue.popleft()

    def step(self, frames: FrameTree, needs_reset: torch.Tensor) -> ControllerState:
        """One frame for every environment: the controller upstream's ``DelayedAgent.step`` would send now."""

        with torch.no_grad():
            self._load(frames, needs_reset)
            self._uniforms.uniform_(generator=self.generator)
            if self.graph:
                if self._graph is None:
                    self._capture()
                self._graph.replay()
                self.replays += 1
            else:
                self._body()
            fresh = self._sample.to("cpu", copy=True)  # the queue must not alias the static buffer
        return decode_indices(self._queue_through(fresh), self.spec)

    def step_with_logits(
        self, frames: FrameTree, needs_reset: torch.Tensor
    ) -> tuple[ControllerState, torch.Tensor, list[torch.Tensor]]:
        """:meth:`step` (eager) that also returns the fresh sample ``[B, 13]`` and its per-component
        logits."""

        with torch.no_grad():
            self._load(frames, needs_reset)
            self._uniforms.uniform_(generator=self.generator)
            logits = [component.clone() for component in self._body()]
            fresh = self._sample.to("cpu", copy=True)
        return decode_indices(self._queue_through(fresh), self.spec), fresh, logits

    def step_forced(
        self, frames: FrameTree, needs_reset: torch.Tensor, forced: torch.Tensor
    ) -> list[torch.Tensor]:
        """One frame teacher-forced on ``forced`` (``[B, 13]``, e.g. the TensorFlow agent's own fresh sample):
        every component's logits, with the state advanced exactly as :meth:`step` would with that sample
        (eager: the parity harness)."""

        with torch.no_grad():
            self._load(frames, needs_reset)
            _, x, hidden, cell = self._prepare()
            out, new_hidden, new_cell = self.network.core(x, hidden, cell)
            sample = forced.to(device=self.device, dtype=torch.long)
            logits = self.network.head_forced(out, self.prev, sample)
            self.hidden.copy_(new_hidden)
            self.cell.copy_(new_cell)
            self.prev.copy_(sample)
            self._queue_through(sample.to("cpu", copy=True))
        return logits


class SlippiAITorchOpponent:
    """A released slippi-ai agent on ``port`` through the PyTorch port: a ``slippi:<release>*<n>`` mix arm and
    the ``slippi:<release>`` evaluation opponent (17 Sep 2026).

    ``act_state`` reads the frame tree the rollout worker hands an opponent -- its own perspective, ``p0`` the
    agent, ``p1`` our Fox, exactly upstream's ``Parser(ports=(mine, theirs))`` -- and returns a
    ``ControllerState`` on Melee's grid (the worker sends it without the codec). ``characters`` records which
    character each Dolphin was booted with (the arm's table), for the per-character metrics; ``refresh`` is a
    no-op."""

    def __init__(
        self,
        network: SlippiAINetwork,
        num_envs: int,
        *,
        names: Sequence[str],
        device: torch.device | str,
        seed: int = 1,
        temperature: float = 1.0,
        graph: bool = False,
        port: int = 1,
        characters: Sequence[int] = (),
        release: str = "",
    ) -> None:
        self.agent = SlippiAIAgent(
            network, num_envs, names=names, device=device, seed=seed, temperature=temperature, graph=graph
        )
        self._port = port
        self.characters = tuple(int(character) for character in characters)
        self.release = release or network.spec.name
        self.calls = 0
        self.seconds = 0.0

    host_frames = True
    """The rollout worker hands this opponent the host frame tree (the agent packs and uploads it itself)."""

    @property
    def port(self) -> int:
        return self._port

    @property
    def controls_port(self) -> bool:
        return True

    @property
    def generator(self) -> torch.Generator:
        """The sampling generator (its state goes into RL checkpoints, like a ``PolicyOpponent``'s)."""

        return self.agent.generator

    def act(self, frames: Any, needs_reset: torch.Tensor) -> Any:
        raise NotImplementedError(
            "the slippi-ai port emits a ControllerState; the worker must call act_state"
        )

    def act_state(self, frames: FrameTree, needs_reset: torch.Tensor) -> ControllerState:
        started = time.perf_counter()
        control = self.agent.step(frames, needs_reset)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return control

    def refresh(self, student: Any) -> None:
        return None

    def reset_state(self) -> None:
        self.agent.reset_all()

    def stats(self) -> dict[str, Any]:
        agent = self.agent
        return {
            "release": self.release,
            "delay": agent.spec.delay,
            "names": list(agent.name_tags),
            "characters": list(self.characters),
            "tech_mask": agent.tech_mask,
            "graph": agent.graph,
            "frames_seen": agent.frames_seen,
            "calls": self.calls,
            "seconds_per_call": self.seconds / self.calls if self.calls else 0.0,
        }


def load_network(path: str | Path, *, name: str, sha256: str | None = None) -> SlippiAINetwork:
    """A release file as a loaded :class:`SlippiAINetwork` on the CPU (``sha256`` checked when given)."""

    source = Path(path)
    if sha256 is not None:
        digest = sha256_file(source)
        if digest != sha256:
            raise ReleaseLoadError(f"{source} has sha256 {digest}, expected {sha256} ({name})")
    state = load_release_state(source)
    spec = ReleaseSpec.from_state(state, name=name)
    network = SlippiAINetwork(spec)
    network.load_arrays(release_arrays(state))
    return network


__all__ = [
    "ACTION_SIZE",
    "CHARACTER_SIZE",
    "COMPONENTS",
    "ITEM_STATE_SIZE",
    "ITEM_TYPE_SIZE",
    "ITEM_WIDTH",
    "P1_ACTION_COLUMN",
    "STAGE_SIZE",
    "TECH_ACTIONS",
    "TECH_MASK_WINDOW",
    "PackedGame",
    "ReleaseLoadError",
    "ReleaseSpec",
    "ReleaseStub",
    "SlippiAIAgent",
    "SlippiAINetwork",
    "SlippiAITorchOpponent",
    "decode_indices",
    "encode_controller",
    "enum_value",
    "expected_shapes",
    "load_network",
    "load_release_state",
    "pack_game",
    "pack_game_shapes",
    "release_arrays",
    "sample_bernoulli",
    "sample_categorical",
    "sha256_file",
    "tech_mask_step",
]
