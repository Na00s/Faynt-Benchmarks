"""The PyTorch port of slippi-ai's released policies (17 Sep 2026, LEAGUE-PLAN.md §3.2).

Everything here runs without TensorFlow or slippi-ai: a synthetic release pickle stands in for the real files
(which the Modal parity harness checks against their own TensorFlow agent, §3.4). The references are written
against the upstream sources at the pin -- ``slippi_ai/tf/embed.py`` (the input), ``tf/networks.py:171-226,
305, 378-424`` (the core), ``tf/controller_heads.py:107-202`` (the head), ``eval_lib.py:90-198`` and
``observations.py:160-200`` (the agent glue) -- line by line in NumPy, so a weight read in the wrong order, a
missing clamp or a swapped gate shows up as a number, not as a shape."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from controller_codec import ControllerState
from melee_rl import slippi_ai_torch as port
from melee_rl.frames import FrameTree, neutral_frame, random_valid_frames
from tests.rl._slippi_release import COMPONENT_SIZES, v4_input_size, v5_input_size, write_release

N_TECH, F_TECH, B_TECH = 0xC7, 0xC8, 0xC9
STANDING = 14


# ---------------------------------------------------------------------------
# a synthetic release
# ---------------------------------------------------------------------------


def _write_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, version: int = 5, **changes: Any
) -> tuple[Path, dict[str, Any], list[np.ndarray]]:
    path = tmp_path / f"release-v{version}.pkl"
    config, arrays = write_release(path, version=version, **changes)
    return path, config, arrays


def _v5_input_size(max_names: int, item_out: int) -> int:
    return v5_input_size(max_names, item_out)


def _v4_input_size(max_names: int) -> int:
    return v4_input_size(max_names)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_a_release_loads_without_slippi_ai_tensorflow_or_arbitrary_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, arrays = _write_release(tmp_path, monkeypatch)
    state = port.load_release_state(path)
    assert state["step"] == 12 and len(state["state"]["policy"]) == len(arrays)
    assert all(np.array_equal(a, b) for a, b in zip(state["state"]["policy"], arrays, strict=True))
    assert port.enum_value(state["config"]["embed"]["items"]["type"]) == "mlp"
    assert port.enum_value("mlp") == "mlp"
    # A pickle that would call a function on load is inert: every non-numpy global is a stub.
    marker = tmp_path / "pwned"
    evil = tmp_path / "evil.pkl"
    evil.write_bytes(b"cos\nsystem\n(S'touch " + str(marker).encode() + b"'\ntR.")
    with pytest.raises(port.ReleaseLoadError, match="not a slippi-ai release"):
        port.load_release_state(evil)
    assert not marker.exists()


def test_a_jax_release_is_refused_by_name_and_pointed_at_the_recorder(tmp_path: Path) -> None:
    """The shared delay-0 Fox (22 Sep 2026) is ``platform: 'jax'`` with flax's *nested* parameter dict,
    not the positional Sonnet list this port reads.  The refusal has to say so rather than "must hold
    numpy arrays": the file is a valid release, just not one the port can play -- the recorder's
    ``[slippi_ai]`` agent runs it through upstream's own JAX code."""

    import pickle

    state = {
        "config": {"policy": {"delay": 0}, "platform": "jax", "version": 1},
        "state": {"policy": {"network": {"kernel": np.zeros((2, 2), np.float32)}}},
        "name_map": {"Master Player": 1},
        "step": 7,
    }
    path = tmp_path / "fox_d0_tx_like_3x512"
    path.write_bytes(pickle.dumps(state))
    with pytest.raises(port.ReleaseLoadError, match="JAX") as caught:
        port.load_release_state(path)
    assert "slippi_ai" in str(caught.value) and "TensorFlow" in str(caught.value)


def test_release_spec_reads_and_upgrades_configs_like_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _ = _write_release(tmp_path, monkeypatch)
    spec = port.ReleaseSpec.from_state(port.load_release_state(path), name="tiny")
    assert (spec.version, spec.delay, spec.max_names, spec.hidden_size, spec.num_layers) == (5, 3, 3, 8, 2)
    assert (spec.ffw_multiplier, spec.activation, spec.residual_size, spec.component_depth) == (
        2,
        "gelu",
        6,
        2,
    )
    assert (spec.axis_spacing, spec.shoulder_spacing, spec.xy_scale, spec.shield_scale) == (
        32,
        10,
        0.05,
        0.01,
    )
    assert spec.with_nana and not spec.legacy_jumps_left and spec.with_randall and spec.with_fod
    assert spec.items_type == "mlp" and spec.item_mlp_sizes == (5, 4) and spec.tech_mask
    assert spec.component_sizes == COMPONENT_SIZES and spec.input_size == _v5_input_size(3, 4)
    assert spec.rl_names == ("Cody", "Hax") and spec.name_code("Hax") == 2
    assert spec.default_name == "Cody"
    with pytest.raises(ValueError, match="Nametag"):
        spec.name_code("Mango")
    # A version-4 file is upgraded as tf/saving.py:17-78 does: no Randall / FoD / items / Nana, legacy jumps.
    path4, _, _ = _write_release(tmp_path, monkeypatch, version=4, max_names=16)
    old = port.ReleaseSpec.from_state(port.load_release_state(path4), name="old")
    assert not (old.with_nana or old.with_randall or old.with_fod) and old.legacy_jumps_left
    assert old.items_type == "skip" and old.tech_mask and old.input_size == _v4_input_size(16)
    # The two real layouts: 2 613 inputs for the 768 family (128 names, items MLP [128, 32]), 1 121 for the v4
    # specialists (16 names).
    assert _v5_input_size(128, 32) == 2613 and _v4_input_size(16) == 1121
    # A version-3 file has no observation config: upstream upgrades it to the null one (no tech mask).
    raw = port.load_release_state(path4)
    raw["config"]["version"] = 3
    del raw["config"]["observation"]
    assert not port.ReleaseSpec.from_state(raw, name="v3").tech_mask
    # Missing keys take upstream's dataclass defaults (a v5 file that never wrote frame_skip).
    raw = port.load_release_state(path)
    del raw["config"]["embed"]["player"]["legacy_jumps_left"]
    assert not port.ReleaseSpec.from_state(raw, name="defaults").legacy_jumps_left
    # What the port cannot express is refused rather than approximated.
    for key, value, message in (
        (("embed", "player", "with_speeds"), True, "speeds"),
        (("embed", "player", "with_controller"), True, "controller"),
        (("network", "name"), "lstm", "tx_like"),
        (("network", "tx_like", "recurrent_layer"), "gru", "lstm"),
        (("controller_head", "name"), "independent", "autoregressive"),
        (("observation", "frame_skip"), {"skip": 2}, "frame_skip"),
    ):
        raw = port.load_release_state(path)
        node = raw["config"]
        for part in key[:-1]:
            node = node[part]
        node[key[-1]] = value
        with pytest.raises(ValueError, match=message):
            port.ReleaseSpec.from_state(raw, name="refused")


def test_weights_load_by_position_with_shape_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _, arrays = _write_release(tmp_path, monkeypatch)
    state = port.load_release_state(path)
    spec = port.ReleaseSpec.from_state(state, name="tiny")
    network = port.SlippiAINetwork(spec)
    network.load_arrays(port.release_arrays(state))
    shapes = port.expected_shapes(spec)
    assert [shape for _, shape in shapes] == [
        array.shape for array in arrays[:-2]
    ]  # the value readout is extra
    assert shapes[0][0].startswith("head/A/decoder") and shapes[-1][0].endswith("linear2/w")
    encoder = [name for name, _ in shapes].index("encoder/w")
    torch.testing.assert_close(network.weight("encoder/w"), torch.from_numpy(arrays[encoder]))
    # Without the trailing value readout the tuple loads too; a wrong shape or count names the position.
    network.load_arrays(tuple(arrays[:-2]))
    broken = list(arrays)
    broken[1] = np.zeros((2, 6), np.float32)
    with pytest.raises(ValueError, match=r"#1 head/A/decoder/w"):
        network.load_arrays(tuple(broken))
    with pytest.raises(ValueError, match="arrays"):
        network.load_arrays(tuple(arrays[:-5]))


# ---------------------------------------------------------------------------
# the input
# ---------------------------------------------------------------------------


def _spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes: Any) -> tuple[port.ReleaseSpec, Any]:
    path, _, _ = _write_release(tmp_path, monkeypatch, **changes)
    state = port.load_release_state(path)
    return port.ReleaseSpec.from_state(state, name="tiny"), state


def _player_width(spec: port.ReleaseSpec) -> int:
    return 4 + 399 + 33 + 1 + (6 if spec.legacy_jumps_left else 7) + 1 + 1


def test_the_input_matches_hand_computed_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, state = _spec(tmp_path, monkeypatch)
    network = port.SlippiAINetwork(spec)
    network.load_arrays(port.release_arrays(state))
    frames = neutral_frame((2,))
    p0, nana, p1 = frames["p0"], frames["p0"]["nana"], frames["p1"]
    p0["percent"][:] = torch.tensor([250, 12])
    p0["facing"][:] = torch.tensor([False, True])
    p0["x"][:] = torch.tensor([300.0, -10.0])
    p0["y"][:] = torch.tensor([-40.0, 20.0])
    p0["action"][:] = torch.tensor([500, 20])
    p0["character"][:] = torch.tensor([22, 1])
    p0["invulnerable"][:] = torch.tensor([True, False])
    p0["jumps_left"][:] = torch.tensor([6, 2])
    p0["shield_strength"][:] = torch.tensor([60.0, 0.0])
    p0["on_ground"][:] = torch.tensor([True, False])
    nana["exists"][:] = torch.tensor([False, True])
    nana["percent"][:] = torch.tensor([0, 30])
    nana["character"][:] = torch.tensor([0, 10])
    p1["action"][:] = torch.tensor([STANDING, 3])
    frames["stage"][:] = torch.tensor([25, 8])
    frames["randall"]["x"][:] = torch.tensor([10.0, 0.0])
    frames["fod_platforms"]["left"][:] = torch.tensor([0.0, -30.0])
    frames["items"]["exists"][1, 3] = True
    frames["items"]["type"][1, 3] = 300
    frames["items"]["state"][1, 3] = 20
    frames["items"]["x"][1, 3] = 400.0
    prev = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 1, 16, 32, 0, 5, 10], [0] * 13])
    names = torch.tensor([1, 7])  # 7 >= max_names 3: an all-zero one-hot (OneHotPolicy.EMPTY)

    x = network.embed(port.pack_game(frames), prev, names)
    assert x.shape == (2, spec.input_size) and x.dtype == torch.float32
    player = _player_width(spec)
    # p0, row 0: percent x 0.01, facing -1, x clamped at +10, y -2, action clamped to 398, character 22, ...
    row = x[0]
    assert (
        row[0] == pytest.approx(2.5) and row[1] == -1.0 and row[2] == 10.0 and row[3] == pytest.approx(-2.0)
    )
    assert row[4 + 398] == 1.0 and row[4:403].sum() == 1.0
    assert row[403 + 22] == 1.0 and row[403:436].sum() == 1.0
    assert row[436] == 1.0  # invulnerable
    assert row[437 + 6] == 1.0 and row[437:444].sum() == 1.0  # jumps one-hot of 7 (the v5 layout)
    assert row[444] == pytest.approx(0.6) and row[445] == 1.0  # shield x 0.01, on_ground
    nana0 = player
    assert x[1, nana0] == pytest.approx(0.3) and x[1, nana0 + 403 + 10] == 1.0 and x[1, nana0 + player] == 1.0
    assert x[0, nana0 + player] == 0.0  # Nana's exists flag is the block's last input
    assert x[1, 4 + 20] == 1.0 and x[1, 1] == 1.0 and x[1, 2] == pytest.approx(-0.5)
    p1_start = player + player + 1
    assert x[0, p1_start + 4 + STANDING] == 1.0 and x[1, p1_start + 4 + 3] == 1.0
    stage = 2 * p1_start
    assert x[0, stage + 25] == 1.0 and x[1, stage + 8] == 1.0 and x[:, stage : stage + 64].sum() == 2.0
    randall = stage + 64
    assert x[0, randall] == pytest.approx(0.5) and x[0, randall + 1] == 0.0
    assert x[1, randall + 2] == pytest.approx(-1.5)  # FoD left x 0.05
    items = randall + 4
    features = network.item_features(port.pack_game(frames))
    assert features.shape == (2, 15, 254)
    slot = features[1, 3]
    assert slot[0] == 1.0 and slot[1 + 237] == 1.0 and slot[1:239].sum() == 1.0  # type 300 -> the EXTRA class
    assert slot[239 + 12] == 1.0 and slot[239:252].sum() == 1.0  # state 20 -> the EXTRA class
    assert slot[252] == 10.0 and slot[253] == 0.0  # x 400 x 0.05 clamped at 10
    empty = features[0, 0]
    assert empty[0] == 0.0 and empty[1] == 1.0 and empty[239] == 1.0  # an empty slot is type 0, state 0
    controller = items + 15 * 4
    assert x[0, controller : controller + 8].tolist() == [1.0, 0, 0, 0, 0, 0, 0, 1.0]
    sticks = controller + 8
    assert x[0, sticks + 16] == 1.0 and x[0, sticks + 33 + 32] == 1.0 and x[0, sticks + 99 + 5] == 1.0
    assert x[1, sticks] == 1.0 and x[1, sticks + 33] == 1.0  # the dummy controller is class 0, not the centre
    shoulder = sticks + 132
    assert x[0, shoulder + 10] == 1.0 and x[0, shoulder : shoulder + 11].sum() == 1.0
    name = shoulder + 11
    assert x[0, name + 1] == 1.0 and x[1, name:].sum() == 0.0 and name + 3 == spec.input_size


def test_the_legacy_layout_has_six_jumps_and_no_nana_stage_extras_or_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, state = _spec(tmp_path, monkeypatch, version=4, max_names=16)
    network = port.SlippiAINetwork(spec)
    network.load_arrays(port.release_arrays(state))
    frames = neutral_frame((2,))
    frames["p0"]["jumps_left"][:] = torch.tensor([6, 5])
    frames["p0"]["nana"]["exists"][:] = True
    frames["items"]["exists"][:, 0] = True
    x = network.embed(port.pack_game(frames), torch.zeros(2, 13, dtype=torch.long), torch.tensor([4, 15]))
    assert x.shape == (2, 1121)
    assert x[0, 437:443].sum() == 0.0  # 6 jumps is outside the legacy one-hot of 6: all zeros (EMPTY)
    assert x[1, 437 + 5] == 1.0
    assert x[1, 1121 - 16 + 15] == 1.0 and x[0, 1121 - 16 + 4] == 1.0


# ---------------------------------------------------------------------------
# the network and the head against a NumPy reference
# ---------------------------------------------------------------------------


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _named(spec: port.ReleaseSpec, arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: array.astype(np.float64)
        for (name, _), array in zip(port.expected_shapes(spec), arrays, strict=False)
    }


def _reference_core(
    spec: port.ReleaseSpec,
    weights: dict[str, np.ndarray],
    x: np.ndarray,
    hidden: np.ndarray,
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tf/networks.py: encoder, then per layer ResidualWrapper(snt.LSTM) and ResBlock(LayerNorm, Linear, gelu,
    Linear)."""

    x = x @ weights["encoder/w"] + weights["encoder/b"]
    new_hidden, new_cell = [], []
    for layer in range(spec.num_layers):
        base = f"layer{layer}"
        gates = (
            x @ weights[f"{base}/lstm/w_i"]
            + hidden[layer] @ weights[f"{base}/lstm/w_h"]
            + weights[f"{base}/lstm/b"]
        )
        i, f, g, o = np.split(gates, 4, axis=1)
        c = _sigmoid(f) * cell[layer] + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        x = x + h
        centred = x - x.mean(-1, keepdims=True)
        normed = centred / np.sqrt(np.mean(centred**2, axis=-1, keepdims=True))
        y = normed * weights[f"{base}/layernorm/scale"] + weights[f"{base}/layernorm/bias"]
        y = _gelu(y @ weights[f"{base}/linear1/w"] + weights[f"{base}/linear1/b"])
        x = x + (y @ weights[f"{base}/linear2/w"] + weights[f"{base}/linear2/b"])
        new_hidden.append(h)
        new_cell.append(c)
    return x, np.stack(new_hidden), np.stack(new_cell)


def _reference_head(
    spec: port.ReleaseSpec,
    weights: dict[str, np.ndarray],
    out: np.ndarray,
    prev: np.ndarray,
    forced: np.ndarray,
) -> list[np.ndarray]:
    """tf/controller_heads.py AutoRegressive.distance: logits per component, the residual teacher-forced."""

    residual = out @ weights["head/to_residual/w"] + weights["head/to_residual/b"]
    logits = []
    for k, (component, size) in enumerate(zip(port.COMPONENTS, spec.component_sizes, strict=True)):

        def embed(value: np.ndarray, size: int = size) -> np.ndarray:
            if size == 1:
                return value.astype(np.float64)[:, None]
            return np.eye(size)[value]

        h = np.concatenate([residual, embed(prev[:, k])], axis=-1)
        for layer in range(spec.component_depth + 1):
            h = (
                h @ weights[f"head/{component}/encoder{layer}/w"]
                + weights[f"head/{component}/encoder{layer}/b"]
            )
            if layer < spec.component_depth:
                h = np.maximum(h, 0.0)
        logits.append(h)
        residual = (
            residual
            + embed(forced[:, k]) @ weights[f"head/{component}/decoder/w"]
            + weights[f"head/{component}/decoder/b"]
        )
    return logits


def _random_controller(batch: int, generator: torch.Generator) -> torch.Tensor:
    buttons = torch.randint(0, 2, (batch, 8), generator=generator)
    sticks = torch.randint(0, 33, (batch, 4), generator=generator)
    shoulder = torch.randint(0, 11, (batch, 1), generator=generator)
    return torch.cat([buttons, sticks, shoulder], dim=1)


def test_core_and_head_match_the_numpy_reference_over_frames_and_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, state = _spec(tmp_path, monkeypatch)
    arrays = list(port.release_arrays(state))
    network = port.SlippiAINetwork(spec)
    network.load_arrays(tuple(arrays))
    weights = _named(spec, arrays)
    generator = torch.Generator().manual_seed(3)
    batch = 3
    hidden = torch.zeros(spec.num_layers, batch, spec.hidden_size)
    cell = torch.zeros_like(hidden)
    ref_hidden, ref_cell = hidden.double().numpy(), cell.double().numpy()
    prev = torch.zeros(batch, 13, dtype=torch.long)
    for frame in range(6):
        frames = random_valid_frames((batch,), generator)
        names = torch.randint(0, 3, (batch,), generator=generator)
        x = network.embed(port.pack_game(frames), prev, names)
        out, hidden, cell = network.core(x, hidden, cell)
        ref_out, ref_hidden, ref_cell = _reference_core(
            spec, weights, x.double().numpy(), ref_hidden, ref_cell
        )
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-4, atol=1e-4)
        forced = _random_controller(batch, generator)
        logits = network.head_forced(out, prev, forced)
        reference = _reference_head(spec, weights, ref_out, prev.numpy(), forced.numpy())
        for mine, theirs in zip(logits, reference, strict=True):
            np.testing.assert_allclose(mine.numpy(), theirs, rtol=1e-4, atol=1e-4)
        prev = forced
        if frame == 3:  # a reset zeroes the recurrent rows only (Network.step_with_reset)
            hidden[:, 1] = 0.0
            cell[:, 1] = 0.0
            ref_hidden[:, 1] = 0.0
            ref_cell[:, 1] = 0.0


def test_swapped_lstm_matrices_load_silently_and_only_parity_catches_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``w_h`` and ``w_i`` are both ``[H, 4H]`` in ``tx_like`` (the input of every LSTM is the residual): a
    read in the wrong order passes every shape check, which is why the parity gate exists (LEAGUE-PLAN.md
    §3.4)."""

    spec, state = _spec(tmp_path, monkeypatch)
    arrays = list(port.release_arrays(state))
    names = [name for name, _ in port.expected_shapes(spec)]
    w_h, w_i = names.index("layer0/lstm/w_h"), names.index("layer0/lstm/w_i")
    swapped = list(arrays)
    swapped[w_h], swapped[w_i] = arrays[w_i], arrays[w_h]
    good, bad = port.SlippiAINetwork(spec), port.SlippiAINetwork(spec)
    good.load_arrays(tuple(arrays))
    bad.load_arrays(tuple(swapped))  # no error
    weights = _named(spec, arrays)
    generator = torch.Generator().manual_seed(5)
    hidden = torch.randn(spec.num_layers, 2, spec.hidden_size, generator=generator)
    cell = torch.randn(spec.num_layers, 2, spec.hidden_size, generator=generator)
    x = good.embed(
        port.pack_game(random_valid_frames((2,), generator)),
        torch.zeros(2, 13, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    reference, _, _ = _reference_core(
        spec, weights, x.double().numpy(), hidden.double().numpy(), cell.double().numpy()
    )
    np.testing.assert_allclose(good.core(x, hidden, cell)[0].numpy(), reference, rtol=1e-4, atol=1e-4)
    assert not np.allclose(bad.core(x, hidden, cell)[0].numpy(), reference, rtol=1e-4, atol=1e-4)


def test_a_batched_step_equals_per_row_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, state = _spec(tmp_path, monkeypatch)
    network = port.SlippiAINetwork(spec)
    network.load_arrays(port.release_arrays(state))
    generator = torch.Generator().manual_seed(11)
    batch = 4
    frames = random_valid_frames((batch,), generator)
    prev = _random_controller(batch, generator)
    names = torch.randint(0, 3, (batch,), generator=generator)
    hidden = torch.randn(spec.num_layers, batch, spec.hidden_size, generator=generator)
    cell = torch.randn(spec.num_layers, batch, spec.hidden_size, generator=generator)
    assert spec.uniforms_per_row == 8 + 4 * 33 + 11
    uniforms = torch.rand(batch, spec.uniforms_per_row, generator=generator)
    x = network.embed(port.pack_game(frames), prev, names)
    out, new_hidden, _ = network.core(x, hidden, cell)
    sample, logits = network.head_sample(out, prev, uniforms, 1.0)
    for row in range(batch):
        one = {key: value for key, value in frames.items()}
        single = port.pack_game(_index(one, row))
        x1 = network.embed(single, prev[row : row + 1], names[row : row + 1])
        out1, hidden1, _ = network.core(x1, hidden[:, row : row + 1], cell[:, row : row + 1])
        sample1, logits1 = network.head_sample(out1, prev[row : row + 1], uniforms[row : row + 1], 1.0)
        torch.testing.assert_close(out1, out[row : row + 1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(hidden1, new_hidden[:, row : row + 1], rtol=1e-5, atol=1e-5)
        assert torch.equal(sample1, sample[row : row + 1])
        for mine, batched in zip(logits1, logits, strict=True):
            torch.testing.assert_close(mine, batched[row : row + 1], rtol=1e-5, atol=1e-5)
    # Sampling with a component's own samples as the forced values reproduces the sampled path's logits.
    for mine, forced in zip(logits, network.head_forced(out, prev, sample), strict=True):
        torch.testing.assert_close(mine, forced)


def _index(tree: FrameTree, row: int) -> FrameTree:
    return {
        key: _index(value, row) if isinstance(value, dict) else value[row : row + 1]
        for key, value in tree.items()
    }


def test_sampling_follows_the_logits() -> None:
    """Bernoulli buttons (``u < sigmoid(logit)``) and Gumbel-max categoricals from pre-drawn uniforms."""

    generator = torch.Generator().manual_seed(2)
    rows = 40_000
    button = torch.full((rows, 1), 0.8)
    stick = torch.log(torch.tensor([0.1, 0.6, 0.3])).repeat(rows, 1)
    uniforms = torch.rand(rows, 1 + 3, generator=generator)
    pressed = port.sample_bernoulli(button[:, 0], uniforms[:, 0], 1.0)
    assert abs(pressed.float().mean().item() - float(torch.sigmoid(torch.tensor(0.8)))) < 0.01
    chosen = port.sample_categorical(stick, uniforms[:, 1:], 1.0)
    frequencies = torch.bincount(chosen, minlength=3).float() / rows
    torch.testing.assert_close(frequencies, torch.tensor([0.1, 0.6, 0.3]), atol=0.01, rtol=0.0)
    cold = port.sample_categorical(stick, uniforms[:, 1:], 0.25)  # temperature sharpens towards the mode
    assert (cold == 1).float().mean() > 0.9


# ---------------------------------------------------------------------------
# the agent glue
# ---------------------------------------------------------------------------


def _agent(
    spec: port.ReleaseSpec, state: Any, batch: int, *, names: tuple[str, ...] | None = None, seed: int = 0
) -> port.SlippiAIAgent:
    network = port.SlippiAINetwork(spec)
    network.load_arrays(port.release_arrays(state))
    return port.SlippiAIAgent(
        network, batch, names=names or (spec.default_name,) * batch, device="cpu", seed=seed
    )


def test_the_delay_queue_is_primed_once_with_dummy_outputs_and_never_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, state = _spec(tmp_path, monkeypatch, delay=3)
    agent = _agent(spec, state, 2)
    generator = torch.Generator().manual_seed(1)
    fresh: list[torch.Tensor] = []
    executed: list[ControllerState] = []
    for frame in range(9):
        reset = torch.tensor([frame == 0, frame in (0, 5)])
        executed.append(agent.step(random_valid_frames((2,), generator), reset))
        fresh.append(agent.last_sample.clone())
    for frame in range(3):  # upstream primes the queue with dummy sample outputs (eval_lib.py:131)
        control = executed[frame]
        assert not any(bool(value.any()) for value in control.buttons.values())
        assert control.main_stick.x.tolist() == [0.0, 0.0] and control.shoulder.tolist() == [0.0, 0.0]
    for frame in range(3, 9):  # the reset at frame 5 does not flush the queue
        assert torch.equal(
            port.decode_indices(fresh[frame - 3], spec).main_stick.x, executed[frame].main_stick.x
        )
        assert torch.equal(port.encode_controller(executed[frame], spec), fresh[frame - 3])


def test_needs_reset_zeroes_the_recurrent_rows_and_keeps_the_previous_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Network.step_with_reset`` zeroes the LSTM rows of a new game; the previous controller input stays the
    agent's own last sample (``BasicAgent._prev_controller`` is never reset) and so does the delay queue."""

    spec, state = _spec(tmp_path, monkeypatch, delay=0)
    agent = _agent(spec, state, 2)
    generator = torch.Generator().manual_seed(4)
    frames = [random_valid_frames((2,), generator) for _ in range(4)]
    for frame in frames:
        frame["p1"]["action"][:] = STANDING  # keep the tech mask out of the replay below
    for frame in frames[:3]:
        agent.step(frame, torch.tensor([False, False]))
    prev = agent.last_sample.clone()
    hidden, cell = agent.hidden.clone(), agent.cell.clone()
    assert prev.shape == (2, 13) and bool(hidden.abs().sum() > 0)
    agent.step(frames[3], torch.tensor([False, True]))
    keep = torch.tensor([True, False])[None, :, None]
    x = agent.network.embed(port.pack_game(frames[3]), prev, agent.names)
    _, expected_hidden, expected_cell = agent.network.core(
        x, torch.where(keep, hidden, 0.0), torch.where(keep, cell, 0.0)
    )
    torch.testing.assert_close(agent.hidden, expected_hidden)
    torch.testing.assert_close(agent.cell, expected_cell)


def test_the_tech_mask_hides_the_opponents_tech_for_seven_frames() -> None:
    """``observations.AnimationFilter.update``: the count restarts on every new action, and neutral / forward
    / back tech read as neutral tech while the count is below 7 (``DEFAULT_TECH_MASK_WINDOW``); a reset
    forgets."""

    actions = [STANDING, *[B_TECH] * 9, F_TECH, F_TECH, N_TECH, STANDING, B_TECH, B_TECH]
    resets = [False] * len(actions)
    resets[15] = True
    prev = torch.zeros(1, dtype=torch.long)
    count = torch.zeros(1, dtype=torch.long)
    masked = []
    for action, reset in zip(actions, resets, strict=True):
        out, prev, count = port.tech_mask_step(torch.tensor([action]), prev, count, torch.tensor([reset]))
        masked.append(int(out))
    assert masked == [
        STANDING,
        *[N_TECH] * 7,
        B_TECH,
        B_TECH,  # the eighth and ninth frames of a back tech are visible
        N_TECH,
        N_TECH,
        N_TECH,
        STANDING,
        N_TECH,
        N_TECH,  # frame 15 continues the back tech but the reset restarted its count
    ]


def test_names_cycle_per_dolphin_and_decode_uses_melees_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, state = _spec(tmp_path, monkeypatch)
    agent = _agent(spec, state, 3, names=("Cody", "Hax", "Cody"))
    assert agent.names.tolist() == [1, 2, 1]
    with pytest.raises(ValueError, match="Nametag"):
        _agent(spec, state, 1, names=("Mango",))
    indices = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 1, 0, 16, 32, 7, 10], [0] * 12 + [3]])
    control = port.decode_indices(indices, spec)
    assert control.buttons.A.tolist() == [True, False] and control.buttons.D_UP.tolist() == [True, False]
    assert control.main_stick.x.tolist() == [0.0, 0.0] and control.main_stick.y.tolist() == [0.5, 0.0]
    assert control.c_stick.x.tolist() == [1.0, 0.0]
    assert control.c_stick.y[0].item() == np.float32(7 / 32) and control.shoulder[1].item() == np.float32(
        3 / 10
    )
    assert control.shoulder.dtype == torch.float32 and control.shoulder.is_contiguous()
    assert torch.equal(port.encode_controller(control, spec), indices)


# ---------------------------------------------------------------------------
# the opponent and the captured step
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")


def test_the_torch_opponent_plays_port_one_from_its_own_perspective(tmp_path: Path) -> None:
    path = tmp_path / "gm"
    write_release(path, names=("Master Player",), delay=2)
    network = port.load_network(path, name="gm")
    opponent = port.SlippiAITorchOpponent(
        network, 3, names=("Master Player",) * 3, device="cpu", seed=4, characters=(1, 22, 22), release="gm"
    )
    assert opponent.port == 1 and opponent.controls_port and opponent.characters == (1, 22, 22)
    assert opponent.generator is opponent.agent.generator and opponent.release == "gm"
    generator = torch.Generator().manual_seed(8)
    frames = [random_valid_frames((3,), generator) for _ in range(5)]
    reset = torch.tensor([True, False, False])
    with pytest.raises(NotImplementedError):
        opponent.act(frames[0], reset)
    twin = port.SlippiAITorchOpponent(
        port.load_network(path, name="gm"), 3, names=("Master Player",) * 3, device="cpu", seed=4
    )
    outputs = [opponent.act_state(frame, reset) for frame in frames]
    assert all(control.shoulder.shape == (3,) for control in outputs)
    for frame, control in zip(frames, outputs, strict=True):  # the same seed plays the same controllers
        assert torch.equal(
            port.encode_controller(twin.act_state(frame, reset), network.spec),
            port.encode_controller(control, network.spec),
        )
    assert opponent.refresh(object()) is None  # type: ignore[func-returns-value]
    opponent.reset_state()  # a relaunch: the queue is primed again with dummy outputs
    first = opponent.act_state(frames[0], reset)
    assert port.encode_controller(first, network.spec).abs().sum() == 0
    stats = opponent.stats()
    assert (
        stats["release"] == "gm"
        and stats["delay"] == 2
        and stats["frames_seen"] == 18
        and stats["calls"] == 6
    )


def test_a_captured_step_needs_a_cuda_device(tmp_path: Path) -> None:
    path = tmp_path / "release"
    write_release(path)
    with pytest.raises(ValueError, match="CUDA"):
        port.SlippiAIAgent(
            port.load_network(path, name="tiny"), 2, names=("Cody",) * 2, device="cpu", graph=True
        )


@cuda_only
def test_the_captured_step_replays_the_eager_step_bit_for_bit(tmp_path: Path) -> None:
    from melee_rl.frames import tree_to

    path = tmp_path / "release"
    write_release(path, delay=2)
    eager = port.SlippiAIAgent(
        port.load_network(path, name="tiny"), 4, names=("Cody",) * 4, device="cuda", seed=9
    )
    graphed = port.SlippiAIAgent(
        port.load_network(path, name="tiny"), 4, names=("Cody",) * 4, device="cuda", seed=9, graph=True
    )
    generator = torch.Generator().manual_seed(1)
    for frame in range(12):
        host = random_valid_frames((4,), generator)
        reset = torch.tensor([frame == 0, frame == 5, False, frame % 4 == 0])
        # The eager agent gets device frames; the captured one host frames (its pinned staging path).
        left = eager.step(tree_to(host, "cuda"), reset.cuda())
        right = graphed.step(host, reset)
        assert torch.equal(eager.last_sample, graphed.last_sample)
        assert torch.equal(
            port.encode_controller(left, eager.spec), port.encode_controller(right, eager.spec)
        )
        if frame == 7:
            eager.reset_all()
            graphed.reset_all()  # in place: the captured graph stays bound to the same state tensors
    assert graphed.replays == 12 and graphed.captures == 1
    for name in ("hidden", "cell", "prev", "tech_prev", "tech_count"):
        assert torch.equal(getattr(eager, name), getattr(graphed, name)), name
