"""The parity harness (17 Sep 2026, LEAGUE-PLAN.md §3.4) with the TensorFlow side faked.

On Modal the reference is slippi-ai's own agent exactly as the recorder plays it (``SlippiAIPlayer``); here it
is a second PyTorch agent on the same weights, which is what the harness's plumbing needs to be proven with:
the port is teacher-forced on the reference's fresh sample every frame, the per-component logits are compared,
and the gate (|Δ logit| <= 1e-3 * max(1, |logit|), mean KL <= 1e-6 per component) reads pass or fail."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from melee_rl import slippi_ai_parity as parity
from melee_rl import slippi_ai_torch as port
from melee_rl.config import CONFIG_DIR, finalize_config, load_config, resolve_model_config
from melee_rl.frames import random_valid_frames
from tensor_batch import BUTTON_ORDER
from tests.rl._slippi_release import write_release


def test_component_kl_matches_the_closed_forms() -> None:
    logits = torch.tensor([[0.3, -1.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(parity.component_kl(logits, logits), torch.zeros(2, dtype=torch.float64))
    other = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], dtype=torch.float64)
    p, q = torch.softmax(logits, -1), torch.softmax(other, -1)
    expected = (p * (p.log() - q.log())).sum(-1)
    torch.testing.assert_close(parity.component_kl(logits, other), expected)
    a, b = (
        torch.tensor([[1.5], [-2.0]], dtype=torch.float64),
        torch.tensor([[0.5], [-2.0]], dtype=torch.float64),
    )
    pa, pb = torch.sigmoid(a[:, 0]), torch.sigmoid(b[:, 0])
    bernoulli = pa * (pa.log() - pb.log()) + (1 - pa) * ((1 - pa).log() - (1 - pb).log())
    torch.testing.assert_close(parity.component_kl(a, b), bernoulli)


def _stick(x: int, y: int) -> SimpleNamespace:
    return SimpleNamespace(x=np.array([x], np.uint8), y=np.array([y], np.uint8))


def test_upstream_sample_outputs_map_by_field_name() -> None:
    """``SampleOutputs.controller_state`` is a ``Controller`` NamedTuple (main_stick, c_stick, shoulder,
    buttons) while the head's order is buttons, main x / y, c x / y, shoulder: the mapping must go by name."""

    buttons = SimpleNamespace(**{name: np.array([name in ("B", "D_UP")]) for name in BUTTON_ORDER})
    controller = SimpleNamespace(
        main_stick=_stick(3, 4), c_stick=_stick(5, 6), shoulder=np.array([7], np.uint8), buttons=buttons
    )
    logits = SimpleNamespace(
        main_stick=SimpleNamespace(x=np.full((1, 33), 1.0), y=np.full((1, 33), 2.0)),
        c_stick=SimpleNamespace(x=np.full((1, 33), 3.0), y=np.full((1, 33), 4.0)),
        shoulder=np.full((1, 11), 5.0),
        buttons=SimpleNamespace(
            **{name: np.full((1, 1), float(index)) for index, name in enumerate(BUTTON_ORDER)}
        ),
    )
    forced, logit_list = parity.sample_outputs_to_forced(
        SimpleNamespace(controller_state=controller, logits=logits)
    )
    assert forced.tolist() == [[0, 1, 0, 0, 0, 0, 0, 1, 3, 4, 5, 6, 7]]
    assert [float(item[0, 0]) for item in logit_list] == [0, 1, 2, 3, 4, 5, 6, 7, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [tuple(item.shape) for item in logit_list] == [(1, 1)] * 8 + [(1, 33)] * 4 + [(1, 11)]


def _agents(path: Path, batch: int, *, perturb: float = 0.0) -> tuple[port.SlippiAIAgent, port.SlippiAIAgent]:
    reference = port.load_network(path, name="tiny")
    shadow = port.load_network(path, name="tiny")
    if perturb:
        with torch.no_grad():
            shadow.weight("layer0/lstm/b").add_(perturb)
    names = ("Cody",) * batch
    return (
        port.SlippiAIAgent(reference, batch, names=names, device="cpu", seed=3),
        port.SlippiAIAgent(shadow, batch, names=names, device="cpu", seed=99),
    )


def test_the_parity_opponent_shadows_the_reference_and_reads_the_gate(tmp_path: Path) -> None:
    path = tmp_path / "release"
    write_release(path, delay=2)
    generator = torch.Generator().manual_seed(6)
    frames = [random_valid_frames((3,), generator) for _ in range(20)]
    reference, shadow = _agents(path, 3)
    # The reference was warmed up (upstream's BasicAgent.warmup): its first controller input is not zeros, and
    # the shadow must start from the same one or its LSTM diverges from the first frame on.
    reference.prev.copy_(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 16, 16, 3, 7, 2]] * 3))
    recorder = parity.ParityRecorder()
    opponent = parity.ParityOpponent(parity.PortSource(reference), shadow, recorder)
    assert torch.equal(shadow.prev, reference.prev)
    assert opponent.port == 1 and opponent.controls_port
    for index, frame in enumerate(frames):
        reset = torch.tensor([index == 0, index in (0, 9), index == 0])
        control = opponent.act_state(frame, reset)
        assert control.shoulder.shape == (3,)
    summary = recorder.summary()
    assert summary["frames"] == 60 and summary["pass"]
    assert summary["components"]["main_x"]["max_abs_logit"] == 0.0
    assert summary["components"]["A"]["mean_kl"] == 0.0 and summary["components"]["shoulder"]["top1"] == 1.0
    # A port that is not the reference -- one LSTM bias nudged -- fails the gate on every component.
    reference, shadow = _agents(path, 3, perturb=0.05)
    recorder = parity.ParityRecorder()
    opponent = parity.ParityOpponent(parity.PortSource(reference), shadow, recorder)
    for index, frame in enumerate(frames):
        opponent.act_state(frame, torch.tensor([index == 0] * 3))
    failed = recorder.summary()
    assert not failed["pass"] and failed["components"]["main_x"]["max_abs_logit"] > 1e-3


def test_run_parity_on_the_dummy_env_writes_one_report_per_case(tmp_path: Path) -> None:
    from melee_rl.checkpoint import save_bc_checkpoint
    from melee_rl.train import build_policy

    base = load_config(CONFIG_DIR / "smoke_tiny.toml")
    model_dir = tmp_path / "releases"
    write_release(model_dir / "fox_d18_ditto_v4", version=4, max_names=16, names=("Cody", "Hax"), delay=2)
    config = finalize_config(
        replace(
            base,
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
            opponent=replace(base.opponent, type="self", train=True),
            slippi_ai=replace(base.slippi_ai, model_dir=str(model_dir), verify_sha256=False),
        ),
        resolve_model_config(base),
    )
    fork = tmp_path / "fork.pt"
    model = resolve_model_config(config)
    save_bc_checkpoint(fork, build_policy(config, model, torch.device("cpu")).state_dict(), model)
    cases = (parity.ParityCase(release="fox_d18_ditto_v4", characters=(1, 1), stages=(25,), frames=48),)

    def reference(case: parity.ParityCase, env: object, agent: port.SlippiAIAgent) -> parity.ReferenceSource:
        twin = port.load_network(model_dir / case.release, name=case.release)
        return parity.PortSource(
            port.SlippiAIAgent(twin, agent.num_envs, names=agent.name_tags, device="cpu")
        )

    output = tmp_path / "parity.json"
    report = parity.run_slippi_parity(
        config, str(fork), cases, device="cpu", output=output, reference_factory=reference
    )
    assert report["pass"] and len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["release"] == "fox_d18_ditto_v4" and case["frames"] >= 48 and case["pass"]
    assert case["names"] == ["Cody", "Cody"] and case["delay"] == 2
    assert json.loads(output.read_text())["pass"]
    assert parity.DEFAULT_CASES[0].release == "gm" and {10, 9, 13, 1} <= set(
        parity.DEFAULT_CASES[0].characters
    )


def test_the_gate_reads_logit_differences_relative_to_their_magnitude() -> None:
    """17 Sep 2026: the first parity run missed an absolute 1e-3 bar on one component (the Falco's D_UP,
    2.6e-3) with mean KL 1e-12 and full argmax agreement, and two correct float32 runs of the port itself (CPU
    vs GPU) differ by 1e-2 at a logit of -494.  The bar is relative: |d| <= 1e-3 * max(1, |logit|); the logit
    at the largest difference is reported beside it."""

    sizes = [1] * 8 + [33] * 4 + [11]
    reference = [torch.zeros(2, size, dtype=torch.float64) for size in sizes]
    reference[7][:, 0] = -500.0  # a component the agent never uses (D_UP): a huge negative logit
    shadow = [tensor.clone() for tensor in reference]
    shadow[7][:, 0] += 0.01  # 2e-5 relative: float32 rounding, not a bug
    recorder = parity.ParityRecorder()
    recorder.add(reference, shadow)
    summary = recorder.summary()
    d_up = summary["components"]["D_UP"]
    assert d_up["max_abs_logit"] == pytest.approx(0.01) and d_up["logit_at_max"] == -500.0
    assert d_up["max_rel_logit"] == pytest.approx(2e-5) and summary["pass"]
    small = [tensor.clone() for tensor in reference]
    small[8][0, 3] += 2e-3  # a unit-scale logit off by 2e-3: 2e-3 relative
    recorder = parity.ParityRecorder()
    recorder.add(reference, small)
    failed = recorder.summary()
    assert not failed["pass"] and failed["components"]["main_x"]["max_rel_logit"] == pytest.approx(2e-3)
    assert failed["thresholds"] == {"max_rel_logit": 1e-3, "mean_kl": 1e-6}
