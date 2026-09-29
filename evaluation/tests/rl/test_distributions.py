from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch.distributions import Categorical, kl_divergence
from torch.utils._python_dispatch import TorchDispatchMode

from controller_codec import ControllerLabels, CustomV1Codec
from melee_rl import distributions

LN2 = math.log(2.0)
LN3 = math.log(3.0)


def _toy_logits() -> dict[str, torch.Tensor]:
    # Row 0: buttons p = (1/4, 1/4, 1/2), main stick p = (3/4, 1/4); row 1: permutations.
    return {
        "buttons": torch.tensor([[0.0, 0.0, LN2], [LN2, 0.0, 0.0]]),
        "main_stick": torch.tensor([[LN3, 0.0], [0.0, LN3]]),
    }


def _uniform_like(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in logits.items()}


def test_logprob_kl_entropy_hand_values() -> None:
    logits = _toy_logits()
    labels = ControllerLabels(buttons=torch.tensor([2, 1]), main_stick=torch.tensor([0, 1]))

    log_p = distributions.log_prob(logits, labels)
    torch.testing.assert_close(log_p, torch.tensor([math.log(3 / 8), math.log(3 / 16)]))
    by_component = distributions.log_probs_by_component(logits, labels)
    torch.testing.assert_close(by_component["buttons"], torch.tensor([math.log(0.5), math.log(0.25)]))
    torch.testing.assert_close(by_component["main_stick"], torch.tensor([math.log(0.75), math.log(0.75)]))

    entropy_buttons = 1.5 * LN2
    entropy_main = 2.0 * LN2 - 0.75 * LN3
    expected_entropy = torch.full((2,), entropy_buttons + entropy_main)
    torch.testing.assert_close(distributions.entropy(logits), expected_entropy)

    uniform = _uniform_like(logits)
    expected_kl = (LN3 - entropy_buttons) + (LN2 - entropy_main)
    torch.testing.assert_close(distributions.kl(logits, uniform), torch.full((2,), expected_kl))
    torch.testing.assert_close(distributions.kl(logits, logits), torch.zeros(2), rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(distributions.kl(uniform, uniform), torch.zeros(2), rtol=0.0, atol=1.0e-7)

    # Mapping labels behave like ControllerLabels, and the entropy of uniform 728/85-way logits is log V.
    torch.testing.assert_close(distributions.log_prob(logits, labels.as_dict()), log_p)
    wide_uniform = {"buttons": torch.zeros(3, 728), "main_stick": torch.zeros(3, 85)}
    wide_entropy = torch.full((3,), math.log(728) + math.log(85))
    torch.testing.assert_close(distributions.entropy(wide_uniform), wide_entropy)


def test_matches_torch_categorical_oracle() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = {
        "buttons": torch.randn(4, 5, 728, generator=generator) * 3.0,
        "main_stick": torch.randn(4, 5, 85, generator=generator) * 3.0,
    }
    other = {
        "buttons": torch.randn(4, 5, 728, generator=generator),
        "main_stick": torch.randn(4, 5, 85, generator=generator),
    }
    labels = ControllerLabels(
        buttons=torch.randint(0, 728, (4, 5), generator=generator),
        main_stick=torch.randint(0, 85, (4, 5), generator=generator),
    )
    expected_log_prob = torch.zeros(4, 5)
    expected_entropy = torch.zeros(4, 5)
    expected_kl = torch.zeros(4, 5)
    for name, taken in (("buttons", labels.buttons), ("main_stick", labels.main_stick)):
        p = Categorical(logits=logits[name])
        q = Categorical(logits=other[name])
        expected_log_prob += p.log_prob(taken)
        expected_entropy += p.entropy()
        expected_kl += kl_divergence(p, q)

    actual_log_prob = distributions.log_prob(logits, labels)
    torch.testing.assert_close(actual_log_prob, expected_log_prob, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(distributions.entropy(logits), expected_entropy, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(distributions.kl(logits, other), expected_kl, rtol=1.0e-5, atol=1.0e-5)
    assert bool((distributions.kl(logits, other) >= 0).all())
    # bf16 inputs are promoted to fp32 before any math.
    low_precision = {name: value.to(torch.bfloat16) for name, value in logits.items()}
    assert distributions.log_prob(low_precision, labels).dtype == torch.float32


def test_sample_matches_codec_sample_and_respects_temperature() -> None:
    codec = CustomV1Codec()
    logits_generator = torch.Generator().manual_seed(15031)
    logits = {
        "buttons": torch.randn(6, 3, 728, generator=logits_generator),
        "main_stick": torch.randn(6, 3, 85, generator=logits_generator),
    }
    ours = distributions.sample(logits, generator=torch.Generator().manual_seed(577965))
    theirs = codec.sample(logits, generator=torch.Generator().manual_seed(577965))
    assert torch.equal(ours.buttons, theirs.buttons)
    assert torch.equal(ours.main_stick, theirs.main_stick)
    assert ours.buttons.dtype == torch.long and ours.main_stick.dtype == torch.long

    tempered = distributions.sample(logits, temperature=0.5, generator=torch.Generator().manual_seed(3))
    expected = codec.sample(logits, temperature=0.5, generator=torch.Generator().manual_seed(3))
    assert torch.equal(tempered.buttons, expected.buttons)
    assert torch.equal(tempered.main_stick, expected.main_stick)

    greedy = distributions.sample(logits, temperature=0.0)
    assert torch.equal(greedy.buttons, logits["buttons"].argmax(dim=-1))
    assert torch.equal(greedy.main_stick, logits["main_stick"].argmax(dim=-1))
    with pytest.raises(ValueError, match="temperature"):
        distributions.sample(logits, temperature=-1.0)


def test_validation_errors() -> None:
    logits = _toy_logits()
    with pytest.raises(KeyError, match="missing components"):
        distributions.entropy({"buttons": logits["buttons"]})
    with pytest.raises(ValueError, match="prefix shape"):
        distributions.entropy({"buttons": logits["buttons"], "main_stick": logits["main_stick"][:1]})
    with pytest.raises(ValueError, match=r"in \[0, 3\)"):
        distributions.log_prob(logits, ControllerLabels(torch.tensor([3, 0]), torch.tensor([0, 0])))
    with pytest.raises(ValueError, match="shape"):
        distributions.log_prob(logits, ControllerLabels(torch.tensor([0]), torch.tensor([0])))
    with pytest.raises(TypeError, match="integer"):
        distributions.log_prob(logits, ControllerLabels(torch.tensor([0.0, 1.0]), torch.tensor([0, 0])))
    with pytest.raises(ValueError, match="same shape"):
        distributions.kl(logits, {"buttons": torch.zeros(2, 4), "main_stick": logits["main_stick"]})


class _ScalarReads(TorchDispatchMode):
    """Counts device-to-host scalar reads (``aten._local_scalar_dense``) inside the block."""

    def __init__(self) -> None:
        self.syncs = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        if func._overloadpacket is torch.ops.aten._local_scalar_dense:
            self.syncs += 1
        return func(*args, **(kwargs or {}))


def test_ppo_terms_equal_the_separate_functions() -> None:
    """5 Sep 2026 (/rl-learner-speedup, commit A): ``ppo_terms`` computes the learner's five per-position
    quantities from one ``log_softmax`` per logits tensor -- the same kernels on the same inputs with the
    same per-term formulas as ``log_prob`` / ``entropy`` / ``kl``, so the values are bit-identical; without
    a teacher the two teacher terms are zeros; ``validate_labels=False`` drops the two label range checks
    (host reads)."""

    generator = torch.Generator().manual_seed(11)

    def logits(scale: float) -> dict[str, torch.Tensor]:
        return {
            "buttons": torch.randn(3, 7, 728, generator=generator) * scale,
            "main_stick": torch.randn(3, 7, 85, generator=generator) * scale,
        }

    new, old, teacher = logits(2.0), logits(1.5), logits(1.0)
    labels = ControllerLabels(
        buttons=torch.randint(0, 728, (3, 7), generator=generator),
        main_stick=torch.randint(0, 85, (3, 7), generator=generator),
    )
    terms = distributions.ppo_terms(new, labels, old, teacher)
    assert set(terms) == {"log_prob", "entropy", "actor_kl", "teacher_kl", "reverse_teacher_kl"}
    assert torch.equal(terms["log_prob"], distributions.log_prob(new, labels))
    assert torch.equal(terms["entropy"], distributions.entropy(new))
    assert torch.equal(terms["actor_kl"], distributions.kl(old, new))
    assert torch.equal(terms["teacher_kl"], distributions.kl(new, teacher))
    assert torch.equal(terms["reverse_teacher_kl"], distributions.kl(teacher, new))
    # No teacher: the teacher terms are zeros shaped like the rows; the rest is unchanged.
    alone = distributions.ppo_terms(new, labels, old)
    assert torch.equal(alone["teacher_kl"], torch.zeros(3, 7))
    assert torch.equal(alone["reverse_teacher_kl"], torch.zeros(3, 7))
    assert torch.equal(alone["log_prob"], terms["log_prob"])
    assert torch.equal(alone["actor_kl"], terms["actor_kl"])
    # Gradients through the new logits agree with the separate functions (a different autograd graph, so
    # fp32 accumulation order differs: close, not equal).
    fused_inputs = {name: value.clone().requires_grad_(True) for name, value in new.items()}
    fused = distributions.ppo_terms(fused_inputs, labels, old, teacher)
    torch.stack([term.sum() for term in fused.values()]).sum().backward()
    plain_inputs = {name: value.clone().requires_grad_(True) for name, value in new.items()}
    (
        distributions.log_prob(plain_inputs, labels).sum()
        + distributions.entropy(plain_inputs).sum()
        + distributions.kl(old, plain_inputs).sum()
        + distributions.kl(plain_inputs, teacher).sum()
        + distributions.kl(teacher, plain_inputs).sum()
    ).backward()
    for name in new:
        assert fused_inputs[name].grad is not None and plain_inputs[name].grad is not None
        torch.testing.assert_close(fused_inputs[name].grad, plain_inputs[name].grad, rtol=1.0e-5, atol=1.0e-6)
    # The label range check is the only host read; the learner drops it for labels a Trajectory validated.
    checked = _ScalarReads()
    with checked:
        distributions.ppo_terms(new, labels, old, teacher)
    unchecked = _ScalarReads()
    with unchecked:
        distributions.ppo_terms(new, labels, old, teacher, validate_labels=False)
    assert checked.syncs == 2 and unchecked.syncs == 0
    bad = ControllerLabels(buttons=labels.buttons.clone(), main_stick=labels.main_stick.clone())
    bad.buttons[0, 0] = 728
    with pytest.raises(ValueError, match="buttons labels must be in"):
        distributions.ppo_terms(new, bad, old, teacher)
    with pytest.raises(ValueError, match="same shape"):
        distributions.ppo_terms(new, labels, {name: value[:, :3] for name, value in old.items()})
