"""Learning signal (PLAN.md §6 P3): the tiny policy learns the dummy env's target buttons label with PPO.

The learnable signal pays ``signal_reward`` whenever the *executed* buttons label equals
``target_buttons``.  At random init the buttons distribution is close to uniform (top label
about 1.6/728, measured 21 Aug 2026), so the test picks the target as the label with the
highest mean probability on a warm-up rollout (the P2 note in PLAN.md §6) and then asserts that
``mean log pi(target)`` on a fixed batch of frames rises by at least ``log 5`` in 20 updates.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import replace

import pytest
import torch

from melee_rl.config import CONFIG_DIR, finalize_config, load_config, resolve_model_config
from melee_rl.protocol import PolicyProtocol
from melee_rl.train import RunOptions, Trainer
from melee_rl.trajectory import Window

LEARN_TINY = CONFIG_DIR / "learn_tiny.toml"


def _mean_log_prob(policy: PolicyProtocol, window: Window, context: int, target: int) -> float:
    assert window.targets is not None
    with torch.no_grad():
        output = policy.unroll_window(
            window.frames, window.controller_prev, window.targets, window.reset_mask, window.padding_mask
        )
    log_probs = torch.log_softmax(output.logits["buttons"][:, context:], dim=-1)
    return float(log_probs[..., target].mean())


def test_learning_signal_tiny() -> None:
    config = load_config(LEARN_TINY)
    assert config.env.dummy.learnable_signal and config.teacher.source == "none"
    assert config.learner.learning_rate == pytest.approx(3.0e-3)
    assert config.learner.ppo.epsilon == pytest.approx(0.2) and config.learner.ppo.num_epochs == 2
    assert config.env.num_envs == 8 and config.actor.rollout_length == 16
    options = RunOptions(wandb_mode="disabled")

    # Warm-up: the same random init as the trainer below (same seeds), one rollout to choose the target
    # and to fix the evaluation batch.  Building it also pays the one-time torch lazy import.
    warmup = Trainer(config, options)
    trajectory = warmup.worker.rollout()
    probabilities = torch.softmax(trajectory.actor_logits()["buttons"], dim=-1).reshape(-1, 728).mean(0)
    target = int(probabilities.argmax())
    if target != config.env.dummy.target_buttons:
        warnings.warn(
            f"learn_tiny.toml targets buttons label {config.env.dummy.target_buttons}, the most likely "
            f"label of this init is {target}; the test uses {target}",
            stacklevel=1,
        )
    tuned = replace(config, env=replace(config.env, dummy=replace(config.env.dummy, target_buttons=target)))
    tuned = finalize_config(tuned, resolve_model_config(tuned))
    trainer = Trainer(tuned, options)
    window = trajectory.policy_window()
    context = trajectory.context_frames
    before = _mean_log_prob(trainer.policy, window, context, target)
    assert before == pytest.approx(_mean_log_prob(warmup.policy, window, context, target))
    assert before < math.log(8.0 / 728.0)  # nearly uniform at init
    warmup.close()

    started = time.perf_counter()
    result = trainer.run(20)
    elapsed = time.perf_counter() - started
    trainer.close()
    after = _mean_log_prob(trainer.policy, window, context, target)
    assert result.steps_run == 20 and len(result.history) == 20
    assert after - before >= math.log(5.0), (before, after, elapsed)
    assert elapsed < 30.0, elapsed
    assert sum(row["ppo/reverted"] for row in result.history) == 0.0
    first, last = result.history[:5], result.history[-5:]
    hits_first = sum(row["reward/env_per_frame"] for row in first) / len(first)
    hits_last = sum(row["reward/env_per_frame"] for row in last) / len(last)
    assert hits_last > hits_first and hits_last > 5.0 / 728.0
    assert all(math.isfinite(row["loss/total"]) for row in result.history)
    assert all(math.isfinite(row["value/loss"]) for row in result.history)
