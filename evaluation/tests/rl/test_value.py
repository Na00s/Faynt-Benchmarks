"""Value network (PLAN.md §3.7, §6 P2): shapes, targets on a hand case, burn-in on a fixed batch, 3m size."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.frames import neutral_frame, neutral_labels, random_valid_frames
from melee_rl.rl_lib import ReturnsConfig
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueConfig, ValueNet, value_model_config, value_targets
from model import ModelConfig

B = 3
C = 16
T = 8


def _hand_trajectory(*, reset_row: int | None = None) -> Trajectory:
    """B = 1, C = 2, T = 4, D = 0: rewards 1..4, an optional game start at rollout row ``reset_row``."""

    context, length = 2, 4
    window = context + length + 1
    is_resetting = torch.zeros((1, window), dtype=torch.bool)
    is_resetting[0, 0] = True
    if reset_row is not None:
        is_resetting[0, context + reset_row] = True
    generator = torch.Generator().manual_seed(0)
    return Trajectory(
        frames=neutral_frame((1, window)),
        controller_exec=neutral_labels((1, window)),
        controller_prev_sample=neutral_labels((1, window)),
        sampled_labels=neutral_labels((1, window)),
        sampled_logits={
            "buttons": torch.randn((1, length, 728), generator=generator),
            "main_stick": torch.randn((1, length, 85), generator=generator),
        },
        rewards=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        env_rewards=torch.zeros((1, length)),
        is_resetting=is_resetting,
        padding=torch.ones((1, window), dtype=torch.bool),
        frame_index=torch.arange(window).unsqueeze(0) - 123,
        context_frames=context,
        rollout_length=length,
        delay_frames=0,
    )


def test_value_net_shapes_padding_and_grads(tiny_config: ModelConfig, tiny_value_config: ValueConfig) -> None:
    torch.manual_seed(0)
    net = ValueNet(tiny_config, tiny_value_config)
    assert net.context_length == tiny_config.context_length + 1 == 25
    assert net.model_config.d_model == 32 and net.model_config.compute_dtype == "float32"
    assert net.model_config.gradient_checkpointing is False
    length = 9
    frames = random_valid_frames((B, length), torch.Generator().manual_seed(1))
    labels = neutral_labels((B, length))
    reset = torch.zeros((B, length), dtype=torch.bool)
    reset[:, 0] = True
    padding = torch.ones((B, length), dtype=torch.bool)
    padding[0, :3] = False  # env 0 has only 6 real rows
    values = net(frames, labels, reset, padding)
    assert values.shape == (B, length) and values.dtype == torch.float32
    assert bool(torch.isfinite(values).all())
    # The head starts at zero, so every value is 0 at init; a padded row stays at the bias.
    assert float(values.detach().abs().max()) == 0.0
    values.sum().backward()
    assert net.head.weight.grad is not None and float(net.head.weight.grad.abs().sum()) > 0.0
    # After a head change the padded rows still equal the bias (hidden is zeroed there, model.py:1496).
    with torch.no_grad():
        net.head.weight.normal_(0.0, 0.1)
        net.head.bias.fill_(0.25)
    values = net(frames, labels, reset, padding)
    assert torch.allclose(values[0, :3], torch.full((3,), 0.25))
    assert not torch.allclose(values[0, 3:], torch.full((6,), 0.25))
    with pytest.raises(ValueError, match="exceeds"):
        too_long = random_valid_frames((B, 26), torch.Generator().manual_seed(2))
        flags = torch.zeros((B, 26), dtype=torch.bool)
        net(too_long, neutral_labels((B, 26)), flags, ~flags)
    with pytest.raises(ValueError, match="value d_model"):
        ValueConfig(d_model=30, n_heads=4, head_dim=8)
    with pytest.raises(ValueError, match="n_kv_heads"):
        ValueConfig(n_heads=2, n_kv_heads=3, head_dim=64, d_model=128)
    assert value_model_config(tiny_config, ValueConfig(context_length=40)).context_length == 40


def test_value_targets_hand_case_with_bootstrap_and_reset() -> None:
    config = ReturnsConfig(reward_halflife=1.0 / 60.0)  # discount 0.5 per frame
    values = torch.tensor([[9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 5.0]], requires_grad=True)
    outputs = value_targets(_hand_trajectory(), values, config)
    # G_3 = 4 + .5 * 5, G_2 = 3 + .5 * G_3, ...
    torch.testing.assert_close(outputs.returns, torch.tensor([[3.5625, 5.125, 6.25, 6.5]]))
    assert outputs.values.shape == (1, 5) and outputs.values.requires_grad
    assert float(outputs.bootstrap.detach()) == 5.0 and not outputs.returns.requires_grad
    torch.testing.assert_close(outputs.advantages, outputs.returns)  # values are 0 on the rollout rows
    assert outputs.discounts.tolist() == [[0.5] * 4]
    assert float(outputs.loss().detach()) == pytest.approx(float((outputs.returns**2).mean()))
    metrics = outputs.metrics()
    assert set(metrics) == {"value/loss", "value/uev", "value/return_mean", "value/value_mean"}
    assert metrics["value/value_mean"] == 0.0 and metrics["value/return_mean"] == pytest.approx(5.359375)
    outputs.loss().backward()
    assert values.grad is not None and float(values.grad[0, 6]) == 0.0  # the bootstrap is detached
    # A game start at rollout row 2 cuts the discount of transition 1 -> 2.
    cut = value_targets(_hand_trajectory(reset_row=2), values.detach(), config)
    assert cut.discounts.tolist() == [[0.5, 0.0, 0.5, 0.5]]
    torch.testing.assert_close(cut.returns, torch.tensor([[2.0, 2.0, 6.25, 6.5]]))
    with pytest.raises(ValueError, match="values_window"):
        value_targets(_hand_trajectory(), values[:, :6], config)


def test_value_burnin_decreases_loss_on_a_fixed_batch(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: Callable[..., list[Trajectory]],
) -> None:
    # Real rollouts (frames, resets, padding) with a synthetic, learnable reward: the toy games
    # rarely score in their first frames, and the test needs returns the frames can explain.
    trajectories = [
        replace(trajectory, rewards=0.01 * trajectory.frames["p0"]["x"][:, C + 1 :].float())
        for trajectory in rollout_trajectories(tiny_adapter, 2)
    ]
    torch.manual_seed(0)
    net = ValueNet(tiny_config, tiny_value_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-2)
    config = ReturnsConfig(reward_halflife=0.5)
    losses: list[float] = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(())
        for trajectory in trajectories:
            window = trajectory.value_window()
            outputs = value_targets(trajectory, net.forward_window(window), config)
            loss = outputs.loss()
            loss.backward()
            total = total + loss.detach()
        optimizer.step()
        losses.append(float(total) / len(trajectories))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[0] > 0.0 and losses[-1] < 0.5 * losses[0], losses
    assert min(losses[-3:]) < min(losses[:3])


def test_value_net_3m_profile_size(three_m_config: ModelConfig) -> None:
    torch.manual_seed(0)
    net = ValueNet(three_m_config)
    count = sum(parameter.numel() for parameter in net.parameters())
    assert 2_000_000 < count < 2_600_000, count  # PLAN.md §3.7: about 2.3 M
    assert net.context_length == 257
    frames = random_valid_frames((2, 9), torch.Generator().manual_seed(4))
    flags = torch.zeros((2, 9), dtype=torch.bool)
    with torch.no_grad():
        values = net(frames, neutral_labels((2, 9)), flags, ~flags)
    assert values.shape == (2, 9) and bool(torch.isfinite(values).all())
