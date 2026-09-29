"""Returns and discounts (PLAN.md §6 P2): slippi-ai hand case, batching, reset zeroing, discount on death."""

from __future__ import annotations

import pytest
import torch

from melee_rl.frames import neutral_frame
from melee_rl.rl_lib import (
    ACTION_ON_HALO_DESCENT,
    ReturnsConfig,
    discount_from_halflife,
    discounted_returns,
    mean_and_variance,
    player_respawn,
    respawn_happened,
    unexplained_variance,
    value_discounts,
)


def test_discounted_returns_hand_case() -> None:
    """``tf/tests/rl_lib_test.py:9-17``: rewards [1,2,3], discounts [1,.5,.25], bootstrap 4 -> [5, 4, 4]."""

    rewards = torch.tensor([1.0, 2.0, 3.0])
    discounts = torch.tensor([1.0, 0.5, 0.25])
    returns = discounted_returns(rewards, discounts, 4.0)
    torch.testing.assert_close(returns, torch.tensor([5.0, 4.0, 4.0]))
    assert returns.dtype == torch.float32


def test_discounted_returns_batched_matches_loop_and_broadcasts() -> None:
    generator = torch.Generator().manual_seed(3)
    rewards = torch.randn((4, 7), generator=generator)
    discounts = torch.rand((4, 7), generator=generator)
    bootstrap = torch.randn(4, generator=generator)
    returns = discounted_returns(rewards, discounts, bootstrap)
    assert returns.shape == (4, 7)
    for b in range(4):
        acc = float(bootstrap[b])
        for t in reversed(range(7)):
            acc = float(rewards[b, t]) + float(discounts[b, t]) * acc
            assert abs(float(returns[b, t]) - acc) < 1.0e-5, (b, t)
    # A scalar discount broadcasts; a scalar bootstrap too; an empty horizon returns an empty tensor.
    scalar = discounted_returns(rewards, 0.9, 0.0)
    reference = discounted_returns(rewards, torch.full((4, 7), 0.9), torch.zeros(4))
    torch.testing.assert_close(scalar, reference)
    assert discounted_returns(rewards[:, :0], 0.9, bootstrap).shape == (4, 0)
    # Gradients flow through the bootstrap when it is not detached (the caller detaches).
    value = torch.tensor(2.0, requires_grad=True)
    discounted_returns(torch.ones(3), 0.5, value).sum().backward()
    assert value.grad is not None and abs(float(value.grad) - (0.125 + 0.25 + 0.5)) < 1.0e-6


def test_discount_from_halflife_and_config() -> None:
    assert abs(discount_from_halflife(4.0) - 0.99712) < 1.0e-5
    assert abs(discount_from_halflife(8.0) - 0.99856) < 1.0e-5
    assert discount_from_halflife(1.0 / 60.0) == pytest.approx(0.5)
    assert ReturnsConfig().discount == discount_from_halflife(4.0)
    with pytest.raises(ValueError, match="positive"):
        discount_from_halflife(0.0)
    with pytest.raises(ValueError, match="reward_halflife"):
        ReturnsConfig(reward_halflife=0.0)
    with pytest.raises(ValueError, match="discount_on_death"):
        ReturnsConfig(discount_on_death=1.5)


def test_value_discounts_zero_at_resets_and_on_death() -> None:
    rewards = torch.zeros((2, 4))
    is_resetting = torch.zeros((2, 5), dtype=torch.bool)
    is_resetting[0, 3] = True  # env 0: a new game starts at row 3 -> transition 2 -> 3 is cut
    config = ReturnsConfig(reward_halflife=1.0 / 60.0)  # discount 0.5
    discounts = value_discounts(config, rewards, is_resetting=is_resetting)
    assert discounts[0].tolist() == [0.5, 0.5, 0.0, 0.5]
    assert discounts[1].tolist() == [0.5] * 4
    # discount_on_death applies on respawn transitions of either player; a reset still wins.
    frames = neutral_frame((2, 5))
    frames["p0"]["action"][1, 2] = ACTION_ON_HALO_DESCENT  # env 1: p0 respawns on transition 1 -> 2
    frames["p1"]["action"][0, 3] = ACTION_ON_HALO_DESCENT  # env 0: p1 respawns on 2 -> 3 (a reset row)
    death = ReturnsConfig(reward_halflife=1.0 / 60.0, discount_on_death=0.25)
    discounts = value_discounts(death, rewards, is_resetting=is_resetting, frames=frames)
    assert discounts[1].tolist() == [0.5, 0.25, 0.5, 0.5]
    assert discounts[0].tolist() == [0.5, 0.5, 0.0, 0.5]
    # Without reset zeroing the death discount shows through.
    no_zero = ReturnsConfig(reward_halflife=1.0 / 60.0, discount_on_death=0.25, zero_discount_on_reset=False)
    assert value_discounts(no_zero, rewards, frames=frames)[0].tolist() == [0.5, 0.5, 0.25, 0.5]
    with pytest.raises(ValueError, match="rollout frames"):
        value_discounts(death, rewards, is_resetting=is_resetting)
    with pytest.raises(ValueError, match="is_resetting"):
        value_discounts(config, rewards)
    with pytest.raises(ValueError, match="shape"):
        value_discounts(config, rewards, is_resetting=is_resetting[:, :4])


def test_player_respawn_edges() -> None:
    action = torch.tensor([[14, 12, 12, 3, 12, 14]])
    assert player_respawn(action).tolist() == [[True, False, False, True, False]]
    frames = neutral_frame((1, 6))
    frames["p1"]["action"][0] = action[0]
    assert torch.equal(respawn_happened(frames), player_respawn(action))


def test_unexplained_variance_and_moments() -> None:
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mean, variance = mean_and_variance(targets)
    assert float(mean) == 2.5 and float(variance) == 1.25
    assert float(unexplained_variance(torch.zeros(4), targets)) == 0.0
    # Predicting the mean everywhere leaves all the variance unexplained (uev ~ 1).
    squared = (targets - 2.5) ** 2
    assert abs(float(unexplained_variance(squared, targets)) - 1.0) < 1.0e-6
