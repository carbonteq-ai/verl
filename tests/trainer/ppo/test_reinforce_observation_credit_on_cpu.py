# Copyright 2026 CarbonTeq
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Observation insertion must not alter action-time REINFORCE++ credit."""

import pytest
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import compute_reinforce_plus_plus_outcome_advantage


@pytest.mark.parametrize("gamma", [1.0, 0.7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_observation_insertion_preserves_action_returns_and_advantages(gamma, dtype):
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, -0.5, 0.0, 2.0]], dtype=dtype)
    mask = torch.ones_like(rewards)
    expected_advantages, expected_returns = compute_reinforce_plus_plus_outcome_advantage(
        rewards, mask, config=AlgoConfig(gamma=gamma)
    )
    positions = [0, 1, 4, 5]
    expanded_rewards = torch.zeros((2, 8), dtype=dtype)
    expanded_rewards[:, positions] = rewards
    expanded_mask = torch.zeros_like(expanded_rewards)
    expanded_mask[:, positions] = 1
    advantages, returns = compute_reinforce_plus_plus_outcome_advantage(
        expanded_rewards, expanded_mask, config=AlgoConfig(gamma=gamma)
    )
    torch.testing.assert_close(returns[:, positions], expected_returns)
    torch.testing.assert_close(advantages[:, positions], expected_advantages)
    assert not torch.count_nonzero(returns[expanded_mask == 0])
    assert not torch.count_nonzero(advantages[expanded_mask == 0])
    # An independent action-time discounted return, not just self-consistency.
    assert returns[0, 0].item() == pytest.approx(gamma**3)


def test_masked_reward_does_not_contaminate_sampled_action_credit():
    rewards = torch.tensor([[0.0, 100.0, 1.0, 100.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    _, returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=AlgoConfig(gamma=0.5))
    torch.testing.assert_close(returns, torch.tensor([[0.5, 0.0, 1.0, 0.0]]))
