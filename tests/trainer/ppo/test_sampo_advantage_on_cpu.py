# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""CPU contract tests for SAMPO's hierarchical advantage estimator."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_sampo_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage


def _config(*, normalization: str = "mean"):
    return SimpleNamespace(
        sampo=SimpleNamespace(
            discount_gamma=0.5,
            step_advantage_weight=1.0,
            advantage_normalization=normalization,
        )
    )


def test_sampo_combines_episode_and_anchor_relative_turn_advantages() -> None:
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 3.0]])
    mask = torch.ones_like(rewards)
    metrics = {}
    advantages, returns = compute_sampo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["prompt", "prompt"], dtype=object),
        turn_spans=np.array([[(0, 2), (2, 4)], [(0, 2), (2, 4)]], dtype=object),
        anchor_state_keys=np.array([["start", "tool"], ["start", "tool"]], dtype=object),
        step_rewards=np.array([[None, None], [None, None]], dtype=object),
        num_repeat=2,
        config=_config(),
        metrics=metrics,
    )

    expected = torch.tensor([[-1.5, -1.5, -2.0, -2.0], [1.5, 1.5, 2.0, 2.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)
    assert metrics == pytest.approx(
        {
            "sampo/episode_advantage_mean": 0.0,
            "sampo/turn_advantage_mean": 0.0,
            "sampo/anchor_group_size_mean": 2.0,
            "sampo/sparse_reward_projection_fraction": 1.0,
        }
    )


def test_sampo_uses_explicit_step_rewards_and_masks_non_policy_tokens() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    advantages, _ = compute_sampo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["prompt", "prompt"], dtype=object),
        turn_spans=np.array([[(0, 1), (2, 3)], [(0, 1), (2, 3)]], dtype=object),
        anchor_state_keys=np.array([["start", "tool"], ["start", "tool"]], dtype=object),
        step_rewards=np.array([[1.0, 0.0], [3.0, 0.0]], dtype=object),
        num_repeat=2,
        config=_config(),
    )

    expected = torch.tensor([[-2.0, 0.0, -1.0], [2.0, 0.0, 1.0]])
    torch.testing.assert_close(advantages, expected)


def test_compute_advantage_routes_agent_loop_metadata_to_sampo() -> None:
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 3.0]]),
            "response_mask": torch.ones((2, 2)),
        },
        non_tensors={
            "uid": np.array(["prompt", "prompt"], dtype=object),
            "sampo_turn_spans": np.array([[(0, 2)], [(0, 2)]], dtype=object),
            "sampo_anchor_state_keys": np.array([["start"], ["start"]], dtype=object),
            "sampo_step_rewards": np.array([[None], [None]], dtype=object),
        },
    )

    result = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.SAMPO,
        num_repeat=2,
        config=_config(),
    )

    torch.testing.assert_close(
        result.batch["advantages"],
        torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
    )
    assert result.meta_info["sampo_metrics"] == pytest.approx(
        {
            "sampo/episode_advantage_mean": 0.0,
            "sampo/turn_advantage_mean": 0.0,
            "sampo/anchor_group_size_mean": 2.0,
            "sampo/sparse_reward_projection_fraction": 1.0,
        }
    )


def test_compute_advantage_rejects_missing_sampo_metadata() -> None:
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 1.0]]),
            "response_mask": torch.ones((1, 2)),
        },
        non_tensors={"uid": np.array(["prompt"], dtype=object)},
    )

    with pytest.raises(ValueError, match="SAMPO rollout metadata is missing"):
        compute_advantage(
            data,
            adv_estimator=AdvantageEstimator.SAMPO,
            config=_config(),
        )


@pytest.mark.parametrize(
    ("spans", "anchors", "step_rewards", "message"),
    [
        ([], [], [], "non-empty"),
        ([(0, 1)], [""], [None], "cannot be empty"),
        ([(0, 2)], ["start"], [None], "sampled policy tokens"),
        ([(0, 1)], ["start"], [None], "cover every sampled"),
        ([(2, 3), (0, 1)], ["a", "b"], [None, None], "ordered"),
        ([(0, 1), (2, 3)], ["a", "b"], [None, 1.0], "complete or entirely absent"),
    ],
)
def test_sampo_rejects_invalid_turn_metadata(spans, anchors, step_rewards, message) -> None:
    with pytest.raises(ValueError, match=message):
        compute_sampo_outcome_advantage(
            token_level_rewards=torch.tensor([[0.0, 0.0, 1.0]]),
            response_mask=torch.tensor([[1.0, 0.0, 1.0]]),
            index=np.array(["prompt"], dtype=object),
            turn_spans=np.array([spans], dtype=object),
            anchor_state_keys=np.array([anchors], dtype=object),
            step_rewards=np.array([step_rewards], dtype=object),
            config=_config(),
        )


def test_sampo_rejects_incomplete_prompt_groups() -> None:
    with pytest.raises(
        ValueError,
        match=r"complete prompt groups matching rollout.n; expected=2, observed_group_sizes=\[1\]",
    ):
        compute_sampo_outcome_advantage(
            token_level_rewards=torch.tensor([[0.0, 1.0]]),
            response_mask=torch.ones((1, 2)),
            index=np.array(["prompt"], dtype=object),
            turn_spans=np.array([[(0, 2)]], dtype=object),
            anchor_state_keys=np.array([["start"]], dtype=object),
            step_rewards=np.array([[None]], dtype=object),
            num_repeat=2,
            config=_config(),
        )
