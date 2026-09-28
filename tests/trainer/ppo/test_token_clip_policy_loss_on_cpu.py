# Copyright 2026 CarbonTeq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The OLMo 3 / DAPO token-clip objective and the unclipped k3 KL estimator."""

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_vanilla,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.workers.config.actor import ActorConfig


def _config(low: float = 0.2, high: float = 0.272, clip_ratio_c: float = 3.0) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=2,
        clip_ratio=0.2,
        clip_ratio_low=low,
        clip_ratio_high=high,
        clip_ratio_c=clip_ratio_c,
    )


def _reference_loss(old, new, adv, mask, low, high, weights=None):
    """TRL GRPOTrainer._compute_loss for loss_type='dapo', token-level ratios, one process."""
    coef_1 = torch.exp(new - old)
    coef_2 = torch.clamp(coef_1, 1 - low, 1 + high)
    per_token = -torch.min(coef_1 * adv, coef_2 * adv)
    if weights is not None:
        per_token = per_token * weights
    return (per_token * mask).sum() / mask.sum()


def _batch():
    generator = torch.Generator().manual_seed(7)
    old = -torch.rand(4, 6, generator=generator, dtype=torch.float64) * 3
    # Ratios from exp(-1.5) to exp(1.5): below the lower bound, inside, above the upper bound,
    # and above dual-clip's 3.0 cap for negative advantages.
    new = old + torch.linspace(-1.5, 1.5, 24, dtype=torch.float64).reshape(4, 6)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.75], dtype=torch.float64).unsqueeze(1).expand(4, 6).clone()
    mask = torch.ones(4, 6, dtype=torch.float64)
    mask[0, 4:] = 0
    mask[2, 1] = 0
    return old, new, advantages, mask


def test_token_clip_is_registered():
    assert get_policy_loss_fn("token_clip").__name__ == "compute_policy_loss_token_clip"


def test_token_clip_matches_trl_asymmetric_clip_with_token_mean():
    old, new, advantages, mask = _batch()
    new = new.clone().requires_grad_(True)
    loss, metrics = get_policy_loss_fn("token_clip")(
        old_log_prob=old, log_prob=new, advantages=advantages, response_mask=mask, config=_config()
    )
    expected = _reference_loss(old, new, advantages, mask, 0.2, 0.272)
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-12)
    (grad,) = torch.autograd.grad(loss, new)
    (expected_grad,) = torch.autograd.grad(_reference_loss(old, new, advantages, mask, 0.2, 0.272), new)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-12)
    assert 0 < metrics["actor/pg_clipfrac"] < 1
    assert metrics["actor/pg_clipfrac_lower"] == 0.0


def test_token_clip_has_no_dual_clip_unlike_vanilla():
    old, new, advantages, mask = _batch()
    config = _config()
    token_clip, _ = get_policy_loss_fn("token_clip")(
        old_log_prob=old, log_prob=new, advantages=advantages, response_mask=mask, config=config
    )
    vanilla, _ = compute_policy_loss_vanilla(
        old_log_prob=old, log_prob=new, advantages=advantages, response_mask=mask, config=config
    )
    # Row 1 has a negative advantage and ratios above 3, which vanilla caps at -A * 3.
    assert token_clip > vanilla


def test_token_clip_applies_rollout_correction_weights_per_token():
    old, new, advantages, mask = _batch()
    weights = torch.linspace(0.25, 2.0, 24, dtype=torch.float64).reshape(4, 6)
    loss, _ = get_policy_loss_fn("token_clip")(
        old_log_prob=old,
        log_prob=new,
        advantages=advantages,
        response_mask=mask,
        config=_config(),
        rollout_is_weights=weights,
    )
    torch.testing.assert_close(
        loss, _reference_loss(old, new, advantages, mask, 0.2, 0.272, weights), rtol=0, atol=1e-12
    )


def test_token_clip_uses_global_token_count_when_provided():
    old, new, advantages, mask = _batch()
    config = _config()
    config.global_batch_info.update({"dp_size": 1, "batch_num_tokens": 40, "global_batch_size": None})
    loss, _ = get_policy_loss_fn("token_clip")(
        old_log_prob=old, log_prob=new, advantages=advantages, response_mask=mask, config=config
    )
    local = _reference_loss(old, new, advantages, mask, 0.2, 0.272)
    torch.testing.assert_close(loss, local * mask.sum() / 40, rtol=0, atol=1e-12)


def test_token_clip_rejects_non_finite_ratios():
    old, new, advantages, mask = _batch()
    new = new.clone()
    new[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        get_policy_loss_fn("token_clip")(
            old_log_prob=old, log_prob=new, advantages=advantages, response_mask=mask, config=_config()
        )


def test_k3_unclipped_matches_trl_and_is_not_clamped():
    logprob = torch.tensor([-0.5, -4.0, -30.0, -0.1], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([-0.4, -0.2, -0.1, -25.0], dtype=torch.float64)
    value = kl_penalty(logprob, ref, "k3_unclipped")
    expected = torch.exp(ref - logprob) - (ref - logprob) - 1
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    # low_var_kl clamps the estimate at 10; the unclipped estimator keeps large drift.
    assert value[1] > 10 and kl_penalty(logprob, ref, "low_var_kl")[1] == 10
    (grad,) = torch.autograd.grad(value.sum(), logprob)
    torch.testing.assert_close(grad, 1 - torch.exp(ref - logprob.detach()), rtol=1e-12, atol=1e-12)
