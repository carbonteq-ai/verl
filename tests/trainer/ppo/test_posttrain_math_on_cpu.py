"""Independent numerical checks for normalized SAMPO policy and KL losses."""

import math
from decimal import Decimal, localcontext

import pytest
import torch

from verl.trainer.ppo import core_algos
from verl.workers.config import ActorConfig, PolicyLossConfig


@pytest.mark.parametrize("delta", [sign * 10.0**-power for sign in (-1, 1) for power in range(2, 9)])
def test_k3_small_gradient_matches_decimal(delta):
    actor = torch.tensor([0.0], requires_grad=True)
    reference = torch.tensor([delta])
    penalty = core_algos.kl_penalty_forward(actor, reference, "k3_unclipped")
    gradient = torch.autograd.grad(penalty.sum(), actor)[0].item()
    with localcontext() as context:
        context.prec = 80
        actual_delta = Decimal.from_float(reference.item())
        expected = float(1 - actual_delta.exp())
    assert math.isclose(gradient, expected, rel_tol=1e-5, abs_tol=1e-15)


@pytest.mark.parametrize("delta", [-12.0, -0.01, 0.0, 0.01, 12.0])
@pytest.mark.parametrize("invalid", [False, True])
def test_sampo_sequence_ratio_preserves_local_credit(delta, invalid):
    loss_fn = core_algos.get_policy_loss_fn("sampo_token_credit")
    actor = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    old = torch.full_like(actor, -delta)
    advantage = torch.tensor([[1.0, -1.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    if invalid:
        old[0, 2] = float("nan")
        advantage[0, 2] = float("inf")
    config = ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, clip_ratio_low=0.003, clip_ratio_high=0.004
    )
    loss, _ = loss_fn(old, actor, advantage, mask, config=config)
    gradient = torch.autograd.grad(loss, actor)[0]
    ratio = math.exp(float((actor[0, 0] - old[0, 0]).detach()))
    clipped = min(max(ratio, 0.997), 1.004)
    expected_loss = (max(-ratio, -clipped) + max(ratio, clipped)) / 2
    expected_gradient = torch.tensor(
        [
            [
                -ratio / 2 if ratio <= 1.004 else 0.0,
                ratio / 2 if ratio >= 0.997 else 0.0,
                0.0,
            ]
        ]
    )
    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-6, abs_tol=1e-7)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=1e-7)


def test_k3_large_delta_keeps_finite_gradient_when_value_is_finite():
    actor = torch.tensor([0.0], requires_grad=True)
    reference = torch.tensor([80.0])
    penalty = core_algos.kl_penalty_forward(actor, reference, "k3_unclipped")
    gradient = torch.autograd.grad(penalty.sum(), actor)[0]
    assert torch.isfinite(penalty).all()
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("delta", [-0.01, -0.0001, -0.000001, 0.000001, 0.0001, 0.01])
def test_k3_half_scores_promote_before_loss_math(dtype, delta):
    actor = torch.tensor([0.0], dtype=dtype, requires_grad=True)
    reference = torch.tensor([delta], dtype=dtype)
    penalty = core_algos.kl_penalty_forward(actor, reference, "k3_unclipped")
    gradient = torch.autograd.grad(penalty.sum(), actor)[0]
    actual_delta = float(reference.item())
    expected_gradient = torch.tensor([-math.expm1(actual_delta)], dtype=dtype)
    assert penalty.dtype == torch.float32
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float(actual_delta)
        expected_value = float(d.exp() - d - 1)
    assert math.isclose(penalty.item(), expected_value, rel_tol=1e-6, abs_tol=1e-15)


@pytest.mark.parametrize("mode", ["gspo", "sampo_token_credit", "token_clip"])
@pytest.mark.parametrize("invalid_field", ["old", "reference", "current", "credit", "weight", "entropy"])
def test_ppo_wrapper_excludes_invalid_scores_before_loss_math(mode, invalid_field):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    def evaluate(inject):
        current = torch.tensor([[0.01, 0.02, 0.0]], requires_grad=True)
        old = torch.zeros_like(current)
        reference = torch.zeros_like(current)
        credit = torch.tensor([[1.0, -1.0, 0.0]])
        weight = torch.ones_like(current)
        entropy = torch.full_like(current, 0.5)
        if inject:
            field = {"old": old, "reference": reference, "credit": credit, "weight": weight, "entropy": entropy}
            if invalid_field == "current":
                with torch.no_grad():
                    current[0, 2] = float("nan")
            else:
                field[invalid_field][0, 2] = 112.0 if invalid_field == "reference" else float("nan")
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            clip_ratio_low=0.003,
            clip_ratio_high=0.004,
            use_kl_loss=True,
            kl_loss_type="k3_unclipped",
            kl_loss_coef=0.005,
            loss_agg_mode="seq-mean-token-mean",
            entropy_coeff=0.01,
            policy_loss=PolicyLossConfig(loss_mode=mode),
        )
        data = TensorDict(
            {
                "prompts": torch.ones(1, 1, dtype=torch.long),
                "responses": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "response_mask": torch.tensor([[1.0, 1.0, 0.0]]),
                "old_log_probs": old,
                "ref_log_prob": reference,
                "advantages": credit,
                "rollout_is_weights": weight,
            },
            batch_size=[1],
        )
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=1)
        output = {
            "log_probs": torch.cat([current.flatten(), torch.zeros(1)]),
            "entropy": torch.cat([entropy.flatten(), torch.zeros(1)]),
        }
        loss, _ = ppo_loss(config, output, data)
        return loss.detach(), torch.autograd.grad(loss, current)[0]

    expected_loss, expected_gradient = evaluate(False)
    actual_loss, actual_gradient = evaluate(True)
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient[0, 2] == 0
