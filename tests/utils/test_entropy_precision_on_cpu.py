"""Entropy must depend on probabilities rather than a common logit offset."""

import math

import pytest
import torch

from verl.utils.torch_functional import entropy_from_logits, entropy_from_logits_with_chunking


def scalar_reference(values):
    centered = [v - max(values) for v in values]
    denominator = math.fsum(math.exp(v) for v in centered)
    log_prob = [v - math.log(denominator) for v in centered]
    probabilities = [math.exp(v) for v in log_prob]
    entropy = -math.fsum(p * lp for p, lp in zip(probabilities, log_prob, strict=True))
    gradient = [-p * (lp + entropy) for p, lp in zip(probabilities, log_prob, strict=True)]
    return entropy, gradient


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("offset", [0.0, 10.0, 1000.0, 10000.0])
@pytest.mark.parametrize("chunked", [False, True])
def test_entropy_values_and_gradients_use_represented_logits(dtype, offset, chunked):
    logits = torch.tensor([[offset, offset + 1], [offset, offset]], dtype=dtype, requires_grad=True)
    function = entropy_from_logits_with_chunking if chunked else entropy_from_logits
    entropy = function(logits, chunk_size=1) if chunked else function(logits)
    gradient = torch.autograd.grad(entropy.sum(), logits)[0]
    expected = [scalar_reference(row) for row in logits.detach().double().tolist()]
    tolerance = 2e-6 if dtype != torch.float64 else 1e-12
    expected_values = torch.tensor([v[0] for v in expected], dtype=torch.float64)
    assert torch.allclose(entropy.double(), expected_values, atol=tolerance, rtol=0)
    rounded_gradient = torch.tensor([v[1] for v in expected], dtype=dtype)
    assert torch.allclose(gradient, rounded_gradient, atol=tolerance, rtol=0)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("chunked", [False, True])
def test_entropy_preserves_uniform_probability_under_large_float32_offset(chunked):
    logits = torch.full((2, 4), 1e8, requires_grad=True)
    entropy = entropy_from_logits_with_chunking(logits, 1) if chunked else entropy_from_logits(logits)
    gradient = torch.autograd.grad(entropy.sum(), logits)[0]
    assert torch.allclose(entropy, torch.full((2,), math.log(4)), atol=2e-6, rtol=0)
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("chunked", [False, True])
def test_entropy_zero_probability_has_finite_value_and_gradient(chunked):
    logits = torch.tensor([[3e38, -3e38]], requires_grad=True)
    entropy = entropy_from_logits_with_chunking(logits, 1) if chunked else entropy_from_logits(logits)
    gradient = torch.autograd.grad(entropy.sum(), logits)[0]
    assert torch.equal(entropy, torch.zeros_like(entropy))
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA entropy precision qualification")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("chunked", [False, True])
def test_cuda_entropy_value_and_gradient(dtype, chunked):
    logits = torch.tensor([[10.0, 11.0], [10000.0, 10000.0]], device="cuda", dtype=dtype, requires_grad=True)
    entropy = entropy_from_logits_with_chunking(logits, 1) if chunked else entropy_from_logits(logits)
    gradient = torch.autograd.grad(entropy.sum(), logits)[0]
    expected = [scalar_reference(row) for row in logits.detach().double().cpu().tolist()]
    expected_values = torch.tensor([v[0] for v in expected], dtype=torch.float64)
    assert torch.allclose(entropy.cpu().double(), expected_values, atol=2e-6, rtol=0)
    rounded_gradient = torch.tensor([v[1] for v in expected], dtype=dtype)
    assert torch.allclose(gradient.cpu(), rounded_gradient, atol=2e-6, rtol=0)
    assert torch.isfinite(gradient).all()
