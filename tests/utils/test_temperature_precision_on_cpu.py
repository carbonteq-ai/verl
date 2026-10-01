import math

import pytest
import torch

from verl.utils.torch_functional import scale_logits_by_temperature, temperature_scaled_logprobs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [0.8, 1.0, 0.1])
@pytest.mark.parametrize("rowwise", [False, True])
def test_temperature_score_and_derivative_against_scalar_reference(dtype, temperature, rowwise):
    logits = torch.tensor([[-12.0, -4.0, 0.0, 2.0], [10.0, 11.0, 11.0, 10.0]], dtype=dtype, requires_grad=True)
    before = logits.detach().clone()
    temperatures = torch.full((2, 1), temperature, dtype=torch.float64)
    scaled = scale_logits_by_temperature(logits, temperatures)
    scores = (
        temperature_scaled_logprobs(logits, torch.zeros(2, dtype=torch.long), temperatures)
        if rowwise else scaled.log_softmax(-1)[:, 0]
    )
    gradient = torch.autograd.grad(scores.sum(), logits)[0]
    expected_scores = []
    expected_gradients = []
    for row in before.tolist():
        values = [value / temperature for value in row]
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        denominator = math.fsum(weights)
        expected_scores.append(values[0] - maximum - math.log(denominator))
        expected_gradients.append(
            [(int(index == 0) - weight / denominator) / temperature for index, weight in enumerate(weights)]
        )
    tolerance = 2e-5 if dtype != torch.float64 else 1e-12
    assert torch.allclose(scores.double(), torch.tensor(expected_scores, dtype=torch.float64), atol=tolerance, rtol=0)
    assert torch.allclose(gradient, torch.tensor(expected_gradients, dtype=dtype), atol=tolerance, rtol=0)
    assert torch.equal(logits.detach(), before)
    assert scaled.dtype == (torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_temperature_scaling_cannot_overflow_before_normalization(dtype):
    logits = torch.tensor([[60000.0, 60000.0]], dtype=dtype, requires_grad=True)
    scaled = scale_logits_by_temperature(logits, torch.tensor([[0.1]]))
    scores = scaled.log_softmax(-1)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.full_like(scores, -math.log(2)))
    assert torch.isfinite(torch.autograd.grad(scores.sum(), logits)[0]).all()


def test_temperature_floor_does_not_underflow_in_fp16():
    scaled = scale_logits_by_temperature(torch.zeros((2, 3), dtype=torch.float16), torch.tensor([[1e-12], [0.8]]))
    assert torch.isfinite(scaled).all()
    assert torch.equal(scaled, torch.zeros_like(scaled))


def test_rowwise_empty_scores_retain_zero_gradient_path():
    logits = torch.empty((0, 4), dtype=torch.bfloat16, requires_grad=True)
    scores = temperature_scaled_logprobs(logits, torch.empty(0, dtype=torch.long), torch.tensor(0.8))
    assert scores.shape == (0,) and scores.dtype == torch.float32
    assert torch.autograd.grad(scores.sum(), logits)[0].shape == logits.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA temperature precision qualification")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rowwise", [False, True])
def test_cuda_autocast_temperature_score_and_derivative(dtype, rowwise):
    logits = torch.tensor([[-12.0, -4.0, 0.0, 2.0], [60000.0] * 4], device="cuda",
                          dtype=dtype, requires_grad=True)
    temperatures = torch.tensor([[0.8], [0.1]], device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        scores = (
            temperature_scaled_logprobs(logits, torch.zeros(2, device="cuda", dtype=torch.long), temperatures)
            if rowwise else scale_logits_by_temperature(logits, temperatures).log_softmax(-1)[:, 0]
        )
    gradient = torch.autograd.grad(scores.sum(), logits)[0]
    expected_scores = []
    expected_gradients = []
    for row, temperature in zip(logits.detach().cpu().tolist(), temperatures.cpu().flatten().tolist(), strict=True):
        values = [value / temperature for value in row]
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        denominator = math.fsum(weights)
        expected_scores.append(values[0] - maximum - math.log(denominator))
        expected_gradients.append([(int(i == 0) - weight / denominator) / temperature
                                   for i, weight in enumerate(weights)])
    assert scores.dtype == torch.float32
    assert torch.allclose(scores.cpu(), torch.tensor(expected_scores), atol=2e-5, rtol=0)
    assert torch.allclose(gradient.cpu(), torch.tensor(expected_gradients, dtype=dtype), atol=2e-5, rtol=0)
    assert torch.isfinite(gradient).all()
