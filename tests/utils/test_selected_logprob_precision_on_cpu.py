import math

import pytest
import torch

from verl.utils.torch_functional import logprobs_from_logits_v2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset", [0.0, 10000.0, 60000.0, 1e8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_selected_logprob_value_and_gradient_with_common_offset(dtype, offset, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA selected-logprob qualification")
    if dtype == torch.float16 and offset > 65504:
        pytest.skip("offset is outside finite FP16 range")
    logits = torch.tensor([[[offset, offset], [offset, offset + 1]]], dtype=dtype,
                          device=device, requires_grad=True)
    before = logits.detach().clone()
    scores = logprobs_from_logits_v2(logits, torch.tensor([[0, 0]], device=device))
    gradient = torch.autograd.grad(scores.sum(), logits)[0]
    expected_scores = []
    expected_gradients = []
    for row in before[0].tolist():
        maximum = max(row)
        weights = [math.exp(value - maximum) for value in row]
        total = math.fsum(weights)
        expected_scores.append(row[0] - maximum - math.log(total))
        expected_gradients.append([1 - weights[0] / total, -weights[1] / total])
    tolerance = {torch.float64: 1e-12, torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 4e-3}[dtype]
    torch.testing.assert_close(scores.cpu().double(), torch.tensor([expected_scores], dtype=torch.float64),
                               atol=tolerance, rtol=0)
    torch.testing.assert_close(gradient.cpu(), torch.tensor([expected_gradients], dtype=dtype), atol=tolerance, rtol=0)
    assert torch.equal(logits.detach(), before)
    assert scores.dtype == dtype


@pytest.mark.parametrize("shape", [(2, 7), (2, 3, 7)])
def test_selected_logprob_ordinary_logits_and_gradient(shape):
    logits = torch.randn(shape, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.zeros(shape[:-1], dtype=torch.long)
    actual = logprobs_from_logits_v2(logits, labels)
    expected = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits)[0],
                               torch.autograd.grad(expected.sum(), logits)[0])
