"""Finite values, overflow flags and actual CUDA-to-host ordering."""

import pytest
import torch

from verl.utils.sharded_grad_scaler import CPUOffloadShardedGradScaler


@pytest.mark.parametrize("nonfinite", [False, True])
def test_cpu_gradient_unscale_and_overflow(nonfinite):
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([1024.0, -2048.0, 512.0, float("inf") if nonfinite else 0.0])
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = CPUOffloadShardedGradScaler(device="cpu")
    flags = scaler._unscale_grads_(optimizer, torch.tensor(1 / 1024), torch.tensor(0.0))
    torch.testing.assert_close(parameter.grad[:3], torch.tensor([1.0, -2.0, 0.5]), rtol=0, atol=0)
    assert float(flags[torch.device("cpu")]) == float(nonfinite)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-to-CPU copy ordering")
@pytest.mark.parametrize("nonfinite", [False, True])
def test_delayed_cuda_inverse_scale_is_ready_for_cpu_kernel(nonfinite):
    parameter = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = CPUOffloadShardedGradScaler()
    expected = torch.tensor([0.01, -0.02, 0.03, -0.04])
    if nonfinite:
        expected[-1] = float("inf")
    for _ in range(4):
        parameter.grad = expected * 1024
        scale = torch.tensor(1024.0, device="cuda")
        torch.cuda._sleep(10_000_000)
        inverse = scale.double().reciprocal().float()
        flags = scaler._unscale_grads_(optimizer, inverse, torch.zeros(1, device="cuda"))
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
        assert float(flags[torch.device("cpu")]) == float(nonfinite)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires device-only unscale")
def test_device_only_gradients_keep_native_scaling():
    parameter = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    expected = torch.tensor([1.0, -2.0, 0.5, 0.0], device="cuda")
    parameter.grad = expected * 1024
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = CPUOffloadShardedGradScaler()
    flags = scaler._unscale_grads_(optimizer, torch.tensor(1 / 1024, device="cuda"), torch.tensor(0.0, device="cuda"))
    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    assert float(flags[torch.device("cuda:0")]) == 0.0
