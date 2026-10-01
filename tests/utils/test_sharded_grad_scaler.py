"""Finite values, overflow flags and actual CUDA-to-host ordering."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

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


@pytest.mark.parametrize("scale_device", ["cpu", "cuda"])
def test_scaler_resume_after_overflow_preserves_adamw_state(tmp_path, scale_device):
    if scale_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA scaling and NCCL with CPU gradients")
    dist.init_process_group(
        "nccl" if scale_device == "cuda" else "gloo",
        init_method="file://" + str(tmp_path / "rendezvous"),
        rank=0,
        world_size=1,
    )
    try:

        def build():
            parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
            optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.01)
            scaler = CPUOffloadShardedGradScaler(device=scale_device, init_scale=1024, growth_interval=2)
            return parameter, optimizer, scaler

        def advance(state, step):
            parameter, optimizer, scaler = state
            optimizer.zero_grad(set_to_none=True)
            before = parameter.detach().clone()
            gradient = torch.tensor([0.1, -0.2]) * (step + 1)
            scaler.scale((parameter * gradient).sum().to(scale_device)).backward()
            if step == 1:
                parameter.grad[0] = float("inf")
            if scale_device == "cuda":
                torch.cuda._sleep(10_000_000)
            scaler.unscale_(optimizer)
            if step != 1:
                torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
            scaler.step(optimizer)
            scaler.update()
            if step == 1:
                torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                assert int(optimizer.state[parameter]["step"]) == 1

        uninterrupted = build()
        resumed = build()
        for step in range(4):
            advance(uninterrupted, step)
            advance(resumed, step)
            if step == 1:
                parameter, optimizer, scaler = resumed
                checkpoint = deepcopy((parameter.detach(), optimizer.state_dict(), scaler.state_dict()))
                resumed = build()
                with torch.no_grad():
                    resumed[0].copy_(checkpoint[0])
                resumed[1].load_state_dict(checkpoint[1])
                resumed[2].load_state_dict(checkpoint[2])
            torch.testing.assert_close(uninterrupted[0], resumed[0], rtol=0, atol=0)
            assert uninterrupted[2].state_dict() == resumed[2].state_dict()
        assert uninterrupted[2].get_scale() == 1024
        expected_state = uninterrupted[1].state[uninterrupted[0]]
        actual_state = resumed[1].state[resumed[0]]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _distributed_overflow_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        parameter = torch.nn.Parameter(torch.tensor([0.25]))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        scaler = CPUOffloadShardedGradScaler(device="cpu", init_scale=1024)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(parameter.sum() * 0.25).backward()
            if step == 0 and rank == 0:
                parameter.grad[0] = float("inf")
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            expected = torch.tensor([0.25 if step == 0 else 0.2475])
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            assert scaler.get_scale() == 512
    finally:
        dist.destroy_process_group()


def test_distributed_cpu_overflow_skips_every_rank(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_overflow_worker, args=(str(tmp_path / "distributed_rendezvous"),), nprocs=2, join=True
    )
