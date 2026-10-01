# Copyright 2026 CarbonTeq
# Licensed under the Apache License, Version 2.0.
"""Loss scaling with ordered scalar copies for CPU-offloaded gradients."""

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


class CPUOffloadShardedGradScaler(ShardedGradScaler):
    """Keep host unscale kernels behind CUDA scalar computation/copies.

    PyTorch's scalar replicator uses nonblocking copies. A CPU foreach kernel
    can consume a CUDA-to-CPU inverse-scale copy before it is complete. Stage
    both scalars synchronously on the host when gradients have host storage;
    leave device-only execution and distributed overflow reduction unchanged.
    """

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16=True):
        def has_cpu_storage(gradient):
            local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
            return local.device.type == "cpu"

        if any(
            parameter.grad is not None and has_cpu_storage(parameter.grad)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            inv_scale = inv_scale.to(device="cpu", non_blocking=False)
            found_inf = found_inf.to(device="cpu", non_blocking=False)
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16)
