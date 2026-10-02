"""Opt-in Linear arithmetic controls; parameter names and optimizer ownership stay native."""

import torch
from torch import nn
from torch.nn import functional as F


class FP32AdapterLinear(nn.Linear):
    """A LoRA leaf whose arithmetic stays FP32 inside a mixed-precision backbone."""

    rowwise_compute = False

    def forward(self, inputs):
        if self.weight.dtype != torch.float32:
            raise RuntimeError("FP32 adapter compute requires FP32 native parameter views")
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            if self.rowwise_compute and inputs.ndim == 3:
                if inputs.shape[0] == 0:
                    raise ValueError("Rowwise adapter compute requires a nonempty batch")
                return torch.stack([F.linear(row.float(), self.weight, self.bias) for row in inputs], dim=0)
            return F.linear(inputs.float(), self.weight, self.bias)


def prepare_fp32_lora_leaves(module, *, rowwise_compute=False):
    """Replace ordinary PEFT A/B leaves without changing values, identities or state keys.

    FSDP must separately wrap these leaves with mixed precision disabled. Merely
    excluding them from synchronization would violate the distributed contract.
    Unsupported specialized leaves fail rather than silently losing their forward.
    """
    leaves = []
    for name, child in module.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] in ("lora_A", "lora_B"):
            if type(child) is not nn.Linear:
                raise ValueError(f"FP32 LoRA compute requires ordinary Linear leaves: {name}")
            if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks:
                raise ValueError(f"Cannot replace a LoRA leaf with existing module hooks: {name}")
            if any(parameter.dtype != torch.float32 for parameter in child.parameters()):
                raise ValueError(f"FP32 LoRA compute requires FP32 initialized adapter parameters: {name}")
            leaves.append((name, child))
    if not leaves:
        raise ValueError("FP32 LoRA compute requested without any LoRA A/B Linear leaves")
    for name, child in leaves:
        replacement = FP32AdapterLinear(child.in_features, child.out_features, child.bias is not None, device="meta")
        replacement.weight = child.weight
        replacement.bias = child.bias
        replacement.rowwise_compute = rowwise_compute
        replacement.train(child.training)
        parent_name, _, leaf_name = name.rpartition(".")
        module.get_submodule(parent_name)._modules[leaf_name] = replacement
    return tuple(name for name, _ in leaves)


def register_contiguous_linear_output_gradients(module):
    """Preserve cotangent values while fixing the GEMM input layout at each Linear.

    Registration is once per model, including checkpoint recomputation. No hook
    is attached to no-grad outputs, and padding/context/objective are untouched.
    This controls one arithmetic path, not all sources of batch-shape sensitivity.
    """

    def on_output(_module, _inputs, output):
        if torch.is_grad_enabled() and isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(lambda gradient: gradient.contiguous())

    return [child.register_forward_hook(on_output) for child in module.modules() if isinstance(child, nn.Linear)]
