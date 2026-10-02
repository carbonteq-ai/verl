"""Value-preserving layout and FP32 LoRA leaf controls before native FSDP wrapping."""

import pytest
import torch
from torch import nn

from verl.utils.linear_precision import (
    FP32AdapterLinear,
    prepare_fp32_lora_leaves,
    register_contiguous_linear_output_gradients,
)
from verl.workers.config.engine import FSDPEngineConfig, QATEngineConfig


def adapter_model():
    model = nn.Module()
    model.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
    model.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
    return model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_adapter_arithmetic_preserves_fp32_parameters_inside_low_precision_autocast(dtype):
    model = adapter_model()
    parameters = dict(model.named_parameters())
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    names = prepare_fp32_lora_leaves(model)
    assert names == ("lora_A.default", "lora_B.default")
    assert torch.equal(rng, torch.get_rng_state())
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    inputs = torch.tensor([[1, -2, 0.5]], dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        output = model.lora_A["default"](inputs)
    reference_inputs = inputs.detach().float().requires_grad_()
    reference_weight = parameters["lora_A.default.weight"].detach().clone().requires_grad_()
    reference = torch.nn.functional.linear(reference_inputs, reference_weight)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    output.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(parameters["lora_A.default.weight"].grad, reference_weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad.to(dtype), rtol=0, atol=0)
    # Checkpoint state remains loadable by the ordinary adapter topology.
    adapter_model().load_state_dict(model.state_dict(), strict=True)


def test_layout_hook_preserves_strided_cotangent_values_and_linear_derivatives():
    layer = nn.Linear(3, 4)
    reference = nn.Linear(3, 4)
    reference.load_state_dict(layer.state_dict())
    handles = register_contiguous_linear_output_gradients(layer)
    inputs = torch.randn(2, 3, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    output = layer(inputs)
    incoming = torch.randn(4, 2).transpose(0, 1)
    assert not incoming.is_contiguous()
    observed = []
    output.register_hook(lambda gradient: observed.append(gradient.clone(memory_format=torch.preserve_format)))
    output.backward(incoming)
    reference(reference_inputs).backward(incoming.contiguous())
    assert observed[0].is_contiguous()
    torch.testing.assert_close(observed[0], incoming, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad, rtol=0, atol=0)
    with torch.no_grad():
        layer(inputs)
    for handle in handles:
        handle.remove()


def test_unsupported_adapter_leaves_fail_before_any_replacement():
    model = adapter_model()
    first = model.lora_A["default"]
    model.lora_B["default"] = nn.Identity()
    with pytest.raises(ValueError, match="ordinary Linear"):
        prepare_fp32_lora_leaves(model)
    assert model.lora_A["default"] is first
    with pytest.raises(ValueError, match="without any LoRA"):
        prepare_fp32_lora_leaves(nn.Linear(2, 2))


def test_adapter_rejects_non_fp32_parameter_view():
    with pytest.raises(RuntimeError, match="FP32 native parameter views"):
        FP32AdapterLinear(2, 2).half()(torch.ones(1, 2))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"strategy": "fsdp2", "use_orig_params": True}, "FSDP1"),
        ({"use_orig_params": False}, "use_orig_params"),
        ({"use_orig_params": True, "model_dtype": "bf16"}, "model_dtype"),
        ({"use_orig_params": True, "qat": QATEngineConfig(enable=True)}, "QAT"),
    ],
)
def test_precision_config_rejects_unqualified_combinations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FSDPEngineConfig(lora_fp32_compute=True, **kwargs)


def test_precision_controls_default_off_and_qualified_config_constructs():
    default = FSDPEngineConfig()
    assert not default.lora_fp32_compute and not default.contiguous_linear_output_gradients
    FSDPEngineConfig(lora_fp32_compute=True, use_orig_params=True, contiguous_linear_output_gradients=True)
