import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import logprobs_from_logits_v2
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_actual_fsdp_output_route_temperature_scores_and_gradients(packed, dtype):
    offsets = torch.tensor([0, 3, 5])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.zeros(5, dtype=torch.long), offsets=offsets)
    temperatures = torch.tensor([0.8, 0.1])
    flat_temperatures = torch.tensor([0.8, 0.8, 0.8, 0.1, 0.1])
    values = [-12.0, -4.0, 0.0, 2.0]
    logits = torch.tensor(values, dtype=dtype).repeat((1, 5, 1) if packed else (2, 3, 1)).requires_grad_()
    batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(batch, use_remove_padding=packed, pad_mode=DatasetPadMode.NO_PADDING,
                         use_fused_kernels=False, calculate_entropy=False, calculate_sum_pi_squared=False)
    output_args = {"input_ids_rmpad_rolled": torch.zeros(5, dtype=torch.long)}
    output_args.update({"temperature_rmpad": flat_temperatures, "pad_size": 0} if packed
                       else {"temperature": temperatures})
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False)
    # Execute the real CPU fallback normalizer, avoiding an optional CUDA-only CE kernel.
    def cpu_logprobs(logits, labels, inplace_backward=True):
        return logprobs_from_logits_v2(logits, labels)

    with patch("verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits", cpu_logprobs):
        output = engine.prepare_model_outputs(SimpleNamespace(logits=logits), output_args, batch, None)
    scores = output["log_probs"].values()
    gradients = torch.autograd.grad(scores.sum(), logits)[0]
    expected_scores = []
    expected_gradients = []
    for temperature in flat_temperatures.tolist():
        scaled = [value / temperature for value in values]
        maximum = max(scaled)
        weights = [math.exp(value - maximum) for value in scaled]
        denominator = math.fsum(weights)
        expected_scores.append(scaled[0] - maximum - math.log(denominator))
        expected_gradients.append(
            [(int(i == 0) - weight / denominator) / temperature for i, weight in enumerate(weights)]
        )
    assert scores.dtype == torch.float32
    assert torch.allclose(scores, torch.tensor(expected_scores), atol=2e-5, rtol=0)
    if packed:
        active_gradients = gradients[0]
    else:
        assert torch.equal(gradients[1, 2], torch.zeros(4, dtype=dtype))
        active_gradients = torch.cat([gradients[0, :3], gradients[1, :2]])
    assert torch.allclose(active_gradients, torch.tensor(expected_gradients, dtype=dtype), atol=2e-5, rtol=0)
