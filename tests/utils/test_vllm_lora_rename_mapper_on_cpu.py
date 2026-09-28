# Copyright 2026 CarbonTeq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LoRA tensors synced to vLLM keep their constituent module names (q_proj, w1, w3)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.lora.lora_model import LoRAModel  # noqa: E402
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager  # noqa: E402

from verl.utils.vllm.utils import TensorLoRARequest, VLLMHijack, lora_weights_mapper  # noqa: E402


class _Mapper:
    def __init__(self, renamed=None):
        self.renamed = renamed

    def get_rename_mapper(self):
        return self.renamed


def test_lora_weights_mapper_drops_stacking_like_vllm():
    renamed = object()
    assert lora_weights_mapper(SimpleNamespace(hf_to_vllm_mapper=_Mapper(renamed))) is renamed
    assert lora_weights_mapper(SimpleNamespace(hf_to_vllm_mapper=None)) is None
    assert lora_weights_mapper(SimpleNamespace()) is None
    legacy = object()  # vLLM releases without get_rename_mapper keep the mapper as is
    assert lora_weights_mapper(SimpleNamespace(hf_to_vllm_mapper=legacy)) is legacy


def test_tensor_lora_request_keeps_lfm2_constituent_modules_apart(monkeypatch):
    """LFM2 stacks q/k/v into qkv_proj and w1/w3 into w13 for base weights only.

    With the full mapper both w1 and w3 LoRA tensors were named ``w13``; the last
    one won and vLLM's merged-layer ``set_lora`` failed with IndexError.
    """

    lfm2 = pytest.importorskip("vllm.model_executor.models.lfm2")
    from vllm.config.lora import LoRAConfig

    # CPU test: vLLM pins host memory only when a CUDA runtime is present.
    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False, raising=False)
    VLLMHijack.hijack()
    rank, hidden, ffn = 4, 16, 32
    torch.manual_seed(0)
    prefix = "base_model.model.model.layers.0"
    shapes = {
        "feed_forward.w1": (ffn, hidden),
        "feed_forward.w3": (ffn, hidden),
        "feed_forward.w2": (hidden, ffn),
        "conv.in_proj": (3 * hidden, hidden),
        "self_attn.q_proj": (hidden, hidden),
        "self_attn.k_proj": (hidden // 2, hidden),
    }
    tensors = {}
    for module, (out_features, in_features) in shapes.items():
        tensors[f"{prefix}.{module}.lora_A.weight"] = torch.randn(rank, in_features)
        tensors[f"{prefix}.{module}.lora_B.weight"] = torch.randn(out_features, rank)
    manager = SimpleNamespace(
        _adapter_manager=SimpleNamespace(
            supported_lora_modules=["qkv_proj", "out_proj", "w13", "w2", "in_proj"],
            packed_modules_mapping=lfm2.Lfm2ForCausalLM.packed_modules_mapping,
            model=SimpleNamespace(hf_to_vllm_mapper=lfm2.Lfm2ForCausalLM.hf_to_vllm_mapper),
        ),
        lora_config=LoRAConfig(max_lora_rank=8, lora_dtype=torch.float32),
        _lora_model_cls=LoRAModel,
        vocab_size=128,
        max_position_embeddings=64,
    )
    request = TensorLoRARequest(
        lora_name="policy",
        lora_int_id=1,
        lora_path="unused",
        peft_config={"r": rank, "lora_alpha": 8, "target_modules": sorted({m.split(".")[-1] for m in shapes})},
        lora_tensors=tensors,
    )

    lora = LRUCacheWorkerLoRAManager._load_adapter(manager, request)

    names = set(lora.loras)
    assert {
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w3",
        "model.layers.0.feed_forward.w2",
        "model.layers.0.short_conv.in_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    } == names
    torch.testing.assert_close(
        lora.loras["model.layers.0.feed_forward.w3"].lora_a.float(),
        tensors[f"{prefix}.feed_forward.w3.lora_A.weight"],
    )
