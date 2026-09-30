from types import SimpleNamespace

import pytest
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.models.transformers.qwen3_5 import qwen3_5_gated_delta_net_forward


@pytest.mark.parametrize("packed", [False, True])
def test_qwen_padded_engine_retains_native_forward(monkeypatch, packed):
    classes = [
        modeling_qwen3_5.Qwen3_5Model,
        modeling_qwen3_5.Qwen3_5DecoderLayer,
        modeling_qwen3_5.Qwen3_5GatedDeltaNet,
        modeling_qwen3_5.Qwen3_5ForConditionalGeneration,
        modeling_qwen3_5_moe.Qwen3_5MoeModel,
        modeling_qwen3_5_moe.Qwen3_5MoeDecoderLayer,
        modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet,
        modeling_qwen3_5_moe.Qwen3_5MoeForConditionalGeneration,
    ]
    originals = {cls: cls.forward for cls in classes}
    for cls in classes:
        monkeypatch.setattr(cls, "forward", cls.forward)
    for cls in [modeling_qwen3_5.Qwen3_5VisionModel, modeling_qwen3_5_moe.Qwen3_5MoeVisionModel]:
        monkeypatch.setattr(cls, "fast_pos_embed_interpolate", cls.fast_pos_embed_interpolate)
    config = SimpleNamespace(model_type="qwen3_5", num_attention_heads=2, num_key_value_heads=1)
    model_type = type("QwenPaddingProbe", (), {"__module__": modeling_qwen3_5.__name__})
    model = model_type()
    model.config = config
    apply_monkey_patch(model, use_remove_padding=packed, use_fused_kernels=False)
    if packed:
        assert modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward is qwen3_5_gated_delta_net_forward
    else:
        for cls in classes:
            assert cls.forward is originals[cls]
