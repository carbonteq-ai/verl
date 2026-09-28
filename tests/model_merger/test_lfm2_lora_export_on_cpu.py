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
"""LoRA export of a hybrid short-convolution model with tied embeddings (LFM2)."""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")

from verl.model_merger.base_model_merger import ModelMergerConfig  # noqa: E402
from verl.model_merger.fsdp_model_merger import FSDPModelMerger  # noqa: E402


def _tiny_lfm2(path):
    config = transformers.Lfm2Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        max_position_embeddings=64,
        tie_word_embeddings=True,
        block_auto_adjust_ff_dim=False,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.save_pretrained(path)
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {f"t{index}": index for index in range(config.vocab_size)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="t0"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0").save_pretrained(path)
    return model


def test_lfm2_all_linear_lora_export_reloads_with_identical_outputs(tmp_path):
    hf_dir = tmp_path / "actor" / "huggingface"
    base = _tiny_lfm2(hf_dir)
    peft_model = peft.get_peft_model(
        base,
        peft.LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules="all-linear"),
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    (tmp_path / "actor" / "lora_train_meta.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    # The FSDP checkpoint carries the PEFT model's own parameter names.
    state_dict = {name: tensor.detach().clone() for name, tensor in peft_model.state_dict().items()}

    target = tmp_path / "model"
    merger = FSDPModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            target_dir=str(target),
            local_dir=str(tmp_path / "actor"),
            hf_model_config_path=str(hf_dir),
        )
    )
    merger.save_hf_model_and_tokenizer(state_dict)

    adapter_config = json.loads((target / "lora_adapter" / "adapter_config.json").read_text())
    # PEFT "all-linear": attention q/k/v/out_proj, short-convolution in/out_proj,
    # feed-forward w1/w2/w3; never the tied output head.
    assert set(adapter_config["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj",
        "w1",
        "w2",
        "w3",
    }
    assert (adapter_config["r"], adapter_config["lora_alpha"]) == (4, 8)
    saved = safetensors_torch.load_file(str(target / "model.safetensors"))
    assert "lm_head.weight" not in saved
    assert not any("lora_" in name or "base_layer" in name for name in saved)

    reloaded = peft.PeftModel.from_pretrained(
        transformers.AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32),
        target / "lora_adapter",
    )
    tokens = torch.tensor([[1, 5, 9, 17, 33, 2, 8, 4]])
    peft_model.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = peft_model(input_ids=tokens).logits
        actual = reloaded(input_ids=tokens).logits
    assert reloaded.base_model.model.lm_head.weight.data_ptr() == (
        reloaded.base_model.model.model.embed_tokens.weight.data_ptr()
    )
    torch.testing.assert_close(actual, expected)
