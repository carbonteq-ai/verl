"""Export must preserve which projections were trained across model towers."""

import json
from types import SimpleNamespace

import peft
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def model():
    torch.manual_seed(19)
    value = AutoModelForCausalLM.from_config(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    value.vision_tower = torch.nn.Module()
    value.vision_tower.q_proj = torch.nn.Linear(16, 16, bias=False)
    return value


def test_exported_adapter_does_not_expand_targets_to_another_tower(tmp_path):
    target = 'model.layers.0.self_attn.q_proj'
    trained = peft.get_peft_model(model(), peft.LoraConfig(task_type='CAUSAL_LM', r=4,
                                                         lora_alpha=8, target_modules=[target]))
    with torch.no_grad():
        trained.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.fill_(0.01)
    (tmp_path / 'lora_train_meta.json').write_text(json.dumps({'r': 4, 'lora_alpha': 8}))
    merger = FSDPModelMerger.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(local_dir=str(tmp_path), target_dir=str(tmp_path / 'export'))
    destination = merger.save_lora_adapter(dict(trained.state_dict()))
    reloaded = peft.PeftModel.from_pretrained(model(), destination)
    actual = {name for name, module in reloaded.base_model.model.named_modules() if hasattr(module, 'lora_A')}
    assert actual == {target}
    tokens = torch.tensor([[1, 2, 3, 4]])
    trained.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(tokens).logits, trained(tokens).logits, rtol=0, atol=0)
