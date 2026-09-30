import builtins

import pytest
import torch

from verl.utils import attention_utils


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_padding_without_flash_attention_preserves_values_and_gradients(monkeypatch, dtype):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "flash_attn.bert_padding":
            raise ModuleNotFoundError("flash attention deliberately absent", name="flash_attn")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr("verl.utils.device.is_torch_npu_available", lambda **kwargs: False)
    x = torch.arange(24, dtype=dtype).reshape(2, 4, 3).requires_grad_()
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    selected, indices, offsets, maximum, lengths = attention_utils.unpad_input(x, mask)
    torch.testing.assert_close(selected, x[mask.bool()])
    assert offsets.tolist() == [0, 3, 5]
    assert lengths.tolist() == [3, 2]
    assert maximum == 3
    padded = attention_utils.pad_input(selected, indices, 2, 4)
    torch.testing.assert_close(padded, x * mask.unsqueeze(-1))
    padded.sum().backward()
    torch.testing.assert_close(x.grad, mask.unsqueeze(-1).expand_as(x).to(dtype))


def test_broken_flash_dependency_is_not_hidden(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "flash_attn.bert_padding":
            raise ModuleNotFoundError("broken transitive dependency", name="flash_attn_internal_missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr("verl.utils.device.is_torch_npu_available", lambda **kwargs: False)
    with pytest.raises(ModuleNotFoundError, match="broken transitive"):
        attention_utils._get_attention_functions()
