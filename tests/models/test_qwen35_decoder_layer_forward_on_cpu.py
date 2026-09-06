import torch
from torch import nn

from verl.models.transformers.qwen3_5 import qwen3_5_decoder_layer_forward


class _LinearAttention(nn.Module):
    def forward(self, *, hidden_states, **_kwargs):
        return hidden_states


class _WrappedLinearDecoder(nn.Module):
    """Approximate an FSDP2 wrapper that has children but no layer_type."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.linear_attn = _LinearAttention()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Identity()


def test_qwen35_decoder_forward_infers_layer_type_for_fsdp2_wrapper():
    layer = _WrappedLinearDecoder()
    hidden_states = torch.ones(1, 2, 3)

    output = qwen3_5_decoder_layer_forward(
        layer,
        hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
    )

    torch.testing.assert_close(output, hidden_states * 4)
