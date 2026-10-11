
from __future__ import annotations

import torch
from torch import nn


def degqa_model(model, adapter) -> None:
    """Convert a GQA model to equivalent MHA in-place. Not reversible."""
    text_config = adapter.get_text_config(model)
    num_kv = text_config.num_key_value_heads
    num_q = text_config.num_attention_heads
    if num_kv == num_q:
        return
    if num_q % num_kv != 0:
        raise ValueError(
            f"num_attention_heads ({num_q}) must be divisible by "
            f"num_key_value_heads ({num_kv})"
        )
    group_size = num_q // num_kv

    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = text_config.hidden_size // num_q

    for layer in adapter.get_decoder_layers(model):
        attn = layer.self_attn
        _expand_proj(attn.k_proj, head_dim, group_size)
        _expand_proj(attn.v_proj, head_dim, group_size)
        _update_attn_module(attn, num_q)

    text_config.num_key_value_heads = num_q
    if hasattr(text_config, "num_key_value_groups"):
        text_config.num_key_value_groups = 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _expand_proj(proj: nn.Linear, head_dim: int, group_size: int) -> None:
    out_features, in_features = proj.weight.shape
    num_kv = out_features // head_dim
    if num_kv * head_dim != out_features:
        raise ValueError(
            f"proj out_features={out_features} not divisible by head_dim={head_dim}"
        )

    new_out = num_kv * group_size * head_dim
    device = proj.weight.device
    dtype = proj.weight.dtype

    new_weight = (
        proj.weight.data.view(num_kv, head_dim, in_features)
        .repeat_interleave(group_size, dim=0)
        .reshape(new_out, in_features)
        .contiguous()
    )

    if proj.bias is not None:
        new_bias = (
            proj.bias.data.view(num_kv, head_dim)
            .repeat_interleave(group_size, dim=0)
            .reshape(new_out)
            .contiguous()
        )
    else:
        new_bias = None

    del proj.weight
    proj.weight = nn.Parameter(new_weight, requires_grad=False)
    proj.out_features = new_out

    if new_bias is not None:
        del proj.bias
        proj.bias = nn.Parameter(new_bias, requires_grad=False)


def _update_attn_module(attn: nn.Module, num_q: int) -> None:
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = num_q
    if hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = 1
    if hasattr(attn, "config"):
        attn.config.num_key_value_heads = num_q
        if hasattr(attn.config, "num_key_value_groups"):
            attn.config.num_key_value_groups = 1
