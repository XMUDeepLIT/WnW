
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.varlen import varlen_attn

from common.kv.flat_cache import FlatVarlenCache
from common.session import AlgorithmSession
from algorithms.wnw.code.cache import WnWCache
from common.kv.fused_attn import q1_decode_attn

import os

_FUSED_ATTN = os.environ.get("WNW_FUSED_ATTN", "0") == "1"


class WnWSession(AlgorithmSession):

    def _make_patched_forward(self, attn_module: nn.Module) -> Callable:
        cache: WnWCache = self.cache
        adapter = self.adapter
        meta = adapter.attention_meta(attn_module)
        num_kv_heads = meta["num_kv_heads"]
        num_kv_groups = meta["num_kv_groups"]
        num_q_heads = num_kv_heads * num_kv_groups
        scaling = meta["scaling"]
        head_dim = meta["head_dim"]
        layer_idx = meta["layer_idx"]
        is_last_layer = (layer_idx == cache.num_layers - 1)

        def patched_forward(
            self_attn,
            hidden_states: Tensor,
            position_embeddings: tuple[Tensor, Tensor] = None,
            attention_mask: Tensor | None = None,
            past_key_values=None,
            **kwargs,
        ):
            assert isinstance(cache, FlatVarlenCache)
            B, S, _ = hidden_states.shape
            input_shape = hidden_states.shape[:-1]

            q, k, v = adapter.compute_qkv(self_attn, hidden_states)
            q, k = adapter.apply_rope(q, k, position_embeddings, self_attn)

            if not cache.prefilled:
                cache.prefill_before_attn(layer_idx)
                cache.update(k, v, layer_idx)

                kg = k.repeat_interleave(num_kv_groups, dim=1)
                vg = v.repeat_interleave(num_kv_groups, dim=1)
                attn_output = F.scaled_dot_product_attention(
                    q, kg, vg, attn_mask=attention_mask,
                    scale=scaling, is_causal=(attention_mask is None),
                )
                del kg, vg

                class_a_scores, per_head_scores = _compute_prefill_scores(
                    q, k, attention_mask, cache, layer_idx,
                    B, S, num_kv_heads, num_kv_groups, scaling,
                )
                del q

                cache.prefill_after_attn(layer_idx, class_a_scores, per_head_scores, k, v)

                if is_last_layer:
                    cache.prefill_all_layers_done()

                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                return self_attn.o_proj(attn_output), None

            else:
                cache.update(k, v, layer_idx)

                k_flat, v_flat, cu_k, max_k = cache.decode_before_attn(layer_idx, k, v)

                q_flat = q.squeeze(2)
                q_flat = q_flat.view(B, num_kv_heads, num_kv_groups, head_dim)
                q_flat = q_flat.reshape(B * num_kv_heads, num_kv_groups, head_dim)

                if _FUSED_ATTN:
                    attn_output = cache.fused_decode_attention(
                        layer_idx, q_flat, q, k_flat, v_flat, cu_k, scaling
                    )
                else:
                    total_kv = k_flat.shape[0]
                    k_exp = k_flat.unsqueeze(1).as_strided(
                        (total_kv, num_kv_groups, head_dim), (head_dim, 0, 1)
                    )
                    v_exp = v_flat.unsqueeze(1).as_strided(
                        (total_kv, num_kv_groups, head_dim), (head_dim, 0, 1)
                    )

                    num_segs = B * num_kv_heads
                    cu_q = torch.arange(num_segs + 1, dtype=torch.int32, device=hidden_states.device)

                    attn_output = varlen_attn(
                        q_flat, k_exp, v_exp,
                        cu_q, cu_k,
                        max_q=1, max_k=max_k,
                        scale=scaling,
                    )
                attn_output = attn_output.view(B, num_kv_heads * num_kv_groups, head_dim)
                attn_output = attn_output.unsqueeze(1)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()

                if not _FUSED_ATTN:
                    cache.decode_after_attn(layer_idx, q)
                if is_last_layer:
                    cache.decode_all_layers_done()

                return self_attn.o_proj(attn_output), None

        return patched_forward


def _compute_prefill_scores(
    query_states: Tensor, key_states: Tensor, attention_mask: Tensor | None,
    cache, layer_idx: int,
    B: int, S: int, num_kv_heads: int, num_kv_groups: int, scaling: float,
) -> tuple[list[Tensor | None], list[list[Tensor]]]:
    if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.dtype == torch.bool:
        attention_mask = torch.where(attention_mask, 0.0, float("-inf")).to(query_states.dtype)

    if attention_mask is None:
        causal = torch.full((S, S), float("-inf"), device=query_states.device, dtype=query_states.dtype)
        causal.triu_(1)
        attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    config = cache.config
    class_a_local = config.get_layer_class_a(layer_idx)

    class_a_scores: list[Tensor | None] = []
    per_head_scores: list[list[Tensor | None]] = []

    for b in range(B):
        ra_start = cache._raw_audio_starts[b]
        ra_end = cache._raw_audio_ends[b]

        if class_a_local:
            kv_h_a = class_a_local[0] // num_kv_groups
            q_last = query_states[b, class_a_local, -1:, :]
            k_a = key_states[b, kv_h_a:kv_h_a + 1, :, :]
            scores = torch.matmul(q_last, k_a.transpose(-1, -2)) * scaling
            if attention_mask is not None and attention_mask.dim() == 4:
                scores = scores + attention_mask[b, :, -1:, :]
            scores = scores.float().softmax(dim=-1)
            class_a_scores.append(scores[:, 0, ra_start:ra_end].sum(dim=0).detach())
        else:
            class_a_scores.append(None)

        head_scores: list[Tensor | None] = []
        for kv_h in range(num_kv_heads):
            if config.is_class_a_kv_head(layer_idx, kv_h):
                head_scores.append(None)
            else:
                q_s = kv_h * num_kv_groups
                q_e = q_s + num_kv_groups
                head_scores.append(_compute_head_scores(
                    query_states[b], key_states[b, kv_h], attention_mask, b,
                    S, q_s, q_e, ra_start, ra_end, scaling,
                    config.prefill_score_mode,
                ))
        per_head_scores.append(head_scores)

    return class_a_scores, per_head_scores


def _compute_head_scores(
    q_all: Tensor, k_single: Tensor, attention_mask: Tensor | None,
    b: int, seq_len: int, q_s: int, q_e: int,
    ra_start: int, ra_end: int, scaling: float, score_mode: str,
) -> Tensor:
    num_audio = ra_end - ra_start
    k = k_single.unsqueeze(0)

    if score_mode == "last":
        q = q_all[q_s:q_e, -1:, :]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
        if attention_mask is not None and attention_mask.dim() == 4:
            scores = scores + attention_mask[b, :, -1:, :]
        scores = scores.float().softmax(dim=-1)
        return scores[:, 0, ra_start:ra_end].sum(dim=0)

    elif score_mode == "mean_text":
        num_text = seq_len - ra_end
        if num_text == 0:
            return torch.zeros(num_audio, device=q_all.device, dtype=torch.float32)
        q = q_all[q_s:q_e, ra_end:, :]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
        if attention_mask is not None and attention_mask.dim() == 4:
            scores = scores + attention_mask[b, :, ra_end:, :]
        scores = scores.float().softmax(dim=-1)
        return scores[:, :, ra_start:ra_end].mean(dim=1).sum(dim=0)

    else:
        raise ValueError(f"Unknown prefill_score_mode: {score_mode}")
