
from __future__ import annotations

import torch
from torch import Tensor
from transformers.cache_utils import Cache, CacheLayerMixin

from common.adapters.base import BaseModelAdapter
from common.monitoring.inference_stats import InferenceStats


class _SeqLengthStub(CacheLayerMixin):

    is_compileable = False
    is_sliding = False

    def __init__(self, get_len_fn):
        super().__init__()
        self._get_len = get_len_fn
        self.is_initialized = True

    def lazy_initialization(self, *args, **kwargs):
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        raise RuntimeError("FlatVarlenCache: layers[].update should never be called directly")

    def get_seq_length(self) -> int:
        return self._get_len()

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int):
        seq_len = self._get_len()
        return seq_len + query_length, seq_len

    def reset(self):
        pass

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search not supported")


class FlatVarlenCache(Cache):
    """Flat KV storage + cu_seqlens for variable-length attention.

    Per-layer storage (allocated by subclass):
        layer_k[i]: Tensor[total_tokens, head_dim]   (flat, concatenated across B * num_kv_heads)
        layer_v[i]: Tensor[total_tokens, head_dim]
        cu_seqlens[i]: Tensor[B * num_kv_heads + 1]  (int32)

    Also stores audio range metadata + algorithm state (no model pollution).
    """

    is_compileable = False

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        device: torch.device,
        adapter: BaseModelAdapter,
    ):
        stubs = [_SeqLengthStub(lambda i=i: self._seq_lengths[i] if i < len(self._seq_lengths) else 0)
                 for i in range(num_layers)]
        super().__init__(layers=stubs)

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_kv_groups = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.adapter = adapter

        self._seq_lengths: list[int] = [0] * num_layers
        self.prefilled: bool = False

        self.audio_starts: Tensor | None = None
        self.audio_ends: Tensor | None = None
        self.pad_lens: Tensor | None = None

        self.stats = InferenceStats(device, interval=5)

    def setup_audio(
        self,
        audio_starts: Tensor,
        audio_ends: Tensor,
        attention_mask: Tensor | None,
    ) -> None:
        """Called by Session before each generate(). Resets state for new sample."""
        self.audio_starts = audio_starts
        self.audio_ends = audio_ends
        self.prefilled = False
        self._seq_lengths = [0] * self.num_layers
        self.stats.reset(audio_starts.shape[0])

        if attention_mask is not None:
            B = attention_mask.shape[0]
            pad_lens = []
            for b in range(B):
                nz = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
                pad_lens.append(int(nz[0].item()) if len(nz) > 0 else 0)
            self.pad_lens = torch.tensor(pad_lens, dtype=torch.int64, device=self.device)
        else:
            self.pad_lens = torch.zeros(audio_starts.shape[0], dtype=torch.int64, device=self.device)


    def update(
        self, key_states: Tensor, value_states: Tensor, layer_idx: int, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """Override Cache.update to bypass layers[] routing.

        During prefill: patched_forward handles storage directly (this is a no-op for seq tracking).
        During decode: patched_forward handles storage directly (this is a no-op for seq tracking).

        We only update the seq_length counter here so get_seq_length works for HF internals.
        The actual KV management is done in prefill_before_attn / decode_before_attn.
        """
        new_tokens = key_states.shape[-2]
        self._seq_lengths[layer_idx] += new_tokens
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= self.num_layers:
            return 0
        return self._seq_lengths[layer_idx]

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        seq_len = self.get_seq_length(layer_idx)
        return seq_len + query_length, seq_len

    def reset(self):
        """Reset all state. Called by Session.__enter__."""
        self.prefilled = False
        self._seq_lengths = [0] * self.num_layers
        self.audio_starts = None
        self.audio_ends = None
        self.pad_lens = None
        self.stats.reset(0)

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("FlatVarlenCache does not support beam search")

    def crop(self, max_length: int):
        raise NotImplementedError("FlatVarlenCache does not support assisted decoding crop")

    def batch_repeat_interleave(self, repeats: int):
        raise NotImplementedError("FlatVarlenCache does not support batch_repeat_interleave")

    def batch_select_indices(self, indices):
        raise NotImplementedError("FlatVarlenCache does not support batch_select_indices")


    def prefill_before_attn(self, layer_idx: int) -> None:
        """Called before attention during prefill. Subclass sets up layer storage."""

    def prefill_after_attn(
        self, layer_idx: int, scores: Tensor, k: Tensor, v: Tensor
    ) -> None:
        """Called after attention during prefill. Subclass applies compression."""

    def prefill_all_layers_done(self) -> None:
        """Called after last layer's prefill completes. Flips prefilled flag."""
        self.prefilled = True

    def decode_before_attn(
        self, layer_idx: int, k_new: Tensor, v_new: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Called before attention during decode. Returns (K_flat, V_flat, cu_seqlens_k, max_seqlen_k)."""
        raise NotImplementedError("Subclass must implement decode_before_attn")

    def decode_after_attn(self, layer_idx: int, q: Tensor) -> None:
        """Called after attention during decode. Subclass collects scores / updates eviction."""

    def decode_all_layers_done(self) -> None:
        """Called after last layer's decode completes. Subclass does chunk maintenance etc."""

    def _snapshot_cache_stats(self, step: int = -1) -> list[dict]:
        raise NotImplementedError("Subclass must implement _snapshot_cache_stats")
