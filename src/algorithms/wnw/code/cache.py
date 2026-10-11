
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor

from common.kv.flat_cache import FlatVarlenCache
from common.kv.kernels import interleave_append
from common.adapters.base import BaseModelAdapter
from algorithms.wnw.code.config import WnWConfig


@dataclass
class SegmentMeta:
    """Position info for a segment across storages."""
    cpu_start: int
    cpu_end: int
    gpu_rel_start: int = -1
    gpu_rel_end: int = -1


class WnWCache(FlatVarlenCache):
    """WnW KV cache manager.

    Lifecycle:
    1. Construct, then call setup_audio() with audio ranges
    2. Prefill: after each layer's attn call prefill_after_attn -> finally prefill_all_layers_done
    3. Decode: each step each layer call decode_before_attn / decode_after_attn -> decode_all_layers_done
    """

    def __init__(self, config: WnWConfig, num_layers: int, num_kv_heads: int,
                 num_q_heads: int, head_dim: int, device: torch.device,
                 adapter: BaseModelAdapter):
        super().__init__(num_layers, num_kv_heads, num_q_heads, head_dim, device, adapter)
        self.config = config

        self.batch_size = 0
        self._audio_starts_list: list[int] = []
        self._audio_ends_list: list[int] = []

        self.segment_ranges: list[list[tuple[int, int]]] = []
        self.chunk_to_segs: list[list[list[int]]] = []

        self.layer_k: list[torch.Tensor | None] = [None] * num_layers
        self.layer_v: list[torch.Tensor | None] = [None] * num_layers
        self.cu_seqlens_k: list[torch.Tensor | None] = [None] * num_layers
        self.cu_seqlens_list: list[list[int] | None] = [None] * num_layers
        self.seq_lens: list[list[int] | None] = [None] * num_layers

        self.cpu_restore_k: torch.Tensor | None = None
        self.cpu_restore_v: torch.Tensor | None = None

        self.restore_meta: list[list[list[list[SegmentMeta | None]]]] = []

        self.loaded_segments: list[set[int]] = []
        self.eviction_counters: list[dict[int, int]] = []

        self._prefill_scores: list[list[torch.Tensor]] = []
        self.score_accumulator: torch.Tensor | None = None
        self.score_offsets: torch.Tensor | None = None

        self._prefill_decisions: list[dict] = []
        self._cpu_buffers_k: list[torch.Tensor] = []
        self._cpu_buffers_v: list[torch.Tensor] = []
        self._cpu_buffer_metas: list[tuple] = []

        self._pending_chunk_update: bool = False

        self._decode_step: int = 0
        self._segments_to_load: list[set[int]] = []
        self._segments_to_evict: list[set[int]] = []


        self.use_triton_rebuild = os.environ.get("WNW_DISABLE_TRITON_REBUILD", "0") != "1"
        self._class_b_heads_by_layer = [
            config.get_layer_class_b(layer_idx) for layer_idx in range(num_layers)
        ]
        self._transfer_stream: torch.cuda.Stream | None = None
        self._pending_transfer_event: torch.cuda.Event | None = None
        self._pending_transfer_waited = False
        self._pending_gpu_k: torch.Tensor | None = None
        self._pending_gpu_v: torch.Tensor | None = None
        self._pending_load_starts: list[torch.Tensor | None] = [None] * num_layers
        self._pending_load_lens: list[torch.Tensor | None] = [None] * num_layers
        self._pending_evict_offsets: list[torch.Tensor | None] = [None] * num_layers
        self._pending_evict_starts: list[torch.Tensor | None] = [None] * num_layers
        self._pending_evict_ends: list[torch.Tensor | None] = [None] * num_layers
        self._pending_evict_max: list[int] = [0] * num_layers
        self._pending_new_lens: list[list[int] | None] = [None] * num_layers
        self._pending_cu_new: list[torch.Tensor | None] = [None] * num_layers
        self._pending_load_starts_2d: torch.Tensor | None = None
        self._pending_load_lens_2d: torch.Tensor | None = None
        self._pending_cu_new_2d: torch.Tensor | None = None
        self._pending_layers_applied: int = 0
        self._empty_i32: torch.Tensor | None = None
        self._staging_k: torch.Tensor | None = None
        self._staging_v: torch.Tensor | None = None
        self._staging_capacity: int = 0

    def _clear_pending_rebuild_state(self):
        self._pending_transfer_event = None
        self._pending_transfer_waited = False
        self._pending_gpu_k = None
        self._pending_gpu_v = None
        self._pending_load_starts = [None] * self.num_layers
        self._pending_load_lens = [None] * self.num_layers
        self._pending_evict_offsets = [None] * self.num_layers
        self._pending_evict_starts = [None] * self.num_layers
        self._pending_evict_ends = [None] * self.num_layers
        self._pending_evict_max = [0] * self.num_layers
        self._pending_new_lens = [None] * self.num_layers
        self._pending_cu_new = [None] * self.num_layers
        self._pending_load_starts_2d = None
        self._pending_load_lens_2d = None
        self._pending_cu_new_2d = None
        self._pending_layers_applied = 0

    def _get_transfer_stream(self):
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=self.device)
        return self._transfer_stream

    def _ensure_staging_capacity(self, n_tokens: int, dtype: torch.dtype):
        if n_tokens <= self._staging_capacity:
            return
        capacity = 1 << max(0, n_tokens - 1).bit_length()
        self._staging_k = torch.empty(capacity, self.head_dim, dtype=dtype, pin_memory=True)
        self._staging_v = torch.empty(capacity, self.head_dim, dtype=dtype, pin_memory=True)
        self._staging_capacity = capacity

    def collect_stats(self, interval: int = 20):
        """Enable stats collection during inference."""
        self.stats.set_interval(interval)

    def get_all_stats(self) -> list[dict]:
        """Return complete stats for each sample after inference."""
        return self.stats.get_all_stats()

    def setup_audio(
        self,
        audio_starts: Tensor,
        audio_ends: Tensor,
        attention_mask: Tensor | None = None,
    ) -> None:
        """Called before inference to initialize all state.

        Args:
            audio_starts/audio_ends: Tensor[B] int64, audio token start/end in original (padded) input_ids
            attention_mask: Tensor[B, seq_len] or None
        """
        super().setup_audio(audio_starts, audio_ends, attention_mask)

        audio_starts_list = audio_starts.tolist()
        audio_ends_list = audio_ends.tolist()
        pad_lens_list = self.pad_lens.tolist()

        self.batch_size = len(audio_starts_list)

        self._setup_segments(audio_starts_list, audio_ends_list, pad_lens_list)
        self._setup_score_buffers(audio_starts_list, audio_ends_list, pad_lens_list)
        self._setup_chunk_state(audio_starts_list, audio_ends_list, pad_lens_list)

    def _setup_segments(self, audio_starts: list[int], audio_ends: list[int],
                        pad_lens: list[int]) -> None:
        self._raw_audio_starts = list(audio_starts)
        self._raw_audio_ends = list(audio_ends)
        self.pad_lens_list = list(pad_lens)
        self._audio_starts_list = [s - p for s, p in zip(audio_starts, pad_lens)]
        self._audio_ends_list = [e - p for e, p in zip(audio_ends, pad_lens)]

        stride = self.config.stride_tokens
        chunk_size = self.config.chunk_size_tokens
        self.segment_ranges = [
            _compute_segments(self._audio_ends_list[b] - self._audio_starts_list[b], stride)
            for b in range(self.batch_size)
        ]
        self.chunk_to_segs = [
            _compute_chunks(self.segment_ranges[b], chunk_size, stride)
            for b in range(self.batch_size)
        ]
        self._chunk_token_bounds = []
        for b in range(self.batch_size):
            bounds = []
            for segs in self.chunk_to_segs[b]:
                if segs:
                    bounds.append((self.segment_ranges[b][segs[0]][0],
                                   self.segment_ranges[b][segs[-1]][1]))
                else:
                    bounds.append((0, 0))
            self._chunk_token_bounds.append(bounds)

        self._chunk_starts_t = []
        self._chunk_ends_t = []
        self._chunk_lens_t = []
        for b in range(self.batch_size):
            bounds = self._chunk_token_bounds[b]
            if bounds:
                starts = torch.tensor([cs for cs, _ in bounds], dtype=torch.int64, device=self.device)
                ends = torch.tensor([ce for _, ce in bounds], dtype=torch.int64, device=self.device)
                lens = (ends - starts).clamp(min=1).to(torch.float32)
            else:
                starts = torch.empty(0, dtype=torch.int64, device=self.device)
                ends = torch.empty(0, dtype=torch.int64, device=self.device)
                lens = torch.empty(0, dtype=torch.float32, device=self.device)
            self._chunk_starts_t.append(starts)
            self._chunk_ends_t.append(ends)
            self._chunk_lens_t.append(lens)

    def _setup_score_buffers(self, audio_starts: list[int], audio_ends: list[int],
                             pad_lens: list[int]) -> None:
        self._prefill_scores = [[] for _ in range(self.batch_size)]

        audio_lens = [audio_ends[b] - audio_starts[b] for b in range(self.batch_size)]
        total_audio = sum(audio_lens)
        offsets = [0] + list(torch.cumsum(torch.tensor(audio_lens), 0).tolist())
        self.score_offsets = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        self._score_offsets_list = offsets
        self.score_accumulator = torch.zeros(total_audio, dtype=torch.float32, device=self.device)

    def _setup_chunk_state(self, audio_starts: list[int], audio_ends: list[int],
                           pad_lens: list[int]) -> None:
        self.loaded_segments = [set() for _ in range(self.batch_size)]
        self.eviction_counters = [{} for _ in range(self.batch_size)]

        self.layer_k = [None] * self.num_layers
        self.layer_v = [None] * self.num_layers
        self.cu_seqlens_k = [None] * self.num_layers
        self.cu_seqlens_list = [None] * self.num_layers
        self.seq_lens = [None] * self.num_layers
        self.restore_meta = [
            [[[None] * len(self.segment_ranges[b]) for _ in range(self.num_kv_heads)]
             for b in range(self.batch_size)]
            for _ in range(self.num_layers)
        ]

        self._prefill_decisions = []
        self._cpu_buffers_k = []
        self._cpu_buffers_v = []
        self._cpu_buffer_metas = []
        self._pending_chunk_update = False
        self._segments_to_load = [set() for _ in range(self.batch_size)]
        self._segments_to_evict = [set() for _ in range(self.batch_size)]

        self._decode_step = 0
        self._clear_pending_rebuild_state()


    def prefill_before_attn(self, layer_idx: int):
        """Before attention during prefill. Record one prefill-before snapshot."""
        self.stats.record_prefill_before(self._snapshot_cache_stats)

    def prefill_after_attn(self, layer_idx: int,
                           class_a_scores: list[torch.Tensor | None],
                           per_head_scores: list[list[torch.Tensor | None]],
                           key_states: torch.Tensor, value_states: torch.Tensor):
        """After attention during prefill: prune KV using pre-computed scores + offload.

        Score computation is done in session layer. Cache only handles
        pruning, flat storage construction, and CPU offload.

        Args:
            class_a_scores: [B] list, each Tensor[num_audio] float32 or None
            per_head_scores: [B][num_kv_heads] list, Tensor[num_audio] float32 or None (Class A heads)
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: same shape
        """
        for b in range(self.batch_size):
            if class_a_scores[b] is not None:
                self._prefill_scores[b].append(class_a_scores[b])

        pieces_k, pieces_v = [], []
        seq_lens_layer = []

        for b in range(self.batch_size):
            ra_start, ra_end = self._raw_audio_starts[b], self._raw_audio_ends[b]
            pad_len = self.pad_lens_list[b]

            for kv_h in range(self.num_kv_heads):
                k_full = key_states[b, kv_h]
                v_full = value_states[b, kv_h]

                if self.config.is_class_a_kv_head(layer_idx, kv_h):
                    k_keep = k_full[pad_len:]
                    v_keep = v_full[pad_len:]
                    pieces_k.append(k_keep)
                    pieces_v.append(v_keep)
                    seq_lens_layer.append(k_keep.shape[0])
                else:
                    ratio = self.config.get_retention(layer_idx, kv_h)
                    head_scores = per_head_scores[b][kv_h]

                    if self.config.prefill_global_topk:
                        kept_mask = _select_topk_global(
                            head_scores, self.segment_ranges[b], ratio
                        )
                    else:
                        kept_mask = _select_topk_per_segment(
                            head_scores, self.segment_ranges[b], ratio
                        )

                    non_audio_k = torch.cat([k_full[pad_len:ra_start], k_full[ra_end:]], dim=0)
                    non_audio_v = torch.cat([v_full[pad_len:ra_start], v_full[ra_end:]], dim=0)
                    audio_k = k_full[ra_start:ra_end]
                    audio_v = v_full[ra_start:ra_end]

                    kept_k = audio_k[kept_mask]
                    kept_v = audio_v[kept_mask]

                    pieces_k.append(torch.cat([non_audio_k, kept_k], dim=0))
                    pieces_v.append(torch.cat([non_audio_v, kept_v], dim=0))
                    seq_lens_layer.append(non_audio_k.shape[0] + kept_k.shape[0])

                    if self.config.is_class_b(layer_idx, kv_h):
                        restore_mask = ~kept_mask
                        restore_idx = restore_mask.nonzero(as_tuple=True)[0]
                        if restore_idx.numel() > 0:
                            restore_k = audio_k.index_select(0, restore_idx).cpu()
                            restore_v = audio_v.index_select(0, restore_idx).cpu()
                            stride = self.config.stride_tokens
                            num_segs = len(self.segment_ranges[b])
                            seg_counts = torch.bincount(
                                restore_idx // stride, minlength=num_segs
                            ).cpu()
                            self.stats.record_cpu_gpu_transfer(
                                b,
                                g2c_tokens=int(restore_idx.numel()),
                                g2c_bytes=int((restore_k.numel() + restore_v.numel()) * restore_k.element_size()),
                            )
                            offset = 0
                            for seg_idx in range(num_segs):
                                n = int(seg_counts[seg_idx])
                                if n > 0:
                                    self._cpu_buffers_k.append(restore_k[offset:offset + n])
                                    self._cpu_buffers_v.append(restore_v[offset:offset + n])
                                    self._cpu_buffer_metas.append(
                                        (layer_idx, b, kv_h, seg_idx, n)
                                    )
                                    offset += n

        self.layer_k[layer_idx] = torch.cat(pieces_k, dim=0)
        self.layer_v[layer_idx] = torch.cat(pieces_v, dim=0)
        self.seq_lens[layer_idx] = seq_lens_layer
        self.cu_seqlens_k[layer_idx] = _build_cu_seqlens(seq_lens_layer, self.device)
        self.cu_seqlens_list[layer_idx] = _build_cu_seqlens_list(seq_lens_layer)

    def prefill_all_layers_done(self):
        """After all layers prefill: organize CPU tensors + select initial chunks + load restore tokens."""
        if self._cpu_buffers_k:
            self.cpu_restore_k = torch.cat(self._cpu_buffers_k, dim=0)
            self.cpu_restore_v = torch.cat(self._cpu_buffers_v, dim=0)

            try:
                self.cpu_restore_k = self.cpu_restore_k.pin_memory()
                self.cpu_restore_v = self.cpu_restore_v.pin_memory()
            except RuntimeError:
                pass

            offset = 0
            for layer_idx, b, kv_h, seg_idx, n in self._cpu_buffer_metas:
                self.restore_meta[layer_idx][b][kv_h][seg_idx] = SegmentMeta(
                    cpu_start=offset, cpu_end=offset + n
                )
                offset += n
        else:
            self.cpu_restore_k = torch.empty(0, self.head_dim)
            self.cpu_restore_v = torch.empty(0, self.head_dim)

        self._cpu_buffers_k.clear()
        self._cpu_buffers_v.clear()
        self._cpu_buffer_metas.clear()

        for b in range(self.batch_size):
            if self._prefill_scores[b]:
                avg_score = torch.stack(self._prefill_scores[b]).sum(dim=0)
            else:
                num_audio = self._audio_ends_list[b] - self._audio_starts_list[b]
                avg_score = torch.ones(num_audio, device=self.device)
            self._select_and_apply_chunks(b, avg_score)
            self._prefill_scores[b].clear()

        self.prefilled = True
        self._warmup_triton()
        self._precompute_decode_constants()

        self.stats.record_prefill(self._snapshot_cache_stats)

    def _warmup_triton(self):
        from algorithms.wnw.code.kernels import varlen_q1_softmax
        d = self.head_dim
        Q = torch.zeros(1, d, dtype=torch.bfloat16, device=self.device)
        K = torch.zeros(1, d, dtype=torch.bfloat16, device=self.device)
        starts = torch.zeros(1, dtype=torch.int32, device=self.device)
        ends = torch.ones(1, dtype=torch.int32, device=self.device)
        varlen_q1_softmax(Q, K, starts, ends)

        N = self.batch_size * self.num_kv_heads
        old = torch.zeros(N, d, dtype=torch.bfloat16, device=self.device)
        new_tok = torch.zeros(N, d, dtype=torch.bfloat16, device=self.device)
        cu_old = torch.arange(N + 1, dtype=torch.int32, device=self.device)
        cu_new = torch.arange(N + 1, dtype=torch.int32, device=self.device) * 2
        interleave_append(old, new_tok, cu_old, cu_new)

    def _precompute_decode_constants(self):
        num_segs = self.batch_size * self.num_kv_heads
        self._cu_step_increment = torch.arange(num_segs + 1, dtype=torch.int32, device=self.device)
        self._empty_i32 = torch.empty(0, dtype=torch.int32, device=self.device)

        self._decode_class_a_local = None
        for layer_idx in range(self.num_layers):
            ca = self.config.get_layer_class_a(layer_idx)
            if ca:
                self._decode_class_a_local = ca
                break
        if self._decode_class_a_local is None:
            return

        self._decode_after_groups = [[] for _ in range(self.num_layers)]
        for l in range(self.num_layers):
            ca = self.config.get_layer_class_a(l)
            if not ca:
                continue
            by_kv = {}
            for qh in ca:
                by_kv.setdefault(qh // self.num_kv_groups, []).append(qh)
            for kv, qheads in sorted(by_kv.items()):
                seg_idx = torch.tensor(
                    [b * self.num_kv_heads + kv for b in range(self.batch_size)],
                    dtype=torch.int64, device=self.device,
                )
                self._decode_after_groups[l].append((qheads, seg_idx))

        self._layer_foldable = []
        self._ca_kv_mask_by_layer = []
        for l in range(self.num_layers):
            ca = self.config.get_layer_class_a(l)
            if not ca:
                self._layer_foldable.append(False)
                self._ca_kv_mask_by_layer.append(None)
                continue
            ca_set = set(ca)
            kv_heads = sorted(set(q // self.num_kv_groups for q in ca))
            foldable = all(
                set(range(kv * self.num_kv_groups, (kv + 1) * self.num_kv_groups)).issubset(ca_set)
                for kv in kv_heads
            )
            if foldable:
                mask = torch.zeros(self.num_kv_heads, dtype=torch.int32, device=self.device)
                for kv in kv_heads:
                    mask[kv] = 1
                self._layer_foldable.append(True)
                self._ca_kv_mask_by_layer.append(mask)
            else:
                self._layer_foldable.append(False)
                self._ca_kv_mask_by_layer.append(None)

        self._decode_audio_starts_b = torch.tensor(
            [self._audio_starts_list[b] for b in range(self.batch_size)],
            dtype=torch.int32, device=self.device,
        )
        self._decode_audio_ends_b = torch.tensor(
            [self._audio_ends_list[b] for b in range(self.batch_size)],
            dtype=torch.int32, device=self.device,
        )

        from algorithms.wnw.code.kernels import fused_score_accumulate
        group_sizes = {len(qh) for groups in self._decode_after_groups for qh, _ in groups}
        if group_sizes:
            d = self.head_dim
            kdt = self.layer_k[0].dtype
            dK = torch.zeros(1, d, dtype=kdt, device=self.device)
            ss = torch.zeros(1, dtype=torch.int32, device=self.device)
            se = torch.ones(1, dtype=torch.int32, device=self.device)
            a_s = torch.zeros(1, dtype=torch.int32, device=self.device)
            a_e = torch.ones(1, dtype=torch.int32, device=self.device)
            acc = torch.zeros(1, dtype=torch.float32, device=self.device)
            off = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
            for nh in group_sizes:
                dQ = torch.zeros(1, nh, d, dtype=kdt, device=self.device)
                fused_score_accumulate(dQ, dK, ss, se, a_s, a_e, acc, off)


    def decode_before_attn(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Before attention during decode: handle chunk updates + append new token.

        Returns: (K, V, cu_seqlens_k, max_seqlen_k) for varlen attention
        """
        if self._pending_chunk_update:
            if self.use_triton_rebuild and self._pending_cu_new[layer_idx] is not None:
                return self._rebuild_append_layer(layer_idx, new_k, new_v)
            self._apply_chunk_update_to_layer(layer_idx)

        new_k_flat = new_k.squeeze(2).reshape(-1, self.head_dim)
        new_v_flat = new_v.squeeze(2).reshape(-1, self.head_dim)

        old_k, old_v = self.layer_k[layer_idx], self.layer_v[layer_idx]
        cu_old = self.cu_seqlens_k[layer_idx]
        lens = self.seq_lens[layer_idx]
        new_lens = [l + 1 for l in lens]
        cu_new = cu_old + self._cu_step_increment

        new_k_tensor = interleave_append(old_k, new_k_flat, cu_old, cu_new)
        new_v_tensor = interleave_append(old_v, new_v_flat, cu_old, cu_new)

        self.layer_k[layer_idx] = new_k_tensor
        self.layer_v[layer_idx] = new_v_tensor
        self.seq_lens[layer_idx] = new_lens
        self.cu_seqlens_k[layer_idx] = cu_new
        self.cu_seqlens_list[layer_idx] = _build_cu_seqlens_list(new_lens)

        return (new_k_tensor, new_v_tensor, cu_new, max(new_lens))

    def decode_after_attn(self, layer_idx: int, q: torch.Tensor):
        """After attention during decode: Class A score collection (fused triton kernel).

        Each Class-A kv-head group is scored against its OWN K segment (full-segment
        softmax over its q-heads, summed, audio-range accumulated). Grouping by kv-head
        gives correct GQA scoring even when a layer's Class-A q-heads span multiple
        kv-heads, and keeps num_heads==group_size for the fast h4/h1 kernel.
        """
        if self._decode_class_a_local is None:
            return
        groups = self._decode_after_groups[layer_idx]
        if not groups:
            return

        from algorithms.wnw.code.kernels import fused_score_accumulate

        cu = self.cu_seqlens_k[layer_idx]
        K = self.layer_k[layer_idx]
        for qheads, seg_indices in groups:
            Q = q[:, qheads, 0, :]
            seg_starts = cu[seg_indices].to(torch.int32)
            seg_ends = cu[seg_indices + 1].to(torch.int32)
            fused_score_accumulate(
                Q, K,
                seg_starts, seg_ends,
                self._decode_audio_starts_b, self._decode_audio_ends_b,
                self.score_accumulator, self.score_offsets,
            )

    def fused_decode_attention(self, layer_idx, q_flat, q, k_flat, v_flat, cu_k, scale):
        """Env-gated fused decode path: custom q1 attention kernel, folding Class-A
        score collection into the same kernel for foldable layers.

        Foldable layer (_fold_ca_kv[layer]>=0): the per-segment softmax for the
        Class-A kv-head already computes what scoring needs, so score is emitted in
        the same launch (near-zero marginal). Non-foldable Class-A layers fall back
        to decode_after_attn; non-Class-A layers do attention only.
        """
        from common.kv.fused_attn import q1_decode_attn, q1_decode_attn_score

        if self._layer_foldable[layer_idx]:
            return q1_decode_attn_score(
                q_flat, k_flat, v_flat, cu_k, scale,
                self._ca_kv_mask_by_layer[layer_idx], self.num_kv_heads,
                self._decode_audio_starts_b, self._decode_audio_ends_b,
                self.score_accumulator, self.score_offsets,
            )
        attn_output = q1_decode_attn(q_flat, k_flat, v_flat, cu_k, scale)
        self.decode_after_attn(layer_idx, q)
        return attn_output

    def decode_all_layers_done(self):
        """After all layers decode: select chunks from scores + mark updates."""
        self._decode_step += 1
        self.stats.flush_decode_step(self._decode_step)
        if self.stats.enabled and self._decode_step % self.stats.interval == 0:
            self.stats.record_decode(self._decode_step, self._snapshot_cache_stats)

        any_change = self._compute_decode_chunk_update()
        self._prepare_decode_rebuild_if_needed(any_change)
        self.score_accumulator.zero_()

    def _compute_decode_chunk_update(self) -> bool:
        self._pending_chunk_update = False
        any_change = False

        for b in range(self.batch_size):
            acc_start = self._score_offsets_list[b]
            acc_end = self._score_offsets_list[b + 1]
            sample_scores = self.score_accumulator[acc_start:acc_end]

            to_load, to_evict = self._compute_chunk_changes(b, sample_scores)
            self._segments_to_load[b] = to_load
            self._segments_to_evict[b] = to_evict
            if to_load or to_evict:
                any_change = True
        return any_change

    def _prepare_decode_rebuild_if_needed(self, any_change: bool, activate: bool = True) -> None:
        if any_change:
            if activate:
                self._pending_chunk_update = True
            if self.use_triton_rebuild:
                self._prepare_pending_rebuild()


    def _select_and_apply_chunks(self, sample_idx: int, scores: torch.Tensor):
        to_load, _ = self._compute_chunk_changes(sample_idx, scores)
        if to_load:
            self._load_segments_for_sample(sample_idx, to_load)
        self.loaded_segments[sample_idx] = to_load

    def _compute_chunk_changes_from_topk(self, sample_idx: int, topk_chunks: list[int]):
        chunk_to_segs = self.chunk_to_segs[sample_idx]
        needed = {si for ci in topk_chunks for si in chunk_to_segs[ci]}

        loaded = self.loaded_segments[sample_idx]
        counters = self.eviction_counters[sample_idx]

        to_load = needed - loaded
        to_evict = set()
        for seg_idx in loaded:
            if seg_idx not in needed:
                counters[seg_idx] = counters.get(seg_idx, 0) + 1
                if counters[seg_idx] >= self.config.eviction_lag:
                    to_evict.add(seg_idx)
            else:
                counters.pop(seg_idx, None)

        self.loaded_segments[sample_idx] = (loaded | to_load) - to_evict
        for seg_idx in to_evict:
            counters.pop(seg_idx, None)

        return to_load, to_evict

    def _compute_chunk_changes(self, sample_idx: int, scores: torch.Tensor):
        chunk_to_segs = self.chunk_to_segs[sample_idx]

        if chunk_to_segs:
            prefix = torch.zeros(scores.shape[0] + 1, dtype=torch.float32, device=scores.device)
            torch.cumsum(scores, dim=0, out=prefix[1:])
            starts = self._chunk_starts_t[sample_idx]
            ends = self._chunk_ends_t[sample_idx]
            chunk_scores = (prefix[ends] - prefix[starts]) / self._chunk_lens_t[sample_idx]
        else:
            chunk_scores = torch.empty(0, device=self.device)

        k = min(self.config.top_k_chunks, len(chunk_scores))
        if k == 0:
            return set(), set()

        topk_chunks = chunk_scores.topk(k).indices.tolist()
        return self._compute_chunk_changes_from_topk(sample_idx, topk_chunks)

    def _load_segments_for_sample(self, sample_idx: int, seg_indices: set[int]):
        sorted_segs = sorted(seg_indices)
        for layer_idx in range(self.num_layers):
            old_k, old_v = self.layer_k[layer_idx], self.layer_v[layer_idx]
            cu = self.cu_seqlens_list[layer_idx]

            transfer_ranges = []
            transfer_targets = []
            for flat_idx in range(self.batch_size * self.num_kv_heads):
                b, kv_h = flat_idx // self.num_kv_heads, flat_idx % self.num_kv_heads
                if b == sample_idx and self.config.is_class_b(layer_idx, kv_h):
                    for si in sorted_segs:
                        m = self.restore_meta[layer_idx][b][kv_h][si]
                        if m is not None:
                            transfer_ranges.append((m.cpu_start, m.cpu_end))
                            transfer_targets.append((flat_idx, si, m.cpu_end - m.cpu_start))

            if transfer_ranges:
                batch_cpu_k = torch.cat([self.cpu_restore_k[s:e] for s, e in transfer_ranges])
                batch_cpu_v = torch.cat([self.cpu_restore_v[s:e] for s, e in transfer_ranges])
                batch_gpu_k = batch_cpu_k.to(self.device)
                batch_gpu_v = batch_cpu_v.to(self.device)
                total_tokens = sum(n for _, _, n in transfer_targets)
                total_bytes = int((batch_cpu_k.numel() + batch_cpu_v.numel()) * batch_cpu_k.element_size())
                self.stats.record_cpu_gpu_transfer(
                    sample_idx, c2g_tokens=total_tokens, c2g_bytes=total_bytes
                )
                sizes = [n for _, _, n in transfer_targets]
                gpu_k_parts = batch_gpu_k.split(sizes)
                gpu_v_parts = batch_gpu_v.split(sizes)
            else:
                gpu_k_parts, gpu_v_parts = [], []

            pieces_k, pieces_v, new_lens = [], [], []
            part_idx = 0
            load_rel_offsets = []
            for flat_idx in range(self.batch_size * self.num_kv_heads):
                s, e = cu[flat_idx], cu[flat_idx + 1]
                pieces_k.append(old_k[s:e])
                pieces_v.append(old_v[s:e])
                rel_cursor = e - s
                extra = 0
                while part_idx < len(transfer_targets) and transfer_targets[part_idx][0] == flat_idx:
                    _, seg_idx, n = transfer_targets[part_idx]
                    pieces_k.append(gpu_k_parts[part_idx])
                    pieces_v.append(gpu_v_parts[part_idx])
                    load_rel_offsets.append((flat_idx, seg_idx, rel_cursor, rel_cursor + n))
                    rel_cursor += n
                    extra += n
                    part_idx += 1
                new_lens.append(e - s + extra)

            self.layer_k[layer_idx] = torch.cat(pieces_k)
            self.layer_v[layer_idx] = torch.cat(pieces_v)
            self.seq_lens[layer_idx] = new_lens
            self.cu_seqlens_k[layer_idx] = _build_cu_seqlens(new_lens, self.device)
            self.cu_seqlens_list[layer_idx] = _build_cu_seqlens_list(new_lens)

            for flat_idx, seg_idx, rel_start, rel_end in load_rel_offsets:
                b, kv_h = flat_idx // self.num_kv_heads, flat_idx % self.num_kv_heads
                meta = self.restore_meta[layer_idx][b][kv_h][seg_idx]
                if meta is not None:
                    meta.gpu_rel_start = rel_start
                    meta.gpu_rel_end = rel_end

    def _rebuild_append_layer(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        from algorithms.wnw.code.kernels import rebuild_append_kv

        if self._pending_transfer_event is not None and not self._pending_transfer_waited:
            torch.cuda.current_stream(self.device).wait_event(self._pending_transfer_event)
            self._pending_transfer_waited = True

        new_k_flat = new_k.squeeze(2).reshape(-1, self.head_dim)
        new_v_flat = new_v.squeeze(2).reshape(-1, self.head_dim)
        old_k, old_v = self.layer_k[layer_idx], self.layer_v[layer_idx]
        cu_old = self.cu_seqlens_k[layer_idx]
        cu_new = self._pending_cu_new[layer_idx]

        new_k_tensor, new_v_tensor = rebuild_append_kv(
            old_k, old_v,
            new_k_flat, new_v_flat,
            self._pending_gpu_k, self._pending_gpu_v,
            cu_old, cu_new,
            self._pending_load_starts[layer_idx],
            self._pending_load_lens[layer_idx],
            self._pending_evict_offsets[layer_idx],
            self._pending_evict_starts[layer_idx],
            self._pending_evict_ends[layer_idx],
            self._pending_evict_max[layer_idx],
        )

        new_lens = self._pending_new_lens[layer_idx]
        self.layer_k[layer_idx] = new_k_tensor
        self.layer_v[layer_idx] = new_v_tensor
        self.seq_lens[layer_idx] = new_lens
        self.cu_seqlens_k[layer_idx] = cu_new
        self.cu_seqlens_list[layer_idx] = _build_cu_seqlens_list(new_lens)

        self._pending_layers_applied += 1
        if self._pending_layers_applied == self.num_layers:
            self._pending_chunk_update = False
            self._clear_pending_rebuild_state()

        return (new_k_tensor, new_v_tensor, cu_new, max(new_lens))

    def _prepare_pending_rebuild(self):
        self._clear_pending_rebuild_state()
        num_segs = self.batch_size * self.num_kv_heads
        all_cpu_ranges = []
        layer_load_records: list[list[tuple[int, int, int, int]]] = [[] for _ in range(self.num_layers)]
        layer_evict_ranges: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(num_segs)] for _ in range(self.num_layers)
        ]
        layer_new_lens: list[list[int]] = []

        global_transfer_start = 0
        changed_samples = [
            b for b in range(self.batch_size)
            if self._segments_to_load[b] or self._segments_to_evict[b]
        ]
        for layer_idx in range(self.num_layers):
            new_lens = [length + 1 for length in self.seq_lens[layer_idx]]
            cu = self.cu_seqlens_list[layer_idx]
            for b in changed_samples:
                for kv_h in self._class_b_heads_by_layer[layer_idx]:
                    flat_idx = b * self.num_kv_heads + kv_h
                    s, e = cu[flat_idx], cu[flat_idx + 1]
                    seg_len = e - s
                    evicted = 0

                    if self._segments_to_evict[b]:
                        evict_ranges = []
                        for seg_idx in self._segments_to_evict[b]:
                            meta = self.restore_meta[layer_idx][b][kv_h][seg_idx]
                            if meta is not None and meta.gpu_rel_start >= 0:
                                evict_ranges.append((meta.gpu_rel_start, meta.gpu_rel_end))
                                meta.gpu_rel_start = meta.gpu_rel_end = -1
                        evict_ranges.sort()
                        layer_evict_ranges[layer_idx][flat_idx] = evict_ranges
                        evicted = sum(ee - es for es, ee in evict_ranges)

                        if evict_ranges:
                            for si in range(len(self.restore_meta[layer_idx][b][kv_h])):
                                meta = self.restore_meta[layer_idx][b][kv_h][si]
                                if meta is None or meta.gpu_rel_start < 0 or si in self._segments_to_evict[b]:
                                    continue
                                shift = sum(ee - es for es, ee in evict_ranges if ee <= meta.gpu_rel_start)
                                meta.gpu_rel_start -= shift
                                meta.gpu_rel_end -= shift

                    rel_cursor = seg_len - evicted
                    loaded = 0
                    if self._segments_to_load[b]:
                        for seg_idx in sorted(self._segments_to_load[b]):
                            meta = self.restore_meta[layer_idx][b][kv_h][seg_idx]
                            if meta is None:
                                continue
                            n = meta.cpu_end - meta.cpu_start
                            all_cpu_ranges.append((meta.cpu_start, meta.cpu_end))
                            layer_load_records[layer_idx].append((flat_idx, seg_idx, global_transfer_start, n))
                            meta.gpu_rel_start = rel_cursor
                            meta.gpu_rel_end = rel_cursor + n
                            rel_cursor += n
                            loaded += n
                            global_transfer_start += n
                            self.stats.record_cpu_gpu_transfer(
                                b,
                                c2g_tokens=int(n),
                                c2g_bytes=int(n * self.head_dim * 2 * self.cpu_restore_k.element_size()),
                            )

                    new_lens[flat_idx] = seg_len - evicted + loaded + 1
            layer_new_lens.append(new_lens)

        load_starts_rows = []
        load_lens_rows = []
        cu_new_rows = []

        for layer_idx in range(self.num_layers):
            load_starts_list = [-1] * num_segs
            load_lens_list = [0] * num_segs
            for flat_idx, _, transfer_start, n in layer_load_records[layer_idx]:
                if load_starts_list[flat_idx] < 0:
                    load_starts_list[flat_idx] = transfer_start
                load_lens_list[flat_idx] += n
            load_starts_rows.append(load_starts_list)
            load_lens_rows.append(load_lens_list)
            cu_new_rows.append(_build_cu_seqlens_list(layer_new_lens[layer_idx]))

            flat_evict_starts, flat_evict_ends = [], []
            offsets = [0]
            for flat_idx in range(num_segs):
                for start, end in layer_evict_ranges[layer_idx][flat_idx]:
                    flat_evict_starts.append(start)
                    flat_evict_ends.append(end)
                offsets.append(len(flat_evict_starts))
            self._pending_evict_max[layer_idx] = max(
                (len(ranges) for ranges in layer_evict_ranges[layer_idx]),
                default=0,
            )
            self._pending_evict_offsets[layer_idx] = torch.tensor(offsets, dtype=torch.int32, device=self.device)
            if flat_evict_starts:
                self._pending_evict_starts[layer_idx] = torch.tensor(flat_evict_starts, dtype=torch.int32, device=self.device)
                self._pending_evict_ends[layer_idx] = torch.tensor(flat_evict_ends, dtype=torch.int32, device=self.device)
            else:
                self._pending_evict_starts[layer_idx] = self._empty_i32
                self._pending_evict_ends[layer_idx] = self._empty_i32
            self._pending_new_lens[layer_idx] = layer_new_lens[layer_idx]

        self._pending_load_starts_2d = torch.tensor(load_starts_rows, dtype=torch.int32, device=self.device)
        self._pending_load_lens_2d = torch.tensor(load_lens_rows, dtype=torch.int32, device=self.device)
        self._pending_cu_new_2d = torch.tensor(cu_new_rows, dtype=torch.int32, device=self.device)
        for layer_idx in range(self.num_layers):
            self._pending_load_starts[layer_idx] = self._pending_load_starts_2d[layer_idx]
            self._pending_load_lens[layer_idx] = self._pending_load_lens_2d[layer_idx]
            self._pending_cu_new[layer_idx] = self._pending_cu_new_2d[layer_idx]

        if all_cpu_ranges:
            total_tokens = sum(e - s for s, e in all_cpu_ranges)
            if self.device.type == "cuda":
                self._ensure_staging_capacity(total_tokens, self.cpu_restore_k.dtype)
                torch.cat([self.cpu_restore_k[s:e] for s, e in all_cpu_ranges], dim=0,
                          out=self._staging_k[:total_tokens])
                torch.cat([self.cpu_restore_v[s:e] for s, e in all_cpu_ranges], dim=0,
                          out=self._staging_v[:total_tokens])
                stream = self._get_transfer_stream()
                self._pending_transfer_event = torch.cuda.Event()
                with torch.cuda.stream(stream):
                    self._pending_gpu_k = self._staging_k[:total_tokens].to(self.device, non_blocking=True)
                    self._pending_gpu_v = self._staging_v[:total_tokens].to(self.device, non_blocking=True)
                    self._pending_transfer_event.record(stream)
            else:
                pending_cpu_k = torch.cat([self.cpu_restore_k[s:e] for s, e in all_cpu_ranges])
                pending_cpu_v = torch.cat([self.cpu_restore_v[s:e] for s, e in all_cpu_ranges])
                self._pending_gpu_k = pending_cpu_k.to(self.device)
                self._pending_gpu_v = pending_cpu_v.to(self.device)
        else:
            self._pending_gpu_k = torch.empty(0, self.head_dim, dtype=self.layer_k[0].dtype, device=self.device)
            self._pending_gpu_v = torch.empty(0, self.head_dim, dtype=self.layer_v[0].dtype, device=self.device)

    def _apply_chunk_update_to_layer(self, layer_idx: int):
        old_k, old_v = self.layer_k[layer_idx], self.layer_v[layer_idx]
        cu = self.cu_seqlens_list[layer_idx]
        num_segs = self.batch_size * self.num_kv_heads

        transfer_ranges, transfer_targets = [], []
        for flat_idx in range(num_segs):
            b, kv_h = flat_idx // self.num_kv_heads, flat_idx % self.num_kv_heads
            if self.config.is_class_b(layer_idx, kv_h) and self._segments_to_load[b]:
                for si in sorted(self._segments_to_load[b]):
                    m = self.restore_meta[layer_idx][b][kv_h][si]
                    if m is not None:
                        transfer_ranges.append((m.cpu_start, m.cpu_end))
                        transfer_targets.append((flat_idx, si, m.cpu_end - m.cpu_start))

        if transfer_ranges:
            batch_cpu_k = torch.cat([self.cpu_restore_k[s:e] for s, e in transfer_ranges])
            batch_cpu_v = torch.cat([self.cpu_restore_v[s:e] for s, e in transfer_ranges])
            batch_gpu_k = batch_cpu_k.to(self.device, non_blocking=True)
            batch_gpu_v = batch_cpu_v.to(self.device, non_blocking=True)
            for flat_idx, si, n in transfer_targets:
                b = flat_idx // self.num_kv_heads
                self.stats.record_cpu_gpu_transfer(
                    b,
                    c2g_tokens=int(n),
                    c2g_bytes=int(n * self.head_dim * 2 * batch_cpu_k.element_size()),
                )
            sizes = [n for _, _, n in transfer_targets]
        else:
            batch_gpu_k = batch_gpu_v = None
            sizes = []

        if transfer_ranges:
            gpu_k_parts = batch_gpu_k.split(sizes)
            gpu_v_parts = batch_gpu_v.split(sizes)
        else:
            gpu_k_parts, gpu_v_parts = [], []

        has_load = {ft[0] for ft in transfer_targets}
        pieces_k, pieces_v, new_lens = [], [], []
        part_idx = 0
        load_rel_offsets = []
        run_start_row = None

        def _flush_run():
            nonlocal run_start_row, run_end_row
            if run_start_row is not None:
                pieces_k.append(old_k[run_start_row:run_end_row])
                pieces_v.append(old_v[run_start_row:run_end_row])
                run_start_row = None

        run_end_row = None
        for flat_idx in range(num_segs):
            s, e = cu[flat_idx], cu[flat_idx + 1]
            b, kv_h = flat_idx // self.num_kv_heads, flat_idx % self.num_kv_heads
            seg_len = e - s

            is_b = self.config.is_class_b(layer_idx, kv_h)
            changed = (flat_idx in has_load) or (is_b and bool(self._segments_to_evict[b]))

            if not changed:
                if run_start_row is None:
                    run_start_row = s
                run_end_row = e
                new_lens.append(seg_len)
                continue

            _flush_run()

            seg_k, seg_v = old_k[s:e], old_v[s:e]

            if is_b and self._segments_to_evict[b]:
                evict_ranges = []
                for seg_idx in self._segments_to_evict[b]:
                    meta = self.restore_meta[layer_idx][b][kv_h][seg_idx]
                    if meta is not None and meta.gpu_rel_start >= 0:
                        evict_ranges.append((meta.gpu_rel_start, meta.gpu_rel_end))
                        meta.gpu_rel_start = meta.gpu_rel_end = -1
                if evict_ranges:
                    evict_ranges.sort()
                    keep_slices = _invert_ranges(evict_ranges, seg_len)
                    seg_k = torch.cat([seg_k[a:b] for a, b in keep_slices])
                    seg_v = torch.cat([seg_v[a:b] for a, b in keep_slices])
                    for si in range(len(self.restore_meta[layer_idx][b][kv_h])):
                        m = self.restore_meta[layer_idx][b][kv_h][si]
                        if m is None or m.gpu_rel_start < 0 or si in self._segments_to_evict[b]:
                            continue
                        shift = sum(ee - es for es, ee in evict_ranges if ee <= m.gpu_rel_start)
                        m.gpu_rel_start -= shift
                        m.gpu_rel_end -= shift

            rel_cursor = seg_k.shape[0]
            pieces_k.append(seg_k)
            pieces_v.append(seg_v)

            extra = 0
            while part_idx < len(transfer_targets) and transfer_targets[part_idx][0] == flat_idx:
                _, seg_idx, n = transfer_targets[part_idx]
                pieces_k.append(gpu_k_parts[part_idx])
                pieces_v.append(gpu_v_parts[part_idx])
                load_rel_offsets.append((flat_idx, seg_idx, rel_cursor, rel_cursor + n))
                rel_cursor += n
                extra += n
                part_idx += 1
            new_lens.append(seg_k.shape[0] + extra)

        _flush_run()

        self.layer_k[layer_idx] = torch.cat(pieces_k)
        self.layer_v[layer_idx] = torch.cat(pieces_v)
        self.seq_lens[layer_idx] = new_lens
        self.cu_seqlens_k[layer_idx] = _build_cu_seqlens(new_lens, self.device)
        self.cu_seqlens_list[layer_idx] = _build_cu_seqlens_list(new_lens)

        for flat_idx, seg_idx, rel_start, rel_end in load_rel_offsets:
            b, kv_h = flat_idx // self.num_kv_heads, flat_idx % self.num_kv_heads
            meta = self.restore_meta[layer_idx][b][kv_h][seg_idx]
            if meta is not None:
                meta.gpu_rel_start = rel_start
                meta.gpu_rel_end = rel_end


    def _snapshot_cache_stats(self, step: int = -1) -> list[dict]:
        cpu_total_all = self.cpu_restore_k.shape[0] if self.cpu_restore_k is not None else 0

        decode_text_per_seg = self._decode_step if self.prefilled else 0
        padded_prefill_len = (self._seq_lengths[0] - self._decode_step) if self._seq_lengths else 0

        stats = []
        for b in range(self.batch_size):
            num_audio = self._audio_ends_list[b] - self._audio_starts_list[b]
            full_audio = self.num_layers * self.num_kv_heads * num_audio

            s_prefill_unpadded = max(0, padded_prefill_len - self.pad_lens_list[b])
            post_audio_text = max(0, s_prefill_unpadded - self._audio_ends_list[b])
            non_audio = self._audio_starts_list[b] + post_audio_text

            gpu_audio_tokens = 0
            gpu_total_tokens = 0
            per_layer_audio = []

            for layer_idx in range(self.num_layers):
                if self.seq_lens[layer_idx] is None:
                    per_layer_audio.append(0)
                    continue
                layer_audio = 0
                for kv_h in range(self.num_kv_heads):
                    flat_idx = b * self.num_kv_heads + kv_h
                    n_total = self.seq_lens[layer_idx][flat_idx]
                    gpu_total_tokens += n_total
                    n_audio = n_total - non_audio - decode_text_per_seg
                    n_audio = max(0, n_audio)
                    gpu_audio_tokens += n_audio
                    layer_audio += n_audio
                per_layer_audio.append(layer_audio)

            cpu_restore = 0
            for layer_idx in range(self.num_layers):
                for kv_h in range(self.num_kv_heads):
                    if not self.config.is_class_b(layer_idx, kv_h):
                        continue
                    for m in self.restore_meta[layer_idx][b][kv_h]:
                        if m is not None:
                            cpu_restore += m.cpu_end - m.cpu_start

            cpu_loaded = 0
            if self.prefilled:
                for seg_idx in self.loaded_segments[b]:
                    for layer_idx in range(self.num_layers):
                        for kv_h in range(self.num_kv_heads):
                            if not self.config.is_class_b(layer_idx, kv_h):
                                continue
                            m = self.restore_meta[layer_idx][b][kv_h][seg_idx]
                            if m is not None:
                                cpu_loaded += m.cpu_end - m.cpu_start

            stats.append({
                "step": step,
                "num_audio": num_audio,
                "gpu_audio_tokens": gpu_audio_tokens,
                "gpu_total_tokens": gpu_total_tokens,
                "audio_retention": gpu_audio_tokens / full_audio if full_audio > 0 else 1.0,
                "per_layer_audio_tokens": per_layer_audio,
                "cpu_total_tokens": cpu_total_all,
                "cpu_loaded_tokens": cpu_loaded,
                "cpu_restore_tokens": cpu_restore,
                "gpu_plus_cpu_retention": (gpu_audio_tokens + cpu_restore) / full_audio
                    if full_audio > 0 else 1.0,
            })
        return stats

    def get_snapshot_stats(self, step: int = -1) -> list[dict]:
        """Backwards-compatible wrapper for _snapshot_cache_stats."""
        return self._snapshot_cache_stats(step=step)

    def get_compression_stats(self) -> list[dict]:
        """Backwards-compatible interface, equivalent to get_snapshot_stats(step=-1)."""
        return self._snapshot_cache_stats(step=-1)



def _compute_segments(num_audio: int, stride: int) -> list[tuple[int, int]]:
    return [(i * stride, min(i * stride + stride, num_audio))
            for i in range(math.ceil(num_audio / stride))]


def _compute_chunks(segments: list[tuple[int, int]], chunk_size: int,
                    stride: int) -> list[list[int]]:
    segs_per_chunk = math.ceil(chunk_size / stride)
    n = len(segments)
    return [list(range(i, min(i + segs_per_chunk, n))) for i in range(n)]


def _build_cu_seqlens(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _build_cu_seqlens_list(seq_lens: list[int]) -> list[int]:
    return [0] + list(accumulate(seq_lens))


def _select_topk_per_segment(scores: torch.Tensor, seg_ranges: list[tuple[int, int]],
                              ratio: float) -> torch.Tensor:
    num_audio = scores.shape[0]
    kept = torch.zeros(num_audio, dtype=torch.bool, device=scores.device)
    if not seg_ranges:
        return kept

    stride = seg_ranges[0][1] - seg_ranges[0][0]
    last_len = seg_ranges[-1][1] - seg_ranges[-1][0]
    n_full = len(seg_ranges) if last_len == stride else len(seg_ranges) - 1

    if n_full > 0:
        n_keep = max(1, int(stride * ratio))
        block = scores[:n_full * stride].reshape(n_full, stride)
        idx = block.topk(n_keep, dim=1).indices
        row_off = (torch.arange(n_full, device=scores.device) * stride).unsqueeze(1)
        kept[(idx + row_off).reshape(-1)] = True

    if n_full < len(seg_ranges):
        seg_start, seg_end = seg_ranges[-1]
        n_keep = max(1, int((seg_end - seg_start) * ratio))
        topk_idx = scores[seg_start:seg_end].topk(n_keep).indices + seg_start
        kept[topk_idx] = True

    return kept


def _select_topk_global(scores: torch.Tensor, seg_ranges: list[tuple[int, int]],
                         ratio: float) -> torch.Tensor:
    num_audio = scores.shape[0]
    n_keep_total = sum(max(1, int((seg_end - seg_start) * ratio))
                       for seg_start, seg_end in seg_ranges)
    n_keep_total = min(n_keep_total, num_audio)
    kept = torch.zeros(num_audio, dtype=torch.bool, device=scores.device)
    topk_idx = scores.topk(n_keep_total).indices
    kept[topk_idx] = True
    return kept


def _invert_ranges(ranges: list[tuple[int, int]], total_len: int) -> list[tuple[int, int]]:
    result = []
    prev = 0
    for s, e in ranges:
        if prev < s:
            result.append((prev, s))
        prev = e
    if prev < total_len:
        result.append((prev, total_len))
    return result



def debug_compare_attention_output(
    varlen_output: torch.Tensor,
    eager_output: torch.Tensor,
    label: str = "",
):
    """Compare varlen and eager attention output consistency."""
    cos = torch.nn.functional.cosine_similarity(
        varlen_output.float().flatten().unsqueeze(0),
        eager_output.float().flatten().unsqueeze(0),
    ).item()
    max_diff = (varlen_output.float() - eager_output.float()).abs().max().item()
    print(f"[DEBUG {label}] cosine={cos:.6f}, max_abs_diff={max_diff:.6f}")
    assert cos > 0.99, f"Cosine too low: {cos}"
