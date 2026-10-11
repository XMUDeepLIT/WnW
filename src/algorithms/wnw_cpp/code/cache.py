
from __future__ import annotations

import os

import torch

from common.kv.kernels import interleave_append
from algorithms.wnw.code.cache import WnWCache


class WnWCppCache(WnWCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fallback = os.environ.get("WNW_CPP_FALLBACK", "0") == "1"
        self._planner = None
        self._cpp_load_starts_2d = None
        self._cpp_load_lens_2d = None
        self._cpp_cu_new_2d = None
        self._cpp_new_lens_2d = None
        self._cpp_evict_offsets_2d = None
        self._cpp_evict_max = None
        self._cpp_evict_layer_ptr = None
        self._topk_pending = False
        self._topk_idx_masked = None

    def setup_audio(self, audio_starts, audio_ends, attention_mask=None):
        self._planner = None
        self._topk_pending = False
        self._topk_idx_masked = None
        super().setup_audio(audio_starts, audio_ends, attention_mask)


    def prefill_all_layers_done(self):
        super().prefill_all_layers_done()
        if self._fallback:
            return
        self._init_planner()

    def _init_planner(self):
        from algorithms.wnw_cpp.code._ext import get_planner_module

        mod = get_planner_module()
        L, H, B = self.num_layers, self.num_kv_heads, self.batch_size
        num_segs = B * H

        self._planner = mod.Planner(L, H, B, self.num_q_heads // self.num_kv_heads)
        self._planner.set_config(self.config.top_k_chunks, self.config.eviction_lag)

        seg_count = [len(self.segment_ranges[b]) for b in range(B)]
        self._planner.set_seg_count(seg_count)

        for b in range(B):
            flat, off = [], [0]
            for segs in self.chunk_to_segs[b]:
                flat.extend(segs)
                off.append(len(flat))
            self._planner.set_chunk_to_segs(
                b,
                torch.tensor(flat, dtype=torch.int32),
                torch.tensor(off, dtype=torch.int32),
            )

        for l in range(L):
            self._planner.set_class_b(
                l, torch.tensor(self._class_b_heads_by_layer[l], dtype=torch.int32)
            )

        cpu_s, cpu_e, gpu_s, gpu_e = [], [], [], []
        for l in range(L):
            for fi in range(num_segs):
                b, kv_h = fi // H, fi % H
                for m in self.restore_meta[l][b][kv_h]:
                    if m is None:
                        cpu_s.append(-1); cpu_e.append(-1)
                        gpu_s.append(-1); gpu_e.append(-1)
                    else:
                        cpu_s.append(m.cpu_start); cpu_e.append(m.cpu_end)
                        gpu_s.append(m.gpu_rel_start); gpu_e.append(m.gpu_rel_end)
        self._planner.set_restore_meta(
            torch.tensor(cpu_s, dtype=torch.int32),
            torch.tensor(cpu_e, dtype=torch.int32),
            torch.tensor(gpu_s, dtype=torch.int32),
            torch.tensor(gpu_e, dtype=torch.int32),
        )

        for b in range(B):
            self._planner.set_loaded(
                b, torch.tensor(sorted(self.loaded_segments[b]), dtype=torch.int32)
            )

        sl = []
        for l in range(L):
            sl.extend(self.seq_lens[l])
        self._planner.set_seq_lens(torch.tensor(sl, dtype=torch.int32))

        self._cpp_load_starts_2d = torch.empty(L, num_segs, dtype=torch.int32)
        self._cpp_load_lens_2d = torch.empty(L, num_segs, dtype=torch.int32)
        self._cpp_cu_new_2d = torch.empty(L, num_segs + 1, dtype=torch.int32)
        self._cpp_new_lens_2d = torch.empty(L, num_segs, dtype=torch.int32)
        self._cpp_evict_offsets_2d = torch.empty(L, num_segs + 1, dtype=torch.int32)
        self._cpp_evict_max = torch.empty(L, dtype=torch.int32)
        self._cpp_evict_layer_ptr = torch.empty(L + 1, dtype=torch.int32)

        if self.cpu_restore_k is not None and self.cpu_restore_k.shape[0] > 0:
            n_restore = self.cpu_restore_k.shape[0]
            self._staging_k = torch.empty(
                n_restore, self.head_dim, dtype=self.cpu_restore_k.dtype, pin_memory=True
            )
            self._staging_v = torch.empty(
                n_restore, self.head_dim, dtype=self.cpu_restore_v.dtype, pin_memory=True
            )
            self._staging_capacity = n_restore
            self._planner.set_restore_buffers(
                self.cpu_restore_k, self.cpu_restore_v, self._staging_k, self._staging_v
            )

        self._setup_topk_vectorization()

    def _setup_topk_vectorization(self):
        B = self.batch_size
        device = self.device
        audio_lens = [
            self._score_offsets_list[b + 1] - self._score_offsets_list[b]
            for b in range(B)
        ]
        num_chunks = [len(self.chunk_to_segs[b]) for b in range(B)]
        self._tk_max_audio = max(audio_lens) if audio_lens else 0
        self._tk_max_chunks = max(num_chunks) if num_chunks else 0
        self._tk_k = min(self.config.top_k_chunks, self._tk_max_chunks)
        W = self._tk_max_audio + 1

        scatter = []
        for b in range(B):
            base = b * W + 1
            scatter.extend(range(base, base + audio_lens[b]))
        self._tk_scatter_idx = torch.tensor(scatter, dtype=torch.int64, device=device)

        gk_start, gk_end, gk_lens, gk_grid = [], [], [], []
        for b in range(B):
            row = b * W
            for c, (cs, ce) in enumerate(self._chunk_token_bounds[b]):
                gk_start.append(row + cs)
                gk_end.append(row + ce)
                gk_lens.append(max(1, ce - cs))
                gk_grid.append(b * self._tk_max_chunks + c)
        self._tk_gk_start = torch.tensor(gk_start, dtype=torch.int64, device=device)
        self._tk_gk_end = torch.tensor(gk_end, dtype=torch.int64, device=device)
        self._tk_gk_lens = torch.tensor(gk_lens, dtype=torch.float32, device=device)
        self._tk_gk_grid = torch.tensor(gk_grid, dtype=torch.int64, device=device)

        self._tk_padded = torch.zeros(B, W, dtype=torch.float32, device=device)
        self._tk_prefix = torch.empty(B, W, dtype=torch.float32, device=device)
        self._tk_padded_flat = self._tk_padded.view(-1)
        self._tk_prefix_flat = self._tk_prefix.view(-1)
        self._tk_grid = torch.full(
            (B * self._tk_max_chunks,), float("-inf"), dtype=torch.float32, device=device
        )

    def _topk_scores_async(self):
        if self._tk_k == 0:
            return None
        self._tk_padded_flat.index_copy_(0, self._tk_scatter_idx, self.score_accumulator)
        torch.cumsum(self._tk_padded, dim=1, out=self._tk_prefix)
        pf = self._tk_prefix_flat
        means = (pf[self._tk_gk_end] - pf[self._tk_gk_start]) / self._tk_gk_lens
        self._tk_grid.index_copy_(0, self._tk_gk_grid, means)
        grid2d = self._tk_grid.view(self.batch_size, self._tk_max_chunks)
        vals, idx = grid2d.topk(self._tk_k, dim=1)
        idx32 = idx.to(torch.int32)
        return torch.where(vals > float("-inf"), idx32, torch.full_like(idx32, -1))

    def _topk_finalize(self, idx_masked):
        if idx_masked is None:
            return (
                torch.empty(0, dtype=torch.int32),
                torch.zeros(self.batch_size + 1, dtype=torch.int32),
            )
        rows = idx_masked.tolist()

        topk_list, offsets = [], [0]
        for row in rows:
            for v in row:
                if v >= 0:
                    topk_list.append(v)
            offsets.append(len(topk_list))
        return (
            torch.tensor(topk_list, dtype=torch.int32),
            torch.tensor(offsets, dtype=torch.int32),
        )


    def decode_before_attn(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        if self._fallback:
            return super().decode_before_attn(layer_idx, new_k, new_v)

        if layer_idx == 0 and self._topk_pending:
            self._finalize_pending_plan()

        if self._pending_chunk_update:
            return super().decode_before_attn(layer_idx, new_k, new_v)

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

        return (new_k_tensor, new_v_tensor, cu_new, max(new_lens))

    def _finalize_pending_plan(self):
        self._topk_pending = False
        topk_t, off_t = self._topk_finalize(self._topk_idx_masked)
        self._topk_idx_masked = None

        any_change, total, ev_s, ev_e, ranges = self._planner.compute_and_prepare(
            topk_t, off_t,
            self._cpp_load_starts_2d, self._cpp_load_lens_2d,
            self._cpp_cu_new_2d, self._cpp_new_lens_2d,
            self._cpp_evict_offsets_2d, self._cpp_evict_max,
            self._cpp_evict_layer_ptr,
        )

        if any_change and self.use_triton_rebuild:
            self._pending_chunk_update = True
            self._build_pending_from_plan(int(total), ev_s, ev_e, ranges)
        else:
            self._pending_chunk_update = False

    def decode_all_layers_done(self):
        if self._fallback:
            return super().decode_all_layers_done()

        self._decode_step += 1
        self.stats.flush_decode_step(self._decode_step)
        if self.stats.enabled and self._decode_step % self.stats.interval == 0:
            self.stats.record_decode(self._decode_step, self._snapshot_cache_stats)

        self._topk_idx_masked = self._topk_scores_async()
        self._topk_pending = True
        self.score_accumulator.zero_()

    def _build_pending_from_plan(self, total: int, ev_s, ev_e, ranges):
        self._clear_pending_rebuild_state()
        L = self.num_layers

        load_starts_2d = self._cpp_load_starts_2d.to(self.device)
        load_lens_2d = self._cpp_load_lens_2d.to(self.device)
        cu_new_2d = self._cpp_cu_new_2d.to(self.device)
        self._pending_load_starts_2d = load_starts_2d
        self._pending_load_lens_2d = load_lens_2d
        self._pending_cu_new_2d = cu_new_2d

        ev_off_gpu = self._cpp_evict_offsets_2d.to(self.device)
        ev_s_gpu = ev_s.to(self.device) if ev_s.numel() else self._empty_i32
        ev_e_gpu = ev_e.to(self.device) if ev_e.numel() else self._empty_i32
        elp = self._cpp_evict_layer_ptr.tolist()
        emax = self._cpp_evict_max.tolist()
        new_lens_all = self._cpp_new_lens_2d.tolist()

        for l in range(L):
            self._pending_load_starts[l] = load_starts_2d[l]
            self._pending_load_lens[l] = load_lens_2d[l]
            self._pending_cu_new[l] = cu_new_2d[l]
            self._pending_new_lens[l] = new_lens_all[l]
            self._pending_evict_offsets[l] = ev_off_gpu[l]
            self._pending_evict_max[l] = emax[l]
            lo, hi = elp[l], elp[l + 1]
            if hi > lo:
                self._pending_evict_starts[l] = ev_s_gpu[lo:hi]
                self._pending_evict_ends[l] = ev_e_gpu[lo:hi]
            else:
                self._pending_evict_starts[l] = self._empty_i32
                self._pending_evict_ends[l] = self._empty_i32

        if total > 0:
            if self.device.type == "cuda":
                self._planner.gather_restore()
                stream = self._get_transfer_stream()
                self._pending_transfer_event = torch.cuda.Event()
                with torch.cuda.stream(stream):
                    self._pending_gpu_k = self._staging_k[:total].to(self.device, non_blocking=True)
                    self._pending_gpu_v = self._staging_v[:total].to(self.device, non_blocking=True)
                    self._pending_transfer_event.record(stream)
            else:
                ranges_pairs = ranges.tolist()
                pending_cpu_k = torch.cat([self.cpu_restore_k[s:e] for s, e in ranges_pairs])
                pending_cpu_v = torch.cat([self.cpu_restore_v[s:e] for s, e in ranges_pairs])
                self._pending_gpu_k = pending_cpu_k.to(self.device)
                self._pending_gpu_v = pending_cpu_v.to(self.device)
        else:
            self._pending_gpu_k = torch.empty(0, self.head_dim, dtype=self.layer_k[0].dtype, device=self.device)
            self._pending_gpu_v = torch.empty(0, self.head_dim, dtype=self.layer_v[0].dtype, device=self.device)


    def _snapshot_cache_stats(self, step: int = -1) -> list[dict]:
        if self._planner is not None:
            for b in range(self.batch_size):
                self.loaded_segments[b] = set(self._planner.get_loaded_segments(b))
        return super()._snapshot_cache_stats(step=step)
