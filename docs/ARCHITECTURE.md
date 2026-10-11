# WnW Architecture & Design

This document describes the design of the framework and the constraints that
keep it correct. WnW is a model-agnostic approach to KV-cache compression on
audio LLMs.

---

## 1. Overview

The core goal is **zero side-channel**: the model object is never polluted with
algorithm-specific attributes. All state lives in a HuggingFace `Cache`
subclass that is passed explicitly to `model.generate(..., past_key_values=cache)`.

Supported models:

- Voxtral (`voxtral`)
- Qwen2.5-Omni (`qwen2_5_omni`)

Adding a new model only requires a new adapter file in `src/common/adapters/`.

Algorithms:

- `baseline` — no compression, used as a reference.
- `wnw` — class A/B head classification plus chunk-level CPU offload (Python).
- `wnw_cpp` — the same decode planning logic implemented in C++ to reduce
  host-side overhead; cache operations and kernels are shared with `wnw`.

---

## 2. Design Principles

| # | Decision | Rationale |
|---|---|---|
| D1 | WnW shares `FlatVarlenCache` for varlen KV storage. | One flat, `cu_seqlens`-indexed store serves every head/sample. |
| D2 | State is held in a HF `Cache` subclass, not on the model. | Zero pollution; the same model can run different algorithms sequentially. |
| D3 | `AlgorithmSession` installs and removes `patched_forward` in a `with` block. | Patches are scoped and always restored on exit. |
| D4 | Audio token ranges are half-open `[start, end)`. | Consistent, off-by-one-safe treatment across Voxtral and Qwen. |
| D5 | RoPE is applied by the adapter; `patched_forward` is model-agnostic. | The only model-specific rope variant lives in `adapter.apply_rope`. |
| D6 | Prefill uses SDPA plus hand-written per-kv-head scores for all models. | Unified prefill path; no fallback to the original forward. |
| D7 | `Cache.update()` writes directly into flat storage. | No per-layer closures or dummy cache layers. |
| D8 | `reorder_cache` / `crop` raise. | Beam search and assisted decoding are intentionally unsupported. |
| D9 | `transformers==5.7.0` is the supported version. | The HF `Cache` contract and RoPE signatures are version-sensitive. |

---

## 3. Core Abstractions

### 3.1 `BaseModelAdapter` — `src/common/adapters/base.py`

The adapter hides all model-specific detail from the algorithm. Key methods:

| Method | Purpose |
|---|---|
| `load(model_path, device, degqa=...)` | Load model + processor. |
| `build_inputs(audio, sr, text, processor, device)` | Build a model input dict from audio + prompt. |
| `get_decoder_layers(model)` / `get_text_config(model)` | Decoder layers and the text config (heads, layers, head_dim). |
| `locate_audio_tokens(input_ids, attention_mask, processor)` | Return `(audio_starts, audio_ends)` as half-open `[start, end)` ranges. |
| `compute_qkv(attn_module, hidden_states)` | Compute Q/K/V shaped `[B, heads, S, head_dim]`. |
| `apply_rope(q, k, position_embeddings, attn_module)` | Apply the model-specific RoPE. |
| `prepare_for_inference` / `restore_after_inference` | Save/restore model state around a session. |
| `generate(model, inputs, **kwargs)` | Model-specific `model.generate` wrapper. |
| `token_rate` | Audio tokens per second (Voxtral 12.5, Qwen2.5-Omni 25.0). |
| `build_analysis_inputs(...)` / `degqa(model)` | Analysis helpers; GQA→MHA conversion. |

`assert_compatible` requires a power-of-2 `head_dim` (a Triton kernel
constraint). To add a model, implement these methods and register the adapter
in `src/common/adapters/__init__.py`.

### 3.2 `FlatVarlenCache` — `src/common/kv/flat_cache.py`

A HuggingFace `Cache` subclass that stores all KV in flat buffers indexed by
`cu_seqlens`. Entries are ordered by `(sample, kv_head)`; the flat index of a
`(batch b, kv-head h)` slice is `b * num_kv_heads + h`, and `cu_seqlens` is an
`int32` prefix-sum of length `B * num_kv_heads + 1`. It provides the required HF
methods (`update`, `get_seq_length`, `get_max_cache_shape`) and raises on
`reorder_cache` / `crop`. Algorithm behavior plugs in through hooks:

```python
def prefill_before_attn(self, layer_idx): ...
def prefill_after_attn(self, layer_idx, scores, k, v): ...
def decode_before_attn(self, layer_idx, k_new, v_new): ...
def decode_after_attn(self, layer_idx, q): ...
def _snapshot_cache_stats(self): ...
```

`WnWCache` implements these hooks and adds segment/score/chunk state.

### 3.3 `AlgorithmSession` — `src/common/session.py`

A context manager that, on enter: resets the cache, calls
`assert_compatible`, calls `adapter.prepare_for_inference(model)` (Qwen forces
eager attention here; Voxtral is a no-op), and installs `patched_forward` on
every decoder self-attention module (saving the originals). On exit it restores
the original forwards and calls `adapter.restore_after_inference`.
`Session.generate` is a thin wrapper that calls
`model.generate(..., past_key_values=self.cache)`.

### 3.4 Model-agnostic `patched_forward`

Every model shares one attention path: compute Q/K/V → apply RoPE via
`cache.adapter.apply_rope(...)` → if not prefilled, run prefill SDPA (with KV
heads repeated for GQA) and compute per-kv-head scores → if prefilled, update the
flat cache and run varlen decode attention → project the output. All algorithm
differences live in the cache hooks.

### 3.5 `InferenceStats` — `src/common/monitoring/inference_stats.py`

Each cache owns an `InferenceStats` instance that records GPU memory around
prefill/decode, cache occupancy (`gpu_total_tokens`, `gpu_audio_tokens`,
`audio_retention`), and per-step CPU↔GPU transfer events. It is written to the
eval output JSONL and consumed by the calibration scripts.

---

## 4. WnW Internals

- **Head classes.** Class A heads keep the full audio KV on GPU; Class B heads
  keep only a configured `retention_ratio` on GPU and offload the rest to CPU,
  streaming chunks back during decode. In the config, a Class A entry is a
  **global query-head id** `layer_idx * num_q_heads + local_q`, a Class B entry
  is a **global kv-head id** `layer_idx * num_kv_heads + local_kv`, and
  `retention_ratios` has length `num_layers * num_kv_heads`.
- **Chunk math** (`src/algorithms/wnw/code/config.py`). From the chunk config,
  `chunk_size_tokens = int(size_s * token_rate)`,
  `overlap_tokens = int(overlap_s * token_rate)`, and
  `stride_tokens = chunk_size_tokens - overlap_tokens`. The config asserts
  `stride_tokens > 0` and `chunk_size_tokens % stride_tokens == 0`. Audio is cut
  into stride-sized segments; each chunk spans `ceil(chunk_size / stride)`
  segments.
- **Selection.** During decode, per-chunk scores drive `top_k_chunks` selection
  per segment (or globally via `prefill_global_topk`); `eviction_lag` counters
  delay evicting a chunk that just dropped out of the top-k. Prefill scoring is
  either `"last"` or `"mean_text"` (`prefill_score_mode`).

---

## 5. GQA, RoPE, and mRoPE

- **GQA / degqa.** KV heads are read from `config.num_key_value_heads`. Prefill
  repeats KV heads to match query heads using `repeat_interleave(num_kv_groups)`.
  `adapter.degqa(model)` (`src/common/adapters/degqa.py`) expands `k_proj`/`v_proj`
  by `repeat_interleave`, sets `num_key_value_heads = num_attention_heads`, and is
  bit-identical to `repeat_kv`. The conversion is in-place and irreversible, so a
  GQA-derived config must be expanded to match (see `expand_wnw_config.py`).
- **RoPE.** `adapter.apply_rope` is the single rope hook. Voxtral uses the Llama
  rotary embedding. Qwen2.5-Omni uses multimodal rotary embeddings with an
  `mrope_section` and therefore needs `input_ids` (3D position ids), so its
  adapter sets `requires_input_ids = True`.

---

## 6. Compute Backends

- **Triton.** WnW uses Triton kernels for the decode-time cache rebuild; set
  `WNW_DISABLE_TRITON_REBUILD=1` to fall back to a non-Triton rebuild. The fused
  decode attention path is opt-in via `WNW_FUSED_ATTN=1`. The power-of-2
  `head_dim` requirement (`assert_compatible`) exists for these kernels.
- **C++ planner (`wnw_cpp`).** The decode-step planner is JIT-compiled with
  torch's `cpp_extension.load` (requires `ninja` and a C++ compiler). It
  implements the same planning logic as the Python planner with lower host-side
  overhead. Set `WNW_CPP_FALLBACK=1` to force the pure-Python path for comparison
  or when the extension cannot build. On some systems the loader needs a
  sufficiently new `libstdc++` on `LD_LIBRARY_PATH` for the compiled extension
  to import.

---

## 7. Extending

**Add an algorithm:** create `src/algorithms/<name>/code/{config,cache,session}.py`,
subclass `FlatVarlenCache`, implement the hooks, and add a driver under
`src/benchmarks/scripts/`.

**Add a model:** create `src/common/adapters/<name>.py` implementing
`BaseModelAdapter`, register it in `src/common/adapters/__init__.py`, and add a
smoke test under `tests/e2e/`. Any algorithm that subclasses `FlatVarlenCache`
works as soon as the adapter implements RoPE, audio-token location, and QKV
computation correctly.

---

## 8. Data Formats

See `data_format.md` for the input/output JSONL schemas, the `stats` object, and
the WnW config schema.
