# WnW Data and Config Formats

This document defines the JSONL and config formats used by the evaluation and
analysis pipelines. The repository ships no data, models, configs, or outputs;
you provide the inputs and generate the config yourself.

For the documentation index, see `README.md`.

---

## Input Sample JSONL

Each line is one JSON object describing an audio+text sample.

### Required fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique sample identifier. |
| `audio_path` | string | Path to the audio file. It is opened directly, so it must resolve to a real file (there is no audio-root prefix option). |
| `question` | string | Prompt / instruction, e.g. `"Transcribe the audio."`. |
| `answer` | string | Reference text (ground truth or target response). |

### Optional field

| Field | Type | Description |
|---|---|---|
| `task` | object \| null | Task metadata. If present, must contain a `name`. `null` means no task-specific routing. |

**`task` object:**

| Task name | Fields | Example |
|---|---|---|
| `ASR` | `name`, `lang` (English name) | `{"name": "ASR", "lang": "English"}` |
| `translation` | `name`, `lang1` (source), `lang2` (target) | `{"name": "translation", "lang1": "English", "lang2": "French"}` |
| `chat` | `name` | `{"name": "chat"}` |

### Example

```json
{
  "id": "sample_001",
  "audio_path": "/abs/path/to/001.flac",
  "task": {"name": "ASR", "lang": "English"},
  "question": "Transcribe the audio into text.",
  "answer": "the reference transcription ..."
}
```

---

## Output JSONL

All eval drivers append one JSON object per line.

### Common fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Same as input `id`. |
| `hyp` | string | Model-generated hypothesis text. |
| `ref` | string \| null | Reference answer from input (`answer`). |
| `n_in` | int | Number of input tokens (including audio tokens). |
| `n_out` | int | Number of generated tokens. |
| `max_new_tokens` | int | Generation cap passed to `generate()`. |
| `time` | float | Wall-clock generation time per sample, in seconds. |
| `timing` | object | `{"prefill": float, "decode": [float, ...]}`, batch-level per-step timings in seconds. |

### Driver-specific fields

| Field | Driver | Description |
|---|---|---|
| `stats` | WnW (`eval_wnw`, `eval_wnw_cpp`) | Runtime statistics from `InferenceStats` (see below). |
| `gpu_peak_bytes` | baseline (`eval_baseline`) | Peak allocated GPU memory during the batch. |

### Example (WnW)

```json
{
  "id": "sample_001",
  "hyp": "the model output ...",
  "ref": "the reference transcription ...",
  "n_in": 12345, "n_out": 200, "max_new_tokens": 200, "time": 12.34,
  "timing": {"prefill": 0.42, "decode": [0.01, 0.01]},
  "stats": {
    "prefill_before": {"memory_used_bytes": 1234567890},
    "prefill": {"gpu_total_tokens": 12345, "gpu_audio_tokens": 12000},
    "decode_snapshots": [],
    "decode_final": {"memory_used_bytes": 1234567890},
    "cpu_gpu_events": [[1, 0, 0, 0, 0]]
  }
}
```

### Example (baseline)

```json
{
  "id": "sample_001",
  "hyp": "the model output ...",
  "ref": "the reference transcription ...",
  "n_in": 12345, "n_out": 200, "max_new_tokens": 200, "time": 12.34,
  "timing": {"prefill": 0.42, "decode": [0.01, 0.01]},
  "gpu_peak_bytes": 4567890123
}
```

---

## Runtime `stats` Schema

`stats` is produced by `InferenceStats` and attached per sample by the WnW
session.

| Field | Type | Description |
|---|---|---|
| `prefill_before` | object | Cache + GPU snapshot before prefill. |
| `prefill` | object | Cache + GPU snapshot after prefill. |
| `decode_snapshots` | list[object] | Snapshots taken every `--stats-interval` decode steps (empty if `--stats-interval 0`). |
| `decode_final` | object | Last snapshot (or the prefill snapshot if none were taken). |
| `cpu_gpu_events` | list[tuple] | Per-step transfer events `(step, c2g_tokens, g2c_tokens, c2g_bytes, g2c_bytes)`. |

Cache snapshots typically include `gpu_index`, `memory_total_bytes`,
`memory_free_bytes`, `memory_used_bytes`, `gpu_total_tokens`,
`gpu_audio_tokens`, and `audio_retention` (`gpu_audio_tokens / audio_len`).

---

## WnW Config JSON

`eval_wnw.py` / `eval_wnw_cpp.py` require `--config`, a JSON file matching
`WnWConfig` (`src/algorithms/wnw/code/config.py`). This snapshot ships no config;
generate one with the pipeline in `wnw_workflow.md`.

| Section | Field | Meaning |
|---|---|---|
| `chunk` | `size_s`, `overlap_s`, `token_rate` | Chunk length, overlap, and audio tokens/sec. Derived: `chunk_size_tokens`, `overlap_tokens`, `stride_tokens = chunk - overlap`, which must satisfy `stride > 0` and `chunk_size % stride == 0`. |
| `heads` | `class_a` | Global query-head ids kept fully on GPU: `layer_idx * num_q_heads + local_q`. |
| `heads` | `class_b` | Global kv-head ids offloaded to CPU: `layer_idx * num_kv_heads + local_kv`. |
| `heads` | `retention_ratios` | Per-kv-head GPU retention, length `num_layers * num_kv_heads`. |
| `selection` | `top_k_chunks`, `eviction_lag` | Chunks kept on GPU per selection, and the delay before evicting a dropped chunk. |
| `score` | `prefill_mode` | `"mean_text"` or `"last"`. |
| `score` | `prefill_global_topk` | Optional; global vs per-segment top-k at prefill. |

### Example

```json
{
  "chunk": {"size_s": 4.0, "overlap_s": 2.0, "token_rate": 12.5},
  "heads": {
    "class_a": [0, 1, 2, 3, 4],
    "class_b": [5, 6, 7],
    "retention_ratios": [0.15, 0.15, 0.15, 0.15, 0.15, 0.8, 0.8, 0.8]
  },
  "selection": {"top_k_chunks": 3, "eviction_lag": 3},
  "score": {"prefill_mode": "mean_text"}
}
```

---

## Adapter Notes

### Voxtral

- Audio is 16 kHz mono.
- For `task.name == "ASR"`, the adapter maps `task.lang` (English name) to an ISO
  639-1 code via `pycountry` and uses `apply_transcription_request`. Otherwise it
  falls back to `apply_chat_template` with `question` as text.
- `--degqa` converts GQA to MHA in-place; default off.

### Qwen2.5-Omni

- Audio is 16 kHz mono.
- All tasks use `apply_chat_template` with `question` as text.
- The talker is disabled and `generation_mode="text"` is set automatically during
  `generate()`.

---

## Directory Conventions (suggested)

These are conventions only; the repository ships none of these directories. Keep
generated artifacts (large `.jsonl`, `.pt`, model outputs) out of version control.

```
outputs/
  <run_id>/
    results/   # eval output jsonl
    stats/     # aggregated stats
```
