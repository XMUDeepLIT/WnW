# WnW End-to-End Workflow

This document describes the complete pipeline that produces a WnW config from
your own audio samples, and how to run the final evaluation. For WnW's design see
`ARCHITECTURE.md`; for schemas see `data_format.md`.

The analysis pipeline needs the `analysis` extra (`pip install -e ".[analysis]"`)
for the WhisperX alignment step. All commands run from the repository root with
`PYTHONPATH=src`. `audio_path` values are opened directly, so they must resolve
to real files. Replace `<MODEL_PATH>` and `<DATASET_JSONL>` with your own paths;
the `outputs/...` paths below are produced by the steps themselves.

The pipeline uses the dataset ground truth in the `answer` field by default.
Set `<ANALYSIS_JSONL>` to `<DATASET_JSONL>` throughout the analysis steps. You
may instead generate model responses as optional analysis targets in step 1.
Evaluation always uses `<DATASET_JSONL>` so `ref` remains the dataset ground
truth.

---

## Pipeline Overview

```
<ANALYSIS_JSONL> (dataset ground truth by default)
    ├─ preprocess_inputs.py               → outputs/preprocess/<model>/{id}.pt
    │
    └─ compute_word_timestamps.py         → outputs/word_timestamps/<model>.jsonl
             │  compute_token_timestamps.py → outputs/token_timestamps/<model>.jsonl
             ▼
compute_hit_ratio.py                      → outputs/hit_ratio/<model>/{id}.pt
    │  aggregate_hit_ratio.py              → outputs/hit_ratio/<model>/summary.json
    │
    ├─ compute_kv_importance.py           → outputs/kv_importance/<model>/{id}.pt
    ▼        ▼
compute_head_classification.py            → outputs/head_scores/<model>/head_scores.pt
    │  generate_wnw_config.py              → outputs/wnw_configs/<model>/config.json
    │        ├─ [optional] calibrate_gpu_only.py / calibrate_gpu_cpu.py
    │        └─ [optional] expand_wnw_config.py  (degqa)
    ▼
eval_wnw.py                               → outputs/wnw_eval.jsonl
```

Token rates: Voxtral `12.5`, Qwen2.5-Omni `25.0`. Use the model's rate
consistently throughout.

---

## 1. Select Analysis Targets

Use the dataset JSONL directly for the standard pipeline:

```bash
ANALYSIS_JSONL=/path/to/dataset.jsonl
```

Each sample must contain its ground-truth text in `answer`. Both preprocessing
and timestamp alignment must consume the same `<ANALYSIS_JSONL>`.

Optionally, generate model responses and use them instead of ground truth for a
separate analysis of model-produced targets:

```bash
PYTHONPATH=src python src/analysis/data_prep/scripts/generate_transcription.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <DATASET_JSONL> \
  --output outputs/model_responses.jsonl \
  --device cuda:0 \
  --max-new-tokens 200

ANALYSIS_JSONL=outputs/model_responses.jsonl
```

This optional command replaces each sample's `answer` field in its output. It
does not modify the original dataset JSONL.

---

## 2. Preprocess Model Inputs

Builds `{id}.pt` files containing `input_ids`, `inputs_embeds`, and labels from
the `answer` field.

```bash
PYTHONPATH=src python src/analysis/data_prep/scripts/preprocess_inputs.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input "$ANALYSIS_JSONL" \
  --output-dir outputs/preprocess/<model> \
  --device-map cuda:0
```

`--degqa` defaults to off. Analysis does not need query-head granularity, and
expanding GQA to MHA can exhaust GPU memory during the audio-encoder forward
pass. Very long audio may still OOM even without `--degqa`; use shorter clips or a
larger GPU.

---

## 3. Word-Level Timestamps (WhisperX)

Aligns each audio file with its text to produce word spans.

```bash
PYTHONPATH=src python src/analysis/importance/scripts/compute_word_timestamps.py \
  --input "$ANALYSIS_JSONL" \
  --output outputs/word_timestamps/<model>.jsonl \
  --device cuda \
  --language en
```

Output columns: `id`, `audio_path`, `text`, `duration`, `word_segments`.

---

## 4. Token-Level Timestamps

Maps word spans onto tokenizer token indices.

```bash
PYTHONPATH=src python src/analysis/importance/scripts/compute_token_timestamps.py \
  --input outputs/word_timestamps/<model>.jsonl \
  --output outputs/token_timestamps/<model>.jsonl \
  --model-path <MODEL_PATH> \
  --min-word-score 0.85
```

Output columns: `id`, `audio_path`, `text`, `duration`, `tokens`.

---

## 5. Hit-Ratio Analysis

For each sample, runs a patched forward pass and measures how often a head's
top-K attended positions fall inside a word's audio span.

```bash
PYTHONPATH=src python src/analysis/importance/scripts/compute_hit_ratio.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --preprocess-dir outputs/preprocess/<model> \
  --token-timestamps outputs/token_timestamps/<model>.jsonl \
  --output-dir outputs/hit_ratio/<model> \
  --top-k 24 \
  --device-map cuda:0
```

---

## 6. Aggregate Hit-Ratio

Averages the per-sample tensors into `summary.json`. Use the same `--top-k` as
step 5 (both default to 24) and the model's token rate.

```bash
PYTHONPATH=src python src/analysis/importance/scripts/aggregate_hit_ratio.py \
  --hit-dir outputs/hit_ratio/<model> \
  --output outputs/hit_ratio/<model>/summary.json \
  --top-k 24 \
  --token-rate 12.5
```

---

## 7. KV Importance / Audio Sensitivity

Computes per-layer, per-head audio sensitivity from KV-cache gradients.
`--tp-plan` enables tensor-parallel execution under `torchrun`.

```bash
PYTHONPATH=src python src/analysis/importance/scripts/compute_kv_importance.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --preprocess-dir outputs/preprocess/<model> \
  --output-dir outputs/kv_importance/<model> \
  --device-map cuda:0
```

---

## 8. Head Classification

Combines hit-ratio and audio sensitivity into `head_scores.pt` (containing
`voice_score_kv`, `audio_sensitivity_k`, `audio_sensitivity_v`, `hit_ratio`).

```bash
PYTHONPATH=src python src/analysis/wnw_config/scripts/compute_head_classification.py \
  --hit-ratio-path outputs/hit_ratio/<model>/summary.json \
  --kv-imp-dir outputs/kv_importance/<model> \
  --output outputs/head_scores/<model>/head_scores.pt \
  --model-path <MODEL_PATH> \
  --max-samples 100
```

---

## 9. Generate WnW Config

Turns `head_scores.pt` into a config JSON that `eval_wnw.py` can load.

```bash
PYTHONPATH=src python src/analysis/wnw_config/scripts/generate_wnw_config.py \
  --head-scores-path outputs/head_scores/<model>/head_scores.pt \
  --output outputs/wnw_configs/<model>/config.json \
  --token-rate 12.5 \
  --num-class-a 5 \
  --retention-scale 0.15 \
  --sensitivity-mode mean \
  --top-voice 0.10 \
  --chunk-size-s 4.0 \
  --overlap-s 2.0
```

Instead of `--retention-scale`, you can target a retention level (the script
bisects the scale internally using an analytic formula):

```bash
PYTHONPATH=src python src/analysis/wnw_config/scripts/generate_wnw_config.py \
  --head-scores-path outputs/head_scores/<model>/head_scores.pt \
  --output outputs/wnw_configs/<model>/target.json \
  --token-rate 12.5 \
  --target-gpucpu-retention 0.15 \
  --duration-file outputs/durations.txt
```

Key flags:

| Flag | Meaning |
|---|---|
| `--num-class-a` | Number of Class A heads (full GPU retention). Default 5. |
| `--retention-scale` | Global multiplier for non-Class-A retention ratios. Default 0.15. |
| `--target-gpu-retention` / `--target-gpucpu-retention` | Target GPU or GPU+CPU retention; mutually exclusive with `--retention-scale`. |
| `--sensitivity-mode` | `k`, `v`, or `mean`. |
| `--top-voice` | Fraction (or count) of top voice heads to consider. Default 0.10. |
| `--chunk-size-s` / `--overlap-s` | Chunk length and overlap in seconds. |
| `--top-k-chunks` | Chunks kept on GPU per selection. Default 3. |
| `--duration-file` | Optional file of per-sample audio durations (seconds, one per line). |
| `--loadback-factor` | Damping on the Class B CPU→GPU loadback term. Default 0.8. |

The script prints predicted GPU and GPU+CPU retention. `--duration-file` only
needs to approximate the audio-length distribution of your evaluation data; a
random sample of 50–100 entries is enough, since the analytic formula has a 2%
tolerance and mainly cares about the short-vs-long shape.

---

## 10. Calibrate Retention (optional)

The generator's analytic formula already accounts for the per-segment
`max(1, ...)` floor and the Class B CPU→GPU loadback term, so its predictions
usually match real runs. Use calibration only to hit an exact target on real
samples. Both scripts generate a config, run `eval_wnw.py` on a small subset,
and scan the scale until the measured retention is within tolerance, then write
the final config to the output directory.

```bash
# GPU-only target
PYTHONPATH=src python src/analysis/wnw_config/scripts/calibrate_gpu_only.py \
  --head-scores-path outputs/head_scores/<model>/head_scores.pt \
  --target-gpu-retention 0.2 \
  --eval-input <DATASET_JSONL> \
  --output-dir outputs/calib/gpu_only \
  --model-path <MODEL_PATH> \
  --adapter voxtral \
  --device cuda:0 \
  --token-rate 12.5

# GPU+CPU target
PYTHONPATH=src python src/analysis/wnw_config/scripts/calibrate_gpu_cpu.py \
  --head-scores-path outputs/head_scores/<model>/head_scores.pt \
  --target-retention 0.4 \
  --eval-input <DATASET_JSONL> \
  --output-dir outputs/calib/gpu_cpu \
  --model-path <MODEL_PATH> \
  --adapter voxtral \
  --device cuda:0 \
  --token-rate 12.5
```

---

## 11. Expand Config for Degqa (optional)

If you evaluate with `--degqa` (GQA converted to MHA), expand the config so each
query head inherits its original KV head's retention ratio and Class B
membership. Pass the model's own dimensions.

```bash
PYTHONPATH=src python src/analysis/wnw_config/scripts/expand_wnw_config.py \
  --input outputs/wnw_configs/<model>/config.json \
  --output outputs/wnw_configs/<model>/config_degqa.json \
  --num_layers <NUM_LAYERS> \
  --orig_kv_heads <ORIG_KV_HEADS> \
  --num_q_heads <NUM_Q_HEADS>
```

---

## 12. Run WnW Evaluation

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_wnw.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <DATASET_JSONL> \
  --config outputs/wnw_configs/<model>/config.json \
  --output outputs/wnw_eval.jsonl \
  --device cuda:0 \
  --max-new-tokens 200 \
  --stats-interval 5
```

Add `--degqa` (with the expanded config) if the model is loaded with GQA→MHA
conversion. Evaluation reads the ground-truth reference from `answer` in
`<DATASET_JSONL>`.

---

## Common Pitfalls

- **Mismatched analysis targets.** Preprocessing and word alignment must use the
  same analysis JSONL. The default is the dataset JSONL with ground truth in
  `answer`; model-generated targets are optional.
- **Overwriting ground truth.** Write optional model responses to a separate
  JSONL and keep `<DATASET_JSONL>` unchanged for final evaluation.
- **Mismatched token rate.** Voxtral is 12.5 tok/s, Qwen2.5-Omni is 25.0. The
  wrong rate makes chunk sizes and retention predictions wrong.
- **OOM during preprocessing.** Keep `--degqa` off unless you need query-head
  granularity and have the VRAM; use shorter clips otherwise.
- **GQA vs MHA config.** A GQA config used with `--degqa` will not match the
  expanded KV head count; run `expand_wnw_config.py` first.
- **Verify predictions.** Compare the generator's predicted retention with the
  `audio_retention` values in `eval_wnw.py` output; large divergence usually
  means the token rate, duration distribution, or `--top-voice` is off.
- **Calibration is per-model.** Do not reuse a retention scale across models.
