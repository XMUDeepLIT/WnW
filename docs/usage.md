# WnW Usage Guide

Concrete commands for installing, running eval drivers, scoring output, running
the distributed launcher, and running tests.

For the documentation index see `README.md`; for design see `ARCHITECTURE.md`;
for schemas see `data_format.md`; for the config-generation pipeline see
`wnw_workflow.md`.

The repository ships no models, data, or configs. Supply your own model
checkpoint and input JSONL, and generate a WnW config with the pipeline in
`wnw_workflow.md`. All examples assume `PYTHONPATH=src` from the repository root.

---

## Install

Requires Python >= 3.12 and `transformers==5.7.0`.

```bash
pip install -e .                 # core
pip install -e ".[analysis]"     # + WhisperX for the analysis pipeline
pip install -e ".[dev]"          # + pytest, formatters, ninja (C++ JIT build)
```

WhisperX is only needed for the analysis pipeline's word-level alignment step;
core evaluation does not require it.

---

## Common CLI (eval drivers)

| Flag | Meaning |
|---|---|
| `--model-path` | Path to the HuggingFace model checkpoint. |
| `--adapter` | Model family: `voxtral` or `qwen2_5_omni`. |
| `--input` | Input sample JSONL. |
| `--output` | Output JSONL path (appended to; existing ids are skipped). |
| `--config` | WnW config JSON (WnW drivers only). |
| `--device` | Torch device, e.g. `cuda:0`. |
| `--max-new-tokens` | Generation budget (default 200). |
| `--sample-rate` | Audio sample rate (default 16000; WnW drivers). |
| `--batch-size` | Batch size (default 1). |
| `--stats-interval` | Record stats every N decode steps (`0` disables; WnW drivers). |
| `--degqa` | Convert GQA to MHA in-place (optional). |
| `--prompt-key` / `--answer-key` | Input fields to read (default `question` / `answer`). |

`audio_path` values in the input JSONL are opened directly; there is no
audio-root prefix option.

---

## Running an Evaluation

### Baseline (no compression)

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_baseline.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <INPUT_JSONL> \
  --output <OUTPUT_JSONL> \
  --device cuda:0 \
  --max-new-tokens 200
```

### WnW (Python)

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_wnw.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <INPUT_JSONL> \
  --config <CONFIG_JSON> \
  --output <OUTPUT_JSONL> \
  --device cuda:0 \
  --max-new-tokens 200 \
  --stats-interval 5
```

### WnW (C++ planner)

Same CLI as `eval_wnw.py`. The planner extension is JIT-compiled on first use
(requires `ninja` and a C++ compiler). Set `WNW_CPP_FALLBACK=1` to force the
pure-Python planner instead.

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_wnw_cpp.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <INPUT_JSONL> \
  --config <CONFIG_JSON> \
  --output <OUTPUT_JSONL>
```

---

## Scoring Output

The metric drivers read the eval **output** JSONL (which already contains `id`,
`hyp`, and `ref`); they take `--output`, not `--input`.

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_wer.py --output <OUTPUT_JSONL>
PYTHONPATH=src python src/benchmarks/scripts/eval_cer.py --output <OUTPUT_JSONL>
PYTHONPATH=src python src/benchmarks/scripts/eval_bleu.py --output <OUTPUT_JSONL>
```

Shared metric flags: `--print-errors`, `--tokenizer`, `--truncate-ratio` (default
1.2). WER additionally accepts `--no-standardize` (by default it lowercases and
strips punctuation before scoring).

---

## Distributed Launcher

`launch_wnw.py` shards the input across GPUs, runs one `eval_wnw.py` process per
GPU (via `CUDA_VISIBLE_DEVICES`), and merges the shards into `merged.jsonl`
sorted by id.

```bash
PYTHONPATH=src python src/benchmarks/scripts/distributed/launch_wnw.py \
  --gpus 0,1,2,3 \
  --model-path <MODEL_PATH> \
  --input <INPUT_JSONL> \
  --config <CONFIG_JSON> \
  --output-dir <OUTPUT_DIR>
```

`--gpus` accepts either a comma-separated list of ids or a count. Note its
`--max-new-tokens` default is 2048.

---

## Analysis Pipeline

Generating a WnW config starts from the dataset ground truth in each sample's
`answer` field: preprocessing → timestamps → hit-ratio → KV importance → head
classification → config. Generating model responses is an optional alternative
for the analysis targets. The complete process is documented in
`wnw_workflow.md`.

---

## Running Tests

End-to-end tests load a model and read a sample JSONL. Provide `--model-path`
and `--samples`; use `--data-root` when sample `audio_path` values are relative.

```bash
PYTHONPATH=src python -m pytest tests/e2e/test_voxtral_wnw.py -v \
  --model-path <MODEL_PATH> \
  --samples <INPUT_JSONL> \
  --data-root <DATA_ROOT> \
  --device cuda:0
```

The Qwen2.5-Omni test (`tests/e2e/test_qwen2_5_omni_degqa.py`) takes the same
options with a Qwen2.5-Omni checkpoint.
