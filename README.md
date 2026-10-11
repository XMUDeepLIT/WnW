# WnW — Waxing-and-Waning KV-cache compression for audio LLMs

WnW is a model-agnostic framework for running and evaluating KV-cache
compression on audio large language models. It keeps the model object clean:
all algorithm state lives in a HuggingFace `Cache` subclass and is passed
explicitly to `model.generate(..., past_key_values=cache)`.

## Scope

This snapshot contains:

- **WnW (Python)** — class A/B head classification plus chunk-level CPU offload.
- **WnW (C++ planner, optional)** — the same planning logic implemented as a
  JIT-compiled C++ extension for lower decode-step planning overhead.
- **Baseline evaluation** — the same models with no compression, for reference.
- **Models** — Voxtral (`voxtral`) and Qwen2.5-Omni (`qwen2_5_omni`).
- **Analysis pipeline** — `analysis/data_prep`, `analysis/importance`, and
  `analysis/wnw_config`, which together produce a WnW config from head scores.
- **Metrics** — WER, CER, and BLEU scorers over eval output.

The snapshot intentionally ships **no** models, datasets, sample JSONL, outputs,
or pre-generated configs. You must supply your own model checkpoint and input
data, and generate a WnW config with the analysis pipeline (see
`docs/wnw_workflow.md`). Input `audio_path` values are opened directly, so they
must already resolve to real files.

## Install

Requires Python >= 3.12 and `transformers==5.7.0` (the HF `Cache` and RoPE
contracts are version-sensitive).

```bash
pip install -e .
# analysis pipeline (WhisperX alignment):
pip install -e ".[analysis]"
# development (tests, formatters, C++ JIT build tool):
pip install -e ".[dev]"
```

All commands below assume `PYTHONPATH=src` from the repository root.

## Quickstart

Baseline (no compression):

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_baseline.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <INPUT_JSONL> \
  --output <OUTPUT_JSONL>
```

WnW (Python) with a config you generated via the analysis pipeline:

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_wnw.py \
  --adapter voxtral \
  --model-path <MODEL_PATH> \
  --input <INPUT_JSONL> \
  --config <CONFIG_JSON> \
  --output <OUTPUT_JSONL>
```

Score the output (metric drivers read `--output`, not `--input`):

```bash
PYTHONPATH=src python src/benchmarks/scripts/eval_wer.py --output <OUTPUT_JSONL>
```

## Repository Layout

```
src/
  common/        # adapters, FlatVarlenCache, AlgorithmSession, monitoring
  algorithms/    # wnw (Python) and wnw_cpp (C++ planner)
  analysis/      # data_prep / importance / wnw_config
  benchmarks/    # model-agnostic eval drivers + distributed launcher
tests/
docs/
  README.md          # documentation index
  ARCHITECTURE.md    # design and correctness constraints
  data_format.md     # input/output JSONL and config schema
  usage.md           # install, commands, tests
  wnw_workflow.md    # full WnW pipeline
```

## Documentation

Start with `docs/README.md`, or jump to:

- `docs/ARCHITECTURE.md` — design and the constraints that keep it correct.
- `docs/data_format.md` — input/output JSONL and config schemas.
- `docs/usage.md` — install, eval drivers, metrics, tests.
- `docs/wnw_workflow.md` — end-to-end pipeline that produces a WnW config.

## Requirements

- Python >= 3.12
- `torch >= 2.11.0`
- `transformers == 5.7.0`
- See `pyproject.toml` for the full dependency list.
