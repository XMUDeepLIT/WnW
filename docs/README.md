# WnW Documentation

This directory documents the WnW release. Each file owns one area with no
overlap.

| Document | What it covers |
|---|---|
| `ARCHITECTURE.md` | Design of the adapter / cache / session abstractions, WnW internals, and the correctness constraints that keep them safe. |
| `data_format.md` | JSONL schemas for input samples and eval output, the runtime `stats` object, and the WnW config schema. |
| `usage.md` | Install, eval drivers (baseline, WnW Python/C++), metrics, the distributed launcher, and tests. |
| `wnw_workflow.md` | The full analysis pipeline that turns audio samples into a WnW config, and how to run the final evaluation. |

Start with `ARCHITECTURE.md` to understand the design, or `usage.md` to run
something immediately.
