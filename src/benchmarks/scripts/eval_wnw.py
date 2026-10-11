
import argparse
import json
import os
import time
import traceback
import warnings

import torch

from algorithms.wnw.code.cache import WnWCache
from algorithms.wnw.code.config import WnWConfig
from algorithms.wnw.code.session import WnWSession
from benchmarks.scripts._batch_utils import chunked, collate_inputs, sort_by_answer_length
from benchmarks.scripts._timing import StepTimer
from common.adapters import get_adapter


def _existing_ids(path):
    if not os.path.exists(path):
        return set()
    existing = set()
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                existing.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                warnings.warn(f"{path}:{lineno}: skipping corrupt line")
    return existing


def main():
    parser = argparse.ArgumentParser(description="WnW evaluation driver")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter", type=str, default="voxtral")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stats-interval", type=int, default=5,
                        help="0 disables InferenceStats collection")
    parser.add_argument("--degqa", action="store_true",
                        help="Convert GQA to MHA in-place (default: keep original GQA)")
    parser.add_argument("--prompt-key", type=str, default="question",
                        help="JSONL field containing the text prompt")
    parser.add_argument("--answer-key", type=str, default="answer",
                        help="JSONL field containing the reference answer")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.input) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    existing = _existing_ids(args.output)
    pending = [s for s in samples if s["id"] not in existing]
    pending = sort_by_answer_length(pending, answer_key=args.answer_key)
    print(f"Total {len(samples)}, existing {len(existing)}, pending {len(pending)} (sorted by answer length)")
    if not pending:
        return

    adapter = get_adapter(args.adapter)
    print("Loading model...")
    model, processor = adapter.load(args.model_path, args.device, degqa=args.degqa)
    text_config = adapter.get_text_config(model)
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    num_q_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // num_q_heads)

    config = WnWConfig.from_json(
        args.config, num_layers=num_layers,
        num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
    )
    print(f"WnW: class_a={len(config.class_a)} class_b={len(config.class_b)} "
          f"chunk={config.chunk_size_s}s top_k={config.top_k_chunks}")

    cache = WnWCache(
        config=config, num_layers=num_layers, num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads, head_dim=head_dim, device=model.device, adapter=adapter,
    )
    cache.stats.set_interval(args.stats_interval)

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with WnWSession(model, adapter, cache, processor) as sess, open(args.output, "a") as out_f:
        for batch_samples in chunked(pending, args.batch_size):
            try:
                input_list = []
                for sample in batch_samples:
                    inputs = adapter.build_inputs(
                        sample["audio_path"], args.sample_rate,
                        sample.get(args.prompt_key, ""), processor, args.device,
                        task=sample.get("task"),
                    )
                    input_list.append(inputs)

                batched_inputs, input_lens = collate_inputs(input_list, pad_token_id, adapter.feature_time_dim)

                sess.setup_audio(batched_inputs["input_ids"],
                                 batched_inputs.get("attention_mask"))

                t0 = time.time()
                timer = StepTimer(model.device)
                with torch.no_grad():
                    output_ids = sess.generate(batched_inputs,
                                              max_new_tokens=args.max_new_tokens,
                                              stopping_criteria=[timer])
                elapsed = time.time() - t0
                timing = timer.result()

                stats_list = sess.get_stats()
                max_in = max(input_lens)

                for i, sample in enumerate(batch_samples):
                    n_in = input_lens[i]
                    hyp = processor.batch_decode(
                        output_ids[i:i+1, max_in:], skip_special_tokens=True
                    )[0]
                    stats = stats_list[i] if i < len(stats_list) else {}
                    out_f.write(json.dumps({
                        "id": sample["id"],
                        "hyp": hyp,
                        "ref": sample.get(args.answer_key),
                        "n_in": int(n_in),
                        "n_out": int(output_ids.shape[1] - max_in),
                        "max_new_tokens": args.max_new_tokens,
                        "time": round(elapsed / len(batch_samples), 2),
                        "timing": timing,
                        "stats": stats,
                    }, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                traceback.print_exc()
                ids = [s["id"] for s in batch_samples]
                print(f"FAIL batch {ids}: {e}", flush=True)
                torch.cuda.empty_cache()

            cache.reset()

    print(f"Done. {args.output}")


if __name__ == "__main__":
    main()
