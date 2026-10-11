
import argparse
import json
import os

import torch
from tqdm import tqdm

from analysis.importance.code.attention_score import AttentionScoreAnalyzer
from analysis.importance.code.hit_ratio import HitRatioCalculator
from common.adapters import get_adapter


def _get_sample_id(filename):
    assert filename.endswith(".pt")
    return filename[:-3]


def _existing_ids(output_dir):
    if not os.path.exists(output_dir):
        return set()
    return {_get_sample_id(f) for f in os.listdir(output_dir) if f.endswith(".pt")}


def main():
    parser = argparse.ArgumentParser(description="Compute hit-ratio scores")
    parser.add_argument("--adapter", type=str, required=True,
                        choices=["voxtral", "qwen2_5_omni"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--preprocess-dir", type=str, required=True)
    parser.add_argument("--token-timestamps", type=str, required=True,
                        help="Token timestamps jsonl (one row per sample)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--audio-only-topk", action="store_true")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)

    calculator = HitRatioCalculator(
        top_k=args.top_k, audio_only_topk=args.audio_only_topk, token_rate=adapter.token_rate,
    )

    print("Loading model...")
    model, _processor = adapter.load(args.model_path, args.device_map, degqa=False)
    model.eval()
    analyzer = AttentionScoreAnalyzer(model, adapter)
    analyzer.patch()

    all_token_timestamps = {}
    with open(args.token_timestamps) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_token_timestamps[data.get("id")] = {
                    "tokens": data.get("tokens", []),
                    "duration": data.get("duration", 0.0),
                }

    all_files = sorted(f for f in os.listdir(args.preprocess_dir) if f.endswith(".pt"))
    existing = _existing_ids(args.output_dir)
    pending = [f for f in all_files if _get_sample_id(f) not in existing]
    print(f"Total: {len(all_files)}, existing: {len(existing)}, pending: {len(pending)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in tqdm(pending, desc="Computing hit-ratio"):
        sample_id = _get_sample_id(filename)
        if sample_id not in all_token_timestamps:
            print(f"Warning: No token timestamps for {sample_id}, skipping")
            continue

        data = torch.load(os.path.join(args.preprocess_dir, filename), weights_only=True)

        attn_result = analyzer.compute(
            inputs_embeds=data["inputs_embeds"],
            labels=data["labels"],
            input_ids=data["input_ids"],
        )

        result = calculator.compute(
            attention_scores=attn_result["scores"],
            num_audio=attn_result["num_audio"],
            token_timestamps=all_token_timestamps[sample_id]["tokens"],
        )

        torch.save(result, os.path.join(args.output_dir, f"{sample_id}.pt"))

        del attn_result, data
        torch.cuda.empty_cache()

    analyzer.unpatch()
    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
