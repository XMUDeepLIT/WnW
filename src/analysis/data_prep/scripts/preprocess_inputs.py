
import argparse
import json
import os

import torch
from tqdm import tqdm

from analysis.importance.code.input_builder import AnalysisInputBuilder
from common.adapters import get_adapter


def _load_samples(input_path):
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _existing_ids(output_dir):
    if not os.path.exists(output_dir):
        return set()
    return {f[:-3] for f in os.listdir(output_dir) if f.endswith(".pt")}


def main():
    parser = argparse.ArgumentParser(description="Preprocess samples for analysis")
    parser.add_argument("--adapter", type=str, required=True,
                        choices=["voxtral", "qwen2_5_omni"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True,
                        help="Input jsonl with id/audio_path/question/answer/task")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--degqa", action="store_true",
                        help="Convert GQA to MHA in-place (default: False for analysis)")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)
    model, processor = adapter.load(args.model_path, args.device_map, degqa=args.degqa)

    builder = AnalysisInputBuilder(model, processor, adapter)

    samples = _load_samples(args.input)
    existing_ids = _existing_ids(args.output_dir)
    pending = [s for s in samples if s["id"] not in existing_ids]
    print(f"Total: {len(samples)}, existing: {len(existing_ids)}, pending: {len(pending)}")

    os.makedirs(args.output_dir, exist_ok=True)
    for sample in tqdm(pending, desc="Preprocessing"):
        audio_path = sample["audio_path"]
        assert os.path.exists(audio_path), f"Audio not found: {audio_path}"

        result = builder.build(
            audio_path=audio_path,
            question=sample["question"],
            answer=sample["answer"],
            task=sample.get("task"),
        )

        torch.save(result, os.path.join(args.output_dir, f"{sample['id']}.pt"))

    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
