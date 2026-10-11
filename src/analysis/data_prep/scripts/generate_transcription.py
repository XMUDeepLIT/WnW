
import argparse
import json
import os
import traceback

import torch
from tqdm import tqdm

from common.adapters import get_adapter


def _existing_ids(path):
    if not os.path.exists(path):
        return set()
    existing = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                existing.add(json.loads(line).get("id"))
    return existing


def main():
    parser = argparse.ArgumentParser(description="Generate model transcription")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Adapter name (voxtral, qwen2_5_omni)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Currently only batch_size=1 is supported")
    parser.add_argument("--prompt-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--task-key", type=str, default="task")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError("Only batch_size=1 is currently supported")

    adapter = get_adapter(args.adapter)
    model, processor = adapter.load(args.model_path, args.device, degqa=False)
    model.eval()

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")

    existing = _existing_ids(args.output)
    pending = [s for s in samples if s["id"] not in existing]
    print(f"Existing: {len(existing)}, pending: {len(pending)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    success = fail = 0
    with open(args.output, "a") as out_f:
        for sample in tqdm(pending, desc="Generating"):
            sample_id = sample["id"]
            audio_path = sample["audio_path"]
            question = sample.get(args.prompt_key, "")
            task = sample.get(args.task_key)
            try:
                inputs = adapter.build_inputs(
                    audio_path, 16000, question, processor, args.device,
                    task=task,
                )
                input_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    output_ids = adapter.generate(
                        model, inputs, max_new_tokens=args.max_new_tokens,
                    )

                hyp = processor.batch_decode(
                    output_ids[:, input_len:], skip_special_tokens=True,
                )[0].strip()

                sample[args.answer_key] = hyp
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1

                del output_ids
            except Exception as e:
                print(f"Failed {sample_id}: {e}")
                traceback.print_exc()
                fail += 1
            torch.cuda.empty_cache()

    print(f"Done. Success={success} Fail={fail}. Saved to: {args.output}")


if __name__ == "__main__":
    main()
