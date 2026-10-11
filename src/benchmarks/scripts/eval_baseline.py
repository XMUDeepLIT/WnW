
import argparse
import json
import os
import time
import traceback
import warnings

import soundfile as sf
import torch
from tqdm import tqdm

from benchmarks.scripts._batch_utils import chunked, collate_inputs, sort_by_answer_length
from benchmarks.scripts._timing import StepTimer
from common.adapters import get_adapter


def load_audio(path: str):
    waveform, sr = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000
    return waveform, sr


def get_existing_ids(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                warnings.warn(f"{output_path}:{lineno}: skipping corrupt line")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation (adapter-based)")
    parser.add_argument("--adapter", type=str, required=True,
                        choices=["voxtral", "qwen2_5_omni"])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--degqa", action="store_true",
                        help="Convert GQA to MHA in-place (default: keep original GQA)")
    parser.add_argument("--stats-interval", type=int, default=0,
                        help="Unused for baseline (kept for CLI parity)")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.input}")

    existing_ids = get_existing_ids(args.output)
    pending = [s for s in samples if s["id"] not in existing_ids]
    pending = sort_by_answer_length(pending, answer_key=args.answer_key)
    print(f"Existing: {len(existing_ids)}, pending: {len(pending)} (sorted by answer length)")
    if not pending:
        print("All done.")
        return

    adapter = get_adapter(args.adapter)
    print(f"Loading model with adapter={adapter.name}...")
    model, processor = adapter.load(args.model_path, args.device, degqa=args.degqa)

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pbar = tqdm(total=len(pending), desc="Baseline eval")

    for batch_samples in chunked(pending, args.batch_size):
        try:
            input_list = []
            for sample in batch_samples:
                audio, sr = load_audio(sample["audio_path"])
                prompt = sample.get(args.prompt_key, "")
                inputs = adapter.build_inputs(audio, sr, prompt, processor,
                                              args.device, task=sample.get("task"))
                input_list.append(inputs)

            batched_inputs, input_lens = collate_inputs(
                input_list, pad_token_id, adapter.feature_time_dim)
            max_in = max(input_lens)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            timer = StepTimer(model.device)
            with torch.no_grad():
                output_ids = adapter.generate(model, batched_inputs,
                                              max_new_tokens=args.max_new_tokens,
                                              stopping_criteria=[timer])
            elapsed = time.time() - t0
            timing = timer.result()
            gpu_peak = torch.cuda.max_memory_allocated()

            n_out = int(output_ids.shape[1] - max_in)
            with open(args.output, "a") as f:
                for i, sample in enumerate(batch_samples):
                    hyp = processor.batch_decode(
                        output_ids[i:i+1, max_in:], skip_special_tokens=True)[0]
                    result = {
                        "id": sample["id"],
                        "hyp": hyp,
                        "ref": sample.get(args.answer_key),
                        "n_in": int(input_lens[i]),
                        "n_out": n_out,
                        "max_new_tokens": args.max_new_tokens,
                        "time": round(elapsed / len(batch_samples), 2),
                        "timing": timing,
                        "gpu_peak_bytes": gpu_peak,
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            traceback.print_exc()
            ids = [s["id"] for s in batch_samples]
            print(f"FAIL batch {ids}: {e}", flush=True)
            torch.cuda.empty_cache()

        pbar.update(len(batch_samples))

    pbar.close()
    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
