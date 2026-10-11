
import argparse
import json
import os

from analysis.importance.code.whisperx_align import WhisperXAligner


def _existing_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    existing = set()
    with open(output_path) as f:
        for line in f:
            if line.strip():
                existing.add(json.loads(line).get("id"))
    return existing


def main():
    parser = argparse.ArgumentParser(description="Compute word-level timestamps using WhisperX")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")

    existing = _existing_ids(args.output)
    pending = [s for s in samples if s.get("id") not in existing]
    print(f"Existing: {len(existing)}, Pending: {len(pending)}")

    aligner = WhisperXAligner(device=args.device, compute_type=args.compute_type)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    success = fail = 0
    with open(args.output, "a") as out_f:
        for idx, sample in enumerate(pending):
            sample_id = sample.get("id", f"sample_{idx}")
            audio_path = sample.get("audio_path")
            text = sample.get("answer", "")
            if not audio_path or not text or not os.path.exists(audio_path):
                print(f"Skipping {sample_id}: bad audio_path/answer")
                fail += 1
                continue
            try:
                result = aligner.align(audio_path, text, language=args.language)
                out_f.write(json.dumps({
                    "id": sample_id,
                    "audio_path": audio_path,
                    "text": text,
                    "duration": result["duration"],
                    "word_segments": result["word_segments"],
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1
                if success % 10 == 0:
                    print(f"Processed {success + fail}/{len(pending)} (ok={success}, fail={fail})")
            except Exception as e:
                print(f"Failed {sample_id}: {e}")
                fail += 1

    aligner.cleanup()
    print(f"Done. Success={success} Fail={fail}. Saved to: {args.output}")


if __name__ == "__main__":
    main()
