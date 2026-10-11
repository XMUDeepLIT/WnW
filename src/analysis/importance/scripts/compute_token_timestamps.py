
import argparse
import json
import os
import traceback

from transformers import AutoProcessor

from analysis.importance.code.word_token_aligner import WordTokenAligner


def main():
    parser = argparse.ArgumentParser(description="Compute token-level timestamps from word-level timestamps")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--min-word-score", type=float, default=0.85)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    aligner = WordTokenAligner(processor.tokenizer)

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    success = fail = 0
    with open(args.output, "w") as out_f:
        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{idx}")
            text = sample.get("text", "")
            word_segments = sample.get("word_segments", [])
            duration = sample.get("duration", 0.0)
            if not text or not word_segments:
                print(f"Skipping {sample_id}: missing text or word_segments")
                fail += 1
                continue
            try:
                token_mapping = aligner.align_words_to_tokens(
                    text, word_segments, min_word_score=args.min_word_score,
                )
                out_f.write(json.dumps({
                    "id": sample_id,
                    "audio_path": sample.get("audio_path", ""),
                    "text": text,
                    "duration": duration,
                    "tokens": token_mapping,
                }, ensure_ascii=False) + "\n")
                success += 1
                if success % 10 == 0:
                    print(f"Processed {success + fail}/{len(samples)} (ok={success}, fail={fail})")
            except Exception as e:
                print(f"Failed {sample_id}: {e}")
                traceback.print_exc()
                fail += 1

    print(f"Done. Success={success} Fail={fail}. Saved to: {args.output}")


if __name__ == "__main__":
    main()
