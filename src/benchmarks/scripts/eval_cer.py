
import argparse

from common.eval.metrics import align_pairs, compute_cer, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="CER evaluation")
    parser.add_argument("--output", required=True,
                        help="Eval output JSONL (with id/hyp/ref)")
    parser.add_argument("--print-errors", action="store_true", help="Print per-sample details")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Optional tokenizer name or path. When set, truncate the "
                             "prediction by token length; otherwise no truncation")
    parser.add_argument("--truncate-ratio", type=float, default=1.2,
                        help="Truncate the prediction to a multiple of the reference token "
                             "count (default 1.2). Only applies when --tokenizer is set")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    pairs = align_pairs(
        args.output,
        standardize=False,
        truncate_ratio=args.truncate_ratio,
        tokenizer=tokenizer,
    )
    if not pairs:
        print("No matching IDs found.")
        return

    r = compute_cer(pairs, print_errors=args.print_errors)
    print(f"\nSamples: {r['samples']}")
    print(f"CER = {r['rate']:.4f} "
          f"(S={r['substitutions']} D={r['deletions']} I={r['insertions']} N={r['n']})")


if __name__ == "__main__":
    main()
