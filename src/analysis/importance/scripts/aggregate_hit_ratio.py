
import argparse
import json
import os
import warnings

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-sample hit ratio into summary")
    parser.add_argument("--hit-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--token-rate", type=float, default=12.5)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.hit_dir) if f.endswith(".pt"))
    assert len(files) > 0, f"No .pt files found in {args.hit_dir}"

    all_ratios = []
    sample_ids = []
    nan_samples = []
    for f in files:
        sid = f.replace(".pt", "")
        data = torch.load(os.path.join(args.hit_dir, f), weights_only=True)
        ratio = data["hit_ratio"]
        if torch.isnan(ratio).any():
            nan_count = int(torch.isnan(ratio).sum().item())
            nan_samples.append((sid, nan_count))
            warnings.warn(f"Sample {sid} has {nan_count} NaN values in hit_ratio, skipping")
            continue
        sample_ids.append(sid)
        all_ratios.append(ratio.numpy())

    if nan_samples:
        print(f"WARNING: {len(nan_samples)} samples skipped due to NaN:")
        for sid, cnt in nan_samples:
            print(f"  {sid}: {cnt} NaN values")

    assert len(all_ratios) > 0, "All samples have NaN values, nothing to aggregate"

    stacked = np.stack(all_ratios, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    num_layers, num_heads = mean.shape
    flat_heads = [
        {"layer": l, "head": h, "hit_ratio": float(mean[l][h]), "std": float(std[l][h])}
        for l in range(num_layers) for h in range(num_heads)
    ]
    flat_heads.sort(key=lambda x: x["hit_ratio"], reverse=True)

    summary = {
        "num_samples": len(all_ratios),
        "num_total": len(files),
        "num_nan_skipped": len(nan_samples),
        "nan_sample_ids": [sid for sid, _ in nan_samples],
        "sample_ids": sample_ids,
        "top_k": args.top_k,
        "token_rate": args.token_rate,
        "statistics": {
            "mean": float(mean.mean()),
            "median": float(np.median(mean)),
            "max": float(mean.max()),
            "min": float(mean.min()),
            "std": float(std.mean()),
        },
        "hit_ratio_mean": mean.tolist(),
        "hit_ratio_std": std.tolist(),
        "top_50_heads": flat_heads[:50],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Mean hit_ratio: {summary['statistics']['mean']:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
