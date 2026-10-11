
import argparse
import json
import math
import os

import torch

from analysis.wnw_config.code.retention_formula import (
    compute_chunk_params,
    predict_retention,
)


def select_sensitivity(data: dict, mode: str) -> torch.Tensor:
    sens_k = data["audio_sensitivity_k"]
    sens_v = data["audio_sensitivity_v"]
    if mode == "k":
        return sens_k
    if mode == "v":
        return sens_v
    if mode == "mean":
        return (sens_k + sens_v) / 2
    raise ValueError(f"Unknown sensitivity mode: {mode}")


def compute_classification_stats(
    head_scores: dict,
    sensitivity_mode: str,
    num_class_a: int,
    top_voice: float,
) -> dict:
    """Return classification stats shared by config building and scale search."""
    hit_ratio = head_scores["hit_ratio"]
    voice_score_kv = head_scores["voice_score_kv"]
    audio_sens_kv = select_sensitivity(head_scores, sensitivity_mode)
    num_layers, num_kv_heads = voice_score_kv.shape
    num_kv_groups = hit_ratio.shape[-1] // num_kv_heads
    total_kv = num_layers * num_kv_heads

    weighted_score = audio_sens_kv.flatten() * voice_score_kv.flatten()

    if top_voice < 1.0:
        num_voice_kv = int(total_kv * top_voice)
    else:
        num_voice_kv = int(top_voice)
    num_voice_kv = max(1, min(num_voice_kv, total_kv))

    _, top_voice_indices = weighted_score.topk(num_voice_kv)
    voice_mask_kv = torch.zeros(total_kv, dtype=torch.bool)
    voice_mask_kv[top_voice_indices] = True

    masked = weighted_score.clone()
    masked[~voice_mask_kv] = float("-inf")
    _, top_class_a = masked.topk(min(num_class_a, int(voice_mask_kv.sum().item())))
    class_a_kv_set = set(top_class_a.tolist())

    class_b_kv_ids = [i for i in range(total_kv) if voice_mask_kv[i] and i not in class_a_kv_set]
    other_kv_ids = [i for i in range(total_kv) if not voice_mask_kv[i]]

    n_A = len(class_a_kv_set)
    n_B = len(class_b_kv_ids)
    n_O = len(other_kv_ids)

    non_a_kv_ids = class_b_kv_ids + other_kv_ids
    non_a_scores = torch.tensor([weighted_score[i].item() for i in non_a_kv_ids])
    max_non_a_score = non_a_scores.max().item() if len(non_a_scores) > 0 else 1
    norm_scores = (non_a_scores / max_non_a_score).tolist()

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "num_kv_groups": num_kv_groups,
        "weighted_score": weighted_score,
        "voice_mask_kv": voice_mask_kv,
        "top_class_a": top_class_a,
        "class_a_kv_set": class_a_kv_set,
        "class_b_kv_ids": class_b_kv_ids,
        "other_kv_ids": other_kv_ids,
        "norm_scores": norm_scores,
        "n_A": n_A,
        "n_B": n_B,
        "n_O": n_O,
    }


def estimate_scale_for_target(
    stats: dict,
    target_gpu: float | None,
    target_gpucpu: float | None,
    token_rate: float,
    durations: list[float] | None,
    chunk_size_s: float,
    overlap_s: float,
    top_k_chunks: int,
    loadback_factor: float,
    max_iter: int = 200,
    scale_hi: float = 200.0,
) -> tuple[float, float, float]:
    """Bisect retention_scale to hit a target GPU or GPU+CPU retention.

    Returns (best_scale, predicted_gpu_retention, predicted_gpucpu_retention).
    """
    if target_gpu is not None and target_gpucpu is not None:
        raise ValueError("Only one of target_gpu_retention or target_gpucpu_retention may be set")
    if target_gpu is None and target_gpucpu is None:
        raise ValueError("One of target_gpu_retention or target_gpucpu_retention must be set")

    target = target_gpu if target_gpu is not None else target_gpucpu
    use_gpu = target_gpu is not None

    n_A = stats["n_A"]
    n_B = stats["n_B"]
    n_O = stats["n_O"]
    norm_scores = stats["norm_scores"]
    total_kv = stats["n_A"] + stats["n_B"] + stats["n_O"]

    chunk_size, stride = compute_chunk_params(chunk_size_s, overlap_s, token_rate)
    segs_per_chunk = math.ceil(chunk_size / stride) if stride > 0 else 2

    if durations:
        num_audio_list = [max(1, int(d * token_rate)) for d in durations]
    else:
        num_audio_list = [max(1, int(100.0 * token_rate))]

    def _metric(scale: float) -> float:
        gpu_ret, gpucpu_ret = predict_retention(
            norm_scores, n_A, n_B, n_O, scale,
            num_audio_list, chunk_size, stride, stride,
            top_k_chunks, segs_per_chunk, total_kv,
            loadback_factor=loadback_factor,
        )
        return gpu_ret if use_gpu else gpucpu_ret

    lo, hi = 0.0, scale_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _metric(mid) < target:
            lo = mid
        else:
            hi = mid
    best_scale = (lo + hi) / 2
    pred_gpu, pred_gpucpu = predict_retention(
        norm_scores, n_A, n_B, n_O, best_scale,
        num_audio_list, chunk_size, stride, stride,
        top_k_chunks, segs_per_chunk, total_kv,
        loadback_factor=loadback_factor,
    )
    return best_scale, pred_gpu, pred_gpucpu


def build_config(
    head_scores: dict,
    token_rate: float,
    num_class_a: int = 5,
    retention_scale: float = 0.15,
    sensitivity_mode: str = "mean",
    top_voice: float = 0.10,
    chunk_size_s: float = 4.0,
    overlap_s: float = 2.0,
    durations: list[float] | None = None,
    top_k_chunks: int = 3,
    loadback_factor: float = 0.8,
) -> tuple[dict, dict]:
    """Return (config_dict, summary_dict).

    head_scores must contain: hit_ratio [L, num_q], voice_score_kv [L, num_kv],
    audio_sensitivity_k [L, num_kv], audio_sensitivity_v [L, num_kv].
    """
    stats = compute_classification_stats(head_scores, sensitivity_mode, num_class_a, top_voice)

    n_A = stats["n_A"]
    n_B = stats["n_B"]
    n_O = stats["n_O"]
    norm_scores = stats["norm_scores"]
    class_a_kv_set = stats["class_a_kv_set"]
    class_b_kv_ids = stats["class_b_kv_ids"]
    other_kv_ids = stats["other_kv_ids"]
    non_a_kv_ids = class_b_kv_ids + other_kv_ids
    num_layers = stats["num_layers"]
    num_kv_heads = stats["num_kv_heads"]
    num_kv_groups = stats["num_kv_groups"]
    total_kv = n_A + n_B + n_O

    r_B_list = [min(norm_scores[i] * retention_scale, 1.0) for i in range(n_B)]
    r_O_list = [min(norm_scores[n_B + i] * retention_scale, 1.0) for i in range(n_O)]
    r_B_avg = sum(r_B_list) / n_B if n_B > 0 else 0
    r_O_avg = sum(r_O_list) / n_O if n_O > 0 else 0

    chunk_size, stride = compute_chunk_params(chunk_size_s, overlap_s, token_rate)
    segs_per_chunk = math.ceil(chunk_size / stride) if stride > 0 else 2

    if durations:
        num_audio_list = [max(1, int(d * token_rate)) for d in durations]
    else:
        num_audio_list = [max(1, int(100.0 * token_rate))]

    gpu_retention, gpu_cpu_retention = predict_retention(
        norm_scores, n_A, n_B, n_O, retention_scale,
        num_audio_list, chunk_size, stride, stride,
        top_k_chunks, segs_per_chunk, total_kv,
        loadback_factor=loadback_factor,
    )

    norm_map = {non_a_kv_ids[i]: norm_scores[i] for i in range(len(non_a_kv_ids))}
    retention_ratios = []
    for i in range(total_kv):
        if i in class_a_kv_set:
            retention_ratios.append(1.0)
        else:
            ratio = min(norm_map[i] * retention_scale, 1.0)
            retention_ratios.append(round(ratio, 6))

    num_q_heads = num_kv_heads * num_kv_groups
    class_a_q_ids = []
    for idx in stats["top_class_a"]:
        layer = int(idx // num_kv_heads)
        kv = int(idx % num_kv_heads)
        q_start = layer * num_q_heads + kv * num_kv_groups
        class_a_q_ids.extend(range(q_start, q_start + num_kv_groups))

    config = {
        "chunk": {
            "size_s": chunk_size_s,
            "overlap_s": overlap_s,
            "token_rate": token_rate,
        },
        "heads": {
            "class_a": sorted(class_a_q_ids),
            "class_b": class_b_kv_ids,
            "retention_ratios": retention_ratios,
        },
        "selection": {"top_k_chunks": top_k_chunks, "eviction_lag": 3},
        "score": {"prefill_mode": "mean_text"},
        "sensitivity_mode": sensitivity_mode,
    }

    summary = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "num_voice_kv": int(stats["voice_mask_kv"].sum()),
        "n_A": n_A,
        "n_B": n_B,
        "n_O": n_O,
        "r_B_avg": r_B_avg,
        "r_O_avg": r_O_avg,
        "gpu_retention": gpu_retention,
        "gpu_cpu_retention": gpu_cpu_retention,
    }
    return config, summary


def main():
    parser = argparse.ArgumentParser(description="Generate WnW config from head_scores.pt")
    parser.add_argument("--head-scores-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--token-rate", type=float, required=True,
                        help="Audio token rate (Voxtral=12.5, Qwen2.5-Omni=25.0)")
    parser.add_argument("--num-class-a", type=int, default=5)

    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--retention-scale", type=float, default=0.15,
                             help="Global multiplier for non-Class-A retention ratios")
    scale_group.add_argument("--target-gpu-retention", type=float, default=None,
                             help="Target GPU-side retention; bisect retention_scale to match it")
    scale_group.add_argument("--target-gpucpu-retention", type=float, default=None,
                             help="Target GPU+CPU retention; bisect retention_scale to match it")

    parser.add_argument("--sensitivity-mode", type=str, default="mean",
                        choices=["k", "v", "mean"])
    parser.add_argument("--top-voice", type=float, default=0.10,
                        help="float=ratio (0.10 -> 10%%), int-valued float=count (10 -> 10 heads)")
    parser.add_argument("--chunk-size-s", type=float, default=4.0)
    parser.add_argument("--overlap-s", type=float, default=2.0)
    parser.add_argument("--duration-file", type=str, default=None,
                        help="File with one float per line (duration in seconds) for accurate retention prediction")
    parser.add_argument("--top-k-chunks", type=int, default=3,
                        help="Number of top chunks loaded during decode (default: 3)")
    parser.add_argument("--loadback-factor", type=float, default=0.8,
                        help="Empirical damping on CPU->GPU loadback (default: 0.8)")
    args = parser.parse_args()

    data = torch.load(args.head_scores_path, weights_only=True)

    durations = None
    if args.duration_file:
        with open(args.duration_file) as f:
            durations = [float(line.strip()) for line in f if line.strip()]

    retention_scale = args.retention_scale
    predicted_gpu = None
    predicted_gpucpu = None

    if args.target_gpu_retention is not None or args.target_gpucpu_retention is not None:
        stats = compute_classification_stats(data, args.sensitivity_mode, args.num_class_a, args.top_voice)
        retention_scale, predicted_gpu, predicted_gpucpu = estimate_scale_for_target(
            stats,
            target_gpu=args.target_gpu_retention,
            target_gpucpu=args.target_gpucpu_retention,
            token_rate=args.token_rate,
            durations=durations,
            chunk_size_s=args.chunk_size_s,
            overlap_s=args.overlap_s,
            top_k_chunks=args.top_k_chunks,
            loadback_factor=args.loadback_factor,
        )

    config, summary = build_config(
        data,
        token_rate=args.token_rate,
        num_class_a=args.num_class_a,
        retention_scale=retention_scale,
        sensitivity_mode=args.sensitivity_mode,
        top_voice=args.top_voice,
        chunk_size_s=args.chunk_size_s,
        overlap_s=args.overlap_s,
        durations=durations,
        top_k_chunks=args.top_k_chunks,
        loadback_factor=args.loadback_factor,
    )

    print(f"Sensitivity mode: {args.sensitivity_mode}")
    if args.target_gpu_retention is not None or args.target_gpucpu_retention is not None:
        target = args.target_gpu_retention if args.target_gpu_retention is not None else args.target_gpucpu_retention
        mode = "GPU" if args.target_gpu_retention is not None else "GPU+CPU"
        print(f"Target {mode} retention: {target:.4f} -> selected retention_scale={retention_scale:.6f}")
        print(f"Predicted GPU retention:     {predicted_gpu:.4f}")
        print(f"Predicted GPU+CPU retention: {predicted_gpucpu:.4f}")
    print(f"Voice KV heads: top_voice={args.top_voice} -> {summary['num_voice_kv']}/"
          f"{summary['num_layers'] * summary['num_kv_heads']} (by weighted_score)")
    print(f"Class A: {summary['n_A']} KV heads (GPU=1.0, CPU=0)")
    print(f"Class B: {summary['n_B']} KV heads (GPU avg={summary['r_B_avg']:.4f}, CPU=1-GPU)")
    print(f"Other:   {summary['n_O']} KV heads (GPU avg={summary['r_O_avg']:.4f}, CPU=0)")
    print(f"GPU retention:     {summary['gpu_retention']:.4f}")
    print(f"GPU+CPU retention: {summary['gpu_cpu_retention']:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    valid = [r for r in config["heads"]["retention_ratios"] if r is not None]
    print(f"\nConfig saved: {args.output}")
    if valid:
        print(f"Non-A retention range: [{min(valid):.4f}, {max(valid):.4f}], "
              f"mean: {sum(valid)/len(valid):.4f}")
    capped = sum(1 for r in valid if abs(r - 1.0) < 1e-6)
    if capped:
        print(f"Warning: {capped} heads capped at 1.0 (consider lowering retention-scale)")


if __name__ == "__main__":
    main()
