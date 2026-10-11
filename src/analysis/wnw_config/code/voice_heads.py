
import json

import torch


def load_hit_ratio(summary_json_path: str) -> dict:
    """Load hit_ratio summary.json produced by the importance pipeline.

    Returns dict with keys: hit_ratio [num_layers, num_q_heads], top_50,
    statistics, sample_ids.
    """
    with open(summary_json_path) as f:
        data = json.load(f)
    hr = torch.tensor(data["hit_ratio_mean"])
    return {
        "hit_ratio": hr,
        "top_50": data["top_50_heads"],
        "statistics": data["statistics"],
        "sample_ids": data["sample_ids"],
    }


def get_top_voice_q_indices(hit_ratio: torch.Tensor, top_pct: float = 0.10) -> list[int]:
    flat = hit_ratio.flatten()
    n_top = max(1, int(len(flat) * top_pct))
    _, indices = flat.topk(n_top)
    return indices.tolist()


def aggregate_voice_score_kv_head(hit_ratio: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
    """Average per-q-head scores into per-kv-head. Returns [num_layers, num_kv_heads]."""
    num_layers, num_q_heads = hit_ratio.shape
    num_kv_heads = num_q_heads // num_kv_groups
    return hit_ratio.reshape(num_layers, num_kv_heads, num_kv_groups).mean(dim=2)
