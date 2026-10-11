
import argparse
import os
import sys

import torch
from tqdm import tqdm

from analysis.common.model_config import get_text_config
from analysis.wnw_config.code.head_classification import load_sample_audio_sensitivity
from analysis.wnw_config.code.voice_heads import (
    aggregate_voice_score_kv_head,
    load_hit_ratio,
)


def main():
    parser = argparse.ArgumentParser(description="Compute head classification scores")
    parser.add_argument("--hit-ratio-path", type=str, required=True)
    parser.add_argument("--kv-imp-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    tc = get_text_config(args.model_path)
    num_q_heads = tc["num_attention_heads"]
    num_kv_heads = tc["num_key_value_heads"]
    num_kv_groups = num_q_heads // num_kv_heads

    hr_data = load_hit_ratio(args.hit_ratio_path)
    hit_ratio = hr_data["hit_ratio"]
    sample_ids = hr_data["sample_ids"]
    num_layers, _ = hit_ratio.shape

    sens_k_sum = torch.zeros(num_layers, num_kv_heads)
    sens_v_sum = torch.zeros(num_layers, num_kv_heads)
    sample_count = 0

    n_samples = min(args.max_samples, len(sample_ids))
    for sid in tqdm(sample_ids[:n_samples], desc="Aggregating sensitivity"):
        result = load_sample_audio_sensitivity(sid, args.kv_imp_dir)
        if result is None:
            continue
        sens_k, sens_v = result
        sens_k_sum += sens_k
        sens_v_sum += sens_v
        sample_count += 1

    if sample_count == 0:
        print("ERROR: No samples processed successfully!")
        sys.exit(1)

    audio_sensitivity_k = sens_k_sum / sample_count
    audio_sensitivity_v = sens_v_sum / sample_count
    print(f"Processed {sample_count}/{n_samples} samples")

    voice_score_kv = aggregate_voice_score_kv_head(hit_ratio, num_kv_groups)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "audio_sensitivity_k": audio_sensitivity_k,
        "audio_sensitivity_v": audio_sensitivity_v,
        "voice_score_kv": voice_score_kv,
        "hit_ratio": hit_ratio,
    }, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
