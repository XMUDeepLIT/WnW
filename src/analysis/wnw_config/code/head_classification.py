
import os

import torch


def load_sample_audio_sensitivity(
    sample_id: str,
    kv_imp_dir: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (audio_sensitivity_k, audio_sensitivity_v), each [num_layers, num_kv_heads].

    Returns None when the sample file is missing.
    """
    kv_path = os.path.join(kv_imp_dir, f"{sample_id}.pt")
    if not os.path.exists(kv_path):
        return None
    kv_data = torch.load(kv_path, weights_only=True)
    return kv_data["audio_sensitivity_k"], kv_data["audio_sensitivity_v"]
