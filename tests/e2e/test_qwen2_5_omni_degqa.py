
import pytest
import torch
import torch.nn.functional as F

from common.adapters import get_adapter
from tests.conftest import resolve_audio_path


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def _get_first_logits(model, inputs):
    with torch.no_grad():
        out = model.thinker(
            **{k: v for k, v in inputs.items() if k in (
                "input_ids", "attention_mask", "input_features",
                "feature_attention_mask",
            )},
            return_dict=True,
        )
    return out.logits[0, -1, :].float().cpu()


def test_degqa_equivalence(first_sample, data_root, model_path, device):
    sample = first_sample
    audio_path = resolve_audio_path(sample, data_root)

    adapter = get_adapter("qwen2_5_omni")

    m_gqa, processor = adapter.load(model_path, device, degqa=False)
    inputs = adapter.build_inputs(audio_path, 16000, "Transcribe the audio.", processor, device)
    logits_gqa = _get_first_logits(m_gqa, inputs)

    del m_gqa
    torch.cuda.empty_cache()

    m_mha, _ = adapter.load(model_path, device, degqa=True)
    logits_mha = _get_first_logits(m_mha, inputs)

    diff = (logits_gqa - logits_mha).abs()
    cos = F.cosine_similarity(logits_gqa.unsqueeze(0), logits_mha.unsqueeze(0)).item()
    top1_gqa = logits_gqa.argmax().item()
    top1_mha = logits_mha.argmax().item()

    assert cos > 0.9999, f"Cosine similarity too low: {cos}"
    assert top1_gqa == top1_mha, f"Top-1 mismatch: {top1_gqa} vs {top1_mha}"
