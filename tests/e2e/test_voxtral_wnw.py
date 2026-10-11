
import pytest
import torch

from common.adapters import get_adapter
from algorithms.wnw.code.config import WnWConfig
from algorithms.wnw.code.cache import WnWCache
from algorithms.wnw.code.session import WnWSession
from tests.conftest import resolve_audio_path


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_wnw_voxtral(first_sample, data_root, model_path, device):
    sample = first_sample

    adapter = get_adapter("voxtral")
    model, processor = adapter.load(model_path, device)

    audio_path = resolve_audio_path(sample, data_root)
    inputs = adapter.build_inputs(audio_path, 16000, sample["question"], processor, device)

    text_config = adapter.get_text_config(model)
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    num_q_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // num_q_heads)

    class_a = [l * num_q_heads + q for l in range(num_layers // 2) for q in range(num_q_heads)]
    class_b = [l * num_kv_heads + h for l in range(num_layers // 2, num_layers) for h in range(num_kv_heads)]

    class_b_set = set(class_b)
    retention_ratios = []
    for l in range(num_layers):
        for h in range(num_kv_heads):
            gid = l * num_kv_heads + h
            retention_ratios.append(0.5 if gid in class_b_set else None)

    config = WnWConfig(
        class_a=class_a,
        class_b=class_b,
        retention_ratios=retention_ratios,
        chunk_size_s=2.0,
        overlap_s=0.5,
        token_rate=adapter.token_rate,
        top_k_chunks=2,
        eviction_lag=1,
        prefill_score_mode="last",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
    )

    cache = WnWCache(
        config=config,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        device=model.device,
        adapter=adapter,
    )

    with WnWSession(model, adapter, cache, processor) as sess:
        sess.setup_audio(inputs["input_ids"], inputs.get("attention_mask"))
        with torch.no_grad():
            output_ids = sess.generate(inputs, max_new_tokens=50)

    input_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
    assert text, "WnW produced empty output"
    assert not hasattr(model, "wnw_cache")
