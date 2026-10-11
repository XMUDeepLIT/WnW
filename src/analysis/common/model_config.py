
from transformers import AutoConfig


def get_text_config(model_path: str) -> dict:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "thinker_config"):
        tc = config.thinker_config.text_config
    elif hasattr(config, "text_config"):
        tc = config.text_config
    else:
        raise ValueError(f"Cannot find text_config in model config at {model_path}")
    return {
        "num_hidden_layers": tc.num_hidden_layers,
        "num_attention_heads": tc.num_attention_heads,
        "num_key_value_heads": tc.num_key_value_heads,
    }
