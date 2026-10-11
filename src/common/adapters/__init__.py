from common.adapters.base import BaseModelAdapter
from common.adapters.voxtral import VoxtralAdapter
from common.adapters.qwen2_5_omni import Qwen25OmniAdapter

_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    "voxtral": VoxtralAdapter,
    "qwen2_5_omni": Qwen25OmniAdapter,
}


def get_adapter(name: str) -> BaseModelAdapter:
    """Get adapter instance by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown adapter: {name!r}. Available: {list(_REGISTRY.keys())}")
    return cls()
