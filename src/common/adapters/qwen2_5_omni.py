
from torch import Tensor, nn
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb

from common.adapters.qwen_omni import QwenOmniBaseAdapter


class Qwen25OmniAdapter(QwenOmniBaseAdapter):
    """Qwen2.5-Omni-3B / 7B: chunked mRoPE with mrope_section."""

    name = "qwen2_5_omni"

    def __init__(self):
        self._mrope_section: list[int] | None = None

    def _get_mrope_section(self, attn_module: nn.Module) -> list[int]:
        if self._mrope_section is None:
            self._mrope_section = attn_module.config.rope_parameters["mrope_section"]
        return self._mrope_section

    def apply_rope(
        self, q: Tensor, k: Tensor, position_embeddings: tuple[Tensor, Tensor],
        attn_module: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        cos, sin = position_embeddings
        mrope_section = self._get_mrope_section(attn_module)
        return apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
