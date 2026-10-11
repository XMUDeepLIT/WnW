from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel


class BaseModelAdapter(ABC):
    """Model-agnostic interface for KV cache compression algorithms.

    Each supported model (Voxtral, Qwen2.5-Omni) implements one subclass.
    The patched_forward and Session only interact with models through this interface.
    """

    name: str

    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs) -> tuple[PreTrainedModel, object]:
        """Load model + processor. Returns (model, processor)."""

    def build_inputs(
        self, audio: "str | np.ndarray", sr: int, text: str, processor, device: str,
        task: dict | None = None,
    ) -> dict[str, Tensor]:
        """Build model inputs from raw audio + text prompt.

        Args:
            task: Optional task metadata dict. If provided, must contain a "name" key.
                  Adapters may use this for task-specific routing (e.g. ASR vs chat).
        """
        if task is not None:
            assert isinstance(task, dict) and "name" in task, (
                f"task must be a dict with a 'name' key, got: {task!r}"
            )
        return self._build_inputs(audio, sr, text, processor, device, task)

    @abstractmethod
    def _build_inputs(
        self, audio: "str | np.ndarray", sr: int, text: str, processor, device: str,
        task: dict | None = None,
    ) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def get_decoder_layers(self, model: PreTrainedModel) -> nn.ModuleList:
        """Return the decoder layer list (e.g. model.language_model.model.layers)."""

    @abstractmethod
    def get_text_config(self, model: PreTrainedModel) -> PretrainedConfig:
        """Return the text config containing num_heads, head_dim, etc."""

    @abstractmethod
    def locate_audio_tokens(
        self, input_ids: Tensor, attention_mask: Tensor | None, processor
    ) -> tuple[Tensor, Tensor]:
        """Return (audio_starts, audio_ends) as Tensor[B] int64, using [bos+1, eos) convention.

        Raises ValueError if any sample in the batch has no audio tokens.
        """

    @abstractmethod
    def compute_qkv(
        self, attn_module: nn.Module, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Q, K, V from hidden_states. Returns tensors shaped [B, heads, S, head_dim]."""

    @abstractmethod
    def apply_rope(
        self, q: Tensor, k: Tensor, position_embeddings: tuple[Tensor, Tensor],
        attn_module: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary position embedding (model-specific variant)."""

    @abstractmethod
    def attention_meta(self, attn_module: nn.Module) -> dict:
        """Return {head_dim, num_kv_heads, num_kv_groups, scaling, layer_idx}."""

    @abstractmethod
    def prepare_for_inference(self, model: PreTrainedModel) -> dict:
        """Prepare model (e.g. set eager attn). Returns state dict for restore."""

    @abstractmethod
    def restore_after_inference(self, model: PreTrainedModel, state: dict) -> None:
        """Restore model state saved by prepare_for_inference."""

    @abstractmethod
    def generate(self, model: PreTrainedModel, inputs: dict, **kwargs) -> Tensor:
        """Run model.generate with model-specific handling."""

    @abstractmethod
    def get_logits(self, model: PreTrainedModel, outputs) -> Tensor:
        """Return logits tensor from model outputs (analysis path)."""

    @property
    def token_rate(self) -> float:
        """Audio tokens per second, used by analysis scripts."""
        return 12.5

    @property
    def requires_input_ids(self) -> bool:
        """Whether the analysis forward needs input_ids alongside inputs_embeds.

        Only models whose position encoding depends on token ids (e.g. Qwen's
        3D RoPE) need this. Llama-style decoders (Voxtral) must not receive both
        input_ids and inputs_embeds, or the forward raises.
        """
        return False

    @property
    def feature_time_dim(self) -> int:
        """Dimension index of the time axis in input_features for batched padding.

        Voxtral: [B, n_mels, T] -> -1 (last dim)
        Qwen:    [B, T, feat_dim] -> -2 (second-to-last dim)
        """
        return -1

    @abstractmethod
    def get_text_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return the text decoder model used for analysis forwards.

        E.g. model.language_model for Voxtral, model.thinker.model for Qwen-Omni.
        """

    @abstractmethod
    def get_audio_token_id(self, model: PreTrainedModel) -> int:
        """Return the audio token id used to locate audio positions."""

    @abstractmethod
    def build_analysis_inputs(
        self,
        model: PreTrainedModel,
        processor,
        audio_path: str,
        question: str,
        answer: str,
        task: dict | None = None,
    ) -> dict[str, Tensor]:
        """Build analysis inputs for one sample.

        Returns dict with at least: input_ids, inputs_embeds, labels.
        All tensors should be on CPU for cheap torch.save.
        """

    def assert_compatible(self, model: PreTrainedModel) -> None:
        """Verify model is structurally compatible. Raises AssertionError if not."""
        text_config = self.get_text_config(model)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = text_config.hidden_size // text_config.num_attention_heads
        assert head_dim & (head_dim - 1) == 0, (
            f"head_dim={head_dim} must be a power of 2 for Triton kernels"
        )
        layers = self.get_decoder_layers(model)
        assert len(layers) > 0, "No decoder layers found"
        first_attn = layers[0].self_attn
        assert hasattr(first_attn, "layer_idx"), "Attention module missing layer_idx"

    def degqa(self, model: PreTrainedModel) -> None:
        """Convert GQA model to equivalent MHA in-place (irreversible)."""
        from common.adapters.degqa import degqa_model
        degqa_model(model, self)
