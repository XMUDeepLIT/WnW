
from __future__ import annotations

import types
from collections.abc import Callable

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from common.adapters.base import BaseModelAdapter
from common.kv.flat_cache import FlatVarlenCache


class AlgorithmSession:
    """Context manager that installs/uninstalls attention patches.

    Usage:
        with WnWSession(model, adapter, config, cache, processor) as sess:
            for sample in samples:
                inputs = adapter.build_inputs(...)
                sess.setup_audio(inputs["input_ids"], inputs.get("attention_mask"))
                output = sess.generate(inputs, max_new_tokens=...)

    The model is NOT modified outside the with-block.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        adapter: BaseModelAdapter,
        cache: FlatVarlenCache,
        processor=None,
    ):
        self.model = model
        self.adapter = adapter
        self.cache = cache
        self.processor = processor
        self._adapter_state: dict | None = None
        self._orig_forwards: list[tuple[nn.Module, Callable]] = []

    def __enter__(self):
        self.cache.reset()
        self.adapter.assert_compatible(self.model)
        self._adapter_state = self.adapter.prepare_for_inference(self.model)
        self._install_patches()
        return self

    def __exit__(self, *exc):
        self._restore_patches()
        self.adapter.restore_after_inference(self.model, self._adapter_state)
        self._adapter_state = None

    def setup_audio(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        """Call before each generate() to set audio range for the current sample/batch."""
        starts, ends = self.adapter.locate_audio_tokens(
            input_ids, attention_mask, self.processor
        )
        self.cache.setup_audio(starts, ends, attention_mask)

    def generate(self, inputs: dict, **gen_kwargs) -> Tensor:
        """Run model.generate with our cache explicitly passed."""
        gen_kwargs["past_key_values"] = self.cache
        return self.adapter.generate(self.model, inputs, **gen_kwargs)

    def get_stats(self) -> list[dict]:
        """Return per-sample runtime statistics collected during inference."""
        stats = getattr(self.cache, "stats", None)
        if stats is None:
            return []
        return stats.get_all_stats()


    def _install_patches(self):
        layers = self.adapter.get_decoder_layers(self.model)
        for layer_module in layers:
            attn = layer_module.self_attn
            self._orig_forwards.append((attn, attn.forward))
            patched = self._make_patched_forward(attn)
            attn.forward = types.MethodType(patched, attn)

    def _restore_patches(self):
        for attn, orig_forward in self._orig_forwards:
            attn.forward = orig_forward
        self._orig_forwards.clear()

    def _make_patched_forward(self, attn_module: nn.Module) -> Callable:
        raise NotImplementedError(
            "AlgorithmSession is abstract. Use a subclass (e.g. WnWSession)."
        )
