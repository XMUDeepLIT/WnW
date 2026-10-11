
import inspect

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from common.adapters.base import BaseModelAdapter


class AttentionScoreAnalyzer:
    def __init__(self, model: PreTrainedModel, adapter: BaseModelAdapter):
        self.model = model
        self.adapter = adapter
        self.text_model = adapter.get_text_model(model)
        self.layers = self.text_model.model.layers
        self.num_layers = len(self.layers)
        self.audio_token_id = adapter.get_audio_token_id(model)

        self.attn_cls = type(self.layers[0].self_attn)
        self._original_forward = self.attn_cls.forward
        self._patched = False

        attn_module = self.layers[0].self_attn
        self._attn_module_pkg = inspect.getmodule(type(attn_module))
        self._ALL_ATTENTION_FUNCTIONS = getattr(self._attn_module_pkg, "ALL_ATTENTION_FUNCTIONS")
        self._eager_attention_forward = getattr(self._attn_module_pkg, "eager_attention_forward")

        self._attn_weights = [None] * self.num_layers
        self._answer_positions = None
        self._target_positions = None

    def patch(self):
        assert not self._patched
        self._attn_state = self.adapter.prepare_for_inference(self.model)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.forward = self._make_patched_forward(layer_idx).__get__(
                layer.self_attn, self.attn_cls
            )
        self._patched = True

    def unpatch(self):
        assert self._patched
        for layer in self.layers:
            layer.self_attn.forward = self._original_forward.__get__(
                layer.self_attn, self.attn_cls
            )
        self.adapter.restore_after_inference(self.model, self._attn_state)
        self._patched = False

    def _make_patched_forward(self, layer_idx):
        analyzer = self

        def patched_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]

            query_states, key_states, value_states = analyzer.adapter.compute_qkv(
                self, hidden_states
            )

            cos, sin = position_embeddings
            query_states, key_states = analyzer.adapter.apply_rope(
                query_states, key_states, (cos, sin), self
            )

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )

            attention_interface = analyzer._ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, analyzer._eager_attention_forward
            )

            attn_kwargs = {
                "dropout": 0.0 if not self.training else self.attention_dropout,
                "scaling": self.scaling,
            }
            if hasattr(self, "sliding_window"):
                attn_kwargs["sliding_window"] = self.sliding_window
            if kwargs.get("position_ids") is not None:
                attn_kwargs["position_ids"] = kwargs["position_ids"]
            if kwargs.get("cache_position") is not None:
                attn_kwargs["cache_position"] = kwargs["cache_position"]

            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                **attn_kwargs,
            )

            repeat_kv = getattr(analyzer._attn_module_pkg, "repeat_kv")
            key_states_for_weights = repeat_kv(key_states, self.num_key_value_groups)
            if analyzer._answer_positions is not None:
                positions = analyzer._answer_positions.to(query_states.device)
                q_slices = query_states[:, :, positions, :]
                attn_rows = (
                    torch.matmul(q_slices, key_states_for_weights.transpose(2, 3))
                    * self.scaling
                )
                if attention_mask is not None and attention_mask.dim() == 4:
                    attn_rows = attn_rows + attention_mask[:, :, positions, :]
                else:
                    raise RuntimeError(
                        "No 4D attention mask reached the attention layer, so the "
                        "hand-computed rows would attend to future tokens. Ensure "
                        "the adapter's prepare_for_inference selects eager attention."
                    )
                attn_rows = F.softmax(attn_rows, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                result = attn_rows[0]
                if analyzer._target_positions is not None:
                    result = result[:, :, analyzer._target_positions.to(result.device)]
                analyzer._attn_weights[layer_idx] = result.detach().cpu()
            else:
                attn_weights = (
                    torch.matmul(query_states, key_states_for_weights.transpose(2, 3))
                    * self.scaling
                )
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                result = attn_weights[0]
                if analyzer._target_positions is not None:
                    result = result[:, :, analyzer._target_positions.to(result.device)]
                analyzer._attn_weights[layer_idx] = result.detach().cpu()

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        return patched_forward

    def compute(self, inputs_embeds, labels, input_ids):
        """Run one forward and capture per-layer answer-row attention.

        Returns dict:
            scores [num_layers, num_heads, num_answer_tokens, num_audio + num_answer]
            prefix_len, answer_ids, num_audio
        """
        assert self._patched

        device = next(self.model.parameters()).device
        inputs_embeds = inputs_embeds.to(device)
        labels = labels.to(device)
        input_ids = input_ids.to(device)
        seq_len = inputs_embeds.shape[1]

        answer_mask = labels[0] != -100
        answer_positions = answer_mask.nonzero(as_tuple=True)[0]
        answer_ids = labels[0, answer_positions].tolist()

        audio_mask = (input_ids[0] == self.audio_token_id)
        audio_positions = audio_mask.nonzero(as_tuple=True)[0]
        num_audio = len(audio_positions)

        target_positions = torch.cat([audio_positions, answer_positions])

        self._attn_weights = [None] * self.num_layers
        self._answer_positions = answer_positions
        self._target_positions = target_positions

        attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
        fwd_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        if self.adapter.requires_input_ids:
            fwd_kwargs["input_ids"] = input_ids
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.text_model(**fwd_kwargs)

        scores = torch.stack(self._attn_weights)

        self._attn_weights = [None] * self.num_layers
        self._answer_positions = None
        self._target_positions = None

        return {
            "scores": scores,
            "prefix_len": answer_positions[0].item(),
            "answer_ids": answer_ids,
            "num_audio": num_audio,
        }
