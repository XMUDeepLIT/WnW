
import torch
import torch.nn.functional as F
from transformers import DynamicCache, PreTrainedModel

from common.adapters.base import BaseModelAdapter


class GradCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.k_checkpoints = {}
        self.v_checkpoints = {}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        super().update(key_states, value_states, layer_idx, *args, **kwargs)
        layer = self.layers[layer_idx]
        layer.keys = layer.keys.detach()
        layer.values = layer.values.detach()

        k_cp = layer.keys.clone().requires_grad_(True)
        v_cp = layer.values.clone().requires_grad_(True)
        k_cp.retain_grad()
        v_cp.retain_grad()
        self.k_checkpoints[layer_idx] = k_cp
        self.v_checkpoints[layer_idx] = v_cp
        return k_cp, v_cp

    def clear_checkpoints(self):
        self.k_checkpoints.clear()
        self.v_checkpoints.clear()


class KVImportanceAnalyzer:
    def __init__(self, model: PreTrainedModel, adapter: BaseModelAdapter):
        self.model = model
        self.adapter = adapter
        self.text_model = adapter.get_text_model(model)
        self.num_layers = len(self.text_model.model.layers)

    def _set_attn_implementation(self, impl: str):
        self.text_model.config._attn_implementation = impl
        self.text_model.model.config._attn_implementation = impl
        for layer in self.text_model.model.layers:
            layer.self_attn.config._attn_implementation = impl

    def compute(self, inputs_embeds, labels, audio_positions, input_ids=None):
        """Return audio-position K/V gradient norms averaged over answer steps.

        audio_positions must all be < prefix_len (audio lives entirely in prefix).
        input_ids: optional, needed by models that require token IDs for position
                   computation (e.g. Qwen2.5-Omni uses input_ids for 3D RoPE).
        """
        device = next(self.model.parameters()).device
        inputs_embeds = inputs_embeds.to(device)
        labels = labels.to(device)
        if input_ids is not None:
            input_ids = input_ids.to(device)

        if isinstance(audio_positions, torch.Tensor):
            audio_pos = audio_positions.to(torch.long).cpu()
        else:
            audio_pos = torch.tensor(audio_positions, dtype=torch.long)
        num_audio = audio_pos.numel()
        assert num_audio > 0, "audio_positions is empty"

        answer_mask = labels[0] != -100
        answer_positions = answer_mask.nonzero(as_tuple=True)[0]
        num_answer_tokens = len(answer_positions)
        answer_ids = labels[0, answer_positions].tolist()
        prefix_len = answer_positions[0].item()

        assert int(audio_pos.max().item()) < prefix_len, (
            f"audio_positions max ({int(audio_pos.max().item())}) must be < prefix_len ({prefix_len})"
        )

        n = self.num_layers
        cache = GradCache()

        orig_attn_impl = getattr(self.text_model.config, "_attn_implementation", None)
        self._set_attn_implementation("sdpa")

        prefix_embeds = inputs_embeds[:, :prefix_len, :]
        prefix_mask = torch.ones(1, prefix_len, device=device, dtype=torch.long)

        fwd_kwargs = dict(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            past_key_values=cache,
            use_cache=True,
        )
        if input_ids is not None:
            fwd_kwargs["input_ids"] = input_ids[:, :prefix_len]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.text_model(**fwd_kwargs)

        logits_t = self.adapter.get_logits(self.model, outputs)[0, -1, :].float()
        loss_t = -F.log_softmax(logits_t, dim=-1)[answer_ids[0]]

        all_k = [cache.k_checkpoints[i] for i in range(n)]
        all_v = [cache.v_checkpoints[i] for i in range(n)]

        grads = torch.autograd.grad(
            loss_t, all_k + all_v, retain_graph=False, create_graph=False
        )

        sens_k_step = torch.stack([
            grads[i][0].float().norm(dim=-1)[:, audio_pos].mean(dim=-1).cpu()
            for i in range(n)
        ])
        sens_v_step = torch.stack([
            grads[n + i][0].float().norm(dim=-1)[:, audio_pos].mean(dim=-1).cpu()
            for i in range(n)
        ])
        accum_k = sens_k_step
        accum_v = sens_v_step

        del outputs, grads
        cache.clear_checkpoints()

        self._set_attn_implementation("eager")

        for t in range(1, num_answer_tokens):
            token_embeds = inputs_embeds[:, prefix_len + t - 1: prefix_len + t, :]
            step_mask = torch.ones(1, prefix_len + t, device=device, dtype=torch.long)

            fwd_kwargs = dict(
                inputs_embeds=token_embeds,
                attention_mask=step_mask,
                past_key_values=cache,
                use_cache=True,
            )
            if input_ids is not None:
                fwd_kwargs["input_ids"] = input_ids[:, prefix_len + t - 1: prefix_len + t]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.text_model(**fwd_kwargs)

            logits_t = self.adapter.get_logits(self.model, outputs)[0, 0, :].float()
            loss_t = -F.log_softmax(logits_t, dim=-1)[answer_ids[t]]

            all_k = [cache.k_checkpoints[i] for i in range(n)]
            all_v = [cache.v_checkpoints[i] for i in range(n)]

            grads = torch.autograd.grad(
                loss_t, all_k + all_v, retain_graph=False, create_graph=False
            )
            sens_k_step = torch.stack([
                grads[i][0].float().norm(dim=-1)[:, audio_pos].mean(dim=-1).cpu()
                for i in range(n)
            ])
            sens_v_step = torch.stack([
                grads[n + i][0].float().norm(dim=-1)[:, audio_pos].mean(dim=-1).cpu()
                for i in range(n)
            ])
            accum_k = accum_k + sens_k_step
            accum_v = accum_v + sens_v_step
            del outputs, grads
            cache.clear_checkpoints()

        torch.cuda.empty_cache()

        if orig_attn_impl is not None:
            self._set_attn_implementation(orig_attn_impl)

        return {
            "audio_sensitivity_k": accum_k / num_answer_tokens,
            "audio_sensitivity_v": accum_v / num_answer_tokens,
            "num_audio": int(num_audio),
            "num_answer_steps": int(num_answer_tokens),
            "prefix_len": prefix_len,
        }
