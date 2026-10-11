import os

import numpy as np
import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from common.adapters.base import BaseModelAdapter


class VoxtralAdapter(BaseModelAdapter):
    """Adapter for VoxtralForConditionalGeneration (Mistral/Llama text decoder)."""

    name = "voxtral"

    def load(self, model_path: str, device: str, degqa: bool = True,
             **kwargs) -> tuple[PreTrainedModel, object]:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, **kwargs,
        )
        model.eval()
        if degqa:
            self.degqa(model)
        return model, processor

    def _build_inputs(
        self, audio, sr: int, text: str, processor, device: str,
        task: dict | None = None,
    ) -> dict[str, Tensor]:
        import soundfile as sf

        if isinstance(audio, str):
            audio_path = audio
            _tmp_path = None
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, sr)
            tmp.close()
            audio_path = tmp.name
            _tmp_path = tmp.name

        try:
            if task and isinstance(task, dict) and task.get("name", "").lower() == "asr":
                lang_code = self._resolve_lang_code(task.get("lang", ""))
                assert lang_code is not None, (
                    f"ASR task requires a valid language name (e.g. 'English'), "
                    f"got: {task.get('lang')!r}"
                )
                inputs = processor.apply_transcription_request(
                    language=lang_code, audio=audio_path,
                    model_id=processor.tokenizer.name_or_path,
                )
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()}

            conversations = [[{"role": "user", "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": text},
            ]}]]
            inputs = processor.apply_chat_template(conversations)
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()}
        finally:
            if _tmp_path:
                os.unlink(_tmp_path)

    @staticmethod
    def _resolve_lang_code(lang_name: str) -> str | None:
        if not lang_name:
            return None
        import pycountry
        match = pycountry.languages.get(name=lang_name)
        if match and hasattr(match, "alpha_2"):
            return match.alpha_2
        return None

    def get_decoder_layers(self, model: PreTrainedModel) -> nn.ModuleList:
        return model.language_model.model.layers

    def get_text_config(self, model: PreTrainedModel) -> PretrainedConfig:
        return model.language_model.model.layers[0].self_attn.config

    def locate_audio_tokens(
        self, input_ids: Tensor, attention_mask: Tensor | None, processor
    ) -> tuple[Tensor, Tensor]:
        audio_token_id = processor.audio_token_id
        B = input_ids.shape[0]
        starts, ends = [], []
        for b in range(B):
            mask = input_ids[b] == audio_token_id
            if not mask.any():
                raise ValueError(f"Sample {b} has no audio tokens")
            positions = mask.nonzero(as_tuple=True)[0]
            starts.append(positions[0].item())
            ends.append(positions[-1].item() + 1)
        return (
            torch.tensor(starts, dtype=torch.int64, device=input_ids.device),
            torch.tensor(ends, dtype=torch.int64, device=input_ids.device),
        )

    def compute_qkv(
        self, attn_module: nn.Module, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, S, _ = hidden_states.shape
        head_dim = attn_module.head_dim
        q = attn_module.q_proj(hidden_states).view(B, S, -1, head_dim).transpose(1, 2)
        k = attn_module.k_proj(hidden_states).view(B, S, -1, head_dim).transpose(1, 2)
        v = attn_module.v_proj(hidden_states).view(B, S, -1, head_dim).transpose(1, 2)
        return q, k, v

    def apply_rope(
        self, q: Tensor, k: Tensor, position_embeddings: tuple[Tensor, Tensor],
        attn_module: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(q, k, cos, sin)

    def attention_meta(self, attn_module: nn.Module) -> dict:
        config = attn_module.config
        num_kv_heads = config.num_key_value_heads
        return {
            "head_dim": attn_module.head_dim,
            "num_kv_heads": num_kv_heads,
            "num_kv_groups": config.num_attention_heads // num_kv_heads,
            "scaling": attn_module.scaling,
            "layer_idx": attn_module.layer_idx,
        }

    def prepare_for_inference(self, model: PreTrainedModel) -> dict:
        return {}

    def restore_after_inference(self, model: PreTrainedModel, state: dict) -> None:
        pass

    def generate(self, model: PreTrainedModel, inputs: dict, **kwargs) -> Tensor:
        return model.generate(**inputs, **kwargs)

    def get_logits(self, model: PreTrainedModel, outputs) -> Tensor:
        return outputs.logits

    def get_text_model(self, model: PreTrainedModel) -> PreTrainedModel:
        return model.language_model

    def get_audio_token_id(self, model: PreTrainedModel) -> int:
        return model.config.audio_token_id

    def build_analysis_inputs(
        self,
        model: PreTrainedModel,
        processor,
        audio_path: str,
        question: str,
        answer: str,
        task: dict | None = None,
    ) -> dict[str, Tensor]:

        device = next(model.parameters()).device

        def _to_device(inputs):
            result = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        result[k] = v.to(device, dtype=torch.bfloat16)
                    else:
                        result[k] = v.to(device)
                else:
                    result[k] = v
            return result

        use_transcription = False
        if task and isinstance(task, dict) and task.get("name", "").lower() == "asr":
            lang_code = self._resolve_lang_code(task.get("lang", ""))
            if lang_code:
                prefix = processor.apply_transcription_request(
                    language=lang_code, audio=audio_path,
                    model_id=processor.tokenizer.name_or_path,
                )
                use_transcription = True

        if not use_transcription:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": question},
                ],
            }]
            prefix = processor.apply_chat_template(conversation)

        prefix = _to_device(prefix)
        prefix_input_ids = prefix["input_ids"]
        input_features = prefix["input_features"]
        prefix_len = prefix_input_ids.shape[1]

        answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        answer_ids.append(processor.tokenizer.eos_token_id)
        answer_tensor = torch.tensor([answer_ids], device=device)
        full_input_ids = torch.cat([prefix_input_ids, answer_tensor], dim=1)

        inputs_embeds = model.get_input_embeddings()(full_input_ids)
        audio_embeds = model.get_audio_features(
            input_features, return_dict=True
        ).pooler_output
        audio_token_mask = (full_input_ids == self.get_audio_token_id(model)).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_token_mask.to(inputs_embeds.device),
            audio_embeds.to(inputs_embeds.device),
        )

        labels = torch.full_like(full_input_ids, -100)
        labels[0, prefix_len:] = answer_tensor[0]

        return {
            "input_ids": full_input_ids.cpu(),
            "inputs_embeds": inputs_embeds.cpu().to(torch.bfloat16),
            "labels": labels.cpu(),
        }
