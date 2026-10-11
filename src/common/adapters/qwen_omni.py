
import os

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel

from common.adapters.base import BaseModelAdapter


class QwenOmniBaseAdapter(BaseModelAdapter):
    """Shared logic for the Qwen2.5-Omni thinker-based audio LLM adapter."""

    name = "qwen_omni_base"

    audio_bos_token = "<|audio_bos|>"
    audio_eos_token = "<|audio_eos|>"
    chat_template_kwargs: dict = {}

    @property
    def feature_time_dim(self) -> int:
        return -2

    @property
    def requires_input_ids(self) -> bool:
        return True

    def _model_class(self):
        from transformers import Qwen2_5OmniForConditionalGeneration
        return Qwen2_5OmniForConditionalGeneration

    def load(self, model_path: str, device: str, degqa: bool = True,
             attn_implementation: str = "flash_attention_2",
             **kwargs) -> tuple[PreTrainedModel, object]:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = self._model_class().from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, attn_implementation=attn_implementation,
            **kwargs,
        )
        if hasattr(model, "disable_talker"):
            model.disable_talker()
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
            waveform, file_sr = sf.read(audio_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if file_sr != 16000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=16000)

            conversations = [[{"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": text},
            ]}]]
            text_input = processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=False,
                **self.chat_template_kwargs,
            )
            inputs = processor(
                text=text_input, audio=[waveform],
                return_tensors="pt", padding=True, sampling_rate=16000,
            )
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()}
        finally:
            if _tmp_path:
                os.unlink(_tmp_path)

    def get_decoder_layers(self, model: PreTrainedModel) -> nn.ModuleList:
        return model.thinker.model.layers

    def get_text_config(self, model: PreTrainedModel) -> PretrainedConfig:
        return model.config.thinker_config.text_config

    def locate_audio_tokens(
        self, input_ids: Tensor, attention_mask: Tensor | None, processor
    ) -> tuple[Tensor, Tensor]:
        tok = processor.tokenizer
        audio_start_id = tok.convert_tokens_to_ids(self.audio_bos_token)
        audio_end_id = tok.convert_tokens_to_ids(self.audio_eos_token)
        B = input_ids.shape[0]
        starts, ends = [], []
        for b in range(B):
            ids = input_ids[b]
            bos_positions = (ids == audio_start_id).nonzero(as_tuple=True)[0]
            eos_positions = (ids == audio_end_id).nonzero(as_tuple=True)[0]
            if len(bos_positions) == 0 or len(eos_positions) == 0:
                raise ValueError(f"Sample {b} has no audio tokens")
            starts.append(bos_positions[0].item() + 1)
            ends.append(eos_positions[0].item())
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
        text_config = self.get_text_config(model)
        orig = text_config._attn_implementation
        text_config._attn_implementation = "eager"
        return {"orig_attn_impl": orig}

    def restore_after_inference(self, model: PreTrainedModel, state: dict) -> None:
        text_config = self.get_text_config(model)
        text_config._attn_implementation = state["orig_attn_impl"]

    def generate(self, model: PreTrainedModel, inputs: dict, **kwargs) -> Tensor:
        kwargs.setdefault("generation_mode", "text")
        if "max_new_tokens" in kwargs:
            kwargs.setdefault("thinker_max_new_tokens", kwargs.pop("max_new_tokens"))
        return model.generate(**inputs, **kwargs)

    def get_logits(self, model: PreTrainedModel, outputs) -> Tensor:
        if hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return model.thinker.lm_head(hidden)

    @property
    def token_rate(self) -> float:
        return 25.0

    def get_text_model(self, model: PreTrainedModel) -> PreTrainedModel:
        return model.thinker

    def get_audio_token_id(self, model: PreTrainedModel) -> int:
        cfg = model.config.thinker_config
        token_id = getattr(cfg, "audio_token_index", None)
        if token_id is None:
            token_id = cfg.audio_token_id
        return token_id

    def build_analysis_inputs(
        self,
        model: PreTrainedModel,
        processor,
        audio_path: str,
        question: str,
        answer: str,
        task: dict | None = None,
    ) -> dict[str, Tensor]:
        import soundfile as sf

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

        waveform, file_sr = sf.read(audio_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if file_sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=16000)

        conversation = [
            [{"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question},
            ]}]
        ]

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
            **self.chat_template_kwargs,
        )
        prefix = _to_device(processor(
            text=text, audio=[waveform], return_tensors="pt",
            padding=True, sampling_rate=16000,
        ))

        prefix_input_ids = prefix["input_ids"]
        input_features = prefix["input_features"]
        feature_attention_mask = prefix["feature_attention_mask"]
        prefix_len = prefix_input_ids.shape[1]

        answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        answer_ids.append(processor.tokenizer.eos_token_id)
        answer_tensor = torch.tensor([answer_ids], device=device)
        full_input_ids = torch.cat([prefix_input_ids, answer_tensor], dim=1)

        inputs_embeds = model.thinker.get_input_embeddings()(full_input_ids)
        audio_embeds = model.thinker.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            return_dict=True,
        ).last_hidden_state
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
