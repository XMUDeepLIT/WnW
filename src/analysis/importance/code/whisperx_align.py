
import gc

import torch
import whisperx
from whisperx.audio import SAMPLE_RATE


class WhisperXAligner:
    def __init__(self, device: str = "cuda", compute_type: str = "float16"):
        self.device = device
        self.compute_type = compute_type
        self._align_model = None
        self._metadata = None

    def _load_audio(self, audio_path: str):
        return whisperx.load_audio(audio_path)

    def _get_audio_duration(self, audio) -> float:
        import numpy as np
        audio_np = np.array(audio) if not isinstance(audio, list) else audio
        num_samples = audio_np.shape[0] if audio_np.ndim == 1 else audio_np.shape[1]
        return num_samples / SAMPLE_RATE

    def _ensure_align_model(self, language_code: str = "en"):
        if self._align_model is None:
            self._align_model, self._metadata = whisperx.load_align_model(
                language_code=language_code, device=self.device,
            )

    def align(self, audio_path: str, text: str, language: str = "en") -> dict:
        audio = self._load_audio(audio_path)
        duration = self._get_audio_duration(audio)
        transcript = [{"start": 0.0, "end": duration, "text": text}]

        self._ensure_align_model(language)
        result = whisperx.align(
            transcript=transcript,
            model=self._align_model,
            align_model_metadata=self._metadata,
            audio=audio,
            device=self.device,
            return_char_alignments=False,
        )
        return {"word_segments": result.get("word_segments", []), "duration": duration}

    def cleanup(self):
        if self._align_model is not None:
            del self._align_model
            self._align_model = None
            self._metadata = None
            torch.cuda.empty_cache()
            gc.collect()
