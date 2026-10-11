
from common.adapters.base import BaseModelAdapter


class AnalysisInputBuilder:
    def __init__(self, model, processor, adapter: BaseModelAdapter):
        self.model = model
        self.processor = processor
        self.adapter = adapter

    def build(self, audio_path: str, question: str, answer: str, task: dict | None = None) -> dict:
        return self.adapter.build_analysis_inputs(
            self.model, self.processor, audio_path, question, answer, task
        )
