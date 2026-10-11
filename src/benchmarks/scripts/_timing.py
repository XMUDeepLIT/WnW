
from __future__ import annotations

import torch
from transformers import StoppingCriteria


class StepTimer(StoppingCriteria):

    def __init__(self, device: torch.device | str | None = None):
        self.device = device
        self._events: list[torch.cuda.Event] = []
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.append(event)
        return False

    def result(self) -> dict:
        """Synchronize once and return prefill + per-decode-step times in seconds."""
        if not self._events:
            return {"prefill": None, "decode": []}
        torch.cuda.synchronize(self.device)
        marks = [self._start] + self._events
        secs = [marks[i - 1].elapsed_time(marks[i]) / 1000.0 for i in range(1, len(marks))]
        return {"prefill": round(secs[0], 4), "decode": [round(s, 4) for s in secs[1:]]}
