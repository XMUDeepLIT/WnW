
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import torch

from common.monitoring.gpu_monitor import GPUMonitor


class InferenceStats:
    """Records cache state, GPU memory, and CPU<->GPU transfer events per sample.

    Designed to be composed into a cache backend. Default interval is 5 decode steps.
    """

    def __init__(self, device: torch.device, interval: int = 5):
        self.device = device
        self.interval = interval
        self.enabled = interval > 0
        self._gpu_monitor = GPUMonitor(device.index or 0)

        self._batch_size: int = 0
        self._prefill_before_stats: list[dict] = []
        self._prefill_stats: list[dict] = []
        self._decode_snapshots: list[list[dict]] = []
        self._cpu_gpu_events: dict[int, list[tuple]] = defaultdict(list)

        self._step_c2g_tokens: dict[int, int] = defaultdict(int)
        self._step_g2c_tokens: dict[int, int] = defaultdict(int)
        self._step_c2g_bytes: dict[int, int] = defaultdict(int)
        self._step_g2c_bytes: dict[int, int] = defaultdict(int)
        self._prefill_before_recorded = False

    def set_interval(self, interval: int) -> None:
        """Change interval at runtime. 0 disables collection."""
        self.interval = interval
        self.enabled = interval > 0

    def reset(self, batch_size: int) -> None:
        """Reset all buffers for a new sample/batch."""
        self._batch_size = batch_size
        self._prefill_before_stats = []
        self._prefill_stats = []
        self._decode_snapshots = []
        self._cpu_gpu_events = defaultdict(list)

        self._step_c2g_tokens = defaultdict(int)
        self._step_g2c_tokens = defaultdict(int)
        self._step_c2g_bytes = defaultdict(int)
        self._step_g2c_bytes = defaultdict(int)
        self._prefill_before_recorded = False

    def record_prefill_before(self, cache_stats_fn: Callable[..., list[dict]]) -> None:
        """Record one snapshot before prefill starts (guarded to run only once)."""
        if not self.enabled or self._prefill_before_recorded:
            return
        stats = cache_stats_fn(step=-1)
        gpu = self._gpu_monitor.snapshot()
        self._prefill_before_stats = [{**s, **gpu} for s in stats]
        self._prefill_before_recorded = True

    def record_prefill(self, cache_stats_fn: Callable[..., list[dict]]) -> None:
        """Record snapshot after prefill completes and flush prefill-stage transfers."""
        if not self.enabled:
            return
        stats = cache_stats_fn(step=-1)
        gpu = self._gpu_monitor.snapshot()
        self._prefill_stats = [{**s, **gpu} for s in stats]
        self.flush_decode_step(-1)

    def record_decode(self, step: int, cache_stats_fn: Callable[..., list[dict]]) -> None:
        """Record a decode snapshot at the configured interval."""
        if not self.enabled:
            return
        stats = cache_stats_fn(step=step)
        gpu = self._gpu_monitor.snapshot()
        self._decode_snapshots.append([{**s, **gpu} for s in stats])

    def record_cpu_gpu_transfer(
        self,
        b: int,
        c2g_tokens: int = 0,
        g2c_tokens: int = 0,
        c2g_bytes: int = 0,
        g2c_bytes: int = 0,
    ) -> None:
        """Accumulate a CPU<->GPU transfer for the current decode step."""
        if not self.enabled:
            return
        self._step_c2g_tokens[b] += int(c2g_tokens)
        self._step_g2c_tokens[b] += int(g2c_tokens)
        self._step_c2g_bytes[b] += int(c2g_bytes)
        self._step_g2c_bytes[b] += int(g2c_bytes)

    def flush_decode_step(self, step: int) -> None:
        """Persist accumulated transfer counters for the current step.

        Skips samples with no activity so non-swap algorithms don't accumulate
        long lists of zero-only entries.
        """
        if not self.enabled:
            return
        touched = set(self._step_c2g_tokens) | set(self._step_g2c_tokens) \
                | set(self._step_c2g_bytes) | set(self._step_g2c_bytes)
        for b in sorted(touched):
            c2g_t = self._step_c2g_tokens[b]
            g2c_t = self._step_g2c_tokens[b]
            c2g_b = self._step_c2g_bytes[b]
            g2c_b = self._step_g2c_bytes[b]
            if c2g_t == 0 and g2c_t == 0 and c2g_b == 0 and g2c_b == 0:
                continue
            self._cpu_gpu_events[b].append((
                int(step), int(c2g_t), int(g2c_t), int(c2g_b), int(g2c_b),
            ))
        self._step_c2g_tokens.clear()
        self._step_g2c_tokens.clear()
        self._step_c2g_bytes.clear()
        self._step_g2c_bytes.clear()

    def get_all_stats(self) -> list[dict]:
        """Return per-sample stats dict after generation finishes."""
        results = []
        for b in range(self._batch_size):
            prefill_before = self._prefill_before_stats[b] if b < len(self._prefill_before_stats) else {}
            prefill = self._prefill_stats[b] if b < len(self._prefill_stats) else {}
            snapshots = [s[b] for s in self._decode_snapshots if b < len(s)]
            final = snapshots[-1] if snapshots else prefill
            results.append({
                "prefill_before": prefill_before,
                "prefill": prefill,
                "decode_snapshots": snapshots,
                "decode_final": final,
                "cpu_gpu_events": self._cpu_gpu_events.get(b, []),
            })
        return results
