
import torch


class GPUMonitor:
    """Snapshot GPU memory for a single CUDA device."""

    def __init__(self, gpu_index: int):
        self.gpu_index = gpu_index

    def snapshot(self) -> dict:
        """Return current GPU memory state.

        Falls back to None values if torch.cuda.mem_get_info fails (e.g. wrong
        index or non-CUDA device).
        """
        try:
            free, total = torch.cuda.mem_get_info(self.gpu_index)
        except Exception:
            return {
                "gpu_index": self.gpu_index,
                "memory_total_bytes": None,
                "memory_free_bytes": None,
                "memory_used_bytes": None,
            }
        return {
            "gpu_index": self.gpu_index,
            "memory_total_bytes": int(total),
            "memory_free_bytes": int(free),
            "memory_used_bytes": int(total - free),
        }
