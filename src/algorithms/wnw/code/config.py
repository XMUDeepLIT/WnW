import json
import warnings
from dataclasses import dataclass, field


@dataclass
class WnWConfig:
    chunk_size_s: float
    overlap_s: float
    token_rate: float

    class_a: list[int]
    class_b: list[int]
    retention_ratios: list[float | None]

    top_k_chunks: int
    eviction_lag: int

    prefill_score_mode: str

    prefill_global_topk: bool = False

    num_layers: int = 0
    num_kv_heads: int = 0
    num_q_heads: int = 0

    _class_a_set: set[int] = field(default_factory=set, repr=False)
    _class_b_set: set[int] = field(default_factory=set, repr=False)
    _class_a_kv_set: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self._class_a_set = set(self.class_a)
        self._class_b_set = set(self.class_b)
        assert self.stride_tokens > 0, (
            f"stride must be positive: chunk_size_tokens={self.chunk_size_tokens} "
            f"<= overlap_tokens={self.overlap_tokens}"
        )
        assert self.chunk_size_tokens % self.stride_tokens == 0, (
            f"chunk_size_tokens ({self.chunk_size_tokens}) must be an integer "
            f"multiple of stride_tokens ({self.stride_tokens}) so segments tile "
            f"each chunk exactly"
        )
        if self.num_q_heads > 0 and self.num_kv_heads > 0:
            self._rebuild_class_a_kv_set()

    def set_model_dims(self, num_layers: int, num_kv_heads: int, num_q_heads: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self._rebuild_class_a_kv_set()

    def _rebuild_class_a_kv_set(self):
        self._class_a_kv_set = set()
        num_kv_groups = self.num_q_heads // self.num_kv_heads
        for gid in self.class_a:
            layer_idx = gid // self.num_q_heads
            local_q = gid % self.num_q_heads
            local_kv = local_q // num_kv_groups
            self._class_a_kv_set.add(layer_idx * self.num_kv_heads + local_kv)

    @property
    def chunk_size_tokens(self) -> int:
        return int(self.chunk_size_s * self.token_rate)

    @property
    def overlap_tokens(self) -> int:
        return int(self.overlap_s * self.token_rate)

    @property
    def stride_tokens(self) -> int:
        return self.chunk_size_tokens - self.overlap_tokens

    def get_layer_class_a(self, layer_idx: int) -> list[int]:
        base = layer_idx * self.num_q_heads
        return [gid - base for gid in self.class_a if base <= gid < base + self.num_q_heads]

    def get_layer_class_b(self, layer_idx: int) -> list[int]:
        base = layer_idx * self.num_kv_heads
        return [gid - base for gid in self.class_b if base <= gid < base + self.num_kv_heads]

    def get_retention(self, layer_idx: int, local_kv_head: int) -> float | None:
        return self.retention_ratios[layer_idx * self.num_kv_heads + local_kv_head]

    def is_class_a(self, layer_idx: int, q_head: int) -> bool:
        return layer_idx * self.num_q_heads + q_head in self._class_a_set

    def is_class_a_kv_head(self, layer_idx: int, kv_head: int) -> bool:
        return layer_idx * self.num_kv_heads + kv_head in self._class_a_kv_set

    def is_class_b(self, layer_idx: int, kv_head: int) -> bool:
        return layer_idx * self.num_kv_heads + kv_head in self._class_b_set

    @classmethod
    def from_json(cls, path: str, num_layers: int = 0, num_kv_heads: int = 0,
                  num_q_heads: int = 0) -> "WnWConfig":
        with open(path) as f:
            raw = json.load(f)

        chunk = raw["chunk"]
        heads = raw["heads"]
        selection = raw["selection"]
        score = raw.get("score", {})

        prefill_score_mode = score.get("prefill_mode", "mean_text")
        assert prefill_score_mode in ("last", "mean_text")

        return cls(
            chunk_size_s=chunk["size_s"],
            overlap_s=chunk["overlap_s"],
            token_rate=chunk["token_rate"],
            class_a=heads["class_a"],
            class_b=heads["class_b"],
            retention_ratios=heads["retention_ratios"],
            top_k_chunks=selection["top_k_chunks"],
            eviction_lag=selection["eviction_lag"],
            prefill_score_mode=prefill_score_mode,
            prefill_global_topk=score.get("prefill_global_topk", False),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
        )
