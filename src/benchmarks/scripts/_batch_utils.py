
import torch
from torch import Tensor


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    items = list(iterable)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    parts = []
    for ch in text:
        cp = ord(ch)
        if (
            0x3040 <= cp <= 0x30FF
            or 0x3400 <= cp <= 0x4DBF
            or 0x4E00 <= cp <= 0x9FFF
            or 0xAC00 <= cp <= 0xD7AF
            or 0xF900 <= cp <= 0xFAFF
            or 0x20000 <= cp <= 0x2FFFF
        ):
            parts.append(" ")
            parts.append(ch)
            parts.append(" ")
        else:
            parts.append(ch)
    return len("".join(parts).split())


def sort_by_answer_length(samples, answer_key: str = "answer"):
    """Sort samples by approximate token count of their ground-truth answer.

    Grouping similar-length answers into the same batch keeps decode steps
    within a batch synchronized, minimizing wasted compute on short samples
    waiting for long ones.
    """
    return sorted(samples, key=lambda s: _approx_token_count(s.get(answer_key, "") or ""))


def collate_inputs(
    input_list: list[dict[str, Tensor]],
    pad_token_id: int,
    feature_time_dim: int = -1,
) -> tuple[dict[str, Tensor], list[int]]:
    """Left-pad and batch a list of single-sample input dicts.

    Each element in input_list is from adapter.build_inputs() with tensors of
    shape [1, S_i, ...]. Returns a batched dict with [B, S_max, ...] tensors
    and a list of original (unpadded) sequence lengths.
    """
    B = len(input_list)
    device = input_list[0]["input_ids"].device

    seq_lens = [inp["input_ids"].shape[1] for inp in input_list]
    max_len = max(seq_lens)

    input_ids_list = []
    attention_mask_list = []

    for inp, slen in zip(input_list, seq_lens):
        pad_len = max_len - slen
        ids = inp["input_ids"][0]
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=device)
            ids = torch.cat([pad, ids])
        input_ids_list.append(ids)

        if "attention_mask" in inp:
            mask = inp["attention_mask"][0]
        else:
            mask = torch.ones(slen, dtype=torch.long, device=device)
        if pad_len > 0:
            pad_mask = torch.zeros(pad_len, dtype=mask.dtype, device=device)
            mask = torch.cat([pad_mask, mask])
        attention_mask_list.append(mask)

    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
    }

    if "input_features" in input_list[0]:
        features = [inp["input_features"] for inp in input_list]
        feat_shape = features[0].shape
        ndim = len(feat_shape)
        if ndim >= 3:
            dim = feature_time_dim % ndim
            time_sizes = [f.shape[dim] for f in features]
            if len(set(time_sizes)) > 1:
                max_t = max(time_sizes)
                padded = []
                for f in features:
                    pad_t = max_t - f.shape[dim]
                    if pad_t > 0:
                        n_pad_pairs = ndim - 1 - dim
                        pad_args = [0, 0] * n_pad_pairs + [0, pad_t]
                        f = torch.nn.functional.pad(f, pad_args)
                    padded.append(f)
                result["input_features"] = torch.cat(padded, dim=0)
            else:
                result["input_features"] = torch.cat(features, dim=0)
        else:
            result["input_features"] = torch.cat(features, dim=0)

    if "feature_attention_mask" in input_list[0]:
        feat_masks = [inp["feature_attention_mask"] for inp in input_list]
        feat_lens = [fm.shape[-1] for fm in feat_masks]
        if len(set(feat_lens)) > 1:
            max_fl = max(feat_lens)
            padded = []
            for fm in feat_masks:
                pad_fl = max_fl - fm.shape[-1]
                if pad_fl > 0:
                    fm = torch.nn.functional.pad(fm, (0, pad_fl), value=0)
                padded.append(fm)
            result["feature_attention_mask"] = torch.cat(padded, dim=0)
        else:
            result["feature_attention_mask"] = torch.cat(feat_masks, dim=0)

    return result, seq_lens
