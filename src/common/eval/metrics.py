
from __future__ import annotations

import json
import re

import jiwer


def load_jsonl(path: str) -> dict[str, dict]:
    data = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data


def standardize_text(text: str) -> str:
    """Lowercase and strip punctuation, keeping only [a-z0-9], space, ' and ."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.]", "", text)
    return text


def truncate_hyp(ref: str, hyp: str, ratio: float, tokenizer=None) -> str:
    """Truncate hyp to len(ref) * ratio tokens. No-op without a tokenizer."""
    if ratio <= 0 or tokenizer is None:
        return hyp
    ref_tokens = tokenizer.encode(ref, add_special_tokens=False)
    if not ref_tokens:
        return hyp
    max_tokens = int(len(ref_tokens) * ratio)
    hyp_tokens = tokenizer.encode(hyp, add_special_tokens=False)
    return tokenizer.decode(hyp_tokens[:max_tokens], skip_special_tokens=True)


def load_tokenizer(name_or_path: str | None):
    """Load an AutoTokenizer, or return None when no name/path is given."""
    if not name_or_path:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)


def align_pairs(
    output_path: str,
    standardize: bool = True,
    truncate_ratio: float = 1.2,
    tokenizer=None,
) -> list[tuple[str, str, str]]:
    """Return [(id, ref, hyp)] from an eval output JSONL.

    The eval drivers write one record per sample as {id, hyp, ref, ...}, so ref
    and hyp both come from output_path. Pairs where both ref and hyp are empty
    are dropped.
    """
    outputs = load_jsonl(output_path)

    pairs = []
    for sid, obj in sorted(outputs.items()):
        ref = obj.get("ref") or ""
        hyp = obj.get("hyp") or ""
        if standardize:
            ref = standardize_text(ref)
            hyp = standardize_text(hyp)
        if not ref and not hyp:
            continue
        hyp = truncate_hyp(ref, hyp, truncate_ratio, tokenizer)
        pairs.append((sid, ref, hyp))
    return pairs


def _accumulate_error_rate(pairs, process_fn, label, print_errors):
    total_s = total_d = total_i = total_n = 0
    for sid, ref, hyp in pairs:
        out = process_fn(ref, hyp)
        s, d, i = out.substitutions, out.deletions, out.insertions
        n = out.hits + s + d
        total_s += s
        total_d += d
        total_i += i
        total_n += n
        if print_errors:
            val = (s + d + i) / n if n > 0 else 0
            print(f"[{sid}] {label}={val:.4f} (S={s} D={d} I={i} N={n})")
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}")

    rate = (total_s + total_d + total_i) / total_n if total_n > 0 else 0
    return {
        "rate": rate,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "n": total_n,
        "samples": len(pairs),
    }


def compute_wer(pairs, print_errors: bool = False) -> dict:
    """Word Error Rate over aligned (id, ref, hyp) pairs."""
    return _accumulate_error_rate(pairs, jiwer.process_words, "WER", print_errors)


def compute_cer(pairs, print_errors: bool = False) -> dict:
    """Character Error Rate over aligned (id, ref, hyp) pairs."""
    return _accumulate_error_rate(pairs, jiwer.process_characters, "CER", print_errors)


def compute_bleu(pairs, print_errors: bool = False) -> dict:
    """Corpus BLEU over aligned (id, ref, hyp) pairs."""
    import sacrebleu

    preds = [hyp for _, _, hyp in pairs]
    refs = [ref for _, ref, _ in pairs]
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    if print_errors:
        for sid, ref, hyp in pairs:
            print(f"[{sid}]")
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}")
    return {"score": bleu.score, "detail": bleu.format(), "samples": len(pairs)}
