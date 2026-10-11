

def compute_chunk_params(chunk_size_s, overlap_s, token_rate):
    chunk_size = int(chunk_size_s * token_rate)
    stride = int((chunk_size_s - overlap_s) * token_rate)
    return chunk_size, stride


def compute_num_segments(num_audio, chunk_size, stride):
    if stride <= 0:
        return 1
    return (num_audio + stride - 1) // stride


def effective_ratio(r_i, seg_size):
    return max(1, int(seg_size * r_i))


def predict_retention(
    norm_scores,
    n_A, n_B, n_O,
    scale,
    num_audio_list,
    chunk_size, stride, seg_size,
    top_k_chunks, segs_per_chunk,
    total_kv,
    loadback_factor: float = 0.8,
):
    """Predict average GPU and GPU+CPU retention over a dataset.

    Args:
        norm_scores: list of n_B + n_O normalized scores (B first, then O).
        n_A, n_B, n_O: head counts per class.
        scale: retention scale factor.
        num_audio_list: list of int, number of audio tokens per sample.
        chunk_size, stride, seg_size: chunk parameters in tokens.
        top_k_chunks: number of top chunks loaded during decode.
        segs_per_chunk: segments per chunk.
        total_kv: total number of KV heads.
        loadback_factor: empirical damping on the CPU->GPU loadback L term,
            because real chunk selection is more correlated (adjacent) than
            random and thus loads fewer unique segments.

    Returns:
        (avg_gpu_retention, avg_gpucpu_retention)
    """
    r_i_list = [min(s * scale, 1.0) for s in norm_scores]

    gpu_sum = 0.0
    gpucpu_sum = 0.0
    count = 0

    for na in num_audio_list:
        n_segs = compute_num_segments(na, chunk_size, stride)

        r_B_eff = 0.0
        for i in range(n_B):
            r_B_eff += effective_ratio(r_i_list[i], seg_size) * n_segs / na

        r_O_eff = 0.0
        for i in range(n_B, n_B + n_O):
            r_O_eff += effective_ratio(r_i_list[i], seg_size) * n_segs / na

        k = min(top_k_chunks, n_segs)
        if n_segs > 0 and segs_per_chunk > 0:
            p_not_covered = ((n_segs - segs_per_chunk) / n_segs) ** k
            loaded_segs = n_segs * (1.0 - p_not_covered)
        else:
            loaded_segs = 0.0
        loaded_frac = loaded_segs / n_segs if n_segs > 0 else 0.0
        L = loadback_factor * loaded_frac * (n_B - r_B_eff)

        gpu_sum += (n_A + r_B_eff + r_O_eff + L) / total_kv
        gpucpu_sum += (n_A + n_B + r_O_eff + L) / total_kv
        count += 1

    if count == 0:
        return 0.0, 0.0
    return gpu_sum / count, gpucpu_sum / count
