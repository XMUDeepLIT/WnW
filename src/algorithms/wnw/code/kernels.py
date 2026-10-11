
import torch
import triton
import triton.language as tl


@triton.jit
def _varlen_q1_softmax_kernel(
    Q_ptr,
    K_ptr,
    Scores_ptr,
    start_ptr,
    end_ptr,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    start = tl.load(start_ptr + pid)
    end = tl.load(end_ptr + pid)
    Nk = end - start

    offs_d = tl.arange(0, d)
    q = tl.load(Q_ptr + pid * d + offs_d)
    scale = 1.0 / tl.sqrt(tl.cast(d, tl.float32))

    max_score = tl.full([], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([], dtype=tl.float32)

    for block_start in range(0, Nk, BLOCK_K):
        offs_k = block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Nk
        k = tl.load(
            K_ptr + (start + offs_k[:, None]) * d + offs_d[None, :],
            mask=mask_k[:, None], other=0.0,
        )
        raw = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(mask_k, raw, float('-inf'))

        m_new = tl.maximum(max_score, tl.max(scores))
        sum_exp = sum_exp * tl.exp(max_score - m_new) + tl.sum(tl.where(mask_k, tl.exp(scores - m_new), 0.0))
        max_score = m_new

    inv_sum = 1.0 / sum_exp
    for block_start in range(0, Nk, BLOCK_K):
        offs_k = block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Nk
        k = tl.load(
            K_ptr + (start + offs_k[:, None]) * d + offs_d[None, :],
            mask=mask_k[:, None], other=0.0,
        )
        raw = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(mask_k, raw, float('-inf'))
        out = tl.exp(scores - max_score) * inv_sum
        tl.store(Scores_ptr + start + offs_k, out, mask=mask_k)


def varlen_q1_softmax(
    Q: torch.Tensor,
    K: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Softmax each query against its variable-length K slice; returns scores [total_kv]."""
    B, d = Q.shape
    total_kv = K.shape[0]
    scores = torch.empty(total_kv, dtype=torch.float32, device=Q.device)

    BLOCK_K = 256 if d <= 128 else 128

    _varlen_q1_softmax_kernel[(B,)](
        Q, K, scores, starts, ends,
        d=d, BLOCK_K=BLOCK_K,
    )
    return scores


@triton.jit
def _score_accumulate_kernel(
    Scores_ptr,
    Accum_ptr,
    src_offsets_ptr,
    dst_offsets_ptr,
    num_heads: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_start = tl.load(dst_offsets_ptr + pid)
    dst_end = tl.load(dst_offsets_ptr + pid + 1)
    n_tokens = dst_end - dst_start

    for block_start in range(0, n_tokens, BLOCK):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < n_tokens

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for h in range(num_heads):
            src_start = tl.load(src_offsets_ptr + pid * num_heads + h)
            vals = tl.load(Scores_ptr + src_start + offs, mask=mask, other=0.0)
            acc += vals

        prev = tl.load(Accum_ptr + dst_start + offs, mask=mask, other=0.0)
        tl.store(Accum_ptr + dst_start + offs, prev + acc, mask=mask)


def score_accumulate(
    scores: torch.Tensor,
    accumulator: torch.Tensor,
    src_offsets: torch.Tensor,
    dst_offsets: torch.Tensor,
    num_heads: int,
):
    """Accumulate per-head scores into a per-token accumulator."""
    num_segments = dst_offsets.shape[0] - 1
    BLOCK = 256

    _score_accumulate_kernel[(num_segments,)](
        scores, accumulator,
        src_offsets, dst_offsets,
        num_heads=num_heads, BLOCK=BLOCK,
    )




@triton.jit
def _fused_score_kernel_h1(
    Q_ptr,
    K_ptr,
    Accum_ptr,
    seg_start_ptr,
    seg_end_ptr,
    audio_start_ptr,
    audio_end_ptr,
    accum_off_ptr,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    seg_s = tl.load(seg_start_ptr + pid)
    seg_e = tl.load(seg_end_ptr + pid)
    Nk = seg_e - seg_s

    audio_s = tl.load(audio_start_ptr + pid)
    audio_e = tl.load(audio_end_ptr + pid)
    accum_off = tl.load(accum_off_ptr + pid)

    offs_d = tl.arange(0, d)
    scale = 1.0 / tl.sqrt(tl.cast(d, tl.float32))

    q0 = tl.load(Q_ptr + pid * d + offs_d)

    max0 = tl.full([], float('-inf'), dtype=tl.float32)
    sum0 = tl.zeros([], dtype=tl.float32)

    for block_start in range(0, Nk, BLOCK_K):
        offs_k = block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Nk
        k = tl.load(
            K_ptr + (seg_s + offs_k[:, None]) * d + offs_d[None, :],
            mask=mask_k[:, None], other=0.0,
        )
        raw0 = tl.sum(k * q0[None, :], axis=1) * scale
        s0 = tl.where(mask_k, raw0, float('-inf'))

        m_new0 = tl.maximum(max0, tl.max(s0))
        sum0 = sum0 * tl.exp(max0 - m_new0) + tl.sum(tl.where(mask_k, tl.exp(s0 - m_new0), 0.0))
        max0 = m_new0

    inv0 = 1.0 / sum0

    Na = audio_e - audio_s
    for block_start in range(0, Na, BLOCK_K):
        offs_k = block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Na
        k = tl.load(
            K_ptr + (seg_s + audio_s + offs_k[:, None]) * d + offs_d[None, :],
            mask=mask_k[:, None], other=0.0,
        )
        raw0 = tl.sum(k * q0[None, :], axis=1) * scale
        s0 = tl.where(mask_k, raw0, float('-inf'))
        out0 = tl.exp(s0 - max0) * inv0

        total = out0

        prev = tl.load(Accum_ptr + accum_off + offs_k, mask=mask_k, other=0.0)
        tl.store(Accum_ptr + accum_off + offs_k, prev + total, mask=mask_k)


@triton.jit
def _fused_score_kernel_h4(
    Q_ptr,
    K_ptr,
    Accum_ptr,
    seg_start_ptr,
    seg_end_ptr,
    audio_start_ptr,
    audio_end_ptr,
    accum_off_ptr,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    seg_s = tl.load(seg_start_ptr + pid)
    seg_e = tl.load(seg_end_ptr + pid)
    Nk = seg_e - seg_s

    audio_s = tl.load(audio_start_ptr + pid)
    audio_e = tl.load(audio_end_ptr + pid)
    accum_off = tl.load(accum_off_ptr + pid)

    offs_d = tl.arange(0, d)
    scale = 1.0 / tl.sqrt(tl.cast(d, tl.float32))

    q_base = pid * 4 * d
    q0 = tl.load(Q_ptr + q_base + offs_d)
    q1 = tl.load(Q_ptr + q_base + d + offs_d)
    q2 = tl.load(Q_ptr + q_base + 2 * d + offs_d)
    q3 = tl.load(Q_ptr + q_base + 3 * d + offs_d)

    max0 = tl.full([], float('-inf'), dtype=tl.float32)
    max1 = tl.full([], float('-inf'), dtype=tl.float32)
    max2 = tl.full([], float('-inf'), dtype=tl.float32)
    max3 = tl.full([], float('-inf'), dtype=tl.float32)
    sum0 = tl.zeros([], dtype=tl.float32)
    sum1 = tl.zeros([], dtype=tl.float32)
    sum2 = tl.zeros([], dtype=tl.float32)
    sum3 = tl.zeros([], dtype=tl.float32)

    for block_start in range(0, Nk, BLOCK_K):
        offs_k = block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Nk
        k = tl.load(
            K_ptr + (seg_s + offs_k[:, None]) * d + offs_d[None, :],
            mask=mask_k[:, None], other=0.0,
        )

        raw0 = tl.sum(k * q0[None, :], axis=1) * scale
        raw1 = tl.sum(k * q1[None, :], axis=1) * scale
        raw2 = tl.sum(k * q2[None, :], axis=1) * scale
        raw3 = tl.sum(k * q3[None, :], axis=1) * scale

        s0 = tl.where(mask_k, raw0, float('-inf'))
        s1 = tl.where(mask_k, raw1, float('-inf'))
        s2 = tl.where(mask_k, raw2, float('-inf'))
        s3 = tl.where(mask_k, raw3, float('-inf'))

        m0 = tl.maximum(max0, tl.max(s0))
        sum0 = sum0 * tl.exp(max0 - m0) + tl.sum(tl.where(mask_k, tl.exp(s0 - m0), 0.0))
        max0 = m0

        m1 = tl.maximum(max1, tl.max(s1))
        sum1 = sum1 * tl.exp(max1 - m1) + tl.sum(tl.where(mask_k, tl.exp(s1 - m1), 0.0))
        max1 = m1

        m2 = tl.maximum(max2, tl.max(s2))
        sum2 = sum2 * tl.exp(max2 - m2) + tl.sum(tl.where(mask_k, tl.exp(s2 - m2), 0.0))
        max2 = m2

        m3 = tl.maximum(max3, tl.max(s3))
        sum3 = sum3 * tl.exp(max3 - m3) + tl.sum(tl.where(mask_k, tl.exp(s3 - m3), 0.0))
        max3 = m3

    inv0 = 1.0 / sum0
    inv1 = 1.0 / sum1
    inv2 = 1.0 / sum2
    inv3 = 1.0 / sum3

    Na = audio_e - audio_s
    for block_start in range(0, Na, BLOCK_K):
        offs_k = block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Na
        k = tl.load(
            K_ptr + (seg_s + audio_s + offs_k[:, None]) * d + offs_d[None, :],
            mask=mask_k[:, None], other=0.0,
        )

        raw0 = tl.sum(k * q0[None, :], axis=1) * scale
        raw1 = tl.sum(k * q1[None, :], axis=1) * scale
        raw2 = tl.sum(k * q2[None, :], axis=1) * scale
        raw3 = tl.sum(k * q3[None, :], axis=1) * scale

        s0 = tl.where(mask_k, raw0, float('-inf'))
        s1 = tl.where(mask_k, raw1, float('-inf'))
        s2 = tl.where(mask_k, raw2, float('-inf'))
        s3 = tl.where(mask_k, raw3, float('-inf'))

        out0 = tl.exp(s0 - max0) * inv0
        out1 = tl.exp(s1 - max1) * inv1
        out2 = tl.exp(s2 - max2) * inv2
        out3 = tl.exp(s3 - max3) * inv3

        total = out0 + out1 + out2 + out3

        prev = tl.load(Accum_ptr + accum_off + offs_k, mask=mask_k, other=0.0)
        tl.store(Accum_ptr + accum_off + offs_k, prev + total, mask=mask_k)




def fused_score_accumulate(
    Q: torch.Tensor,
    K: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    audio_starts: torch.Tensor,
    audio_ends: torch.Tensor,
    accumulator: torch.Tensor,
    accum_offsets: torch.Tensor,
):
    """Fused kernel: full-segment softmax + multi-head sum into a per-token accumulator.

    For each sample and head, softmax(Q @ K) is computed over the full segment,
    then the audio-range scores are summed across heads and added to the accumulator.

    Args:
        Q: [num_samples, num_heads, d] Class A query heads per sample.
        K: [total_kv, d] flat K cache for one layer.
        seg_starts: [num_samples] segment start in K.
        seg_ends: [num_samples] segment end in K.
        audio_starts: [num_samples] audio-range start offset within the segment.
        audio_ends: [num_samples] audio-range end offset within the segment.
        accumulator: [total_audio] global score accumulator (updated in place).
        accum_offsets: [num_samples + 1] per-sample offsets into the accumulator.
    """
    num_samples, num_heads, d = Q.shape
    BLOCK_K = 256 if d <= 128 else 128

    Q_flat = Q.contiguous()

    if num_heads == 4:
        _fused_score_kernel_h4[(num_samples,)](
            Q_flat, K, accumulator,
            seg_starts, seg_ends,
            audio_starts, audio_ends, accum_offsets,
            d=d, BLOCK_K=BLOCK_K,
        )
    elif num_heads == 1:
        _fused_score_kernel_h1[(num_samples,)](
            Q_flat, K, accumulator,
            seg_starts, seg_ends,
            audio_starts, audio_ends, accum_offsets,
            d=d, BLOCK_K=BLOCK_K,
        )
    else:
        _fused_score_pytorch(Q_flat, K, seg_starts, seg_ends,
                             audio_starts, audio_ends, accumulator, accum_offsets)


def _fused_score_pytorch(Q, K, seg_starts, seg_ends, audio_starts, audio_ends,
                         accumulator, accum_offsets):
    num_samples, num_heads, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    for s in range(num_samples):
        seg_s = seg_starts[s].item()
        seg_e = seg_ends[s].item()
        a_s = audio_starts[s].item()
        a_e = audio_ends[s].item()
        acc_off = accum_offsets[s].item()

        K_seg = K[seg_s:seg_e]

        for h in range(num_heads):
            q = Q[s, h]
            logits = (K_seg @ q) * scale
            scores = torch.softmax(logits.float(), dim=0)
            audio_scores = scores[a_s:a_e]
            accumulator[acc_off:acc_off + (a_e - a_s)] += audio_scores




def rebuild_append_kv(
    old_k: torch.Tensor,
    old_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    load_k: torch.Tensor,
    load_v: torch.Tensor,
    old_cu: torch.Tensor,
    new_cu: torch.Tensor,
    load_starts: torch.Tensor,
    load_lens: torch.Tensor,
    evict_offsets: torch.Tensor,
    evict_starts: torch.Tensor,
    evict_ends: torch.Tensor,
    max_evict: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    N = new_k.shape[0]
    D = old_k.shape[1]
    total_out = int(new_cu[-1].item())
    out_k = torch.empty(total_out, D, dtype=old_k.dtype, device=old_k.device)
    out_v = torch.empty(total_out, D, dtype=old_v.dtype, device=old_v.device)
    BLOCK_D = triton.next_power_of_2(D)

    _rebuild_append_kv_kernel[(total_out,)](
        old_k, old_v, new_k, new_v, load_k, load_v, out_k, out_v,
        old_cu, new_cu, load_starts, load_lens,
        evict_offsets, evict_starts, evict_ends,
        N=N, D=D, BLOCK_D=BLOCK_D, MAX_EVICT=max(max_evict, 1),
    )
    return out_k, out_v


@triton.jit
def _rebuild_append_kv_kernel(
    OldK, OldV,
    NewK, NewV,
    LoadK, LoadV,
    OutK, OutV,
    OldCu, NewCu,
    LoadStarts, LoadLens,
    EvictOffsets, EvictStarts, EvictEnds,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_EVICT: tl.constexpr,
):
    row_id = tl.program_id(0)

    lo = 0
    hi = N
    for _ in range(10):
        mid = (lo + hi) // 2
        val = tl.load(NewCu + mid + 1)
        lo = tl.where(val <= row_id, mid + 1, lo)
        hi = tl.where(val <= row_id, hi, mid)
    seg_id = lo

    new_start = tl.load(NewCu + seg_id)
    old_start = tl.load(OldCu + seg_id)
    old_end = tl.load(OldCu + seg_id + 1)
    old_len = old_end - old_start
    row_in_new = row_id - new_start

    evict_start_idx = tl.load(EvictOffsets + seg_id)
    evict_end_idx = tl.load(EvictOffsets + seg_id + 1)
    num_evict = evict_end_idx - evict_start_idx
    kept_len = old_len
    for i in range(0, MAX_EVICT):
        active = i < num_evict
        es = tl.load(EvictStarts + evict_start_idx + i, mask=active, other=0)
        ee = tl.load(EvictEnds + evict_start_idx + i, mask=active, other=0)
        kept_len -= tl.where(active, ee - es, 0)

    load_start = tl.load(LoadStarts + seg_id)
    load_len = tl.load(LoadLens + seg_id)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    if row_in_new < kept_len:
        old_rel = row_in_new
        for i in range(0, MAX_EVICT):
            active = i < num_evict
            es = tl.load(EvictStarts + evict_start_idx + i, mask=active, other=0)
            ee = tl.load(EvictEnds + evict_start_idx + i, mask=active, other=0)
            old_rel += tl.where(active & (old_rel >= es), ee - es, 0)
        src = old_start + old_rel
        vals_k = tl.load(OldK + src * D + offs_d, mask=mask_d)
        vals_v = tl.load(OldV + src * D + offs_d, mask=mask_d)
    elif row_in_new < kept_len + load_len:
        src = load_start + (row_in_new - kept_len)
        vals_k = tl.load(LoadK + src * D + offs_d, mask=mask_d)
        vals_v = tl.load(LoadV + src * D + offs_d, mask=mask_d)
    else:
        vals_k = tl.load(NewK + seg_id * D + offs_d, mask=mask_d)
        vals_v = tl.load(NewV + seg_id * D + offs_d, mask=mask_d)

    tl.store(OutK + row_id * D + offs_d, vals_k, mask=mask_d)
    tl.store(OutV + row_id * D + offs_d, vals_v, mask=mask_d)
