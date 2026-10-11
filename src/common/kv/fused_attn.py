
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q1_decode_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, cu_ptr,
    scale,
    G: tl.constexpr, GPAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    seg = tl.program_id(0)
    d = tl.arange(0, D)
    gp = tl.arange(0, GPAD)
    valid_g = gp < G

    start = tl.load(cu_ptr + seg)
    end = tl.load(cu_ptr + seg + 1)

    q = tl.load(Q_ptr + seg * G * D + gp[:, None] * D + d[None, :],
                mask=valid_g[:, None], other=0.0).to(tl.float32)

    m = tl.full((GPAD,), -1e30, tl.float32)
    l = tl.zeros((GPAD,), tl.float32)
    acc = tl.zeros((GPAD, D), tl.float32)
    for s in range(start, end, BLOCK_N):
        n = s + tl.arange(0, BLOCK_N)
        mask = n < end
        k = tl.load(K_ptr + n[:, None] * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_ptr + n[:, None] * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        qk = tl.where(mask[None, :] & valid_g[:, None], tl.dot(q, tl.trans(k)) * scale, -1e30)
        m2 = tl.maximum(m, tl.max(qk, axis=1))
        alpha = tl.exp(m - m2)
        p = tl.exp(qk - m2[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m = m2

    o = acc / l[:, None]
    tl.store(O_ptr + seg * G * D + gp[:, None] * D + d[None, :],
             o.to(O_ptr.dtype.element_ty), mask=valid_g[:, None])


def q1_decode_attn(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    cu_k: torch.Tensor,
    scale: float,
    block_n: int = 64,
) -> torch.Tensor:
    """Return attention output [nseg, G, D] (same dtype as q_flat)."""
    nseg, G, D = q_flat.shape
    q_flat = q_flat.contiguous()
    GPAD = max(16, triton.next_power_of_2(G))
    out = torch.empty(nseg, G, D, device=q_flat.device, dtype=q_flat.dtype)
    _q1_decode_attn_kernel[(nseg,)](
        q_flat, k_flat, v_flat, out, cu_k,
        scale,
        G=G, GPAD=GPAD, D=D, BLOCK_N=block_n,
    )
    return out


@triton.jit
def _q1_decode_attn_score_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, cu_ptr,
    ACC_ptr, ascore_off_ptr, audio_s_ptr, audio_e_ptr, ca_mask_ptr,
    num_kv_heads,
    scale,
    G: tl.constexpr, GPAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    seg = tl.program_id(0)
    d = tl.arange(0, D)
    gp = tl.arange(0, GPAD)
    valid_g = gp < G

    start = tl.load(cu_ptr + seg)
    end = tl.load(cu_ptr + seg + 1)

    q = tl.load(Q_ptr + seg * G * D + gp[:, None] * D + d[None, :],
                mask=valid_g[:, None], other=0.0).to(tl.float32)

    m = tl.full((GPAD,), -1e30, tl.float32)
    l = tl.zeros((GPAD,), tl.float32)
    acc = tl.zeros((GPAD, D), tl.float32)
    for s in range(start, end, BLOCK_N):
        n = s + tl.arange(0, BLOCK_N)
        mask = n < end
        k = tl.load(K_ptr + n[:, None] * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_ptr + n[:, None] * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        qk = tl.where(mask[None, :] & valid_g[:, None], tl.dot(q, tl.trans(k)) * scale, -1e30)
        m2 = tl.maximum(m, tl.max(qk, axis=1))
        alpha = tl.exp(m - m2)
        p = tl.exp(qk - m2[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m = m2

    o = acc / l[:, None]
    tl.store(O_ptr + seg * G * D + gp[:, None] * D + d[None, :],
             o.to(O_ptr.dtype.element_ty), mask=valid_g[:, None])

    kv = seg % num_kv_heads
    if tl.load(ca_mask_ptr + kv) != 0:
        b = seg // num_kv_heads
        a0 = tl.load(audio_s_ptr + b)
        a1 = tl.load(audio_e_ptr + b)
        acc_base = tl.load(ascore_off_ptr + b)
        for s in range(start + a0, start + a1, BLOCK_N):
            n = s + tl.arange(0, BLOCK_N)
            mask = n < (start + a1)
            k = tl.load(K_ptr + n[:, None] * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
            qk = tl.where(mask[None, :] & valid_g[:, None], tl.dot(q, tl.trans(k)) * scale, -1e30)
            w = tl.exp(qk - m[:, None]) / l[:, None]
            s_key = tl.sum(tl.where(valid_g[:, None], w, 0.0), axis=0)
            idx = acc_base + (n - start - a0)
            tl.atomic_add(ACC_ptr + idx, s_key, mask=mask)


def q1_decode_attn_score(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    cu_k: torch.Tensor,
    scale: float,
    ca_kv_mask: torch.Tensor,
    num_kv_heads: int,
    audio_starts: torch.Tensor,
    audio_ends: torch.Tensor,
    score_accumulator: torch.Tensor,
    score_offsets: torch.Tensor,
    block_n: int = 64,
) -> torch.Tensor:
    """Fused attention output + per-kv-head Class-A score accumulation.

    Every kv-head flagged in ca_kv_mask emits its group's per-audio-token score
    (sum over the group's G q-heads of the full-segment softmax weight) into
    score_accumulator[score_offsets[b]:], scoring each group against its OWN K
    segment. Multiple flagged kv-heads per layer accumulate via atomic_add.
    """
    nseg, G, D = q_flat.shape
    q_flat = q_flat.contiguous()
    GPAD = max(16, triton.next_power_of_2(G))
    out = torch.empty(nseg, G, D, device=q_flat.device, dtype=q_flat.dtype)
    _q1_decode_attn_score_kernel[(nseg,)](
        q_flat, k_flat, v_flat, out, cu_k,
        score_accumulator, score_offsets, audio_starts, audio_ends, ca_kv_mask,
        num_kv_heads,
        scale,
        G=G, GPAD=GPAD, D=D, BLOCK_N=block_n,
    )
    return out
