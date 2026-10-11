
import torch
import triton
import triton.language as tl


@triton.jit
def _interleave_append_kernel(
    Old_ptr,
    New_ptr,
    Out_ptr,
    cu_old_ptr,
    cu_new_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)

    lo = 0
    hi = N
    for _ in range(10):
        mid = (lo + hi) // 2
        val = tl.load(cu_new_ptr + mid + 1)
        lo = tl.where(val <= row_id, mid + 1, lo)
        hi = tl.where(val <= row_id, hi, mid)
    seg_id = lo

    new_start = tl.load(cu_new_ptr + seg_id)
    old_start = tl.load(cu_old_ptr + seg_id)
    old_len = tl.load(cu_old_ptr + seg_id + 1) - old_start
    row_in_seg = row_id - new_start

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    if row_in_seg < old_len:
        vals = tl.load(Old_ptr + (old_start + row_in_seg) * D + offs_d, mask=mask_d)
    else:
        vals = tl.load(New_ptr + seg_id * D + offs_d, mask=mask_d)

    tl.store(Out_ptr + row_id * D + offs_d, vals, mask=mask_d)


def interleave_append(
    old: torch.Tensor,
    new_tokens: torch.Tensor,
    cu_old: torch.Tensor,
    cu_new: torch.Tensor,
) -> torch.Tensor:
    """Append one new token per segment into flat cache, producing new flat tensor.

    Args:
        old: [total_old, D] existing flat cache
        new_tokens: [N, D] one token per segment
        cu_old: [N+1] int32 cumulative lengths of old segments
        cu_new: [N+1] int32 cumulative lengths of new segments (each = old + 1)

    Returns: [total_old + N, D]
    """
    N = new_tokens.shape[0]
    D = old.shape[1]
    total_out = old.shape[0] + N
    out = torch.empty(total_out, D, dtype=old.dtype, device=old.device)
    BLOCK_D = triton.next_power_of_2(D)

    with torch.cuda.device(old.device):
        _interleave_append_kernel[(total_out,)](
            old, new_tokens, out,
            cu_old, cu_new,
            N=N, D=D, BLOCK_D=BLOCK_D,
        )
    return out
