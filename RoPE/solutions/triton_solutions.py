import triton
import torch
import triton.language as tl
from typing import Union
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@triton.jit
def RoPE_base(
    t_ptr,
    seq_len,
    t_stride,
    batch_size,
    batch_stride,
    head_size,
    head_stride,
    dim,
    dim_stride,
    f_ptr,
    f_stride,
    f_dim_stride,
    out_ptr,
    out_stride,
    out_dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_starts = pid * BLOCK_SIZE
    offsets = block_starts + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim

    org_out_ptrs = out_ptr + offsets * out_dim_stride
    org_cos_t_ptrs = t_ptr + offsets * dim_stride
    org_f_ptrs = f_ptr + offsets * f_dim_stride

    is_reversed = offsets >= (dim // 2)
    sign_sin = -1 + 2 * is_reversed
    org_sin_t_ptrs = org_cos_t_ptrs + (1 - 2 * is_reversed) * (dim // 2) * dim_stride
    for s in range(seq_len):
        f_ptrs = org_f_ptrs + s * f_stride
        mthetas = tl.load(f_ptrs, mask=mask)
        cos_theta = tl.cos(mthetas)
        sin_theta = tl.sin(mthetas)
        for b in range(batch_size):
            for h in range(head_size):
                cos_t_ptrs = (
                    org_cos_t_ptrs + s * t_stride + b * batch_stride + h * head_stride
                )
                sin_t_ptrs = (
                    org_sin_t_ptrs + s * t_stride + b * batch_stride + h * head_stride
                )
                out_ptrs = (
                    org_out_ptrs + s * out_stride + b * batch_stride + h * head_stride
                )
                cos_t = tl.load(cos_t_ptrs, mask=mask)
                sin_t = tl.load(sin_t_ptrs, mask=mask)
                o = cos_t * cos_theta + sign_sin * sin_t * sin_theta
                tl.store(out_ptrs, o, mask=mask)


@triton.jit
def RoPE_v1(
    t_ptr,
    seq_len,
    t_stride,
    batch_size,
    batch_stride,
    head_size,
    head_stride,
    dim,
    dim_stride,
    f_ptr,
    f_stride,
    f_dim_stride,
    out_ptr,
    out_stride,
    out_dim_stride,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    starts_d = pid_d * BLOCK_SIZE_D
    starts_t = pid_t * BLOCK_SIZE_T
    offsets = (starts_d + tl.arange(0, BLOCK_SIZE_D)) % dim
    offsets_t = (starts_t + tl.arange(0, BLOCK_SIZE_T)) % seq_len
    org_out_ptrs = (
        out_ptr + offsets[None, :] * out_dim_stride + offsets_t[:, None] * out_stride
    )
    org_cos_t_ptrs = (
        t_ptr + offsets[None, :] * dim_stride + offsets_t[:, None] * t_stride
    )
    is_reversed = offsets >= (dim // 2)
    sin_offsets = offsets + (1 - 2 * is_reversed) * (dim // 2) * dim_stride
    org_sin_t_ptrs = (
        t_ptr + sin_offsets[None, :] * dim_stride + offsets_t[:, None] * t_stride
    )
    org_f_ptrs = f_ptr + offsets[None, :] * f_dim_stride + offsets_t[:, None] * f_stride

    sign_sin = -1 + 2 * is_reversed
    mthetas = tl.load(org_f_ptrs)
    cos_theta = tl.cos(mthetas)
    sin_theta = tl.sin(mthetas)
    for b in range(batch_size):
        for h in range(head_size):
            cos_t_ptrs = org_cos_t_ptrs + b * batch_stride + h * head_stride
            sin_t_ptrs = org_sin_t_ptrs + b * batch_stride + h * head_stride
            out_ptrs = org_out_ptrs + b * batch_stride + h * head_stride
            cos_t = tl.load(cos_t_ptrs)

            sin_t = tl.load(sin_t_ptrs)
            o = cos_t * cos_theta + sign_sin[None, :] * sin_t * sin_theta
            tl.store(out_ptrs, o)


@triton.jit
def RoPE_v2(
    t_ptr,
    t_stride,
    batch_size,
    batch_stride,
    head_stride,
    remained_dim_power2: tl.constexpr,
    remained_dim,
    dim,
    dim_stride,
    f_ptr,
    f_stride,
    f_dim_stride,
    out_ptr,
    out_stride,
    out_dim_stride,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1) // batch_size
    pid_h = tl.program_id(1) % batch_size
    half_dim = dim // 2
    offsets = tl.arange(0, BLOCK_SIZE_D)  # BLOCK_SIZE_D is always larger than half_dim
    mask = offsets < half_dim
    base_out_ptr = (
        out_ptr + pid_b * batch_stride + pid_h * head_stride + pid_t * out_stride
    )
    base_t_ptr = t_ptr + pid_b * batch_stride + pid_h * head_stride + pid_t * t_stride
    base_f_ptr = f_ptr + pid_t * f_stride
    t_ptr1 = base_t_ptr + offsets * dim_stride
    t_ptr2 = base_t_ptr + (half_dim + offsets) * dim_stride
    out_ptr1 = base_out_ptr + offsets * out_dim_stride
    out_ptr2 = base_out_ptr + (half_dim + offsets) * out_dim_stride

    f_ptr1 = base_f_ptr + offsets * f_dim_stride
    f_ptr2 = base_f_ptr + (half_dim + offsets) * f_dim_stride
    t1 = tl.load(t_ptr1, mask=mask)
    t2 = tl.load(t_ptr2, mask=mask)

    f1 = tl.load(f_ptr1, mask=mask)
    f2 = tl.load(f_ptr2, mask=mask)

    o1 = t1 * tl.cos(f1) - t2 * tl.sin(f1)
    o2 = t2 * tl.cos(f2) + t1 * tl.sin(f2)
    tl.store(out_ptr1, o1, mask=mask)
    tl.store(out_ptr2, o2, mask=mask)

    if remained_dim > 0:
        offsets_remained = tl.arange(0, remained_dim_power2)
        org_t = tl.load(
            base_t_ptr + (dim + offsets_remained) * dim_stride,
            mask=offsets_remained < remained_dim,
        )
        tl.store(
            base_out_ptr + (dim + offsets_remained) * out_dim_stride,
            org_t,
            mask=offsets_remained < remained_dim,
        )


def triton_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
):

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )
    out = torch.empty_like(t)  # t.clone().detach()
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    seq_stride = t.stride(1) if tensor_format == "bshd" else t.stride(0)
    batch_size = t.shape[0] if tensor_format == "bshd" else t.shape[1]
    batch_stride = t.stride(0) if tensor_format == "bshd" else t.stride(1)
    out_stride = out.stride(1) if tensor_format == "bshd" else out.stride(0)

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert (
        cur_seq_len <= max_seq_len
    ), f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    f_stride = freqs.stride(1) if tensor_format == "bshd" else freqs.stride(0)
    # cos/sin first then dtype conversion for better precision
    rot_dim = freqs.shape[-1]

    return (
        freqs,
        rot_dim,
        cur_seq_len,
        seq_stride,
        batch_size,
        batch_stride,
        f_stride,
        out,
        out_stride,
    )


def triton_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
) -> torch.Tensor:
    (
        freqs,
        rot_dim,
        cur_seq_len,
        seq_stride,
        batch_size,
        batch_stride,
        f_stride,
        out,
        out_stride,
    ) = triton_base(t, freqs, tensor_format)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(rot_dim, BLOCK_SIZE),)
    RoPE_base[grid](
        t,
        cur_seq_len,
        seq_stride,
        batch_size,
        batch_stride,
        t.shape[2],
        t.stride(2),
        rot_dim,
        t.stride(-1),
        freqs,
        f_stride,
        freqs.stride(-1),
        out,
        out_stride,
        out.stride(-1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_apply_rotary_pos_emb_v1(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
) -> torch.Tensor:
    (
        freqs,
        rot_dim,
        cur_seq_len,
        seq_stride,
        batch_size,
        batch_stride,
        f_stride,
        out,
        out_stride,
    ) = triton_base(t, freqs, tensor_format)
    BLOCK_SIZE_D = 8
    BLOCK_SIZE_T = 8
    grid = (
        triton.cdiv(cur_seq_len, BLOCK_SIZE_T),
        triton.cdiv(cur_seq_len, BLOCK_SIZE_D),
    )
    RoPE_v1[grid](
        t,
        cur_seq_len,
        seq_stride,
        batch_size,
        batch_stride,
        t.shape[2],
        t.stride(2),
        rot_dim,
        t.stride(-1),
        freqs,
        f_stride,
        freqs.stride(-1),
        out,
        out_stride,
        out.stride(-1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
    )
    return out


def triton_apply_rotary_pos_emb_v2(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
) -> torch.Tensor:
    (
        freqs,
        rot_dim,
        cur_seq_len,
        seq_stride,
        batch_size,
        batch_stride,
        f_stride,
        out,
        out_stride,
    ) = triton_base(t, freqs, tensor_format)
    BLOCK_SIZE_D = triton.next_power_of_2(rot_dim // 2)
    num_warps = 4
    grid = (cur_seq_len, batch_size * t.shape[2])
    remained_amount = triton.next_power_of_2(t.shape[-1] - rot_dim)
    RoPE_v2[grid](
        t,
        seq_stride,
        batch_size,
        batch_stride,
        t.stride(2),
        remained_amount,
        t.shape[-1] - rot_dim,
        rot_dim,
        t.stride(-1),
        freqs,
        f_stride,
        freqs.stride(-1),
        out,
        out_stride,
        out.stride(-1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
    )
    return out
