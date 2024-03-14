import triton
import torch
import triton.language as tl
from typing import Union

@triton.jit
def RoPE_base(t_ptr, seq_len, t_stride, batch_size, batch_stride, head_size, head_stride, dim, dim_stride, f_ptr, f_stride, f_dim_stride, out_ptr, out_stride, out_dim_stride, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    block_starts = pid * BLOCK_SIZE
    offsets = block_starts + tl.arange(0,BLOCK_SIZE)
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
                cos_t_ptrs = org_cos_t_ptrs + s * t_stride + b * batch_stride + h * head_stride
                sin_t_ptrs = org_sin_t_ptrs + s * t_stride + b * batch_stride + h * head_stride
                out_ptrs = org_out_ptrs + s * out_stride + b * batch_stride + h * head_stride
                cos_t = tl.load(cos_t_ptrs, mask=mask)
                sin_t = tl.load(sin_t_ptrs, mask=mask)
                o = cos_t * cos_theta + sign_sin * sin_t * sin_theta
                tl.store(out_ptrs, o, mask=mask)
@triton.jit
def RoPE_v1(t_ptr, seq_len, t_stride, batch_size, batch_stride, head_size, head_stride, dim, dim_stride, f_ptr, f_stride, f_dim_stride, out_ptr, out_stride, out_dim_stride, BLOCK_SIZE_D:tl.constexpr, BLOCK_SIZE_T:tl.constexpr):
    pid = tl.program_id(0)
    num_t_block = tl.cdiv(seq_len, BLOCK_SIZE_T)
    starts_d = (pid % num_t_block) * BLOCK_SIZE_D
    starts_t = (pid // num_t_block) * BLOCK_SIZE_T
    offsets = (starts_d + tl.arange(0,BLOCK_SIZE_D)) % dim
    offsets_t = (starts_t + tl.arange(0, BLOCK_SIZE_T)) % seq_len
    org_out_ptrs = out_ptr + offsets[None,:] * out_dim_stride + offsets_t[:,None] * t_stride
    tl.device_print("cos ", org_out_ptrs)
    org_cos_t_ptrs = t_ptr + offsets[None,:] * dim_stride + offsets_t[:,None] * t_stride
    is_reversed = offsets >= (dim // 2)
    sin_offsets = offsets + (1 - 2 * is_reversed) * (dim // 2) * dim_stride
    org_sin_t_ptrs = t_ptr + sin_offsets[None,:] * dim_stride + offsets_t[:,None] * t_stride
    org_f_ptrs = f_ptr + offsets[None,:] * f_dim_stride + offsets_t[:,None] * t_stride

    
    sign_sin = -1 + 2 * is_reversed
    f_ptrs = org_f_ptrs
    mthetas = tl.load(f_ptrs)
    cos_theta = tl.cos(mthetas)
    sin_theta = tl.sin(mthetas)
    for b in range(batch_size):
        for h in range(head_size):
            cos_t_ptrs = org_cos_t_ptrs + b * batch_stride + h * head_stride
            sin_t_ptrs = org_sin_t_ptrs + b * batch_stride + h * head_stride
            out_ptrs = org_out_ptrs + b * batch_stride + h * head_stride
            cos_t = tl.load(cos_t_ptrs)
            
            sin_t = tl.load(sin_t_ptrs)
            o = cos_t * cos_theta + sign_sin * sin_t * sin_theta
            tl.store(out_ptrs, o)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def rope_torch_jit(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[0]

    assert cur_seq_len <= max_seq_len

    freqs = freqs[:cur_seq_len]

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)




def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[s, b, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
    """
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        return
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def triton_base(t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,):

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )
    out = t.clone().detach()
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    seq_stride = t.stride(1) if tensor_format == "bshd" else t.stride(0)
    batch_size = t.shape[0] if tensor_format == "bshd" else t.shape[1]
    batch_stride = t.stride(0) if tensor_format == "bshd" else t.stride(1)
    out_stride = out.stride(1) if tensor_format == "bshd" else out.stride(0)

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    f_stride = freqs.stride(1) if tensor_format == "bshd" else freqs.stride(0)
    # cos/sin first then dtype conversion for better precision
    rot_dim = freqs.shape[-1]

    return rot_dim, cur_seq_len, seq_stride, batch_size, batch_stride, f_stride, out, out_stride
def triton_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        # TODO
        return
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)

    
    rot_dim, cur_seq_len, seq_stride, batch_size, batch_stride, f_stride, out, out_stride = triton_base(t, freqs, tensor_format, fused, cu_seqlens)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(rot_dim, BLOCK_SIZE), )
    RoPE_base[grid](t, cur_seq_len, seq_stride, batch_size, batch_stride, t.shape[2], t.stride(2), rot_dim, t.stride(-1), freqs, f_stride, freqs.stride(-1), out, out_stride, out.stride(-1), BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_apply_rotary_pos_emb_v1(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        # TODO
        return
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)
    rot_dim, cur_seq_len, seq_stride, batch_size, batch_stride, f_stride, out, out_stride = triton_base(t, freqs, tensor_format, fused, cu_seqlens)
    BLOCK_SIZE_D = 128
    BLOCK_SIZE_T = 128
    grid = (triton.cdiv(rot_dim, BLOCK_SIZE_D) * triton.cdiv(cur_seq_len, BLOCK_SIZE_T), )
    RoPE_v1[grid](t, cur_seq_len, seq_stride, batch_size, batch_stride, t.shape[2], t.stride(2), rot_dim, t.stride(-1), freqs, f_stride, freqs.stride(-1), out, out_stride, out.stride(-1), BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_T=BLOCK_SIZE_T)
    return out

def make_freqs(seq_len, dim):
  out = torch.empty((seq_len, 1, 1,dim), device='cuda')
  for s in range(seq_len):
    for d in range(dim):
      i = d // 2
      theta = 10000.0 ** (-2 * i / dim)
      out[s, 0,0,i] = s * theta
  return out

def test_RoPE():
    torch.manual_seed(1234)
    x = torch.randn(334, 2, 3, 356, device='cuda')
    x = x * 100
    freqs = make_freqs(512, 356)
    import time
    tic = time.time()
    # y_jit = rope_torch_jit(x, freqs)
    y_torch = apply_rotary_pos_emb(x, freqs)
    y_triton = triton_apply_rotary_pos_emb_v1(x, freqs)
    print(time.time() - tic)
    torch.testing.assert_close(y_triton, y_torch)#, (y_triton, y_torch)
    print("PASS")
    # for i in range(2):#y_triton.shape[0]):
    #     for b in range(y_triton.shape[1]):
    #         for h in range(y_triton.shape[2]):
    #             for d in range(10):#y_triton.shape[3]):
    #                 try:
    #                     torch.testing.assert_close(y_triton[i,b,h,d], y_torch[i,b,h,d])
    #                 except AssertionError as e:
    #                     print(e)
    #                     print(i, b, h, d, y_triton[i,b,h,d], y_torch[i,b,h,d])

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128 * i for i in range(2, 10)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch_jit', 'cuda'],
        line_names=['Triton', 'Torch', 'torch_jit', 'cuda'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'), ('black', '-')],
        ylabel='Gbps',
        plot_name='RoPE_forward',
        args={'dim': 1024, 'head':10, 'batch':1, 'dtype': torch.float32, 'mode': 'forward'},
    ))
def bench_RoPE(dim, seq_len, head, batch, dtype, provider, mode='forward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (seq_len, batch, head, dim)
    f = make_freqs(seq_len, dim)
    x = torch.randn(*x_shape, device='cuda')
    x = x * 100
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':

        def y_fwd():
            return triton_apply_rotary_pos_emb(x, f)

    if provider == 'torch':

        def y_fwd():
            return apply_rotary_pos_emb(x, f)
    if provider == 'torch_jit':

        def y_fwd():
            return rope_torch_jit(x, f)
    if provider == 'cuda':
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb as cuda_rotary
        def y_fwd():
            return cuda_rotary(x, f, fused=True)
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':

        def gbps(ms):
            return 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704

        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)




if __name__ == "__main__":
    
    test_RoPE()
    
    # bench_RoPE.run(save_path='.', print_data=True)

   