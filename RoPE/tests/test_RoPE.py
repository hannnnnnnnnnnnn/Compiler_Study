import torch
from solutions.triton_solutions import (
    triton_apply_rotary_pos_emb,
    triton_apply_rotary_pos_emb_v1,
    triton_apply_rotary_pos_emb_v2,
)
from bench_RoPE import make_freqs
from solutions.torch_solution import apply_rotary_pos_emb
import pytest


@pytest.mark.parametrize("seq_len", list(range(128, 2048, 64)))
@pytest.mark.parametrize("dim", list(range(128, 2048, 64)))
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("head", list(range(1, 2, 64)))
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
# @pytest.mark.parametrize("seq_len", [640])
# @pytest.mark.parametrize("dim", [1728])
# @pytest.mark.parametrize("batch", [16])
# @pytest.mark.parametrize("head", [1])
# @pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
def test_RoPE(seq_len, dim, batch, head, tensor_format):
    torch.manual_seed(1234)
    x = torch.randn(seq_len, batch, head, dim, device="cuda")
    x = x * 100
    if tensor_format == "bshd":
        x = x.transpose(0, 1)

    # real freqs
    # freqs = make_freqs(seq_len * 2, dim // 2)

    # fake freqs, but faster
    freqs = torch.randn(seq_len * 2, 1, 1, dim // 2, device="cuda")

    # jit
    # y_jit = rope_torch_jit(x, freqs)

    # torch
    y_torch = apply_rotary_pos_emb(x, freqs, tensor_format=tensor_format)

    # triton
    # y_triton = triton_apply_rotary_pos_emb(x, freqs, tensor_format=tensor_format)
    # y_triton = triton_apply_rotary_pos_emb_v1(x, freqs, tensor_format=tensor_format)
    y_triton = triton_apply_rotary_pos_emb_v2(x, freqs, tensor_format=tensor_format)

    torch.testing.assert_close(
        y_triton, y_torch, rtol=1e-05, atol=2e-05
    )  # , (y_triton, y_torch)
    # for i in range(y_triton.shape[0]):
    #     for b in range(y_triton.shape[1]):
    #         for h in range(y_triton.shape[2]):
    #             for d in range(y_triton.shape[3]):
    #                 try:
    #                     torch.testing.assert_close(y_triton[i,b,h,d], y_torch[i,b,h,d])
    #                 except AssertionError as e:
    #                     print(i, b, h, d, y_triton[i,b,h,d], y_torch[i,b,h,d])
    #                     raise e
    print("PASS")
