import torch
from solutions.triton_solutions import FusedRoPE
from bench_RoPE import make_freqs
from solutions.torch_solution import apply_rotary_pos_emb
import pytest

fused_rope = FusedRoPE.apply


@pytest.fixture
def get_data(seq_len, batch, dim, head):
    torch.manual_seed(1234)
    x = torch.randn(seq_len, batch, head, dim, device="cuda")
    x = x * 100
    # real freqs
    # freqs = make_freqs(seq_len * 2, dim // 2)

    # fake freqs, but faster
    freqs = torch.randn(seq_len * 2, 1, 1, dim // 2, device="cuda")
    return x, freqs


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
def test_RoPE(get_data, tensor_format):
    x, freqs = get_data
    if tensor_format == "bshd":
        x = x.transpose(0, 1)

    # jit
    # y_jit = rope_torch_jit(x, freqs)

    # torch
    y_torch = apply_rotary_pos_emb(x, freqs, tensor_format=tensor_format)

    # triton
    y_triton = fused_rope(x, freqs, tensor_format)

    torch.testing.assert_close(y_triton, y_torch, rtol=1e-05, atol=2e-05)  # , (y_triton, y_torch)


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
def test_RoPE_bw(get_data, tensor_format):
    x, freqs = get_data
    if tensor_format == "bshd":
        x = x.transpose(0, 1)

    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    # torch
    y_torch = apply_rotary_pos_emb(x, freqs, tensor_format=tensor_format)

    y_torch.backward(dy, retain_graph=True)
    dx_torch = x.grad.clone()

    x.grad = None
    # triton
    y_triton = fused_rope(x, freqs, tensor_format)
    y_triton.backward(dy, retain_graph=True)
    dx_triton = x.grad.clone()

    torch.testing.assert_close(dx_torch, dx_triton, atol=1e-2, rtol=0)
