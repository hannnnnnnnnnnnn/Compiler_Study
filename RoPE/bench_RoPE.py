import torch
from solutions.torch_solution import apply_rotary_pos_emb, rope_torch_jit
from solutions.triton_solutions import (
    triton_apply_rotary_pos_emb_v1,
    triton_apply_rotary_pos_emb_v2,
)
import triton


def make_freqs(seq_len, dim):
    out = torch.empty((seq_len, 1, 1, dim), device="cuda")
    for s in range(seq_len):
        for d in range(dim):
            i = d // 2
            theta = 10000.0 ** (-2 * i / dim)
            out[s, 0, 0, i] = s * theta
    return out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[128 * i for i in range(2, 10)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_jit", "cuda"],
        line_names=["Triton", "Torch", "torch_jit", "cuda"],
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("black", "-"),
            ("red", "-"),
        ],
        ylabel="Gbps",
        plot_name="RoPE_bench_v2",
        args={
            "dim": 512,
            "head": 10,
            "batch": 16,
            "dtype": torch.float32,
            "mode": "forward",
        },
    )
)
def bench_RoPE(
    dim, seq_len, head, batch, dtype, provider, mode="forward", eps=1e-5, device="cuda"
):
    # create data
    print(seq_len)
    x_shape = (seq_len, batch, head, dim)
    f = torch.randn(seq_len * 2, 1, 1, dim // 2, device="cuda")
    x = torch.randn(*x_shape, device="cuda")
    x = x * 100
    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":

        def y_fwd():
            return triton_apply_rotary_pos_emb_v2(x, f)

    if provider == "torch":

        def y_fwd():
            return apply_rotary_pos_emb(x, f)

    if provider == "torch_jit":

        def y_fwd():
            return rope_torch_jit(x, f)

    if provider == "cuda":
        from transformer_engine.pytorch.attention import (
            apply_rotary_pos_emb as cuda_rotary,
        )

        def y_fwd():
            return cuda_rotary(x, f, fused=True)

    # forward pass
    if mode == "forward":
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(
            y_fwd, quantiles=quantiles, rep=500
        )
    # backward pass
    if mode == "backward":

        def gbps(ms):
            return 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704

        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x],
            rep=500,
        )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    bench_RoPE.run(save_path="results/", print_data=True)
