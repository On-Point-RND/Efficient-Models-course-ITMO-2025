"""
Task 8: Long Softmax

Вычислите построчный softmax для матрицы x формы (N0, T).

Формула:

row_max = max_j x[i, j]
row_exp = exp(x[i, j] - row_max)
out[i, j] = row_exp / sum_k row_exp


Требования:
- одномерная сетка: pid0 перебирает строки i;
- внутри ядра используйте тайлы длины B1 < T и маски для хвостов;
- посчитать row_max (первый проход);
- затем в цикле посчитать сумму экспонент в стабильной форме;
- в последнем проходе — нормализовать и сохранить;
- используйте exp2(x * log2(e)) вместо exp(x) для стабильности и скорости;
- решение должно работать на GPU (CUDA).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _long_softmax_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
    # Row pid
    pid0 = tl.program_id(0)
    row_start = pid0 * T

    # Constants
    LOG2E = 1.4426950408889634  # log2(e)

    # Pass 1: compute row-wise max for numerical stability
    row_max = -float("inf")
    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T
        vals = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
        tile_max = tl.max(vals, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # Pass 2: compute denominator = sum(exp(x - row_max)) using exp2
    denom = tl.zeros((), dtype=tl.float32)
    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T
        vals = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
        exps = tl.exp2((vals - row_max) * LOG2E)
        denom += tl.sum(exps, axis=0)

    # Pass 3: normalize and store
    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T
        vals = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
        exps = tl.exp2((vals - row_max) * LOG2E)
        out_vals = exps / denom
        tl.store(out_ptr + row_start + offs, out_vals, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2, "x must be 2D (N0, T)"
    N0, T = x.shape
    B1 = 128
    out = torch.empty_like(x)
    grid = (N0,)
    _long_softmax_kernel[grid](x, out, N0, T, B1)
    return out


if __name__ == "__main__":
    # Simple correctness check vs PyTorch softmax
    N0, T = 64, 257  # choose T not divisible by B1 to test tails
    x = torch.randn(N0, T, device="cuda", dtype=torch.float32)
    out = softmax_triton(x)
    ref = torch.softmax(x, dim=1)
    max_err = (out - ref).abs().max().item()
    print("max_abs_err:", max_err)
    print(out[0, :8])
    print(ref[0, :8])
