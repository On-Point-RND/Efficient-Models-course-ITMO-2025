"""
Task 7: Long Sum

Вычислите сумму по строкам матрицы x формы (N0, T).

Формула:
out[i] = sum_{j=0}^{T-1} x[i, j] для i = 0 … N0-1.

Требования:
- сетка одномерная: pid0 перебирает строки i;
- внутри ядра используйте цикл по тайлам длины B1 < T;
- в каждом шаге цикла загружайте блок tl.arange(0, B1) и аккумулируйте сумму;
- используйте маску для хвостового блока, если T не делится на B1;
- решение должно работать на GPU (CUDA).

Подсказка
- постройте индексы: row = pid0, offs = start + tl.arange(0, B1);
- вычисляйте глобальные смещения: row * T + offs;
- аккумулируйте локальную сумму в переменной acc и по завершении цикла сохраните её в out[row].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _long_sum_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
    # Row pid
    pid0 = tl.program_id(0)
    row_start = pid0 * T

    # Accumulate row sum in scalar accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over the row in tiles of size B1
    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T
        vals = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Store the result to out[row]
    tl.store(out_ptr + pid0, acc)


def sum_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2, "x must be 2D (N0, T)"
    N0, T = x.shape
    B1 = 128
    out = torch.empty((N0,), device=x.device, dtype=x.dtype)
    grid = (N0,)
    _long_sum_kernel[grid](x, out, N0, T, B1)
    return out


if __name__ == "__main__":
    # Simple correctness check vs PyTorch
    N0, T = 100, 257  # pick T not divisible by B1 to test tail masking
    x = torch.randn(N0, T, device="cuda", dtype=torch.float32)
    out = sum_triton(x)
    ref = x.sum(dim=1)
    max_err = (out - ref).abs().max().item()
    print("max_abs_err:", max_err)
    print(out[:5])
    print(ref[:5])
