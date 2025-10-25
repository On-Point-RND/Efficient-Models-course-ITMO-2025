"""
Task 3: Outer Vector Add

В этой задаче требуется реализовать вычисление *внешней суммы* двух векторов x и y, то есть построить матрицу z размера (N1, N0):

Формула:
z[i, j] = x[j] + y[i] для i = 0 … N1-1, j = 0 … N0-1.

Требования:
- сетка состоит из одной оси программ-блоков, но размеры блоков равны размерам входных векторов (B0 == N0, B1 == N1);
- маска не требуется (нет неполных блоков);
- использовать tl.arange, tl.load, tl.store;
- решение должно работать на GPU (CUDA).

Подсказка
- используйте row_ids = tl.arange(0, N1) и col_ids = tl.arange(0, N0) для построения индексов;
- загрузите x и y через tl.load по этим индексам;
- результат можно собрать как y[:, None] + x[None, :];
- для записи используйте смещения в виде offs = row_ids[:, None] * N0 + col_ids[None, :] и tl.store(z_ptr + offs, z).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _outer_add_kernel(x_ptr, y_ptr, z_ptr, N0: tl.constexpr, N1: tl.constexpr):
    # Create indices for rows and columns
    row_ids = tl.arange(0, N1)  # indices for y (N1 elements)
    col_ids = tl.arange(0, N0)  # indices for x (N0 elements)

    # Load x and y vectors
    x = tl.load(x_ptr + col_ids)  # shape: (N0,)
    y = tl.load(y_ptr + row_ids)  # shape: (N1,)

    # Compute offsets for storing the result in row-major order
    offs = row_ids[:, None] * N0 + col_ids[None, :]

    # Compute outer sum: z[i, j] = x[j] + y[i]
    # Using broadcasting: y[:, None] + x[None, :]
    z = y[:, None] + x[None, :]  # shape: (N1, N0)

    # Store the result
    tl.store(z_ptr + offs, z)


def outer_vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    N0 = x.numel()
    N1 = y.numel()
    grid = (1,)  # Single program block (one axis)
    z = torch.empty(size=(N1, N0), device="cuda")
    _outer_add_kernel[grid](x, y, z, N0, N1)
    return z


if __name__ == "__main__":
    x = torch.randn(8, device="cuda")
    y = torch.randn(16, device="cuda")
    z = outer_vector_add_triton(x, y)
    print(z)