"""
Task 6: Fused Outer Multiplication + ReLU — Backward

Нужно реализовать вычисление градиента по x для операции

z = relu(x[None, :] * y[:, None])

Формула:
dx[j] = sum_i 1[x[j]*y[i] > 0] * y[i] * dz[i,j]

где dz — апстрим-градиент размера (N1, N0).

Требования:
- двумерная сетка (pid1, pid0);
- размеры блоков B0 < N0, B1 < N1, требуется маскирование на обоих осях;
- использовать tl.arange, tl.load, tl.store, tl.where;
- накапливать частичные суммы и писать в dx с атомарным tl.atomic_add.

Подсказка
- индексы блоков: pid0, pid1;
- offs0 — столбцы (ось x), offs1 — строки (ось y);
- загрузите куски x, y, dz;
- вычислите u = y[:,None] * x[None,:];
- маска ReLU: relu_mask = u > 0;
- локальный градиент: g = tl.where(relu_mask, y[:,None] * dz_tile, 0);
- просуммируйте по оси i (строки) и прибавьте к dx[offs0] через tl.atomic_add.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_relu_block_back_kernel(
    x_ptr,
    y_ptr,
    dz_ptr,
    dx_ptr,
    N0,
    N1,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    # Create pids for rows and columns
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Create offsets
    offs0 = pid0 * B0 + tl.arange(0, B0)  # [B0]
    offs1 = pid1 * B1 + tl.arange(0, B1)  # [B1]

    # Create masks
    mask0 = offs0 < N0  # [B0] bool
    mask1 = offs1 < N1  # [B1] bool

    # Load x, y with masks
    x = tl.load(x_ptr + offs0, mask0, 0)  # [B0]
    y = tl.load(y_ptr + offs1, mask1, 0)  # [B1]
    # Load dz tile with 2D offsets and mask
    dz_offs = offs1[:, None] * N0 + offs0[None, :]  # [B1, B0]
    dz_mask = mask1[:, None] & mask0[None, :]
    dz_tile = tl.load(dz_ptr + dz_offs, dz_mask, 0)  # [B1, B0]

    # Compute local gradient
    u = y[:, None] * x[None, :]  # [B1, B0]
    relu_mask = u > 0
    g = tl.where(relu_mask, y[:, None] * dz_tile, 0)  # [B1, B0]

    # Reduce over rows (axis 0) to get per-column partial sums
    col_partial = tl.sum(g, axis=0)  # [B0]

    # Add to dx with atomic add
    tl.atomic_add(dx_ptr + offs0, col_partial, mask0)


def mul_relu_block_back_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    dz: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and dz.is_cuda
    N0 = x.numel()
    N1 = y.numel()
    assert dz.numel() == N0 * N1 and dz.shape == (N1, N0)
    B0 = 128
    B1 = 128
    grid = (triton.cdiv(N0, B0), triton.cdiv(N1, B1))
    dx = torch.zeros_like(x)
    _mul_relu_block_back_kernel[grid](x, y, dz, dx, N0, N1, B0, B1)
    return dx


if __name__ == "__main__":
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    dz = torch.randn(100, 100).cuda()
    dx = mul_relu_block_back_triton(x, y, dz)
    print(dx)
