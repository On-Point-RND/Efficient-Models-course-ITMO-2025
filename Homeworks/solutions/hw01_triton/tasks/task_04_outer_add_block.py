"""
Task 4: Outer Vector Add (Block)

Теперь реализуем внешнюю сумму двух векторов с разбиением на блоки.

Формула:
z[i, j] = x[j] + y[i] для i = 0 … N1-1, j = 0 … N0-1.

Требования:
- сетка двумерная: (pid1, pid0) — строки и столбцы;
- размеры блоков B0 < N0, B1 < N1, поэтому нужно маскировать хвосты;
- использовать tl.arange, tl.load, tl.store;
- решение должно работать на GPU (CUDA).

Подсказка
- внутри ядра постройте глобальные индексы:

  pid0 = tl.program_id(0)
  pid1 = tl.program_id(1)
  offs0 = pid0 * B0 + tl.arange(0, B0)
  offs1 = pid1 * B1 + tl.arange(0, B1)

- создайте маски: mask0 = offs0 < N0, mask1 = offs1 < N1;
- загрузите x и y с масками, сделайте broadcast: y[:, None] + x[None, :];
- вычислите смещения: offs = offs1[:, None] * N0 + offs0[None, :] и сохраните результат через tl.store с маской mask1[:, None] & mask0[None, :].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _outer_add_block_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    # Create pids for rows and columns
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Create offsets
    offs0 = pid0 * B0 + tl.arange(0, B0)
    offs1 = pid1 * B1 + tl.arange(0, B1)

    # Create masks
    mask0 = offs0 < N0
    mask1 = offs1 < N1

    # Load x, y with masks
    x = tl.load(x_ptr + offs0, mask0, 0)
    y = tl.load(y_ptr + offs1, mask1, 0)

    # Compute outer sum: z[i, j] = x[j] + y[i]
    z = y[:, None] + x[None, :]  # shape: (N1, N0)

    # Count offsets&masks and store the result
    offs = offs1[:, None] * N0 + offs0[None, :]
    mask = mask1[:, None] & mask0[None, :]
    tl.store(z_ptr + offs, z, mask)


def add_vec_block_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    B0: int = 128,
    B1: int = 128,
) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    N0 = x.numel()
    N1 = y.numel()
    # axis 0 iterates over columns (N0, block B0); axis 1 over rows (N1, block B1)
    grid = (triton.cdiv(N0, B0), triton.cdiv(N1, B1))
    z = torch.empty(size=(N1, N0), device=x.device, dtype=x.dtype)
    _outer_add_block_kernel[grid](x, y, z, N0, N1, B0, B1)
    return z


if __name__ == "__main__":
    x = torch.randn(1024, device="cuda")
    y = torch.randn(2048, device="cuda")
    z = add_vec_block_triton(x, y)
    print(z)
