"""
Task 2 constant_add_block
Теперь доработаем решение из первого задания: добавим работу с блоками и необходимость маскировать хвост.

В отличие от Task 1, длина вектора N может быть произвольной (не обязательно степень двойки), а размер блока B0 меньше N. Поэтому один блок не покрывает весь вектор, и на последнем блоке нужно использовать маску.

Формула:
z[i] = const_val + x[i] для i = 0 … N-1.

Требования:
- одна ось идентификатора программы (pid 0);
- блоки размера B0 обрабатывают вектор по частям;
- использовать tl.arange, tl.load и tl.store с маской для корректной обработки хвоста;
- решение должно работать на GPU (CUDA);

Подсказка
Индексы внутри блока можно построить так:
offs = pid * B0 + tl.arange(0, B0).
При загрузке и записи используйте аргумент
mask = offs < N,
чтобы игнорировать выход за границы.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_const_block_kernel(
    x_ptr,
    z_ptr,
    const_val,
    N: tl.constexpr,
    B0: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * B0
    offsets = block_start + tl.arange(0, B0)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + const_val
    tl.store(z_ptr + offsets, output, mask=mask)


def add_block_triton(x: torch.Tensor, const_val: int) -> torch.Tensor:
    assert x.is_cuda, "Тензор должен быть на GPU (CUDA)."
    N = x.numel()
    B0 = 128  # например, фиксированный размер блока
    grid = ((N + B0 - 1) // B0,)
    z = torch.empty_like(x)
    _add_const_block_kernel[grid](x, z, const_val, N, B0)
    return z


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    const_val = 10
    z = add_triton(x, const_val)
    print(z)
