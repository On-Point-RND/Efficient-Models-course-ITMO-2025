"""
Task 1 constant_add
Прибавьте произвольную константу const_val к каждому элементу входного вектора x и сохраните результат в выходном векторе z.

Формула:
z[i] = const_val + x[i] для i = 0 … N-1.

Требования:
- одна ось идентификатора программы (pid 0);
- размер блока B0 равен длине вектора N (один блок покрывает весь вектор);
- использовать tl.arange, tl.load и tl.store без маски;
- решение должно работать на GPU (CUDA);

Подсказка: индексы можно построить так:
```
offs = pid * B0 + tl.arange(0, B0)
```
и затем загрузить `x_ptr + offs`
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_const_kernel(x_ptr, z_ptr, const_val, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * N
    offsets = block_start + tl.arange(0, N)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + const_val
    tl.store(z_ptr + offsets, output, mask=mask)



def add_triton(x: torch.Tensor, const_val: int) -> torch.Tensor:
    assert x.is_cuda, 'Тензор должен быть на GPU (CUDA).'
    N = x.numel()
    # по условию один блок полностью покрывает вектор
    grid = (1,)
    z = torch.empty_like(x)
    _add_const_kernel[grid](x, z, const_val, N)
    return z


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    const_val = 10
    z = add_triton(x, const_val)
    print(z)
