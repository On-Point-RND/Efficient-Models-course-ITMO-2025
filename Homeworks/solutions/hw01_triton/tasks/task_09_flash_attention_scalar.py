"""
Task 9: Simple FlashAttention (Scalar)

Реализуйте простую версию FlashAttention для одного запроса.

Даны тензоры:
- q формы (T,) — запрос;
- k формы (T,) — ключи;
- v формы (T,) — значения.

Нужно вычислить:

s_j = q * k[j]
soft = softmax(s)
res = sum_j soft[j] * v[j]


Требования:
- процессить последовательность тайлами B1 < T в одном ядре (один pid0 = одна голова);
- использовать стабильную форму softmax с трюком *running max / renormalization*:

  m_new = max(m, s_j)
  l = l * 2**(m - m_new) + 2**(s_j - m_new)
  acc = acc * 2**(m - m_new) + v_j * 2**(s_j - m_new)
  m = m_new

- результат = acc / l.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flashatt_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, B1: tl.constexpr):
    # Streaming softmax with running max / renormalization over tiles
    LOG2E = 1.4426950408889634  # log2(e)

    # Pass 1: compute running max (m) and denominator (l)
    m = -float("inf")
    l = tl.zeros((), dtype=tl.float32)

    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T

        q_tile = tl.load(q_ptr + offs, mask=mask, other=0.0)
        k_tile = tl.load(k_ptr + offs, mask=mask, other=0.0)
        s_tile = q_tile * k_tile

        tile_max = tl.max(tl.where(mask, s_tile, -float("inf")), axis=0)
        m_new = tl.maximum(m, tile_max)
        alpha = tl.exp2((m - m_new) * LOG2E)
        exps = tl.where(mask, tl.exp2((s_tile - m_new) * LOG2E), 0.0)
        l = l * alpha + tl.sum(exps, axis=0)
        m = m_new

    # Pass 2: write per-element softmax(s) * v to out
    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T
        q_tile = tl.load(q_ptr + offs, mask=mask, other=0.0)
        k_tile = tl.load(k_ptr + offs, mask=mask, other=0.0)
        v_tile = tl.load(v_ptr + offs, mask=mask, other=0.0)
        s_tile = q_tile * k_tile
        weights = tl.where(mask, tl.exp2((s_tile - m) * LOG2E) / l, 0.0)
        out_vals = weights * v_tile
        tl.store(out_ptr + offs, out_vals, mask=mask)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 1 and k.dim() == 1 and v.dim() == 1
    T = q.numel()
    assert k.numel() == T and v.numel() == T
    B1 = 128
    out = torch.empty(T, device=q.device, dtype=q.dtype)
    grid = (1,)
    _flashatt_kernel[grid](q, k, v, out, T, B1)
    return out


if __name__ == "__main__":
    # Simple correctness check vs PyTorch reference
    T = 257  # not divisible by B1 to test tail masking
    device = "cuda"
    dtype = torch.float32

    q = torch.randn(T, device=device, dtype=dtype)
    k = torch.randn(T, device=device, dtype=dtype)
    v = torch.randn(T, device=device, dtype=dtype)

    out = flashatt_triton(q, k, v)

    # PyTorch reference vectors
    s = q * k
    weights = torch.softmax(s, dim=0)
    ref_vec = weights * v

    max_err = (out - ref_vec).abs().max().item()
    print("out[:8]", out[:8])
    print("ref[:8]", ref_vec[:8])
    print("max_abs_err:", max_err)
