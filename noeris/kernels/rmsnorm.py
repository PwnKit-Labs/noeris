"""Triton RMSNorm kernel.

Single-pass fused RMSNorm that replaces PyTorch's multi-op decomposition.
Supports both standard (y = x * rstd * w) and Gemma-mode (y = x * rstd * (1+w)).
"""

from __future__ import annotations

_kernel = None


def _ensure_compiled():
    global _kernel
    if _kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr, w_ptr, y_ptr,
        x_row_stride, y_row_stride,
        n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
        AFFINE_MODE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        x_ptr += row_idx * x_row_stride
        y_ptr += row_idx * y_row_stride
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if AFFINE_MODE == 0:
            y = x * rstd * w
        else:
            y = x * rstd * (1.0 + w)
        tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)

    _kernel = _rmsnorm_kernel


def rmsnorm_forward(x, w, eps=1e-6, affine_mode=0):
    """Launch Triton RMSNorm kernel.

    Args:
        x: (n_rows, hidden_dim) fp16 tensor.
        w: (hidden_dim,) fp16/fp32 weight tensor.
        eps: Epsilon for numerical stability.
        affine_mode: 0 = standard, 1 = Gemma (1+w).

    Returns:
        (n_rows, hidden_dim) fp16 output tensor.
    """
    import torch
    import triton

    _ensure_compiled()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _kernel[(n_rows,)](
        x, w, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        AFFINE_MODE=affine_mode,
        num_warps=min(16, max(1, BLOCK_SIZE // 256)),
        num_stages=1,
    )
    return y
