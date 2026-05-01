"""Fused gated activation kernels and standalone GELU.

GeGLU:  out = gate * GELU_tanh(up)     (Gemma 2/3/4)
SwiGLU: out = gate * SiLU(up)          (LLaMA, Mistral, Qwen, Phi)
GELU:   out = GELU(x)                  (standalone, no gate tensor)
"""

from __future__ import annotations

_geglu_kernel = None
_swiglu_kernel = None
_gelu_tanh_kernel = None
_gelu_exact_kernel = None


def _ensure_geglu():
    global _geglu_kernel
    if _geglu_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _geglu_k(
        gate_ptr, up_ptr, out_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        gate_ptr = gate_ptr + row_idx * n_cols
        up_ptr = up_ptr + row_idx * n_cols
        out_ptr = out_ptr + row_idx * n_cols
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (up + coeff * up * up * up)
        gelu_up = 0.5 * up * (1.0 + tl.extra.libdevice.tanh(inner))
        out = gate * gelu_up
        tl.store(out_ptr + offs, out.to(tl.float16), mask=mask)

    _geglu_kernel = _geglu_k


def _ensure_swiglu():
    global _swiglu_kernel
    if _swiglu_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_k(
        gate_ptr, up_ptr, out_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        gate_ptr = gate_ptr + row_idx * n_cols
        up_ptr = up_ptr + row_idx * n_cols
        out_ptr = out_ptr + row_idx * n_cols
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        silu_up = up * tl.sigmoid(up)
        out = gate * silu_up
        tl.store(out_ptr + offs, out.to(tl.float16), mask=mask)

    _swiglu_kernel = _swiglu_k


def _launch_gated(kernel_fn, gate, up):
    import torch
    import triton

    n_rows, n_cols = gate.shape
    out = torch.empty_like(gate)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    kernel_fn[(n_rows,)](
        gate, up, out, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(16, max(1, BLOCK_SIZE // 256)),
        num_stages=1,
    )
    return out


def geglu_forward(gate, up):
    """Fused GeGLU: gate * GELU_tanh(up). Used by Gemma models."""
    _ensure_geglu()
    return _launch_gated(_geglu_kernel, gate, up)


def swiglu_forward(gate, up):
    """Fused SwiGLU: gate * SiLU(up). Used by LLaMA/Mistral/Qwen/Phi."""
    _ensure_swiglu()
    return _launch_gated(_swiglu_kernel, gate, up)


# ---------------------------------------------------------------------------
# Standalone GELU kernels (no gate tensor, 2D tiled grid)
# ---------------------------------------------------------------------------

def _ensure_gelu_tanh():
    global _gelu_tanh_kernel
    if _gelu_tanh_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _gelu_tanh_k(
        x_ptr, out_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_block = tl.program_id(1)
        x_ptr = x_ptr + row_idx * n_cols
        out_ptr = out_ptr + row_idx * n_cols
        offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        out = 0.5 * x * (1.0 + tl.extra.libdevice.tanh(inner))
        tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

    _gelu_tanh_kernel = _gelu_tanh_k


def _ensure_gelu_exact():
    global _gelu_exact_kernel
    if _gelu_exact_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _gelu_exact_k(
        x_ptr, out_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_block = tl.program_id(1)
        x_ptr = x_ptr + row_idx * n_cols
        out_ptr = out_ptr + row_idx * n_cols
        offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476
        out = x * 0.5 * (1.0 + tl.extra.libdevice.erf(x * inv_sqrt2))
        tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

    _gelu_exact_kernel = _gelu_exact_k


def _launch_gelu(kernel_fn, x):
    """Launch a standalone GELU kernel with 2D grid (rows x col_tiles)."""
    import torch
    import triton

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    num_col_blocks = triton.cdiv(n_cols, BLOCK_SIZE)
    kernel_fn[(n_rows, num_col_blocks)](
        x, out, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(16, max(1, BLOCK_SIZE // 256)),
        num_stages=1,
    )
    return out


def gelu_forward(x, approximate="tanh"):
    """Standalone GELU activation (no gate tensor).

    Args:
        x: (n_rows, n_cols) tensor, any dtype supported by Triton.
        approximate: "tanh" (default) or "none" for exact erf variant.

    Returns:
        Same-shape, same-dtype output tensor.
    """
    if approximate == "tanh":
        _ensure_gelu_tanh()
        return _launch_gelu(_gelu_tanh_kernel, x)
    else:
        _ensure_gelu_exact()
        return _launch_gelu(_gelu_exact_kernel, x)
