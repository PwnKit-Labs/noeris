"""Fused QK-RMSNorm + RoPE kernel (forward and backward).

Fuses RMSNorm (with Gemma-mode (1+w) affine) and split-pair RoPE into a
single kernel launch per tensor. The backward pass recomputes forward
intermediates (FlashAttention-style) to avoid doubling activation memory.
"""

from __future__ import annotations

_fwd_kernel = None
_bwd_kernel = None


def _ensure_fwd():
    global _fwd_kernel
    if _fwd_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _qk_norm_rope_fwd(
        x_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
        row_stride, heads, seq_len, head_dim, eps,
        BLOCK_SIZE: tl.constexpr,
        AFFINE_MODE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s_idx = pid % seq_len
        x_base = x_ptr + pid * row_stride
        out_base = out_ptr + pid * row_stride
        half = head_dim // 2
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < half

        x_even = tl.load(x_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

        sum_sq = tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)
        mean_sq = sum_sq / head_dim
        rstd = 1.0 / tl.sqrt(mean_sq + eps)

        s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

        if AFFINE_MODE == 0:
            n_even = x_even * rstd * s_even
            n_odd = x_odd * rstd * s_odd
        else:
            n_even = x_even * rstd * (1.0 + s_even)
            n_odd = x_odd * rstd * (1.0 + s_odd)

        cos_row = cos_ptr + s_idx * half
        sin_row = sin_ptr + s_idx * half
        c = tl.load(cos_row + offs, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sin_row + offs, mask=mask, other=0.0).to(tl.float32)

        out_even = n_even * c - n_odd * sn
        out_odd = n_even * sn + n_odd * c

        tl.store(out_base + 2 * offs, out_even.to(tl.float16), mask=mask)
        tl.store(out_base + 2 * offs + 1, out_odd.to(tl.float16), mask=mask)

    _fwd_kernel = _qk_norm_rope_fwd


def _ensure_bwd():
    global _bwd_kernel
    if _bwd_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _qk_norm_rope_bwd(
        x_ptr, scale_ptr, cos_ptr, sin_ptr,
        dout_ptr, dx_ptr, dscale_ptr,
        row_stride, heads, seq_len, head_dim, eps,
        BLOCK_SIZE: tl.constexpr,
        AFFINE_MODE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s_idx = pid % seq_len
        x_base = x_ptr + pid * row_stride
        dout_base = dout_ptr + pid * row_stride
        dx_base = dx_ptr + pid * row_stride
        half = head_dim // 2
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < half

        # Recompute forward intermediates
        x_even = tl.load(x_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)
        mean_sq = sum_sq / head_dim
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        x_norm_even = x_even * rstd
        x_norm_odd = x_odd * rstd

        s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

        cos_row = cos_ptr + s_idx * half
        sin_row = sin_ptr + s_idx * half
        c = tl.load(cos_row + offs, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sin_row + offs, mask=mask, other=0.0).to(tl.float32)

        # Load upstream gradient
        dout_even = tl.load(dout_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        dout_odd = tl.load(dout_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

        # Undo RoPE (inverse rotation)
        dx_scaled_even = dout_even * c + dout_odd * sn
        dx_scaled_odd = -dout_even * sn + dout_odd * c

        # Undo affine -> dscale
        if AFFINE_MODE == 0:
            dx_norm_even = dx_scaled_even * s_even
            dx_norm_odd = dx_scaled_odd * s_odd
        else:
            dx_norm_even = dx_scaled_even * (1.0 + s_even)
            dx_norm_odd = dx_scaled_odd * (1.0 + s_odd)

        dscale_local_even = dx_scaled_even * x_norm_even
        dscale_local_odd = dx_scaled_odd * x_norm_odd
        tl.atomic_add(dscale_ptr + 2 * offs, dscale_local_even, mask=mask)
        tl.atomic_add(dscale_ptr + 2 * offs + 1, dscale_local_odd, mask=mask)

        # RMSNorm backward
        dot_even = tl.sum(dx_norm_even * x_norm_even, axis=0)
        dot_odd = tl.sum(dx_norm_odd * x_norm_odd, axis=0)
        dot_prod = (dot_even + dot_odd) / head_dim

        dx_even = (dx_norm_even - x_norm_even * dot_prod) * rstd
        dx_odd = (dx_norm_odd - x_norm_odd * dot_prod) * rstd

        tl.store(dx_base + 2 * offs, dx_even.to(tl.float16), mask=mask)
        tl.store(dx_base + 2 * offs + 1, dx_odd.to(tl.float16), mask=mask)

    _bwd_kernel = _qk_norm_rope_bwd


def qk_norm_rope_forward(q, k, cos, sin, q_scale, k_scale, eps=1e-6, affine_mode=1):
    """Fused QK-RMSNorm+RoPE forward.

    Args:
        q: (B, H, S, D) fp16.
        k: (B, H_kv, S, D) fp16.
        cos: (S, D/2) fp32.
        sin: (S, D/2) fp32.
        q_scale: (D,) fp32 learnable scale.
        k_scale: (D,) fp32 learnable scale.
        affine_mode: 0 = standard (y = x*rstd*w), 1 = Gemma (y = x*rstd*(1+w)).
    """
    import torch
    import triton

    _ensure_fwd()
    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    half = D // 2
    BLOCK_SIZE = triton.next_power_of_2(half)
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    _fwd_kernel[(B * H * S,)](
        q.reshape(B * H * S, D).contiguous(), q_scale, cos_c, sin_c,
        q_out.reshape(B * H * S, D),
        D, H, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE, AFFINE_MODE=affine_mode,
        num_warps=min(8, max(1, BLOCK_SIZE // 32)),
        num_stages=1,
    )
    _fwd_kernel[(B * H_kv * S,)](
        k.reshape(B * H_kv * S, D).contiguous(), k_scale, cos_c, sin_c,
        k_out.reshape(B * H_kv * S, D),
        D, H_kv, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE, AFFINE_MODE=affine_mode,
        num_warps=min(8, max(1, BLOCK_SIZE // 32)),
        num_stages=1,
    )
    return q_out, k_out


def qk_norm_rope_backward(dout_q, dout_k, q, k, cos, sin, q_scale, k_scale, eps=1e-6, affine_mode=1):
    """Fused QK-RMSNorm+RoPE backward. Recomputes forward intermediates.

    Returns (dq, dk, dq_scale, dk_scale).
    """
    import torch
    import triton

    _ensure_bwd()
    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dq_scale = torch.zeros((D,), device=q.device, dtype=torch.float32)
    dk_scale = torch.zeros((D,), device=k.device, dtype=torch.float32)

    half = D // 2
    BLOCK_SIZE = triton.next_power_of_2(half)
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    _bwd_kernel[(B * H * S,)](
        q.reshape(B * H * S, D).contiguous(), q_scale, cos_c, sin_c,
        dout_q.reshape(B * H * S, D).contiguous(),
        dq.reshape(B * H * S, D), dq_scale,
        D, H, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE, AFFINE_MODE=affine_mode,
        num_warps=min(8, max(1, BLOCK_SIZE // 32)),
        num_stages=1,
    )
    _bwd_kernel[(B * H_kv * S,)](
        k.reshape(B * H_kv * S, D).contiguous(), k_scale, cos_c, sin_c,
        dout_k.reshape(B * H_kv * S, D).contiguous(),
        dk.reshape(B * H_kv * S, D), dk_scale,
        D, H_kv, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE, AFFINE_MODE=affine_mode,
        num_warps=min(8, max(1, BLOCK_SIZE // 32)),
        num_stages=1,
    )
    return dq, dk, dq_scale, dk_scale
