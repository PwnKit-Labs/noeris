"""torch.autograd.Function wrappers for Noeris fused Triton kernels.

Enables training (backward pass) through the fused kernels, not just inference.
Works with HF Trainer, mixed precision (AMP), and gradient accumulation.

Strategy:
- FusedRMSNorm: forward = Triton kernel, backward = PyTorch autograd (v1).
- FusedQKNormRoPE: forward = Triton kernel, backward = fused Triton backward kernel.
- FusedGeGLU: forward = Triton kernel, backward = PyTorch autograd (v1).
"""

from __future__ import annotations

import torch
from torch.autograd import Function


class FusedRMSNorm(Function):
    """Fused RMSNorm with Triton forward, PyTorch backward (v1 safe fallback)."""

    @staticmethod
    def forward(ctx, x, weight, eps, affine_mode, config):
        from .triton_rmsnorm import rmsnorm

        # x arrives as (n_rows, hidden_dim) fp16; weight as (hidden_dim,) fp16
        y = rmsnorm(x, weight, config=config, eps=eps, affine_mode=affine_mode)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.affine_mode = affine_mode
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        affine_mode = ctx.affine_mode

        # Recompute forward in fp32 for stable gradients
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        x_norm = x_f * rstd  # (n_rows, D)

        if affine_mode == 1:
            w_eff = (1.0 + weight.float())
        else:
            w_eff = weight.float()

        # grad through y = x_norm * w_eff
        # dy/dx_norm = w_eff, dy/dw = x_norm
        grad_f = grad_output.float()

        # dweight
        grad_weight = (grad_f * x_norm).sum(dim=0)
        if affine_mode == 1:
            # w_eff = 1+w, so dL/dw = dL/dw_eff = grad_weight (chain rule passes through)
            pass

        # dx through RMSNorm: standard formula
        dx_norm = grad_f * w_eff  # (n_rows, D)
        D = x.shape[-1]
        dot = (dx_norm * x_norm).sum(-1, keepdim=True) / D
        dx = (dx_norm - x_norm * dot) * rstd

        return dx.to(x.dtype), grad_weight.to(weight.dtype), None, None, None


class FusedQKNormRoPE(Function):
    """Fused QK-RMSNorm+RoPE with Triton forward AND Triton backward."""

    @staticmethod
    def forward(ctx, q, k, cos, sin, q_scale, k_scale, eps, config):
        from .triton_qk_norm_rope import apply_qk_norm_rope

        q_out, k_out = apply_qk_norm_rope(
            q, k, cos, sin, q_scale, k_scale, config=config, eps=eps,
        )
        ctx.save_for_backward(q, k, cos, sin, q_scale, k_scale)
        ctx.eps = eps
        ctx.config = config
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        from .triton_qk_norm_rope_bwd import apply_qk_norm_rope_bwd

        q, k, cos, sin, q_scale, k_scale = ctx.saved_tensors
        dq, dk, dq_scale, dk_scale = apply_qk_norm_rope_bwd(
            grad_q, grad_k, q, k, cos, sin, q_scale, k_scale,
            config=ctx.config, eps=ctx.eps,
        )
        # Returns: dq, dk, d_cos, d_sin, dq_scale, dk_scale, d_eps, d_config
        return dq, dk, None, None, dq_scale, dk_scale, None, None


class FusedGeGLU(Function):
    """Fused GeGLU with Triton forward, PyTorch backward (v1 safe fallback)."""

    @staticmethod
    def forward(ctx, gate, up, config):
        from .triton_geglu import geglu

        # gate, up are (n_rows, ffn_dim) fp16
        out = geglu(gate, up, config=config)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors

        # out = gate * gelu_tanh(up)
        # d_gate = grad * gelu_tanh(up)
        # d_up = grad * gate * gelu_tanh'(up)
        gate_f = gate.float()
        up_f = up.float()
        grad_f = grad_output.float()

        gelu_up = torch.nn.functional.gelu(up_f, approximate="tanh")

        d_gate = grad_f * gelu_up

        # gelu_tanh'(up) via autograd on a small local graph
        up_var = up_f.detach().requires_grad_(True)
        gelu_var = torch.nn.functional.gelu(up_var, approximate="tanh")
        # d_up = grad * gate * gelu'(up)
        # Use autograd to get gelu'(up)
        gelu_grad = torch.autograd.grad(
            gelu_var, up_var, grad_outputs=torch.ones_like(gelu_var),
            create_graph=False,
        )[0]
        d_up = grad_f * gate_f * gelu_grad

        return d_gate.to(gate.dtype), d_up.to(up.dtype), None


# ---------------------------------------------------------------------------
# Quick smoke test (requires CUDA)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .triton_rmsnorm import RMSNORM_CURATED_CONFIGS
    from .triton_geglu import GEGLU_CURATED_CONFIGS

    device = "cuda"
    print("=== FusedRMSNorm forward+backward ===")
    x = torch.randn(4, 128, device=device, dtype=torch.float16, requires_grad=True)
    w = torch.randn(128, device=device, dtype=torch.float16, requires_grad=True)
    y = FusedRMSNorm.apply(x, w, 1e-6, 0, RMSNORM_CURATED_CONFIGS[0])
    loss = y.sum()
    loss.backward()
    print(f"  x.grad shape: {x.grad.shape}, w.grad shape: {w.grad.shape}")
    print(f"  x.grad abs max: {x.grad.abs().max().item():.6f}")

    print("\n=== FusedGeGLU forward+backward ===")
    gate = torch.randn(4, 256, device=device, dtype=torch.float16, requires_grad=True)
    up = torch.randn(4, 256, device=device, dtype=torch.float16, requires_grad=True)
    out = FusedGeGLU.apply(gate, up, GEGLU_CURATED_CONFIGS[0])
    loss = out.sum()
    loss.backward()
    print(f"  gate.grad shape: {gate.grad.shape}, up.grad shape: {up.grad.shape}")

    print("\nAll smoke tests passed.")
