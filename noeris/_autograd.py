"""torch.autograd.Function wrappers for Triton kernels.

These connect the Triton kernels to PyTorch's autograd graph so they work
during training (loss.backward() propagates through them).
"""

from __future__ import annotations

import torch


class TritonRMSNorm(torch.autograd.Function):
    """Triton-accelerated RMSNorm with PyTorch backward."""

    @staticmethod
    def forward(ctx, x, weight, eps, affine_mode):
        from .kernels.rmsnorm import rmsnorm_forward

        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.affine_mode = affine_mode
        return rmsnorm_forward(x, weight, eps=eps, affine_mode=affine_mode)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        affine_mode = ctx.affine_mode

        # Standard RMSNorm backward in PyTorch
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        x_norm = x_f * rstd

        if affine_mode == 1:
            w_eff = (1.0 + weight.float())
        else:
            w_eff = weight.float()

        grad_f = grad_output.float()

        # dy/dx through norm
        dx_norm = grad_f * w_eff
        # RMSNorm backward: dx = (dx_norm - x_norm * mean(dx_norm * x_norm)) * rstd
        n_cols = x.shape[-1]
        dot = (dx_norm * x_norm).sum(-1, keepdim=True) / n_cols
        dx = (dx_norm - x_norm * dot) * rstd

        # dw
        if affine_mode == 1:
            dw = (grad_f * x_norm).sum(dim=0)
        else:
            dw = (grad_f * x_norm).sum(dim=0)

        return dx.to(x.dtype), dw.to(weight.dtype), None, None


class TritonQKNormRoPE(torch.autograd.Function):
    """Fused QK-RMSNorm+RoPE with Triton forward AND backward."""

    @staticmethod
    def forward(ctx, q, k, cos, sin, q_scale, k_scale, eps, affine_mode):
        from .kernels.qk_norm_rope import qk_norm_rope_forward

        # Save inputs for backward recomputation (FlashAttention-style)
        ctx.save_for_backward(q, k, cos, sin, q_scale, k_scale)
        ctx.eps = eps
        ctx.affine_mode = affine_mode
        return qk_norm_rope_forward(
            q, k, cos, sin, q_scale, k_scale,
            eps=eps, affine_mode=affine_mode,
        )

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        from .kernels.qk_norm_rope import qk_norm_rope_backward

        q, k, cos, sin, q_scale, k_scale = ctx.saved_tensors
        dq, dk, dq_scale, dk_scale = qk_norm_rope_backward(
            dq_out, dk_out, q, k, cos, sin, q_scale, k_scale,
            eps=ctx.eps, affine_mode=ctx.affine_mode,
        )
        # Return grads for: q, k, cos, sin, q_scale, k_scale, eps, affine_mode
        return dq, dk, None, None, dq_scale, dk_scale, None, None


class TritonGeGLU(torch.autograd.Function):
    """Triton-accelerated GeGLU with PyTorch backward."""

    @staticmethod
    def forward(ctx, gate, up):
        from .kernels.geglu import geglu_forward

        ctx.save_for_backward(gate, up)
        return geglu_forward(gate, up)

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        # GeGLU: out = gate * gelu(up)
        # d_gate = gelu(up) * grad_output
        # d_up = gate * gelu'(up) * grad_output
        gelu_up = torch.nn.functional.gelu(up.float(), approximate="tanh")
        d_gate = (gelu_up * grad_output.float()).to(gate.dtype)

        # GELU' with tanh approximation
        import math
        x = up.float()
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        tanh_inner = torch.tanh(inner)
        # d_gelu = 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh^2) * sqrt(2/pi) * (1 + 3*coeff*x^2)
        d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner * tanh_inner) * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
        d_up = (gate.float() * d_gelu * grad_output.float()).to(up.dtype)

        return d_gate, d_up


class TritonSwiGLU(torch.autograd.Function):
    """Triton-accelerated SwiGLU with PyTorch backward."""

    @staticmethod
    def forward(ctx, gate, up):
        from .kernels.geglu import swiglu_forward

        ctx.save_for_backward(gate, up)
        return swiglu_forward(gate, up)

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        # SwiGLU: out = gate * silu(up) = gate * up * sigmoid(up)
        up_f = up.float()
        sig = torch.sigmoid(up_f)
        silu_up = up_f * sig

        d_gate = (silu_up * grad_output.float()).to(gate.dtype)
        # d_silu = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        d_silu = sig * (1.0 + up_f * (1.0 - sig))
        d_up = (gate.float() * d_silu * grad_output.float()).to(up.dtype)

        return d_gate, d_up


class TritonGELU(torch.autograd.Function):
    """Triton-accelerated standalone GELU with PyTorch backward."""

    @staticmethod
    def forward(ctx, x, approximate="tanh"):
        from .kernels.geglu import gelu_forward

        ctx.save_for_backward(x)
        ctx.approximate = approximate
        return gelu_forward(x, approximate=approximate)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        approximate = ctx.approximate

        if approximate == "tanh":
            # GELU'(x) with tanh approximation
            import math
            xf = x.float()
            sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
            coeff = 0.044715
            inner = sqrt_2_over_pi * (xf + coeff * xf * xf * xf)
            tanh_inner = torch.tanh(inner)
            d_gelu = (
                0.5 * (1.0 + tanh_inner)
                + 0.5 * xf * (1.0 - tanh_inner * tanh_inner)
                * sqrt_2_over_pi * (1.0 + 3.0 * coeff * xf * xf)
            )
        else:
            # Exact GELU'(x) = 0.5*(1+erf(x/sqrt2)) + x * exp(-x^2/2) / sqrt(2*pi)
            import math
            xf = x.float()
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
            d_gelu = (
                0.5 * (1.0 + torch.erf(xf * inv_sqrt2))
                + xf * torch.exp(-0.5 * xf * xf) * inv_sqrt_2pi
            )

        dx = (d_gelu * grad_output.float()).to(x.dtype)
        return dx, None  # None for the `approximate` arg
