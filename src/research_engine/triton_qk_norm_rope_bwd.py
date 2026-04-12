"""Backward pass for fused QK-RMSNorm + RoPE (Gemma 3/4 prologue).

Complements the forward kernel in ``triton_qk_norm_rope.py``. Together they
turn the 10-13x inference fusion into a TRAINING-time optimization — no
framework fuses the Gemma prologue backward pass.

The backward kernel:
1. **Recomputes** forward intermediates (x_norm, rstd) from the original input
   x rather than saving them — this follows FlashAttention's recomputation
   philosophy and avoids doubling activation memory.
2. Inverts the RoPE rotation on the upstream gradient (dout).
3. Computes dscale via atomic accumulation across all (B, H, S) rows.
4. Propagates through the RMSNorm backward: the standard formula
   ``dx = (dx_norm - x_norm * mean(dx_norm * x_norm)) * rstd``.

dscale accumulation uses **tl.atomic_add** — at Gemma 4 shapes
(4096 seq x 32 heads = 131k programs) the atomics serialize somewhat but the
overall kernel is still faster than 4 separate backward launches.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator
from .triton_qk_norm_rope import (
    QK_NORM_ROPE_CURATED_CONFIGS,
    QK_NORM_ROPE_PARAM_SPACE,
    QK_NORM_ROPE_SHAPE_BUCKETS,
    generate_qk_norm_rope_grid,
    qk_norm_rope_config_id,
    qk_norm_rope_shape_bucket_key,
    qk_norm_rope_shared_memory_check,
)


def generate_qk_norm_rope_bwd_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained backward-pass benchmark for fused QK-RMSNorm+RoPE."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated fused QK-RMSNorm+RoPE BACKWARD benchmark (Gemma 3/4 prologue)."""

import json
import platform

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


# ---------------------------------------------------------------------------
# Forward kernel (needed for combined forward+backward benchmarking)
# ---------------------------------------------------------------------------

@triton.jit
def qk_norm_rope_kernel(
    x_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
    row_stride,
    heads, seq_len, head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm (Gemma-mode affine) + split-pair RoPE for one row."""
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


# ---------------------------------------------------------------------------
# Backward kernel — fused training prologue (no known prior art for backward)
# ---------------------------------------------------------------------------

@triton.jit
def qk_norm_rope_bwd_kernel(
    # Forward inputs (recompute forward intermediates from x)
    x_ptr, scale_ptr, cos_ptr, sin_ptr,
    # Upstream gradient
    dout_ptr,
    # Output gradients
    dx_ptr, dscale_ptr,
    # Strides and dims
    row_stride,
    heads, seq_len, head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for fused RMSNorm (Gemma-mode affine) + split-pair RoPE.

    Recomputes forward intermediates (x_norm, rstd) from the original input x
    rather than loading saved activations — matches FlashAttention recomputation.

    dscale accumulation uses atomic_add across all (B, H, S) rows.
    """
    pid = tl.program_id(0)
    s_idx = pid % seq_len

    x_base = x_ptr + pid * row_stride
    dout_base = dout_ptr + pid * row_stride
    dx_base = dx_ptr + pid * row_stride

    half = head_dim // 2
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < half

    # ---- Recompute forward intermediates ----
    x_even = tl.load(x_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)
    mean_sq = sum_sq / head_dim
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # x_norm (recomputed, not loaded from saved)
    x_norm_even = x_even * rstd
    x_norm_odd = x_odd * rstd

    # Load scale
    s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin for RoPE inverse rotation
    cos_row = cos_ptr + s_idx * half
    sin_row = sin_ptr + s_idx * half
    c = tl.load(cos_row + offs, mask=mask, other=1.0).to(tl.float32)
    sn = tl.load(sin_row + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- Load upstream gradient (dout) ----
    dout_even = tl.load(dout_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    dout_odd = tl.load(dout_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

    # ---- Undo RoPE (inverse rotation) ----
    dx_scaled_even = dout_even * c + dout_odd * sn
    dx_scaled_odd = -dout_even * sn + dout_odd * c

    # ---- Undo affine: dx_norm = dx_scaled * (1 + scale) ----
    # dscale contribution: dx_scaled * x_norm (accumulated across rows)
    dx_norm_even = dx_scaled_even * (1.0 + s_even)
    dx_norm_odd = dx_scaled_odd * (1.0 + s_odd)

    # Accumulate dscale via atomic_add
    dscale_local_even = dx_scaled_even * x_norm_even
    dscale_local_odd = dx_scaled_odd * x_norm_odd
    tl.atomic_add(dscale_ptr + 2 * offs, dscale_local_even, mask=mask)
    tl.atomic_add(dscale_ptr + 2 * offs + 1, dscale_local_odd, mask=mask)

    # ---- RMSNorm backward ----
    # dx = (dx_norm - x_norm * mean(dx_norm * x_norm)) * rstd
    dot_even = tl.sum(dx_norm_even * x_norm_even, axis=0)
    dot_odd = tl.sum(dx_norm_odd * x_norm_odd, axis=0)
    dot_prod = (dot_even + dot_odd) / head_dim

    dx_even = (dx_norm_even - x_norm_even * dot_prod) * rstd
    dx_odd = (dx_norm_odd - x_norm_odd * dot_prod) * rstd

    # ---- Store dx ----
    tl.store(dx_base + 2 * offs, dx_even.to(tl.float16), mask=mask)
    tl.store(dx_base + 2 * offs + 1, dx_odd.to(tl.float16), mask=mask)


def apply_qk_norm_rope_forward(q, k, cos, sin, q_scale, k_scale, config, eps=1e-6):
    """Launch forward kernel for Q and K."""
    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_flat = q.reshape(B * H * S, D).contiguous()
    q_out_flat = q_out.reshape(B * H * S, D)
    k_flat = k.reshape(B * H_kv * S, D).contiguous()
    k_out_flat = k_out.reshape(B * H_kv * S, D)

    half = D // 2
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(half))
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    grid_q = (B * H * S,)
    qk_norm_rope_kernel[grid_q](
        q_flat, q_scale, cos_c, sin_c, q_out_flat,
        D, H, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    grid_k = (B * H_kv * S,)
    qk_norm_rope_kernel[grid_k](
        k_flat, k_scale, cos_c, sin_c, k_out_flat,
        D, H_kv, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return q_out, k_out


def apply_qk_norm_rope_backward(dout_q, dout_k, q, k, cos, sin, q_scale, k_scale, config, eps=1e-6):
    """Launch backward kernel for Q and K.

    Returns (dq, dk, dq_scale, dk_scale).
    """
    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dq_scale = torch.zeros((D,), device=q.device, dtype=torch.float32)
    dk_scale = torch.zeros((D,), device=k.device, dtype=torch.float32)

    q_flat = q.reshape(B * H * S, D).contiguous()
    dout_q_flat = dout_q.reshape(B * H * S, D).contiguous()
    dq_flat = dq.reshape(B * H * S, D)

    k_flat = k.reshape(B * H_kv * S, D).contiguous()
    dout_k_flat = dout_k.reshape(B * H_kv * S, D).contiguous()
    dk_flat = dk.reshape(B * H_kv * S, D)

    half = D // 2
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(half))
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    # Q backward
    grid_q = (B * H * S,)
    qk_norm_rope_bwd_kernel[grid_q](
        q_flat, q_scale, cos_c, sin_c,
        dout_q_flat,
        dq_flat, dq_scale,
        D, H, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    # K backward (GQA-aware)
    grid_k = (B * H_kv * S,)
    qk_norm_rope_bwd_kernel[grid_k](
        k_flat, k_scale, cos_c, sin_c,
        dout_k_flat,
        dk_flat, dk_scale,
        D, H_kv, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    return dq, dk, dq_scale, dk_scale


# ---------------------------------------------------------------------------
# PyTorch reference backward (for correctness checking)
# ---------------------------------------------------------------------------

def torch_qk_norm_rope_backward(dout_q, dout_k, q, k, cos, sin, q_scale, k_scale, eps=1e-6):
    """Compute gradients for the fused prologue — pure PyTorch reference."""
    D = q.shape[-1]

    # --- Q backward ---
    q_f = q.float()
    q_var = q_f.pow(2).mean(-1, keepdim=True)
    q_rstd = torch.rsqrt(q_var + eps)
    q_norm = q_f * q_rstd

    # Undo RoPE on dout
    c = cos[None, None, :, :]
    sn = sin[None, None, :, :]
    dq_even = dout_q[..., 0::2].float() * c + dout_q[..., 1::2].float() * sn
    dq_odd = -dout_q[..., 0::2].float() * sn + dout_q[..., 1::2].float() * c
    dq_scaled = torch.stack([dq_even, dq_odd], dim=-1).reshape(q.shape).float()

    # dscale
    dq_scale = (dq_scaled * q_norm).sum(dim=(0, 1, 2))

    # dx through RMSNorm backward
    dq_norm = dq_scaled * (1 + q_scale.float())
    dot = (dq_norm * q_norm).sum(-1, keepdim=True) / D
    dq = (dq_norm - q_norm * dot) * q_rstd

    # --- K backward ---
    k_f = k.float()
    k_var = k_f.pow(2).mean(-1, keepdim=True)
    k_rstd = torch.rsqrt(k_var + eps)
    k_norm = k_f * k_rstd

    dk_even = dout_k[..., 0::2].float() * c + dout_k[..., 1::2].float() * sn
    dk_odd = -dout_k[..., 0::2].float() * sn + dout_k[..., 1::2].float() * c
    dk_scaled = torch.stack([dk_even, dk_odd], dim=-1).reshape(k.shape).float()

    dk_scale = (dk_scaled * k_norm).sum(dim=(0, 1, 2))

    dk_norm = dk_scaled * (1 + k_scale.float())
    dot_k = (dk_norm * k_norm).sum(-1, keepdim=True) / D
    dk = (dk_norm - k_norm * dot_k) * k_rstd

    return dq.half(), dk.half(), dq_scale.float(), dk_scale.float()


# ---------------------------------------------------------------------------
# Separated backward baseline (4 separate autograd ops, no fusion)
# ---------------------------------------------------------------------------

def separated_backward_baseline(dout_q, dout_k, q, k, cos, sin, q_scale, k_scale, eps=1e-6):
    """vLLM-style separated backward: 4 distinct backward ops, no fusion.

    1. Q-RoPE backward
    2. K-RoPE backward
    3. Q-RMSNorm backward
    4. K-RMSNorm backward
    """
    D = q.shape[-1]
    c = cos[None, None, :, :]
    sn = sin[None, None, :, :]

    # 1. Q-RoPE backward
    dq_even = dout_q[..., 0::2].float() * c + dout_q[..., 1::2].float() * sn
    dq_odd = -dout_q[..., 0::2].float() * sn + dout_q[..., 1::2].float() * c
    dq_scaled = torch.stack([dq_even, dq_odd], dim=-1).reshape(q.shape).float()

    # 2. K-RoPE backward
    dk_even = dout_k[..., 0::2].float() * c + dout_k[..., 1::2].float() * sn
    dk_odd = -dout_k[..., 0::2].float() * sn + dout_k[..., 1::2].float() * c
    dk_scaled = torch.stack([dk_even, dk_odd], dim=-1).reshape(k.shape).float()

    # 3. Q-RMSNorm backward
    q_f = q.float()
    q_var = q_f.pow(2).mean(-1, keepdim=True)
    q_rstd = torch.rsqrt(q_var + eps)
    q_norm = q_f * q_rstd
    dq_scale = (dq_scaled * q_norm).sum(dim=(0, 1, 2))
    dq_norm = dq_scaled * (1 + q_scale.float())
    dot_q = (dq_norm * q_norm).sum(-1, keepdim=True) / D
    dq = (dq_norm - q_norm * dot_q) * q_rstd

    # 4. K-RMSNorm backward
    k_f = k.float()
    k_var = k_f.pow(2).mean(-1, keepdim=True)
    k_rstd = torch.rsqrt(k_var + eps)
    k_norm = k_f * k_rstd
    dk_scale = (dk_scaled * k_norm).sum(dim=(0, 1, 2))
    dk_norm = dk_scaled * (1 + k_scale.float())
    dot_k = (dk_norm * k_norm).sum(-1, keepdim=True) / D
    dk = (dk_norm - k_norm * dot_k) * k_rstd

    return dq.half(), dk.half(), dq_scale.float(), dk_scale.float()


# ---------------------------------------------------------------------------
# Benchmark function — runs forward+backward as a unit
# ---------------------------------------------------------------------------

def benchmark_one(batch, heads, num_kv_heads, seq, head_dim, config):
    try:
        q = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, num_kv_heads, seq, head_dim), device="cuda", dtype=torch.float16)
        cos = torch.randn((seq, head_dim // 2), device="cuda", dtype=torch.float32)
        sin = torch.randn((seq, head_dim // 2), device="cuda", dtype=torch.float32)
        q_scale = torch.randn((head_dim,), device="cuda", dtype=torch.float32) * 0.1
        k_scale = torch.randn((head_dim,), device="cuda", dtype=torch.float32) * 0.1
        dout_q = torch.randn_like(q)
        dout_k = torch.randn_like(k)

        # Correctness: compare fused backward vs PyTorch reference
        dq_ref, dk_ref, dqs_ref, dks_ref = torch_qk_norm_rope_backward(
            dout_q, dout_k, q, k, cos, sin, q_scale, k_scale,
        )
        dq_out, dk_out, dqs_out, dks_out = apply_qk_norm_rope_backward(
            dout_q, dout_k, q, k, cos, sin, q_scale, k_scale, config,
        )

        dq_err = (dq_out - dq_ref).abs().max().item()
        dk_err = (dk_out - dk_ref).abs().max().item()
        dqs_err = (dqs_out - dqs_ref).abs().max().item()
        dks_err = (dks_out - dks_ref).abs().max().item()
        max_err = max(dq_err, dk_err, dqs_err, dks_err)
        if max_err > 0.5:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        # Time fused forward+backward
        def fused_fwd_bwd():
            apply_qk_norm_rope_forward(q, k, cos, sin, q_scale, k_scale, config)
            apply_qk_norm_rope_backward(dout_q, dout_k, q, k, cos, sin, q_scale, k_scale, config)

        ms = triton.testing.do_bench(fused_fwd_bwd, warmup=25, rep=100)

        # Time separated forward+backward (4 separate ops each direction)
        def sep_fwd_bwd():
            # Forward (separated)
            D = q.shape[-1]
            q_var = q.float().pow(2).mean(-1, keepdim=True)
            q_normed = (q.float() * torch.rsqrt(q_var + 1e-6)).to(torch.float16) * (1.0 + q_scale).to(torch.float16)
            k_var = k.float().pow(2).mean(-1, keepdim=True)
            k_normed = (k.float() * torch.rsqrt(k_var + 1e-6)).to(torch.float16) * (1.0 + k_scale).to(torch.float16)
            c_b = cos[None, None, :, :].to(torch.float16)
            sn_b = sin[None, None, :, :].to(torch.float16)
            q_out = torch.stack([q_normed[..., 0::2] * c_b - q_normed[..., 1::2] * sn_b,
                                 q_normed[..., 0::2] * sn_b + q_normed[..., 1::2] * c_b], dim=-1).reshape(q.shape)
            k_out = torch.stack([k_normed[..., 0::2] * c_b - k_normed[..., 1::2] * sn_b,
                                 k_normed[..., 0::2] * sn_b + k_normed[..., 1::2] * c_b], dim=-1).reshape(k.shape)
            # Backward (separated)
            separated_backward_baseline(dout_q, dout_k, q, k, cos, sin, q_scale, k_scale)

        sep_ms = triton.testing.do_bench(sep_fwd_bwd, warmup=25, rep=100)

        # Memory bandwidth (backward reads x + dout, writes dx + dscale atomics)
        q_bytes = batch * heads * seq * head_dim * 2
        k_bytes = batch * num_kv_heads * seq * head_dim * 2
        trig_bytes = 2 * seq * (head_dim // 2) * 4
        scale_bytes = 2 * head_dim * 4
        # Backward reads: x, dout, scale, cos, sin; writes: dx, dscale (atomic)
        bwd_bytes = 2 * (q_bytes + k_bytes) + trig_bytes + scale_bytes  # reads
        bwd_bytes += 2 * (q_bytes + k_bytes) + 2 * head_dim * 4  # writes
        # Total = fwd + bwd
        fwd_bytes = 2 * (q_bytes + k_bytes) + trig_bytes + scale_bytes
        bytes_moved = fwd_bytes + bwd_bytes
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        backward_fusion_speedup = sep_ms / ms if ms > 0 else 0.0
        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "separated_ms": round(sep_ms, 4),
            "backward_fusion_speedup": round(backward_fusion_speedup, 3),
            "gb_per_s": round(gb_per_s, 2),
            "tflops": round(gb_per_s, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None, "tflops": None}}


def main():
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bs{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            batch = shape["batch"]
            heads = shape["heads"]
            num_kv_heads = shape["num_kv_heads"]
            seq = shape["seq"]
            head_dim = shape["head_dim"]
            result = benchmark_one(batch, heads, num_kv_heads, seq, head_dim, config)
            result["shape"] = f"{{batch}}x{{heads}}x{{num_kv_heads}}x{{seq}}x{{head_dim}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "qk_norm_rope_bwd",
        "hardware": {{
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        }},
        "configs_tested": len(configs),
        "config_results": all_results,
    }}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
'''


QK_NORM_ROPE_BWD_SPEC = register_operator(TritonOperatorSpec(
    name="qk_norm_rope_bwd",
    param_space=QK_NORM_ROPE_PARAM_SPACE,
    curated_configs=QK_NORM_ROPE_CURATED_CONFIGS,
    shape_buckets=QK_NORM_ROPE_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=qk_norm_rope_config_id,
    shape_bucket_fn=qk_norm_rope_shape_bucket_key,
    benchmark_script_fn=generate_qk_norm_rope_bwd_benchmark_script,
    grid_generator_fn=generate_qk_norm_rope_grid,
    shared_memory_check_fn=qk_norm_rope_shared_memory_check,
    description=(
        "Backward pass for fused Gemma 3/4 attention prologue: QK-RMSNorm "
        "(1+scale affine) + split-pair RoPE. Recomputes forward intermediates "
        "(FlashAttention-style), uses atomic_add for dscale accumulation. "
        "To our knowledge, no existing framework fuses the full Gemma "
        "prologue backward pass (existing fusions are inference-only)."
    ),
))
