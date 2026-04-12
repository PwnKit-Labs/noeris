"""End-to-end Gemma 4 decoder layer benchmark: Noeris fused vs PyTorch separated.

Generates a self-contained GPU benchmark script that measures full decoder
layer latency using two paths:

  Path A (noeris_fused): Uses Noeris Triton kernels for RMSNorm, QK-RMSNorm+
    RoPE, FlashAttention (GQA + sliding-window), and GeGLU.

  Path B (pytorch_separated): Uses standard PyTorch ops (F.rms_norm, separate
    Q/K norm and RoPE, F.scaled_dot_product_attention, separate GELU + mul).

The headline metric is ``layer_speedup = pytorch_separated_ms / noeris_fused_ms``,
which predicts the real training/inference speedup from Noeris on Gemma 4
decoder layers.

Issue #51: co-optimizing across the full pipeline.
"""

from __future__ import annotations

import json
from typing import Any


GEMMA4_LAYER_CONFIGS = [
    # 31B Dense -- local layer (5 of 6)
    {
        "name": "gemma4_31b_local",
        "batch": 1,
        "seq_len": 2048,
        "hidden_dim": 5376,
        "num_heads": 32,
        "num_kv_heads": 16,
        "head_dim": 256,
        "ffn_dim": 21504,
        "window_size": 1024,
        "is_causal": True,
    },
    # 31B Dense -- global layer (1 of 6)
    {
        "name": "gemma4_31b_global",
        "batch": 1,
        "seq_len": 2048,
        "hidden_dim": 5376,
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 512,
        "ffn_dim": 21504,
        "window_size": -1,
        "is_causal": True,
    },
    # E2B -- local layer
    {
        "name": "gemma4_e2b_local",
        "batch": 1,
        "seq_len": 4096,
        "hidden_dim": 1536,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
        "ffn_dim": 6144,
        "window_size": 512,
        "is_causal": True,
    },
]


def generate_gemma4_layer_benchmark_script(
    configs: list[dict[str, Any]] | None = None,
) -> str:
    """Return a self-contained benchmark script for Gemma 4 decoder layers.

    The generated script is designed to run on Colab T4 or any CUDA device.
    It benchmarks both the Noeris fused path and the PyTorch separated path,
    reports per-step timings and the overall ``layer_speedup``.

    Args:
        configs: Layer configurations to benchmark.  Defaults to
            ``GEMMA4_LAYER_CONFIGS``.
    """
    if configs is None:
        configs = GEMMA4_LAYER_CONFIGS
    configs_json = json.dumps(configs)

    return f'''#!/usr/bin/env python3
"""Auto-generated Gemma 4 decoder layer benchmark — Noeris fused vs PyTorch separated.

Measures end-to-end latency of a full Gemma 4 decoder layer using:
  Path A: Noeris fused kernels (RMSNorm, QK-RMSNorm+RoPE, Attention, GeGLU)
  Path B: PyTorch separated ops (F.rms_norm, separate norm+rope, SDPA, GELU+mul)

Headline metric: layer_speedup = pytorch_separated_ms / noeris_fused_ms
"""

import json
import math
import platform
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


LAYER_CONFIGS = json.loads({configs_json!r})


# =============================================================================
# Triton kernels — Noeris fused operators
# =============================================================================

# --- RMSNorm (affine_mode=1: Gemma style y = x * rstd * (1 + w)) ---

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rstd * (1.0 + w)
    tl.store(out_ptr + row * n_cols + offs, out.to(tl.float16), mask=mask)


def noeris_rmsnorm(x, w, eps=1e-6):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](x, w, out, n_cols, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
                              num_warps=min(8, max(1, BLOCK_SIZE // 256)),
                              num_stages=1)
    return out


# --- Fused QK-RMSNorm + RoPE ---

@triton.jit
def qk_norm_rope_kernel(
    qk_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_b, stride_h, stride_s, stride_d,
    cos_stride_s, cos_stride_d,
    n_heads, seq_len, head_dim,
    eps: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    total_heads = tl.load(qk_ptr - qk_ptr)  # dummy; we use pid arithmetic
    # Decompose pid -> (batch, head, seq)
    bs_idx = pid // (n_heads * seq_len)
    rem = pid % (n_heads * seq_len)
    h_idx = rem // seq_len
    s_idx = rem % seq_len

    base = bs_idx * stride_b + h_idx * stride_h + s_idx * stride_s

    # Load even and odd halves
    offs = tl.arange(0, HALF_DIM)
    mask = offs < (head_dim // 2)

    x_even = tl.load(qk_ptr + base + offs * 2, mask=mask, other=0.0).to(tl.float32)
    x_odd  = tl.load(qk_ptr + base + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm over full head_dim: var = (sum_even^2 + sum_odd^2) / head_dim
    var = (tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)) / head_dim
    rstd = 1.0 / tl.sqrt(var + eps)

    # Affine: (1 + scale)
    s_even = tl.load(scale_ptr + offs * 2, mask=mask, other=0.0).to(tl.float32)
    s_odd  = tl.load(scale_ptr + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    x_even = x_even * rstd * (1.0 + s_even)
    x_odd  = x_odd  * rstd * (1.0 + s_odd)

    # RoPE rotation
    cos_val = tl.load(cos_ptr + s_idx * cos_stride_s + offs * cos_stride_d, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + s_idx * cos_stride_s + offs * cos_stride_d, mask=mask, other=0.0).to(tl.float32)

    out_even = x_even * cos_val - x_odd * sin_val
    out_odd  = x_even * sin_val + x_odd * cos_val

    tl.store(out_ptr + base + offs * 2,     out_even.to(tl.float16), mask=mask)
    tl.store(out_ptr + base + offs * 2 + 1, out_odd.to(tl.float16),  mask=mask)


def noeris_qk_norm_rope(qk, scale, cos, sin):
    """Fused QK-RMSNorm + RoPE for one tensor (Q or K)."""
    B, H, S, D = qk.shape
    out = torch.empty_like(qk)
    HALF_DIM = triton.next_power_of_2(D // 2)
    grid = (B * H * S,)

    qk_norm_rope_kernel[grid](
        qk, scale, cos, sin, out,
        qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
        cos.stride(0), cos.stride(1),
        H, S, D,
        eps=1e-6, HALF_DIM=HALF_DIM,
        num_warps=min(8, max(1, HALF_DIM // 32)),
        num_stages=1,
    )
    return out


# --- Fused GeGLU ---

@triton.jit
def geglu_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    gate = tl.load(gate_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(up_ptr   + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (up + coeff * up * up * up)
    gelu_up = 0.5 * up * (1.0 + tl.extra.libdevice.tanh(inner))
    out = gate * gelu_up

    tl.store(out_ptr + row * n_cols + offs, out.to(tl.float16), mask=mask)


def noeris_geglu(gate, up):
    n_rows, n_cols = gate.shape
    out = torch.empty_like(gate)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    geglu_kernel[(n_rows,)](gate, up, out, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                            num_warps=min(16, max(1, BLOCK_SIZE // 256)),
                            num_stages=1)
    return out


# --- FlashAttention (with GQA + sliding-window) ---

@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    seq_len, head_dim, num_kv_heads, scale,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // tl.load(Q_ptr - Q_ptr + Q_ptr.dtype.element_ty(0) + 1)  # placeholder
    # Simplified: compute via strides
    # We flatten batch*heads into pid_bh
    num_q_heads = stride_qb // stride_qh if stride_qh > 0 else 1
    q_head_idx = pid_bh % num_q_heads if num_q_heads > 0 else 0
    b_idx = pid_bh // num_q_heads if num_q_heads > 0 else 0
    kv_head_idx = q_head_idx * num_kv_heads // num_q_heads if num_q_heads > 0 else 0

    q_off = b_idx * stride_qb + q_head_idx * stride_qh
    k_off = b_idx * stride_kb + kv_head_idx * stride_kh
    v_off = b_idx * stride_vb + kv_head_idx * stride_vh
    o_off = b_idx * stride_ob + q_head_idx * stride_oh

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)

    q = tl.load(Q_ptr + q_off + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
                mask=(m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim), other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Determine key range
    if IS_CAUSAL:
        max_kn = tl.minimum(m_start + BLOCK_M, seq_len)
    else:
        max_kn = seq_len

    if WINDOW_SIZE > 0 and IS_CAUSAL:
        min_kn = tl.maximum(0, m_start - WINDOW_SIZE + 1)
    else:
        min_kn = 0

    n_start = (min_kn // BLOCK_N) * BLOCK_N

    for n_off in range(0, (max_kn + BLOCK_N - 1) // BLOCK_N * BLOCK_N, BLOCK_N):
        if n_off >= n_start:
            n_offs = n_off + tl.arange(0, BLOCK_N)
            k = tl.load(K_ptr + k_off + n_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd,
                        mask=(n_offs[None, :] < seq_len) & (d_offs[:, None] < head_dim), other=0.0)
            qk = tl.dot(q.to(tl.float16), k.to(tl.float16)).to(tl.float32) * scale

            # Masking
            if IS_CAUSAL:
                causal_mask = m_offs[:, None] >= n_offs[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            if WINDOW_SIZE > 0:
                window_mask = (m_offs[:, None] - n_offs[None, :]) < WINDOW_SIZE
                qk = tl.where(window_mask, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v = tl.load(V_ptr + v_off + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                        mask=(n_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim), other=0.0)
            acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
            m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O_ptr + o_off + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
             acc.to(tl.float16),
             mask=(m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim))


def noeris_attention(Q, K, V, num_kv_heads, window_size=-1, is_causal=True):
    """FlashAttention with GQA + optional sliding window."""
    B, H, S, D = Q.shape
    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    flash_attn_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        S, D, num_kv_heads, scale,
        WINDOW_SIZE=window_size if window_size > 0 else -1,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )
    return O


# =============================================================================
# Timing utilities
# =============================================================================

def cuda_event_timer(fn, warmup=5, trials=20):
    """Time a function using CUDA events. Returns median ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


# =============================================================================
# RoPE frequency helpers
# =============================================================================

def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    """Build cos/sin caches for RoPE: shape (seq_len, head_dim//2)."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(torch.float16), torch.sin(angles).to(torch.float16)


# =============================================================================
# PyTorch separated ops (Path B)
# =============================================================================

def pytorch_rmsnorm(x, w, eps=1e-6):
    """Standard RMSNorm: y = x * rstd * (1 + w) (Gemma affine mode)."""
    x32 = x.to(torch.float32)
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return ((x32 * rstd) * (1.0 + w.to(torch.float32))).to(torch.float16)


def pytorch_qk_norm_rope_separated(qk, scale, cos, sin):
    """4 separate ops: RMSNorm Q/K + RoPE Q/K."""
    B, H, S, D = qk.shape
    half = D // 2
    # RMSNorm
    x32 = qk.to(torch.float32)
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x32 * rstd * (1.0 + scale.to(torch.float32))).to(torch.float16)
    # RoPE
    x_even = normed[..., 0::2].to(torch.float32)
    x_odd  = normed[..., 1::2].to(torch.float32)
    cos_v = cos[:S, :half].to(torch.float32)
    sin_v = sin[:S, :half].to(torch.float32)
    out_even = x_even * cos_v - x_odd * sin_v
    out_odd  = x_even * sin_v + x_odd * cos_v
    out = torch.empty_like(normed)
    out[..., 0::2] = out_even.to(torch.float16)
    out[..., 1::2] = out_odd.to(torch.float16)
    return out


def pytorch_geglu_separated(gate, up):
    """gate * GELU_tanh(up) as 2 ops."""
    gelu_up = F.gelu(up.to(torch.float32), approximate="tanh").to(torch.float16)
    return gate * gelu_up


# =============================================================================
# GQA helpers
# =============================================================================

def expand_kv_for_gqa(kv, num_heads, num_kv_heads):
    """Expand KV from (B, Hkv, S, D) to (B, H, S, D) by repeating."""
    if num_kv_heads == num_heads:
        return kv
    repeat = num_heads // num_kv_heads
    B, Hkv, S, D = kv.shape
    return kv.unsqueeze(2).expand(B, Hkv, repeat, S, D).reshape(B, num_heads, S, D)


# =============================================================================
# Layer benchmark
# =============================================================================

def benchmark_layer(cfg):
    """Benchmark one Gemma 4 decoder layer config, returning per-step timings."""
    B     = cfg["batch"]
    S     = cfg["seq_len"]
    D     = cfg["hidden_dim"]
    H     = cfg["num_heads"]
    Hkv   = cfg["num_kv_heads"]
    Dh    = cfg["head_dim"]
    Dff   = cfg["ffn_dim"]
    ws    = cfg["window_size"]
    causal = cfg["is_causal"]
    name  = cfg["name"]
    device = "cuda"

    print(f"\\n=== {{name}} ===")
    print(f"  B={{B}} S={{S}} D={{D}} H={{H}} Hkv={{Hkv}} Dh={{Dh}} Dff={{Dff}} ws={{ws}}")

    # --- Allocate weights (random, fp16) ---
    hidden = torch.randn(B, S, D, device=device, dtype=torch.float16)
    w_pre_attn   = torch.randn(D, device=device, dtype=torch.float16)
    w_q          = torch.randn(D, H * Dh, device=device, dtype=torch.float16)
    w_k          = torch.randn(D, Hkv * Dh, device=device, dtype=torch.float16)
    w_v          = torch.randn(D, Hkv * Dh, device=device, dtype=torch.float16)
    q_scale      = torch.randn(Dh, device=device, dtype=torch.float16)
    k_scale      = torch.randn(Dh, device=device, dtype=torch.float16)
    cos, sin     = build_rope_cache(S, Dh, device)
    w_pre_mlp    = torch.randn(D, device=device, dtype=torch.float16)
    w_gate_up    = torch.randn(D, Dff * 2, device=device, dtype=torch.float16)
    w_down       = torch.randn(Dff, D, device=device, dtype=torch.float16)

    # =====================================================================
    # Path A: Noeris fused
    # =====================================================================

    noeris_step_times = {{}}

    def noeris_fused_layer(hidden_in):
        residual = hidden_in

        # Step 1: Pre-attention RMSNorm (Gemma affine_mode=1)
        h = noeris_rmsnorm(hidden_in.view(-1, D), w_pre_attn).view(B, S, D)

        # Step 2: QKV projection
        q = (h.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3)
        k = (h.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3)
        v = (h.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3)

        # Step 3: Fused QK-RMSNorm + RoPE (our headline kernel!)
        q = noeris_qk_norm_rope(q, q_scale, cos, sin)
        k = noeris_qk_norm_rope(k, k_scale, cos, sin)

        # Step 4: Attention (GQA + sliding-window)
        k_exp = expand_kv_for_gqa(k, H, Hkv)
        v_exp = expand_kv_for_gqa(v, H, Hkv)
        attn_out = noeris_attention(q, k_exp, v_exp, Hkv,
                                    window_size=ws, is_causal=causal)

        # Step 5: Post-attention residual
        h = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        h = residual + h

        # Step 6: Pre-MLP RMSNorm
        residual2 = h
        h = noeris_rmsnorm(h.view(-1, D), w_pre_mlp).view(B, S, D)

        # Step 7: GeGLU MLP
        gate_up = h.view(-1, D) @ w_gate_up  # (B*S, 2*Dff)
        gate = gate_up[:, :Dff]
        up   = gate_up[:, Dff:]
        mlp_out = noeris_geglu(gate, up)
        h = (mlp_out @ w_down).view(B, S, D)

        # Step 8: Post-MLP residual
        h = residual2 + h
        return h

    # Per-step timing for Noeris
    def step1_noeris():
        return noeris_rmsnorm(hidden.view(-1, D), w_pre_attn)
    noeris_step_times["1_pre_attn_rmsnorm"] = cuda_event_timer(step1_noeris)

    h_norm = step1_noeris().view(B, S, D)
    def step2_noeris():
        q = (h_norm.view(-1, D) @ w_q)
        k = (h_norm.view(-1, D) @ w_k)
        v = (h_norm.view(-1, D) @ w_v)
        return q, k, v
    noeris_step_times["2_qkv_proj"] = cuda_event_timer(step2_noeris)

    q_t = (h_norm.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3)
    k_t = (h_norm.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3)
    v_t = (h_norm.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3)

    def step3_noeris():
        noeris_qk_norm_rope(q_t, q_scale, cos, sin)
        noeris_qk_norm_rope(k_t, k_scale, cos, sin)
    noeris_step_times["3_qk_norm_rope"] = cuda_event_timer(step3_noeris)

    q_nr = noeris_qk_norm_rope(q_t, q_scale, cos, sin)
    k_nr = noeris_qk_norm_rope(k_t, k_scale, cos, sin)
    k_exp = expand_kv_for_gqa(k_nr, H, Hkv)
    v_exp = expand_kv_for_gqa(v_t, H, Hkv)
    def step4_noeris():
        return noeris_attention(q_nr, k_exp, v_exp, Hkv,
                                window_size=ws, is_causal=causal)
    noeris_step_times["4_attention"] = cuda_event_timer(step4_noeris)

    attn_out = step4_noeris().permute(0, 2, 1, 3).reshape(B, S, D)
    h_post_attn = hidden + attn_out

    def step6_noeris():
        return noeris_rmsnorm(h_post_attn.view(-1, D), w_pre_mlp)
    noeris_step_times["6_pre_mlp_rmsnorm"] = cuda_event_timer(step6_noeris)

    h_mlp = step6_noeris().view(B, S, D)
    gate_up = h_mlp.view(-1, D) @ w_gate_up
    gate_t = gate_up[:, :Dff]
    up_t   = gate_up[:, Dff:]
    def step7_noeris():
        return noeris_geglu(gate_t, up_t)
    noeris_step_times["7_geglu"] = cuda_event_timer(step7_noeris)

    # Total end-to-end Noeris fused
    noeris_fused_ms = cuda_event_timer(lambda: noeris_fused_layer(hidden))

    # =====================================================================
    # Path B: PyTorch separated
    # =====================================================================

    pytorch_step_times = {{}}

    def pytorch_separated_layer(hidden_in):
        residual = hidden_in

        # Step 1: Pre-attention RMSNorm
        h = pytorch_rmsnorm(hidden_in.view(-1, D), w_pre_attn).view(B, S, D)

        # Step 2: QKV projection
        q = (h.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3)
        k = (h.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3)
        v = (h.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3)

        # Step 3: Separate Q-RMSNorm, K-RMSNorm, Q-RoPE, K-RoPE (4 ops)
        q = pytorch_qk_norm_rope_separated(q, q_scale, cos, sin)
        k = pytorch_qk_norm_rope_separated(k, k_scale, cos, sin)

        # Step 4: Attention (SDPA with GQA expansion)
        k_exp = expand_kv_for_gqa(k, H, Hkv)
        v_exp = expand_kv_for_gqa(v, H, Hkv)

        if ws > 0:
            # Build sliding-window mask
            rows = torch.arange(S, device=hidden_in.device).unsqueeze(1)
            cols = torch.arange(S, device=hidden_in.device).unsqueeze(0)
            attn_mask = (cols >= (rows - ws + 1)) & (cols <= rows)
            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, is_causal=causal)

        # Step 5: Residual add
        h = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        h = residual + h

        # Step 6: Pre-MLP RMSNorm
        residual2 = h
        h = pytorch_rmsnorm(h.view(-1, D), w_pre_mlp).view(B, S, D)

        # Step 7: GeGLU MLP (separated)
        gate_up = h.view(-1, D) @ w_gate_up
        gate = gate_up[:, :Dff]
        up   = gate_up[:, Dff:]
        mlp_out = pytorch_geglu_separated(gate, up)
        h = (mlp_out @ w_down).view(B, S, D)

        # Step 8: Residual add
        h = residual2 + h
        return h

    # Per-step timing for PyTorch
    def step1_pytorch():
        return pytorch_rmsnorm(hidden.view(-1, D), w_pre_attn)
    pytorch_step_times["1_pre_attn_rmsnorm"] = cuda_event_timer(step1_pytorch)

    def step3_pytorch():
        pytorch_qk_norm_rope_separated(q_t, q_scale, cos, sin)
        pytorch_qk_norm_rope_separated(k_t, k_scale, cos, sin)
    pytorch_step_times["3_qk_norm_rope_separated"] = cuda_event_timer(step3_pytorch)

    def step4_pytorch():
        k_e = expand_kv_for_gqa(k_t, H, Hkv)
        v_e = expand_kv_for_gqa(v_t, H, Hkv)
        if ws > 0:
            rows = torch.arange(S, device="cuda").unsqueeze(1)
            cols = torch.arange(S, device="cuda").unsqueeze(0)
            mask = ((cols >= (rows - ws + 1)) & (cols <= rows)).unsqueeze(0).unsqueeze(0)
            return F.scaled_dot_product_attention(q_t, k_e, v_e, attn_mask=mask, is_causal=False)
        return F.scaled_dot_product_attention(q_t, k_e, v_e, is_causal=causal)
    pytorch_step_times["4_attention_sdpa"] = cuda_event_timer(step4_pytorch)

    def step6_pytorch():
        return pytorch_rmsnorm(h_post_attn.view(-1, D), w_pre_mlp)
    pytorch_step_times["6_pre_mlp_rmsnorm"] = cuda_event_timer(step6_pytorch)

    def step7_pytorch():
        return pytorch_geglu_separated(gate_t, up_t)
    pytorch_step_times["7_geglu_separated"] = cuda_event_timer(step7_pytorch)

    # Total end-to-end PyTorch separated
    pytorch_separated_ms = cuda_event_timer(lambda: pytorch_separated_layer(hidden))

    # =====================================================================
    # Correctness check
    # =====================================================================
    noeris_out = noeris_fused_layer(hidden)
    pytorch_out = pytorch_separated_layer(hidden)
    max_err = (noeris_out.to(torch.float32) - pytorch_out.to(torch.float32)).abs().max().item()
    correct = max_err < 0.1  # fp16 accumulates error over full layer

    # =====================================================================
    # Report
    # =====================================================================
    layer_speedup = pytorch_separated_ms / noeris_fused_ms if noeris_fused_ms > 0 else 0.0

    print(f"\\n  --- Per-step timing (ms) ---")
    print(f"  Noeris fused steps:")
    for step, ms in sorted(noeris_step_times.items()):
        print(f"    {{step}}: {{ms:.3f}} ms")
    print(f"  PyTorch separated steps:")
    for step, ms in sorted(pytorch_step_times.items()):
        print(f"    {{step}}: {{ms:.3f}} ms")
    print(f"\\n  --- Total layer ---")
    print(f"  noeris_fused_ms:      {{noeris_fused_ms:.3f}}")
    print(f"  pytorch_separated_ms: {{pytorch_separated_ms:.3f}}")
    print(f"  layer_speedup:        {{layer_speedup:.2f}}x")
    print(f"  correct:              {{correct}} (max_err={{max_err:.4f}})")

    return {{
        "name": name,
        "noeris_fused_ms": round(noeris_fused_ms, 4),
        "pytorch_separated_ms": round(pytorch_separated_ms, 4),
        "layer_speedup": round(layer_speedup, 4),
        "correct": correct,
        "max_err": round(max_err, 6),
        "noeris_step_times": {{k: round(v, 4) for k, v in noeris_step_times.items()}},
        "pytorch_step_times": {{k: round(v, 4) for k, v in pytorch_step_times.items()}},
        "config": cfg,
    }}


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Gemma 4 Decoder Layer Benchmark")
    print("Noeris fused vs PyTorch separated")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {{gpu_name}}")
    print(f"PyTorch: {{torch.__version__}}")
    print(f"Python: {{platform.python_version()}}")

    results = []
    for cfg in LAYER_CONFIGS:
        try:
            result = benchmark_layer(cfg)
            results.append(result)
        except Exception as exc:
            print(f"\\nERROR on {{cfg[\'name\']}}: {{exc}}")
            results.append({{"name": cfg["name"], "error": str(exc)[:300]}})

    # Summary
    print("\\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {{r[\'name\']}}: ERROR — {{r[\'error\'][:80]}}")
        else:
            print(f"  {{r[\'name\']}}: layer_speedup={{r[\'layer_speedup\']:.2f}}x "
                  f"({{r[\'noeris_fused_ms\']:.2f}} vs {{r[\'pytorch_separated_ms\']:.2f}} ms) "
                  f"correct={{r[\'correct\']}}")

    # Machine-readable JSON output
    print("\\n--- JSON_RESULTS_START ---")
    print(json.dumps({{"layer_results": results, "gpu": gpu_name}}, indent=2))
    print("--- JSON_RESULTS_END ---")

    return 0


if __name__ == "__main__":
    main()
'''
