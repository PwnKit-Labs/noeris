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

# Import Noeris operator launchers from the installed research_engine package.
# This avoids inlining kernel source that can drift out of sync with the
# canonical operator modules.
from research_engine.triton_rmsnorm import rmsnorm as _noeris_rmsnorm_raw
from research_engine.triton_qk_norm_rope import apply_qk_norm_rope as _noeris_qk_norm_rope_raw
from research_engine.triton_geglu import geglu as _noeris_geglu_raw
from research_engine.triton_attention_v2 import flash_attn_v2 as _noeris_flash_attn_raw


LAYER_CONFIGS = json.loads({configs_json!r})


# =============================================================================
# Noeris operator wrappers (delegate to installed Triton kernels)
# =============================================================================

# Default configs that work well on T4/A100/H100
_RMSNORM_CONFIG = {{"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}}
_QK_NORM_ROPE_CONFIG = {{"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}}
_GEGLU_CONFIG = {{"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}}
_ATTN_CONFIG = {{"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 3}}

def noeris_rmsnorm(x, w, eps=1e-6):
    """RMSNorm via the canonical Triton kernel (Gemma affine_mode=1)."""
    return _noeris_rmsnorm_raw(x, w, _RMSNORM_CONFIG, eps=eps, affine_mode=1)


def noeris_qk_norm_rope(q, k, q_scale, k_scale, cos, sin):
    """Fused QK-RMSNorm + RoPE via the canonical Triton kernel."""
    return _noeris_qk_norm_rope_raw(q, k, cos, sin, q_scale, k_scale, _QK_NORM_ROPE_CONFIG)


def noeris_geglu(gate, up):
    """Fused GeGLU via the canonical Triton kernel."""
    return _noeris_geglu_raw(gate, up, _GEGLU_CONFIG)


def noeris_attention(Q, K, V, num_kv_heads, window_size=-1, is_causal=True):
    """FlashAttention with GQA + optional sliding window via the canonical Triton kernel."""
    return _noeris_flash_attn_raw(Q, K, V, _ATTN_CONFIG, is_causal=is_causal,
                                  window_size=window_size, num_kv_heads=num_kv_heads)


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
    return torch.cos(angles), torch.sin(angles)


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
    q_scale      = torch.randn(Dh, device=device, dtype=torch.float32) * 0.1
    k_scale      = torch.randn(Dh, device=device, dtype=torch.float32) * 0.1
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

        # Step 2: QKV projection — contiguous (B, H, S, D) for Triton kernels
        q = (h.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = (h.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
        v = (h.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()

        # Step 3: Fused QK-RMSNorm + RoPE (our headline kernel!)
        q, k = noeris_qk_norm_rope(q, k, q_scale, k_scale, cos, sin)

        # Step 4: Attention (GQA + sliding-window) — flash_attn handles GQA internally
        attn_out = noeris_attention(q, k, v, Hkv,
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
        q = (h_norm.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = (h_norm.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
        v = (h_norm.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
        return q, k, v
    noeris_step_times["2_qkv_proj"] = cuda_event_timer(step2_noeris)

    q_t = (h_norm.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
    k_t = (h_norm.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
    v_t = (h_norm.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()

    def step3_noeris():
        noeris_qk_norm_rope(q_t, k_t, q_scale, k_scale, cos, sin)
    noeris_step_times["3_qk_norm_rope_fused"] = cuda_event_timer(step3_noeris)

    q_nr, k_nr = noeris_qk_norm_rope(q_t, k_t, q_scale, k_scale, cos, sin)
    def step4_noeris():
        return noeris_attention(q_nr, k_nr, v_t, Hkv,
                                window_size=ws, is_causal=causal)
    noeris_step_times["4_attention_flash"] = cuda_event_timer(step4_noeris)

    attn_out = step4_noeris().permute(0, 2, 1, 3).reshape(B, S, D)
    h_post_attn = hidden + attn_out

    def step6_noeris():
        return noeris_rmsnorm(h_post_attn.view(-1, D), w_pre_mlp)
    noeris_step_times["6_pre_mlp_rmsnorm"] = cuda_event_timer(step6_noeris)

    h_mlp = step6_noeris().view(B, S, D)
    def step7a_noeris():
        return h_mlp.view(-1, D) @ w_gate_up
    noeris_step_times["7a_gate_up_proj"] = cuda_event_timer(step7a_noeris)

    gate_up = h_mlp.view(-1, D) @ w_gate_up
    gate_t = gate_up[:, :Dff]
    up_t   = gate_up[:, Dff:]
    def step7b_noeris():
        return noeris_geglu(gate_t, up_t)
    noeris_step_times["7b_geglu_fused"] = cuda_event_timer(step7b_noeris)

    mlp_act = noeris_geglu(gate_t, up_t)
    def step7c_noeris():
        return mlp_act @ w_down
    noeris_step_times["7c_down_proj"] = cuda_event_timer(step7c_noeris)

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

        # Step 2: QKV projection — contiguous (B, H, S, D) for SDPA
        q = (h.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = (h.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
        v = (h.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()

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

    h_norm_pt = step1_pytorch().view(B, S, D)
    def step2_pytorch():
        q = (h_norm_pt.view(-1, D) @ w_q).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = (h_norm_pt.view(-1, D) @ w_k).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
        v = (h_norm_pt.view(-1, D) @ w_v).view(B, S, Hkv, Dh).permute(0, 2, 1, 3).contiguous()
        return q, k, v
    pytorch_step_times["2_qkv_proj"] = cuda_event_timer(step2_pytorch)

    def step3_pytorch():
        pytorch_qk_norm_rope_separated(q_t, q_scale, cos, sin)
        pytorch_qk_norm_rope_separated(k_t, k_scale, cos, sin)
    pytorch_step_times["3_qk_norm_rope_4ops"] = cuda_event_timer(step3_pytorch)

    q_pt = pytorch_qk_norm_rope_separated(q_t, q_scale, cos, sin)
    k_pt = pytorch_qk_norm_rope_separated(k_t, k_scale, cos, sin)
    def step4_pytorch():
        k_e = expand_kv_for_gqa(k_pt, H, Hkv)
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

    h_mlp_pt = step6_pytorch().view(B, S, D)
    def step7a_pytorch():
        return h_mlp_pt.view(-1, D) @ w_gate_up
    pytorch_step_times["7a_gate_up_proj"] = cuda_event_timer(step7a_pytorch)

    def step7b_pytorch():
        return pytorch_geglu_separated(gate_t, up_t)
    pytorch_step_times["7b_geglu_separated"] = cuda_event_timer(step7b_pytorch)

    def step7c_pytorch():
        return mlp_act @ w_down
    pytorch_step_times["7c_down_proj"] = cuda_event_timer(step7c_pytorch)

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

    print("\\n  --- Per-step timing (ms) ---")
    hdr = "  {{:<30s}} {{:>10s}} {{:>10s}} {{:>10s}}".format("Step", "Noeris", "PyTorch", "Speedup")
    print(hdr)
    print("  " + "─" * 62)
    for n_key, n_ms in sorted(noeris_step_times.items()):
        # Find matching pytorch step by step number prefix
        prefix = n_key[:2]
        p_keys = [k for k in pytorch_step_times if k.startswith(prefix)]
        if p_keys:
            p_ms = pytorch_step_times[p_keys[0]]
            spd = p_ms / n_ms if n_ms > 0 else 0.0
            row = "    {{:<28s}} {{:>8.3f}}   {{:>8.3f}}   {{:>8.2f}}x".format(n_key, n_ms, p_ms, spd)
        else:
            row = "    {{:<28s}} {{:>8.3f}}   {{:>8s}}   {{:>8s}}".format(n_key, n_ms, "—", "—")
        print(row)
    # Print pytorch-only steps
    for p_key, p_ms in sorted(pytorch_step_times.items()):
        prefix = p_key[:2]
        n_keys = [k for k in noeris_step_times if k.startswith(prefix)]
        if not n_keys:
            row = "    {{:<28s}} {{:>8s}}   {{:>8.3f}}   {{:>8s}}".format(p_key, "—", p_ms, "—")
            print(row)
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
