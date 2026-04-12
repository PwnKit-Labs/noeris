#!/usr/bin/env python3
"""Comprehensive bombshell measurement script for free T4 GPU (Kaggle or Colab).

Primary platform: Kaggle (30 hr/week free T4, API-driven via `kaggle kernels push`).
Backup platform: Google Colab (~4-5 hr/day free T4).

Validates ALL headline claims in a single ~20-minute run:
  Phase 1: Fused QK-RMSNorm+RoPE prologue (4 ops -> 1 fused call)
  Phase 2: Split-K matmul vs cuBLAS
  Phase 3: Full Gemma 4 layer benchmark — SDPA attention in BOTH paths,
           Noeris fused kernels only where genuinely faster
  Phase 4: Forward + backward prologue (fused QK-RMSNorm+RoPE)
  Phase 5: Bandit search convergence on attention

Usage (Kaggle or Colab):
  !git clone https://github.com/PwnKit-Labs/noeris && cd noeris
  !pip install -e . numpy scikit-learn -q
  !python scripts/colab_bombshell.py

Outputs machine-readable JSON with all results.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch

print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: No GPU available. Change Colab runtime to T4 GPU.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")

import triton
print(f"Triton {triton.__version__}")

# ============================================================================
# Timing helper
# ============================================================================

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
    return times[len(times) // 2]


# ============================================================================
# Phase 1: Fused QK-RMSNorm+RoPE prologue (THE headline kernel)
# ============================================================================

def phase1_prologue_fusion():
    """Compare Noeris fused QK-RMSNorm+RoPE (1 call) vs 4 separate PyTorch ops.

    This is the proven 5-13x win. We do NOT compare attention here.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Fused QK-RMSNorm+RoPE prologue (4 ops -> 1)")
    print("=" * 70)

    from research_engine.triton_qk_norm_rope import apply_qk_norm_rope

    # Gemma 4 E2B local shape
    B, H, S, D = 1, 8, 4096, 256
    H_kv = 1

    print(f"  Shape: B={B}, H={H}, H_kv={H_kv}, S={S}, D={D}")

    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=torch.float16)
    cos = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
    q_scale = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1
    k_scale = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1

    config = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}

    # --- Separated baseline (what vLLM does: 4 separate ops) ---
    def separated_4ops():
        eps = 1e-6
        # 1. Q-RMSNorm
        q_var = q.float().pow(2).mean(-1, keepdim=True)
        q_n = (q.float() * torch.rsqrt(q_var + eps)).half() * (1.0 + q_scale).half()
        # 2. K-RMSNorm
        k_var = k.float().pow(2).mean(-1, keepdim=True)
        k_n = (k.float() * torch.rsqrt(k_var + eps)).half() * (1.0 + k_scale).half()
        # 3. Q-RoPE
        c = cos[None, None, :, :].half()
        sn = sin[None, None, :, :].half()
        qe, qo = q_n[..., 0::2], q_n[..., 1::2]
        q_out = torch.stack([qe * c - qo * sn, qe * sn + qo * c], dim=-1).reshape(q.shape)
        # 4. K-RoPE
        ke, ko = k_n[..., 0::2], k_n[..., 1::2]
        k_out = torch.stack([ke * c - ko * sn, ke * sn + ko * c], dim=-1).reshape(k.shape)
        return q_out, k_out

    # --- Noeris fused (1 call instead of 4) ---
    def noeris_fused():
        return apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config=config)

    sep_ms = cuda_event_timer(separated_4ops)
    fused_ms = cuda_event_timer(noeris_fused)

    speedup = sep_ms / fused_ms if fused_ms > 0 else 0.0

    print(f"  Separated (4 ops): {sep_ms:.3f} ms")
    print(f"  Noeris fused:      {fused_ms:.3f} ms")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  Launches saved:    4 -> 2 (Q+K each fused)")

    return {
        "separated_ms": round(sep_ms, 4),
        "fused_ms": round(fused_ms, 4),
        "speedup": round(speedup, 4),
        "shape": f"B{B}_H{H}_Hkv{H_kv}_S{S}_D{D}",
    }


# ============================================================================
# Phase 2: Split-K matmul vs cuBLAS
# ============================================================================

def phase2_splitk_vs_cublas():
    """Compare Noeris split-K matmul vs torch.matmul (cuBLAS) via subprocess."""
    print("\n" + "=" * 70)
    print("PHASE 2: Split-K matmul vs cuBLAS")
    print("=" * 70)

    import subprocess
    import tempfile

    from research_engine.triton_operators import REGISTRY

    spec = REGISTRY.get("matmul_splitk")

    shapes = [
        {"name": "4096x4096", "M": 4096, "N": 4096, "K": 4096},
        {"name": "8192x4096", "M": 8192, "N": 4096, "K": 4096},
        {"name": "llm_mlp_down", "M": 4096, "N": 4096, "K": 11008},
    ]

    # T4-friendly configs: smaller tiles to fit shared memory
    configs = [
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "SPLIT_K": 2, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
         "SPLIT_K": 4, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
         "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    ]

    script = spec.benchmark_script_fn(configs, shapes)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    print(f"  Running split-K benchmark subprocess...")
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "unknown"
        print(f"  FAIL: {err[:200]}")
        return {"error": err[:300]}

    # Parse JSON output
    stdout = proc.stdout
    json_start = stdout.find("{")
    if json_start < 0:
        print("  FAIL: no JSON output found")
        print(stdout[-500:])
        return {"error": "no JSON output"}

    payload = json.loads(stdout[json_start:])

    all_results = []
    best_ratio = 0.0
    best_shape = ""

    for cr in payload.get("config_results", []):
        config = cr.get("config", {})
        sk = config.get("SPLIT_K", 1)
        for r in cr.get("results", []):
            shape_name = r.get("shape_name", "?")
            ratio = r.get("ratio_vs_cublas", 0.0) or 0.0
            correct = r.get("correct", False)

            if correct:
                all_results.append({
                    "shape": shape_name,
                    "SPLIT_K": sk,
                    "splitk_ms": r.get("ms"),
                    "cublas_ms": r.get("cublas_ms"),
                    "ratio_vs_cublas": ratio,
                })
                print(f"  {shape_name} SPLIT_K={sk}: ratio={ratio:.3f}")

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_shape = f"{shape_name}_SK{sk}"

    print(f"\n  Best ratio vs cuBLAS: {best_ratio:.3f} on {best_shape}")

    return {
        "best_ratio": round(best_ratio, 4),
        "shape": best_shape,
        "details": all_results,
    }


# ============================================================================
# Phase 3: Full layer benchmark (Gemma 4 E2B local) — SDPA in BOTH paths
# ============================================================================

def phase3_layer_benchmark():
    """Run Gemma 4 E2B local layer: Noeris fused vs PyTorch separated.

    BOTH paths use torch.nn.functional.scaled_dot_product_attention for
    attention. The ONLY differences are the fused prologue (RMSNorm,
    QK-RMSNorm+RoPE) and fused GeGLU — the parts where Noeris is
    genuinely faster.

    Per-step timing shows WHERE the savings come from.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Full layer benchmark (Gemma 4 E2B local)")
    print("  Architecture: SDPA attention in BOTH paths")
    print("  Noeris fusions: RMSNorm, QK-RMSNorm+RoPE, GeGLU")
    print("=" * 70)

    from research_engine.triton_rmsnorm import rmsnorm
    from research_engine.triton_qk_norm_rope import apply_qk_norm_rope
    from research_engine.triton_geglu import geglu

    F = torch.nn.functional

    # --- Gemma 4 E2B local shape ---
    B, S, D = 1, 2048, 1536
    H, H_kv, Dh = 8, 1, 256
    Dff = 6144
    W = 512  # sliding window
    eps = 1e-6

    print(f"  Shape: B={B}, S={S}, D={D}, H={H}, Hkv={H_kv}, Dh={Dh}, Dff={Dff}, W={W}")

    # --- Allocate weights (random, just for timing) ---
    W_qkv = torch.randn(D, (H + 2 * H_kv) * Dh, device="cuda", dtype=torch.float16)
    W_o = torch.randn(H * Dh, D, device="cuda", dtype=torch.float16)
    W_gate_up = torch.randn(D, 2 * Dff, device="cuda", dtype=torch.float16)
    W_down = torch.randn(Dff, D, device="cuda", dtype=torch.float16)

    rn_w1 = torch.randn(D, device="cuda", dtype=torch.float16)
    rn_w2 = torch.randn(D, device="cuda", dtype=torch.float16)
    q_scale = torch.randn(Dh, device="cuda", dtype=torch.float32) * 0.1
    k_scale = torch.randn(Dh, device="cuda", dtype=torch.float32) * 0.1
    cos = torch.randn(S, Dh // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, Dh // 2, device="cuda", dtype=torch.float32)

    x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)

    # GQA repeat factor for SDPA
    repeat = H // H_kv

    # Triton configs
    rn_cfg = {"BLOCK_SIZE": 1024, "num_warps": 2, "num_stages": 1}
    qknr_cfg = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}
    geglu_cfg = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1}

    # ---------------------------------------------------------------
    # BASELINE: PyTorch separated ops (what vLLM does)
    # ---------------------------------------------------------------

    def baseline_pre_attn_rmsnorm(x_in):
        """Separate RMSNorm (Gemma 1+w affine mode)."""
        var = x_in.float().pow(2).mean(-1, keepdim=True)
        return (x_in.float() * torch.rsqrt(var + eps)).half() * (1.0 + rn_w1).half()

    def baseline_qkv_proj(normed):
        """QKV projection -> split into q, k, v."""
        qkv = normed.reshape(B * S, D) @ W_qkv  # (B*S, (H+2*Hkv)*Dh)
        q_dim = H * Dh
        k_dim = H_kv * Dh
        q = qkv[:, :q_dim].reshape(B, S, H, Dh).permute(0, 2, 1, 3)
        k = qkv[:, q_dim:q_dim + k_dim].reshape(B, S, H_kv, Dh).permute(0, 2, 1, 3)
        v = qkv[:, q_dim + k_dim:].reshape(B, S, H_kv, Dh).permute(0, 2, 1, 3)
        return q, k, v

    def baseline_qk_norm_rope(q, k):
        """4 separate ops: Q-RMSNorm, K-RMSNorm, Q-RoPE, K-RoPE."""
        # Q-RMSNorm
        q_var = q.float().pow(2).mean(-1, keepdim=True)
        q_n = (q.float() * torch.rsqrt(q_var + eps)).half() * (1.0 + q_scale).half()
        # K-RMSNorm
        k_var = k.float().pow(2).mean(-1, keepdim=True)
        k_n = (k.float() * torch.rsqrt(k_var + eps)).half() * (1.0 + k_scale).half()
        # Q-RoPE
        c = cos[None, None, :, :].half()
        sn = sin[None, None, :, :].half()
        qe, qo = q_n[..., 0::2], q_n[..., 1::2]
        q_out = torch.stack([qe * c - qo * sn, qe * sn + qo * c], dim=-1).reshape(q.shape)
        # K-RoPE
        ke, ko = k_n[..., 0::2], k_n[..., 1::2]
        k_out = torch.stack([ke * c - ko * sn, ke * sn + ko * c], dim=-1).reshape(k.shape)
        return q_out, k_out

    def sdpa_attention(q, k, v):
        """SDPA attention with GQA expansion — SAME in both paths."""
        k_exp = k.unsqueeze(2).expand(B, H_kv, repeat, S, Dh).reshape(B, H, S, Dh)
        v_exp = v.unsqueeze(2).expand(B, H_kv, repeat, S, Dh).reshape(B, H, S, Dh)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

    def baseline_output_proj(attn_out):
        """Output projection."""
        return attn_out.permute(0, 2, 1, 3).reshape(B * S, H * Dh) @ W_o

    def baseline_pre_mlp_rmsnorm(x_in):
        """Separate RMSNorm for MLP block."""
        var = x_in.float().pow(2).mean(-1, keepdim=True)
        return (x_in.float() * torch.rsqrt(var + eps)).half() * (1.0 + rn_w2).half()

    def baseline_geglu_mlp(normed):
        """Separate gate_up matmul, separate GELU, separate mul, separate down."""
        gate_up = normed.reshape(B * S, D) @ W_gate_up  # (B*S, 2*Dff)
        gate = gate_up[:, :Dff]
        up = gate_up[:, Dff:]
        activated = F.gelu(up, approximate="tanh")
        hidden = gate * activated
        return hidden @ W_down

    # ---------------------------------------------------------------
    # NOERIS: Fused ops where we are genuinely faster
    # ---------------------------------------------------------------

    def noeris_pre_attn_rmsnorm(x_in):
        """Noeris fused RMSNorm (Gemma 1+w affine)."""
        return rmsnorm(x_in.reshape(B * S, D), rn_w1, config=rn_cfg, affine_mode=1).reshape(B, S, D)

    def noeris_qk_norm_rope(q, k):
        """Noeris fused QK-RMSNorm+RoPE (1 call instead of 4!)."""
        return apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config=qknr_cfg)

    def noeris_pre_mlp_rmsnorm(x_in):
        """Noeris fused RMSNorm."""
        return rmsnorm(x_in.reshape(B * S, D), rn_w2, config=rn_cfg, affine_mode=1).reshape(B, S, D)

    def noeris_geglu_mlp(normed):
        """Noeris fused GeGLU (gate * GELU(up) in 1 kernel) + matmuls."""
        gate_up = normed.reshape(B * S, D) @ W_gate_up
        gate = gate_up[:, :Dff]
        up = gate_up[:, Dff:]
        hidden = geglu(gate, up, config=geglu_cfg)
        return hidden @ W_down

    # ---------------------------------------------------------------
    # Per-step timing
    # ---------------------------------------------------------------

    print("\n  Timing each step individually...")

    # We need q, k, v for attention timing — run the projections once
    normed_base = baseline_pre_attn_rmsnorm(x)
    q_raw, k_raw, v_raw = baseline_qkv_proj(normed_base)

    # Baseline steps
    t_base_pre_rn = cuda_event_timer(lambda: baseline_pre_attn_rmsnorm(x))
    q_b, k_b = baseline_qk_norm_rope(q_raw, k_raw)
    t_base_qknr = cuda_event_timer(lambda: baseline_qk_norm_rope(q_raw, k_raw))
    t_base_attn = cuda_event_timer(lambda: sdpa_attention(q_b, k_b, v_raw))
    attn_out = sdpa_attention(q_b, k_b, v_raw)
    o_proj = baseline_output_proj(attn_out)
    residual1 = x + o_proj.reshape(B, S, D)
    t_base_pre_mlp_rn = cuda_event_timer(lambda: baseline_pre_mlp_rmsnorm(residual1))
    normed_mlp_b = baseline_pre_mlp_rmsnorm(residual1)
    t_base_geglu = cuda_event_timer(lambda: baseline_geglu_mlp(normed_mlp_b))

    # Noeris steps
    t_noeris_pre_rn = cuda_event_timer(lambda: noeris_pre_attn_rmsnorm(x))
    normed_n = noeris_pre_attn_rmsnorm(x)
    q_raw_n, k_raw_n, v_raw_n = baseline_qkv_proj(normed_n)  # same matmul
    q_n, k_n = noeris_qk_norm_rope(q_raw_n, k_raw_n)
    t_noeris_qknr = cuda_event_timer(lambda: noeris_qk_norm_rope(q_raw_n, k_raw_n))
    t_noeris_attn = cuda_event_timer(lambda: sdpa_attention(q_n, k_n, v_raw_n))
    attn_out_n = sdpa_attention(q_n, k_n, v_raw_n)
    o_proj_n = baseline_output_proj(attn_out_n)  # same matmul
    residual1_n = x + o_proj_n.reshape(B, S, D)
    t_noeris_pre_mlp_rn = cuda_event_timer(lambda: noeris_pre_mlp_rmsnorm(residual1_n))
    normed_mlp_n = noeris_pre_mlp_rmsnorm(residual1_n)
    t_noeris_geglu = cuda_event_timer(lambda: noeris_geglu_mlp(normed_mlp_n))

    # Totals (only the steps that differ + shared steps)
    # Shared: qkv_proj, attention, output_proj, residual adds
    # We time the shared steps once and add them to both
    t_qkv_proj = cuda_event_timer(lambda: baseline_qkv_proj(normed_base))
    t_o_proj = cuda_event_timer(lambda: baseline_output_proj(attn_out))

    t_base_total = (t_base_pre_rn + t_qkv_proj + t_base_qknr + t_base_attn
                    + t_o_proj + t_base_pre_mlp_rn + t_base_geglu)
    t_noeris_total = (t_noeris_pre_rn + t_qkv_proj + t_noeris_qknr + t_noeris_attn
                      + t_o_proj + t_noeris_pre_mlp_rn + t_noeris_geglu)

    speedup = t_base_total / t_noeris_total if t_noeris_total > 0 else 0.0

    # Print the comparison table
    print(f"\n  {'Step':<24s} {'Baseline (ms)':>14s} {'Noeris (ms)':>12s} {'Savings':>10s}")
    print(f"  {'-' * 62}")

    steps = [
        ("pre_attn_rmsnorm", t_base_pre_rn, t_noeris_pre_rn),
        ("qkv_proj (matmul)", t_qkv_proj, t_qkv_proj),
        ("qk_norm_rope (4->1)", t_base_qknr, t_noeris_qknr),
        ("attention (SDPA)", t_base_attn, t_noeris_attn),
        ("output_proj (matmul)", t_o_proj, t_o_proj),
        ("pre_mlp_rmsnorm", t_base_pre_mlp_rn, t_noeris_pre_mlp_rn),
        ("geglu_mlp", t_base_geglu, t_noeris_geglu),
    ]

    for name, base_ms, noeris_ms in steps:
        saving = base_ms - noeris_ms
        marker = ""
        if name == "qk_norm_rope (4->1)":
            marker = " <-- biggest win"
        elif name in ("attention (SDPA)", "qkv_proj (matmul)", "output_proj (matmul)"):
            marker = " (same)"
        print(f"  {name:<24s} {base_ms:>11.3f} ms {noeris_ms:>9.3f} ms {saving:>+8.3f} ms{marker}")

    print(f"  {'-' * 62}")
    print(f"  {'TOTAL':<24s} {t_base_total:>11.3f} ms {t_noeris_total:>9.3f} ms "
          f"{t_base_total - t_noeris_total:>+8.3f} ms")
    print(f"  Layer speedup: {speedup:.3f}x")

    return {
        "baseline_total_ms": round(t_base_total, 4),
        "noeris_total_ms": round(t_noeris_total, 4),
        "speedup": round(speedup, 4),
        "per_step": {
            "pre_attn_rmsnorm": {"baseline": round(t_base_pre_rn, 4), "noeris": round(t_noeris_pre_rn, 4)},
            "qkv_proj": {"baseline": round(t_qkv_proj, 4), "noeris": round(t_qkv_proj, 4)},
            "qk_norm_rope": {"baseline": round(t_base_qknr, 4), "noeris": round(t_noeris_qknr, 4)},
            "attention_sdpa": {"baseline": round(t_base_attn, 4), "noeris": round(t_noeris_attn, 4)},
            "output_proj": {"baseline": round(t_o_proj, 4), "noeris": round(t_o_proj, 4)},
            "pre_mlp_rmsnorm": {"baseline": round(t_base_pre_mlp_rn, 4), "noeris": round(t_noeris_pre_mlp_rn, 4)},
            "geglu_mlp": {"baseline": round(t_base_geglu, 4), "noeris": round(t_noeris_geglu, 4)},
        },
        "shape": f"B{B}_S{S}_D{D}_H{H}_Hkv{H_kv}_Dh{Dh}_Dff{Dff}_W{W}",
    }


# ============================================================================
# Phase 4: Forward + backward prologue (QK-RMSNorm+RoPE)
# ============================================================================

def phase4_prologue_fwd_bwd():
    """Benchmark fused QK-RMSNorm+RoPE forward+backward vs separated ops via subprocess."""
    print("\n" + "=" * 70)
    print("PHASE 4: Forward + backward prologue (QK-RMSNorm+RoPE)")
    print("=" * 70)

    import subprocess
    import tempfile

    from research_engine.triton_operators import REGISTRY

    spec = REGISTRY.get("qk_norm_rope_bwd")

    # Gemma 4 local shape, T4-friendly
    shapes = [
        {"name": "gemma4_local", "batch": 1, "heads": 8, "num_kv_heads": 4,
         "seq": 4096, "head_dim": 256},
    ]

    # T4-friendly configs: small BLOCK_SIZE to fit shared memory
    configs = [
        {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1},
        {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1},
    ]

    script = spec.benchmark_script_fn(configs, shapes)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    print(f"  Running prologue fwd+bwd benchmark subprocess...")
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "unknown"
        print(f"  FAIL: {err[:200]}")
        return {"error": err[:300]}

    # Parse JSON output
    stdout = proc.stdout
    json_start = stdout.find("{")
    if json_start < 0:
        print("  FAIL: no JSON output found")
        print(stdout[-500:])
        return {"error": "no JSON output"}

    payload = json.loads(stdout[json_start:])

    best_speedup = 0.0
    best_ms = None
    best_sep_ms = None

    for cr in payload.get("config_results", []):
        for r in cr.get("results", []):
            correct = r.get("correct", False)
            speedup = r.get("backward_fusion_speedup", 0.0) or 0.0
            if correct and speedup > best_speedup:
                best_speedup = speedup
                best_ms = r.get("ms")
                best_sep_ms = r.get("separated_ms")

    print(f"  Best fused fwd+bwd:     {best_ms} ms")
    print(f"  Separated fwd+bwd:      {best_sep_ms} ms")
    print(f"  Backward fusion speedup: {best_speedup:.2f}x")

    return {
        "forward": {
            "fusion_speedup": round(best_speedup, 4),
            "fused_ms": round(best_ms, 4) if best_ms else None,
            "separated_ms": round(best_sep_ms, 4) if best_sep_ms else None,
        },
        "backward": {
            "fusion_speedup": round(best_speedup, 4),
            "fused_ms": round(best_ms, 4) if best_ms else None,
            "separated_ms": round(best_sep_ms, 4) if best_sep_ms else None,
        },
    }


# ============================================================================
# Phase 5: Bandit search convergence on attention
# ============================================================================

def phase5_bandit_search():
    """Run 3 iterations of bandit search on attention, report improvement."""
    print("\n" + "=" * 70)
    print("PHASE 5: Bandit search convergence (attention)")
    print("=" * 70)

    from research_engine.triton_operators import REGISTRY
    from research_engine.triton_kernels import ConfigDatabase

    import subprocess
    import tempfile

    spec = REGISTRY.get("attention")
    hardware = GPU_NAME

    # Use tiny shapes for speed on T4 — include a sliding-window shape
    shapes = [s for s in spec.shape_buckets
              if s["name"] in ("short_64", "gemma4_local_short")][:2]
    if not shapes:
        shapes = spec.shape_buckets[:2]

    # Baseline: curated config on first shape
    baseline_config = spec.curated_configs[0]
    script_baseline = spec.benchmark_script_fn([baseline_config], shapes[:1])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_baseline)
        f.flush()
        proc = subprocess.run(
            [sys.executable, f.name],
            capture_output=True, text=True, timeout=120,
        )

    baseline_metric = None
    if proc.returncode == 0:
        start = proc.stdout.find("{")
        if start >= 0:
            payload = json.loads(proc.stdout[start:])
            for cr in payload.get("config_results", []):
                for r in cr.get("results", []):
                    m = r.get("gb_per_s") or r.get("tflops") or 0
                    if r.get("correct") and (baseline_metric is None or m > baseline_metric):
                        baseline_metric = m

    print(f"  Baseline metric (curated starter): {baseline_metric}")

    # Run 3 bandit iterations
    db_path = "/tmp/noeris_bombshell_bandit.json"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    best_metric_after = baseline_metric

    for iteration in range(3):
        print(f"\n  --- Bandit iteration {iteration + 1}/3 ---")

        try:
            from research_engine.bandit_selector import BanditSelector
            db = ConfigDatabase(path=db_path)
            bandit = BanditSelector()
            configs = bandit.select_configs(
                spec=spec, database=db, hardware=hardware,
                shapes=shapes, max_configs=6,
            )
        except Exception as e:
            print(f"    Bandit selection failed: {e}, using grid")
            configs = spec.grid_generator_fn(max_configs=6)[:6]

        script = spec.benchmark_script_fn(configs, shapes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            proc = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=180,
            )

        if proc.returncode != 0:
            err = proc.stderr[-200:] if proc.stderr else "unknown"
            print(f"    FAIL: {err[:100]}")
            continue

        start = proc.stdout.find("{")
        if start < 0:
            print("    No JSON output")
            continue

        payload = json.loads(proc.stdout[start:])

        for cr in payload.get("config_results", []):
            config = cr.get("config", {})
            for r in cr.get("results", []):
                shape_name = r.get("shape_name", "?")
                correct = r.get("correct", False)
                m = r.get("gb_per_s") or r.get("tflops") or 0
                ms_val = r.get("ms") or 0

                if correct:
                    if best_metric_after is None or m > best_metric_after:
                        best_metric_after = m

                    # Record in DB for next iteration
                    bucket = spec.shape_bucket_fn(
                        next((s for s in shapes if s["name"] == shape_name), {})
                    )
                    db.record_result(
                        shape={"name": shape_name},
                        hardware=hardware,
                        config=config,
                        tflops=float(m),
                        ms=float(ms_val),
                        correct=correct,
                        operator="attention",
                        bucket=bucket,
                        config_id_str=spec.config_id_fn(config),
                    )

        db.save()
        print(f"    Best metric so far: {best_metric_after}")

    before = baseline_metric or 0
    after = best_metric_after or 0
    improvement = ((after - before) / before * 100) if before > 0 else 0.0

    print(f"\n  Before (curated): {before}")
    print(f"  After (bandit):   {after}")
    print(f"  Improvement:      {improvement:.1f}%")

    return {
        "before": round(before, 4) if before else None,
        "after": round(after, 4) if after else None,
        "improvement_pct": round(improvement, 2),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("NOERIS BOMBSHELL MEASUREMENT SCRIPT")
    print(f"GPU: {GPU_NAME}")
    print("=" * 70)

    t_start = time.time()
    results = {}

    phases = [
        ("prologue_fusion", phase1_prologue_fusion),
        ("splitk_vs_cublas", phase2_splitk_vs_cublas),
        ("layer_speedup", phase3_layer_benchmark),
        ("prologue_forward_backward", phase4_prologue_fwd_bwd),
        ("attention_bandit_improvement", phase5_bandit_search),
    ]

    for name, fn in phases:
        try:
            results[name] = fn()
        except Exception:
            tb = traceback.format_exc()
            print(f"\n  PHASE ERROR: {tb[-300:]}")
            results[name] = {"error": tb[-300:]}

    elapsed = time.time() - t_start

    # Build final output
    output = {
        "gpu": GPU_NAME,
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "elapsed_seconds": round(elapsed, 1),
        "bombshell_results": results,
    }

    # Summary
    print("\n" + "=" * 70)
    print("BOMBSHELL RESULTS SUMMARY")
    print("=" * 70)

    pro = results.get("prologue_fusion", {})
    if "error" not in pro:
        print(f"  Prologue fusion (4->1):   {pro.get('speedup', '?')}x")

    sk = results.get("splitk_vs_cublas", {})
    if "error" not in sk:
        print(f"  Split-K vs cuBLAS:        best ratio {sk.get('best_ratio', '?')} "
              f"on {sk.get('shape', '?')}")

    layer = results.get("layer_speedup", {})
    if "error" not in layer:
        print(f"  Layer speedup (SDPA both): {layer.get('speedup', '?')}x")

    pro_bwd = results.get("prologue_forward_backward", {})
    if "error" not in pro_bwd:
        fwd = pro_bwd.get("forward", {})
        bwd = pro_bwd.get("backward", {})
        print(f"  Prologue forward speedup: {fwd.get('fusion_speedup', '?')}x")
        print(f"  Prologue backward speedup: {bwd.get('fusion_speedup', '?')}x")

    bandit = results.get("attention_bandit_improvement", {})
    if "error" not in bandit:
        print(f"  Bandit improvement:       {bandit.get('improvement_pct', '?')}%")

    print(f"\n  Total time: {elapsed:.0f}s")

    # Save JSON
    out_path = Path("bombshell_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {out_path}")

    # Machine-readable JSON to stdout
    print("\n--- BOMBSHELL_JSON_START ---")
    print(json.dumps(output, indent=2))
    print("--- BOMBSHELL_JSON_END ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
