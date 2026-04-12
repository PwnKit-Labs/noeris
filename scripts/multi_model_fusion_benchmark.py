#!/usr/bin/env python3
"""Multi-model fusion benchmark: fused QK-RMSNorm+RoPE across model families.

Tests the Noeris fused kernel on 15+ model families (Gemma 4, LLaMA 3/4,
Qwen 3, Mistral/Mixtral, Phi-3/4, Falcon 3, DBRX, OLMo 2, InternLM 3)
to prove the fusion is a universal transformer optimization. Models with
QK-norm (Gemma, Qwen 3, OLMo 2) benefit most from the fused kernel.

Usage (Kaggle T4 or Colab):
  !git clone https://github.com/PwnKit-Labs/noeris && cd noeris
  !pip install -e . numpy scikit-learn -q
  !python scripts/multi_model_fusion_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch

print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: No GPU available. Change runtime to T4 GPU.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")

import triton
print(f"Triton {triton.__version__}")

from research_engine.triton_qk_norm_rope import apply_qk_norm_rope

# ============================================================================
# Timing helper (CUDA events, median ms)
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
# Model shapes and T4-optimized configs
# ============================================================================

MODELS = [
    # --- Gemma family (QK-norm: YES, affine_mode=1) ---
    {"name": "gemma4_e2b_local",  "B": 1, "H": 8,  "H_kv": 1,  "S": 4096, "D": 256, "qk_norm": True},
    {"name": "gemma4_31b_global", "B": 1, "H": 32, "H_kv": 4,  "S": 4096, "D": 512, "qk_norm": True},

    # --- Llama family (QK-norm: NO, standard RMSNorm, RoPE) ---
    {"name": "llama3_8b",         "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "llama3_70b",        "B": 1, "H": 64, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    # Llama 4 Scout/Maverick (MoE, 17B active): same attn dims, H=40, H_kv=8, D=128
    {"name": "llama4_scout",      "B": 1, "H": 40, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},

    # --- Qwen 3 family (QK-norm: YES — "qk layernorm" confirmed in tech report) ---
    {"name": "qwen3_8b",          "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": True},
    {"name": "qwen3_32b",         "B": 1, "H": 64, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": True},
    # Qwen3-235B-A22B (MoE, 22B active): H=64, H_kv=4, D=128
    {"name": "qwen3_235b_a22b",   "B": 1, "H": 64, "H_kv": 4,  "S": 4096, "D": 128, "qk_norm": True},

    # --- Mistral / Mixtral (QK-norm: NO, RoPE, GQA) ---
    {"name": "mistral_7b",        "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    # Mixtral 8x22B (MoE, 39B active): H=48, H_kv=8, D=128
    {"name": "mixtral_8x22b",     "B": 1, "H": 48, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},

    # --- Phi family (QK-norm: NO confirmed for Phi-4, RoPE, GQA) ---
    {"name": "phi3_mini",         "B": 1, "H": 32, "H_kv": 32, "S": 4096, "D": 96,  "qk_norm": False},
    {"name": "phi4_mini",         "B": 1, "H": 24, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "phi4_14b",          "B": 1, "H": 40, "H_kv": 10, "S": 4096, "D": 128, "qk_norm": False},

    # --- Falcon 3 (QK-norm: NO, RoPE, GQA, head_dim=256) ---
    {"name": "falcon3_7b",        "B": 1, "H": 12, "H_kv": 4,  "S": 4096, "D": 256, "qk_norm": False},
    {"name": "falcon3_10b",       "B": 1, "H": 12, "H_kv": 4,  "S": 4096, "D": 256, "qk_norm": False},

    # --- DBRX (MoE, 36B active, QK-norm: NO, RoPE, GQA) ---
    {"name": "dbrx",              "B": 1, "H": 48, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},

    # --- OLMo 2 (QK-norm: YES — applies RMSNorm to Q and K, RoPE) ---
    {"name": "olmo2_7b",          "B": 1, "H": 32, "H_kv": 32, "S": 4096, "D": 128, "qk_norm": True},
    {"name": "olmo2_32b",         "B": 1, "H": 40, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": True},

    # --- InternLM 3 (QK-norm: NO confirmed, RoPE, GQA) ---
    {"name": "internlm3_8b",      "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
]

# T4-optimized: fewer warps reduce register pressure on 40-SM Turing
T4_CONFIGS = {
    96:  {"BLOCK_SIZE": 64,  "num_warps": 2, "num_stages": 1},
    128: {"BLOCK_SIZE": 64,  "num_warps": 2, "num_stages": 1},
    256: {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1},
    512: {"BLOCK_SIZE": 256, "num_warps": 2, "num_stages": 1},
}

# ============================================================================
# Separated baseline (what vLLM does: 4 separate PyTorch ops)
# ============================================================================

def make_separated_fn(q, k, cos, sin, q_scale, k_scale):
    """Return a closure that runs 4 separate ops (Q-RMSNorm, K-RMSNorm, Q-RoPE, K-RoPE)."""
    eps = 1e-6
    def separated_4ops():
        # 1. Q-RMSNorm with (1+w) affine
        q_var = q.float().pow(2).mean(-1, keepdim=True)
        q_n = (q.float() * torch.rsqrt(q_var + eps)).half() * (1.0 + q_scale).half()
        # 2. K-RMSNorm with (1+w) affine
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
    return separated_4ops

# ============================================================================
# Benchmark loop
# ============================================================================

def compute_bandwidth_gbps(model, fused_ms):
    """Estimate effective HBM bandwidth: read Q+K+scales+cos+sin, write Q+K."""
    B, H, H_kv, S, D = model["B"], model["H"], model["H_kv"], model["S"], model["D"]
    # fp16 tensors: Q(B,H,S,D) + K(B,H_kv,S,D) read + write = 2x each
    q_bytes = B * H * S * D * 2       # fp16
    k_bytes = B * H_kv * S * D * 2    # fp16
    # fp32 tensors: cos(S,D/2) + sin(S,D/2) + q_scale(D) + k_scale(D)
    trig_bytes = 2 * S * (D // 2) * 4  # cos + sin
    scale_bytes = 2 * D * 4            # q_scale + k_scale
    total_bytes = 2 * (q_bytes + k_bytes) + trig_bytes + scale_bytes  # read + write
    return (total_bytes / 1e9) / (fused_ms / 1e3)  # GB/s

def main():
    print("\n" + "=" * 74)
    print("MODEL FAMILY FUSION BENCHMARK: Fused QK-RMSNorm+RoPE (4 ops -> 1)")
    print("=" * 74)
    print(f"GPU: {GPU_NAME}")
    print(f"Models: {len(MODELS)} shapes across 13 model families")
    print("NOTE: Kernel uses (1+w) affine for all models. For LLaMA/Mistral")
    print("      (standard RMSNorm w*x), this is a weight convention difference")
    print("      only — kernel performance is identical either way.")
    print("      QK-Norm column marks models where the fused kernel directly")
    print("      replaces the QK-norm + RoPE pipeline (highest value).\n")

    header = f"{'Model':<22s} {'QK-Norm':>7s} {'Separated (ms)':>14s} {'Fused (ms)':>10s} {'Speedup':>8s} {'GB/s':>7s}"
    print(header)
    print("-" * len(header))

    results = []
    for m in MODELS:
        B, H, H_kv, S, D = m["B"], m["H"], m["H_kv"], m["S"], m["D"]
        config = T4_CONFIGS[D]

        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H_kv, S, D, device="cuda", dtype=torch.float16)
        cos = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
        sin = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
        q_scale = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1
        k_scale = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1

        sep_fn = make_separated_fn(q, k, cos, sin, q_scale, k_scale)
        fused_fn = lambda: apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config=config)

        sep_ms = cuda_event_timer(sep_fn)
        fused_ms = cuda_event_timer(fused_fn)
        speedup = sep_ms / fused_ms if fused_ms > 0 else 0.0
        gbps = compute_bandwidth_gbps(m, fused_ms)

        qk_tag = "YES" if m.get("qk_norm") else "no"
        print(f"{m['name']:<22s} {qk_tag:>7s} {sep_ms:>14.3f} {fused_ms:>10.3f} {speedup:>7.1f}x {gbps:>7.0f}")

        results.append({
            "model": m["name"],
            "shape": f"B{B}_H{H}_Hkv{H_kv}_S{S}_D{D}",
            "qk_norm": m.get("qk_norm", False),
            "config": config,
            "separated_ms": round(sep_ms, 4),
            "fused_ms": round(fused_ms, 4),
            "speedup": round(speedup, 2),
            "bandwidth_gbps": round(gbps, 1),
        })

        # Free VRAM between shapes
        del q, k, cos, sin, q_scale, k_scale
        torch.cuda.empty_cache()

    # Save JSON
    out_path = REPO / "multi_model_results.json"
    payload = {"gpu": GPU_NAME, "results": results}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
