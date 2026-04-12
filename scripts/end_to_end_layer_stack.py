#!/usr/bin/env python3
"""End-to-end Gemma 4 E2B forward pass: 26-layer Noeris fused vs PyTorch baseline.

Usage (Kaggle T4):
  !pip install -e . numpy scikit-learn -q && python scripts/end_to_end_layer_stack.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F

print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: No GPU available. Change runtime to T4 GPU.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")

import triton
print(f"Triton {triton.__version__}")

from research_engine.triton_rmsnorm import rmsnorm
from research_engine.triton_qk_norm_rope import apply_qk_norm_rope
from research_engine.triton_geglu import geglu

# -- Gemma 4 E2B shapes --
N_LAYERS = 26
B, S, D = 1, 2048, 1536
H, H_KV, DH = 8, 1, 256
D_QKV = (H + 2 * H_KV) * DH  # 2560
D_FF = 6144
EPS = 1e-6
# T4-tuned Triton configs
CFG_RMSNORM = {"BLOCK_SIZE": 2048, "num_warps": 2, "num_stages": 1}
CFG_QK_ROPE = {"BLOCK_SIZE": 128,  "num_warps": 2, "num_stages": 1}
CFG_GEGLU   = {"BLOCK_SIZE": 1024, "num_warps": 2, "num_stages": 1}

def cuda_event_timer(fn, warmup=5, trials=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]

# -- Baseline (PyTorch native) ops --
def baseline_rmsnorm(x, w):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + EPS)).half() * (1.0 + w).half()

def baseline_qk_norm_rope(q, k, cos, sin, q_scale, k_scale):
    q_n, k_n = baseline_rmsnorm(q, q_scale), baseline_rmsnorm(k, k_scale)
    c, sn = cos[None, None, :, :].half(), sin[None, None, :, :].half()
    qe, qo = q_n[..., 0::2], q_n[..., 1::2]
    q_out = torch.stack([qe * c - qo * sn, qe * sn + qo * c], dim=-1).reshape(q.shape)
    ke, ko = k_n[..., 0::2], k_n[..., 1::2]
    k_out = torch.stack([ke * c - ko * sn, ke * sn + ko * c], dim=-1).reshape(k.shape)
    return q_out, k_out

def baseline_geglu(gate, up):
    return gate * F.gelu(up, approximate="tanh")

# -- Layer weights (pre-allocated, unique per layer) --
def make_layer_weights(device="cuda"):
    return {
        "attn_norm_w":  torch.randn(D, device=device, dtype=torch.float16) * 0.02,
        "qkv_w":        torch.randn(D, D_QKV, device=device, dtype=torch.float16) * 0.02,
        "q_scale":      torch.randn(DH, device=device, dtype=torch.float32) * 0.1,
        "k_scale":      torch.randn(DH, device=device, dtype=torch.float32) * 0.1,
        "o_proj_w":     torch.randn(H * DH, D, device=device, dtype=torch.float16) * 0.02,
        "mlp_norm_w":   torch.randn(D, device=device, dtype=torch.float16) * 0.02,
        "gate_up_w":    torch.randn(D, 2 * D_FF, device=device, dtype=torch.float16) * 0.02,
        "down_w":       torch.randn(D_FF, D, device=device, dtype=torch.float16) * 0.02,
    }

# -- Forward pass (one layer) --
def layer_forward(x, w, cos, sin, *, use_noeris=False):
    BS = B * S
    # Pre-attention RMSNorm
    x_flat = x.reshape(BS, D)
    if use_noeris:
        h = rmsnorm(x_flat, w["attn_norm_w"], config=CFG_RMSNORM, eps=EPS, affine_mode=1)
    else:
        h = baseline_rmsnorm(x_flat, w["attn_norm_w"])
    h = h.reshape(B, S, D)

    # QKV projection
    qkv = h.reshape(BS, D) @ w["qkv_w"]
    q = qkv[:, :H * DH].reshape(B, H, S, DH)
    k = qkv[:, H * DH:H * DH + H_KV * DH].reshape(B, H_KV, S, DH)
    v = qkv[:, H * DH + H_KV * DH:].reshape(B, H_KV, S, DH)

    # QK-RMSNorm + RoPE
    if use_noeris:
        q, k = apply_qk_norm_rope(q, k, cos, sin, w["q_scale"], w["k_scale"],
                                   config=CFG_QK_ROPE, eps=EPS)
    else:
        q, k = baseline_qk_norm_rope(q, k, cos, sin, w["q_scale"], w["k_scale"])

    # SDPA (identical both paths, GQA expand)
    k_exp = k.expand(B, H, S, DH)
    v_exp = v.expand(B, H, S, DH)
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

    # Output projection + residual
    o = attn_out.reshape(BS, H * DH) @ w["o_proj_w"]
    x = x + o.reshape(B, S, D)

    # Pre-MLP RMSNorm
    x_flat = x.reshape(BS, D)
    if use_noeris:
        h = rmsnorm(x_flat, w["mlp_norm_w"], config=CFG_RMSNORM, eps=EPS, affine_mode=1)
    else:
        h = baseline_rmsnorm(x_flat, w["mlp_norm_w"])

    # GeGLU MLP
    gate_up = h @ w["gate_up_w"]
    gate, up = gate_up.chunk(2, dim=-1)
    if use_noeris:
        mlp_h = geglu(gate, up, config=CFG_GEGLU)
    else:
        mlp_h = baseline_geglu(gate, up)

    # Down projection + residual
    down = mlp_h @ w["down_w"]
    x = x + down.reshape(B, S, D)
    return x

def full_forward(x, layers, cos, sin, *, use_noeris=False):
    for w in layers:
        x = layer_forward(x, w, cos, sin, use_noeris=use_noeris)
    return x

def main():
    print(f"\nAllocating {N_LAYERS}-layer model weights (B={B}, S={S}, D={D}, "
          f"H={H}, H_kv={H_KV}, Dh={DH}, Dff={D_FF})...")
    layers = [make_layer_weights() for _ in range(N_LAYERS)]

    # Precompute RoPE sin/cos tables
    cos = torch.randn(S, DH // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, DH // 2, device="cuda", dtype=torch.float32)

    # Input
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)

    print("Warming up and benchmarking...\n")

    baseline_fn = lambda: full_forward(x, layers, cos, sin, use_noeris=False)
    noeris_fn   = lambda: full_forward(x, layers, cos, sin, use_noeris=True)

    baseline_ms = cuda_event_timer(baseline_fn, warmup=5, trials=10)
    noeris_ms   = cuda_event_timer(noeris_fn,   warmup=5, trials=10)

    speedup = baseline_ms / noeris_ms if noeris_ms > 0 else 0.0
    per_layer_savings = (baseline_ms - noeris_ms) / N_LAYERS

    print("END-TO-END GEMMA 4 E2B (26 LAYERS)")
    print("=" * 36)
    print(f"Baseline (PyTorch):  {baseline_ms:>7.1f} ms")
    print(f"Noeris (fused):      {noeris_ms:>7.1f} ms")
    print(f"Speedup:             {speedup:>7.2f}x")
    print(f"Per-layer savings:   {per_layer_savings:>7.2f} ms")

    results = {
        "gpu": GPU_NAME,
        "model": "gemma4_e2b",
        "n_layers": N_LAYERS,
        "shapes": {"B": B, "S": S, "D": D, "H": H, "H_kv": H_KV, "Dh": DH, "Dff": D_FF},
        "configs": {
            "rmsnorm": CFG_RMSNORM,
            "qk_norm_rope": CFG_QK_ROPE,
            "geglu": CFG_GEGLU,
        },
        "baseline_ms": round(baseline_ms, 2),
        "noeris_ms": round(noeris_ms, 2),
        "speedup": round(speedup, 3),
        "per_layer_savings_ms": round(per_layer_savings, 3),
    }

    out_path = REPO / "end_to_end_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
