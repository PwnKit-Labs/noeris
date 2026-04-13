#!/usr/bin/env python3
"""Run 26-layer Gemma 4 E2B end-to-end benchmark on Modal A100.

Generates a self-contained script (all three Triton kernels inlined),
sends it to a Modal A100-40GB container, saves results, and prints
T4 vs A100 comparison.

Usage:
    python3.11 scripts/modal_e2e_a100.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from research_engine.modal_session import ModalBenchmarkSession

# ---------------------------------------------------------------------------
# Self-contained benchmark script sent to Modal
# ---------------------------------------------------------------------------

REMOTE_SCRIPT = r'''
import json
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

GPU_NAME = torch.cuda.get_device_name(0)
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {GPU_NAME}")
print(f"Triton {triton.__version__}")

# -- Gemma 4 E2B shapes --
N_LAYERS = 26
B, S, D = 1, 2048, 1536
H, H_KV, DH = 8, 1, 256
D_QKV = (H + 2 * H_KV) * DH  # 2560
D_FF = 6144
EPS = 1e-6

# A100-tuned Triton configs: more warps than T4 (108 SMs vs 40)
CFG_RMSNORM = {"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 1}
CFG_QK_ROPE = {"BLOCK_SIZE": 128,  "num_warps": 4, "num_stages": 1}
CFG_GEGLU   = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1}


# =====================================================================
# Triton kernel 1: RMSNorm (Gemma-mode affine)
# =====================================================================

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    x_row_stride, y_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
    AFFINE_MODE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if AFFINE_MODE == 0:
        y = x * rstd * w
    else:
        y = x * rstd * (1.0 + w)
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


def rmsnorm(x, w, config, eps=1e-6, affine_mode=0):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    rmsnorm_kernel[(n_rows,)](
        x, w, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        AFFINE_MODE=affine_mode,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


# =====================================================================
# Triton kernel 2: Fused QK-RMSNorm + RoPE
# =====================================================================

@triton.jit
def qk_norm_rope_kernel(
    x_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
    row_stride,
    heads, seq_len, head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
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


def apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config, eps=1e-6):
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
    qk_norm_rope_kernel[(B * H * S,)](
        q_flat, q_scale, cos_c, sin_c, q_out_flat,
        D, H, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    qk_norm_rope_kernel[(B * H_kv * S,)](
        k_flat, k_scale, cos_c, sin_c, k_out_flat,
        D, H_kv, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return q_out, k_out


# =====================================================================
# Triton kernel 3: Fused GeGLU
# =====================================================================

@triton.jit
def geglu_kernel(
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


def geglu(gate, up, config):
    n_rows, n_cols = gate.shape
    out = torch.empty_like(gate)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    geglu_kernel[(n_rows,)](
        gate, up, out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


# =====================================================================
# Baselines (PyTorch native)
# =====================================================================

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


# =====================================================================
# Timer
# =====================================================================

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


# =====================================================================
# Layer weights + forward pass
# =====================================================================

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


def layer_forward(x, w, cos, sin, *, use_noeris=False):
    BS = B * S
    x_flat = x.reshape(BS, D)
    if use_noeris:
        h = rmsnorm(x_flat, w["attn_norm_w"], config=CFG_RMSNORM, eps=EPS, affine_mode=1)
    else:
        h = baseline_rmsnorm(x_flat, w["attn_norm_w"])
    h = h.reshape(B, S, D)

    qkv = h.reshape(BS, D) @ w["qkv_w"]
    q = qkv[:, :H * DH].reshape(B, H, S, DH)
    k = qkv[:, H * DH:H * DH + H_KV * DH].reshape(B, H_KV, S, DH)
    v = qkv[:, H * DH + H_KV * DH:].reshape(B, H_KV, S, DH)

    if use_noeris:
        q, k = apply_qk_norm_rope(q, k, cos, sin, w["q_scale"], w["k_scale"],
                                   config=CFG_QK_ROPE, eps=EPS)
    else:
        q, k = baseline_qk_norm_rope(q, k, cos, sin, w["q_scale"], w["k_scale"])

    k_exp = k.expand(B, H, S, DH)
    v_exp = v.expand(B, H, S, DH)
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

    o = attn_out.reshape(BS, H * DH) @ w["o_proj_w"]
    x = x + o.reshape(B, S, D)

    x_flat = x.reshape(BS, D)
    if use_noeris:
        h = rmsnorm(x_flat, w["mlp_norm_w"], config=CFG_RMSNORM, eps=EPS, affine_mode=1)
    else:
        h = baseline_rmsnorm(x_flat, w["mlp_norm_w"])

    gate_up = h @ w["gate_up_w"]
    gate, up = gate_up.chunk(2, dim=-1)
    if use_noeris:
        mlp_h = geglu(gate, up, config=CFG_GEGLU)
    else:
        mlp_h = baseline_geglu(gate, up)

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
    cos = torch.randn(S, DH // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, DH // 2, device="cuda", dtype=torch.float32)
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

    # Print JSON for ModalBenchmarkSession to extract
    print(json.dumps({"config_results": [results], **results}))


if __name__ == "__main__":
    main()
'''


def run_a100_e2e() -> dict:
    """Send E2E benchmark to Modal A100, return parsed results."""
    print("Launching 26-layer Gemma 4 E2B end-to-end benchmark on Modal A100-40GB...")
    print("(Budget cap: $0.30, timeout: 300s)\n")

    with ModalBenchmarkSession(gpu="A100", timeout_seconds=300, max_cost_usd=0.30) as session:
        result = session.run_script(REMOTE_SCRIPT)

    if not result.success:
        print(f"FAILED: {result.error}", file=sys.stderr)
        if result.stderr:
            print(f"STDERR:\n{result.stderr[:2000]}", file=sys.stderr)
        if result.stdout:
            print(f"STDOUT preview:\n{result.stdout[:2000]}", file=sys.stderr)
        sys.exit(1)

    stdout = result.stdout
    print("--- Remote stdout ---")
    print(stdout[:3000])
    print("---")

    # Extract JSON from stdout (last JSON object)
    json_start = stdout.rfind('{"config_results"')
    if json_start < 0:
        # Try the session's parsed config_results
        if result.config_results:
            return result.config_results[0]
        print("ERROR: No JSON found in stdout", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(stdout[json_start:])
    return payload


def main():
    # Load T4 results
    t4_path = REPO / "docs" / "results" / "t4-end-to-end-26layer.json"
    if not t4_path.exists():
        print(f"ERROR: T4 results not found at {t4_path}", file=sys.stderr)
        sys.exit(1)
    t4 = json.loads(t4_path.read_text())
    print(f"T4 results: {t4['baseline_ms']}ms baseline, {t4['noeris_ms']}ms fused, "
          f"{t4['speedup']}x speedup on {t4['gpu']}\n")

    # Run A100 benchmark
    a100 = run_a100_e2e()

    # Save results
    out_path = REPO / "docs" / "results" / "a100-end-to-end-26layer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove config_results wrapper if present
    save_data = {k: v for k, v in a100.items() if k != "config_results"}
    out_path.write_text(json.dumps(save_data, indent=2))
    print(f"\nA100 results saved to {out_path}\n")

    # Print comparison
    print("=" * 60)
    print("CROSS-HARDWARE COMPARISON: T4 vs A100")
    print("END-TO-END GEMMA 4 E2B (26 LAYERS)")
    print("=" * 60)
    print(f"{'':>20s} {'T4':>12s} {'A100':>12s}")
    print("-" * 48)
    print(f"{'Baseline (ms)':>20s} {t4['baseline_ms']:>12.2f} {a100['baseline_ms']:>12.2f}")
    print(f"{'Noeris (ms)':>20s} {t4['noeris_ms']:>12.2f} {a100['noeris_ms']:>12.2f}")
    print(f"{'Speedup':>20s} {t4['speedup']:>11.3f}x {a100['speedup']:>11.3f}x")
    print(f"{'Per-layer savings':>20s} {t4['per_layer_savings_ms']:>11.3f}ms {a100['per_layer_savings_ms']:>11.3f}ms")
    print()
    print(f"A100 raw throughput: {t4['baseline_ms'] / a100['baseline_ms']:.1f}x faster than T4 baseline")
    print(f"A100 fused speedup: {a100['speedup']:.3f}x  (T4: {t4['speedup']:.3f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
