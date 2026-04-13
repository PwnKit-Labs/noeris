#!/usr/bin/env python3
"""Run multi-model fusion benchmark on Modal A100, compare rankings with T4.

Generates a self-contained benchmark script (Triton kernel inlined),
sends it to a Modal A100-40GB container, saves results, and computes
Spearman rho between T4 and A100 speedup rankings.

Usage:
    python3.11 scripts/modal_multimodel_a100.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from research_engine.modal_session import ModalBenchmarkSession

# ---------------------------------------------------------------------------
# Self-contained benchmark script sent to Modal
# ---------------------------------------------------------------------------

REMOTE_SCRIPT = r'''
import json
import platform
import torch
import triton
import triton.language as tl

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}", flush=True)

# ---------------------------------------------------------------------------
# Triton fused QK-RMSNorm+RoPE kernel (inlined from triton_qk_norm_rope.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Separated baseline (vLLM-style 4 separate ops)
# ---------------------------------------------------------------------------

def make_separated_fn(q, k, cos, sin, q_scale, k_scale):
    eps = 1e-6
    def separated_4ops():
        q_var = q.float().pow(2).mean(-1, keepdim=True)
        q_n = (q.float() * torch.rsqrt(q_var + eps)).half() * (1.0 + q_scale).half()
        k_var = k.float().pow(2).mean(-1, keepdim=True)
        k_n = (k.float() * torch.rsqrt(k_var + eps)).half() * (1.0 + k_scale).half()
        c = cos[None, None, :, :].half()
        sn = sin[None, None, :, :].half()
        qe, qo = q_n[..., 0::2], q_n[..., 1::2]
        q_out = torch.stack([qe * c - qo * sn, qe * sn + qo * c], dim=-1).reshape(q.shape)
        ke, ko = k_n[..., 0::2], k_n[..., 1::2]
        k_out = torch.stack([ke * c - ko * sn, ke * sn + ko * c], dim=-1).reshape(k.shape)
        return q_out, k_out
    return separated_4ops


# ---------------------------------------------------------------------------
# CUDA event timer
# ---------------------------------------------------------------------------

def cuda_event_timer(fn, warmup=5, trials=20):
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


# ---------------------------------------------------------------------------
# Model definitions (same 19 as T4 benchmark)
# ---------------------------------------------------------------------------

MODELS = [
    {"name": "gemma4_e2b_local",  "B": 1, "H": 8,  "H_kv": 1,  "S": 4096, "D": 256, "qk_norm": True},
    {"name": "gemma4_31b_global", "B": 1, "H": 32, "H_kv": 4,  "S": 4096, "D": 512, "qk_norm": True},
    {"name": "llama3_8b",         "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "llama3_70b",        "B": 1, "H": 64, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "llama4_scout",      "B": 1, "H": 40, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "qwen3_8b",          "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": True},
    {"name": "qwen3_32b",         "B": 1, "H": 64, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": True},
    {"name": "qwen3_235b_a22b",   "B": 1, "H": 64, "H_kv": 4,  "S": 4096, "D": 128, "qk_norm": True},
    {"name": "mistral_7b",        "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "mixtral_8x22b",     "B": 1, "H": 48, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "phi3_mini",         "B": 1, "H": 32, "H_kv": 32, "S": 4096, "D": 96,  "qk_norm": False},
    {"name": "phi4_mini",         "B": 1, "H": 24, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "phi4_14b",          "B": 1, "H": 40, "H_kv": 10, "S": 4096, "D": 128, "qk_norm": False},
    {"name": "falcon3_7b",        "B": 1, "H": 12, "H_kv": 4,  "S": 4096, "D": 256, "qk_norm": False},
    {"name": "falcon3_10b",       "B": 1, "H": 12, "H_kv": 4,  "S": 4096, "D": 256, "qk_norm": False},
    {"name": "dbrx",              "B": 1, "H": 48, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
    {"name": "olmo2_7b",          "B": 1, "H": 32, "H_kv": 32, "S": 4096, "D": 128, "qk_norm": True},
    {"name": "olmo2_32b",         "B": 1, "H": 40, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": True},
    {"name": "internlm3_8b",      "B": 1, "H": 32, "H_kv": 8,  "S": 4096, "D": 128, "qk_norm": False},
]

# A100-tuned configs: A100 has 108 SMs, more register file, wider warps
# Use same per-head-dim mapping but with more warps for A100
A100_CONFIGS = {
    96:  {"BLOCK_SIZE": 64,  "num_warps": 4, "num_stages": 2},
    128: {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2},
    256: {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2},
    512: {"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 2},
}


def compute_bandwidth_gbps(model, fused_ms):
    B, H, H_kv, S, D = model["B"], model["H"], model["H_kv"], model["S"], model["D"]
    q_bytes = B * H * S * D * 2
    k_bytes = B * H_kv * S * D * 2
    trig_bytes = 2 * S * (D // 2) * 4
    scale_bytes = 2 * D * 4
    total_bytes = 2 * (q_bytes + k_bytes) + trig_bytes + scale_bytes
    return (total_bytes / 1e9) / (fused_ms / 1e3)


def main():
    results = []
    for m in MODELS:
        B, H, H_kv, S, D = m["B"], m["H"], m["H_kv"], m["S"], m["D"]
        config = A100_CONFIGS[D]

        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H_kv, S, D, device="cuda", dtype=torch.float16)
        cos = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
        sin = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
        q_scale = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1
        k_scale = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1

        sep_fn = make_separated_fn(q, k, cos, sin, q_scale, k_scale)
        fused_fn = lambda q=q, k=k, cos=cos, sin=sin, q_scale=q_scale, k_scale=k_scale, config=config: \
            apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config=config)

        sep_ms = cuda_event_timer(sep_fn)
        fused_ms = cuda_event_timer(fused_fn)
        speedup = sep_ms / fused_ms if fused_ms > 0 else 0.0
        gbps = compute_bandwidth_gbps(m, fused_ms)

        print(f"{m['name']:<22s} sep={sep_ms:.3f}ms  fused={fused_ms:.3f}ms  {speedup:.1f}x  {gbps:.0f} GB/s", flush=True)

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

        del q, k, cos, sin, q_scale, k_scale
        torch.cuda.empty_cache()

    # Print JSON payload (ModalBenchmarkSession extracts the first JSON object)
    payload = {"gpu": GPU_NAME, "config_results": results, "results": results}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
'''


def run_a100_benchmark() -> dict:
    """Send self-contained benchmark to Modal A100, return parsed results."""
    print("Launching multi-model fusion benchmark on Modal A100-40GB...")
    print("(Budget cap: $1.00, timeout: 600s)\n")

    with ModalBenchmarkSession(gpu="A100", timeout_seconds=600, max_cost_usd=1.00) as session:
        result = session.run_script(REMOTE_SCRIPT)

    if not result.success:
        print(f"FAILED: {result.error}", file=sys.stderr)
        if result.stderr:
            print(f"STDERR:\n{result.stderr[:1000]}", file=sys.stderr)
        if result.stdout:
            print(f"STDOUT preview:\n{result.stdout[:1000]}", file=sys.stderr)
        sys.exit(1)

    # The session extracts config_results; but we also want the full stdout
    # to get the GPU name. Parse from stdout directly.
    stdout = result.stdout
    json_start = stdout.rfind('{"gpu"')
    if json_start < 0:
        # Fall back to config_results from session
        print("Warning: could not find JSON in stdout, using session results")
        return {"gpu": "A100", "results": result.config_results}

    payload = json.loads(stdout[json_start:])
    return payload


def compare_rankings(t4_results: list[dict], a100_results: list[dict]) -> float:
    """Compute Spearman rho between T4 and A100 speedup rankings."""
    # Build model -> speedup maps
    t4_map = {r["model"]: r["speedup"] for r in t4_results}
    a100_map = {r["model"]: r["speedup"] for r in a100_results}

    # Only compare models present in both
    common = sorted(set(t4_map.keys()) & set(a100_map.keys()))
    if len(common) < 3:
        print(f"ERROR: Only {len(common)} common models, need at least 3")
        return 0.0

    t4_speedups = [t4_map[m] for m in common]
    a100_speedups = [a100_map[m] for m in common]

    rho, pvalue = spearmanr(t4_speedups, a100_speedups)
    return rho, pvalue, common, t4_speedups, a100_speedups


def main():
    # Load T4 results
    t4_path = REPO / "docs" / "results" / "t4-19model-generalization.json"
    if not t4_path.exists():
        print(f"ERROR: T4 results not found at {t4_path}", file=sys.stderr)
        sys.exit(1)
    t4_data = json.loads(t4_path.read_text())
    t4_results = t4_data["results"]
    print(f"Loaded T4 results: {len(t4_results)} models on {t4_data['gpu']}\n")

    # Run A100 benchmark
    a100_data = run_a100_benchmark()
    a100_results = a100_data.get("results", a100_data.get("config_results", []))
    gpu_name = a100_data.get("gpu", "A100")
    print(f"\nA100 benchmark complete: {len(a100_results)} models on {gpu_name}\n")

    # Save A100 results
    out_path = REPO / "docs" / "results" / "a100-19model-generalization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {"gpu": gpu_name, "results": a100_results}
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"A100 results saved to {out_path}\n")

    # Print side-by-side comparison
    print("=" * 80)
    print("CROSS-HARDWARE COMPARISON: T4 vs A100 fusion speedups")
    print("=" * 80)
    t4_map = {r["model"]: r for r in t4_results}
    a100_map = {r["model"]: r for r in a100_results}
    common = sorted(set(t4_map.keys()) & set(a100_map.keys()))

    print(f"\n{'Model':<22s} {'QK-Norm':>7s} {'T4 speedup':>11s} {'A100 speedup':>13s} {'T4 GB/s':>8s} {'A100 GB/s':>10s}")
    print("-" * 75)
    for m in common:
        t4 = t4_map[m]
        a100 = a100_map[m]
        qk = "YES" if t4.get("qk_norm") else "no"
        print(f"{m:<22s} {qk:>7s} {t4['speedup']:>10.2f}x {a100['speedup']:>12.2f}x "
              f"{t4['bandwidth_gbps']:>7.0f} {a100['bandwidth_gbps']:>9.0f}")

    # Compute Spearman rho
    rho, pvalue, models, t4_sp, a100_sp = compare_rankings(t4_results, a100_results)

    print(f"\n{'=' * 60}")
    print(f"Spearman rho (speedup ranking): {rho:.4f}")
    print(f"p-value:                        {pvalue:.2e}")
    print(f"Models compared:                {len(models)}")
    print(f"{'=' * 60}")

    if rho > 0.8:
        print("\nSTRONG cross-hardware transfer: config rankings are highly")
        print("correlated between T4 (Turing) and A100 (Ampere).")
    elif rho > 0.5:
        print("\nMODERATE cross-hardware transfer.")
    else:
        print("\nWEAK cross-hardware transfer -- rankings differ significantly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
