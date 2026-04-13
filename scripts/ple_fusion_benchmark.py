#!/usr/bin/env python3
"""Standalone PLE fusion benchmark: fused vs separated baseline.

Compares Noeris fused PLE injection (1 matvec + 1 kernel) against the
separated PyTorch baseline (5 ops: lookup, project, RMSNorm, scale, add).

Shapes:
  - Gemma 4 E2B: D=1536, E=256, B=1, S=2048
  - Gemma 4 31B: D=5376, E=256, B=1, S=2048

Usage:
    python scripts/ple_fusion_benchmark.py
"""

import json
import platform

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def ple_norm_scale_add_kernel(
    hidden_ptr,
    projected_ptr,
    norm_weight_ptr,
    scale_ptr,
    M, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused RMSNorm(projected) * scale + hidden[row].

    One program per row. projected is broadcast (same for all rows).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    proj = tl.load(projected_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sq = proj * proj
    mean_sq = tl.sum(sq, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(norm_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = proj * rstd * w

    s = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scaled = normed * s

    h_base = hidden_ptr + row * D
    h = tl.load(h_base + offs, mask=mask, other=0.0).to(tl.float32)
    out = h + scaled
    tl.store(h_base + offs, out.to(tl.float16), mask=mask)


def apply_fused_ple(hidden, ple_embedding, proj_weight, norm_weight, scale, config, eps=1e-6):
    B, S, D = hidden.shape
    projected = ple_embedding.float() @ proj_weight.float().T
    M = B * S
    hidden_flat = hidden.reshape(M, D)
    BLOCK_D = max(config["BLOCK_D"], triton.next_power_of_2(D))
    ple_norm_scale_add_kernel[(M,)](
        hidden_flat, projected, norm_weight, scale,
        M, D, eps,
        BLOCK_D=BLOCK_D,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return hidden


# ---------------------------------------------------------------------------
# Separated baseline (5 separate PyTorch ops)
# ---------------------------------------------------------------------------

def separated_baseline(hidden, ple_embedding, proj_weight, norm_weight, scale, eps=1e-6):
    """5-op baseline: lookup (done), project, RMSNorm, scale, residual add."""
    # 1. Project
    projected = ple_embedding.float() @ proj_weight.float().T
    # 2. RMSNorm
    rms = projected.pow(2).mean().sqrt()
    normed = projected / (rms + eps) * norm_weight.float()
    # 3. Scale
    scaled = normed * scale.float()
    # 4. Residual add (broadcast)
    out = hidden.float() + scaled.unsqueeze(0).unsqueeze(0)
    return out.to(torch.float16)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

SHAPES = [
    {"name": "gemma4_e2b", "batch": 1, "seq_len": 2048, "hidden_dim": 1536, "embed_dim": 256},
    {"name": "gemma4_31b", "batch": 1, "seq_len": 2048, "hidden_dim": 5376, "embed_dim": 256},
    {"name": "gemma4_e2b_4k", "batch": 1, "seq_len": 4096, "hidden_dim": 1536, "embed_dim": 256},
]

CONFIGS = [
    {"BLOCK_D": 2048, "num_warps": 4, "num_stages": 1},
    {"BLOCK_D": 2048, "num_warps": 8, "num_stages": 1},
    {"BLOCK_D": 4096, "num_warps": 8, "num_stages": 1},
    {"BLOCK_D": 4096, "num_warps": 4, "num_stages": 2},
]


def benchmark_one(shape, config):
    B = shape["batch"]
    S = shape["seq_len"]
    D = shape["hidden_dim"]
    E = shape["embed_dim"]

    hidden = torch.randn((B, S, D), device="cuda", dtype=torch.float16)
    ple_embedding = torch.randn((E,), device="cuda", dtype=torch.float16) * 0.01
    proj_weight = torch.randn((D, E), device="cuda", dtype=torch.float16) * 0.02
    norm_weight = torch.ones((D,), device="cuda", dtype=torch.float32)
    scale = torch.ones((D,), device="cuda", dtype=torch.float32) * 0.1

    # Correctness
    ref = separated_baseline(hidden, ple_embedding, proj_weight, norm_weight, scale)
    hidden_copy = hidden.clone()
    out = apply_fused_ple(hidden_copy, ple_embedding, proj_weight, norm_weight, scale, config)
    max_err = (out - ref).abs().max().item()

    if max_err > 0.1:
        return {"correct": False, "max_err": max_err}

    # Timing
    def fused_fn():
        h = hidden.clone()
        apply_fused_ple(h, ple_embedding, proj_weight, norm_weight, scale, config)

    ms = triton.testing.do_bench(fused_fn, warmup=25, rep=100)
    sep_ms = triton.testing.do_bench(
        lambda: separated_baseline(hidden, ple_embedding, proj_weight, norm_weight, scale),
        warmup=25, rep=100,
    )

    M = B * S
    bytes_moved = M * D * 2 + D * 4 * 3 + M * D * 2
    gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
    speedup = sep_ms / ms if ms > 0 else 0.0

    return {
        "correct": True,
        "max_err": round(max_err, 6),
        "fused_ms": round(ms, 4),
        "separated_ms": round(sep_ms, 4),
        "speedup": round(speedup, 2),
        "gb_per_s": round(gb_per_s, 2),
    }


def main():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    results = []
    for shape in SHAPES:
        for config in CONFIGS:
            cid = f"bd{config['BLOCK_D']}_w{config['num_warps']}_s{config['num_stages']}"
            result = benchmark_one(shape, config)
            result["shape"] = shape["name"]
            result["config"] = cid
            results.append(result)

            status = "PASS" if result.get("correct") else "FAIL"
            speedup = result.get("speedup", "N/A")
            fused = result.get("fused_ms", "N/A")
            sep = result.get("separated_ms", "N/A")
            bw = result.get("gb_per_s", "N/A")
            print(f"  [{status}] {shape['name']:20s} {cid:25s}  "
                  f"fused={fused}ms  sep={sep}ms  speedup={speedup}x  bw={bw} GB/s")

    print()
    print(json.dumps({"operator": "ple_fusion", "gpu": gpu_name, "results": results}, indent=2))


if __name__ == "__main__":
    main()
