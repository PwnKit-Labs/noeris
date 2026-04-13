#!/usr/bin/env python3
"""Standalone SSM scan benchmark: Noeris bandit configs vs Mamba-3 fixed defaults.

Compares parameterized Triton selective scan kernel across Mamba-3 model
sizes (130M, 370M, 1.3B, 2.8B) to prove architecture-agnostic autotuning.

The selective scan is a sequential recurrence — fundamentally different
from the parallel reductions in transformer attention:
    h[t] = A * h[t-1] + B * x[t]
    y[t] = C * h[t]

Usage:
    python scripts/ssm_scan_benchmark.py
"""

import json
import platform
import sys

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton selective scan kernel
# ---------------------------------------------------------------------------

@triton.jit
def selective_scan_kernel(
    x_ptr, A_ptr, B_ptr, C_ptr, y_ptr,
    batch, seq_len, d_model, d_state,
    stride_x_b, stride_x_s, stride_x_d,
    stride_A_b, stride_A_s, stride_A_d,
    stride_B_b, stride_B_s, stride_B_n,
    stride_C_b, stride_C_s, stride_C_n,
    stride_y_b, stride_y_s, stride_y_d,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """Selective SSM scan: one program per (batch, d_model_channel).

    Runs the recurrence h[t] = exp(A[t]) * h[t-1] + B[t] * x[t] across
    the full sequence, then outputs y[t] = C[t]^T * h[t].
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_DSTATE)
    n_mask = offs_n < d_state

    # Hidden state: [BLOCK_DSTATE]
    h = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    for t in range(seq_len):
        x_val = tl.load(
            x_ptr + pid_b * stride_x_b + t * stride_x_s + pid_d * stride_x_d,
        ).to(tl.float32)

        a_val = tl.load(
            A_ptr + pid_b * stride_A_b + t * stride_A_s + pid_d * stride_A_d,
        ).to(tl.float32)

        b_ptrs = B_ptr + pid_b * stride_B_b + t * stride_B_s + offs_n * stride_B_n
        b_vals = tl.load(b_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        c_ptrs = C_ptr + pid_b * stride_C_b + t * stride_C_s + offs_n * stride_C_n
        c_vals = tl.load(c_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        a_discrete = tl.exp(a_val)
        h = a_discrete * h + b_vals * x_val

        y_val = tl.sum(c_vals * h, axis=0)
        tl.store(
            y_ptr + pid_b * stride_y_b + t * stride_y_s + pid_d * stride_y_d,
            y_val,
        )


def triton_selective_scan(x, A, B, C, config):
    """Launch Triton selective scan."""
    batch, seq_len, d_model = x.shape
    _, _, d_state = B.shape

    y = torch.empty((batch, seq_len, d_model), device=x.device, dtype=torch.float32)

    grid = (batch, d_model)
    selective_scan_kernel[grid](
        x, A, B, C, y,
        batch, seq_len, d_model, d_state,
        x.stride(0), x.stride(1), x.stride(2),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SEQ=config["BLOCK_SEQ"],
        BLOCK_DSTATE=triton.next_power_of_2(d_state),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


# ---------------------------------------------------------------------------
# PyTorch baseline (sequential loop)
# ---------------------------------------------------------------------------

def torch_selective_scan(x, A, B, C):
    """PyTorch reference: pure sequential scan."""
    batch, seq_len, d_model = x.shape
    _, _, d_state = B.shape

    h = torch.zeros(batch, d_model, d_state, device=x.device, dtype=torch.float32)
    y = torch.empty(batch, seq_len, d_model, device=x.device, dtype=torch.float32)

    A_discrete = torch.exp(A.float())

    for t in range(seq_len):
        h = A_discrete[:, t, :].unsqueeze(-1) * h + B[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1).float()
        y[:, t, :] = (C[:, t, :].unsqueeze(1) * h).sum(-1)

    return y


# ---------------------------------------------------------------------------
# Shapes and configs
# ---------------------------------------------------------------------------

MAMBA3_SHAPES = [
    {"name": "mamba3_130m", "batch": 1, "seq_len": 2048, "d_model": 768, "d_state": 64},
    {"name": "mamba3_370m", "batch": 1, "seq_len": 2048, "d_model": 1024, "d_state": 64},
    {"name": "mamba3_1.3b", "batch": 1, "seq_len": 2048, "d_model": 2048, "d_state": 64},
    {"name": "mamba3_2.8b", "batch": 1, "seq_len": 2048, "d_model": 2560, "d_state": 128},
]

# Mamba-3 shipped fixed config (baseline)
MAMBA3_FIXED_CONFIG = {"BLOCK_SEQ": 128, "BLOCK_DSTATE": 64, "num_warps": 4, "num_stages": 2}

# Noeris search candidates
NOERIS_SEARCH_CONFIGS = [
    {"BLOCK_SEQ": 32, "BLOCK_DSTATE": 32, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SEQ": 64, "BLOCK_DSTATE": 32, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SEQ": 64, "BLOCK_DSTATE": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SEQ": 128, "BLOCK_DSTATE": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SEQ": 128, "BLOCK_DSTATE": 64, "num_warps": 4, "num_stages": 2},  # == fixed
    {"BLOCK_SEQ": 256, "BLOCK_DSTATE": 64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SEQ": 512, "BLOCK_DSTATE": 64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SEQ": 32, "BLOCK_DSTATE": 16, "num_warps": 4, "num_stages": 1},
]


def config_id(cfg):
    return f"bs{cfg['BLOCK_SEQ']}_bd{cfg['BLOCK_DSTATE']}_w{cfg['num_warps']}_s{cfg['num_stages']}"


def benchmark_one(shape, config):
    """Benchmark a single (shape, config) pair."""
    batch = shape.get("batch", 1)
    seq_len = shape["seq_len"]
    d_model = shape["d_model"]
    d_state = shape["d_state"]

    try:
        torch.manual_seed(0)
        x = torch.randn(batch, seq_len, d_model, device="cuda", dtype=torch.float32) * 0.1
        A = torch.randn(batch, seq_len, d_model, device="cuda", dtype=torch.float32) * 0.01 - 1.0
        B = torch.randn(batch, seq_len, d_state, device="cuda", dtype=torch.float32) * 0.1
        C = torch.randn(batch, seq_len, d_state, device="cuda", dtype=torch.float32) * 0.1

        # Correctness
        ref_y = torch_selective_scan(x, A, B, C)
        out_y = triton_selective_scan(x, A, B, C, config)
        max_err = (out_y - ref_y).abs().max().item()
        if max_err > 0.1:
            return {"correct": False, "max_err": round(max_err, 6), "ms": None}

        # Timing
        ms = triton.testing.do_bench(
            lambda: triton_selective_scan(x, A, B, C, config),
            warmup=25, rep=100,
        )

        x_bytes = batch * seq_len * d_model * 4
        a_bytes = batch * seq_len * d_model * 4
        b_bytes = batch * seq_len * d_state * 4
        c_bytes = batch * seq_len * d_state * 4
        y_bytes = batch * seq_len * d_model * 4
        bytes_moved = x_bytes + a_bytes + b_bytes + c_bytes + y_bytes
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9

        return {
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "gb_per_s": round(gb_per_s, 2),
        }
    except Exception as exc:
        return {"correct": False, "error": str(exc)[:200], "ms": None}


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — cannot run benchmark.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Python: {platform.python_version()}")
    print()

    all_configs = [MAMBA3_FIXED_CONFIG] + NOERIS_SEARCH_CONFIGS
    # Deduplicate
    seen = set()
    unique_configs = []
    for c in all_configs:
        cid = config_id(c)
        if cid not in seen:
            seen.add(cid)
            unique_configs.append(c)

    results = {}
    for shape in MAMBA3_SHAPES:
        shape_name = shape["name"]
        print(f"=== {shape_name}: batch={shape['batch']} seq={shape['seq_len']} d_model={shape['d_model']} d_state={shape['d_state']} ===")

        # Baseline: PyTorch sequential
        torch.manual_seed(0)
        x = torch.randn(shape["batch"], shape["seq_len"], shape["d_model"], device="cuda", dtype=torch.float32) * 0.1
        A = torch.randn(shape["batch"], shape["seq_len"], shape["d_model"], device="cuda", dtype=torch.float32) * 0.01 - 1.0
        B = torch.randn(shape["batch"], shape["seq_len"], shape["d_state"], device="cuda", dtype=torch.float32) * 0.1
        C = torch.randn(shape["batch"], shape["seq_len"], shape["d_state"], device="cuda", dtype=torch.float32) * 0.1

        baseline_ms = triton.testing.do_bench(
            lambda: torch_selective_scan(x, A, B, C),
            warmup=10, rep=50,
        )
        print(f"  PyTorch baseline: {baseline_ms:.2f} ms")

        best_ms = float("inf")
        best_cid = ""
        fixed_ms = None

        for config in unique_configs:
            cid = config_id(config)
            result = benchmark_one(shape, config)
            is_fixed = (config == MAMBA3_FIXED_CONFIG)
            tag = " [MAMBA-3 FIXED]" if is_fixed else ""

            if result["correct"] and result["ms"] is not None:
                speedup = baseline_ms / result["ms"] if result["ms"] > 0 else 0
                print(f"  {cid}{tag}: {result['ms']:.2f} ms ({speedup:.2f}x vs PyTorch, {result['gb_per_s']:.1f} GB/s)")
                if is_fixed:
                    fixed_ms = result["ms"]
                if result["ms"] < best_ms:
                    best_ms = result["ms"]
                    best_cid = cid
            else:
                err = result.get("error", result.get("max_err", "unknown"))
                print(f"  {cid}{tag}: FAILED ({err})")

        if fixed_ms and best_ms < fixed_ms:
            improvement = (fixed_ms - best_ms) / fixed_ms * 100
            print(f"  >>> Best: {best_cid} ({improvement:.1f}% faster than Mamba-3 fixed)")
        elif best_ms < float("inf"):
            print(f"  >>> Best: {best_cid} ({best_ms:.2f} ms)")
        print()

    # JSON summary
    summary = {
        "operator": "ssm_scan",
        "hardware": {"gpu": gpu_name, "cuda_version": torch.version.cuda or "unknown"},
        "shapes_tested": len(MAMBA3_SHAPES),
        "configs_tested": len(unique_configs),
    }
    print("JSON_SUMMARY:", json.dumps(summary))


if __name__ == "__main__":
    main()
