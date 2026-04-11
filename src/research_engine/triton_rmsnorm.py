"""Parameterized Triton RMSNorm kernel and operator spec.

RMSNorm is where Triton beats PyTorch hardest — AutoKernel reports 5.29x
over PyTorch eager on H100, reaching 83% of peak memory bandwidth. The
gap comes from fusing torch's multi-op decomposition into a single pass.

This module provides a parameterized RMSNorm kernel with search space:
- BLOCK_SIZE: rows processed per block
- num_warps: parallelism per block
- num_stages: pipeline depth

The kernel is memory-bound, so the metric is GB/s of memory bandwidth.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


RMSNORM_PARAM_SPACE = {
    "BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3, 4],
}


RMSNORM_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 512, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 256, "num_warps": 1, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 2},
]


# Shape buckets: (batch*seq, hidden_dim) pairs typical of LLM workloads
RMSNORM_SHAPE_BUCKETS = [
    {"name": "small_hidden", "n_rows": 1024, "hidden_dim": 768},
    {"name": "gpt2_base", "n_rows": 4096, "hidden_dim": 768},
    {"name": "llama_160m", "n_rows": 2048, "hidden_dim": 1024},
    {"name": "bert_base", "n_rows": 4096, "hidden_dim": 768},
    {"name": "llama_7b", "n_rows": 4096, "hidden_dim": 4096},
    {"name": "llama_13b", "n_rows": 4096, "hidden_dim": 5120},
    {"name": "llama_70b", "n_rows": 2048, "hidden_dim": 8192},
    {"name": "mixtral", "n_rows": 8192, "hidden_dim": 4096},
]


def rmsnorm_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def rmsnorm_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify an RMSNorm shape into a bucket."""
    h = shape.get("hidden_dim", 0)
    n = shape.get("n_rows", 0)
    if h <= 768:
        if n <= 2048:
            return "small_hidden"
        return "gpt2_base"
    if h <= 1024:
        return "llama_160m"
    if h <= 4096:
        return "llama_7b" if n <= 4096 else "mixtral"
    if h <= 5120:
        return "llama_13b"
    return "llama_70b"


def rmsnorm_shared_memory_check(config: dict[str, int]) -> bool:
    """Approximate shared memory limit check for A100 (192 KB per SM)."""
    bs = config.get("BLOCK_SIZE", 0)
    ns = config.get("num_stages", 1)
    # Each block loads BLOCK_SIZE fp16 elements, plus a few reductions
    shmem = bs * 2 * ns + 1024
    return shmem <= 192_000


def generate_rmsnorm_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in RMSNORM_CURATED_CONFIGS:
            cid = rmsnorm_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in RMSNORM_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in RMSNORM_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:  # most memory-bound kernels use 1-2 stages
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                if not rmsnorm_shared_memory_check(config):
                    continue
                cid = rmsnorm_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_rmsnorm_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained script benchmarking all RMSNorm configs."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton RMSNorm benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    x_row_stride,
    y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Parameterized RMSNorm kernel.

    Computes y = x * w / sqrt(mean(x^2) + eps) for each row.
    One program per row, BLOCK_SIZE elements processed in parallel.
    """
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    offs = tl.arange(0, BLOCK_SIZE)

    # Load x row (masked for the tail)
    mask = offs < n_cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Apply weight and store
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


def rmsnorm(x, w, config, eps=1e-6):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    rmsnorm_kernel[(n_rows,)](
        x, w, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


def torch_rmsnorm(x, w, eps=1e-6):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * w).to(torch.float16)


def benchmark_one(n_rows, hidden_dim, config, dtype=torch.float16):
    try:
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=dtype)
        w = torch.randn((hidden_dim,), device="cuda", dtype=dtype)
        ref = torch_rmsnorm(x, w)
        out = rmsnorm(x, w, config)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.05:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}
        ms = triton.testing.do_bench(lambda: rmsnorm(x, w, config), warmup=25, rep=100)
        # Memory bandwidth: read x (n_rows*hidden_dim*2 bytes) + read w (hidden_dim*2 bytes) + write y
        bytes_moved = 2 * n_rows * hidden_dim * 2 + hidden_dim * 2
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "gb_per_s": round(gb_per_s, 2),
            "tflops": round(gb_per_s, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None, "tflops": None}}


def main():
    configs = {configs_json}
    shapes = {shapes_json}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bs{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            n_rows = shape["n_rows"]
            hidden_dim = shape["hidden_dim"]
            result = benchmark_one(n_rows, hidden_dim, config)
            result["shape"] = f"{{n_rows}}x{{hidden_dim}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "rmsnorm",
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


# Register the operator
RMSNORM_SPEC = register_operator(TritonOperatorSpec(
    name="rmsnorm",
    param_space=RMSNORM_PARAM_SPACE,
    curated_configs=RMSNORM_CURATED_CONFIGS,
    shape_buckets=RMSNORM_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=rmsnorm_config_id,
    shape_bucket_fn=rmsnorm_shape_bucket_key,
    benchmark_script_fn=generate_rmsnorm_benchmark_script,
    grid_generator_fn=generate_rmsnorm_grid,
    shared_memory_check_fn=rmsnorm_shared_memory_check,
    description="Memory-bound RMSNorm. Triton fuses torch's multi-op decomposition.",
))
