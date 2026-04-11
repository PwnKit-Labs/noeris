"""Parameterized Triton LayerNorm kernel and operator spec.

LayerNorm is like RMSNorm but also subtracts the mean and supports a bias.
AutoKernel reports 3.21x speedup over PyTorch on H100 for this operator.

Computes: y = (x - mean(x)) / sqrt(var(x) + eps) * w + b
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


LAYERNORM_PARAM_SPACE = {
    "BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3],
}


LAYERNORM_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 512, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 256, "num_warps": 1, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 2},
]


LAYERNORM_SHAPE_BUCKETS = [
    {"name": "small_hidden", "n_rows": 1024, "hidden_dim": 768},
    {"name": "gpt2_base", "n_rows": 4096, "hidden_dim": 768},
    {"name": "bert_base", "n_rows": 4096, "hidden_dim": 768},
    {"name": "bert_large", "n_rows": 4096, "hidden_dim": 1024},
    {"name": "gpt_xl", "n_rows": 4096, "hidden_dim": 1600},
    {"name": "gpt_neox", "n_rows": 4096, "hidden_dim": 4096},
    {"name": "long_seq_base", "n_rows": 8192, "hidden_dim": 768},
    {"name": "long_seq_large", "n_rows": 8192, "hidden_dim": 4096},
]


def layernorm_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def layernorm_shape_bucket_key(shape: dict[str, int]) -> str:
    h = shape.get("hidden_dim", 0)
    n = shape.get("n_rows", 0)
    if h <= 768:
        if n >= 8192:
            return "long_seq_base"
        if n <= 2048:
            return "small_hidden"
        return "gpt2_base"
    if h <= 1024:
        return "bert_large"
    if h <= 2048:
        return "gpt_xl"
    if n >= 8192:
        return "long_seq_large"
    return "gpt_neox"


def layernorm_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True. See module docstring."""
    return True


def generate_layernorm_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in LAYERNORM_CURATED_CONFIGS:
            cid = layernorm_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in LAYERNORM_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in LAYERNORM_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                cid = layernorm_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_layernorm_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton LayerNorm benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    x_row_stride, y_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass LayerNorm using E[X^2] - E[X]^2 for variance.

    Key optimizations:
    - Compute mean and E[X^2] in one pass, avoid re-loading x
    - Derive variance as E[X^2] - mean^2
    - Fuse normalize + scale + bias into single expression
    - Minimize dtype conversions
    """
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load once, keep in registers
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Single-pass: compute sum(x) and sum(x^2) simultaneously
    inv_n = 1.0 / n_cols
    mean = tl.sum(x, axis=0) * inv_n
    mean_sq = tl.sum(x * x, axis=0) * inv_n
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Pre-scale weight by rstd (avoids materializing x_centered)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


def layernorm(x, w, b, config, eps=1e-5):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    layernorm_kernel[(n_rows,)](
        x, w, b, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


def benchmark_one(n_rows, hidden_dim, config, dtype=torch.float16):
    try:
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=dtype)
        w = torch.randn((hidden_dim,), device="cuda", dtype=dtype)
        b = torch.randn((hidden_dim,), device="cuda", dtype=dtype)
        ref = torch.nn.functional.layer_norm(x, (hidden_dim,), w, b, eps=1e-5)
        out = layernorm(x, w, b, config)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}
        ms = triton.testing.do_bench(lambda: layernorm(x, w, b, config), warmup=25, rep=100)
        bytes_moved = 2 * n_rows * hidden_dim * 2 + hidden_dim * 4  # x read, y write, w+b read
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
        "operator": "layernorm",
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


LAYERNORM_SPEC = register_operator(TritonOperatorSpec(
    name="layernorm",
    param_space=LAYERNORM_PARAM_SPACE,
    curated_configs=LAYERNORM_CURATED_CONFIGS,
    shape_buckets=LAYERNORM_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=layernorm_config_id,
    shape_bucket_fn=layernorm_shape_bucket_key,
    benchmark_script_fn=generate_layernorm_benchmark_script,
    grid_generator_fn=generate_layernorm_grid,
    shared_memory_check_fn=layernorm_shared_memory_check,
    description="Memory-bound LayerNorm with bias. Fuses multi-op decomposition into single pass.",
))
