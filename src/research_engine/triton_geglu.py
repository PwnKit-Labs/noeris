"""Parameterized Triton fused GeGLU kernel.

GeGLU (Gated GELU) is the activation used in Gemma 2/3/4 MLP blocks:

    out = gate * GELU_tanh(up)

where GELU_tanh uses the tanh approximation:

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

This fused kernel reads both gate and up tensors in a single pass and writes
the output, making it memory-bandwidth-bound. Fusing avoids materializing
the intermediate GELU result and reduces memory traffic by ~1/3 compared to
two separate ops (GELU + elementwise multiply).

Gemma 4 FFN dimensions (ffn_dim = intermediate size):
  E2B  : hidden=2304,  ffn_dim=5632   (SwiGLU-compatible geometry)
  E4B  : hidden=3584,  ffn_dim=14336
  26B A4B: hidden=2048, ffn_dim=16384  (MoE)
  31B Dense: hidden=4096, ffn_dim=24576

Reference: Gemma 2 tech report (arXiv:2408.00118).
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


GEGLU_PARAM_SPACE = {
    "BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3, 4],
}


# 8 curated configs spanning different block sizes — skewed toward large
# blocks because ffn_dim for Gemma models is 5632–24576.
GEGLU_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 1024, "num_warps": 4,  "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 8,  "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 512,  "num_warps": 2,  "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8,  "num_stages": 2},
    {"BLOCK_SIZE": 2048, "num_warps": 4,  "num_stages": 2},
    {"BLOCK_SIZE": 256,  "num_warps": 1,  "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 8,  "num_stages": 2},
]


# Shape buckets: (n_rows, ffn_dim) pairs for Gemma 4 and a small test shape.
# n_rows = batch_size * seq_len (token count hitting the MLP)
# ffn_dim = intermediate size (each of gate and up has this width)
GEGLU_SHAPE_BUCKETS = [
    {"name": "test_small",    "n_rows": 512,  "ffn_dim": 1024},
    {"name": "gemma4_e2b",   "n_rows": 2048, "ffn_dim": 5632},
    {"name": "gemma4_e4b",   "n_rows": 2048, "ffn_dim": 14336},
    {"name": "gemma4_26b",   "n_rows": 2048, "ffn_dim": 16384},
    {"name": "gemma4_31b",   "n_rows": 2048, "ffn_dim": 24576},
]


def geglu_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def geglu_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a GeGLU shape into the nearest Gemma 4 bucket."""
    fd = shape.get("ffn_dim", 0)
    if fd <= 1024:
        return "test_small"
    if fd <= 5632:
        return "gemma4_e2b"
    if fd <= 14336:
        return "gemma4_e4b"
    if fd <= 16384:
        return "gemma4_26b"
    return "gemma4_31b"


def geglu_shared_memory_check(config: dict[str, int]) -> bool:
    """Approximate shared memory limit check for A100 (192 KB per SM).

    The kernel loads gate + up (2 * BLOCK_SIZE fp16) per stage.
    """
    bs = config.get("BLOCK_SIZE", 0)
    ns = config.get("num_stages", 1)
    # 2 tensors * BLOCK_SIZE * 2 bytes (fp16) * num_stages + overhead
    shmem = 2 * bs * 2 * ns + 1024
    return shmem <= 192_000


def generate_geglu_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in GEGLU_CURATED_CONFIGS:
            cid = geglu_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in GEGLU_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in GEGLU_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:  # memory-bound kernels rarely need >2 stages
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                if not geglu_shared_memory_check(config):
                    continue
                cid = geglu_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_geglu_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained script benchmarking all GeGLU configs."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton fused GeGLU benchmark."""

import json
import math
import platform

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GeGLU: out[row, col] = gate[row, col] * GELU_tanh(up[row, col]).

    One program per row; BLOCK_SIZE columns processed in parallel.
    GELU uses the tanh approximation (what Gemma uses):
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    row_idx = tl.program_id(0)
    gate_ptr = gate_ptr + row_idx * n_cols
    up_ptr   = up_ptr   + row_idx * n_cols
    out_ptr  = out_ptr  + row_idx * n_cols

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(up_ptr   + offs, mask=mask, other=0.0).to(tl.float32)

    # tanh-approximated GELU
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (up + coeff * up * up * up)
    gelu_up = 0.5 * up * (1.0 + tl.extra.libdevice.tanh(inner))

    out = gate * gelu_up
    tl.store(out_ptr + offs, out.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def geglu(gate, up, config):
    """Run fused GeGLU with the given config."""
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


def torch_geglu(gate, up):
    """PyTorch reference: gate * GELU_tanh(up)."""
    return torch.nn.functional.gelu(up.to(torch.float32), approximate="tanh").to(torch.float16) * gate


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark_one(n_rows, ffn_dim, config, dtype=torch.float16):
    try:
        gate = torch.randn((n_rows, ffn_dim), device="cuda", dtype=dtype)
        up   = torch.randn((n_rows, ffn_dim), device="cuda", dtype=dtype)

        ref = torch_geglu(gate, up)
        out = geglu(gate, up, config)
        max_err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
        if max_err > 1e-2:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        ms = triton.testing.do_bench(lambda: geglu(gate, up, config), warmup=25, rep=100)
        # Memory: read gate + up (2 tensors) + write out (1 tensor)
        # Each element is 2 bytes (fp16), so bytes = 3 * n_rows * ffn_dim * 2
        bytes_moved = 3 * n_rows * ffn_dim * 2
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "gb_per_s": round(gb_per_s, 2),
            "tflops": round(gb_per_s, 2),  # alias: tflops field = gb_per_s for memory-bound
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None, "tflops": None}}


def main():
    configs = {configs_json}
    shapes  = {shapes_json}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bs{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            n_rows  = shape["n_rows"]
            ffn_dim = shape["ffn_dim"]
            result  = benchmark_one(n_rows, ffn_dim, config)
            result["shape"]      = f"{{n_rows}}x{{ffn_dim}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config":    config,
            "results":   shape_results,
        }})

    output = {{
        "operator": "geglu",
        "hardware": {{
            "gpu":          gpu_name,
            "cuda_version": torch.version.cuda or "unknown",
            "python":       platform.python_version(),
        }},
        "configs_tested": len(configs),
        "config_results": all_results,
    }}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Register the operator
# ---------------------------------------------------------------------------

GEGLU_SPEC = register_operator(TritonOperatorSpec(
    name="geglu",
    param_space=GEGLU_PARAM_SPACE,
    curated_configs=GEGLU_CURATED_CONFIGS,
    shape_buckets=GEGLU_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=geglu_config_id,
    shape_bucket_fn=geglu_shape_bucket_key,
    benchmark_script_fn=generate_geglu_benchmark_script,
    grid_generator_fn=generate_geglu_grid,
    shared_memory_check_fn=geglu_shared_memory_check,
    description=(
        "Fused GeGLU (gate * GELU_tanh(up)) used in Gemma 2/3/4 MLP blocks. "
        "Memory-bound; fusing avoids materializing intermediate GELU output."
    ),
))
