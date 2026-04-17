"""Parameterized Triton standalone GELU kernel.

Unlike GeGLU (gate * GELU(up)), this is a plain GELU activation:

    GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    GELU_exact(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

Uses a 2D grid (rows x col_tiles) with fixed BLOCK_SIZE so it handles
arbitrarily wide rows without requiring BLOCK_SIZE >= n_cols.

KernelBench upstream eval showed 8.40x speedup on H100 for the tiled
tanh-approximate variant.  This module promotes that inline kernel to a
first-class registered operator with shape buckets and curated configs.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


GELU_PARAM_SPACE = {
    "BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3, 4],
}


GELU_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 1024, "num_warps": 4,  "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 8,  "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 512,  "num_warps": 2,  "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8,  "num_stages": 2},
    {"BLOCK_SIZE": 256,  "num_warps": 1,  "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 8,  "num_stages": 2},
    # T4-optimized: fewer warps, smaller blocks
    {"BLOCK_SIZE": 1024, "num_warps": 2,  "num_stages": 1},
    {"BLOCK_SIZE": 512,  "num_warps": 4,  "num_stages": 1},
]


# Exact GELU is still the same elementwise algorithm, but the upstream
# KernelBench #26 shape is wide enough that it deserves its own launch
# defaults instead of inheriting the tanh-GELU starter config.
GELU_EXACT_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 4096, "num_warps": 8,  "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 8,  "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 1024, "num_warps": 8,  "num_stages": 1},
]


def default_gelu_exact_config(*, n_cols: int = 0) -> dict[str, int]:
    if n_cols >= 131072:
        return dict(GELU_EXACT_CURATED_CONFIGS[0])
    return dict(GELU_EXACT_CURATED_CONFIGS[1])


# Shape buckets: standalone GELU is applied to (batch*seq_len, hidden_dim)
# or (batch*seq_len, ffn_dim) tensors depending on architecture.
GELU_SHAPE_BUCKETS = [
    {"name": "test_small",            "n_rows": 512,  "n_cols": 1024},
    {"name": "bert_base",             "n_rows": 2048, "n_cols": 3072},
    {"name": "bert_large",            "n_rows": 2048, "n_cols": 4096},
    {"name": "gpt2_small",            "n_rows": 2048, "n_cols": 3072},
    {"name": "gpt2_medium",           "n_rows": 2048, "n_cols": 4096},
    {"name": "gpt2_large",            "n_rows": 2048, "n_cols": 5120},
    {"name": "gemma4_e2b",            "n_rows": 2048, "n_cols": 6144},
    {"name": "gemma4_e4b",            "n_rows": 2048, "n_cols": 10240},
    {"name": "llama3_8b",             "n_rows": 2048, "n_cols": 14336},
    {"name": "gemma4_31b",            "n_rows": 2048, "n_cols": 21504},
    {"name": "llama3_70b",            "n_rows": 2048, "n_cols": 28672},
]


def gelu_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def gelu_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a GELU shape into the nearest bucket."""
    _ALL_BUCKET_NAMES = {b["name"] for b in GELU_SHAPE_BUCKETS}
    name = shape.get("name", "")
    if name in _ALL_BUCKET_NAMES:
        return name

    nc = shape.get("n_cols", 0)
    if nc <= 1024:
        return "test_small"
    if nc <= 3072:
        return "bert_base"
    if nc <= 4096:
        return "bert_large"
    if nc <= 6144:
        return "gemma4_e2b"
    if nc <= 10240:
        return "gemma4_e4b"
    if nc <= 14336:
        return "llama3_8b"
    if nc <= 21504:
        return "gemma4_31b"
    return "llama3_70b"


def gelu_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only -- always returns True."""
    return True


def generate_gelu_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in GELU_CURATED_CONFIGS:
            cid = gelu_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in GELU_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in GELU_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                cid = gelu_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


_triton_gelu_tanh_available = False
_gelu_tanh_kernel_compiled = None
_triton_gelu_exact_available = False
_gelu_exact_kernel_compiled = None


def _ensure_triton_gelu_tanh():
    global _triton_gelu_tanh_available, _gelu_tanh_kernel_compiled
    if _gelu_tanh_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _gelu_tanh_kernel(
            x_ptr, out_ptr,
            n_cols,
            BLOCK_SIZE: tl.constexpr,
        ):
            """Standalone GELU (tanh approx) with 2D grid (rows x col_tiles)."""
            row_idx = tl.program_id(0)
            col_block = tl.program_id(1)
            x_ptr = x_ptr + row_idx * n_cols
            out_ptr = out_ptr + row_idx * n_cols
            offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            sqrt_2_over_pi = 0.7978845608028654
            coeff = 0.044715
            inner = sqrt_2_over_pi * (x + coeff * x * x * x)
            out = 0.5 * x * (1.0 + tl.extra.libdevice.tanh(inner))
            tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

        _gelu_tanh_kernel_compiled = _gelu_tanh_kernel
        _triton_gelu_tanh_available = True
    except ImportError:
        _triton_gelu_tanh_available = False


def _ensure_triton_gelu_exact():
    global _triton_gelu_exact_available, _gelu_exact_kernel_compiled
    if _gelu_exact_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _gelu_exact_kernel(
            x_ptr, out_ptr,
            n_cols,
            BLOCK_SIZE: tl.constexpr,
        ):
            """Standalone GELU (exact via erf) with 2D grid (rows x col_tiles)."""
            row_idx = tl.program_id(0)
            col_block = tl.program_id(1)
            x_ptr = x_ptr + row_idx * n_cols
            out_ptr = out_ptr + row_idx * n_cols
            offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
            out = x * 0.5 * (1.0 + tl.extra.libdevice.erf(x * inv_sqrt2))
            tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

        _gelu_exact_kernel_compiled = _gelu_exact_kernel
        _triton_gelu_exact_available = True
    except ImportError:
        _triton_gelu_exact_available = False


def gelu_tanh(x, config=None):
    """Standalone GELU (tanh approximation). No gate tensor.

    Args:
        x: (n_rows, n_cols) tensor, any dtype.
        config: Triton config dict. Defaults to first curated config.

    Returns:
        Same-shape, same-dtype output tensor.
    """
    import torch
    import triton

    _ensure_triton_gelu_tanh()
    if not _triton_gelu_tanh_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = GELU_CURATED_CONFIGS[0]

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = config["BLOCK_SIZE"]
    num_col_blocks = triton.cdiv(n_cols, BLOCK_SIZE)
    _gelu_tanh_kernel_compiled[(n_rows, num_col_blocks)](
        x, out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def gelu_exact(x, config=None):
    """Standalone GELU (exact via erf). No gate tensor.

    Args:
        x: (n_rows, n_cols) tensor, any dtype.
        config: Triton config dict. Defaults to first curated config.

    Returns:
        Same-shape, same-dtype output tensor.
    """
    import torch
    import triton

    _ensure_triton_gelu_exact()
    if not _triton_gelu_exact_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = default_gelu_exact_config(n_cols=x.shape[1])

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = config["BLOCK_SIZE"]
    num_col_blocks = triton.cdiv(n_cols, BLOCK_SIZE)
    _gelu_exact_kernel_compiled[(n_rows, num_col_blocks)](
        x, out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def generate_gelu_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained script benchmarking all GELU configs."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton standalone GELU benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_tanh_kernel(
    x_ptr, out_ptr, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Standalone GELU (tanh approx) -- 2D grid (rows x col_tiles)."""
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    x_ptr   = x_ptr   + row_idx * n_cols
    out_ptr = out_ptr + row_idx * n_cols
    offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    out = 0.5 * x * (1.0 + tl.extra.libdevice.tanh(inner))
    tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


def gelu(x, config):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = config["BLOCK_SIZE"]
    num_col_blocks = triton.cdiv(n_cols, BLOCK_SIZE)
    gelu_tanh_kernel[(n_rows, num_col_blocks)](
        x, out, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def torch_gelu(x):
    return torch.nn.functional.gelu(x.to(torch.float32), approximate="tanh").to(x.dtype)


def benchmark_one(n_rows, n_cols, config, dtype=torch.float16):
    try:
        x = torch.randn((n_rows, n_cols), device="cuda", dtype=dtype)
        ref = torch_gelu(x)
        out = gelu(x, config)
        max_err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
        if max_err > 1e-2:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        ms = triton.testing.do_bench(lambda: gelu(x, config), warmup=25, rep=100)
        bytes_moved = 2 * n_rows * n_cols * 2  # read x + write out, fp16
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
    shapes  = {shapes_json}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bs{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            n_rows = shape["n_rows"]
            n_cols = shape["n_cols"]
            result = benchmark_one(n_rows, n_cols, config)
            result["shape"]      = f"{{n_rows}}x{{n_cols}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config":    config,
            "results":   shape_results,
        }})

    output = {{
        "operator": "gelu",
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

GELU_SPEC = register_operator(TritonOperatorSpec(
    name="gelu",
    param_space=GELU_PARAM_SPACE,
    curated_configs=GELU_CURATED_CONFIGS,
    shape_buckets=GELU_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=gelu_config_id,
    shape_bucket_fn=gelu_shape_bucket_key,
    benchmark_script_fn=generate_gelu_benchmark_script,
    grid_generator_fn=generate_gelu_grid,
    shared_memory_check_fn=gelu_shared_memory_check,
    description=(
        "Standalone GELU activation (tanh approximation). 2D tiled grid "
        "handles arbitrarily wide rows. 8.40x H100 speedup on KernelBench."
    ),
))
