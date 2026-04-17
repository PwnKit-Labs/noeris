"""Parameterized Triton fused GeGLU kernel.

GeGLU (Gated GELU) is the activation used in Gemma 2/3/4 MLP blocks:

    out = gate * GELU_tanh(up)

where GELU_tanh uses the tanh approximation:

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

This fused kernel reads both gate and up tensors in a single pass and writes
the output, making it memory-bandwidth-bound. Fusing avoids materializing
the intermediate GELU result and reduces memory traffic by ~1/3 compared to
two separate ops (GELU + elementwise multiply).

Gemma 4 FFN dimensions (ffn_dim = intermediate size). Verified against
the authoritative HF config.json files on 2026-04-11:
  E2B            : hidden=1536,  ffn_dim=6144    (google/gemma-4-E2B-it)
  E4B            : hidden=2560,  ffn_dim=10240   (google/gemma-4-E4B-it)
  26B-A4B expert : hidden=2816,  ffn_dim=2112    (per expert, 128 experts,
                                                  top-8 active + 1 shared;
                                                  google/gemma-4-26B-A4B-it)
  31B Dense      : hidden=5376,  ffn_dim=21504   (google/gemma-4-31B)

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
    # T4-optimized: 40 SMs prefer fewer warps; good for small ffn_dim (2112)
    {"BLOCK_SIZE": 1024, "num_warps": 2,  "num_stages": 1},
    {"BLOCK_SIZE": 512,  "num_warps": 4,  "num_stages": 1},
    # Gemma 4 31B (ffn_dim=21504): large block + deep pipeline
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 2},
]


# Shape buckets: (n_rows, ffn_dim) pairs for Gemma 4 and a small test shape.
# n_rows = batch_size * seq_len (token count hitting the MLP)
# ffn_dim = intermediate size (each of gate and up has this width)
#
# Values are pulled directly from each model's HF config.json (2026-04-11).
# Note: "gemma4_26b_a4b_expert" is per-expert; the MoE block dispatches
# top-8 of 128 experts per token + 1 shared expert, so the aggregate work
# per token is ~9 * this bucket.
GEGLU_SHAPE_BUCKETS = [
    {"name": "test_small",              "n_rows": 512,  "ffn_dim": 1024},
    {"name": "gemma4_e2b",              "n_rows": 2048, "ffn_dim": 6144},
    {"name": "gemma4_e4b",              "n_rows": 2048, "ffn_dim": 10240},
    {"name": "gemma4_26b_a4b_expert",   "n_rows": 2048, "ffn_dim": 2112},
    {"name": "gemma4_31b",              "n_rows": 2048, "ffn_dim": 21504},
    # Non-Gemma architectures — SwiGLU uses the same fused kernel structure
    {"name": "llama3_8b",               "n_rows": 2048, "ffn_dim": 14336},
    {"name": "llama3_70b",              "n_rows": 2048, "ffn_dim": 28672},
    {"name": "mistral_7b",              "n_rows": 2048, "ffn_dim": 14336},
    {"name": "phi3_mini",               "n_rows": 2048, "ffn_dim": 8192},
    # ---------- April 2026 model expansion ----------
    # Llama 4 Scout/Maverick (ffn_dim=16384 per MoE expert, SwiGLU)
    {"name": "llama4_scout",            "n_rows": 2048, "ffn_dim": 16384},
    # Qwen 3 family (SwiGLU)
    {"name": "qwen3_8b",               "n_rows": 2048, "ffn_dim": 12288},
    {"name": "qwen3_32b",              "n_rows": 2048, "ffn_dim": 25600},
    # Mixtral 8x22B (ffn_dim=16384, SwiGLU)
    {"name": "mixtral_8x22b",          "n_rows": 2048, "ffn_dim": 16384},
    # Phi-4 family (SwiGLU)
    {"name": "phi4_mini",              "n_rows": 2048, "ffn_dim": 8192},
    {"name": "phi4_14b",               "n_rows": 2048, "ffn_dim": 17920},
    # Falcon 3 (SwiGLU)
    {"name": "falcon3_7b",             "n_rows": 2048, "ffn_dim": 8192},
    {"name": "falcon3_10b",            "n_rows": 2048, "ffn_dim": 14336},
    # DBRX (SwiGLU, ffn_dim=10752 per expert)
    {"name": "dbrx",                   "n_rows": 2048, "ffn_dim": 10752},
    # OLMo 2 (SwiGLU)
    {"name": "olmo2_7b",               "n_rows": 2048, "ffn_dim": 11008},
    {"name": "olmo2_32b",              "n_rows": 2048, "ffn_dim": 13824},
    # InternLM 3 (SwiGLU, ffn=14336)
    {"name": "internlm3_8b",           "n_rows": 2048, "ffn_dim": 14336},
]


def geglu_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def geglu_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a GeGLU shape into the nearest bucket.

    Exact name match is checked first — this handles all non-Gemma
    architectures and any shape dict that carries a ``name`` field
    matching a known bucket.  For unnamed shapes the function falls
    through to ascending ffn_dim bands.
    """
    _ALL_BUCKET_NAMES = {b["name"] for b in GEGLU_SHAPE_BUCKETS}
    name = shape.get("name", "")
    # Exact name match for any known bucket
    if name in _ALL_BUCKET_NAMES:
        return name

    fd = shape.get("ffn_dim", 0)
    if fd <= 1024:
        return "test_small"
    if fd <= 2112:
        return "gemma4_26b_a4b_expert"
    if fd <= 6144:
        return "gemma4_e2b"
    if fd <= 8192:
        return "phi3_mini"
    if fd <= 10240:
        return "gemma4_e4b"
    if fd <= 14336:
        return "llama3_8b"
    if fd <= 21504:
        return "gemma4_31b"
    # Anything wider than 21504 falls to the widest dense bucket.
    return "gemma4_31b"


def geglu_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True.

    Feasibility is learned from runtime failures (reward=0). Retained for
    backward compatibility.
    """
    return True


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
                cid = geglu_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


_triton_geglu_available = False
_geglu_kernel_compiled = None


def _ensure_triton_geglu():
    global _triton_geglu_available, _geglu_kernel_compiled
    if _geglu_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _geglu_kernel(
            gate_ptr, up_ptr, out_ptr,
            n_cols,
            stride_gate_row,
            stride_up_row,
            stride_out_row,
            BLOCK_SIZE: tl.constexpr,
        ):
            row_idx = tl.program_id(0)
            gate_ptr = gate_ptr + row_idx * stride_gate_row
            up_ptr = up_ptr + row_idx * stride_up_row
            out_ptr = out_ptr + row_idx * stride_out_row
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

        _geglu_kernel_compiled = _geglu_kernel
        _triton_geglu_available = True
    except ImportError:
        _triton_geglu_available = False


def geglu(gate, up, config=None):
    """Module-level fused GeGLU launcher. Requires CUDA GPU.

    Computes: out = gate * GELU_tanh(up)

    Args:
        gate: (n_rows, ffn_dim) fp16 tensor.
        up: (n_rows, ffn_dim) fp16 tensor.
        config: Triton config dict with BLOCK_SIZE, num_warps, num_stages.
            Defaults to the first curated config.

    Returns:
        (n_rows, ffn_dim) fp16 output tensor.
    """
    import torch
    import triton

    _ensure_triton_geglu()
    if not _triton_geglu_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = GEGLU_CURATED_CONFIGS[0]

    n_rows, n_cols = gate.shape
    assert up.shape == gate.shape, f"Shape mismatch: gate={gate.shape}, up={up.shape}"
    out = torch.empty_like(gate)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    _geglu_kernel_compiled[(n_rows,)](
        gate, up, out,
        n_cols,
        gate.stride(0),
        up.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


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
    stride_gate_row,
    stride_up_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GeGLU: out[row, col] = gate[row, col] * GELU_tanh(up[row, col]).

    One program per row; BLOCK_SIZE columns processed in parallel.
    GELU uses the tanh approximation (what Gemma uses):
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    row_idx = tl.program_id(0)
    gate_ptr = gate_ptr + row_idx * stride_gate_row
    up_ptr   = up_ptr   + row_idx * stride_up_row
    out_ptr  = out_ptr  + row_idx * stride_out_row

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
    assert up.shape == gate.shape, f"Shape mismatch: gate={{gate.shape}}, up={{up.shape}}"
    out = torch.empty_like(gate)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    geglu_kernel[(n_rows,)](
        gate, up, out,
        n_cols,
        gate.stride(0),
        up.stride(0),
        out.stride(0),
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
