"""Parameterized Triton Softmax kernel and operator spec.

Softmax is another memory-bound kernel where Triton beats PyTorch
substantially. AutoKernel reports 2.82x over PyTorch eager and 3.44x
over torch.compile on H100.

The kernel computes y = exp(x - max(x)) / sum(exp(x - max(x))) per row
in a single pass using online-softmax accumulation.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


SOFTMAX_PARAM_SPACE = {
    "BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3],
}


SOFTMAX_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 8192, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 512, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8, "num_stages": 2},
    # T4-optimized: fewer warps for 40-SM GPU
    {"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 1024, "num_warps": 2, "num_stages": 1},
    # KernelBench L1 #23 (393k cols) and Gemma 4 262k vocab: max block + warps
    {"BLOCK_SIZE": 16384, "num_warps": 16, "num_stages": 1},
]


SOFTMAX_SHAPE_BUCKETS = [
    {"name": "small", "n_rows": 1024, "n_cols": 512},
    {"name": "medium", "n_rows": 2048, "n_cols": 1024},
    {"name": "large", "n_rows": 4096, "n_cols": 4096},
    {"name": "attn_short", "n_rows": 8192, "n_cols": 1024},
    {"name": "attn_long", "n_rows": 4096, "n_cols": 8192},
    {"name": "vocab_small", "n_rows": 1024, "n_cols": 32000},
    {"name": "vocab_llama", "n_rows": 2048, "n_cols": 32000},
    # Gemma 4 final-logits path — 262144 vocab with optional softcap=30.
    # Gemma 4 31B retains a final-logits softcap (per Kaitchup arch breakdown).
    # Smaller Gemma 4 variants do not use a softcap but still use the 262k vocab.
    {"name": "gemma4_262k_vocab", "n_rows": 2048, "n_cols": 262144},
    {"name": "gemma4_31b_softcap", "n_rows": 2048, "n_cols": 262144, "softcap": 30.0},
    # Upstream KernelBench L1 #23 (Softmax) shape — (4096, 393216) fp32
    # 7× wider than vocab_llama; separate bucket so bandit does not mix priors.
    {"name": "kb_l1_23_huge", "n_rows": 4096, "n_cols": 393216},
]


def softmax_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def softmax_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a softmax shape.

    Post-#40/#43: the softcap variant gets its own bucket so bandit priors
    do not mix between `softcap=0` (standard softmax) and `softcap>0`
    (Gemma 4 31B final logits).  The KernelBench L1 #23 wide-softmax shape
    (4096 × 393216 fp32) also gets its own bucket since no existing
    vocab bucket is within 7× of that column count.
    """
    cols = shape.get("n_cols", 0)
    rows = shape.get("n_rows", 0)
    softcap = float(shape.get("softcap", 0.0) or 0.0)
    # Upstream KernelBench L1 #23: (4096, 393216) — separate huge bucket
    if cols >= 300_000:
        return "kb_l1_23_huge"
    # Gemma 4 262k vocab: two buckets depending on whether the final-logits
    # softcap is active (31B Dense uses softcap=30; other variants do not).
    if cols >= 200_000:
        return "gemma4_31b_softcap" if softcap > 0.0 else "gemma4_262k_vocab"
    if cols >= 16384:
        return "vocab_llama" if rows >= 2048 else "vocab_small"
    if cols >= 4096:
        return "attn_long"
    if rows >= 4096:
        return "attn_short" if cols < 4096 else "large"
    if rows >= 2048:
        return "medium"
    return "small"


def softmax_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True.

    Feasibility is now learned from runtime failures (reward=0 in the
    bandit) rather than enforced via a hand-curated formula. Retained
    on the operator spec for backward compatibility.
    """
    return True


def generate_softmax_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in SOFTMAX_CURATED_CONFIGS:
            cid = softmax_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in SOFTMAX_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in SOFTMAX_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                cid = softmax_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


# ---------------------------------------------------------------------------
# Module-level launcher (lazy-init pattern, matches triton_rmsnorm.py)
# ---------------------------------------------------------------------------

_triton_softmax_available = False
_softmax_kernel_compiled = None


def _ensure_triton_softmax():
    global _triton_softmax_available, _softmax_kernel_compiled
    if _softmax_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _softmax_kernel(
            x_ptr, y_ptr,
            x_row_stride,
            y_row_stride,
            n_cols,
            softcap,
            BLOCK_SIZE: tl.constexpr,
            USE_SOFTCAP: tl.constexpr,
        ):
            row_idx = tl.program_id(0)
            x_ptr += row_idx * x_row_stride
            y_ptr += row_idx * y_row_stride
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols

            x = tl.load(x_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
            if USE_SOFTCAP:
                inv_softcap = 1.0 / softcap
                x = softcap * (2.0 / (1.0 + tl.exp(-2.0 * x * inv_softcap)) - 1.0)
            row_max = tl.max(x, axis=0)
            x_shifted = x - row_max
            exp_x = tl.exp(x_shifted)
            denom = tl.sum(exp_x, axis=0)
            y = exp_x / denom

            tl.store(y_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)

        _softmax_kernel_compiled = _softmax_kernel
        _triton_softmax_available = True
    except ImportError:
        _triton_softmax_available = False


def softmax(x, config=None, softcap=0.0):
    """Module-level Softmax launcher. Requires CUDA GPU.

    Args:
        x: (n_rows, n_cols) tensor.
        config: Triton config dict with BLOCK_SIZE, num_warps, num_stages.
            Defaults to the first curated config.
        softcap: If > 0, applies softcap pre-step (Gemma 4 31B style).

    Returns:
        Same-shape output tensor with the same dtype as `x`.
    """
    import torch
    import triton

    _ensure_triton_softmax()
    if not _triton_softmax_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = SOFTMAX_CURATED_CONFIGS[0]

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    use_softcap = float(softcap) > 0.0
    _softmax_kernel_compiled[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        float(softcap) if use_softcap else 1.0,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_SOFTCAP=use_softcap,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


def generate_softmax_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton Softmax benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    x_row_stride,
    y_row_stride,
    n_cols,
    softcap,
    BLOCK_SIZE: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
):
    """Numerically-stable softmax with optional softcap pre-step.

    Standard:         y = exp(x - max(x)) / sum(...)
    With softcap:     x = softcap * tanh(x / softcap);  then standard softmax.
    Gemma 4 31B final logits use softcap=30.0.
    """
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    if USE_SOFTCAP:
        # softcap * tanh(x / softcap) — bounds magnitudes to [-softcap, softcap]
        inv_softcap = 1.0 / softcap
        x = softcap * (2.0 / (1.0 + tl.exp(-2.0 * x * inv_softcap)) - 1.0)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    denom = tl.sum(exp_x, axis=0)
    y = exp_x / denom

    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


def softmax(x, config, softcap=0.0):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    use_softcap = float(softcap) > 0.0
    softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        float(softcap) if use_softcap else 1.0,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_SOFTCAP=use_softcap,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


def benchmark_one(n_rows, n_cols, config, dtype=torch.float16, softcap=0.0):
    try:
        x = torch.randn((n_rows, n_cols), device="cuda", dtype=dtype)
        x_f32 = x.to(torch.float32)
        if float(softcap) > 0.0:
            sc = float(softcap)
            x_f32 = sc * torch.tanh(x_f32 / sc)
        ref = torch.softmax(x_f32, dim=-1).to(torch.float16)
        out = softmax(x, config, softcap=softcap)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.05:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}
        ms = triton.testing.do_bench(lambda: softmax(x, config, softcap=softcap), warmup=25, rep=100)
        bytes_moved = 2 * n_rows * n_cols * 2  # read x + write y
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
            n_cols = shape["n_cols"]
            softcap = float(shape.get("softcap", 0.0) or 0.0)
            result = benchmark_one(n_rows, n_cols, config, softcap=softcap)
            result["shape"] = f"{{n_rows}}x{{n_cols}}"
            result["shape_name"] = shape.get("name", "")
            result["softcap"] = softcap
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "softmax",
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


SOFTMAX_SPEC = register_operator(TritonOperatorSpec(
    name="softmax",
    param_space=SOFTMAX_PARAM_SPACE,
    curated_configs=SOFTMAX_CURATED_CONFIGS,
    shape_buckets=SOFTMAX_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=softmax_config_id,
    shape_bucket_fn=softmax_shape_bucket_key,
    benchmark_script_fn=generate_softmax_benchmark_script,
    grid_generator_fn=generate_softmax_grid,
    shared_memory_check_fn=softmax_shared_memory_check,
    description="Memory-bound softmax with online computation. Beats torch eager by 2-3x.",
))
