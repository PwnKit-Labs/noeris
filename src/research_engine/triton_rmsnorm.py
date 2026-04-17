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
    # T4-optimized: fewer warps for 40-SM GPU; Gemma 4 E2B hidden=1536
    {"BLOCK_SIZE": 1024, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 512, "num_warps": 4, "num_stages": 1},
    # Large hidden (llama_70b hidden=8192): max throughput
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 2},
]


# Shape buckets: (batch*seq, hidden_dim) pairs typical of LLM workloads.
# affine_mode: 0 = standard y = x * rstd * w
#              1 = gemma y = x * rstd * (1 + w)  (Gemma 3/4 style)
RMSNORM_SHAPE_BUCKETS = [
    {"name": "small_hidden", "n_rows": 1024, "hidden_dim": 768, "affine_mode": 0},
    {"name": "gpt2_base", "n_rows": 4096, "hidden_dim": 768, "affine_mode": 0},
    {"name": "llama_160m", "n_rows": 2048, "hidden_dim": 1024, "affine_mode": 0},
    {"name": "bert_base", "n_rows": 4096, "hidden_dim": 768, "affine_mode": 0},
    {"name": "llama_7b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    {"name": "llama_13b", "n_rows": 4096, "hidden_dim": 5120, "affine_mode": 0},
    {"name": "llama_70b", "n_rows": 2048, "hidden_dim": 8192, "affine_mode": 0},
    {"name": "mixtral", "n_rows": 8192, "hidden_dim": 4096, "affine_mode": 0},
    # Gemma 4 family (released April 2026) — hidden_dim values from HF config.json.
    # Gemma 3/4 use affine_mode=1: y = x * rstd * (1 + w), because trained weights
    # are small perturbations around 0. See vLLM's GemmaRMSNorm CustomOp, issue #46.
    # hidden_dim values per HF config.json (issue #45):
    #   E2B -> 1536, E4B -> 2560, 26B-A4B -> 2816, 31B -> 5376
    {"name": "gemma4_e2b", "n_rows": 2048, "hidden_dim": 1536, "affine_mode": 1},
    {"name": "gemma4_e4b", "n_rows": 2048, "hidden_dim": 2560, "affine_mode": 1},
    {"name": "gemma4_26b", "n_rows": 4096, "hidden_dim": 2816, "affine_mode": 1},
    {"name": "gemma4_31b", "n_rows": 4096, "hidden_dim": 5376, "affine_mode": 1},
    # Non-Gemma architectures — standard affine (affine_mode=0)
    {"name": "llama3_8b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    {"name": "llama3_70b", "n_rows": 4096, "hidden_dim": 8192, "affine_mode": 0},
    {"name": "mistral_7b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    {"name": "phi3_mini", "n_rows": 4096, "hidden_dim": 3072, "affine_mode": 0},
    # ---------- April 2026 model expansion ----------
    # Llama 4 Scout/Maverick (hidden=5120, RoPE, standard affine)
    {"name": "llama4_scout", "n_rows": 4096, "hidden_dim": 5120, "affine_mode": 0},
    # Qwen 3 family (QK-norm uses (1+w) Gemma-style affine on Q/K;
    # layer-norm on hidden uses standard affine)
    {"name": "qwen3_8b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    {"name": "qwen3_32b", "n_rows": 4096, "hidden_dim": 5120, "affine_mode": 0},
    # Mixtral 8x22B (hidden=6144)
    {"name": "mixtral_8x22b", "n_rows": 4096, "hidden_dim": 6144, "affine_mode": 0},
    # Phi-4 family
    {"name": "phi4_mini", "n_rows": 4096, "hidden_dim": 3072, "affine_mode": 0},
    {"name": "phi4_14b", "n_rows": 4096, "hidden_dim": 5120, "affine_mode": 0},
    # Falcon 3 (hidden=3072 for 7B, hidden~4096 for 10B)
    {"name": "falcon3_7b", "n_rows": 4096, "hidden_dim": 3072, "affine_mode": 0},
    {"name": "falcon3_10b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    # DBRX (hidden=6144)
    {"name": "dbrx", "n_rows": 4096, "hidden_dim": 6144, "affine_mode": 0},
    # OLMo 2 (uses RMSNorm, standard affine)
    {"name": "olmo2_7b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    {"name": "olmo2_32b", "n_rows": 4096, "hidden_dim": 5120, "affine_mode": 0},
    # InternLM 3 (hidden=4096)
    {"name": "internlm3_8b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
]


def rmsnorm_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def rmsnorm_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify an RMSNorm shape into a bucket.

    Exact name match is checked first for non-Gemma architectures, then
    Gemma 4 family shapes are detected by their exact hidden_dim values:
    - 1536 → gemma4_e2b  (HF config.json, issue #45)
    - 2048 → gemma4_e2b  (legacy alias; preserved for backward compat)
    - 2560 → gemma4_e4b  (unique to Gemma; LLaMA does not use this width)
    - 2816 → gemma4_26b  (HF config.json for gemma-4-26B-A4B, issue #45)
    - 3072 → phi3_mini   (Phi-3 mini hidden size)
    - 4096 with n_rows >= 4096 → gemma4_26b  (disambiguated from llama_7b by row count)
    - 5376 → gemma4_31b  (unique to Gemma; sits between llama_13b and llama_70b)
    """
    name = shape.get("name", "")
    # Exact name match for non-Gemma buckets
    if name in ("llama3_8b", "llama3_70b", "mistral_7b", "phi3_mini"):
        return name

    h = shape.get("hidden_dim", 0)
    n = shape.get("n_rows", 0)
    if h <= 768:
        if n <= 2048:
            return "small_hidden"
        return "gpt2_base"
    if h <= 1024:
        return "llama_160m"
    if h == 1536:
        return "gemma4_e2b"
    if h == 2048:
        return "gemma4_e2b"
    if h == 2560:
        return "gemma4_e4b"
    if h == 2816:
        return "gemma4_26b"
    if h == 3072:
        return "phi3_mini"
    if h <= 4096:
        if n > 4096:
            return "mixtral"
        if n >= 4096:
            return "gemma4_26b"
        return "llama_7b"
    if h <= 5120:
        return "llama_13b"
    if h == 5376:
        return "gemma4_31b"
    return "llama_70b"


def rmsnorm_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True.

    Feasibility is learned from runtime failures (recorded as reward=0 in
    the bandit). Retained on the spec for backward compatibility.
    """
    return True


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
                cid = rmsnorm_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


_triton_available = False
_rmsnorm_kernel_compiled = None
_rmsnorm_strided_kernel_compiled = None
_rmsnorm_nchw_dim1_kernel_compiled = None


def _ensure_triton_rmsnorm():
    global _triton_available, _rmsnorm_kernel_compiled, _rmsnorm_strided_kernel_compiled, _rmsnorm_nchw_dim1_kernel_compiled
    if (
        _rmsnorm_kernel_compiled is not None
        and _rmsnorm_strided_kernel_compiled is not None
        and _rmsnorm_nchw_dim1_kernel_compiled is not None
    ):
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _rmsnorm_kernel(
            x_ptr, w_ptr, y_ptr,
            x_row_stride,
            y_row_stride,
            n_cols,
            eps,
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

        @triton.jit
        def _rmsnorm_strided_kernel(
            x_ptr, w_ptr, y_ptr,
            outer_stride,
            norm_stride,
            n_norm,
            eps,
            BLOCK_SIZE: tl.constexpr,
        ):
            """RMSNorm over a strided axis without materializing a permute."""
            pid = tl.program_id(0)
            base = x_ptr + pid * outer_stride
            y_base = y_ptr + pid * outer_stride
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < n_norm
            x = tl.load(base + offs * norm_stride, mask=mask, other=0.0).to(tl.float32)
            mean_sq = tl.sum(x * x, axis=0) / n_norm
            rstd = 1.0 / tl.sqrt(mean_sq + eps)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(y_base + offs * norm_stride, y.to(tl.float16), mask=mask)

        @triton.jit
        def _rmsnorm_nchw_dim1_kernel(
            x_ptr, w_ptr, y_ptr,
            batch_stride,
            channel_stride,
            hw_size,
            eps,
            BLOCK_HW: tl.constexpr,
            CHANNELS: tl.constexpr,
        ):
            """RMSNorm for contiguous NCHW tensors normalized over C."""
            hw_pid = tl.program_id(0)
            batch_pid = tl.program_id(1)

            hw_offsets = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
            mask = hw_offsets < hw_size

            x_batch = x_ptr + batch_pid * batch_stride
            y_batch = y_ptr + batch_pid * batch_stride
            sum_sq = tl.zeros((BLOCK_HW,), dtype=tl.float32)

            for c in range(CHANNELS):
                x_vals = tl.load(
                    x_batch + c * channel_stride + hw_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                sum_sq += x_vals * x_vals

            rstd = tl.rsqrt(sum_sq / CHANNELS + eps)

            for c in range(CHANNELS):
                x_vals = tl.load(
                    x_batch + c * channel_stride + hw_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                w_val = tl.load(w_ptr + c).to(tl.float32)
                y_vals = x_vals * rstd * w_val
                tl.store(
                    y_batch + c * channel_stride + hw_offsets,
                    y_vals.to(tl.float16),
                    mask=mask,
                )

        _rmsnorm_kernel_compiled = _rmsnorm_kernel
        _rmsnorm_strided_kernel_compiled = _rmsnorm_strided_kernel
        _rmsnorm_nchw_dim1_kernel_compiled = _rmsnorm_nchw_dim1_kernel
        _triton_available = True
    except ImportError:
        _triton_available = False


def rmsnorm(x, w, config=None, eps=1e-6, affine_mode=0):
    """Module-level RMSNorm launcher. Requires CUDA GPU.

    Args:
        x: (n_rows, hidden_dim) fp16 tensor.
        w: (hidden_dim,) fp16 weight tensor.
        config: Triton config dict with BLOCK_SIZE, num_warps, num_stages.
            Defaults to the first curated config.
        eps: Epsilon for numerical stability.
        affine_mode: 0 = standard (y = x * rstd * w),
                     1 = Gemma (y = x * rstd * (1 + w)).

    Returns:
        (n_rows, hidden_dim) fp16 output tensor.
    """
    import torch
    import triton

    _ensure_triton_rmsnorm()
    if not _triton_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = RMSNORM_CURATED_CONFIGS[0]

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    _rmsnorm_kernel_compiled[(n_rows,)](
        x, w, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        AFFINE_MODE=affine_mode,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return y


def rmsnorm_strided(x, w, *, outer_stride: int, norm_stride: int, n_norm: int, eps=1e-6):
    """RMSNorm over a non-contiguous axis.

    Typical upstream use is NCHW tensors normalized along C:
    `outer_stride=1`, `norm_stride=H*W`, `n_norm=C`, grid=`B*H*W`.
    """
    import torch
    import triton

    _ensure_triton_rmsnorm()
    if not _triton_available:
        raise RuntimeError("Triton not available")

    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_norm)
    n_outer = x.numel() // n_norm
    _rmsnorm_strided_kernel_compiled[(n_outer,)](
        x, w, y,
        outer_stride, norm_stride, n_norm, eps,
        BLOCK_SIZE=block_size,
        num_warps=min(4, max(1, block_size // 32)),
        num_stages=1,
    )
    return y


def rmsnorm_nchw_dim1(x, w, *, eps=1e-6, block_hw: int = 256):
    """RMSNorm on a contiguous NCHW tensor normalized over the C dimension."""
    import torch
    import triton

    _ensure_triton_rmsnorm()
    if not _triton_available:
        raise RuntimeError("Triton not available")

    if x.ndim != 4:
        raise ValueError(f"expected 4D NCHW tensor, got shape={tuple(x.shape)!r}")

    batch, channels, height, width = x.shape
    if channels != w.numel():
        raise ValueError(f"weight length {w.numel()} does not match channels {channels}")

    y = torch.empty_like(x)
    hw_size = height * width
    grid = (triton.cdiv(hw_size, block_hw), batch)
    _rmsnorm_nchw_dim1_kernel_compiled[grid](
        x, w, y,
        x.stride(0),
        x.stride(1),
        hw_size,
        eps,
        BLOCK_HW=block_hw,
        CHANNELS=channels,
        num_warps=4 if block_hw <= 256 else 8,
        num_stages=1,
    )
    return y


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
    AFFINE_MODE: tl.constexpr,
):
    """Parameterized RMSNorm kernel.

    Computes y = x * w / sqrt(mean(x^2) + eps) for each row.
    One program per row, BLOCK_SIZE elements processed in parallel.

    AFFINE_MODE=0: standard y = x * rstd * w
    AFFINE_MODE=1: gemma    y = x * rstd * (1 + w)
    Constexpr branch is compiled away, so there is zero runtime cost.
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
    if AFFINE_MODE == 0:
        y = x * rstd * w
    else:
        y = x * rstd * (1.0 + w)
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


def rmsnorm(x, w, config, eps=1e-6, affine_mode: int = 0):
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


def torch_rmsnorm(x, w, eps=1e-6, affine_mode: int = 0):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_n = x * torch.rsqrt(variance + eps)
    if affine_mode == 1:
        return (x_n * (1.0 + w)).to(torch.float16)
    return (x_n * w).to(torch.float16)


def benchmark_one(n_rows, hidden_dim, config, dtype=torch.float16, affine_mode: int = 0):
    try:
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=dtype)
        w = torch.randn((hidden_dim,), device="cuda", dtype=dtype)
        ref = torch_rmsnorm(x, w, affine_mode=affine_mode)
        out = rmsnorm(x, w, config, affine_mode=affine_mode)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.05:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}
        ms = triton.testing.do_bench(lambda: rmsnorm(x, w, config, affine_mode=affine_mode), warmup=25, rep=100)
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
            affine_mode = shape.get("affine_mode", 0)
            result = benchmark_one(n_rows, hidden_dim, config, affine_mode=affine_mode)
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
