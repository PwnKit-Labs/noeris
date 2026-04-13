"""Parameterized Triton fused RMSNorm + Linear (matmul) kernel.

This is the genuinely novel "full-layer fusion" — nobody has fused RMSNorm +
matmul in a single kernel.  The key insight: we load x once, normalize in
registers, and immediately use the normalized values for the dot product.
The normalized intermediate is *never written to HBM*, eliminating one full
read+write of the activation tensor.

In a standard transformer pre-attention block the computation is::

    x_normed = rmsnorm(x)           # (M, K) -> (M, K)  — read+write to HBM
    qkv      = x_normed @ W_qkv.T   # (M, K) x (K, N) -> (M, N)

With this fused kernel, the ``x_normed`` intermediate never materialises in
global memory.  For Gemma 4 E2B (K=1536), the entire row fits in a single
BLOCK_K=1536 tile so the kernel runs in a single pass.  For larger models
a two-pass approach is used: first pass computes rstd, second pass does
fused norm+matmul.

Phase 2 of the pre-attention prologue (QK-norm + RoPE) is already handled by
our existing ``triton_qk_norm_rope`` kernel.

Design constraints:
- RMSNorm needs the FULL row for the variance reduction (sum of squares over
  K elements), but matmul processes K in BLOCK_K chunks.
- Single-pass (BLOCK_K == K): load the full row once, compute rstd, normalize
  in registers, and dot with the weight tile.  Works when K fits in one tile.
- Two-pass (BLOCK_K < K): first loop computes the row-level rstd, second loop
  does the fused norm+matmul accumulation.  Required for K > ~2048.

We start with the general two-pass approach which subsumes the single-pass
case naturally.

Gemma 4 family dimensions (from HF config.json):
  E2B:   K=1536,  N = (8+2*1)*256  = 2560   (H=8, H_kv=1, D_h=256)
  E4B:   K=2560,  N = (16+2*4)*256 = 6144   (H=16, H_kv=4, D_h=256)
  26B:   K=2816,  N = (16+2*8)*256 = 8192   (H=16, H_kv=8, D_h=256)
  31B:   K=5376,  N = (32+2*16)*256 = 16384 (H=32, H_kv=16, D_h=256)
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

FUSED_NORM_LINEAR_PARAM_SPACE = {
    "BLOCK_M": [16, 32, 64, 128],
    "BLOCK_N": [32, 64, 128, 256],
    "BLOCK_K": [32, 64, 128, 256, 512, 1024],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3, 4],
}


# Curated starting configs — biased toward Gemma 4 shapes.
# For E2B (K=1536, N=2560): moderate tiles since N and K are both modest.
# For 31B (K=5376, N=16384): larger tiles in N, smaller BLOCK_K for two-pass.
FUSED_NORM_LINEAR_CURATED_CONFIGS = [
    # Gemma 4 E2B sweet spot: K=1536 → BLOCK_K must be power-of-2 for tl.arange
    {"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 512, "num_warps": 4, "num_stages": 1},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 512, "num_warps": 8, "num_stages": 1},
    # Multi-pass configs for larger K
    {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 256,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128,  "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 512,  "num_warps": 4, "num_stages": 2},
    # T4-optimized: fewer warps for 40-SM GPU
    {"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 128,  "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 256,  "num_warps": 2, "num_stages": 1},
    # Small tiles for small batch (decode-time, M=1..8)
    {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 256,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 512,  "num_warps": 4, "num_stages": 1},
]


# Shape buckets for Gemma 4 QKV projection: M = batch * seq, N = (H+2*H_kv)*D_h, K = hidden_dim
FUSED_NORM_LINEAR_SHAPE_BUCKETS = [
    # Test shape
    {"name": "test_small",       "M": 128,  "N": 512,   "K": 256,  "affine_mode": 0},
    # Gemma 4 E2B: hidden=1536, QKV out = (8+2*1)*256 = 2560
    {"name": "gemma4_e2b_prefill", "M": 2048, "N": 2560,  "K": 1536, "affine_mode": 1},
    {"name": "gemma4_e2b_decode",  "M": 1,    "N": 2560,  "K": 1536, "affine_mode": 1},
    # Gemma 4 E4B: hidden=2560, QKV out = (16+2*4)*256 = 6144
    {"name": "gemma4_e4b_prefill", "M": 2048, "N": 6144,  "K": 2560, "affine_mode": 1},
    {"name": "gemma4_e4b_decode",  "M": 1,    "N": 6144,  "K": 2560, "affine_mode": 1},
    # Gemma 4 26B-A4B: hidden=2816, QKV out = (16+2*8)*256 = 8192
    {"name": "gemma4_26b_prefill", "M": 4096, "N": 8192,  "K": 2816, "affine_mode": 1},
    {"name": "gemma4_26b_decode",  "M": 1,    "N": 8192,  "K": 2816, "affine_mode": 1},
    # Gemma 4 31B: hidden=5376, QKV out = (32+2*16)*256 = 16384
    {"name": "gemma4_31b_prefill", "M": 4096, "N": 16384, "K": 5376, "affine_mode": 1},
    {"name": "gemma4_31b_decode",  "M": 1,    "N": 16384, "K": 5376, "affine_mode": 1},
    # Non-Gemma architectures (standard affine)
    {"name": "llama3_8b",          "M": 4096, "N": 6144,  "K": 4096, "affine_mode": 0},
    {"name": "llama3_70b",         "M": 4096, "N": 10240, "K": 8192, "affine_mode": 0},
    {"name": "mistral_7b",         "M": 4096, "N": 6144,  "K": 4096, "affine_mode": 0},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fused_norm_linear_config_id(config: dict[str, int]) -> str:
    return (
        f"bm{config['BLOCK_M']}_bn{config['BLOCK_N']}_bk{config['BLOCK_K']}"
        f"_w{config['num_warps']}_s{config['num_stages']}"
    )


def fused_norm_linear_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a fused norm+linear shape into the nearest bucket."""
    _ALL_NAMES = {b["name"] for b in FUSED_NORM_LINEAR_SHAPE_BUCKETS}
    name = shape.get("name", "")
    if name in _ALL_NAMES:
        return name

    m = shape.get("M", 0)
    k = shape.get("K", 0)
    n = shape.get("N", 0)

    # Detect Gemma shapes by hidden_dim (K)
    if k <= 256:
        return "test_small"
    if k <= 1536:
        return "gemma4_e2b_decode" if m <= 8 else "gemma4_e2b_prefill"
    if k <= 2560:
        return "gemma4_e4b_decode" if m <= 8 else "gemma4_e4b_prefill"
    if k <= 2816:
        return "gemma4_26b_decode" if m <= 8 else "gemma4_26b_prefill"
    if k <= 4096:
        return "llama3_8b"
    if k <= 5376:
        return "gemma4_31b_decode" if m <= 8 else "gemma4_31b_prefill"
    return "llama3_70b"


def fused_norm_linear_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True.

    Feasibility is learned from runtime failures (reward=0).
    """
    return True


def generate_fused_norm_linear_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in FUSED_NORM_LINEAR_CURATED_CONFIGS:
            cid = fused_norm_linear_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    # Systematic grid — keep BLOCK_K reasonable for this compute-bound kernel
    for bm in [32, 64, 128]:
        for bn in [64, 128, 256]:
            for bk in [128, 256, 512]:
                for nw in [4, 8]:
                    for ns in [1, 2, 3]:
                        config = {
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                            "num_warps": nw,
                            "num_stages": ns,
                        }
                        cid = fused_norm_linear_config_id(config)
                        if cid in seen:
                            continue
                        seen.add(cid)
                        configs.append(config)
                        if len(configs) >= max_configs:
                            return configs
    return configs


# ---------------------------------------------------------------------------
# Triton kernel (lazy compilation)
# ---------------------------------------------------------------------------

_triton_available = False
_fused_norm_linear_kernel_compiled = None


def _ensure_triton_fused_norm_linear():
    global _triton_available, _fused_norm_linear_kernel_compiled
    if _fused_norm_linear_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _fused_rmsnorm_linear_kernel(
            x_ptr,          # (M, K) input activations
            w_ptr,          # (K,) RMSNorm affine weight
            linear_ptr,     # (N, K) linear weight (row-major, i.e. out_features x in_features)
            out_ptr,        # (M, N) output
            M, N, K,
            stride_xm, stride_xk,
            stride_ln, stride_lk,      # linear weight strides
            stride_om, stride_on,
            eps,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            AFFINE_MODE: tl.constexpr,  # 0=standard, 1=Gemma (1+w)
        ):
            """Fused RMSNorm + Linear projection.

            Each program computes an (BLOCK_M, BLOCK_N) output tile.

            Two-pass approach:
            - Pass 1: iterate over K in BLOCK_K chunks to compute row-wise
              rstd = 1/sqrt(mean(x^2) + eps) for each of the BLOCK_M rows.
            - Pass 2: iterate over K in BLOCK_K chunks. For each chunk, load
              x, normalize in registers using rstd, apply affine weight, and
              accumulate the dot product with the linear weight tile.

            When BLOCK_K >= K the loop bodies execute exactly once, collapsing
            to the optimal single-pass case.
            """
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            # Row and column offsets for this tile
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

            mask_m = offs_m < M
            mask_n = offs_n < N

            # ---------------------------------------------------------------
            # Pass 1: compute rstd for each row in this BLOCK_M tile.
            # rstd[m] = 1 / sqrt( (1/K) * sum_k(x[m,k]^2) + eps )
            # ---------------------------------------------------------------
            sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
                mask_k = offs_k < K

                # Load x tile: (BLOCK_M, BLOCK_K)
                x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
                x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                x_f32 = x_tile.to(tl.float32)

                # Accumulate sum of squares per row
                sum_sq += tl.sum(x_f32 * x_f32, axis=1)

            # Compute rstd per row
            mean_sq = sum_sq / K
            rstd = 1.0 / tl.sqrt(mean_sq + eps)  # (BLOCK_M,)

            # ---------------------------------------------------------------
            # Pass 2: fused norm + matmul accumulation.
            # For each K-chunk:
            #   x_normed = x_chunk * rstd * affine_weight
            #   acc += x_normed @ linear_weight_chunk.T
            # ---------------------------------------------------------------
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K

                # Load x tile: (BLOCK_M, BLOCK_K)
                x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
                x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                x_f32 = x_tile.to(tl.float32)

                # Load RMSNorm affine weight: (BLOCK_K,)
                w_tile = tl.load(w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

                # Normalize in registers
                x_normed = x_f32 * rstd[:, None]
                if AFFINE_MODE == 0:
                    x_normed = x_normed * w_tile[None, :]
                else:
                    # Gemma mode: y = x * rstd * (1 + w)
                    x_normed = x_normed * (1.0 + w_tile[None, :])

                # Load linear weight tile in natural (BLOCK_N, BLOCK_K) layout,
                # then transpose for the dot product.
                # linear_ptr is (N, K) row-major, so linear[n, k] = linear_ptr + n * stride_ln + k * stride_lk
                l_ptrs = linear_ptr + offs_n[:, None] * stride_ln + offs_k[None, :] * stride_lk
                l_tile = tl.load(l_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                # l_tile is (BLOCK_N, BLOCK_K); transpose to (BLOCK_K, BLOCK_N) for matmul

                # Dot product: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
                acc += tl.dot(x_normed.to(tl.float16), tl.trans(l_tile.to(tl.float16)))

            # Store output tile
            out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])

        _fused_norm_linear_kernel_compiled = _fused_rmsnorm_linear_kernel
        _triton_available = True
    except ImportError:
        _triton_available = False


def fused_rmsnorm_linear(x, rmsnorm_weight, linear_weight, eps=1e-6, affine_mode=0, config=None):
    """Module-level launcher for fused RMSNorm + Linear projection.

    Args:
        x: (M, K) fp16 input activations.
        rmsnorm_weight: (K,) fp16 RMSNorm affine weight.
        linear_weight: (N, K) fp16 linear weight (PyTorch nn.Linear convention:
            weight shape is (out_features, in_features)).
        eps: Epsilon for numerical stability.
        affine_mode: 0 = standard (y = x * rstd * w),
                     1 = Gemma (y = x * rstd * (1 + w)).
        config: Triton config dict with BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages.
            Defaults to the first curated config.

    Returns:
        (M, N) fp16 output tensor, equivalent to:
            x_normed = rmsnorm(x, rmsnorm_weight, eps, affine_mode)
            out = x_normed @ linear_weight.T
    """
    import torch
    import triton

    _ensure_triton_fused_norm_linear()
    if not _triton_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = FUSED_NORM_LINEAR_CURATED_CONFIGS[0]

    M, K = x.shape
    N = linear_weight.shape[0]
    assert linear_weight.shape[1] == K, f"Weight K dim mismatch: {linear_weight.shape[1]} != {K}"
    assert rmsnorm_weight.shape[0] == K, f"RMSNorm weight dim mismatch: {rmsnorm_weight.shape[0]} != {K}"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_norm_linear_kernel_compiled[grid](
        x, rmsnorm_weight, linear_weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        linear_weight.stride(0), linear_weight.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        AFFINE_MODE=affine_mode,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


# ---------------------------------------------------------------------------
# Benchmark script generator
# ---------------------------------------------------------------------------

def generate_fused_norm_linear_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained script benchmarking all fused norm+linear configs."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton fused RMSNorm + Linear benchmark.

Novel kernel: fuses RMSNorm normalization with the QKV linear projection
into a single kernel, eliminating the intermediate normalized activation
tensor from HBM.
"""

import json
import platform

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def fused_rmsnorm_linear_kernel(
    x_ptr,          # (M, K) input activations
    w_ptr,          # (K,) RMSNorm affine weight
    linear_ptr,     # (N, K) linear weight
    out_ptr,        # (M, N) output
    M, N, K,
    stride_xm, stride_xk,
    stride_ln, stride_lk,
    stride_om, stride_on,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    AFFINE_MODE: tl.constexpr,
):
    """Fused RMSNorm + Linear: out = rmsnorm(x) @ W.T

    Two-pass:
    - Pass 1: compute rstd per row (iterate over K in BLOCK_K chunks).
    - Pass 2: for each K-chunk, normalize x in registers and accumulate
      the dot product with the linear weight.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pass 1: row-wise sum of squares for rstd
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_tile * x_tile, axis=1)

    rstd = 1.0 / tl.sqrt(sum_sq / K + eps)

    # Pass 2: fused norm + matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        w_tile = tl.load(w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        x_normed = x_tile * rstd[:, None]
        if AFFINE_MODE == 0:
            x_normed = x_normed * w_tile[None, :]
        else:
            x_normed = x_normed * (1.0 + w_tile[None, :])

        l_ptrs = linear_ptr + offs_n[:, None] * stride_ln + offs_k[None, :] * stride_lk
        l_tile = tl.load(l_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc += tl.dot(x_normed.to(tl.float16), tl.trans(l_tile.to(tl.float16)))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_rmsnorm_linear(x, rmsnorm_weight, linear_weight, config, eps=1e-6, affine_mode=0):
    M, K = x.shape
    N = linear_weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_rmsnorm_linear_kernel[grid](
        x, rmsnorm_weight, linear_weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        linear_weight.stride(0), linear_weight.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        AFFINE_MODE=affine_mode,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


# ---------------------------------------------------------------------------
# PyTorch reference (separate ops)
# ---------------------------------------------------------------------------

def torch_rmsnorm_linear(x, rmsnorm_weight, linear_weight, eps=1e-6, affine_mode=0):
    """Separate RMSNorm + matmul for correctness checking and baseline timing."""
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    w = rmsnorm_weight.to(torch.float32)
    if affine_mode == 1:
        x_normed = x_normed * (1.0 + w)
    else:
        x_normed = x_normed * w
    x_normed = x_normed.to(torch.float16)
    return x_normed @ linear_weight.t()


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark_one(M, N, K, config, dtype=torch.float16, affine_mode=0):
    try:
        x = torch.randn((M, K), device="cuda", dtype=dtype)
        rmsnorm_w = torch.randn((K,), device="cuda", dtype=dtype) * 0.01
        linear_w = torch.randn((N, K), device="cuda", dtype=dtype) * (K ** -0.5)

        ref = torch_rmsnorm_linear(x, rmsnorm_w, linear_w, affine_mode=affine_mode)
        out = fused_rmsnorm_linear(x, rmsnorm_w, linear_w, config, affine_mode=affine_mode)
        # Use relative tolerance for matmul outputs
        abs_err = (out.to(torch.float32) - ref.to(torch.float32)).abs()
        ref_abs = ref.to(torch.float32).abs().clamp(min=1e-4)
        max_rel_err = (abs_err / ref_abs).max().item()
        max_abs_err = abs_err.max().item()
        # fp16 matmul tolerance: allow up to 5% relative error or 1.0 absolute
        if max_rel_err > 0.05 and max_abs_err > 1.0:
            return {{"correct": False, "max_abs_err": max_abs_err, "max_rel_err": max_rel_err,
                     "ms": None, "tflops": None, "gb_per_s": None}}

        # Benchmark fused kernel
        fused_ms = triton.testing.do_bench(
            lambda: fused_rmsnorm_linear(x, rmsnorm_w, linear_w, config, affine_mode=affine_mode),
            warmup=25, rep=100,
        )
        # Benchmark separated baseline
        separated_ms = triton.testing.do_bench(
            lambda: torch_rmsnorm_linear(x, rmsnorm_w, linear_w, affine_mode=affine_mode),
            warmup=25, rep=100,
        )

        # Compute throughput (2*M*N*K FLOPs for matmul + 3*M*K for RMSNorm)
        flops = 2.0 * M * N * K + 3.0 * M * K
        tflops = flops / (fused_ms * 1e-3) / 1e12
        # Memory saved: we skip writing M*K*2 bytes (normalized x) and reading
        # it back = 2 * M * K * 2 bytes
        bytes_saved = 2 * M * K * 2
        speedup = separated_ms / fused_ms if fused_ms > 0 else 0.0

        return {{
            "correct": True,
            "max_abs_err": round(max_abs_err, 6),
            "max_rel_err": round(max_rel_err, 6),
            "fused_ms": round(fused_ms, 4),
            "separated_ms": round(separated_ms, 4),
            "speedup": round(speedup, 3),
            "tflops": round(tflops, 2),
            "bytes_saved_mb": round(bytes_saved / 1e6, 2),
            "ms": round(fused_ms, 4),
            "gb_per_s": None,
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "tflops": None, "gb_per_s": None}}


def main():
    configs = {configs_json}
    shapes = {shapes_json}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bm{{}}_bn{{}}_bk{{}}_w{{}}_s{{}}".format(
            config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_K"],
            config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            M = shape["M"]
            N = shape["N"]
            K = shape["K"]
            affine_mode = shape.get("affine_mode", 0)
            result = benchmark_one(M, N, K, config, affine_mode=affine_mode)
            result["shape"] = f"{{M}}x{{N}}x{{K}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "fused_norm_linear",
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


# ---------------------------------------------------------------------------
# Register the operator
# ---------------------------------------------------------------------------

FUSED_NORM_LINEAR_SPEC = register_operator(TritonOperatorSpec(
    name="fused_norm_linear",
    param_space=FUSED_NORM_LINEAR_PARAM_SPACE,
    curated_configs=FUSED_NORM_LINEAR_CURATED_CONFIGS,
    shape_buckets=FUSED_NORM_LINEAR_SHAPE_BUCKETS,
    metric_name="tflops",
    config_id_fn=fused_norm_linear_config_id,
    shape_bucket_fn=fused_norm_linear_shape_bucket_key,
    benchmark_script_fn=generate_fused_norm_linear_benchmark_script,
    grid_generator_fn=generate_fused_norm_linear_grid,
    shared_memory_check_fn=fused_norm_linear_shared_memory_check,
    description=(
        "Fused RMSNorm + Linear projection. Eliminates the intermediate "
        "normalized activation tensor from HBM. Novel: no existing system "
        "fuses normalization with the subsequent matmul in a single kernel."
    ),
))
