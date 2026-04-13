"""Hand-written CUDA fused QK-RMSNorm+RoPE for Turing T4 (SM 7.5).

Bypasses Triton codegen (3-5x PTX overhead on T4) with warp-shuffle
reduction, shared-memory cos/sin caching, one block per row.
Same semantics as triton_qk_norm_rope.py.  Issue: #64
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator

CUDA_QK_NORM_ROPE_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// One block per (batch, head, seq_pos) row.
// blockDim.x = half = head_dim / 2  (capped at 256 threads for T4).
// Each thread handles one even/odd pair: x[2*tid] and x[2*tid+1].

__global__ void qk_norm_rope_kernel(
    const half* __restrict__ x,         // [rows, head_dim]
    const float* __restrict__ scale,    // [head_dim]
    const float* __restrict__ cos_tab,  // [seq_len, half]
    const float* __restrict__ sin_tab,  // [seq_len, half]
    half* __restrict__ out,             // [rows, head_dim]
    int seq_len,
    int head_dim,
    int partial_rotary_pairs,
    float eps
) {
    // Row index = blockIdx.x; thread index covers one pair
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int half_dim = head_dim / 2;

    // Bounds check: threads beyond half_dim are inactive
    const bool active = (tid < half_dim);

    const int base = row * head_dim;
    float xe = 0.0f, xo = 0.0f;
    if (active) {
        xe = __half2float(x[base + 2 * tid]);
        xo = __half2float(x[base + 2 * tid + 1]);
    }

    // Warp-level sum-of-squares reduction
    float local_sq = xe * xe + xo * xo;
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq += __shfl_xor_sync(mask, local_sq, offset);
    }
    // Cross-warp reduction via shared memory (max 8 warps on T4)
    __shared__ float warp_sums[8];
    __shared__ float cos_cache[256];
    __shared__ float sin_cache[256];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sq;
    }
    __syncthreads();

    float total_sq = 0.0f;
    int num_warps = (blockDim.x + 31) / 32;
    if (tid < num_warps) {
        total_sq = warp_sums[tid];
    }
    for (int offset = 4; offset > 0; offset >>= 1) {
        total_sq += __shfl_xor_sync(mask, total_sq, offset);
    }

    __shared__ float rstd_shared;
    if (tid == 0) {
        float mean_sq = total_sq / (float)head_dim;
        rstd_shared = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rstd = rstd_shared;

    // Cache cos/sin row in shared memory; s_idx = row % seq_len
    const int s_idx = row % seq_len;
    const int cos_base = s_idx * half_dim;
    if (active) {
        cos_cache[tid] = cos_tab[cos_base + tid];
        sin_cache[tid] = sin_tab[cos_base + tid];
    }
    __syncthreads();

    if (!active) return;

    // Gemma-mode affine: normed = x * rstd * (1 + scale)
    float se = scale[2 * tid];
    float so = scale[2 * tid + 1];
    float ne = xe * rstd * (1.0f + se);
    float no = xo * rstd * (1.0f + so);

    float out_e, out_o;
    if (tid < partial_rotary_pairs) {
        float c  = cos_cache[tid];
        float sn = sin_cache[tid];
        out_e = ne * c - no * sn;
        out_o = ne * sn + no * c;
    } else {
        out_e = ne;
        out_o = no;
    }

    out[base + 2 * tid]     = __float2half(out_e);
    out[base + 2 * tid + 1] = __float2half(out_o);
}
"""

CUDA_LAUNCHER_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void qk_norm_rope_kernel(
    const half*, const float*, const float*, const float*,
    half*, int, int, int, float);

torch::Tensor cuda_qk_norm_rope_forward(
    torch::Tensor x_flat, torch::Tensor scale,
    torch::Tensor cos_tab, torch::Tensor sin_tab,
    int seq_len, int partial_rotary_pairs, float eps) {
    int rows = x_flat.size(0), hd = x_flat.size(1), half_dim = hd / 2;
    auto out = torch::empty_like(x_flat);
    int threads = ((half_dim + 31) / 32) * 32;
    if (threads > 256) threads = 256;
    qk_norm_rope_kernel<<<rows, threads>>>(
        reinterpret_cast<const half*>(x_flat.data_ptr<at::Half>()),
        scale.data_ptr<float>(), cos_tab.data_ptr<float>(),
        sin_tab.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        seq_len, hd, partial_rotary_pairs, eps);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cuda_qk_norm_rope_forward, "Fused QK-RMSNorm+RoPE (CUDA)");
}
"""

_cuda_module = None


def _ensure_cuda_compiled():
    """Compile the CUDA kernel via load_inline (cached after first call)."""
    global _cuda_module
    if _cuda_module is not None:
        return

    from torch.utils.cpp_extension import load_inline

    _cuda_module = load_inline(
        name="cuda_qk_norm_rope",
        cpp_sources=[CUDA_LAUNCHER_SRC],
        cuda_sources=[CUDA_QK_NORM_ROPE_SRC],
        functions=["cuda_qk_norm_rope_forward"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-arch=sm_75",  # Turing (T4)
        ],
        verbose=False,
    )


def apply_cuda_qk_norm_rope(
    q, k, cos, sin, q_scale, k_scale, eps=1e-6, partial_ratio=1.0,
):
    """Launch CUDA fused QK-RMSNorm+RoPE. Same API as the Triton version."""
    import torch

    _ensure_cuda_compiled()

    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape
    half = D // 2
    partial_rotary_pairs = int(half * partial_ratio)

    q_flat = q.reshape(B * H * S, D).contiguous()
    k_flat = k.reshape(B * H_kv * S, D).contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    q_out_flat = _cuda_module.forward(
        q_flat, q_scale, cos_c, sin_c, S, partial_rotary_pairs, eps,
    )
    k_out_flat = _cuda_module.forward(
        k_flat, k_scale, cos_c, sin_c, S, partial_rotary_pairs, eps,
    )

    q_out = q_out_flat.reshape(B, H, S, D)
    k_out = k_out_flat.reshape(B, H_kv, S, D)
    return q_out, k_out


CUDA_QK_NORM_ROPE_SHAPE_BUCKETS = [
    # T4-relevant subset: smaller batch for 16 GB VRAM
    {"name": "gemma4_e2b_local_t4", "batch": 1, "heads": 8, "num_kv_heads": 1, "seq": 2048, "head_dim": 256},
    {"name": "llama3_8b_t4", "batch": 1, "heads": 32, "num_kv_heads": 8, "seq": 2048, "head_dim": 128},
    {"name": "gemma4_26b_a4b_local_t4", "batch": 1, "heads": 16, "num_kv_heads": 8, "seq": 2048, "head_dim": 256},
]


def cuda_qk_norm_rope_config_id(config: dict[str, int]) -> str:
    return f"cuda_t4_threads{config.get('threads', 128)}"


def cuda_qk_norm_rope_shape_bucket_key(shape: dict[str, int]) -> str:
    hd = shape.get("head_dim", 0)
    h = shape.get("heads", 0)
    if hd >= 256 and h >= 16:
        return "gemma4_26b_a4b_local_t4"
    if hd >= 256:
        return "gemma4_e2b_local_t4"
    return "llama3_8b_t4"


def generate_cuda_qk_norm_rope_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained CUDA QK-RMSNorm+RoPE benchmark for T4."""
    shapes_json = json.dumps(shapes)
    return f'''#!/usr/bin/env python3
"""Auto-generated CUDA fused QK-RMSNorm+RoPE benchmark (T4)."""
import json, time, torch
from research_engine.cuda_qk_norm_rope import apply_cuda_qk_norm_rope

SHAPES = json.loads({shapes_json!r})

def separated(q, k, cos, sin, qs, ks, eps=1e-6, pr=1.0):
    qv = q.float().pow(2).mean(-1, keepdim=True)
    qn = (q.float() * torch.rsqrt(qv + eps)).half() * (1 + qs).half()
    kv = k.float().pow(2).mean(-1, keepdim=True)
    kn = (k.float() * torch.rsqrt(kv + eps)).half() * (1 + ks).half()
    h = q.shape[-1] // 2; pp = int(h * pr)
    c, sn = cos[None,None,:,:pp].half(), sin[None,None,:,:pp].half()
    qe, qo = qn[...,0::2].clone(), qn[...,1::2].clone()
    qe[...,:pp], qo[...,:pp] = qe[...,:pp]*c - qo[...,:pp]*sn, qe[...,:pp]*sn + qo[...,:pp]*c
    ke, ko = kn[...,0::2].clone(), kn[...,1::2].clone()
    ke[...,:pp], ko[...,:pp] = ke[...,:pp]*c - ko[...,:pp]*sn, ke[...,:pp]*sn + ko[...,:pp]*c
    return torch.stack([qe,qo],-1).reshape_as(q), torch.stack([ke,ko],-1).reshape_as(k)

def bench(fn, *a, warmup=10, rep=50, **kw):
    for _ in range(warmup): fn(*a, **kw)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(rep): fn(*a, **kw)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / rep * 1000

for s in SHAPES:
    B,H,Hk,S,D = s["batch"],s["heads"],s["num_kv_heads"],s["seq"],s["head_dim"]
    pr = s.get("partial_ratio", 1.0)
    q = torch.randn(B,H,S,D, device="cuda", dtype=torch.float16)
    k = torch.randn(B,Hk,S,D, device="cuda", dtype=torch.float16)
    cs = torch.randn(S,D//2, device="cuda", dtype=torch.float32)
    sn = torch.randn(S,D//2, device="cuda", dtype=torch.float32)
    qs = torch.randn(D, device="cuda", dtype=torch.float32) * 0.01
    ks = torch.randn(D, device="cuda", dtype=torch.float32) * 0.01
    ms_sep = bench(separated, q, k, cs, sn, qs, ks, pr=pr)
    ms_cuda = bench(apply_cuda_qk_norm_rope, q, k, cs, sn, qs, ks, partial_ratio=pr)
    print(f"{{s['name']}}: sep={{ms_sep:.3f}}ms cuda={{ms_cuda:.3f}}ms speedup={{ms_sep/ms_cuda:.1f}}x")
'''


def generate_cuda_qk_norm_rope_grid(**_kwargs) -> list[dict[str, int]]:
    """CUDA kernel has no tunable grid -- single config."""
    return [{"threads": 128}]


CUDA_QK_NORM_ROPE_SPEC = register_operator(TritonOperatorSpec(
    name="cuda_qk_norm_rope",
    param_space={"threads": [64, 128, 256]},
    curated_configs=[{"threads": 128}],
    shape_buckets=CUDA_QK_NORM_ROPE_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=cuda_qk_norm_rope_config_id,
    shape_bucket_fn=cuda_qk_norm_rope_shape_bucket_key,
    benchmark_script_fn=generate_cuda_qk_norm_rope_benchmark_script,
    grid_generator_fn=generate_cuda_qk_norm_rope_grid,
    description=(
        "Hand-written CUDA fused QK-RMSNorm+RoPE targeting Turing (T4, SM 7.5). "
        "Bypasses Triton codegen to avoid 3-5x PTX overhead on SM 7.5. "
        "Uses warp-shuffle reduction, shared-memory cos/sin caching, and "
        "half2 vectorized stores. Same semantics as triton_qk_norm_rope."
    ),
))
