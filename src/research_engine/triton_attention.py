"""Parameterized Triton FlashAttention kernel and operator spec.

Tiled scaled dot-product attention with online softmax (FlashAttention-style).
Computes: out = softmax(Q @ K.T * scale) @ V per head.

The kernel supports causal masking and operates on tiles of BLOCK_M queries
by BLOCK_N keys at a time, accumulating the online softmax normalizer so no
O(N^2) intermediate attention matrix is materialized.

References:
- FlashAttention-2 (Dao 2023): https://arxiv.org/abs/2307.08691
- FlashAttention-3 (Shah et al 2024): for Hopper/H100 TMA/WGMMA optimizations

Search space: BLOCK_M, BLOCK_N, num_warps, num_stages.
Different head dims and sequence lengths prefer different configurations.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


ATTENTION_PARAM_SPACE = {
    "BLOCK_M": [32, 64, 128],
    "BLOCK_N": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}


ATTENTION_CURATED_CONFIGS = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
]


# Shape buckets: (batch, num_heads, seq_len, head_dim)
# These match LLM workload patterns.
ATTENTION_SHAPE_BUCKETS = [
    {"name": "short_64", "batch": 4, "heads": 32, "seq_len": 512, "head_dim": 64},
    {"name": "short_128", "batch": 2, "heads": 16, "seq_len": 1024, "head_dim": 128},
    {"name": "med_128", "batch": 2, "heads": 16, "seq_len": 2048, "head_dim": 128},
    {"name": "long_64", "batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 64},
    {"name": "long_128", "batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 128},
    {"name": "llama7b", "batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128},
    {"name": "mistral", "batch": 1, "heads": 32, "seq_len": 8192, "head_dim": 128},
]


def attention_config_id(config: dict[str, int]) -> str:
    return f"m{config['BLOCK_M']}_n{config['BLOCK_N']}_w{config['num_warps']}_s{config['num_stages']}"


def attention_shape_bucket_key(shape: dict[str, int]) -> str:
    seq = shape.get("seq_len", 0)
    hd = shape.get("head_dim", 0)
    heads = shape.get("heads", 0)
    if seq >= 8192:
        return "mistral"
    if seq >= 4096:
        if hd <= 64:
            return "long_64"
        return "llama7b" if heads >= 32 else "long_128"
    if seq >= 2048:
        return "med_128"
    if hd <= 64:
        return "short_64"
    return "short_128"


def attention_shared_memory_check(config: dict[str, int]) -> bool:
    """Approximate shared memory limit for A100 (192 KB per SM).

    Attention tiles: BLOCK_M x head_dim (Q) + BLOCK_N x head_dim (K) + BLOCK_N x head_dim (V)
    Using max head_dim=128 and fp16 (2 bytes).
    """
    bm = config.get("BLOCK_M", 0)
    bn = config.get("BLOCK_N", 0)
    ns = config.get("num_stages", 1)
    max_head_dim = 128
    shmem = (bm * max_head_dim + 2 * bn * max_head_dim) * 2 * ns + 2048
    return shmem <= 192_000


def generate_attention_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in ATTENTION_CURATED_CONFIGS:
            cid = attention_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bm in ATTENTION_PARAM_SPACE["BLOCK_M"]:
        for bn in ATTENTION_PARAM_SPACE["BLOCK_N"]:
            for nw in ATTENTION_PARAM_SPACE["num_warps"]:
                for ns in [2, 3]:
                    config = {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "num_warps": nw,
                        "num_stages": ns,
                    }
                    if not attention_shared_memory_check(config):
                        continue
                    cid = attention_config_id(config)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    configs.append(config)
                    if len(configs) >= max_configs:
                        return configs
    return configs


def generate_attention_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained script benchmarking all attention configs.

    Uses a minimal Triton FlashAttention kernel derived from the Triton
    tutorial. Non-causal for simplicity; causal masking can be added later.
    """
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton FlashAttention benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled FlashAttention forward (non-causal) with online softmax.

    Each program handles one (batch, head, query-block) tile.
    """
    pid = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q block once, keep resident
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax state
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e) for tl.exp2

    for start_n in range(0, N, BLOCK_N):
        curr_n = start_n + offs_n
        k_ptrs = k_base + curr_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = v_base + curr_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        n_mask = curr_n[:, None] < N
        k = tl.load(k_ptrs, mask=n_mask, other=0.0)
        v = tl.load(v_ptrs, mask=n_mask, other=0.0)

        # Scores: Q @ K.T
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        qk = tl.where(curr_n[None, :] < N, qk, -float("inf"))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def flash_attn(q, k, v, config, sm_scale=None):
    """q, k, v: (batch, heads, seq, head_dim) float16. Returns out."""
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N,
        sm_scale,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def benchmark_one(batch, heads, seq_len, head_dim, config):
    try:
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

        # Reference: non-causal SDPA
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = flash_attn(q, k, v, config)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "tflops": None}}

        ms = triton.testing.do_bench(lambda: flash_attn(q, k, v, config), warmup=10, rep=50)
        # FLOPs: 2 * B * H * M * N * D (QK) + 2 * B * H * M * N * D (PV)
        flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
        tflops = flops / (ms * 1e-3) / 1e12
        return {{
            "correct": True,
            "max_err": round(max_err, 5),
            "ms": round(ms, 4),
            "tflops": round(tflops, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "tflops": None}}


def main():
    configs = {configs_json}
    shapes = {shapes_json}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "m{{}}_n{{}}_w{{}}_s{{}}".format(
            config["BLOCK_M"], config["BLOCK_N"],
            config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            batch = shape["batch"]
            heads = shape["heads"]
            seq_len = shape["seq_len"]
            head_dim = shape["head_dim"]
            result = benchmark_one(batch, heads, seq_len, head_dim, config)
            result["shape"] = f"{{batch}}x{{heads}}x{{seq_len}}x{{head_dim}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "attention",
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


ATTENTION_SPEC = register_operator(TritonOperatorSpec(
    name="attention",
    param_space=ATTENTION_PARAM_SPACE,
    curated_configs=ATTENTION_CURATED_CONFIGS,
    shape_buckets=ATTENTION_SHAPE_BUCKETS,
    metric_name="tflops",
    config_id_fn=attention_config_id,
    shape_bucket_fn=attention_shape_bucket_key,
    benchmark_script_fn=generate_attention_benchmark_script,
    grid_generator_fn=generate_attention_grid,
    shared_memory_check_fn=attention_shared_memory_check,
    description="FlashAttention-style scaled dot-product attention with tiled online softmax.",
))
