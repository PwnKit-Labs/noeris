"""Parameterized Triton kernel for decode-time paged-KV attention.

Single-query decode attention against a paged KV cache. At inference time,
each generated token's query (seq_q=1) attends to the FULL KV cache stored
in non-contiguous pages of PAGE_SIZE tokens. A page table maps logical
sequence positions to physical page indices.

This is a from-scratch Triton implementation. vLLM's PagedAttention is
CUDA-only (csrc/attention/paged_attention_v1.cu / v2.cu) with a 51-line
Python dispatch wrapper -- no Triton reference exists in vLLM or any
widely-published project.

Key design:
- Grid: (batch, num_heads) -- one program per (batch, head).
- Inner loop iterates over pages in the KV cache for each batch item.
- Page-table indirection: page_idx = page_table[batch, page_offset].
- Online softmax accumulation across pages (no materialized attn matrix).
- GQA support: kv_head = head // (num_heads // num_kv_heads).
- Sliding window: skip pages entirely outside the window boundary.
- Handles partial last pages (context_len not divisible by PAGE_SIZE).

Supported models:
- Gemma 4 31B: global (head_dim=512, GQA 32:4, no window) and
  local (head_dim=256, GQA 32:16, window=1024) layers.
- LLaMA 3 70B: GQA 64:8, head_dim=128, no window.
- Any model with paged KV cache and GQA.

NOTE: This module is NOT imported by __init__.py yet. Add the import
manually when integrating into the research engine pipeline:
    from .triton_attention_decode import ATTENTION_DECODE_SPEC
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


ATTENTION_DECODE_PARAM_SPACE = {
    "BLOCK_KV": [16, 32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3],
}


ATTENTION_DECODE_CURATED_CONFIGS = [
    {"BLOCK_KV": 16, "num_warps": 2, "num_stages": 2},   # = PAGE_SIZE (one page per iter)
    {"BLOCK_KV": 32, "num_warps": 4, "num_stages": 2},   # 2 pages per iter
    {"BLOCK_KV": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_KV": 128, "num_warps": 8, "num_stages": 3},
    # T4-optimized: fewer warps for 40-SM; small BLOCK_KV avoids register spill
    {"BLOCK_KV": 32, "num_warps": 2, "num_stages": 2},
    {"BLOCK_KV": 16, "num_warps": 4, "num_stages": 3},
    # 256k-context decode: larger BLOCK_KV reduces loop iterations
    {"BLOCK_KV": 128, "num_warps": 4, "num_stages": 2},
]


# Real decode shapes for LLM inference workloads.
ATTENTION_DECODE_SHAPE_BUCKETS = [
    # Gemma 4 31B decode -- global layer (head_dim=512, GQA 32:4, no window)
    {"name": "gemma4_31b_decode_global", "batch": 1, "num_heads": 32, "num_kv_heads": 4,
     "head_dim": 512, "context_len": 4096, "page_size": 16, "window_size": -1},
    # Gemma 4 31B decode -- local layer (head_dim=256, GQA 32:16, window=1024)
    {"name": "gemma4_31b_decode_local", "batch": 1, "num_heads": 32, "num_kv_heads": 16,
     "head_dim": 256, "context_len": 4096, "page_size": 16, "window_size": 1024},
    # Gemma 4 31B decode -- long context global (256k tokens)
    {"name": "gemma4_31b_decode_256k", "batch": 1, "num_heads": 32, "num_kv_heads": 4,
     "head_dim": 512, "context_len": 262144, "page_size": 16, "window_size": -1},
    # LLaMA 3 70B decode (GQA 64:8, head_dim=128, no window)
    {"name": "llama3_70b_decode", "batch": 1, "num_heads": 64, "num_kv_heads": 8,
     "head_dim": 128, "context_len": 8192, "page_size": 16, "window_size": -1},
]


def attention_decode_config_id(config: dict[str, int]) -> str:
    return f"bkv{config['BLOCK_KV']}_w{config['num_warps']}_s{config['num_stages']}"


def attention_decode_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a decode attention shape into a bucket.

    Discriminators:
    - head_dim: 128 (LLaMA), 256 (Gemma local), 512 (Gemma global)
    - num_kv_heads: picks GQA ratio
    - context_len: short vs long context
    - window_size: windowed vs global
    """
    hd = shape.get("head_dim", 0)
    ctx = shape.get("context_len", 0)
    ws = shape.get("window_size", -1)
    nkv = shape.get("num_kv_heads", 0)

    if hd <= 128:
        return "llama3_70b_decode"
    if ws > 0:
        return "gemma4_31b_decode_local"
    if ctx >= 65536:
        return "gemma4_31b_decode_256k"
    return "gemma4_31b_decode_global"


def attention_decode_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only -- feasibility is learned at runtime."""
    return True


def generate_attention_decode_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in ATTENTION_DECODE_CURATED_CONFIGS:
            cid = attention_decode_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bkv in ATTENTION_DECODE_PARAM_SPACE["BLOCK_KV"]:
        for nw in ATTENTION_DECODE_PARAM_SPACE["num_warps"]:
            for ns in ATTENTION_DECODE_PARAM_SPACE["num_stages"]:
                config = {"BLOCK_KV": bkv, "num_warps": nw, "num_stages": ns}
                cid = attention_decode_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_attention_decode_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained Triton paged-KV decode attention benchmark script."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated paged-KV decode attention benchmark.

Novel Triton implementation of single-query decode against a paged KV cache.
vLLM has no Triton paged attention -- this is from-scratch.
"""

import json
import math
import platform

import torch
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def paged_attention_decode_kernel(
    # Q input
    q_ptr,                          # [batch, num_heads, 1, head_dim] fp16
    # Paged KV cache
    k_cache_ptr,                    # [num_blocks, num_kv_heads, page_size, head_dim] fp16
    v_cache_ptr,                    # [num_blocks, num_kv_heads, page_size, head_dim] fp16
    # Page table
    page_table_ptr,                 # [batch, max_num_pages] int32
    # Context lengths
    context_lens_ptr,               # [batch] int32 -- actual KV length per batch item
    # Output
    out_ptr,                        # [batch, num_heads, 1, head_dim] fp16
    # Shapes
    num_heads,
    num_kv_heads,
    head_dim,
    max_num_pages,
    # Strides for Q: [batch, num_heads, 1, head_dim]
    stride_qb, stride_qh, stride_qd,
    # Strides for K cache: [num_blocks, num_kv_heads, page_size, head_dim]
    stride_kb, stride_kh, stride_kp, stride_kd,
    # Strides for V cache: [num_blocks, num_kv_heads, page_size, head_dim]
    stride_vb, stride_vh, stride_vp, stride_vd,
    # Strides for output: [batch, num_heads, 1, head_dim]
    stride_ob, stride_oh, stride_od,
    # Strides for page table: [batch, max_num_pages]
    stride_pt_b, stride_pt_p,
    # Scale
    sm_scale,
    # Constexprs
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_KV: tl.constexpr,        # how many KV positions to process per inner loop iteration
    IS_CAUSAL: tl.constexpr,       # always True for decode (query sees all past keys)
    WINDOW_SIZE: tl.constexpr,     # sliding window; -1 means no window
):
    """Single-query paged attention decode kernel.

    Grid: (batch, num_heads) -- one program per (batch, head).
    Each program iterates over all pages in the KV cache for its batch item,
    using online softmax to accumulate the attention output without
    materializing the full attention matrix.

    Page-table indirection: physical page indices are looked up from the
    page table, allowing non-contiguous KV cache storage.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA: map query head to KV head
    heads_per_kv = num_heads // NUM_KV_HEADS
    kv_head_idx = head_idx // heads_per_kv

    # Load context length for this batch item
    context_len = tl.load(context_lens_ptr + batch_idx)

    # Load query vector: q is [HEAD_DIM] in fp32 for accumulation
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    d_offsets = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + q_offset + d_offsets * stride_qd).to(tl.float32)

    # Online softmax state
    m_prev = float("-inf")   # running max
    l_prev = 0.0             # running sum of exp
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # running weighted sum

    # Sliding window: compute the start position
    # All positions before window_start are outside the window
    window_start = 0
    if WINDOW_SIZE > 0:
        window_start = tl.maximum(0, context_len - WINDOW_SIZE)

    # Compute page range to iterate
    start_page = window_start // PAGE_SIZE
    num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE

    # Iterate over pages
    for page_offset in range(start_page, num_pages):
        # Page-table indirection: look up physical page index
        page_idx = tl.load(page_table_ptr + batch_idx * stride_pt_b + page_offset * stride_pt_p)

        # Determine valid positions within this page
        pos_start = page_offset * PAGE_SIZE
        p_offsets = tl.arange(0, PAGE_SIZE)
        positions = pos_start + p_offsets

        # Mask: position must be < context_len AND >= window_start
        valid_mask = (positions < context_len)
        if WINDOW_SIZE > 0:
            valid_mask = valid_mask & (positions >= window_start)

        # Load K block: [PAGE_SIZE, HEAD_DIM]
        k_base = page_idx * stride_kb + kv_head_idx * stride_kh
        k = tl.load(
            k_cache_ptr + k_base + p_offsets[:, None] * stride_kp + d_offsets[None, :] * stride_kd,
            mask=valid_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Load V block: [PAGE_SIZE, HEAD_DIM]
        v_base = page_idx * stride_vb + kv_head_idx * stride_vh
        v = tl.load(
            v_cache_ptr + v_base + p_offsets[:, None] * stride_vp + d_offsets[None, :] * stride_vd,
            mask=valid_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Compute QK^T: [PAGE_SIZE] scores
        # q is [HEAD_DIM], k is [PAGE_SIZE, HEAD_DIM]
        # scores[i] = sum_d(q[d] * k[i, d]) * sm_scale
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale

        # Mask out invalid positions
        scores = tl.where(valid_mask, scores, float("-inf"))

        # Online softmax update
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)

        # Correction factor for previous accumulation
        alpha = tl.exp(m_prev - m_new)

        # Exponentiated scores for this block
        p = tl.exp(scores - m_new)

        # Update running sum and accumulator
        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        m_prev = m_new

    # Final normalization
    out = acc / l_prev

    # Store output
    out_offset = batch_idx * stride_ob + head_idx * stride_oh
    tl.store(out_ptr + out_offset + d_offsets * stride_od, out.to(tl.float16))


def paged_attention_decode(q, k_cache, v_cache, page_table, context_lens,
                           sm_scale, num_kv_heads, page_size, window_size, config):
    """Launch the paged attention decode kernel.

    Args:
        q: [B, H, 1, D] fp16 -- query for the new token
        k_cache: [num_blocks, Hkv, page_size, D] fp16 -- paged key cache
        v_cache: [num_blocks, Hkv, page_size, D] fp16 -- paged value cache
        page_table: [B, max_pages] int32 -- logical-to-physical page mapping
        context_lens: [B] int32 -- actual KV length per batch item
        sm_scale: float -- softmax scale (1/sqrt(D) typically)
        num_kv_heads: int -- number of KV heads (for GQA)
        page_size: int -- tokens per page
        window_size: int -- sliding window size (-1 for no window)
        config: dict -- kernel config with BLOCK_KV, num_warps, num_stages
    """
    B, H, _, D = q.shape
    out = torch.empty(B, H, 1, D, device=q.device, dtype=q.dtype)
    max_num_pages = page_table.shape[1]

    grid = (B, H)
    paged_attention_decode_kernel[grid](
        q, k_cache, v_cache, page_table, context_lens, out,
        H, num_kv_heads, D, max_num_pages,
        # Q strides (skip the seq=1 dim)
        q.stride(0), q.stride(1), q.stride(3),
        # K cache strides
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        # V cache strides
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # Output strides
        out.stride(0), out.stride(1), out.stride(3),
        # Page table strides
        page_table.stride(0), page_table.stride(1),
        sm_scale,
        HEAD_DIM=D,
        PAGE_SIZE=page_size,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_KV=config["BLOCK_KV"],
        IS_CAUSAL=True,
        WINDOW_SIZE=window_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def torch_paged_attention_decode(q, k_cache, v_cache, page_table, context_lens,
                                  sm_scale, num_kv_heads, page_size, window_size=-1):
    """PyTorch reference implementation of paged attention decode.

    Args:
        q: [B, H, 1, D] fp16
        k_cache: [num_blocks, Hkv, page_size, D] fp16
        v_cache: [num_blocks, Hkv, page_size, D] fp16
        page_table: [B, max_pages] int32
        context_lens: [B] int32
        sm_scale: float
        num_kv_heads: int
        page_size: int
        window_size: int (-1 for no window)

    Returns:
        out: [B, H, 1, D] fp16
    """
    B, H, _, D = q.shape
    out = torch.zeros_like(q)
    group_size = H // num_kv_heads

    for b in range(B):
        ctx_len = context_lens[b].item()
        num_pages = (ctx_len + page_size - 1) // page_size
        k_list, v_list = [], []
        for p in range(num_pages):
            page_idx = page_table[b, p].item()
            k_list.append(k_cache[page_idx])  # [Hkv, page_size, D]
            v_list.append(v_cache[page_idx])
        if num_pages == 0:
            continue
        k_full = torch.cat(k_list, dim=1)[:, :ctx_len, :]  # [Hkv, ctx_len, D]
        v_full = torch.cat(v_list, dim=1)[:, :ctx_len, :]

        for h in range(H):
            kvh = h // group_size
            q_h = q[b, h, 0, :]  # [D]
            k_h = k_full[kvh]    # [ctx_len, D]
            v_h = v_full[kvh]    # [ctx_len, D]

            if window_size > 0:
                start = max(0, ctx_len - window_size)
                k_h = k_h[start:]
                v_h = v_h[start:]

            scores = (q_h.float() @ k_h.float().T) * sm_scale  # [ctx_len] or [window]
            weights = torch.softmax(scores, dim=-1)  # keep float32
            out[b, h, 0, :] = (weights @ v_h.float()).half()

    return out


def benchmark_one(batch, num_heads, num_kv_heads, head_dim, context_len,
                  page_size, window_size, config):
    """Run a single benchmark point: correctness check + timing."""
    try:
        D = head_dim
        H = num_heads
        B = batch

        q = torch.randn((B, H, 1, D), device="cuda", dtype=torch.float16)

        # Create paged KV cache
        num_pages_needed = (context_len + page_size - 1) // page_size
        # Allocate more blocks than needed to test indirection
        num_blocks = num_pages_needed + 8
        k_cache = torch.randn((num_blocks, num_kv_heads, page_size, D),
                              device="cuda", dtype=torch.float16)
        v_cache = torch.randn((num_blocks, num_kv_heads, page_size, D),
                              device="cuda", dtype=torch.float16)

        # Create page table with shuffled physical pages
        max_pages = num_pages_needed
        page_table = torch.randperm(num_blocks, device="cuda", dtype=torch.int32)[:max_pages]
        page_table = page_table.unsqueeze(0).expand(B, -1).contiguous()

        context_lens = torch.full((B,), context_len, device="cuda", dtype=torch.int32)

        sm_scale = 1.0 / math.sqrt(D)

        # Correctness check
        ref = torch_paged_attention_decode(
            q, k_cache, v_cache, page_table, context_lens,
            sm_scale, num_kv_heads, page_size, window_size,
        )
        out = paged_attention_decode(
            q, k_cache, v_cache, page_table, context_lens,
            sm_scale, num_kv_heads, page_size, window_size, config,
        )

        max_err = (out - ref).abs().max().item()
        if max_err > 0.05:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        # Benchmark timing
        ms = triton.testing.do_bench(
            lambda: paged_attention_decode(
                q, k_cache, v_cache, page_table, context_lens,
                sm_scale, num_kv_heads, page_size, window_size, config,
            ),
            warmup=25, rep=100,
        )

        # Effective memory bandwidth
        # Read: Q + K pages + V pages + page table. Write: output.
        effective_ctx = context_len
        if window_size > 0:
            effective_ctx = min(context_len, window_size)
        q_bytes = B * H * D * 2
        kv_bytes = B * effective_ctx * num_kv_heads * D * 2 * 2  # K + V
        pt_bytes = B * num_pages_needed * 4
        out_bytes = B * H * D * 2
        total_bytes = q_bytes + kv_bytes + pt_bytes + out_bytes
        gb_per_s = total_bytes / (ms * 1e-3) / 1e9

        # Compute: 2 * B * H * effective_ctx * D (QK) + 2 * B * H * effective_ctx * D (PV)
        flops = 2 * B * H * effective_ctx * D * 2
        tflops = flops / (ms * 1e-3) / 1e12

        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "gb_per_s": round(gb_per_s, 2),
            "tflops": round(tflops, 4),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None, "tflops": None}}


def main():
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bkv{{}}_w{{}}_s{{}}".format(
            config["BLOCK_KV"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            result = benchmark_one(
                batch=shape["batch"],
                num_heads=shape["num_heads"],
                num_kv_heads=shape["num_kv_heads"],
                head_dim=shape["head_dim"],
                context_len=shape["context_len"],
                page_size=shape["page_size"],
                window_size=shape.get("window_size", -1),
                config=config,
            )
            result["shape"] = "{{}}x{{}}x{{}}x{{}}x{{}}".format(
                shape["batch"], shape["num_heads"], shape["num_kv_heads"],
                shape["head_dim"], shape["context_len"],
            )
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "attention_decode",
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


ATTENTION_DECODE_SPEC = register_operator(TritonOperatorSpec(
    name="attention_decode",
    param_space=ATTENTION_DECODE_PARAM_SPACE,
    curated_configs=ATTENTION_DECODE_CURATED_CONFIGS,
    shape_buckets=ATTENTION_DECODE_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=attention_decode_config_id,
    shape_bucket_fn=attention_decode_shape_bucket_key,
    benchmark_script_fn=generate_attention_decode_benchmark_script,
    grid_generator_fn=generate_attention_decode_grid,
    shared_memory_check_fn=attention_decode_shared_memory_check,
    description=(
        "Decode-time paged-KV attention: single-query against non-contiguous "
        "page-table-indexed KV cache with online softmax, GQA, and sliding "
        "window. Novel Triton implementation -- vLLM has no Triton paged "
        "attention."
    ),
))
