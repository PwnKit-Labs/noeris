"""Parameterized Triton kernel for grouped/segmented GEMM (MoE expert FFN w1).

This is the second half of the Gemma 4 26B-A4B MoE path. Given a router
output (top-k expert assignments per token, see ``triton_moe_router``),
this kernel computes ``hidden @ expert_weights[e].T`` for *every*
(token, expert) pair in a single launch — replacing what would otherwise
be ~num_experts (128) separate matmul launches.

The "trick" is borrowed from vLLM's ``fused_moe_kernel``
(``vllm/model_executor/layers/fused_moe/fused_moe.py``, lines 311-572):
instead of materializing a permuted A matrix, we precompute
``sorted_token_ids`` such that ``sorted_token_ids[i] // top_k`` is the
original A-row index, and tokens going to the same expert end up at
contiguous positions. The kernel can then dispatch one expert per
``BLOCK_SIZE_M`` block via ``expert_ids[pid_m]`` and load A rows by
indexed gather. No permuted A matrix is materialized.

This module ships **only the w1 (gate+up) pass** — ``MUL_ROUTED_WEIGHT``
is wired but always set to ``False`` here. The w2 (down) pass would set
it to ``True`` so the per-token expert weights fold into the output
before the un-sort scatter-add.

References:
- vLLM ``fused_moe_kernel``: signature lines 311-365, body 395-572.
- vLLM ``get_default_config``: line 1223 — block-size table for H100/A100.
- Gemma 4 26B-A4B: 128 experts, top_k=8, hidden_dim=2816, ffn_dim=2112,
  so 2*ffn_dim = 4224.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


GROUPED_GEMM_PARAM_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128],
    "GROUP_SIZE_M": [1, 8, 16],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}


# Curated configs sourced from vLLM ``get_default_config`` for bf16 dense
# (non-quantized) FusedMoE on H100/A100 across the M ranges that matter
# for Gemma 4 26B-A4B (1k–16k tokens with top_k=8).
GROUPED_GEMM_CURATED_CONFIGS = [
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
]


# Real Gemma 4 26B-A4B w1 grouped-GEMM shapes (verified from HF config.json:
# 128 experts, top_k=8, hidden_dim=2816, ffn_dim=2112 → 2*ffn_dim=4224).
GROUPED_GEMM_SHAPE_BUCKETS = [
    {
        "name": "gemma4_26b_a4b_w1_small",
        "num_tokens": 1024, "num_experts": 128, "top_k": 8,
        "hidden_dim": 2816, "intermediate_dim": 4224,
    },
    {
        "name": "gemma4_26b_a4b_w1_med",
        "num_tokens": 4096, "num_experts": 128, "top_k": 8,
        "hidden_dim": 2816, "intermediate_dim": 4224,
    },
    {
        "name": "gemma4_26b_a4b_w1_long",
        "num_tokens": 8192, "num_experts": 128, "top_k": 8,
        "hidden_dim": 2816, "intermediate_dim": 4224,
    },
    {
        "name": "gemma4_26b_a4b_w1_xlong",
        "num_tokens": 16384, "num_experts": 128, "top_k": 8,
        "hidden_dim": 2816, "intermediate_dim": 4224,
    },
]


def grouped_gemm_config_id(config: dict[str, int]) -> str:
    return (
        f"bm{config['BLOCK_SIZE_M']}_bn{config['BLOCK_SIZE_N']}_"
        f"bk{config['BLOCK_SIZE_K']}_gm{config['GROUP_SIZE_M']}_"
        f"w{config['num_warps']}_s{config['num_stages']}"
    )


def grouped_gemm_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a grouped-GEMM shape into a Gemma 4 26B-A4B w1 bucket.

    Discriminator is the (rounded) num_tokens dimension. All Gemma 4
    26B-A4B w1 layers share num_experts=128, top_k=8, hidden=2816,
    intermediate=4224, so num_tokens is the only varying axis.
    """
    n = shape.get("num_tokens", 0)
    if n <= 2048:
        return "gemma4_26b_a4b_w1_small"
    if n <= 6144:
        return "gemma4_26b_a4b_w1_med"
    if n <= 12288:
        return "gemma4_26b_a4b_w1_long"
    return "gemma4_26b_a4b_w1_xlong"


def grouped_gemm_shared_memory_check(config: dict[str, int]) -> bool:
    """Permissive feasibility check — actual feasibility is learned at runtime.

    Triton will reject configs whose smem usage exceeds the device limit
    when it tries to compile. The cost model + bandit handle filtering.
    """
    # Cheap sanity bound: a single A tile is BLOCK_M * BLOCK_K fp16, a
    # single B tile is BLOCK_K * BLOCK_N fp16, multiplied by num_stages
    # for software pipelining. Anything ≥ ~228 KB is going to fail on
    # H100 (228 KB smem); reject obviously bad shapes early.
    bm = config["BLOCK_SIZE_M"]
    bn = config["BLOCK_SIZE_N"]
    bk = config["BLOCK_SIZE_K"]
    stages = config["num_stages"]
    bytes_per_element = 2  # fp16
    tile_bytes = (bm * bk + bk * bn) * bytes_per_element * stages
    return tile_bytes <= 228 * 1024


def generate_grouped_gemm_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in GROUPED_GEMM_CURATED_CONFIGS:
            cid = grouped_gemm_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bm in GROUPED_GEMM_PARAM_SPACE["BLOCK_SIZE_M"]:
        for bn in GROUPED_GEMM_PARAM_SPACE["BLOCK_SIZE_N"]:
            for bk in GROUPED_GEMM_PARAM_SPACE["BLOCK_SIZE_K"]:
                for gm in [1, 8]:
                    for nw in [4, 8]:
                        for ns in [2, 3]:
                            config = {
                                "BLOCK_SIZE_M": bm,
                                "BLOCK_SIZE_N": bn,
                                "BLOCK_SIZE_K": bk,
                                "GROUP_SIZE_M": gm,
                                "num_warps": nw,
                                "num_stages": ns,
                            }
                            cid = grouped_gemm_config_id(config)
                            if cid in seen:
                                continue
                            if not grouped_gemm_shared_memory_check(config):
                                continue
                            seen.add(cid)
                            configs.append(config)
                            if len(configs) >= max_configs:
                                return configs
    return configs


def generate_grouped_gemm_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained Triton grouped-GEMM benchmark script."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated grouped GEMM benchmark (Gemma 4 26B-A4B MoE w1)."""

import json
import platform

import torch
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def grouped_gemm_kernel(
    # Inputs
    a_ptr,                       # hidden_states [num_tokens, K]
    b_ptr,                       # expert_weights [num_experts, N, K]
    c_ptr,                       # output [num_tokens * top_k, N]
    # Dispatch
    sorted_token_ids_ptr,        # [EM] int32; sorted_token_ids[i] // TOP_K = real row
    expert_ids_ptr,              # [EM // BLOCK_M] int32; expert per M-block
    num_tokens_post_padded_ptr,  # scalar int32
    topk_weights_ptr,            # [num_tokens * top_k] fp16 (only used if MUL_ROUTED_WEIGHT)
    # Shapes (runtime)
    num_valid_tokens, K, N,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    # constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Fused grouped GEMM for MoE expert FFN.

    Sort-free A dispatch: sorted_token_ids[i] is (orig_token_idx * TOP_K + k),
    so original_row = sorted_token_ids[m_offs] // TOP_K. All tokens routed
    to the same expert are contiguous in sorted_token_ids, and one expert
    is dispatched per BLOCK_M block via expert_ids_ptr[pid_m].

    For Noeris #37 (w1, gate+up), MUL_ROUTED_WEIGHT is always False — the
    multiplication by topk_weights happens in the w2 pass, not here.
    """
    # --- Block id with L2-friendly group swizzling ---
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(tl.load(num_tokens_post_padded_ptr), BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Bounds check on M (block beyond padded len does nothing).
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # --- Load sorted token ids for this M-block ---
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # Original A-matrix rows recovered via integer-divide by TOP_K.
    # This is the sort-free A dispatch trick from vLLM fused_moe_kernel.
    a_row = offs_token // TOP_K

    # --- Per-tile expert lookup ---
    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts < 0:
        # Expert not present on this rank — write zeros and exit early.
        offs_cn_skip = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs_skip = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn_skip[None, :]
        c_mask_skip = token_mask[:, None] & (offs_cn_skip[None, :] < N)
        tl.store(c_ptrs_skip, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask_skip)
        return

    # --- Pointer setup ---
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (a_row[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # --- K loop ---
    for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_idx * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        # b is loaded as [N, K]; tl.dot wants [M, K] x [K, N]. Transpose b.
        accumulator += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    out = accumulator.to(compute_type)

    # --- Store ---
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


def moe_align_block_size_torch(topk_indices, block_size_m, num_experts):
    """Pure PyTorch ``moe_align_block_size`` — produces sorted_token_ids,
    expert_ids, num_tokens_post_padded.

    sorted_token_ids[i] is (orig_token_idx * top_k + k) for the i-th slot
    in the per-expert-padded layout. expert_ids[block] tells the kernel
    which expert each BLOCK_M-sized chunk handles. Padded slots are filled
    with ``num_tokens * top_k`` so that the in-kernel ``token_mask`` based
    on ``num_valid_tokens`` masks them out automatically.
    """
    num_tokens, top_k = topk_indices.shape
    flat_ids = topk_indices.flatten()  # length num_tokens * top_k
    num_valid = flat_ids.numel()
    # Per-expert assigned slot indices, in original order.
    sorted_blocks = []
    expert_ids_list = []
    for e in range(num_experts):
        mask = (flat_ids == e)
        slot_ids = torch.nonzero(mask, as_tuple=False).flatten()
        n = slot_ids.numel()
        if n == 0:
            continue
        # Pad up to multiple of block_size_m.
        padded_n = ((n + block_size_m - 1) // block_size_m) * block_size_m
        pad_amt = padded_n - n
        if pad_amt > 0:
            pad = torch.full((pad_amt,), num_valid, dtype=slot_ids.dtype, device=slot_ids.device)
            slot_ids = torch.cat([slot_ids, pad])
        sorted_blocks.append(slot_ids)
        expert_ids_list.extend([e] * (padded_n // block_size_m))

    if not sorted_blocks:
        sorted_token_ids = torch.full((block_size_m,), num_valid, dtype=torch.int32, device=topk_indices.device)
        expert_ids = torch.full((1,), -1, dtype=torch.int32, device=topk_indices.device)
        return sorted_token_ids, expert_ids, torch.tensor(block_size_m, dtype=torch.int32, device=topk_indices.device)

    sorted_token_ids = torch.cat(sorted_blocks).to(torch.int32)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=topk_indices.device)
    num_tokens_post_padded = torch.tensor(sorted_token_ids.numel(), dtype=torch.int32, device=topk_indices.device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def grouped_gemm(hidden, expert_weights, topk_indices, topk_weights, config):
    """Launch the grouped GEMM kernel for MoE expert w1 (gate+up).

    hidden: [num_tokens, hidden_dim] fp16
    expert_weights: [num_experts, 2*ffn_dim, hidden_dim] fp16
    topk_indices: [num_tokens, top_k] int32
    topk_weights: [num_tokens, top_k] fp16  (unused in w1, present for symmetry)
    Returns: out [num_tokens * top_k, 2*ffn_dim] fp16
    """
    num_tokens, top_k = topk_indices.shape
    num_experts, N, K = expert_weights.shape
    BLOCK_M = config["BLOCK_SIZE_M"]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_torch(
        topk_indices, BLOCK_M, num_experts,
    )

    out = torch.zeros((num_tokens * top_k, N), device=hidden.device, dtype=torch.float16)
    num_valid_tokens = num_tokens * top_k

    flat_topk_weights = topk_weights.flatten().contiguous().to(torch.float16)

    EM = sorted_token_ids.numel()
    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)

    grouped_gemm_kernel[grid](
        hidden, expert_weights, out,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        flat_topk_weights,
        num_valid_tokens, K, N,
        hidden.stride(0), hidden.stride(1),
        expert_weights.stride(0), expert_weights.stride(1), expert_weights.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=False,
        TOP_K=top_k,
        compute_type=tl.float16,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def torch_grouped_gemm(hidden, expert_weights, topk_indices):
    """PyTorch reference: per-(token, k) matmul, slow but obviously correct."""
    num_tokens, top_k = topk_indices.shape
    num_experts, intermediate, hidden_dim = expert_weights.shape
    out = torch.zeros(num_tokens, top_k, intermediate, device=hidden.device, dtype=torch.float16)
    for t in range(num_tokens):
        for k in range(top_k):
            e = topk_indices[t, k].item()
            out[t, k] = (hidden[t].float() @ expert_weights[e].float().T).to(torch.float16)
    return out


def separated_grouped_gemm(hidden, expert_weights, topk_indices):
    """Naive separated baseline: one torch.matmul per expert.

    This is what the un-fused MoE forward pass would do — N_experts (128
    for Gemma 4 26B-A4B) separate GEMM launches, each with a small
    assigned-token count, pathologically launch-overhead-bound.

    fusion_speedup = separated_ms / fused_ms is the headline number.
    """
    num_tokens, top_k = topk_indices.shape
    num_experts, intermediate, hidden_dim = expert_weights.shape
    out = torch.zeros(num_tokens, top_k, intermediate, device=hidden.device, dtype=torch.float16)
    for e in range(num_experts):
        mask = (topk_indices == e)
        if mask.sum().item() == 0:
            continue
        token_idx, k_idx = mask.nonzero(as_tuple=True)
        expert_in = hidden[token_idx]
        expert_out = expert_in @ expert_weights[e].T
        out[token_idx, k_idx] = expert_out.to(torch.float16)
    return out


def benchmark_one(num_tokens, num_experts, top_k, hidden_dim, intermediate_dim, config):
    try:
        torch.manual_seed(0)
        hidden = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.float16) * 0.1
        expert_weights = torch.randn((num_experts, intermediate_dim, hidden_dim), device="cuda", dtype=torch.float16) * 0.05
        topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda", dtype=torch.int32)
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float16).float(), dim=-1).to(torch.float16)

        # Correctness on a tiny slice (the full reference is O(num_tokens * top_k * matmul) — too slow).
        ref_n = min(num_tokens, 32)
        ref_out = torch_grouped_gemm(hidden[:ref_n], expert_weights, topk_indices[:ref_n])
        fused_out = grouped_gemm(hidden[:ref_n], expert_weights, topk_indices[:ref_n], topk_weights[:ref_n], config)
        # fused_out is [ref_n * top_k, intermediate]; ref_out is [ref_n, top_k, intermediate]
        fused_reshaped = fused_out.view(ref_n, top_k, intermediate_dim)
        max_err = (fused_reshaped.float() - ref_out.float()).abs().max().item()
        if max_err > 0.5:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        ms = triton.testing.do_bench(
            lambda: grouped_gemm(hidden, expert_weights, topk_indices, topk_weights, config),
            warmup=10, rep=50,
        )
        sep_ms = triton.testing.do_bench(
            lambda: separated_grouped_gemm(hidden, expert_weights, topk_indices),
            warmup=5, rep=20,
        )
        # FLOPs: each (token, k) pair does a (hidden_dim x intermediate_dim) matmul.
        flops = 2.0 * num_tokens * top_k * hidden_dim * intermediate_dim
        tflops = flops / (ms * 1e-3) / 1e12
        # Bytes: read hidden once per (token, k), B-tile (small), write output.
        bytes_in = num_tokens * hidden_dim * 2 + num_experts * intermediate_dim * hidden_dim * 2
        bytes_out = num_tokens * top_k * intermediate_dim * 2
        gb_per_s = (bytes_in + bytes_out) / (ms * 1e-3) / 1e9
        fusion_speedup = sep_ms / ms if ms > 0 else 0.0
        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "separated_ms": round(sep_ms, 4),
            "fusion_speedup": round(fusion_speedup, 3),
            "tflops": round(tflops, 2),
            "gb_per_s": round(gb_per_s, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "tflops": None, "gb_per_s": None}}


def main():
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bm{{}}_bn{{}}_bk{{}}_gm{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"],
            config["GROUP_SIZE_M"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            result = benchmark_one(
                shape["num_tokens"], shape["num_experts"], shape["top_k"],
                shape["hidden_dim"], shape["intermediate_dim"], config,
            )
            result["shape"] = "{{}}t_{{}}e_k{{}}_{{}}h_{{}}i".format(
                shape["num_tokens"], shape["num_experts"], shape["top_k"],
                shape["hidden_dim"], shape["intermediate_dim"],
            )
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "grouped_gemm",
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


GROUPED_GEMM_SPEC = register_operator(TritonOperatorSpec(
    name="grouped_gemm",
    param_space=GROUPED_GEMM_PARAM_SPACE,
    curated_configs=GROUPED_GEMM_CURATED_CONFIGS,
    shape_buckets=GROUPED_GEMM_SHAPE_BUCKETS,
    metric_name="tflops",
    config_id_fn=grouped_gemm_config_id,
    shape_bucket_fn=grouped_gemm_shape_bucket_key,
    benchmark_script_fn=generate_grouped_gemm_benchmark_script,
    grid_generator_fn=generate_grouped_gemm_grid,
    shared_memory_check_fn=grouped_gemm_shared_memory_check,
    description=(
        "Grouped/segmented GEMM for MoE expert FFN w1 (gate+up). Replaces "
        "~num_experts (128 for Gemma 4 26B-A4B) separate matmul launches "
        "with one fused launch using vLLM's sort-free A-dispatch trick "
        "(sorted_token_ids // TOP_K) and per-tile expert selection."
    ),
))
