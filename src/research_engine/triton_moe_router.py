"""Parameterized Triton kernel for a fused MoE router (Gemma 4 26B-A4B).

The Gemma 4 26B-A4B MoE forward pass gates each token over 128 experts and
activates the top-8 in parallel with one shared dense MLP. vLLM today runs
the router as four separate kernel launches:

    1. Router projection matmul:    logits = hidden @ router_weight
    2. Softmax over experts:         probs = softmax(logits, dim=-1)
    3. Top-k selection:              topk_w, topk_i = probs.topk(k)
    4. Renormalization:              topk_w /= topk_w.sum(-1, keepdim=True)

We fuse all four into a single Triton program per BLOCK_M tokens. Each
program:

  * Tiled-K matmul over ``hidden_dim`` to materialize an ``[BLOCK_M,
    NUM_EXPERTS]`` logit tile in registers (router_weight at
    ``[hidden_dim=2816, num_experts=128]`` is 720 KB — too big for shared
    memory, so we stream K-tiles of the weight).
  * Numerically stable softmax across the NUM_EXPERTS axis.
  * Iterative max-and-mask top-k (k=8, num_experts=128 → 1024 scalar ops
    per token, negligible next to the 2816 K-dim matmul).
  * Renormalization of the selected weights so they sum to 1.

The shared expert is a SEPARATE dense MLP path in the Gemma 4 architecture
and is NOT handled here — this kernel only returns the top-k gating tuple
for the routed experts.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


MOE_ROUTER_PARAM_SPACE = {
    "BLOCK_M": [16, 32, 64, 128, 256],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3],
}


MOE_ROUTER_CURATED_CONFIGS = [
    {"BLOCK_M": 32, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 256, "num_warps": 8, "num_stages": 2},
]


# Real Gemma 4 26B-A4B router shapes at several prefill / decode token counts.
MOE_ROUTER_SHAPE_BUCKETS = [
    {"name": "gemma4_26b_a4b_router_small", "num_tokens": 1024, "hidden_dim": 2816, "num_experts": 128, "top_k": 8},
    {"name": "gemma4_26b_a4b_router_med", "num_tokens": 4096, "hidden_dim": 2816, "num_experts": 128, "top_k": 8},
    {"name": "gemma4_26b_a4b_router_long", "num_tokens": 8192, "hidden_dim": 2816, "num_experts": 128, "top_k": 8},
    {"name": "gemma4_26b_a4b_router_xlong", "num_tokens": 16384, "hidden_dim": 2816, "num_experts": 128, "top_k": 8},
]


def moe_router_config_id(config: dict[str, int]) -> str:
    return f"bm{config['BLOCK_M']}_w{config['num_warps']}_s{config['num_stages']}"


def moe_router_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a router shape into a Gemma 4 26B-A4B token-count bucket."""
    num_tokens = int(shape.get("num_tokens", 0))
    if num_tokens <= 2048:
        return "gemma4_26b_a4b_router_small"
    if num_tokens <= 6144:
        return "gemma4_26b_a4b_router_med"
    if num_tokens <= 12288:
        return "gemma4_26b_a4b_router_long"
    return "gemma4_26b_a4b_router_xlong"


def moe_router_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — feasibility is learned at runtime.

    A BLOCK_M x NUM_EXPERTS=128 logits tile in fp32 plus a streamed
    K-tile of the router_weight fits comfortably even at BLOCK_M=256
    (256*128*4 = 128 KB logits, <64 KB weight tile), so we always pass.
    """
    return True


def generate_moe_router_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in MOE_ROUTER_CURATED_CONFIGS:
            cid = moe_router_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bm in MOE_ROUTER_PARAM_SPACE["BLOCK_M"]:
        for nw in MOE_ROUTER_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {"BLOCK_M": bm, "num_warps": nw, "num_stages": ns}
                cid = moe_router_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_moe_router_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained Triton MoE router benchmark script."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated fused MoE router benchmark (Gemma 4 26B-A4B)."""

import json
import platform

import torch
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def moe_router_kernel(
    hidden_ptr, router_weight_ptr,
    topk_weights_ptr, topk_indices_ptr,
    num_tokens, hidden_dim, num_experts,
    stride_hidden_m, stride_hidden_k,
    stride_weight_k, stride_weight_n,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused router: logits = hidden @ W; softmax; top-k; renormalize.

    One program handles BLOCK_M tokens. The router weight is streamed in
    BLOCK_K-sized K-tiles; NUM_EXPERTS is the full expert dimension (we
    materialize the full logit row in registers since 128 is small).

    The shared expert is a SEPARATE dense MLP in Gemma 4; this kernel only
    computes the routed top-k gate weights and indices.
    """
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NUM_EXPERTS)
    offs_k = tl.arange(0, BLOCK_K)

    token_mask = offs_m < num_tokens

    # Accumulate logits in fp32: [BLOCK_M, NUM_EXPERTS].
    acc = tl.zeros((BLOCK_M, NUM_EXPERTS), dtype=tl.float32)

    # Tiled K-matmul over hidden_dim.
    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < hidden_dim

        # hidden: [BLOCK_M, BLOCK_K]
        h_ptrs = (
            hidden_ptr
            + offs_m[:, None] * stride_hidden_m
            + k_idx[None, :] * stride_hidden_k
        )
        h_mask = token_mask[:, None] & k_mask[None, :]
        h_tile = tl.load(h_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        # router_weight: [BLOCK_K, NUM_EXPERTS]
        w_ptrs = (
            router_weight_ptr
            + k_idx[:, None] * stride_weight_k
            + offs_n[None, :] * stride_weight_n
        )
        w_mask = k_mask[:, None] & (offs_n[None, :] < num_experts)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(h_tile, w_tile, allow_tf32=True)

    # Mask out-of-range experts to -inf (defensive; NUM_EXPERTS == num_experts
    # in practice) and out-of-range tokens to -inf so they do not disturb
    # any downstream reductions.
    expert_mask = offs_n[None, :] < num_experts
    acc = tl.where(expert_mask, acc, float("-inf"))
    acc = tl.where(token_mask[:, None], acc, float("-inf"))

    # Stable softmax over the NUM_EXPERTS axis.
    max_logit = tl.max(acc, axis=1)
    shifted = acc - max_logit[:, None]
    exp_x = tl.exp(shifted)
    denom = tl.sum(exp_x, axis=1)
    probs = exp_x / denom[:, None]

    # Iterative top-k: k passes of argmax+mask. k=8, NUM_EXPERTS=128 ->
    # 1024 scalar ops per token, negligible vs. the matmul.
    cur = probs
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i in tl.static_range(0, TOP_K):
        idx = tl.argmax(cur, axis=1).to(tl.int32)
        # Gather the max value per row.
        max_val = tl.max(cur, axis=1)
        running_sum += max_val

        # Store raw top-k weight + index at column i.
        out_w_ptr = topk_weights_ptr + offs_m * TOP_K + i
        out_i_ptr = topk_indices_ptr + offs_m * TOP_K + i
        tl.store(out_w_ptr, max_val, mask=token_mask)
        tl.store(out_i_ptr, idx, mask=token_mask)

        # Mask the winning column for the next pass.
        one_hot = (offs_n[None, :] == idx[:, None])
        cur = tl.where(one_hot, float("-inf"), cur)

    # Renormalize the stored top-k weights so each row sums to 1.
    inv_sum = 1.0 / running_sum
    for i in tl.static_range(0, TOP_K):
        w_ptr = topk_weights_ptr + offs_m * TOP_K + i
        w = tl.load(w_ptr, mask=token_mask, other=0.0)
        w = w * inv_sum
        tl.store(w_ptr, w, mask=token_mask)


def moe_router(hidden, router_weight, top_k, config):
    """Launch the fused MoE router kernel.

    hidden:        [num_tokens, hidden_dim] fp16
    router_weight: [hidden_dim, num_experts] fp16
    top_k:         int
    Returns: (topk_weights [T, k] fp32, topk_indices [T, k] int32)
    """
    num_tokens, hidden_dim = hidden.shape
    _, num_experts = router_weight.shape

    topk_weights = torch.empty((num_tokens, top_k), device=hidden.device, dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, top_k), device=hidden.device, dtype=torch.int32)

    NUM_EXPERTS = triton.next_power_of_2(num_experts)
    HIDDEN_DIM = hidden_dim
    BLOCK_M = config["BLOCK_M"]
    BLOCK_K = 64

    grid = (triton.cdiv(num_tokens, BLOCK_M),)
    moe_router_kernel[grid](
        hidden, router_weight,
        topk_weights, topk_indices,
        num_tokens, hidden_dim, num_experts,
        hidden.stride(0), hidden.stride(1),
        router_weight.stride(0), router_weight.stride(1),
        NUM_EXPERTS=NUM_EXPERTS,
        TOP_K=top_k,
        HIDDEN_DIM=HIDDEN_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return topk_weights, topk_indices


def torch_moe_router(hidden, router_weight, top_k):
    """PyTorch reference: matmul -> softmax -> topk -> renormalize."""
    logits = hidden.float() @ router_weight.float()
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_indices = probs.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)


def separated_moe_router(hidden, router_weight, top_k):
    """vLLM-style separated baseline: 4 distinct GPU launches.

    1. matmul
    2. softmax
    3. topk
    4. renormalize divide

    Reporting ``separated_ms / fused_ms`` against this baseline is the
    headline fusion_speedup number.
    """
    logits = hidden @ router_weight                                  # 1: matmul
    probs = torch.softmax(logits, dim=-1)                            # 2: softmax
    topk_weights, topk_indices = probs.topk(top_k, dim=-1)           # 3: topk
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True) # 4: renorm
    return topk_weights, topk_indices


def benchmark_one(num_tokens, hidden_dim, num_experts, top_k, config):
    try:
        torch.manual_seed(0)
        hidden = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.float16) * 0.1
        router_weight = torch.randn((hidden_dim, num_experts), device="cuda", dtype=torch.float16) * 0.1

        ref_w, ref_i = torch_moe_router(hidden, router_weight, top_k)
        out_w, out_i = moe_router(hidden, router_weight, top_k, config)

        # Compare on the SET of chosen experts (order within tied probs may
        # differ slightly, but for random fp16 inputs ties are vanishingly
        # rare). We compare sorted weights + sorted indices.
        ref_i_sorted, _ = ref_i.sort(dim=-1)
        out_i_sorted, _ = out_i.sort(dim=-1)
        idx_match = (ref_i_sorted == out_i_sorted).float().mean().item()

        w_err = (out_w.float() - ref_w.float()).abs().max().item()

        if idx_match < 0.99 or w_err > 0.05:
            return {{
                "correct": False,
                "idx_match": round(idx_match, 4),
                "max_err": round(w_err, 6),
                "ms": None, "gb_per_s": None, "tflops": None,
            }}

        ms = triton.testing.do_bench(
            lambda: moe_router(hidden, router_weight, top_k, config),
            warmup=25, rep=100,
        )
        sep_ms = triton.testing.do_bench(
            lambda: separated_moe_router(hidden, router_weight, top_k),
            warmup=25, rep=100,
        )

        # Dominant FLOPs: the router matmul. 2 * T * H * E MACs.
        flops = 2.0 * num_tokens * hidden_dim * num_experts
        tflops = flops / (ms * 1e-3) / 1e12

        # Bytes: read hidden + router_weight, write topk_w + topk_i.
        hidden_bytes = num_tokens * hidden_dim * 2
        weight_bytes = hidden_dim * num_experts * 2
        out_bytes = num_tokens * top_k * (4 + 4)
        bytes_moved = hidden_bytes + weight_bytes + out_bytes
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9

        fusion_speedup = sep_ms / ms if ms > 0 else 0.0
        return {{
            "correct": True,
            "max_err": round(w_err, 6),
            "idx_match": round(idx_match, 4),
            "ms": round(ms, 4),
            "separated_ms": round(sep_ms, 4),
            "fusion_speedup": round(fusion_speedup, 3),
            "gb_per_s": round(gb_per_s, 2),
            "tflops": round(tflops, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None, "tflops": None}}


def main():
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bm{{}}_w{{}}_s{{}}".format(
            config["BLOCK_M"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            num_tokens = shape["num_tokens"]
            hidden_dim = shape["hidden_dim"]
            num_experts = shape["num_experts"]
            top_k = shape["top_k"]
            result = benchmark_one(num_tokens, hidden_dim, num_experts, top_k, config)
            result["shape"] = f"{{num_tokens}}x{{hidden_dim}}x{{num_experts}}x{{top_k}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "moe_router",
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


MOE_ROUTER_SPEC = register_operator(TritonOperatorSpec(
    name="moe_router",
    param_space=MOE_ROUTER_PARAM_SPACE,
    curated_configs=MOE_ROUTER_CURATED_CONFIGS,
    shape_buckets=MOE_ROUTER_SHAPE_BUCKETS,
    metric_name="tflops",
    config_id_fn=moe_router_config_id,
    shape_bucket_fn=moe_router_shape_bucket_key,
    benchmark_script_fn=generate_moe_router_benchmark_script,
    grid_generator_fn=generate_moe_router_grid,
    shared_memory_check_fn=moe_router_shared_memory_check,
    description=(
        "Fused Gemma 4 26B-A4B MoE router: hidden @ W -> softmax -> top-8 -> "
        "renormalize, in ONE Triton kernel. Novel vs vLLM which launches 4 "
        "separate kernels per router. Shared expert is a separate dense path."
    ),
))
