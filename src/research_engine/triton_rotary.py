"""Parameterized Triton Rotary Position Embedding (RoPE) kernel.

Rotary embeddings rotate pairs of elements in each head's embedding by
position-dependent angles. Used in LLaMA, Mistral, GPT-NeoX, and most
modern LLMs.

Implementation: split-pair variant. Given ``x`` of shape
``(batch, seq, heads, head_dim)`` and precomputed ``cos`` / ``sin``
tensors of shape ``(seq, head_dim // 2)``, output:

    x_even' = x_even * cos - x_odd * sin
    x_odd'  = x_even * sin + x_odd * cos

The kernel processes one (batch, seq, head) position at a time, with
BLOCK_SIZE covering the head_dim pairs in parallel.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


ROTARY_PARAM_SPACE = {
    "BLOCK_SIZE": [16, 32, 64, 128, 256],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
}


ROTARY_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 32, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 64, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 64, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SIZE": 32, "num_warps": 1, "num_stages": 1},
    {"BLOCK_SIZE": 128, "num_warps": 8, "num_stages": 2},
]


# (batch, seq, heads, head_dim)
ROTARY_SHAPE_BUCKETS = [
    {"name": "llama7b_short", "batch": 1, "seq": 512, "heads": 32, "head_dim": 128},
    {"name": "llama7b_med", "batch": 1, "seq": 2048, "heads": 32, "head_dim": 128},
    {"name": "llama7b_long", "batch": 1, "seq": 4096, "heads": 32, "head_dim": 128},
    {"name": "mistral_long", "batch": 1, "seq": 8192, "heads": 32, "head_dim": 128},
    {"name": "gpt_neox", "batch": 2, "seq": 2048, "heads": 16, "head_dim": 64},
    {"name": "gqa_small", "batch": 2, "seq": 1024, "heads": 8, "head_dim": 128},
    # Gemma 4 family: head_dim=256 (vs LLaMA's 128); BLOCK_SIZE must cover 128 pairs.
    # E2B/E4B use 8 heads; 26B MoE uses 16 heads.
    {"name": "gemma4_2b_rope", "batch": 1, "seq": 4096, "heads": 8, "head_dim": 256},
    {"name": "gemma4_26b_rope", "batch": 1, "seq": 4096, "heads": 16, "head_dim": 256},
]


def rotary_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def rotary_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a RoPE shape into a bucket.

    Gemma 4 uses head_dim=256 (LLaMA uses 128), which is the discriminating
    feature for the gemma4_*_rope buckets.  Within Gemma, heads=8 maps to the
    E2B/E4B variants and heads=16 maps to the 26B MoE.
    """
    seq = shape.get("seq", 0)
    hd = shape.get("head_dim", 0)
    heads = shape.get("heads", 0)
    # Gemma 4: head_dim=256 distinguishes it from all other tracked models.
    if hd >= 256:
        return "gemma4_26b_rope" if heads >= 16 else "gemma4_2b_rope"
    if seq >= 8192:
        return "mistral_long"
    if seq >= 4096:
        return "llama7b_long"
    if seq >= 2048:
        if hd <= 64:
            return "gpt_neox"
        return "llama7b_med"
    if hd >= 128 and heads >= 16:
        return "llama7b_short"
    return "gqa_small"


def rotary_shared_memory_check(config: dict[str, int]) -> bool:
    bs = config.get("BLOCK_SIZE", 0)
    ns = config.get("num_stages", 1)
    # Each block loads 2*BLOCK_SIZE fp16 (x_even, x_odd) plus cos/sin floats
    shmem = bs * 2 * 2 * ns + bs * 2 * 4 * ns + 1024
    return shmem <= 192_000


def generate_rotary_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in ROTARY_CURATED_CONFIGS:
            cid = rotary_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in ROTARY_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in ROTARY_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                if not rotary_shared_memory_check(config):
                    continue
                cid = rotary_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_rotary_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton Rotary Embedding benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def rotary_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    x_stride_b, x_stride_s, x_stride_h, x_stride_d,
    out_stride_b, out_stride_s, out_stride_h, out_stride_d,
    cos_stride_s,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary embedding to one (batch, seq, head) position.

    Rotates pairs: x[..., 2i], x[..., 2i+1] -> rotated pair.
    grid = (batch * seq * heads,)
    """
    pid = tl.program_id(0)
    # Recover (b, s, h) from pid
    # Assumes x is contiguous in (b, s, h, d) order
    total_heads = seq_len * tl.num_programs(0) // seq_len  # placeholder
    # Simpler: launch a 3D grid below. For now, use 1D and compute offsets from pid.
    # We use pid as a flat index into (batch * seq * heads).
    half = head_dim // 2
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < half

    # x row pointer: pid indexes into the flattened (b*s*h) dimension.
    # Each position has head_dim elements, with even/odd pairs.
    x_base = x_ptr + pid * head_dim
    out_base = out_ptr + pid * head_dim

    # Compute which (s) row this pid corresponds to for cos/sin lookup.
    # We pass cos/sin as (seq_len, half) row-major, so we need pid % (batch*heads) / batch
    # Simpler: caller passes s_idx via another kernel arg. For this kernel we assume
    # the launcher computes s_idx = (pid // heads) % seq_len.

    x_even = tl.load(x_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

    # cos/sin lookup — use a simpler indexing: we require the launcher
    # to compute s_idx and pass it via cos_ptr_offset on the host side.
    # For simplicity, cos/sin for this (batch,seq,head) row live at cos_ptr.
    c = tl.load(cos_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c

    tl.store(out_base + 2 * offs, out_even.to(tl.float16), mask=mask)
    tl.store(out_base + 2 * offs + 1, out_odd.to(tl.float16), mask=mask)


def apply_rotary(x, cos, sin, config):
    """Reference: x is (B, S, H, D), cos/sin is (S, D//2)."""
    B, S, H, D = x.shape
    out = torch.empty_like(x)

    # Launch one program per (batch, seq, head) position, but use a 3D grid
    # and pass the correct cos/sin row via pointer arithmetic.
    for b in range(B):
        for s in range(S):
            cos_row = cos[s]  # (D//2,)
            sin_row = sin[s]
            for h in range(H):
                x_row = x[b, s, h]  # (D,)
                out_row = out[b, s, h]
                grid = (1,)
                rotary_kernel[grid](
                    x_row.contiguous(), cos_row.contiguous(), sin_row.contiguous(), out_row.contiguous(),
                    0, 0, 0, 1,
                    0, 0, 0, 1,
                    D // 2,
                    S, D,
                    BLOCK_SIZE=config["BLOCK_SIZE"],
                    num_warps=config["num_warps"],
                    num_stages=config["num_stages"],
                )
    return out


def apply_rotary_flat(x, cos, sin, config):
    """Flat launcher: one program per (b*s*h) position."""
    B, S, H, D = x.shape
    out = torch.empty_like(x)
    # Flatten to (B*S*H, D)
    x_flat = x.reshape(B * S * H, D).contiguous()
    out_flat = out.reshape(B * S * H, D).contiguous()

    # We need per-position cos/sin which depends only on seq index.
    # Build a per-position cos/sin table of shape (B*S*H, D//2).
    cos_expanded = cos.unsqueeze(0).unsqueeze(2).expand(B, S, H, D // 2).reshape(-1, D // 2).contiguous()
    sin_expanded = sin.unsqueeze(0).unsqueeze(2).expand(B, S, H, D // 2).reshape(-1, D // 2).contiguous()

    grid = (B * S * H,)
    rotary_kernel[grid](
        x_flat, cos_expanded, sin_expanded, out_flat,
        x_flat.stride(0), 0, 0, 1,
        out_flat.stride(0), 0, 0, 1,
        cos_expanded.stride(0),
        S, D,
        BLOCK_SIZE=max(config["BLOCK_SIZE"], triton.next_power_of_2(D // 2)),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def torch_rotary(x, cos, sin):
    """PyTorch reference implementation."""
    B, S, H, D = x.shape
    # x: (B, S, H, D) with pairs (even, odd)
    x_even = x[..., 0::2]  # (B, S, H, D/2)
    x_odd = x[..., 1::2]
    c = cos[None, :, None, :]  # (1, S, 1, D/2)
    s = sin[None, :, None, :]
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c
    out = torch.stack([out_even, out_odd], dim=-1).reshape(B, S, H, D)
    return out.to(torch.float16)


def benchmark_one(batch, seq, heads, head_dim, config):
    try:
        x = torch.randn((batch, seq, heads, head_dim), device="cuda", dtype=torch.float16)
        cos = torch.randn((seq, head_dim // 2), device="cuda", dtype=torch.float16)
        sin = torch.randn((seq, head_dim // 2), device="cuda", dtype=torch.float16)

        ref = torch_rotary(x, cos, sin)
        out = apply_rotary_flat(x, cos, sin, config)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        ms = triton.testing.do_bench(
            lambda: apply_rotary_flat(x, cos, sin, config),
            warmup=25, rep=100,
        )
        bytes_moved = 2 * batch * seq * heads * head_dim * 2  # read x + write out
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
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bs{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            batch = shape["batch"]
            seq = shape["seq"]
            heads = shape["heads"]
            head_dim = shape["head_dim"]
            result = benchmark_one(batch, seq, heads, head_dim, config)
            result["shape"] = f"{{batch}}x{{seq}}x{{heads}}x{{head_dim}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "rotary",
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


ROTARY_SPEC = register_operator(TritonOperatorSpec(
    name="rotary",
    param_space=ROTARY_PARAM_SPACE,
    curated_configs=ROTARY_CURATED_CONFIGS,
    shape_buckets=ROTARY_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=rotary_config_id,
    shape_bucket_fn=rotary_shape_bucket_key,
    benchmark_script_fn=generate_rotary_benchmark_script,
    grid_generator_fn=generate_rotary_grid,
    shared_memory_check_fn=rotary_shared_memory_check,
    description="Rotary position embedding with split-pair variant used in LLaMA/Mistral/GPT-NeoX.",
))
