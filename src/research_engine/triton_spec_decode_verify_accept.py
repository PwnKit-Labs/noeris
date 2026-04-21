"""Triton fused verify+accept kernel for speculative decoding.

Fuses three steps over draft tokens for each batch row:
1) token equality check (target vs draft)
2) first mismatch index (accept length)
3) accepted-prefix mask write

Argmax over logits is intentionally left outside this first iteration to keep
register pressure and kernel complexity low.
"""

from __future__ import annotations

import json
from typing import Any


VERIFY_ACCEPT_PARAM_SPACE = {
    "BLOCK_SIZE": [32, 64, 128, 256, 512],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
}

VERIFY_ACCEPT_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 32, "num_warps": 1, "num_stages": 1},
    {"BLOCK_SIZE": 64, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 2},
]

VERIFY_ACCEPT_SHAPE_BUCKETS = [
    {"name": "draft4_vocab32k", "batch": 1, "draft_len": 4, "vocab": 32000},
    {"name": "draft8_vocab32k", "batch": 1, "draft_len": 8, "vocab": 32000},
    {"name": "draft16_vocab32k", "batch": 1, "draft_len": 16, "vocab": 32000},
    {"name": "draft16_vocab128k", "batch": 1, "draft_len": 16, "vocab": 128000},
    {"name": "draft32_vocab128k", "batch": 1, "draft_len": 32, "vocab": 128000},
]


def verify_accept_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def verify_accept_shape_bucket_key(shape: dict[str, int]) -> str:
    name = shape.get("name", "")
    if name:
        return name
    draft_len = int(shape.get("draft_len", 0))
    vocab = int(shape.get("vocab", 0))
    if draft_len <= 4 and vocab <= 32000:
        return "draft4_vocab32k"
    if draft_len <= 8 and vocab <= 32000:
        return "draft8_vocab32k"
    if draft_len <= 16 and vocab <= 32000:
        return "draft16_vocab32k"
    if draft_len <= 16:
        return "draft16_vocab128k"
    return "draft32_vocab128k"


_triton_available = False
_verify_accept_kernel = None


def _ensure_triton_kernel() -> None:
    global _triton_available, _verify_accept_kernel
    if _verify_accept_kernel is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _kernel(
            target_ptr,
            draft_ptr,
            accept_len_ptr,
            prefix_mask_ptr,
            draft_len,
            stride_tokens,
            stride_mask,
            BLOCK_SIZE: tl.constexpr,
        ):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < draft_len

            t = tl.load(target_ptr + row * stride_tokens + offs, mask=mask, other=0)
            d = tl.load(draft_ptr + row * stride_tokens + offs, mask=mask, other=-1)

            mismatch = (t != d) & mask
            idx = tl.where(mismatch, offs, draft_len)
            first_mismatch = tl.min(idx, axis=0)

            accept_len = first_mismatch.to(tl.int32)
            tl.store(accept_len_ptr + row, accept_len)

            prefix = offs < accept_len
            tl.store(prefix_mask_ptr + row * stride_mask + offs, prefix, mask=mask)

        _verify_accept_kernel = _kernel
        _triton_available = True
    except ImportError:
        _triton_available = False


def verify_accept_reference(target_tokens, draft_tokens):
    """Reference implementation using PyTorch ops."""
    import torch

    if target_tokens.shape != draft_tokens.shape:
        raise ValueError(
            f"Shape mismatch: target_tokens={target_tokens.shape}, draft_tokens={draft_tokens.shape}"
        )
    match = target_tokens == draft_tokens
    mismatch = ~match
    first_mismatch = torch.argmax(mismatch.to(torch.int32), dim=-1)
    all_match = match.all(dim=-1)
    accept_len = torch.where(all_match, torch.full_like(first_mismatch, match.shape[1]), first_mismatch)
    idx = torch.arange(match.shape[1], device=match.device).unsqueeze(0)
    prefix_mask = idx < accept_len.unsqueeze(-1)
    return accept_len.to(torch.int32), prefix_mask


def verify_accept_fused(target_tokens, draft_tokens, config=None):
    """Fused Triton verify+accept over token IDs.

    Args:
        target_tokens: [batch, draft_len] int tensor.
        draft_tokens: [batch, draft_len] int tensor.
        config: optional kernel config dict.

    Returns:
        accept_len: [batch] int32
        prefix_mask: [batch, draft_len] bool
    """
    import torch
    import triton

    _ensure_triton_kernel()
    if not _triton_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = VERIFY_ACCEPT_CURATED_CONFIGS[0]

    if target_tokens.shape != draft_tokens.shape:
        raise ValueError(
            f"Shape mismatch: target_tokens={target_tokens.shape}, draft_tokens={draft_tokens.shape}"
        )
    if not target_tokens.is_cuda or not draft_tokens.is_cuda:
        raise ValueError("verify_accept_fused requires CUDA tensors")

    bsz, draft_len = target_tokens.shape
    target_i32 = target_tokens.to(torch.int32).contiguous()
    draft_i32 = draft_tokens.to(torch.int32).contiguous()

    accept_len = torch.empty((bsz,), device=target_tokens.device, dtype=torch.int32)
    prefix_mask_u8 = torch.empty((bsz, draft_len), device=target_tokens.device, dtype=torch.uint8)

    block_size = max(int(config["BLOCK_SIZE"]), int(triton.next_power_of_2(draft_len)))
    _verify_accept_kernel[(bsz,)](
        target_i32,
        draft_i32,
        accept_len,
        prefix_mask_u8,
        draft_len,
        target_i32.stride(0),
        prefix_mask_u8.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    return accept_len, prefix_mask_u8.to(torch.bool)


def generate_verify_accept_grid(*, include_curated: bool = True, max_configs: int = 120) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for c in VERIFY_ACCEPT_CURATED_CONFIGS:
            cid = verify_accept_config_id(c)
            if cid not in seen:
                seen.add(cid)
                configs.append(c)

    for bs in VERIFY_ACCEPT_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in VERIFY_ACCEPT_PARAM_SPACE["num_warps"]:
            for ns in VERIFY_ACCEPT_PARAM_SPACE["num_stages"]:
                c = {"BLOCK_SIZE": bs, "num_warps": nw, "num_stages": ns}
                cid = verify_accept_config_id(c)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(c)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_verify_accept_benchmark_script(configs: list[dict[str, int]], shapes: list[dict[str, Any]]) -> str:
    """Generate a self-contained benchmark script for fused verify+accept."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated speculative decode verify+accept benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def verify_accept_kernel(
    target_ptr,
    draft_ptr,
    accept_len_ptr,
    prefix_mask_ptr,
    draft_len,
    stride_tokens,
    stride_mask,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < draft_len
    t = tl.load(target_ptr + row * stride_tokens + offs, mask=mask, other=0)
    d = tl.load(draft_ptr + row * stride_tokens + offs, mask=mask, other=-1)
    mismatch = (t != d) & mask
    idx = tl.where(mismatch, offs, draft_len)
    first_mismatch = tl.min(idx, axis=0)
    accept_len = first_mismatch.to(tl.int32)
    tl.store(accept_len_ptr + row, accept_len)
    prefix = offs < accept_len
    tl.store(prefix_mask_ptr + row * stride_mask + offs, prefix, mask=mask)


def verify_accept_fused(target_tokens, draft_tokens, config):
    bsz, draft_len = target_tokens.shape
    target_i32 = target_tokens.to(torch.int32).contiguous()
    draft_i32 = draft_tokens.to(torch.int32).contiguous()
    accept_len = torch.empty((bsz,), device=target_tokens.device, dtype=torch.int32)
    prefix_mask_u8 = torch.empty((bsz, draft_len), device=target_tokens.device, dtype=torch.uint8)
    block_size = max(config["BLOCK_SIZE"], triton.next_power_of_2(draft_len))
    verify_accept_kernel[(bsz,)](
        target_i32,
        draft_i32,
        accept_len,
        prefix_mask_u8,
        draft_len,
        target_i32.stride(0),
        prefix_mask_u8.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return accept_len, prefix_mask_u8.to(torch.bool)


def verify_accept_ref(target_tokens, draft_tokens):
    match = target_tokens == draft_tokens
    mismatch = ~match
    first_mismatch = torch.argmax(mismatch.to(torch.int32), dim=-1)
    all_match = match.all(dim=-1)
    accept_len = torch.where(all_match, torch.full_like(first_mismatch, match.shape[1]), first_mismatch)
    idx = torch.arange(match.shape[1], device=match.device).unsqueeze(0)
    prefix_mask = idx < accept_len.unsqueeze(-1)
    return accept_len.to(torch.int32), prefix_mask


def bench_ms(fn, warmup=25, trials=120):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        vals.append(st.elapsed_time(en))
    vals.sort()
    return float(vals[len(vals)//2])


def run_one(shape, config):
    b = int(shape["batch"])
    l = int(shape["draft_len"])
    v = int(shape["vocab"])
    torch.manual_seed(0)
    target = torch.randint(0, v, (b, l), device="cuda", dtype=torch.int64)
    draft = target.clone()
    if l > 1:
        draft[:, l // 2:] = (draft[:, l // 2:] + 1) % v

    ref_len, ref_mask = verify_accept_ref(target, draft)
    out_len, out_mask = verify_accept_fused(target, draft, config)
    correct = bool(torch.equal(ref_len, out_len) and torch.equal(ref_mask, out_mask))

    fused_ms = bench_ms(lambda: verify_accept_fused(target, draft, config))
    sep_ms = bench_ms(lambda: verify_accept_ref(target, draft))
    speedup = sep_ms / fused_ms if fused_ms > 0 else 0.0
    return {{
        "shape": f"{{b}}x{{l}}x{{v}}",
        "shape_name": shape.get("name", ""),
        "correct": correct,
        "ms": round(fused_ms, 4),
        "separated_ms": round(sep_ms, 4),
        "tflops": round((b * l) / fused_ms, 4),
        "speedup": round(speedup, 4),
    }}


def main():
    configs = {configs_json}
    shapes = {shapes_json}
    all_results = []
    for config in configs:
        cid = f"bs{{config['BLOCK_SIZE']}}_w{{config['num_warps']}}_s{{config['num_stages']}}"
        rows = [run_one(shape, config) for shape in shapes]
        all_results.append({{"config_id": cid, "config": config, "results": rows}})

    print(json.dumps({{
        "hardware": {{
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        }},
        "config_results": all_results,
    }}, indent=2))


if __name__ == "__main__":
    main()
'''
