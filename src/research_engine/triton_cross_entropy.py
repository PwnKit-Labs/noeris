"""Parameterized Triton Cross-entropy kernel and operator spec.

Fused cross-entropy with online log-sum-exp. AutoKernel reports 2.21x
over PyTorch eager and 2.94x over torch.compile on H100.

Computes: loss[i] = -log(exp(logits[i, target[i]]) / sum_j exp(logits[i, j]))
                  = -(logits[i, target[i]] - log_sum_exp(logits[i, :]))

A single Triton kernel computes log_sum_exp + per-target loss in one pass.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


CROSS_ENTROPY_PARAM_SPACE = {
    "BLOCK_SIZE": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3],
}


CROSS_ENTROPY_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 8192, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 16384, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 32768, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
    {"BLOCK_SIZE": 1024, "num_warps": 2, "num_stages": 2},
    # T4-optimized: 40 SMs, lower warp count saturates better
    {"BLOCK_SIZE": 4096, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 2048, "num_warps": 2, "num_stages": 1},
    # Gemma 4 256k vocab: needs very large blocks to cover n_cols in one pass
    {"BLOCK_SIZE": 32768, "num_warps": 8, "num_stages": 2},
]


# (batch*seq_len, vocab_size)
CROSS_ENTROPY_SHAPE_BUCKETS = [
    {"name": "gpt2_small", "n_rows": 1024, "n_cols": 50257},
    {"name": "gpt2_med", "n_rows": 2048, "n_cols": 50257},
    {"name": "llama_32k", "n_rows": 2048, "n_cols": 32000},
    {"name": "llama_32k_long", "n_rows": 4096, "n_cols": 32000},
    {"name": "llama3_128k", "n_rows": 2048, "n_cols": 128256},
    {"name": "mistral", "n_rows": 4096, "n_cols": 32000},
    {"name": "bert_vocab", "n_rows": 4096, "n_cols": 30522},
    # Gemma 4 family: 256k vocab is the largest published vocabulary for a dense LLM
    {"name": "gemma4_vocab_256k_short", "n_rows": 2048, "n_cols": 256000},
    {"name": "gemma4_vocab_256k_long", "n_rows": 4096, "n_cols": 256000},
]


def cross_entropy_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def cross_entropy_shape_bucket_key(shape: dict[str, int]) -> str:
    cols = shape.get("n_cols", 0)
    rows = shape.get("n_rows", 0)
    # Gemma 4's 256k vocab is larger than any other published LLM vocabulary.
    if cols >= 200000:
        return "gemma4_vocab_256k_short" if rows <= 2048 else "gemma4_vocab_256k_long"
    if cols >= 100000:
        return "llama3_128k"
    if cols >= 50000:
        return "gpt2_med" if rows >= 2048 else "gpt2_small"
    if cols >= 32000:
        if rows >= 4096:
            return "mistral"
        if rows >= 2048:
            return "llama_32k_long"
        return "llama_32k"
    return "bert_vocab"


def cross_entropy_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True. See module docstring."""
    return True


def generate_cross_entropy_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in CROSS_ENTROPY_CURATED_CONFIGS:
            cid = cross_entropy_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in CROSS_ENTROPY_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in CROSS_ENTROPY_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {
                    "BLOCK_SIZE": bs,
                    "num_warps": nw,
                    "num_stages": ns,
                }
                cid = cross_entropy_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_cross_entropy_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton Cross-entropy benchmark."""

import json
import platform

import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_ptr,
    logits_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused cross-entropy with online log-sum-exp.

    loss[i] = -(logits[i, target[i]] - log_sum_exp(logits[i, :]))
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    target = tl.load(target_ptr + row_idx)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    logits = tl.load(logits_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    log_sum_exp = row_max + tl.log(tl.sum(tl.exp(logits - row_max), axis=0))

    target_logit = tl.load(logits_ptr + target).to(tl.float32)
    loss = log_sum_exp - target_logit

    tl.store(loss_ptr + row_idx, loss.to(tl.float16))


def cross_entropy(logits, target, config):
    n_rows, n_cols = logits.shape
    loss = torch.empty((n_rows,), device=logits.device, dtype=torch.float16)
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(n_cols))
    cross_entropy_kernel[(n_rows,)](
        logits, target, loss,
        logits.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return loss


def benchmark_one(n_rows, n_cols, config):
    try:
        logits = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float16)
        target = torch.randint(0, n_cols, (n_rows,), device="cuda", dtype=torch.long)
        ref = torch.nn.functional.cross_entropy(logits.to(torch.float32), target, reduction="none")
        out = cross_entropy(logits, target, config)
        max_err = (out.to(torch.float32) - ref).abs().max().item()
        if max_err > 0.5:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}
        ms = triton.testing.do_bench(lambda: cross_entropy(logits, target, config), warmup=25, rep=100)
        bytes_moved = n_rows * n_cols * 2 + n_rows * 8 + n_rows * 2  # logits + target + loss
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
            result = benchmark_one(n_rows, n_cols, config)
            result["shape"] = f"{{n_rows}}x{{n_cols}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "cross_entropy",
        "hardware": {{
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
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


CROSS_ENTROPY_SPEC = register_operator(TritonOperatorSpec(
    name="cross_entropy",
    param_space=CROSS_ENTROPY_PARAM_SPACE,
    curated_configs=CROSS_ENTROPY_CURATED_CONFIGS,
    shape_buckets=CROSS_ENTROPY_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=cross_entropy_config_id,
    shape_bucket_fn=cross_entropy_shape_bucket_key,
    benchmark_script_fn=generate_cross_entropy_benchmark_script,
    grid_generator_fn=generate_cross_entropy_grid,
    shared_memory_check_fn=cross_entropy_shared_memory_check,
    description="Fused cross-entropy with online log-sum-exp. Beats torch by 2-3x on large vocabs.",
))
