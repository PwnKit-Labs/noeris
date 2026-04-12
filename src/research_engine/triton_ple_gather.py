"""Parameterized Triton kernel for Per-Layer Embedding (PLE) gather+add.

Gemma 4 E2B/E4B small variants use a second embedding table that feeds a
residual signal into every decoder layer (carried from Gemma-3n). The PLE
table is indexed by token ID and layer index, producing a ple_dim-wide
vector that is added into the first ple_dim channels of the residual stream.

This kernel is bandwidth-bound: it gathers from a large table
(vocab_size * num_layers * ple_dim) and adds elementwise into the residual.

Grid: (batch * seq_len,) — one program per (batch, seq) position.
BLOCK_SIZE covers ple_dim elements.

NOTE: To wire into the registry, import this module from __init__.py or
ensure it is imported before operator dispatch. Left for manual wiring.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


PLE_GATHER_PARAM_SPACE = {
    "BLOCK_SIZE": [64, 128, 256, 512],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
}


PLE_GATHER_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 256, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 512, "num_warps": 8, "num_stages": 2},
    # T4-optimized: ple_dim=256 fits in BLOCK_SIZE=256, fewer warps
    {"BLOCK_SIZE": 256, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 128, "num_warps": 1, "num_stages": 1},
    # Deep pipeline for bandwidth-bound gather
    {"BLOCK_SIZE": 512, "num_warps": 4, "num_stages": 3},
]


PLE_GATHER_SHAPE_BUCKETS = [
    {"name": "gemma4_e2b_ple", "batch": 1, "seq_len": 4096, "hidden_dim": 1536, "ple_dim": 256, "vocab_size": 262144, "num_layers": 35},
    {"name": "gemma4_e4b_ple", "batch": 1, "seq_len": 4096, "hidden_dim": 2560, "ple_dim": 256, "vocab_size": 262144, "num_layers": 42},
]


def ple_gather_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def ple_gather_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a PLE gather shape into a Gemma 4 bucket.

    Discriminator: hidden_dim distinguishes E2B (1536) from E4B (2560).
    """
    hd = shape.get("hidden_dim", 0)
    if hd > 2000:
        return "gemma4_e4b_ple"
    return "gemma4_e2b_ple"


def ple_gather_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — feasibility is learned at runtime."""
    return True


def generate_ple_gather_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in PLE_GATHER_CURATED_CONFIGS:
            cid = ple_gather_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in PLE_GATHER_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in PLE_GATHER_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {"BLOCK_SIZE": bs, "num_warps": nw, "num_stages": ns}
                cid = ple_gather_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_ple_gather_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained Triton PLE gather+add benchmark script."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated PLE gather+add benchmark (Gemma 4 E2B/E4B per-layer embedding)."""

import json
import platform

import torch
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def ple_gather_kernel(
    residual_ptr, ple_table_ptr, token_ids_ptr, out_ptr,
    hidden_dim, ple_dim, num_layers, layer_idx,
    residual_row_stride, out_row_stride,
    ple_layer_stride, ple_dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather from PLE table and add into the first ple_dim channels.

    Grid: (batch * seq_len,) — one program per (b, s) position.

    For each position:
      out[b, s, :ple_dim] = residual[b, s, :ple_dim] + ple_table[token_ids[b,s], layer_idx, :]
      out[b, s, ple_dim:] = residual[b, s, ple_dim:]  (pass-through)
    """
    pid = tl.program_id(0)

    # Load the token id for this (b, s) position.
    # Cast to int64 to avoid int32 overflow: tok (up to 262143) * ple_layer_stride
    # (up to 8960) = 2.35B which exceeds INT32_MAX (2.15B).
    tok = tl.load(token_ids_ptr + pid).to(tl.int64)

    res_base = residual_ptr + pid * residual_row_stride
    out_base = out_ptr + pid * out_row_stride

    # PLE table offset: ple_table[tok, layer_idx, :]
    # Layout: (vocab_size, num_layers, ple_dim), row-major
    # ple_layer_stride = num_layers * ple_dim (vocab-dim stride, already includes num_layers)
    ple_base = ple_table_ptr + tok * tl.cast(ple_layer_stride, tl.int64) + layer_idx * ple_dim_stride

    # Process the ple_dim region: gather + add
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < ple_dim

    res_ple = tl.load(res_base + offs, mask=mask, other=0.0).to(tl.float32)
    ple_val = tl.load(ple_base + offs, mask=mask, other=0.0).to(tl.float32)
    out_ple = res_ple + ple_val
    tl.store(out_base + offs, out_ple.to(tl.float16), mask=mask)

    # Pass-through for dims beyond ple_dim.
    # We handle this in a loop over BLOCK_SIZE chunks.
    remaining = hidden_dim - ple_dim
    num_chunks = (remaining + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk_idx in range(0, num_chunks):
        chunk_offs = tl.arange(0, BLOCK_SIZE) + chunk_idx * BLOCK_SIZE
        chunk_mask = chunk_offs < remaining
        vals = tl.load(res_base + ple_dim + chunk_offs, mask=chunk_mask, other=0.0)
        tl.store(out_base + ple_dim + chunk_offs, vals, mask=chunk_mask)


def apply_ple_gather(residual, ple_table, token_ids, layer_idx, config):
    """Launch the PLE gather+add kernel.

    residual: (B, S, hidden_dim) fp16
    ple_table: (vocab_size, num_layers, ple_dim) fp16
    token_ids: (B, S) int32
    layer_idx: int
    """
    B, S, hidden_dim = residual.shape
    vocab_size, num_layers, ple_dim = ple_table.shape

    out = torch.empty_like(residual)

    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(ple_dim))

    grid = (B * S,)
    ple_gather_kernel[grid](
        residual, ple_table, token_ids.reshape(-1), out,
        hidden_dim, ple_dim, num_layers, layer_idx,
        hidden_dim,  # residual_row_stride (elements per row)
        hidden_dim,  # out_row_stride
        num_layers * ple_dim,  # ple_layer_stride (stride for vocab dim)
        ple_dim,  # ple_dim_stride (stride for layer dim)
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def torch_ple_gather(residual, ple_table, token_ids, layer_idx):
    """PyTorch reference for PLE gather+add."""
    B, S, hidden_dim = residual.shape
    ple_dim = ple_table.shape[2]

    # Gather: ple_table[token_ids, layer_idx, :] -> (B, S, ple_dim)
    gathered = ple_table[token_ids.long(), layer_idx, :]  # (B, S, ple_dim)

    out = residual.clone()
    out[:, :, :ple_dim] = residual[:, :, :ple_dim] + gathered
    return out


def benchmark_one(batch, seq_len, hidden_dim, ple_dim, vocab_size, num_layers, config):
    try:
        residual = torch.randn((batch, seq_len, hidden_dim), device="cuda", dtype=torch.float16)
        ple_table = torch.randn((vocab_size, num_layers, ple_dim), device="cuda", dtype=torch.float16) * 0.01
        token_ids = torch.randint(0, vocab_size, (batch, seq_len), device="cuda", dtype=torch.int32)
        layer_idx = num_layers // 2

        ref = torch_ple_gather(residual, ple_table, token_ids, layer_idx)
        out = apply_ple_gather(residual, ple_table, token_ids, layer_idx, config)

        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None}}

        ms = triton.testing.do_bench(
            lambda: apply_ple_gather(residual, ple_table, token_ids, layer_idx, config),
            warmup=25, rep=100,
        )

        # Memory bandwidth: read residual + gathered PLE rows + token_ids, write out
        res_bytes = batch * seq_len * hidden_dim * 2
        ple_bytes = batch * seq_len * ple_dim * 2  # gathered rows
        tok_bytes = batch * seq_len * 4
        out_bytes = batch * seq_len * hidden_dim * 2
        bytes_moved = res_bytes + ple_bytes + tok_bytes + out_bytes
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9

        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "gb_per_s": round(gb_per_s, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None}}


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
            result = benchmark_one(
                shape["batch"], shape["seq_len"], shape["hidden_dim"],
                shape["ple_dim"], shape["vocab_size"], shape["num_layers"],
                config,
            )
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "ple_gather",
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


PLE_GATHER_SPEC = register_operator(TritonOperatorSpec(
    name="ple_gather",
    param_space=PLE_GATHER_PARAM_SPACE,
    curated_configs=PLE_GATHER_CURATED_CONFIGS,
    shape_buckets=PLE_GATHER_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=ple_gather_config_id,
    shape_bucket_fn=ple_gather_shape_bucket_key,
    benchmark_script_fn=generate_ple_gather_benchmark_script,
    grid_generator_fn=generate_ple_gather_grid,
    shared_memory_check_fn=ple_gather_shared_memory_check,
    description=(
        "PLE gather+add for Gemma 4 E2B/E4B: gathers per-layer embeddings "
        "from a (vocab_size, num_layers, ple_dim) table and adds into the "
        "first ple_dim channels of the residual stream. Bandwidth-bound."
    ),
))
