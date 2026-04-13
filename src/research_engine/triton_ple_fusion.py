"""Parameterized Triton kernel for fused PLE injection (Gemma 4).

Gemma 4 Per-Layer Experts (PLE) inject a 256-dim per-layer embedding into
each decoder layer via 5 separate ops: lookup, project, RMSNorm, scale,
residual add. This kernel fuses the last 3 (RMSNorm + scale + residual add)
into a single launch, after a tiny matvec projection done in PyTorch.

Key insight: the PLE embedding is the SAME for all tokens in a layer, so the
projection yields a single (D,) vector that broadcasts across all M=B*S rows.
The kernel is bandwidth-bound — same pattern as QK-RMSNorm+RoPE.

Grid: (M,) — one program per (batch, seq) position.
BLOCK_D covers hidden_dim elements per row.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


PLE_FUSION_PARAM_SPACE = {
    "BLOCK_D": [512, 1024, 1536, 2048, 4096],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
}


PLE_FUSION_CURATED_CONFIGS = [
    # E2B: D=1536 — BLOCK_D=2048 covers it
    {"BLOCK_D": 2048, "num_warps": 4, "num_stages": 1},
    {"BLOCK_D": 2048, "num_warps": 8, "num_stages": 1},
    {"BLOCK_D": 2048, "num_warps": 2, "num_stages": 1},
    {"BLOCK_D": 2048, "num_warps": 4, "num_stages": 2},
    # 31B: D=5376 — BLOCK_D needs to cover via loop or large block
    {"BLOCK_D": 4096, "num_warps": 8, "num_stages": 1},
    {"BLOCK_D": 4096, "num_warps": 4, "num_stages": 2},
    # Smaller blocks (lower register pressure, T4-friendly)
    {"BLOCK_D": 1024, "num_warps": 2, "num_stages": 1},
    {"BLOCK_D": 512, "num_warps": 1, "num_stages": 1},
]


PLE_FUSION_SHAPE_BUCKETS = [
    {"name": "gemma4_e2b_ple_fuse", "batch": 1, "seq_len": 2048, "hidden_dim": 1536, "embed_dim": 256},
    {"name": "gemma4_e2b_ple_fuse_4k", "batch": 1, "seq_len": 4096, "hidden_dim": 1536, "embed_dim": 256},
    {"name": "gemma4_31b_ple_fuse", "batch": 1, "seq_len": 2048, "hidden_dim": 5376, "embed_dim": 256},
]


def ple_fusion_config_id(config: dict[str, int]) -> str:
    return f"bd{config['BLOCK_D']}_w{config['num_warps']}_s{config['num_stages']}"


def ple_fusion_shape_bucket_key(shape: dict[str, int]) -> str:
    hd = shape.get("hidden_dim", 0)
    if hd > 3000:
        return "gemma4_31b_ple_fuse"
    return "gemma4_e2b_ple_fuse"


def ple_fusion_shared_memory_check(config: dict[str, int]) -> bool:
    return True


def generate_ple_fusion_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in PLE_FUSION_CURATED_CONFIGS:
            cid = ple_fusion_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bd in PLE_FUSION_PARAM_SPACE["BLOCK_D"]:
        for nw in PLE_FUSION_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {"BLOCK_D": bd, "num_warps": nw, "num_stages": ns}
                cid = ple_fusion_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


# ---------------------------------------------------------------------------
# Module-level Triton kernel (lazy init to avoid import errors without GPU)
# ---------------------------------------------------------------------------

_triton_ple_available = False
_ple_norm_scale_add_kernel_compiled = None


def _ensure_triton_ple_fusion():
    global _triton_ple_available, _ple_norm_scale_add_kernel_compiled
    if _ple_norm_scale_add_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _ple_norm_scale_add_kernel(
            hidden_ptr,       # (M, D) main hidden state — read+write
            projected_ptr,    # (D,) projected PLE vector (broadcast)
            norm_weight_ptr,  # (D,) RMSNorm weight
            scale_ptr,        # (D,) learned scale factor
            M, D,
            eps,
            BLOCK_D: tl.constexpr,
        ):
            """Fused RMSNorm(projected) * scale + hidden[row].

            One program per row of hidden state. The projected vector is the
            same for every row (broadcast from a single PLE embedding).

            Steps per row:
              1. Load projected (D,) — shared across all rows
              2. RMSNorm: rstd = 1/sqrt(mean(proj^2) + eps)
                 normed = proj * rstd * norm_weight
              3. Scale: scaled = normed * scale
              4. hidden[row] += scaled
            """
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_D)
            mask = offs < D

            # Load projected PLE vector (same for all rows — L1 cache hit)
            proj = tl.load(projected_ptr + offs, mask=mask, other=0.0).to(tl.float32)

            # RMSNorm: compute rstd over the projected vector
            sq = proj * proj
            mean_sq = tl.sum(sq, axis=0) / D
            rstd = 1.0 / tl.sqrt(mean_sq + eps)

            # Normalize with learnable weight
            w = tl.load(norm_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            normed = proj * rstd * w

            # Scale
            s = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            scaled = normed * s

            # Residual add
            h_base = hidden_ptr + row * D
            h = tl.load(h_base + offs, mask=mask, other=0.0).to(tl.float32)
            out = h + scaled
            tl.store(h_base + offs, out.to(tl.float16), mask=mask)

        _ple_norm_scale_add_kernel_compiled = _ple_norm_scale_add_kernel
        _triton_ple_available = True
    except ImportError:
        _triton_ple_available = False


def fused_ple_inject(hidden, ple_embedding, proj_weight, norm_weight, scale, config=None, eps=1e-6):
    """Fused PLE injection: project + normalize + scale + residual add.

    Args:
        hidden: (B, S, D) fp16 — modified in-place.
        ple_embedding: (E,) fp16 — the layer's PLE embedding (already looked up).
        proj_weight: (D, E) fp16 — projection weight.
        norm_weight: (D,) fp32 — RMSNorm weight.
        scale: (D,) fp32 — learned scale factor.
        config: Triton config dict with BLOCK_D, num_warps, num_stages.
        eps: Epsilon for RMSNorm.

    Returns:
        hidden (modified in-place).
    """
    import torch
    import triton

    _ensure_triton_ple_fusion()
    if not _triton_ple_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = PLE_FUSION_CURATED_CONFIGS[0]

    B, S, D = hidden.shape

    # Step 1: Small matvec projection (E -> D), done once per layer
    projected = ple_embedding.float() @ proj_weight.float().T  # (D,)

    # Step 2: Fused kernel — RMSNorm(projected) * scale + hidden[row]
    M = B * S
    hidden_flat = hidden.reshape(M, D)

    BLOCK_D = max(config["BLOCK_D"], triton.next_power_of_2(D))

    _ple_norm_scale_add_kernel_compiled[(M,)](
        hidden_flat, projected, norm_weight, scale,
        M, D, eps,
        BLOCK_D=BLOCK_D,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    return hidden


# ---------------------------------------------------------------------------
# Benchmark script generator
# ---------------------------------------------------------------------------

def generate_ple_fusion_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained Triton PLE fusion benchmark script."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated fused PLE injection benchmark (Gemma 4 per-layer experts)."""

import json
import platform

import torch
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def ple_norm_scale_add_kernel(
    hidden_ptr,
    projected_ptr,
    norm_weight_ptr,
    scale_ptr,
    M, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused RMSNorm(projected) * scale + hidden[row].

    One program per row. projected is broadcast (same for all rows).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    proj = tl.load(projected_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sq = proj * proj
    mean_sq = tl.sum(sq, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(norm_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = proj * rstd * w

    s = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scaled = normed * s

    h_base = hidden_ptr + row * D
    h = tl.load(h_base + offs, mask=mask, other=0.0).to(tl.float32)
    out = h + scaled
    tl.store(h_base + offs, out.to(tl.float16), mask=mask)


def apply_fused_ple(hidden, ple_embedding, proj_weight, norm_weight, scale, config, eps=1e-6):
    """Fused PLE injection: project + norm + scale + add."""
    B, S, D = hidden.shape
    projected = ple_embedding.float() @ proj_weight.float().T
    M = B * S
    hidden_flat = hidden.reshape(M, D)
    BLOCK_D = max(config["BLOCK_D"], triton.next_power_of_2(D))
    ple_norm_scale_add_kernel[(M,)](
        hidden_flat, projected, norm_weight, scale,
        M, D, eps,
        BLOCK_D=BLOCK_D,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return hidden


def separated_baseline(hidden, ple_embedding, proj_weight, norm_weight, scale, eps=1e-6):
    """PyTorch separated baseline: 5 distinct ops (lookup already done).

    1. Project: matvec
    2. RMSNorm
    3. Scale
    4. Residual add
    """
    B, S, D = hidden.shape
    # 1. Project
    projected = ple_embedding.float() @ proj_weight.float().T  # (D,)
    # 2. RMSNorm
    rms = projected.pow(2).mean().sqrt()
    normed = projected / (rms + eps) * norm_weight.float()
    # 3. Scale
    scaled = normed * scale.float()
    # 4. Residual add (broadcast)
    out = hidden.float() + scaled.unsqueeze(0).unsqueeze(0)
    return out.to(torch.float16)


def benchmark_one(batch, seq_len, hidden_dim, embed_dim, config):
    try:
        hidden = torch.randn((batch, seq_len, hidden_dim), device="cuda", dtype=torch.float16)
        ple_embedding = torch.randn((embed_dim,), device="cuda", dtype=torch.float16) * 0.01
        proj_weight = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16) * 0.02
        norm_weight = torch.ones((hidden_dim,), device="cuda", dtype=torch.float32)
        scale = torch.ones((hidden_dim,), device="cuda", dtype=torch.float32) * 0.1

        # Correctness check
        ref = separated_baseline(hidden, ple_embedding, proj_weight, norm_weight, scale)
        hidden_copy = hidden.clone()
        out = apply_fused_ple(hidden_copy, ple_embedding, proj_weight, norm_weight, scale, config)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None}}

        # Benchmark fused
        def fused_fn():
            h = hidden.clone()
            apply_fused_ple(h, ple_embedding, proj_weight, norm_weight, scale, config)

        ms = triton.testing.do_bench(fused_fn, warmup=25, rep=100)

        # Benchmark separated
        sep_ms = triton.testing.do_bench(
            lambda: separated_baseline(hidden, ple_embedding, proj_weight, norm_weight, scale),
            warmup=25, rep=100,
        )

        # Bandwidth: read hidden + projected + norm_w + scale, write hidden
        M = batch * seq_len
        bytes_moved = M * hidden_dim * 2 + hidden_dim * 4 * 3 + M * hidden_dim * 2
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        fusion_speedup = sep_ms / ms if ms > 0 else 0.0

        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "separated_ms": round(sep_ms, 4),
            "fusion_speedup": round(fusion_speedup, 3),
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
        cid = "bd{{}}_w{{}}_s{{}}".format(
            config["BLOCK_D"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            result = benchmark_one(
                shape["batch"], shape["seq_len"], shape["hidden_dim"],
                shape["embed_dim"], config,
            )
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "ple_fusion",
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
# Registry
# ---------------------------------------------------------------------------

PLE_FUSION_SPEC = register_operator(TritonOperatorSpec(
    name="ple_fusion",
    param_space=PLE_FUSION_PARAM_SPACE,
    curated_configs=PLE_FUSION_CURATED_CONFIGS,
    shape_buckets=PLE_FUSION_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=ple_fusion_config_id,
    shape_bucket_fn=ple_fusion_shape_bucket_key,
    benchmark_script_fn=generate_ple_fusion_benchmark_script,
    grid_generator_fn=generate_ple_fusion_grid,
    shared_memory_check_fn=ple_fusion_shared_memory_check,
    description=(
        "Fused PLE injection for Gemma 4: RMSNorm + scale + broadcast residual "
        "add of a per-layer embedding projection. Replaces 5 separate kernel "
        "launches (lookup, project, norm, scale, add) with a tiny matvec + one "
        "fused kernel. Bandwidth-bound."
    ),
))
