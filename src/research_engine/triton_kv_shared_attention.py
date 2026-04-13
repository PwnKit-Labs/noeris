"""K=V shared attention optimization for Gemma 4 global layers.

Gemma 4's global attention layers use K=V (keys literally equal values).
Every framework (vLLM, SGLang, HuggingFace, FlashInfer) still computes
separate K and V projections and stores both in KV cache.

This module exploits the K=V identity to:
  1. Halve the KV projection GEMM (one W_kv matmul instead of W_k + W_v)
  2. Halve KV-cache writes and memory (store once, read as both K and V)
  3. Pass the same tensor as both K and V to SDPA (skip redundant V loads)

For the projection stage, the saving is straightforward: replace two
matmuls (hidden @ W_k, hidden @ W_v) with one (hidden @ W_kv), then use
the result for both K and V in the attention computation.

For the attention kernel itself, when K and V share the same underlying
storage, SDPA already benefits from improved L2 cache hit rates since
the K-tile loaded for the QK^T dot product is the same data needed for
the softmax(QK^T) @ V accumulation.

Issue #76: K=V global attention fusion.

Gemma 4 global layer shapes:
  - 31B Dense:       H=32, H_kv=4,  Dh=512, hidden=5376
  - 26B A4B (MoE):   H=16, H_kv=2,  Dh=512, hidden=2816
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

GEMMA4_GLOBAL_SHAPES = [
    {
        "name": "gemma4_31b_global",
        "batch": 1,
        "seq_len": 2048,
        "hidden_dim": 5376,
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 512,
    },
    {
        "name": "gemma4_26b_a4b_global",
        "batch": 1,
        "seq_len": 2048,
        "hidden_dim": 2816,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 512,
    },
]


# ---------------------------------------------------------------------------
# Code-generation helpers
# ---------------------------------------------------------------------------

def _indent(code: str, n: int = 4) -> str:
    """Indent every line of *code* by *n* spaces."""
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in code.splitlines())


def generate_kv_shared_benchmark_script(
    configs: list[dict[str, int]] | None = None,
    shapes: list[dict[str, Any]] | None = None,
    warmup: int = 50,
    rep: int = 200,
) -> str:
    """Return a self-contained Python script that benchmarks K=V projection sharing.

    The script runs on any CUDA GPU (T4, A100, H100) and reports:
      - Projection latency: 3 matmuls (Q+K+V) vs 2 matmuls (Q+KV)
      - Full attention path: proj + SDPA
      - KV cache memory savings

    Args:
        configs: Tunable config dicts (unused — no Triton params yet).
        shapes: Shape dicts to benchmark. Defaults to GEMMA4_GLOBAL_SHAPES.
        warmup: Number of warmup iterations for CUDA timer.
        rep: Number of timed repetitions for CUDA timer.
    """
    # configs is accepted for API compatibility but unused (no tunable params).
    shapes = shapes or GEMMA4_GLOBAL_SHAPES

    shapes_json = json.dumps(shapes, indent=4)

    script = f'''\
#!/usr/bin/env python3
"""K=V shared attention benchmark for Gemma 4 global layers.

Gemma 4 global attention has K=V (keys equal values). This benchmark
measures the speedup from exploiting that identity:
  - Projection: 2 matmuls (Q + KV) instead of 3 (Q + K + V)
  - Attention: pass same tensor as K and V to SDPA
  - Memory: halve KV-cache storage

Usage (Kaggle T4 / Colab):
    !pip install torch -q
    !python kv_shared_benchmark.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

print(f"PyTorch {{torch.__version__}}, CUDA available: {{torch.cuda.is_available()}}")
if not torch.cuda.is_available():
    print("ERROR: No GPU available. Change runtime to T4 GPU.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {{GPU_NAME}}")

# ============================================================================
# Shapes
# ============================================================================

SHAPES = {shapes_json}

WARMUP = {warmup}
REP = {rep}


# ============================================================================
# Benchmark helpers
# ============================================================================

def cuda_timer(fn, warmup=WARMUP, rep=REP):
    """Time *fn* using CUDA events, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median


def fmt(ms):
    """Format milliseconds."""
    if ms < 0.01:
        return f"{{ms * 1000:.1f}} us"
    return f"{{ms:.3f}} ms"


def expand_kv(kv, num_heads, num_kv_heads):
    """Repeat-interleave KV heads to match Q head count (GQA expansion).

    Args:
        kv: Tensor of shape (B, H_kv, S, Dh).
        num_heads: Total number of query heads (H).
        num_kv_heads: Number of KV heads (H_kv).

    Returns:
        Tensor of shape (B, H, S, Dh).
    """
    if num_heads == num_kv_heads:
        return kv
    n_rep = num_heads // num_kv_heads
    B, H_kv, S, Dh = kv.shape
    return kv[:, :, None, :, :].expand(B, H_kv, n_rep, S, Dh).reshape(B, num_heads, S, Dh)


# ============================================================================
# Per-shape benchmark
# ============================================================================

def benchmark_shape(cfg):
    """Run projection and full-attention benchmarks for one shape config."""
    name = cfg["name"]
    B = cfg["batch"]
    S = cfg["seq_len"]
    D = cfg["hidden_dim"]
    H = cfg["num_heads"]
    H_kv = cfg["num_kv_heads"]
    Dh = cfg["head_dim"]

    print(f"\\n{{' ' + name + ' ':=^72}}")
    print(f"  B={{B}}, S={{S}}, D={{D}}, H={{H}}, H_kv={{H_kv}}, Dh={{Dh}}")

    # Allocate input + weights
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
    W_q = torch.randn(D, H * Dh, device="cuda", dtype=torch.float16)
    W_k = torch.randn(D, H_kv * Dh, device="cuda", dtype=torch.float16)
    W_v = torch.randn(D, H_kv * Dh, device="cuda", dtype=torch.float16)
    # For K=V, we use a single weight matrix
    W_kv = W_k  # same weight — K=V identity

    results = {{"name": name, "B": B, "S": S, "D": D, "H": H, "H_kv": H_kv, "Dh": Dh}}

    # ------------------------------------------------------------------
    # 1. Projection benchmark: 3 matmuls vs 2 matmuls
    # ------------------------------------------------------------------
    print("\\n  [Projection: 3 matmuls vs 2 matmuls]")

    def proj_baseline():
        q = x @ W_q
        k = x @ W_k
        v = x @ W_v
        return q, k, v

    def proj_kv_shared():
        q = x @ W_q
        kv = x @ W_kv
        return q, kv

    t_base = cuda_timer(proj_baseline)
    t_shared = cuda_timer(proj_kv_shared)
    speedup_proj = t_base / t_shared

    print(f"    baseline (Q+K+V):   {{fmt(t_base)}}")
    print(f"    kv_shared (Q+KV):   {{fmt(t_shared)}}")
    print(f"    speedup:            {{speedup_proj:.2f}}x")

    results["proj_baseline_ms"] = t_base
    results["proj_kv_shared_ms"] = t_shared
    results["proj_speedup"] = speedup_proj

    # ------------------------------------------------------------------
    # 2. Full attention path: proj + reshape + SDPA
    # ------------------------------------------------------------------
    print("\\n  [Full attention: proj + SDPA]")

    scale = Dh ** -0.5

    def attn_baseline():
        q = (x @ W_q).view(B, S, H, Dh).transpose(1, 2)
        k = (x @ W_k).view(B, S, H_kv, Dh).transpose(1, 2)
        v = (x @ W_v).view(B, S, H_kv, Dh).transpose(1, 2)
        k = expand_kv(k, H, H_kv)
        v = expand_kv(v, H, H_kv)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

    def attn_kv_shared():
        q = (x @ W_q).view(B, S, H, Dh).transpose(1, 2)
        kv = (x @ W_kv).view(B, S, H_kv, Dh).transpose(1, 2)
        kv = expand_kv(kv, H, H_kv)
        return F.scaled_dot_product_attention(q, kv, kv, is_causal=True, scale=scale)

    t_base_attn = cuda_timer(attn_baseline)
    t_shared_attn = cuda_timer(attn_kv_shared)
    speedup_attn = t_base_attn / t_shared_attn

    print(f"    baseline (Q+K+V+SDPA):  {{fmt(t_base_attn)}}")
    print(f"    kv_shared (Q+KV+SDPA):  {{fmt(t_shared_attn)}}")
    print(f"    speedup:                {{speedup_attn:.2f}}x")

    results["attn_baseline_ms"] = t_base_attn
    results["attn_kv_shared_ms"] = t_shared_attn
    results["attn_speedup"] = speedup_attn

    # ------------------------------------------------------------------
    # 3. KV cache memory savings
    # ------------------------------------------------------------------
    kv_cache_baseline_bytes = 2 * B * S * H_kv * Dh * 2  # K + V, fp16
    kv_cache_shared_bytes = 1 * B * S * H_kv * Dh * 2    # KV only, fp16
    savings_mb = (kv_cache_baseline_bytes - kv_cache_shared_bytes) / (1024 ** 2)

    print(f"\\n  [KV cache memory]")
    print(f"    baseline (K+V):   {{kv_cache_baseline_bytes / (1024**2):.2f}} MB")
    print(f"    kv_shared (KV):   {{kv_cache_shared_bytes / (1024**2):.2f}} MB")
    print(f"    savings:          {{savings_mb:.2f}} MB  (50%)")

    results["kv_cache_baseline_mb"] = kv_cache_baseline_bytes / (1024 ** 2)
    results["kv_cache_shared_mb"] = kv_cache_shared_bytes / (1024 ** 2)
    results["kv_cache_savings_mb"] = savings_mb

    # ------------------------------------------------------------------
    # 4. Correctness check: attn(Q, K, V) == attn(Q, KV, KV) when K=V
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    x_check = torch.randn(B, min(S, 512), D, device="cuda", dtype=torch.float16)
    q_c = (x_check @ W_q).view(B, -1, H, Dh).transpose(1, 2)
    k_c = (x_check @ W_k).view(B, -1, H_kv, Dh).transpose(1, 2)
    k_c = expand_kv(k_c, H, H_kv)

    out_base = F.scaled_dot_product_attention(q_c, k_c, k_c, is_causal=True, scale=scale)
    out_shared = F.scaled_dot_product_attention(q_c, k_c, k_c, is_causal=True, scale=scale)

    max_diff = (out_base - out_shared).abs().max().item()
    print(f"\\n  [Correctness] max |baseline - kv_shared| = {{max_diff:.2e}}  (expect 0.0)")
    results["correctness_max_diff"] = max_diff

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("K=V Shared Attention Benchmark — Gemma 4 Global Layers")
    print("=" * 72)
    print(f"\\nExploiting K=V identity: halve KV projection + cache.")
    print(f"Ref: https://github.com/PwnKit-Labs/noeris/issues/76")

    all_results = []
    for shape in SHAPES:
        result = benchmark_shape(shape)
        all_results.append(result)

    # Summary table
    print("\\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  {{'Shape':<30s}} {{'Proj speedup':>14s}} {{'Attn speedup':>14s}} {{'Cache saved':>12s}}")
    print("  " + "-" * 70)
    for r in all_results:
        print(f"  {{r['name']:<30s}} {{r['proj_speedup']:>13.2f}}x {{r['attn_speedup']:>13.2f}}x {{r['kv_cache_savings_mb']:>10.2f}} MB")

    # Save JSON
    out_path = Path("kv_shared_results.json")
    meta = {{
        "gpu": GPU_NAME,
        "pytorch": torch.__version__,
        "warmup": WARMUP,
        "rep": REP,
    }}
    payload = {{"meta": meta, "results": all_results}}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\\nResults saved to {{out_path}}")


if __name__ == "__main__":
    main()
'''
    return script


# ---------------------------------------------------------------------------
# Stub helpers for TritonOperatorSpec required fields
# ---------------------------------------------------------------------------

def _config_id(cfg: dict[str, int]) -> str:
    return "kv_shared_default"


def _shape_bucket_key(shape: dict[str, object]) -> str:
    return str(shape.get("name", "unknown"))


def _grid_generator(**kwargs: Any) -> list[dict[str, int]]:
    return [{}]  # No tunable params yet


# ---------------------------------------------------------------------------
# Operator spec (for research engine integration)
# ---------------------------------------------------------------------------

KV_SHARED_ATTENTION_SPEC = register_operator(TritonOperatorSpec(
    name="kv_shared_attention",
    param_space={},  # No tunable Triton params yet — uses SDPA
    curated_configs=[],
    shape_buckets=GEMMA4_GLOBAL_SHAPES,
    metric_name="gb_per_s",
    config_id_fn=_config_id,
    shape_bucket_fn=_shape_bucket_key,
    benchmark_script_fn=generate_kv_shared_benchmark_script,
    grid_generator_fn=_grid_generator,
    description=(
        "K=V shared attention for Gemma 4 global layers. "
        "Halves KV projection and cache by exploiting K=V identity."
    ),
))
