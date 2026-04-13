#!/usr/bin/env python3
"""Benchmark adaptive vs fixed Triton config selection.

Simulates an inference server processing varying batch sizes: for each
shape the adaptive selector picks the best known config from the database,
while the fixed baseline always uses the first curated config.

Usage (requires CUDA GPU):
    python scripts/adaptive_benchmark.py [--db-path .noeris/triton-configs.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_engine.adaptive_config import AdaptiveConfigSelector
from research_engine.triton_rmsnorm import RMSNORM_CURATED_CONFIGS, rmsnorm


SHAPES = [
    {"label": "llama_7b",   "n_rows": 4096, "hidden_dim": 4096},
    {"label": "gemma4_e2b", "n_rows": 2048, "hidden_dim": 1536},
    {"label": "llama_70b",  "n_rows": 2048, "hidden_dim": 8192},
    {"label": "phi3_mini",  "n_rows": 4096, "hidden_dim": 3072},
    {"label": "gemma4_31b", "n_rows": 4096, "hidden_dim": 5376},
]


def bench_one(x, w, config, warmup=10, rep=50):
    """Return median latency in ms using triton.testing.do_bench."""
    import triton
    ms = triton.testing.do_bench(
        lambda: rmsnorm(x, w, config=config),
        warmup=warmup,
        rep=rep,
    )
    return ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive vs fixed config benchmark")
    parser.add_argument(
        "--db-path", nargs="*",
        default=[".noeris/triton-configs.json", ".noeris/colab-configs.json"],
        help="Config database path(s)",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for this benchmark.", file=sys.stderr)
        sys.exit(1)

    db_paths = [Path(p) for p in args.db_path]
    selector = AdaptiveConfigSelector(
        operator_name="rmsnorm",
        db_paths=db_paths,
    )

    fixed_config = RMSNORM_CURATED_CONFIGS[0]
    print(f"Hardware:       {selector.hardware}")
    print(f"Fixed config:   {fixed_config}")
    print(f"Known buckets:  {selector.known_buckets}")
    print()

    results = []
    for shape in SHAPES:
        n_rows = shape["n_rows"]
        hidden_dim = shape["hidden_dim"]
        label = shape["label"]

        adaptive_cfg = selector.select(
            hidden_dim=hidden_dim, n_rows=n_rows,
        )

        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=torch.float16)
        w = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)

        fixed_ms = bench_one(x, w, fixed_config)
        adaptive_ms = bench_one(x, w, adaptive_cfg)
        speedup = fixed_ms / adaptive_ms if adaptive_ms > 0 else 0.0
        same = adaptive_cfg == fixed_config

        result = {
            "shape": label,
            "n_rows": n_rows,
            "hidden_dim": hidden_dim,
            "fixed_config": fixed_config,
            "adaptive_config": adaptive_cfg,
            "same_config": same,
            "fixed_ms": round(fixed_ms, 4),
            "adaptive_ms": round(adaptive_ms, 4),
            "speedup": round(speedup, 3),
        }
        results.append(result)
        marker = "==" if same else ">>"
        print(
            f"  {label:20s}  fixed={fixed_ms:7.3f}ms  adaptive={adaptive_ms:7.3f}ms  "
            f"speedup={speedup:.3f}x  {marker} {adaptive_cfg}"
        )

    # Aggregate
    total_fixed = sum(r["fixed_ms"] for r in results)
    total_adaptive = sum(r["adaptive_ms"] for r in results)
    agg_speedup = total_fixed / total_adaptive if total_adaptive > 0 else 0.0
    n_different = sum(1 for r in results if not r["same_config"])

    print()
    print(f"Aggregate:  fixed={total_fixed:.3f}ms  adaptive={total_adaptive:.3f}ms  "
          f"speedup={agg_speedup:.3f}x  ({n_different}/{len(results)} shapes used different config)")

    output = {
        "hardware": selector.hardware,
        "fixed_config": fixed_config,
        "known_buckets": selector.known_buckets,
        "results": results,
        "aggregate_speedup": round(agg_speedup, 3),
    }
    print()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
