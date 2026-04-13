#!/usr/bin/env python3
"""NAS-style experiment: sweep architecture configs and predict layer latency.

Uses the kernel-aware ArchitectureCostModel to predict end-to-end decoder
layer latency from architecture hyperparameters.  Key goals:

1. Compare known architectures (Gemma 4 E2B, 31B, Llama 3 8B).
2. Explore novel configs (wider FFN, narrower heads, etc.).
3. Test "kernel cliff" hypothesis: do tile-unaligned dimensions (e.g.
   hidden_dim=4000 vs 4096) cause measurable slowdowns?

Usage::

    python scripts/nas_experiment.py [--hardware a100|t4|h100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Direct import to avoid research_engine.__init__ pulling in torch
import importlib.util
_mod_path = Path(__file__).resolve().parent.parent / "src" / "research_engine" / "arch_cost_model.py"
_spec = importlib.util.spec_from_file_location("arch_cost_model", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ArchitectureCostModel = _mod.ArchitectureCostModel
_tile_efficiency = _mod._tile_efficiency


# -----------------------------------------------------------------------
# Architecture configs
# -----------------------------------------------------------------------

KNOWN_CONFIGS = [
    {
        "name": "gemma4_e2b",
        "hidden_dim": 1536, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 6144, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 512,
    },
    {
        "name": "gemma4_31b",
        "hidden_dim": 5376, "num_heads": 32, "num_kv_heads": 16,
        "head_dim": 256, "ffn_dim": 21504, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "llama3_8b",
        "hidden_dim": 4096, "num_heads": 32, "num_kv_heads": 8,
        "head_dim": 128, "ffn_dim": 14336, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    },
    {
        "name": "llama3_70b",
        "hidden_dim": 8192, "num_heads": 64, "num_kv_heads": 8,
        "head_dim": 128, "ffn_dim": 28672, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    },
]

NOVEL_CONFIGS = [
    {
        "name": "optimal_2b",
        "hidden_dim": 2048, "num_heads": 16, "num_kv_heads": 2,
        "head_dim": 128, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "wide_ffn_2b",
        "hidden_dim": 1536, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "narrow_heads",
        "hidden_dim": 2048, "num_heads": 32, "num_kv_heads": 4,
        "head_dim": 64, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "deep_narrow",
        "hidden_dim": 1024, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 128, "ffn_dim": 4096, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 512,
    },
    {
        "name": "big_head_gqa",
        "hidden_dim": 2048, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
]


def run_comparison(model: ArchitectureCostModel) -> None:
    """Compare all configs and print a table."""
    all_configs = KNOWN_CONFIGS + NOVEL_CONFIGS

    print(f"\n{'='*90}")
    print(f"  Layer Latency Predictions  ({model.hardware.upper()})")
    print(f"{'='*90}")
    header = f"{'Config':20s} {'Total ms':>9s} {'Bottleneck':>18s} {'Attn ms':>8s} {'MLP ms':>8s} {'MatMul ms':>9s}"
    print(header)
    print("-" * 90)

    results = []
    for cfg in all_configs:
        name = cfg.pop("name")
        pred = model.predict_layer_ms(cfg)
        cfg["name"] = name  # restore

        pk = pred["per_kernel"]
        attn_ms = pk.get("attention", 0)
        mlp_ms = pk.get("geglu_mlp", 0)
        matmul_ms = pk.get("qkv_projection", 0) + pk.get("output_projection", 0)

        print(f"{name:20s} {pred['total_ms']:9.3f} {pred['bottleneck']:>18s} "
              f"{attn_ms:8.3f} {mlp_ms:8.3f} {matmul_ms:9.3f}")
        results.append((name, pred))

    # Efficiency-normalized: ms per parameter proxy (hidden_dim * ffn_dim)
    print(f"\n{'='*90}")
    print("  Efficiency: ms per M-params proxy  (hidden * ffn / 1e6)")
    print(f"{'='*90}")
    for cfg in all_configs:
        name = cfg["name"]
        pred = [r for n, r in results if n == name][0]
        param_proxy = cfg["hidden_dim"] * cfg["ffn_dim"] / 1e6
        eff = pred["total_ms"] / param_proxy
        print(f"  {name:20s}  {eff:.4f} ms/M-param-proxy  "
              f"(tile_eff D={_tile_efficiency(cfg['hidden_dim']):.2f} "
              f"FFN={_tile_efficiency(cfg['ffn_dim']):.2f})")


def run_kernel_cliff_test(model: ArchitectureCostModel) -> None:
    """Test whether tile-unaligned dimensions cause performance cliffs."""
    print(f"\n{'='*90}")
    print("  Kernel Cliff Test: hidden_dim sweep around 4096")
    print(f"{'='*90}")

    base = {
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "ffn_dim": 14336, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    }

    dims = list(range(3840, 4352, 32))
    results = model.sweep_dimension(base, "hidden_dim", dims)

    print(f"  {'hidden_dim':>10s} {'total_ms':>9s} {'aligned':>8s} {'tile_eff':>9s} {'bottleneck':>18s}")
    print("  " + "-" * 60)
    for r in results:
        eff = _tile_efficiency(r["hidden_dim"])
        aligned = "yes" if r["aligned_128"] else "NO"
        marker = " ***" if not r["aligned_128"] else ""
        print(f"  {r['hidden_dim']:10d} {r['total_ms']:9.3f} {aligned:>8s} "
              f"{eff:9.3f} {r['bottleneck']:>18s}{marker}")

    # Also sweep ffn_dim
    print(f"\n  Kernel Cliff Test: ffn_dim sweep around 8192")
    print("  " + "-" * 60)

    base2 = {
        "hidden_dim": 2048, "num_heads": 16, "num_kv_heads": 2,
        "head_dim": 128, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    }

    ffn_dims = list(range(7936, 8448, 64))
    results2 = model.sweep_dimension(base2, "ffn_dim", ffn_dims)

    print(f"  {'ffn_dim':>10s} {'total_ms':>9s} {'aligned':>8s} {'tile_eff':>9s}")
    print("  " + "-" * 60)
    for r in results2:
        eff = _tile_efficiency(r["ffn_dim"])
        aligned = "yes" if r["aligned_128"] else "NO"
        marker = " ***" if not r["aligned_128"] else ""
        print(f"  {r['ffn_dim']:10d} {r['total_ms']:9.3f} {aligned:>8s} {eff:9.3f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS architecture latency experiment")
    parser.add_argument("--hardware", default="a100", choices=["a100", "t4", "h100"],
                        help="Target GPU for predictions (default: a100)")
    args = parser.parse_args()

    model = ArchitectureCostModel(hardware=args.hardware)

    run_comparison(model)
    run_kernel_cliff_test(model)

    print(f"\n{'='*90}")
    print("  Key takeaways:")
    print("  - MLP (GeGLU) dominates for large models; attention dominates for long seq + many heads")
    print("  - Tile-unaligned dims (not multiple of 128) pay a measurable penalty")
    print("  - GQA (low num_kv_heads) saves attention time but QKV projection is still large")
    print("  - NAS should prefer hidden_dim, ffn_dim that are multiples of 128 (or 256)")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
