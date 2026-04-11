"""Noeris Modal smoke + scaled validation runner.

Runs bounded Modal experiments to validate the shipped kernels end-to-end:

  1. Fused QK-RMSNorm+RoPE across all 6 Gemma 3/4 shape buckets × 5 curated
     configs (A100 and/or H100). Reports GB/s and fusion_speedup vs vLLM-style
     separated baseline.
  2. Upstream KernelBench L1 problems via real Noeris adapters (cuda_event +
     L2 flush timer, fp32 upstream inputs, apples-to-apples speedup).

Usage:
    python scripts/smoke_modal.py                 # quick smoke (1 shape × 1 config, 2 problems) ~$0.04
    python scripts/smoke_modal.py --full          # full sweep A100 only ~$0.15
    python scripts/smoke_modal.py --full --h100   # full sweep A100+H100 ~$0.40
    python scripts/smoke_modal.py --qk-only       # just the fused kernel experiment
    python scripts/smoke_modal.py --upstream-only # just the upstream L1 experiment

Cost is dominated by the session container uptime (~$2.10/hr A100,
~$3.90/hr H100). Scripts emit JSON to stdout; a compact summary is
printed at the end. Results get committed if --write-results is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from research_engine.triton_qk_norm_rope import (
    QK_NORM_ROPE_SHAPE_BUCKETS,
    generate_qk_norm_rope_benchmark_script,
)
from research_engine.kernelbench_upstream import (
    UPSTREAM_PROBLEMS,
    generate_kernelbench_upstream_script,
)
from research_engine.modal_session import ModalBenchmarkSession
from research_engine.timing_snippet import install_noeris_timing


QK_SMOKE_CONFIGS = [
    {"BLOCK_SIZE": 32,  "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 256, "num_warps": 8, "num_stages": 2},
]


def _render_qk_result(payload: dict[str, Any]) -> list[dict]:
    rows = []
    for cfg in payload.get("config_results", []):
        for r in cfg.get("results", []):
            rows.append({
                "config_id": cfg["config_id"],
                "shape_name": r.get("shape_name"),
                "correct": r.get("correct"),
                "ms": r.get("ms"),
                "gb_per_s": r.get("gb_per_s"),
                "fusion_speedup": r.get("fusion_speedup"),
                "max_err": r.get("max_err"),
            })
    return rows


def _render_upstream_result(payload: dict[str, Any]) -> list[dict]:
    return [
        {
            "problem_file": r.get("problem_file"),
            "upstream_ms": r.get("upstream_ms"),
            "noeris_ms": r.get("noeris_ms"),
            "speedup": r.get("speedup"),
            "correct": r.get("correct"),
            "error": r.get("error"),
        }
        for r in payload.get("upstream_results", [])
    ]


def _print_qk_table(rows: list[dict], header: str) -> None:
    print(f"\n### {header}")
    if not rows:
        print("  (no results)")
        return
    print(f"  {'config':20s} {'shape':22s} {'ok':>3s} {'gb/s':>9s} {'ms':>10s} {'fusion_speedup':>16s}")
    for r in rows:
        ok = "✓" if r["correct"] else "✗"
        gb = f"{r['gb_per_s']:.1f}" if r["gb_per_s"] is not None else "—"
        ms = f"{r['ms']:.4f}" if r["ms"] is not None else "—"
        fs = f"{r['fusion_speedup']:.3f}x" if r.get("fusion_speedup") is not None else "—"
        print(f"  {r['config_id']:20s} {r['shape_name'] or '—':22s} {ok:>3s} {gb:>9s} {ms:>10s} {fs:>16s}")


def _print_upstream_table(rows: list[dict], header: str) -> None:
    print(f"\n### {header}")
    if not rows:
        print("  (no results)")
        return
    print(f"  {'problem':45s} {'ok':>3s} {'upstream_ms':>12s} {'noeris_ms':>12s} {'speedup':>10s}")
    for r in rows:
        ok = "✓" if r["correct"] else "✗"
        up = f"{r['upstream_ms']:.3f}" if r["upstream_ms"] is not None else "—"
        nr = f"{r['noeris_ms']:.3f}" if r["noeris_ms"] is not None else "—"
        sp = f"{r['speedup']:.3f}x" if r["speedup"] is not None else "—"
        print(f"  {(r['problem_file'] or '—')[:45]:45s} {ok:>3s} {up:>12s} {nr:>12s} {sp:>10s}")
        if r.get("error"):
            print(f"    ! {r['error'][:200]}")


def run_qk(gpu: str, shapes: list[dict], configs: list[dict]) -> list[dict]:
    script = generate_qk_norm_rope_benchmark_script(configs, shapes)
    script = install_noeris_timing(script, timer="cuda_event")
    with ModalBenchmarkSession(gpu=gpu, timeout_seconds=420) as session:
        result = session.run_script(script)
    if not result.success:
        print(f"  FAILED: {(result.error or result.stderr)[:400]}", file=sys.stderr)
        return []
    start = result.stdout.find("{")
    if start < 0:
        return []
    payload = json.loads(result.stdout[start:])
    return _render_qk_result(payload)


def run_upstream(gpu: str, problems: list) -> list[dict]:
    script = generate_kernelbench_upstream_script(problems)
    script = install_noeris_timing(script, timer="cuda_event")
    with ModalBenchmarkSession(gpu=gpu, timeout_seconds=900) as session:
        result = session.run_script(script)
    if not result.success:
        print(f"  FAILED: {(result.error or result.stderr)[:400]}", file=sys.stderr)
        return []
    start = result.stdout.find("{")
    if start < 0:
        return []
    payload = json.loads(result.stdout[start:])
    return _render_upstream_result(payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full sweep (6 shapes × 5 configs, all 12 L1 problems)")
    ap.add_argument("--h100", action="store_true", help="also run on H100")
    ap.add_argument("--qk-only", action="store_true", help="only the fused kernel experiment")
    ap.add_argument("--upstream-only", action="store_true", help="only upstream L1")
    ap.add_argument("--write-results", action="store_true", help="write results to docs/results/")
    args = ap.parse_args()

    if args.full:
        qk_shapes = QK_NORM_ROPE_SHAPE_BUCKETS
        qk_configs = QK_SMOKE_CONFIGS
        upstream_problems = UPSTREAM_PROBLEMS
    else:
        qk_shapes = [b for b in QK_NORM_ROPE_SHAPE_BUCKETS if b["name"] == "gemma4_e2b_local"][:1]
        qk_configs = QK_SMOKE_CONFIGS[2:3]  # 1 config
        upstream_problems = [
            p for p in UPSTREAM_PROBLEMS
            if p.problem_file in (
                "7_Matmul_with_small_K_dimension_.py",
                "95_CrossEntropyLoss.py",
            )
        ]

    gpus = ["A100"]
    if args.h100:
        gpus.append("H100")

    results = {"qk": {}, "upstream": {}}

    if not args.upstream_only:
        print(f"\n=== Fused QK-RMSNorm+RoPE ({len(qk_shapes)} shapes × {len(qk_configs)} configs) ===")
        for gpu in gpus:
            rows = run_qk(gpu, qk_shapes, qk_configs)
            results["qk"][gpu] = rows
            _print_qk_table(rows, f"QK-norm+RoPE on {gpu}")

    if not args.qk_only:
        print(f"\n=== Upstream KernelBench L1 ({len(upstream_problems)} problems) ===")
        for gpu in gpus:
            rows = run_upstream(gpu, upstream_problems)
            results["upstream"][gpu] = rows
            _print_upstream_table(rows, f"Upstream L1 on {gpu}")

    if args.write_results:
        out_dir = REPO / "docs" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "full" if args.full else "smoke"
        for gpu, rows in results["qk"].items():
            (out_dir / f"qk-norm-rope-{gpu.lower()}-{suffix}.json").write_text(
                json.dumps({"gpu": gpu, "shapes": len(qk_shapes), "configs": len(qk_configs), "results": rows}, indent=2)
            )
        for gpu, rows in results["upstream"].items():
            (out_dir / f"upstream-l1-{gpu.lower()}-{suffix}.json").write_text(
                json.dumps({"gpu": gpu, "problems": len(upstream_problems), "results": rows}, indent=2)
            )
        print(f"\nResults written to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
