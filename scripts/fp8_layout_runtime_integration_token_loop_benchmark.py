#!/usr/bin/env python3
"""Benchmark FP8 runtime integration using repeated token-loop dispatch."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_engine.executors import MatmulPythonExecutor
from research_engine.models import ExperimentSpec, ResearchTopic


MODES = [
    {
        "mode": "auto_no_cache",
        "layout_preference": "auto",
        "prepack_cache_enabled": False,
    },
    {
        "mode": "auto_with_cache",
        "layout_preference": "auto",
        "prepack_cache_enabled": True,
    },
    {
        "mode": "force_kn_no_cache",
        "layout_preference": "kn",
        "prepack_cache_enabled": False,
    },
]


def _run_mode(*, mode: dict[str, object], repetitions: int, token_loop_iterations: int) -> dict[str, object]:
    executor = MatmulPythonExecutor(
        repetitions=max(1, repetitions),
        warmup_repetitions=0,
        max_candidates_per_run=3,
        fp8_layout_preference=str(mode["layout_preference"]),
        fp8_prepack_cache_enabled=bool(mode["prepack_cache_enabled"]),
        fp8_runtime_token_loop_iterations=max(1, token_loop_iterations),
    )
    results = executor.run(
        ResearchTopic(
            name="matrix multiplication speedup",
            objective="measure fp8 runtime integration behavior",
            benchmark_id="matmul-speedup",
        ),
        [
            ExperimentSpec(
                name=f"runtime-{mode['mode']}",
                benchmark_id="matmul-speedup",
                hypothesis_title="FP8 runtime dispatch mode comparison",
                success_metric="runtime dispatch latency",
                budget="small",
                baseline="naive_ijk",
                protocol=["run"],
            )
        ],
    )
    payloads = results[0].artifact_payloads
    fp8_summary = payloads.get("fp8-runtime-layout-summary.json", {})
    return {
        "mode": str(mode["mode"]),
        "layout_preference": str(mode["layout_preference"]),
        "prepack_cache_enabled": bool(mode["prepack_cache_enabled"]),
        "fp8_fixture_count": int(fp8_summary.get("fp8_fixture_count", 0) or 0),
        "layout_counts": fp8_summary.get("layout_counts", {}),
        "runtime_dispatch_total_ms": float(fp8_summary.get("runtime_dispatch_total_ms", 0.0) or 0.0),
        "runtime_dispatch_avg_ms_per_token": float(
            fp8_summary.get("runtime_dispatch_avg_ms_per_token", 0.0) or 0.0
        ),
        "runtime_token_iterations": int(fp8_summary.get("runtime_token_iterations", 0) or 0),
        "runtime_prepack_ops": int(fp8_summary.get("runtime_prepack_ops", 0) or 0),
        "runtime_cache_hits": int(fp8_summary.get("runtime_cache_hits", 0) or 0),
        "runtime_cache_misses": int(fp8_summary.get("runtime_cache_misses", 0) or 0),
        "runtime_cache_evictions": int(fp8_summary.get("runtime_cache_evictions", 0) or 0),
        "runtime_cache_hit_rate": float(fp8_summary.get("runtime_cache_hit_rate", 0.0) or 0.0),
    }


def _to_md(payload: dict) -> str:
    lines = [
        "# FP8 Runtime Integration Token-Loop Benchmark",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Mode | cache | dispatch total ms | avg ms/token | prepack ops | cache hits | cache misses | vs baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["mode_results"]:
        lines.append(
            f"| {row['mode']} | {str(row['prepack_cache_enabled']).lower()} | {row['runtime_dispatch_total_ms']:.4f} | "
            f"{row['runtime_dispatch_avg_ms_per_token']:.4f} | {row['runtime_prepack_ops']} | "
            f"{row['runtime_cache_hits']} | {row['runtime_cache_misses']} | {row['vs_baseline_dispatch_total']:.4f} |"
        )
    lines += [
        "",
        "`auto_with_cache` should show cache hits during repeated FP8 token-loop dispatch.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--token-loop-iterations", type=int, default=48)
    parser.add_argument(
        "--output-json",
        default="docs/results/fp8-layout-runtime-integration-token-loop.json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/results/fp8-layout-runtime-integration-token-loop.md",
    )
    args = parser.parse_args()

    mode_results = [
        _run_mode(
            mode=mode,
            repetitions=args.repetitions,
            token_loop_iterations=args.token_loop_iterations,
        )
        for mode in MODES
    ]
    baseline_total = max(1e-9, float(mode_results[0]["runtime_dispatch_total_ms"]))
    for row in mode_results:
        row["vs_baseline_dispatch_total"] = round(float(row["runtime_dispatch_total_ms"]) / baseline_total, 4)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
        },
        "token_loop_iterations": max(1, args.token_loop_iterations),
        "repetitions": max(1, args.repetitions),
        "mode_results": mode_results,
    }

    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_to_md(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
