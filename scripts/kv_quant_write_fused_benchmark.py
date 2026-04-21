#!/usr/bin/env python3
"""Run KV quantize-on-write fused benchmark on Modal."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_engine.modal_session import ModalBenchmarkSession
from research_engine.triton_kv_quant_write import (
    KV_QUANT_WRITE_CURATED_CONFIGS,
    KV_QUANT_WRITE_SHAPE_BUCKETS,
    generate_kv_quant_write_benchmark_script,
)


def _summarize(config_results: list[dict]) -> dict:
    best_by_shape: dict[str, dict] = {}
    for cfg in config_results:
        for row in cfg.get("results", []):
            if not row.get("correct"):
                continue
            shape = row.get("shape_name", "")
            prev = best_by_shape.get(shape)
            if prev is None or float(row.get("speedup", 0.0)) > float(prev.get("speedup", 0.0)):
                best_by_shape[shape] = {
                    "config_id": cfg.get("config_id", ""),
                    "speedup": float(row.get("speedup", 0.0)),
                    "fused_ms": float(row.get("ms", 0.0)),
                    "separated_ms": float(row.get("separated_ms", 0.0)),
                    "max_err": float(row.get("max_err", 0.0)),
                }
    return best_by_shape


def _to_md(payload: dict) -> str:
    lines = [
        "# KV Quantize-on-Write Fused v1",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| GPU | Shape | best speedup | fused ms | separated ms | max err | config |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for gpu in payload["gpus"]:
        best = payload["results"][gpu]["best_by_shape"]
        for shape in sorted(best.keys()):
            row = best[shape]
            lines.append(
                f"| {gpu} | {shape} | {row['speedup']:.4f}x | {row['fused_ms']:.4f} | "
                f"{row['separated_ms']:.4f} | {row['max_err']:.6f} | {row['config_id']} |"
            )
    lines += [
        "",
        "Fused kernel computes per-row scale and writes INT8 quantized cache rows in one pass.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="A100,H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--output-json", default="docs/results/kv-quant-write-fused-v1.json")
    parser.add_argument("--output-md", default="docs/results/kv-quant-write-fused-v1.md")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    script = generate_kv_quant_write_benchmark_script(
        KV_QUANT_WRITE_CURATED_CONFIGS,
        KV_QUANT_WRITE_SHAPE_BUCKETS,
    )
    results: dict[str, dict] = {}

    for gpu in gpus:
        with ModalBenchmarkSession(
            gpu=gpu,
            timeout_seconds=args.timeout_seconds,
            max_cost_usd=args.max_cost_usd,
            local_source_dir=str((ROOT / "src/research_engine").resolve()),
        ) as session:
            out = session.run_script(script)
        if not out.success:
            raise RuntimeError(f"{gpu} kv-quant benchmark failed: {out.error}")
        results[gpu] = {
            "hardware": out.hardware,
            "config_results": out.config_results,
            "best_by_shape": _summarize(out.config_results),
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "results": results,
    }
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
