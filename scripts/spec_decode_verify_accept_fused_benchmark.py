#!/usr/bin/env python3
"""Run first Triton fused verify+accept benchmark on Modal."""

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
from research_engine.triton_spec_decode_verify_accept import (
    VERIFY_ACCEPT_CURATED_CONFIGS,
    VERIFY_ACCEPT_SHAPE_BUCKETS,
    generate_verify_accept_benchmark_script,
)


def _summarize(config_results: list[dict]) -> dict:
    best_by_shape: dict[str, dict] = {}
    for cfg in config_results:
        for row in cfg.get("results", []):
            if not row.get("correct"):
                continue
            name = row.get("shape_name", "")
            prev = best_by_shape.get(name)
            if prev is None or float(row.get("speedup", 0.0)) > float(prev.get("speedup", 0.0)):
                best_by_shape[name] = {
                    "config_id": cfg.get("config_id", ""),
                    "speedup": float(row.get("speedup", 0.0)),
                    "fused_ms": float(row.get("ms", 0.0)),
                    "separated_ms": float(row.get("separated_ms", 0.0)),
                    "tokens_per_ms": float(row.get("tflops", 0.0)),
                    "correct": bool(row.get("correct", False)),
                }
    return best_by_shape


def _to_md(payload: dict) -> str:
    lines = [
        "# Spec Decode Verify+Accept Fused v1",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| GPU | Shape | best speedup | fused ms | separated ms | config |",
        "|---|---|---:|---:|---:|---|",
    ]
    for gpu in payload["gpus"]:
        best = payload["results"][gpu]["best_by_shape"]
        for shape in sorted(best.keys()):
            row = best[shape]
            lines.append(
                f"| {gpu} | {shape} | {row['speedup']:.4f}x | {row['fused_ms']:.4f} | "
                f"{row['separated_ms']:.4f} | {row['config_id']} |"
            )
    lines += [
        "",
        "Fused path includes token match, first mismatch detection, and accepted-prefix mask write.",
        "Argmax over logits remains outside this v1 kernel.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="A100,H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--output-json", default="docs/results/spec-decode-verify-accept-fused-v1.json")
    parser.add_argument("--output-md", default="docs/results/spec-decode-verify-accept-fused-v1.md")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    script = generate_verify_accept_benchmark_script(VERIFY_ACCEPT_CURATED_CONFIGS, VERIFY_ACCEPT_SHAPE_BUCKETS)

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
            raise RuntimeError(f"{gpu} fused benchmark failed: {out.error}")
        best = _summarize(out.config_results)
        results[gpu] = {
            "hardware": out.hardware,
            "config_results": out.config_results,
            "best_by_shape": best,
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
