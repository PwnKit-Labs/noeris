#!/usr/bin/env python3
"""Cross-vendor zero-shot prediction scaffold.

Trains the existing Noeris cost model on source (NVIDIA) data and emits
predicted top configs for an unseen target hardware label (default: AMD MI300X)
without using any target-device benchmark points.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_engine.cost_model import CostModel
from research_engine.cross_vendor_transfer import collect_source_bucket_candidates


def _default_operators() -> list[str]:
    return ["matmul", "rmsnorm", "softmax", "layernorm", "cross_entropy", "attention"]


def _to_md(report: dict) -> str:
    lines = [
        "# Cross-Vendor Zero-Shot Scaffold",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        f"Source hardware filter: `{report['source_hardware_filter']}`",
        f"Target hardware label: `{report['target_hardware_label']}`",
        "",
        "| Operator | Buckets with candidates | total candidates |",
        "|---|---:|---:|",
    ]
    for op in report["operators"]:
        meta = report["summary"].get(op, {})
        lines.append(
            f"| {op} | {meta.get('bucket_count', 0)} | {meta.get('total_candidates', 0)} |"
        )
    lines += [
        "",
        "This is a scaffold prediction artifact only; no AMD ground-truth measurements are used.",
        "Evaluation can be computed later with scripts/cross_vendor_transfer_eval.py.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=".noeris/triton-configs.json")
    parser.add_argument("--source-hardware-filter", default="A100")
    parser.add_argument("--target-hardware-label", default="AMD MI300X")
    parser.add_argument("--operators", nargs="*", default=_default_operators())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates-per-bucket", type=int, default=40)
    parser.add_argument("--output-json", default="docs/results/cross-vendor-zero-shot-scaffold-mi300x.json")
    parser.add_argument("--output-md", default="docs/results/cross-vendor-zero-shot-scaffold-mi300x.md")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    data = json.loads(db_path.read_text(encoding="utf-8"))
    records = data.get("records", {})

    model = CostModel()
    train_result = model.train_from_databases([db_path])

    predictions: dict[str, dict] = {}
    summary: dict[str, dict] = {}

    for operator in args.operators:
        buckets = collect_source_bucket_candidates(
            records=records,
            operator=operator,
            source_hardware_substr=args.source_hardware_filter,
            max_candidates_per_bucket=args.max_candidates_per_bucket,
        )

        op_pred: dict[str, Any] = {}
        total_candidates = 0
        for bucket, entry in buckets.items():
            total_candidates += len(entry.configs)
            candidates = [row["config"] for row in entry.configs]
            ranked = model.rank_configs(
                configs=candidates,
                shapes=[entry.shape],
                hardware=args.target_hardware_label,
                operator=operator,
                top_k=args.top_k,
            )
            id_by_config = {
                json.dumps(row["config"], sort_keys=True): row.get("config_id", "")
                for row in entry.configs
            }
            ranked_rows = [
                {
                    "config_id": id_by_config.get(json.dumps(cfg, sort_keys=True), ""),
                    "config": cfg,
                    "predicted_metric": float(score),
                }
                for cfg, score in ranked
            ]
            source_top = entry.configs[: args.top_k]
            op_pred[bucket] = {
                "shape": entry.shape,
                "source_top": source_top,
                "target_predicted_top": ranked_rows,
            }

        predictions[operator] = op_pred
        summary[operator] = {
            "bucket_count": len(op_pred),
            "total_candidates": total_candidates,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "training": train_result,
        "source_hardware_filter": args.source_hardware_filter,
        "target_hardware_label": args.target_hardware_label,
        "operators": args.operators,
        "top_k": args.top_k,
        "summary": summary,
        "predictions": predictions,
        "limitations": [
            "No target-hardware measurements are used in this scaffold artifact.",
            "Predicted ranking quality across vendors must be validated on real AMD measurements.",
            "Unknown hardware labels are mapped to the model's unseen-hardware path.",
        ],
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
