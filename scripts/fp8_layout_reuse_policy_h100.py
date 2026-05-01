#!/usr/bin/env python3
"""Derive FP8 layout policy by expected weight reuse count on H100."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _best_ms_by_shape(config_results: list[dict], layout: str) -> dict[str, float]:
    best: dict[str, float] = {}
    for cfg in config_results:
        cfg_layout = cfg.get("config", {}).get("layout", "")
        if cfg_layout != layout:
            continue
        for row in cfg.get("results", []):
            if not row.get("correct"):
                continue
            shape_name = str(row.get("shape_name", ""))
            ms = float(row.get("ms", 0.0))
            prev = best.get(shape_name)
            if prev is None or ms < prev:
                best[shape_name] = ms
    return best


def _policy_rows(v3_payload: dict, prepack_payload: dict, reuse_counts: list[int]) -> list[dict]:
    kn_ms = _best_ms_by_shape(v3_payload.get("config_results", []), layout="kn")
    nk_ms = _best_ms_by_shape(v3_payload.get("config_results", []), layout="nk")
    prepack_ms = {
        str(row.get("shape_name", "")): float(row.get("prepack_ms", 0.0))
        for row in prepack_payload.get("prepack_results", [])
    }

    rows: list[dict] = []
    for shape_name in sorted(prepack_ms.keys()):
        if shape_name not in kn_ms or shape_name not in nk_ms:
            continue
        shape_row: dict[str, object] = {
            "shape_name": shape_name,
            "kn_ms": round(kn_ms[shape_name], 4),
            "nk_ms": round(nk_ms[shape_name], 4),
            "prepack_ms": round(prepack_ms[shape_name], 4),
            "decisions": [],
        }
        for reuse in reuse_counts:
            kn_total = kn_ms[shape_name] * reuse
            nk_total = prepack_ms[shape_name] + nk_ms[shape_name] * reuse
            winner = "nk" if nk_total < kn_total else "kn"
            shape_row["decisions"].append(
                {
                    "reuse_count": reuse,
                    "kn_total_ms": round(kn_total, 4),
                    "nk_total_ms": round(nk_total, 4),
                    "winner": winner,
                }
            )
        rows.append(shape_row)
    return rows


def _to_md(payload: dict) -> str:
    lines = [
        "# FP8 Layout Reuse Policy (H100)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
    ]
    reuse_counts = payload.get("reuse_counts", [])
    header = "| Shape | " + " | ".join([f"reuse={r}" for r in reuse_counts]) + " |"
    sep = "|---|" + "---|" * len(reuse_counts)
    lines.append(header)
    lines.append(sep)
    for row in payload.get("policy", []):
        winners = [d["winner"] for d in row["decisions"]]
        lines.append("| " + row["shape_name"] + " | " + " | ".join(winners) + " |")

    lines += ["", "Detailed effective latencies (ms):", ""]
    for row in payload.get("policy", []):
        lines.append(f"- {row['shape_name']} (kn={row['kn_ms']}, nk={row['nk_ms']}, prepack={row['prepack_ms']})")
        for decision in row["decisions"]:
            lines.append(
                "  "
                + f"reuse={decision['reuse_count']}: kn_total={decision['kn_total_ms']}, "
                + f"nk_total={decision['nk_total_ms']} -> {decision['winner']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-json", default="docs/results/fp8-triton-matmul-autotune-h100-v3.json")
    parser.add_argument("--prepack-json", default="docs/results/fp8-prepack-amortization-h100.json")
    parser.add_argument("--reuse-counts", default="1,2,4,8,16")
    parser.add_argument("--output-json", default="docs/results/fp8-layout-reuse-policy-h100.json")
    parser.add_argument("--output-md", default="docs/results/fp8-layout-reuse-policy-h100.md")
    args = parser.parse_args()

    v3_payload = json.loads((ROOT / args.v3_json).read_text(encoding="utf-8"))
    prepack_payload = json.loads((ROOT / args.prepack_json).read_text(encoding="utf-8"))
    reuse_counts = [int(x.strip()) for x in args.reuse_counts.split(",") if x.strip()]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_v3": args.v3_json,
        "source_prepack": args.prepack_json,
        "reuse_counts": reuse_counts,
        "policy": _policy_rows(v3_payload, prepack_payload, reuse_counts),
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
