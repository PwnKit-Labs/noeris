#!/usr/bin/env python3
"""Emit FP8 config packs for matmul and grouped-gemm autotune lanes."""

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

from research_engine.fp8_config_space import (  # noqa: E402
    build_fp8_grouped_gemm_config_space,
    build_fp8_matmul_config_space,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="docs/results/fp8-config-space-pack.json")
    args = parser.parse_args()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matmul": {
            "count": 0,
            "configs": [],
        },
        "grouped_gemm": {
            "count": 0,
            "configs": [],
        },
    }
    payload["matmul"]["configs"] = build_fp8_matmul_config_space()
    payload["matmul"]["count"] = len(payload["matmul"]["configs"])
    payload["grouped_gemm"]["configs"] = build_fp8_grouped_gemm_config_space()
    payload["grouped_gemm"]["count"] = len(payload["grouped_gemm"]["configs"])

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
