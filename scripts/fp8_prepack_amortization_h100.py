#!/usr/bin/env python3
"""Estimate FP8 B-prepack amortization threshold on H100."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_engine.modal_session import ModalBenchmarkSession


SCRIPT = r'''
import json
import platform
import torch

SHAPES = [
    {"name": "fp8_mm_1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "fp8_mm_2048x1024x2048", "M": 2048, "N": 1024, "K": 2048},
    {"name": "fp8_mm_4096x4096x4096", "M": 4096, "N": 4096, "K": 4096},
]


def bench_ms(fn, warmup=40, trials=150):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        vals.append(st.elapsed_time(en))
    vals.sort()
    return float(vals[len(vals)//2])


def run_shape(shape):
    K, N = int(shape["K"]), int(shape["N"])
    b_kn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

    def prepack_once():
        _ = b_kn.t().contiguous()

    ms = bench_ms(prepack_once)
    return {
        "shape_name": shape["name"],
        "k": K,
        "n": N,
        "prepack_ms": round(ms, 4),
    }


rows = [run_shape(shape) for shape in SHAPES]
print(json.dumps({
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda or "unknown",
        "python": platform.python_version(),
    },
    "config_results": [],
    "prepack_results": rows,
}, indent=2))
'''


def _best_ms_by_shape(config_results: list[dict], *, layout: str) -> dict[str, float]:
    best: dict[str, float] = {}
    for cfg in config_results:
        cfg_layout = cfg.get("config", {}).get("layout", "")
        if cfg_layout != layout:
            continue
        for row in cfg.get("results", []):
            if not row.get("correct"):
                continue
            shape_name = row.get("shape_name", "")
            ms = float(row.get("ms", 0.0))
            prev = best.get(shape_name)
            if prev is None or ms < prev:
                best[shape_name] = ms
    return best


def _merge_amortization(v3_payload: dict, prepack_rows: list[dict]) -> list[dict]:
    kn_best_ms = _best_ms_by_shape(v3_payload.get("config_results", []), layout="kn")
    nk_best_ms = _best_ms_by_shape(v3_payload.get("config_results", []), layout="nk")
    prepack_ms = {row.get("shape_name", ""): float(row.get("prepack_ms", 0.0)) for row in prepack_rows}

    out = []
    for shape_name in sorted(prepack_ms.keys()):
        kn = kn_best_ms.get(shape_name)
        nk = nk_best_ms.get(shape_name)
        pack = prepack_ms.get(shape_name, 0.0)
        delta = (kn - nk) if (kn is not None and nk is not None) else None
        if delta is None or delta <= 0.0:
            break_even = None
        else:
            break_even = int(math.ceil(pack / delta))
        out.append(
            {
                "shape_name": shape_name,
                "kn_ms": round(kn, 4) if kn is not None else None,
                "nk_ms": round(nk, 4) if nk is not None else None,
                "delta_ms_per_run": round(delta, 4) if delta is not None else None,
                "prepack_ms": round(pack, 4),
                "break_even_runs": break_even,
            }
        )
    return out


def _to_md(payload: dict) -> str:
    lines = [
        "# FP8 Prepack Amortization (H100)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Shape | best kn ms | best nk ms | delta/run ms | prepack ms | break-even runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["amortization"]:
        break_even = row["break_even_runs"]
        break_even_str = str(break_even) if break_even is not None else "n/a"
        lines.append(
            f"| {row['shape_name']} | {row['kn_ms']} | {row['nk_ms']} | {row['delta_ms_per_run']} | {row['prepack_ms']} | {break_even_str} |"
        )
    lines += [
        "",
        "Break-even runs means how many repeated matmuls with the same B are needed",
        "for prepacked `nk` to recover one-time transpose+pack cost.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--v3-json", default="docs/results/fp8-triton-matmul-autotune-h100-v3.json")
    parser.add_argument("--output-json", default="docs/results/fp8-prepack-amortization-h100.json")
    parser.add_argument("--output-md", default="docs/results/fp8-prepack-amortization-h100.md")
    args = parser.parse_args()

    v3_path = ROOT / args.v3_json
    v3_payload = json.loads(v3_path.read_text(encoding="utf-8"))

    with ModalBenchmarkSession(
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        local_source_dir=str((ROOT / "src/research_engine").resolve()),
    ) as session:
        out = session.run_script(SCRIPT)
    if not out.success:
        raise RuntimeError(f"FP8 prepack amortization probe failed: {out.error}")

    prepack_rows = out.extra.get("prepack_results", [])
    amortization = _merge_amortization(v3_payload, prepack_rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": out.hardware,
        "source_v3": args.v3_json,
        "prepack_results": prepack_rows,
        "amortization": amortization,
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
