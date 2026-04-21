#!/usr/bin/env python3
"""Benchmark runtime-hook integration for KV quantize-on-write."""

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


SCRIPT = r'''
import json
import platform
import torch

from research_engine.kv_cache_quant_runtime import quantize_kv_pair_write

SHAPES = [
    {"name": "b1_kv16_d256_t1", "rows": 16, "head_dim": 256},
    {"name": "b1_kv4_d512_t1", "rows": 4, "head_dim": 512},
    {"name": "b4_kv16_d256_t1", "rows": 64, "head_dim": 256},
    {"name": "b1_kv16_d256_t8", "rows": 128, "head_dim": 256},
]


def median_ms(fn, warmup=25, trials=120):
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
    rows = int(shape["rows"])
    d = int(shape["head_dim"])
    torch.manual_seed(0)
    k = torch.randn((rows, d), device="cuda", dtype=torch.float16)
    v = torch.randn((rows, d), device="cuda", dtype=torch.float16)

    sq, ss, vq, vs, s_backend = quantize_kv_pair_write(k, v, prefer="separated")
    fq, fs, fvq, fvs, f_backend = quantize_kv_pair_write(k, v, prefer="fused")
    aq, ass, avq, avs, a_backend = quantize_kv_pair_write(k, v, prefer="auto")

    # Compare in dequantized space for correctness.
    s_k_deq = sq.to(torch.float32) * ss.unsqueeze(1)
    f_k_deq = fq.to(torch.float32) * fs.unsqueeze(1)
    s_v_deq = vq.to(torch.float32) * vs.unsqueeze(1)
    f_v_deq = fvq.to(torch.float32) * fvs.unsqueeze(1)
    max_err = max((s_k_deq - f_k_deq).abs().max().item(), (s_v_deq - f_v_deq).abs().max().item())
    correct = bool(max_err <= 1.5)

    sep_ms = median_ms(lambda: quantize_kv_pair_write(k, v, prefer="separated"))
    fused_ms = median_ms(lambda: quantize_kv_pair_write(k, v, prefer="fused"))
    auto_ms = median_ms(lambda: quantize_kv_pair_write(k, v, prefer="auto"))

    return {
        "shape_name": shape["name"],
        "correct": correct,
        "max_err": round(max_err, 6),
        "separated_ms": round(sep_ms, 4),
        "fused_ms": round(fused_ms, 4),
        "auto_ms": round(auto_ms, 4),
        "fused_speedup": round(sep_ms / fused_ms, 4) if fused_ms > 0 else 0.0,
        "auto_speedup": round(sep_ms / auto_ms, 4) if auto_ms > 0 else 0.0,
        "backends": {
            "separated": s_backend,
            "fused": f_backend,
            "auto": a_backend,
        },
    }


def main():
    rows = [run_shape(s) for s in SHAPES]
    print(json.dumps({
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        },
        "config_results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
'''


def _to_md(payload: dict) -> str:
    lines = [
        "# KV Quantize-on-Write Runtime Integration",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| GPU | Shape | separated ms | fused ms | auto ms | fused speedup | auto backend |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for gpu in payload["gpus"]:
        for row in payload["results"][gpu]["config_results"]:
            lines.append(
                f"| {gpu} | {row['shape_name']} | {row['separated_ms']:.4f} | {row['fused_ms']:.4f} | "
                f"{row['auto_ms']:.4f} | {row['fused_speedup']:.4f}x | {row['backends']['auto']} |"
            )
    lines += ["", "Auto backend should select fused in this environment (no external callback).", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="A100,H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--output-json", default="docs/results/kv-quant-write-runtime-integration.json")
    parser.add_argument("--output-md", default="docs/results/kv-quant-write-runtime-integration.md")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    results: dict[str, dict] = {}
    for gpu in gpus:
        with ModalBenchmarkSession(
            gpu=gpu,
            timeout_seconds=args.timeout_seconds,
            max_cost_usd=args.max_cost_usd,
            local_source_dir=str((ROOT / "src/research_engine").resolve()),
        ) as session:
            out = session.run_script(SCRIPT)
        if not out.success:
            raise RuntimeError(f"{gpu} runtime benchmark failed: {out.error}")
        results[gpu] = {
            "hardware": out.hardware,
            "config_results": out.config_results,
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
