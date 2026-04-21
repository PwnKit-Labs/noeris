#!/usr/bin/env python3
"""Benchmark runtime-hook integration for verify+accept path."""

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
import traceback
import torch

from research_engine.spec_decode_runtime import verify_accept_from_logits

SHAPES = [
    {"name": "draft4_vocab32k", "batch": 1, "draft_len": 4, "vocab": 32000},
    {"name": "draft8_vocab32k", "batch": 1, "draft_len": 8, "vocab": 32000},
    {"name": "draft16_vocab32k", "batch": 1, "draft_len": 16, "vocab": 32000},
    {"name": "draft16_vocab128k", "batch": 1, "draft_len": 16, "vocab": 128000},
    {"name": "draft32_vocab128k", "batch": 1, "draft_len": 32, "vocab": 128000},
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
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        vals.append(st.elapsed_time(en))
    vals.sort()
    return float(vals[len(vals)//2])


def run_shape(shape):
    try:
        b = int(shape["batch"])
        l = int(shape["draft_len"])
        v = int(shape["vocab"])
        torch.manual_seed(0)
        logits = torch.randn((b, l, v), device="cuda", dtype=torch.float16)
        draft = torch.randint(0, v, (b, l), device="cuda", dtype=torch.int64)

        # correctness and backend checks
        s_len, s_mask, s_backend = verify_accept_from_logits(logits, draft, prefer="separated")
        f_len, f_mask, f_backend = verify_accept_from_logits(logits, draft, prefer="fused")
        a_len, a_mask, a_backend = verify_accept_from_logits(logits, draft, prefer="auto")

        correct = bool(torch.equal(s_len, f_len) and torch.equal(s_mask, f_mask) and torch.equal(s_len, a_len) and torch.equal(s_mask, a_mask))

        sep_ms = median_ms(lambda: verify_accept_from_logits(logits, draft, prefer="separated"))
        fused_ms = median_ms(lambda: verify_accept_from_logits(logits, draft, prefer="fused"))
        auto_ms = median_ms(lambda: verify_accept_from_logits(logits, draft, prefer="auto"))

        return {
            "shape_name": shape["name"],
            "correct": correct,
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
    except Exception as exc:  # noqa: BLE001
        return {
            "shape_name": shape.get("name", "unknown"),
            "correct": False,
            "error": str(exc),
            "traceback": traceback.format_exc()[-1200:],
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
        "# Spec Decode Verify+Accept Runtime Integration",
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
    lines += ["", "Auto backend should select fused in this environment (no flashinfer callback).", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="A100,H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--output-json", default="docs/results/spec-decode-verify-accept-runtime-integration.json")
    parser.add_argument("--output-md", default="docs/results/spec-decode-verify-accept-runtime-integration.md")
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
