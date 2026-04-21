#!/usr/bin/env python3
"""Baseline benchmark for speculative decoding verify+accept path.

This establishes a reproducible latency baseline for the current non-fused
verify/accept implementation (argmax + compare + prefix-length extraction).
"""

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


BENCH_SCRIPT = r'''
import json
import platform
import torch

SHAPES = [
    {"name": "draft4_vocab32k", "batch": 1, "draft_len": 4, "vocab": 32000},
    {"name": "draft8_vocab32k", "batch": 1, "draft_len": 8, "vocab": 32000},
    {"name": "draft16_vocab32k", "batch": 1, "draft_len": 16, "vocab": 32000},
    {"name": "draft16_vocab128k", "batch": 1, "draft_len": 16, "vocab": 128000},
    {"name": "draft32_vocab128k", "batch": 1, "draft_len": 32, "vocab": 128000},
]


def _median_cuda_ms(fn, warmup=25, trials=120):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return float(times[len(times) // 2])


def verify_accept_separated(target_logits, draft_tokens):
    # target_logits: [B, L, V], draft_tokens: [B, L]
    target_tokens = torch.argmax(target_logits, dim=-1)
    match = (target_tokens == draft_tokens)
    mismatch = ~match
    # First mismatch index per batch row; if all match, accept full L.
    first_mismatch = torch.argmax(mismatch.to(torch.int32), dim=-1)
    all_match = match.all(dim=-1)
    accept_len = torch.where(all_match, torch.full_like(first_mismatch, match.shape[1]), first_mismatch)
    # Build accepted-prefix mask [B, L] to model downstream mask usage.
    idx = torch.arange(match.shape[1], device=match.device).unsqueeze(0)
    accepted_prefix_mask = idx < accept_len.unsqueeze(-1)
    return accept_len, accepted_prefix_mask


def run_shape(shape):
    b = int(shape["batch"])
    l = int(shape["draft_len"])
    v = int(shape["vocab"])

    torch.manual_seed(0)
    logits = torch.randn((b, l, v), device="cuda", dtype=torch.float16)
    draft = torch.randint(0, v, (b, l), device="cuda", dtype=torch.int64)

    # Correctness sanity: returned lengths within [0, L].
    accept_len, mask = verify_accept_separated(logits, draft)
    correct = bool(
        (accept_len >= 0).all().item()
        and (accept_len <= l).all().item()
        and mask.shape == (b, l)
    )

    ms = _median_cuda_ms(lambda: verify_accept_separated(logits, draft))
    tokens_per_ms = (b * l) / ms if ms > 0 else 0.0

    return {
        "shape_name": shape["name"],
        "batch": b,
        "draft_len": l,
        "vocab": v,
        "correct": correct,
        "separated_ms": round(ms, 4),
        "tokens_per_ms": round(tokens_per_ms, 4),
    }


def main():
    results = [run_shape(s) for s in SHAPES]
    print(json.dumps({
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        },
        "config_results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
'''


def _to_md(payload: dict) -> str:
    lines = [
        "# Spec Decode Verify+Accept Baseline",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| GPU | Shape | ms | tokens/ms | correct |",
        "|---|---|---:|---:|---|",
    ]
    for gpu in payload["gpus"]:
        for row in payload["results"][gpu]["config_results"]:
            lines.append(
                f"| {gpu} | {row['shape_name']} | {row['separated_ms']:.4f} | "
                f"{row['tokens_per_ms']:.4f} | {str(row['correct']).lower()} |"
            )
    lines += [
        "",
        "Baseline path: argmax(target logits) + draft compare + prefix-accept mask extraction.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="A100,H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--output-json", default="docs/results/spec-decode-verify-accept-baseline.json")
    parser.add_argument("--output-md", default="docs/results/spec-decode-verify-accept-baseline.md")
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
            out = session.run_script(BENCH_SCRIPT)
        if not out.success:
            raise RuntimeError(f"{gpu} benchmark failed: {out.error}")
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
