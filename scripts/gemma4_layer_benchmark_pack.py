#!/usr/bin/env python3
"""Run canonical Gemma4 deeper-fusion benchmark pack on Modal.

Produces a single canonical JSON + markdown summary containing A100 and H100
results for:
- gemma4_31b_local
- gemma4_31b_global
- gemma4_e2b_local
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

from research_engine.gemma4_layer_benchmark import generate_gemma4_layer_benchmark_script
from research_engine.modal_session import ModalBenchmarkSession


DEFAULT_JSON = "docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json"
DEFAULT_MD = "docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md"


def _run_one_gpu(
    *,
    gpu: str,
    timeout_seconds: int,
    max_cost_usd: float,
    source_dir: Path,
) -> dict:
    script = generate_gemma4_layer_benchmark_script()
    with ModalBenchmarkSession(
        gpu=gpu,
        timeout_seconds=timeout_seconds,
        max_cost_usd=max_cost_usd,
        local_source_dir=str(source_dir.resolve()),
    ) as session:
        result = session.run_script(script)
    if not result.success:
        raise RuntimeError(f"{gpu} benchmark failed: {result.error}")
    layer_results = result.extra.get("layer_results", result.config_results)
    return {
        "hardware": result.hardware,
        "layer_results": layer_results,
    }


def _render_markdown(pack: dict) -> str:
    lines: list[str] = []
    lines.append("# Gemma4 Deeper-Fusion Canonical Pack")
    lines.append("")
    lines.append(f"Generated: {pack['generated_at_utc']}")
    lines.append("")
    lines.append("| GPU | 31b_local | 31b_global | e2b_local | Correctness |")
    lines.append("|---|---:|---:|---:|---|")

    for gpu in pack["gpus"]:
        layer_map = {r["name"]: r for r in pack["results"][gpu]["layer_results"]}
        c0 = layer_map["gemma4_31b_local"]["correct"]
        c1 = layer_map["gemma4_31b_global"]["correct"]
        c2 = layer_map["gemma4_e2b_local"]["correct"]
        all_ok = "all true" if c0 and c1 and c2 else "mixed"
        lines.append(
            f"| {gpu} | {layer_map['gemma4_31b_local']['layer_speedup']:.4f}x | "
            f"{layer_map['gemma4_31b_global']['layer_speedup']:.4f}x | "
            f"{layer_map['gemma4_e2b_local']['layer_speedup']:.4f}x | {all_ok} |"
        )

    lines.append("")
    lines.append("Source script: `research_engine.gemma4_layer_benchmark.generate_gemma4_layer_benchmark_script()`")
    lines.append("")
    lines.append(f"JSON artifact: `{pack['json_path']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="A100,H100", help="Comma-separated GPU list")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_MD)
    parser.add_argument("--source-dir", default="src/research_engine")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    source_dir = Path(args.source_dir)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    results: dict[str, dict] = {}
    for gpu in gpus:
        print(f"Running canonical Gemma4 layer benchmark on {gpu}...")
        results[gpu] = _run_one_gpu(
            gpu=gpu,
            timeout_seconds=args.timeout_seconds,
            max_cost_usd=args.max_cost_usd,
            source_dir=source_dir,
        )

    pack = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "json_path": str(out_json),
        "results": results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")

    md_text = _render_markdown(pack)
    out_md.write_text(md_text, encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
