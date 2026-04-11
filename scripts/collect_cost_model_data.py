#!/usr/bin/env python3
"""Collect training data for the learned cost model.

Runs grid-heavy benchmarks across all registered operators on a target
GPU, persisting results to a ConfigDatabase that becomes the cost model
training corpus. Uses a persistent Modal session to minimize overhead.

Usage:

    python scripts/collect_cost_model_data.py \
        --gpu A100 \
        --operators rmsnorm softmax layernorm cross_entropy rotary \
        --configs-per-operator 20 \
        --shapes-per-operator 6 \
        --output .noeris/cost-model-training.json

Approximately $0.10-0.30 per operator depending on config count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_engine.modal_session import ModalBenchmarkSession
from research_engine.triton_kernels import ConfigDatabase
from research_engine.triton_operators import REGISTRY


def _parse_shape(operator: str, shape_str: str) -> dict | None:
    parts = shape_str.split("x")
    try:
        if operator == "matmul":
            return {"M": int(parts[0]), "N": int(parts[1]), "K": int(parts[2])}
        if operator in ("rmsnorm", "layernorm"):
            return {"n_rows": int(parts[0]), "hidden_dim": int(parts[1])}
        if operator in ("softmax", "cross_entropy"):
            return {"n_rows": int(parts[0]), "n_cols": int(parts[1])}
        if operator == "attention":
            return {
                "batch": int(parts[0]),
                "heads": int(parts[1]),
                "seq_len": int(parts[2]),
                "head_dim": int(parts[3]),
            }
        if operator == "rotary":
            return {
                "batch": int(parts[0]),
                "seq": int(parts[1]),
                "heads": int(parts[2]),
                "head_dim": int(parts[3]),
            }
    except (ValueError, IndexError):
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="A100")
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["rmsnorm", "softmax", "layernorm", "cross_entropy"],
        help="Which operators to collect data for",
    )
    parser.add_argument(
        "--configs-per-operator",
        type=int,
        default=20,
        help="How many configs to benchmark per operator",
    )
    parser.add_argument(
        "--shapes-per-operator",
        type=int,
        default=6,
        help="How many shape buckets to use per operator",
    )
    parser.add_argument(
        "--output",
        default=".noeris/cost-model-training.json",
        help="Path to the accumulated ConfigDatabase",
    )
    args = parser.parse_args()

    db_path = Path(args.output)
    db = ConfigDatabase(path=db_path)

    print(f"Collecting cost model training data on {args.gpu}")
    print(f"Database: {db_path} (starting with {sum(len(r.results) for r in db.records.values())} points)")
    print(f"Operators: {args.operators}")
    print(f"Configs/op: {args.configs_per_operator}, Shapes/op: {args.shapes_per_operator}")
    print()

    total_added = 0
    with ModalBenchmarkSession(gpu=args.gpu, timeout_seconds=900) as session:
        for op_name in args.operators:
            try:
                spec = REGISTRY.get(op_name)
            except KeyError:
                print(f"[{op_name}] unknown operator, skipping")
                continue

            shapes = spec.shape_buckets[: args.shapes_per_operator]

            # Pull a large systematic grid of configs
            grid = spec.grid_generator_fn(
                include_curated=True,
                max_configs=args.configs_per_operator,
            )[: args.configs_per_operator]

            print(f"[{op_name}] running {len(grid)} configs on {len(shapes)} shapes...")

            script = spec.benchmark_script_fn(grid, shapes)
            batch = session.run_batch(script)

            if not batch.success:
                print(f"[{op_name}] failed: {batch.error[:200]}")
                continue

            hw_name = batch.hardware.get("gpu", args.gpu)
            added = 0
            for config_result in batch.config_results:
                cid = config_result.get("config_id", "")
                config = config_result.get("config", {})
                for sr in config_result.get("results", []):
                    if not sr.get("correct") or not sr.get("tflops"):
                        continue
                    shape = _parse_shape(op_name, sr.get("shape", ""))
                    if shape is None:
                        continue
                    bucket = spec.shape_bucket_fn(shape)
                    db.record_result(
                        shape=shape,
                        hardware=hw_name,
                        config=config,
                        tflops=sr["tflops"],
                        ms=sr.get("ms", 0),
                        correct=True,
                        run_id=cid,
                        operator=op_name,
                        bucket=bucket,
                        config_id_str=cid,
                    )
                    added += 1
            print(f"[{op_name}] added {added} points (hardware={hw_name})")
            total_added += added

    db.save()
    total_points = sum(len(r.results) for r in db.records.values())
    print()
    print(f"Added {total_added} new points this session.")
    print(f"Database now has {total_points} total results across {len(db.records)} shape buckets.")
    print(f"Saved to {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
