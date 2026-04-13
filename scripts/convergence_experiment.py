#!/usr/bin/env python3
"""Bandit convergence experiment: demonstrate near-optimal in <5 iterations.

For each of 3 operators (rmsnorm, qk_norm_rope, geglu), picks 2 representative
shapes, runs exhaustive grid search as the baseline "optimal", then runs 10
iterations of bandit search (6 configs each) and tracks best-so-far throughput.

Outputs a convergence table showing iteration vs % of optimal achieved, plus
a JSON file with full data.

Usage::

    python3 scripts/convergence_experiment.py [--output convergence.json]

Requires a CUDA GPU (designed for Kaggle T4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_engine.bandit_selector import BanditSelector
from research_engine.triton_kernels import ConfigDatabase
from research_engine.triton_operators import REGISTRY


# ---------------------------------------------------------------------------
# Shape string parsers (match benchmark_script_fn output formats)
# ---------------------------------------------------------------------------

def _parse_shape(operator: str, shape_str: str) -> dict | None:
    parts = shape_str.split("x")
    try:
        if operator == "rmsnorm":
            return {"n_rows": int(parts[0]), "hidden_dim": int(parts[1])}
        if operator == "geglu":
            return {"n_rows": int(parts[0]), "ffn_dim": int(parts[1])}
        if operator == "qk_norm_rope":
            return {
                "batch": int(parts[0]), "heads": int(parts[1]),
                "num_kv_heads": int(parts[2]), "seq": int(parts[3]),
                "head_dim": int(parts[4]),
            }
    except (ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Subprocess benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(spec: Any, configs: list[dict], shapes: list[dict]) -> list[dict]:
    """Run benchmark via subprocess for Triton JIT isolation.

    Returns list of {config_id, config, results: [{correct, tflops, shape, ...}]}.
    """
    script = spec.benchmark_script_fn(configs, shapes)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            print(f"    [WARN] benchmark subprocess failed: {proc.stderr[:200]}")
            return []
        output = json.loads(proc.stdout)
        return output.get("config_results", [])
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"    [WARN] benchmark error: {exc}")
        return []
    finally:
        Path(script_path).unlink(missing_ok=True)


def record_results(
    spec: Any, db: ConfigDatabase, config_results: list[dict],
    hardware: str, operator: str,
) -> float:
    """Record benchmark results into database. Returns best tflops in batch."""
    best = 0.0
    for cr in config_results:
        cid = cr.get("config_id", "")
        config = cr.get("config", {})
        for sr in cr.get("results", []):
            if not sr.get("correct") or not sr.get("tflops"):
                continue
            tflops = sr["tflops"]
            parsed = _parse_shape(operator, sr.get("shape", ""))
            if parsed is None:
                continue
            bucket = spec.shape_bucket_fn(parsed)
            db.record_result(
                shape=parsed, hardware=hardware, config=config,
                tflops=tflops, ms=sr.get("ms", 0), correct=True,
                run_id=cid, operator=operator, bucket=bucket,
                config_id_str=cid,
            )
            if tflops > best:
                best = tflops
    db.save()
    return best


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

OPERATORS = ["rmsnorm", "qk_norm_rope", "geglu"]
BANDIT_ITERATIONS = 10
CONFIGS_PER_ITER = 6
EXHAUSTIVE_MAX = 50


def run_experiment(output_path: str) -> int:
    import torch
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "T4"
    print(f"Hardware: {hardware}\n")

    all_data: dict[str, Any] = {"hardware": hardware, "operators": {}}

    for operator in OPERATORS:
        spec = REGISTRY.get(operator)
        shapes = spec.shape_buckets[:2]
        shape_names = [s.get("name", "?") for s in shapes]
        print(f"{'='*60}")
        print(f"Operator: {operator}")
        print(f"Shapes:   {shape_names}")

        # --- Phase 1: Exhaustive baseline ---
        print(f"\n  Phase 1: Exhaustive search ({EXHAUSTIVE_MAX} configs)...")
        all_configs = spec.grid_generator_fn(max_configs=EXHAUSTIVE_MAX)
        exhaustive_results = run_benchmark(spec, all_configs, shapes)

        exhaustive_best = 0.0
        for cr in exhaustive_results:
            for sr in cr.get("results", []):
                if sr.get("correct") and sr.get("tflops", 0) > exhaustive_best:
                    exhaustive_best = sr["tflops"]

        if exhaustive_best <= 0:
            print(f"  [SKIP] No valid exhaustive results for {operator}")
            continue
        print(f"  Exhaustive best: {exhaustive_best:.1f} (metric: {spec.metric_name})")

        # --- Phase 2: Bandit search ---
        print(f"\n  Phase 2: Bandit search ({BANDIT_ITERATIONS} iters x {CONFIGS_PER_ITER} configs)...")
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        db_path = Path(tmp.name)
        tmp.close()

        try:
            db = ConfigDatabase(path=db_path)
            bandit = BanditSelector(seed=42)
            best_so_far = 0.0
            trajectory: list[float] = []

            for it in range(BANDIT_ITERATIONS):
                configs = bandit.select_configs(
                    spec=spec, database=db, hardware=hardware,
                    shapes=shapes, max_configs=CONFIGS_PER_ITER,
                )
                batch_results = run_benchmark(spec, configs, shapes)
                batch_best = record_results(
                    spec, db, batch_results, hardware, operator,
                )
                if batch_best > best_so_far:
                    best_so_far = batch_best
                trajectory.append(best_so_far)

                pct = best_so_far / exhaustive_best * 100
                print(f"    Iter {it+1:2d}: {best_so_far:.1f} ({pct:.0f}% of optimal)")

            # --- Report ---
            print(f"\n  Convergence table for {operator}:")
            print(f"  {'Iter':>6} {'Best':>10} {'% Optimal':>10}")
            print(f"  {'-'*30}")
            for i, best in enumerate(trajectory):
                pct = best / exhaustive_best * 100
                print(f"  {i+1:>6} {best:>10.1f} {pct:>9.0f}%")

            # Find first iteration reaching 90%
            first_90 = None
            for i, best in enumerate(trajectory):
                if best / exhaustive_best >= 0.90:
                    first_90 = i + 1
                    break

            summary = (
                f"Bandit reaches {trajectory[-1]/exhaustive_best*100:.0f}% "
                f"of exhaustive by iter {BANDIT_ITERATIONS}"
            )
            if first_90 is not None:
                summary += f"; >=90% at iter {first_90}"
            print(f"\n  {summary}")

            all_data["operators"][operator] = {
                "shapes": shape_names,
                "exhaustive_best": exhaustive_best,
                "exhaustive_configs_tested": len(all_configs),
                "trajectory": trajectory,
                "trajectory_pct": [
                    round(b / exhaustive_best * 100, 1) for b in trajectory
                ],
                "first_90pct_iter": first_90,
                "summary": summary,
            }
        finally:
            db_path.unlink(missing_ok=True)

        print()

    # --- Global summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for op, data in all_data.get("operators", {}).items():
        print(f"  {op}: {data['summary']}")

    # Save JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_data, indent=2) + "\n")
    print(f"\nSaved: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bandit convergence experiment")
    parser.add_argument(
        "--output", default="convergence.json",
        help="Path for JSON output (default: convergence.json)",
    )
    args = parser.parse_args()
    return run_experiment(args.output)


if __name__ == "__main__":
    sys.exit(main())
