"""Run the Noeris autonomous search loop on Google Colab's free T4 GPU.

This is the CORE of the Noeris system: generate configs → benchmark on GPU →
update the config database → learn → repeat. The bandit selector uses
Thompson sampling with Beta posteriors to explore the config space, and the
cost model (if available) filters candidates before GPU evaluation.

Usage in Colab:
  !git clone https://github.com/peaktwilight/noeris && cd noeris
  !pip install -e . numpy scikit-learn
  !python scripts/colab_iterate.py --operator qk_norm_rope --iterations 3 --configs-per-iter 8

Each iteration benchmarks `configs-per-iter` configurations across all shape
buckets for the chosen operator. Results are saved to a local config database
that persists across iterations (but not across Colab sessions unless you
download it).

No Modal needed. Free T4 GPU. This is how Noeris finds better configs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch


def run_iteration(
    operator: str,
    configs_per_iter: int,
    db_path: str,
    use_bandit: bool = True,
    shapes_mode: str = "standard",
) -> dict:
    """Run one iteration of the Noeris search loop on the local GPU."""
    from research_engine.triton_operators import REGISTRY
    from research_engine.triton_kernels import ConfigDatabase
    from research_engine.timing_snippet import install_noeris_timing

    spec = REGISTRY.get(operator)
    if spec is None:
        return {"error": f"Unknown operator: {operator}"}

    db = ConfigDatabase(path=db_path)
    hardware = torch.cuda.get_device_name(0)

    # Select shapes
    if shapes_mode == "tiny":
        shapes = spec.shape_buckets[:2]
    elif shapes_mode == "full":
        shapes = spec.shape_buckets
    else:
        shapes = spec.shape_buckets[:6]

    # Select configs — use bandit if available, else grid sample
    if use_bandit:
        try:
            from research_engine.bandit_selector import BanditSelector
            bandit = BanditSelector()
            configs = bandit.select_configs(
                spec=spec, database=db, hardware=hardware,
                shapes=shapes, max_configs=configs_per_iter,
                proposed_configs=[],
            )
            selector = "bandit"
        except Exception as e:
            print(f"  Bandit failed ({e}), falling back to grid sample")
            configs = spec.grid_generator_fn(max_configs=configs_per_iter)[:configs_per_iter]
            selector = "grid"
    else:
        configs = spec.grid_generator_fn(max_configs=configs_per_iter)[:configs_per_iter]
        selector = "grid"

    print(f"  Selector: {selector}, {len(configs)} configs, {len(shapes)} shapes")

    # Generate and run the benchmark script
    script = spec.benchmark_script_fn(configs, shapes)
    script = install_noeris_timing(script, timer="cuda_event")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {
            "error": proc.stderr[-500:] if proc.stderr else f"returncode={proc.returncode}",
            "elapsed": elapsed,
        }

    # Parse results
    stdout = proc.stdout
    start = stdout.find("{")
    if start < 0:
        return {"error": "No JSON in stdout", "elapsed": elapsed}

    payload = json.loads(stdout[start:])

    # Update the config database with results
    new_bests = 0
    total_correct = 0
    total_tested = 0
    best_per_shape = {}

    for cfg_result in payload.get("config_results", []):
        cid = cfg_result.get("config_id", "?")
        config = cfg_result.get("config", {})
        for r in cfg_result.get("results", []):
            total_tested += 1
            shape_name = r.get("shape_name", "?")
            correct = r.get("correct", False)
            metric = r.get("gb_per_s") or r.get("tflops") or 0
            ms = r.get("ms") or 0

            if correct:
                total_correct += 1

            # Record in database
            bucket = spec.shape_bucket_fn(
                next((s for s in spec.shape_buckets if s["name"] == shape_name), {})
            ) if spec.shape_bucket_fn else shape_name

            is_new_best = db.record_result(
                shape={"name": shape_name},
                hardware=hardware,
                config=config,
                tflops=float(metric),
                ms=float(ms),
                correct=correct,
                operator=operator,
                bucket=bucket,
                config_id_str=cid,
            )
            if is_new_best:
                new_bests += 1

            # Track best per shape for display
            if correct and (shape_name not in best_per_shape or metric > best_per_shape[shape_name]["metric"]):
                best_per_shape[shape_name] = {
                    "config_id": cid, "metric": metric, "ms": ms,
                    "fusion_speedup": r.get("fusion_speedup"),
                }

    # Save database
    db.save()

    return {
        "operator": operator,
        "hardware": hardware,
        "selector": selector,
        "configs_tested": len(configs),
        "shapes_tested": len(shapes),
        "total_measurements": total_tested,
        "correct": total_correct,
        "new_bests": new_bests,
        "elapsed": round(elapsed, 1),
        "best_per_shape": best_per_shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Noeris search loop on Colab T4")
    parser.add_argument("--operator", default="qk_norm_rope",
                       help="Which operator to search (default: qk_norm_rope)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of search iterations (default: 3)")
    parser.add_argument("--configs-per-iter", type=int, default=8,
                       help="Configs to test per iteration (default: 8)")
    parser.add_argument("--db-path", default=".noeris/colab-configs.json",
                       help="Path to config database (default: .noeris/colab-configs.json)")
    parser.add_argument("--shapes", default="standard", choices=["tiny", "standard", "full"],
                       help="Shape set (default: standard)")
    parser.add_argument("--no-bandit", action="store_true",
                       help="Use grid sampling instead of bandit")
    parser.add_argument("--all-operators", action="store_true",
                       help="Run on ALL operators (1 iteration each)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)

    if args.all_operators:
        from research_engine.triton_operators import REGISTRY
        operators = sorted(REGISTRY.names())
    else:
        operators = [args.operator]

    for op in operators:
        print(f"\n{'='*60}")
        print(f"OPERATOR: {op}")
        print(f"{'='*60}")

        for i in range(args.iterations):
            print(f"\n--- Iteration {i+1}/{args.iterations} ---")
            result = run_iteration(
                operator=op,
                configs_per_iter=args.configs_per_iter,
                db_path=args.db_path,
                use_bandit=not args.no_bandit,
                shapes_mode=args.shapes,
            )

            if "error" in result:
                print(f"  ERROR: {result['error'][:200]}")
                continue

            print(f"  {result['correct']}/{result['total_measurements']} correct, "
                  f"{result['new_bests']} new bests, {result['elapsed']}s")

            for shape, best in result.get("best_per_shape", {}).items():
                fs = f" fusion={best['fusion_speedup']:.2f}x" if best.get("fusion_speedup") else ""
                print(f"    {shape:30s} {best['metric']:>10.2f} GB/s  {best['config_id']}{fs}")

    # Final summary from DB
    from research_engine.triton_kernels import ConfigDatabase
    db = ConfigDatabase(path=args.db_path)
    insights = db.get_insights()
    if insights:
        print(f"\n{'='*60}")
        print(f"DATABASE INSIGHTS ({len(insights)} buckets)")
        print(f"{'='*60}")
        for ins in insights:
            print(f"  {ins.get('shape_bucket', '?'):40s} best={ins.get('best_tflops', 0):.2f} "
                  f"config={ins.get('best_config_id', '?')} "
                  f"experiments={ins.get('total_experiments', 0)}")


if __name__ == "__main__":
    main()
