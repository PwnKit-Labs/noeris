#!/usr/bin/env python3
"""Cost-model-filtered vs unfiltered ablation.

Compares two config selection strategies on the same operator:

  **A) Baseline:** frontier-slot selector (incumbent / proposed / curated /
       exploration). Grid exploration slots are appended in natural order
       from the grid generator.

  **B) Cost-model-filtered:** identical slotting, but grid candidates are
       pre-ranked by a trained CostModel before being appended.

Both conditions use the same operator, shapes, and Modal GPU session.
Within-session we avoid cold starts entirely — one warm container runs
everything.

Metric: best avg metric (TFLOPS or GB/s) after N iterations, per condition.
We also record the per-iteration trajectory so the "iterations to 95%"
comparison is possible.

Usage:

    python scripts/cost_model_ablation.py \
        --operator rmsnorm \
        --gpu A100 \
        --iterations 5 \
        --configs-per-run 6 \
        --cost-model .noeris/cost-model.pkl \
        --output docs/results/cost-model-ablation-rmsnorm.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_engine.ablation import _run_one_session_iteration
from research_engine.cost_model import CostModel
from research_engine.modal_session import ModalBenchmarkSession
from research_engine.triton_kernels import ConfigDatabase
from research_engine.triton_operators import REGISTRY


def _run_condition(
    *,
    condition_name: str,
    spec,
    session,
    shapes,
    configs_per_run: int,
    iterations: int,
    cost_model=None,
    include_curated: bool = True,
) -> list[float]:
    """Run N iterations of triton-iterate under a single condition.

    Returns: list of best-metric-so-far after each iteration.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    db_path = Path(tmp.name)
    tmp.close()
    try:
        db = ConfigDatabase(path=db_path)
        trajectory: list[float] = []
        best_so_far = 0.0
        for iter_idx in range(iterations):
            metric = _run_one_session_iteration_with_cost_model(
                spec=spec,
                session=session,
                database=db,
                shapes=shapes,
                configs_per_run=configs_per_run,
                llm_client=None,  # no LLM for this experiment — isolate the cost model
                operator=spec.name,
                gpu="A100",
                cost_model=cost_model,
                include_curated=include_curated,
            )
            if metric > best_so_far:
                best_so_far = metric
            trajectory.append(best_so_far)
            print(f"  [{condition_name}] iter {iter_idx + 1}: metric={metric:.2f}, best_so_far={best_so_far:.2f}")
        return trajectory
    finally:
        db_path.unlink(missing_ok=True)


def _run_one_session_iteration_with_cost_model(
    *, spec, session, database, shapes, configs_per_run,
    llm_client, operator, gpu, cost_model,
    include_curated: bool = True,
) -> float:
    """Variant of _run_one_session_iteration that uses the cost model in
    the selector."""
    from research_engine.triton_operators import select_configs_for_operator

    configs = select_configs_for_operator(
        spec=spec,
        database=database,
        hardware=gpu,
        shapes=shapes,
        max_configs=configs_per_run,
        proposed_configs=None,
        cost_model=cost_model,
        include_curated=include_curated,
    )
    if not configs:
        return 0.0

    script = spec.benchmark_script_fn(configs, shapes)
    batch = session.run_batch(script)
    if not batch.success:
        return 0.0

    hw_name = batch.hardware.get("gpu", gpu)
    best_metric = 0.0

    from research_engine.ablation import _parse_shape_for_operator
    for config_result in batch.config_results:
        cid = config_result.get("config_id", "")
        config = config_result.get("config", {})
        shape_results = config_result.get("results", [])
        for shape_result in shape_results:
            if not shape_result.get("correct") or not shape_result.get("tflops"):
                continue
            parsed_shape = _parse_shape_for_operator(operator, shape_result.get("shape", ""))
            if parsed_shape is None:
                continue
            bucket = spec.shape_bucket_fn(parsed_shape)
            database.record_result(
                shape=parsed_shape,
                hardware=hw_name,
                config=config,
                tflops=shape_result["tflops"],
                ms=shape_result.get("ms", 0),
                correct=True,
                run_id=cid,
                operator=operator,
                bucket=bucket,
                config_id_str=cid,
            )

        correct_results = [r for r in shape_results if r.get("correct") and r.get("tflops")]
        if correct_results:
            avg = sum(r["tflops"] for r in correct_results) / len(correct_results)
            if avg > best_metric:
                best_metric = avg

    database.save()
    return best_metric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--configs-per-run", type=int, default=6)
    parser.add_argument("--cost-model", required=True)
    parser.add_argument("--shapes-set", default="standard", choices=["tiny", "standard", "full"])
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Strip curated starter configs so the cost model has room to filter grid candidates.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = REGISTRY.get(args.operator)
    cm_path = Path(args.cost_model)
    if not cm_path.exists():
        print(f"Cost model not found: {cm_path}")
        return 1
    cost_model = CostModel.load(cm_path)
    print(f"Loaded cost model ({cost_model.training_size} training points)")

    if args.shapes_set == "tiny":
        shapes = spec.shape_buckets[:2]
    elif args.shapes_set == "full":
        shapes = spec.shape_buckets
    else:
        shapes = spec.shape_buckets[:6]

    print(f"Ablation on {args.operator}: {args.iterations} iterations x {args.configs_per_run} configs")
    print(f"Shapes: {[s.get('name') for s in shapes]}")
    print()

    include_curated = not args.no_curated
    with ModalBenchmarkSession(gpu=args.gpu, timeout_seconds=900) as session:
        print(f"Condition A: baseline (no cost model, include_curated={include_curated})")
        baseline = _run_condition(
            condition_name="baseline",
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run,
            iterations=args.iterations,
            cost_model=None,
            include_curated=include_curated,
        )

        print()
        print(f"Condition B: cost-model-filtered (include_curated={include_curated})")
        filtered = _run_condition(
            condition_name="cost_model",
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run,
            iterations=args.iterations,
            cost_model=cost_model,
            include_curated=include_curated,
        )

    baseline_final = max(baseline) if baseline else 0
    filtered_final = max(filtered) if filtered else 0
    delta_pct = (
        (filtered_final - baseline_final) / baseline_final * 100
        if baseline_final > 0 else 0
    )

    report = {
        "operator": args.operator,
        "gpu": args.gpu,
        "iterations": args.iterations,
        "configs_per_run": args.configs_per_run,
        "include_curated": include_curated,
        "shapes": [s.get("name") for s in shapes],
        "baseline_trajectory": baseline,
        "filtered_trajectory": filtered,
        "baseline_final": baseline_final,
        "filtered_final": filtered_final,
        "delta_pct": round(delta_pct, 2),
        "cost_model": {
            "path": str(cm_path),
            "training_size": cost_model.training_size,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print("=" * 60)
    print(f"Baseline final:        {baseline_final:.2f}")
    print(f"Cost-model-filtered:   {filtered_final:.2f}")
    print(f"Delta:                 {delta_pct:+.2f}%")
    print("=" * 60)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
