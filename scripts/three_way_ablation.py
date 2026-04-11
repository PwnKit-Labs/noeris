#!/usr/bin/env python3
"""Three-way config selection strategy ablation.

Compares three independent config selection strategies on the same operator
within a single persistent Modal session:

  **A) baseline** — frontier-slot selector (incumbent / proposed / curated /
       exploration). Grid exploration slots are filled in natural generator
       order. No cost model, no bandit.

  **B) cost_model** — identical slot structure, but grid candidates are
       pre-ranked by a trained CostModel before being appended.

  **C) bandit** — Thompson sampling over Beta(alpha, beta) posteriors per
       (operator, shape_bucket, hardware, config_id) cell, with curated and
       random-grid fallback for empty slots.

All three conditions share:
- The same operator spec and shape list.
- The same ModalBenchmarkSession (one warm GPU container, no cold starts).
- An independent, freshly-initialised ConfigDatabase so learned state from
  one condition does not bleed into another.

Metric: best avg TFLOPS (or GB/s) after each iteration.  We record the
per-iteration best-so-far trajectory so "iterations to 95%" comparisons are
possible in post-processing.

Usage::

    python3.11 scripts/three_way_ablation.py \\
        --operator softmax \\
        --gpu A100 \\
        --iterations 6 \\
        --configs-per-run 6 \\
        --cost-model .noeris/cost-model.pkl \\
        --no-curated \\
        --output docs/results/three-way-softmax.json

The ``--cost-model`` argument is optional. When omitted, condition B falls
back to baseline behaviour (the cost model is None, same as condition A).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_engine.bandit_selector import BanditSelector
from research_engine.modal_session import ModalBenchmarkSession
from research_engine.triton_kernels import ConfigDatabase
from research_engine.triton_operators import REGISTRY, select_configs_for_operator


# ---------------------------------------------------------------------------
# Condition names — these are the canonical identifiers used in JSON output
# ---------------------------------------------------------------------------

CONDITION_BASELINE = "baseline"
CONDITION_COST_MODEL = "cost_model"
CONDITION_BANDIT = "bandit"


# ---------------------------------------------------------------------------
# Per-iteration execution helpers
# ---------------------------------------------------------------------------


def _parse_shape_for_operator(operator: str, shape_str: str) -> dict | None:
    """Parse a shape string from a benchmark result into a shape dict.

    Delegates to the ablation module's parser to avoid duplicating logic.
    """
    from research_engine.ablation import _parse_shape_for_operator as _parse
    return _parse(operator, shape_str)


def _execute_one_iteration(
    *,
    spec: Any,
    session: Any,
    database: ConfigDatabase,
    shapes: list[dict],
    configs_per_run: int,
    operator: str,
    gpu: str,
    configs: list[dict],
) -> float:
    """Run one benchmark batch and record results into *database*.

    Args:
        spec: TritonOperatorSpec for the operator under test.
        session: Active ModalBenchmarkSession providing ``run_batch``.
        database: ConfigDatabase to record results into (mutated in-place).
        shapes: Target shape dicts.
        configs_per_run: Maximum number of configs (used only for logging).
        operator: Operator name string (e.g. "softmax").
        gpu: Hardware identifier (e.g. "A100").
        configs: Pre-selected list of config dicts to benchmark.

    Returns:
        Best average TFLOPS across all correct shape results in the batch,
        or 0.0 if the batch fails or produces no correct results.
    """
    if not configs:
        return 0.0

    script = spec.benchmark_script_fn(configs, shapes)
    batch = session.run_batch(script)
    if not batch.success:
        return 0.0

    hw_name = batch.hardware.get("gpu", gpu)
    best_metric = 0.0

    for config_result in batch.config_results:
        cid = config_result.get("config_id", "")
        config = config_result.get("config", {})
        shape_results = config_result.get("results", [])

        for shape_result in shape_results:
            if not shape_result.get("correct") or not shape_result.get("tflops"):
                continue
            parsed_shape = _parse_shape_for_operator(
                operator, shape_result.get("shape", "")
            )
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

        correct_results = [
            r for r in shape_results if r.get("correct") and r.get("tflops")
        ]
        if correct_results:
            avg = sum(r["tflops"] for r in correct_results) / len(correct_results)
            if avg > best_metric:
                best_metric = avg

    database.save()
    return best_metric


def select_configs_baseline(
    *,
    spec: Any,
    database: ConfigDatabase,
    hardware: str,
    shapes: list[dict],
    max_configs: int,
    include_curated: bool = True,
) -> list[dict]:
    """Select configs using the frontier-slot baseline selector (no cost model)."""
    return select_configs_for_operator(
        spec=spec,
        database=database,
        hardware=hardware,
        shapes=shapes,
        max_configs=max_configs,
        proposed_configs=None,
        cost_model=None,
        include_curated=include_curated,
    )


def select_configs_cost_model(
    *,
    spec: Any,
    database: ConfigDatabase,
    hardware: str,
    shapes: list[dict],
    max_configs: int,
    cost_model: Any,
    include_curated: bool = True,
) -> list[dict]:
    """Select configs using cost-model-pre-ranked grid candidates."""
    return select_configs_for_operator(
        spec=spec,
        database=database,
        hardware=hardware,
        shapes=shapes,
        max_configs=max_configs,
        proposed_configs=None,
        cost_model=cost_model,
        include_curated=include_curated,
    )


def select_configs_bandit(
    *,
    spec: Any,
    database: ConfigDatabase,
    hardware: str,
    shapes: list[dict],
    max_configs: int,
    bandit: BanditSelector,
) -> list[dict]:
    """Select configs using Thompson-sampling BanditSelector."""
    return bandit.select_configs(
        spec=spec,
        database=database,
        hardware=hardware,
        shapes=shapes,
        max_configs=max_configs,
        proposed_configs=None,
    )


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    """Trajectory and summary for a single ablation condition.

    Attributes:
        name: Canonical condition name (one of the CONDITION_* constants).
        trajectory: Best-metric-so-far after each iteration.
        final_best: Maximum observed metric across all iterations.
    """

    name: str
    trajectory: list[float] = field(default_factory=list)
    final_best: float = 0.0


def run_condition(
    *,
    condition_name: str,
    spec: Any,
    session: Any,
    shapes: list[dict],
    configs_per_run: int,
    iterations: int,
    gpu: str,
    include_curated: bool = True,
    cost_model: Any = None,
    bandit: BanditSelector | None = None,
) -> ConditionResult:
    """Run N iterations of benchmark under a single condition.

    Each condition starts with a fresh, empty ConfigDatabase so there is no
    cross-contamination between conditions.  The Modal session (and its warm
    GPU container) is shared across all conditions.

    Args:
        condition_name: Human-readable label for logging and output JSON.
        spec: TritonOperatorSpec for the operator under test.
        session: Shared ModalBenchmarkSession.
        shapes: List of target shape dicts.
        configs_per_run: Max configs to benchmark per iteration.
        iterations: Number of benchmark iterations to run.
        gpu: Hardware identifier used for database queries.
        include_curated: Whether to include curated starter configs in slot
            allocation (only applies to baseline/cost-model conditions).
        cost_model: Optional trained CostModel (condition B only).
        bandit: Optional BanditSelector instance (condition C only).

    Returns:
        ConditionResult with per-iteration best-so-far trajectory.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    result = ConditionResult(name=condition_name)
    try:
        db = ConfigDatabase(path=db_path)
        best_so_far = 0.0

        for iter_idx in range(iterations):
            # ---- Config selection ----------------------------------------
            if bandit is not None:
                configs = select_configs_bandit(
                    spec=spec,
                    database=db,
                    hardware=gpu,
                    shapes=shapes,
                    max_configs=configs_per_run,
                    bandit=bandit,
                )
            elif cost_model is not None:
                configs = select_configs_cost_model(
                    spec=spec,
                    database=db,
                    hardware=gpu,
                    shapes=shapes,
                    max_configs=configs_per_run,
                    cost_model=cost_model,
                    include_curated=include_curated,
                )
            else:
                configs = select_configs_baseline(
                    spec=spec,
                    database=db,
                    hardware=gpu,
                    shapes=shapes,
                    max_configs=configs_per_run,
                    include_curated=include_curated,
                )

            # ---- Execute on GPU ------------------------------------------
            metric = _execute_one_iteration(
                spec=spec,
                session=session,
                database=db,
                shapes=shapes,
                configs_per_run=configs_per_run,
                operator=spec.name,
                gpu=gpu,
                configs=configs,
            )

            if metric > best_so_far:
                best_so_far = metric
            result.trajectory.append(best_so_far)

            print(
                f"  [{condition_name}] iter {iter_idx + 1:2d}: "
                f"metric={metric:.2f}, best_so_far={best_so_far:.2f}"
            )

        result.final_best = max(result.trajectory) if result.trajectory else 0.0
    finally:
        db_path.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------


def compute_delta_pct(candidate: float, baseline: float) -> float:
    """Return percentage improvement of *candidate* over *baseline*."""
    if baseline <= 0.0:
        return 0.0
    return (candidate - baseline) / baseline * 100.0


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(results: dict[str, ConditionResult], *, baseline_key: str) -> None:
    """Print a formatted summary table to stdout."""
    baseline_final = results[baseline_key].final_best

    print()
    print("=" * 65)
    print(f"{'Condition':<22} {'Final best':>12}  {'vs baseline':>12}")
    print("-" * 65)
    for name, res in results.items():
        delta = compute_delta_pct(res.final_best, baseline_final)
        marker = "" if name == baseline_key else f"  ({delta:+.2f}%)"
        print(f"  {name:<20} {res.final_best:>12.2f}{marker}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments, run all three conditions, write JSON report."""
    parser = argparse.ArgumentParser(
        description="Three-way ablation: baseline vs cost-model vs bandit."
    )
    parser.add_argument("--operator", required=True, help="Operator name (e.g. softmax)")
    parser.add_argument("--gpu", default="A100", help="GPU identifier (default: A100)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per condition")
    parser.add_argument("--configs-per-run", type=int, default=6, help="Max configs per iteration")
    parser.add_argument(
        "--cost-model",
        default=None,
        help="Path to a trained CostModel .pkl file (required for condition B)",
    )
    parser.add_argument(
        "--shapes-set",
        default="standard",
        choices=["tiny", "standard", "full"],
        help="Which shape bucket slice to use (default: standard = first 6)",
    )
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Strip curated starter configs from slot allocation (baseline & cost-model).",
    )
    parser.add_argument("--output", required=True, help="Path for JSON output file")
    args = parser.parse_args()

    # ---- Resolve operator spec ------------------------------------------
    spec = REGISTRY.get(args.operator)
    if spec is None:
        print(f"Unknown operator: {args.operator!r}. Available: {REGISTRY.names()}")
        return 1

    # ---- Load cost model (optional) ------------------------------------
    cost_model = None
    cost_model_meta: dict = {}
    if args.cost_model:
        cm_path = Path(args.cost_model)
        if not cm_path.exists():
            print(f"Cost model not found: {cm_path}")
            return 1
        from research_engine.cost_model import CostModel
        cost_model = CostModel.load(cm_path)
        cost_model_meta = {
            "path": str(cm_path),
            "training_size": cost_model.training_size,
        }
        print(f"Loaded cost model ({cost_model.training_size} training points)")
    else:
        print("No cost model provided — condition B will behave like condition A.")

    # ---- Shape subset --------------------------------------------------
    if args.shapes_set == "tiny":
        shapes = spec.shape_buckets[:2]
    elif args.shapes_set == "full":
        shapes = spec.shape_buckets
    else:
        shapes = spec.shape_buckets[:6]

    include_curated = not args.no_curated

    print()
    print(f"Operator:         {args.operator}")
    print(f"GPU:              {args.gpu}")
    print(f"Iterations:       {args.iterations}")
    print(f"Configs per run:  {args.configs_per_run}")
    print(f"Include curated:  {include_curated}")
    print(f"Shapes ({len(shapes)}):     {[s.get('name') for s in shapes]}")
    print()

    # ---- Create a fresh BanditSelector for condition C -----------------
    bandit = BanditSelector(seed=42)

    # ---- Run all three conditions in a shared Modal session ------------
    all_results: dict[str, ConditionResult] = {}

    with ModalBenchmarkSession(gpu=args.gpu, timeout_seconds=900) as session:
        print(f"Condition A: {CONDITION_BASELINE}")
        all_results[CONDITION_BASELINE] = run_condition(
            condition_name=CONDITION_BASELINE,
            spec=spec,
            session=session,
            shapes=shapes,
            configs_per_run=args.configs_per_run,
            iterations=args.iterations,
            gpu=args.gpu,
            include_curated=include_curated,
            cost_model=None,
            bandit=None,
        )

        print()
        print(f"Condition B: {CONDITION_COST_MODEL}")
        all_results[CONDITION_COST_MODEL] = run_condition(
            condition_name=CONDITION_COST_MODEL,
            spec=spec,
            session=session,
            shapes=shapes,
            configs_per_run=args.configs_per_run,
            iterations=args.iterations,
            gpu=args.gpu,
            include_curated=include_curated,
            cost_model=cost_model,
            bandit=None,
        )

        print()
        print(f"Condition C: {CONDITION_BANDIT}")
        all_results[CONDITION_BANDIT] = run_condition(
            condition_name=CONDITION_BANDIT,
            spec=spec,
            session=session,
            shapes=shapes,
            configs_per_run=args.configs_per_run,
            iterations=args.iterations,
            gpu=args.gpu,
            include_curated=True,  # bandit always uses curated as fallback
            cost_model=None,
            bandit=bandit,
        )

    # ---- Assemble report -----------------------------------------------
    baseline_final = all_results[CONDITION_BASELINE].final_best
    report = {
        "operator": args.operator,
        "gpu": args.gpu,
        "iterations": args.iterations,
        "configs_per_run": args.configs_per_run,
        "include_curated": include_curated,
        "shapes": [s.get("name") for s in shapes],
        "cost_model": cost_model_meta,
        "conditions": {
            CONDITION_BASELINE: {
                "trajectory": all_results[CONDITION_BASELINE].trajectory,
                "final_best": all_results[CONDITION_BASELINE].final_best,
                "delta_vs_baseline_pct": 0.0,
            },
            CONDITION_COST_MODEL: {
                "trajectory": all_results[CONDITION_COST_MODEL].trajectory,
                "final_best": all_results[CONDITION_COST_MODEL].final_best,
                "delta_vs_baseline_pct": round(
                    compute_delta_pct(
                        all_results[CONDITION_COST_MODEL].final_best, baseline_final
                    ),
                    2,
                ),
            },
            CONDITION_BANDIT: {
                "trajectory": all_results[CONDITION_BANDIT].trajectory,
                "final_best": all_results[CONDITION_BANDIT].final_best,
                "delta_vs_baseline_pct": round(
                    compute_delta_pct(
                        all_results[CONDITION_BANDIT].final_best, baseline_final
                    ),
                    2,
                ),
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print_summary_table(all_results, baseline_key=CONDITION_BASELINE)
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
