#!/usr/bin/env python3
"""Four-way config selection strategy ablation including the ensemble.

Compares four independent config selection strategies on the same operator
within a single persistent Modal session:

  **A) baseline** — frontier-slot selector (incumbent / proposed / curated /
       exploration). No cost model, no bandit.

  **B) cost_model** — identical slot structure, but grid candidates are
       pre-ranked by a trained CostModel.

  **C) bandit** — Thompson sampling over Beta posteriors per
       (operator, shape_bucket, hardware, config_id) cell, with curated
       and random-grid fallback for empty slots.

  **D) ensemble** — interleaved cost-model + bandit picks via
       ``select_configs_ensemble``, combining the best of B and C.

All four conditions share:
- The same operator spec and shape list.
- The same ModalBenchmarkSession (one warm GPU container, no cold starts).
- An independent, freshly-initialised ConfigDatabase so learned state from
  one condition does not bleed into another.

Metric: best avg TFLOPS (or GB/s) after each iteration.  We record the
per-iteration best-so-far trajectory so "iterations to 95%" comparisons are
possible in post-processing.

Usage::

    python3.11 scripts/ensemble_ablation.py \\
        --operator matmul \\
        --gpu A100 \\
        --iterations 6 \\
        --configs-per-run 6 \\
        --cost-model .noeris/cost-model.pkl \\
        --no-curated \\
        --output docs/results/ensemble-matmul.json

The ``--cost-model`` argument is optional. When omitted, conditions B and D
behave like baseline and bandit respectively.
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
from research_engine.ensemble_selector import select_configs_ensemble
from research_engine.modal_session import ModalBenchmarkSession
from research_engine.triton_kernels import ConfigDatabase
from research_engine.triton_operators import REGISTRY, select_configs_for_operator


# ---------------------------------------------------------------------------
# Condition names — canonical identifiers used in JSON output
# ---------------------------------------------------------------------------

CONDITION_BASELINE = "baseline"
CONDITION_COST_MODEL = "cost_model"
CONDITION_BANDIT = "bandit"
CONDITION_ENSEMBLE = "ensemble"


# ---------------------------------------------------------------------------
# Per-iteration execution helpers
# ---------------------------------------------------------------------------


def _parse_shape_for_operator(operator: str, shape_str: str) -> dict | None:
    """Parse a shape string from a benchmark result into a shape dict."""
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

    Returns:
        Best average TFLOPS across all correct shape results, or 0.0 on failure.
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


# ---------------------------------------------------------------------------
# Per-condition config selectors
# ---------------------------------------------------------------------------


def select_configs_baseline(
    *,
    spec: Any,
    database: ConfigDatabase,
    hardware: str,
    shapes: list[dict],
    max_configs: int,
    include_curated: bool = True,
) -> list[dict]:
    """Frontier-slot baseline selector (no cost model, no bandit)."""
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
    """Cost-model-pre-ranked grid candidates."""
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
    """Thompson-sampling BanditSelector."""
    return bandit.select_configs(
        spec=spec,
        database=database,
        hardware=hardware,
        shapes=shapes,
        max_configs=max_configs,
        proposed_configs=None,
    )


def select_configs_ensemble_condition(
    *,
    spec: Any,
    database: ConfigDatabase,
    hardware: str,
    shapes: list[dict],
    max_configs: int,
    cost_model: Any,
    bandit: BanditSelector,
    include_curated: bool = True,
) -> list[dict]:
    """Interleaved ensemble of cost model + bandit."""
    return select_configs_ensemble(
        spec=spec,
        database=database,
        hardware=hardware,
        shapes=shapes,
        max_configs=max_configs,
        proposed_configs=None,
        cost_model=cost_model,
        bandit_selector=bandit,
        include_curated=include_curated,
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
    is_ensemble: bool = False,
) -> ConditionResult:
    """Run N iterations of benchmark under a single condition.

    Each condition starts with a fresh, empty ConfigDatabase.

    Args:
        condition_name: Label for logging and output JSON.
        spec: TritonOperatorSpec for the operator under test.
        session: Shared ModalBenchmarkSession.
        shapes: List of target shape dicts.
        configs_per_run: Max configs per iteration.
        iterations: Number of benchmark iterations to run.
        gpu: Hardware identifier.
        include_curated: Whether to include curated starter configs.
        cost_model: Optional trained CostModel.
        bandit: Optional BanditSelector instance.
        is_ensemble: When True, route through ``select_configs_ensemble``.

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
            if is_ensemble:
                configs = select_configs_ensemble_condition(
                    spec=spec,
                    database=db,
                    hardware=gpu,
                    shapes=shapes,
                    max_configs=configs_per_run,
                    cost_model=cost_model,
                    bandit=bandit,
                    include_curated=include_curated,
                )
            elif bandit is not None:
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


def print_summary_table(
    results: dict[str, ConditionResult],
    *,
    baseline_key: str,
) -> None:
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
    """Parse arguments, run all four conditions, write JSON report."""
    parser = argparse.ArgumentParser(
        description=(
            "Four-way ablation: baseline vs cost-model vs bandit vs ensemble."
        )
    )
    parser.add_argument(
        "--operator", required=True, help="Operator name (e.g. matmul, softmax)"
    )
    parser.add_argument("--gpu", default="A100", help="GPU identifier (default: A100)")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Iterations per condition"
    )
    parser.add_argument(
        "--configs-per-run",
        type=int,
        default=6,
        help="Max configs per iteration",
    )
    parser.add_argument(
        "--cost-model",
        default=None,
        help="Path to a trained CostModel .pkl file",
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
        help="Strip curated configs from slot allocation (baseline & cost-model).",
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
        print("No cost model — conditions B and D will behave like A and C.")

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

    # ---- Create BanditSelectors for conditions C and D -----------------
    # Each condition gets its own bandit instance with the same seed so
    # posterior state is independent between conditions.
    bandit_c = BanditSelector(seed=42)
    bandit_d = BanditSelector(seed=42)

    # ---- Run all four conditions in a shared Modal session -------------
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
            bandit=bandit_c,
        )

        print()
        print(f"Condition D: {CONDITION_ENSEMBLE}")
        all_results[CONDITION_ENSEMBLE] = run_condition(
            condition_name=CONDITION_ENSEMBLE,
            spec=spec,
            session=session,
            shapes=shapes,
            configs_per_run=args.configs_per_run,
            iterations=args.iterations,
            gpu=args.gpu,
            include_curated=include_curated,
            cost_model=cost_model,
            bandit=bandit_d,
            is_ensemble=True,
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
            cname: {
                "trajectory": res.trajectory,
                "final_best": res.final_best,
                "delta_vs_baseline_pct": round(
                    compute_delta_pct(res.final_best, baseline_final), 2
                ) if cname != CONDITION_BASELINE else 0.0,
            }
            for cname, res in all_results.items()
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
