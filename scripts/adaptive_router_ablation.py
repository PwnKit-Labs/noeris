#!/usr/bin/env python3
"""Five-way config selection strategy ablation including the adaptive router.

Compares five independent config selection strategies on the same operator
within a single persistent Modal session:

  **A) baseline**         — frontier-slot selector. No model, no bandit.
  **B) cost_model**       — grid candidates pre-ranked by a trained CostModel.
  **C) bandit**           — Thompson sampling over Beta posteriors.
  **D) naive_ensemble**   — interleaved cost-model + bandit picks (alternation).
  **E) adaptive_router**  — bandit-over-selectors: Thompson-samples one of
                            {cost_model, bandit, baseline} per iteration,
                            updates arm posteriors from observed improvement.

All five conditions share:
- The same operator spec and shape list.
- The same ModalBenchmarkSession (one warm GPU container, no cold starts).
- Independent, freshly-initialised ConfigDatabases so learned state from
  one condition does not bleed into another.

The adaptive router starts from uniform priors Beta(1,1) over the three arms.
After each iteration it observes whether the best metric improved and updates
the chosen arm's posterior.  Over time the router concentrates probability on
the winning selector for this operator.

Metric: best avg TFLOPS (or GB/s) after each iteration.  Per-iteration
best-so-far trajectories are recorded for "iterations to 95%" analysis.

Usage::

    python3.11 scripts/adaptive_router_ablation.py \\
        --operator matmul \\
        --gpu A100 \\
        --iterations 6 \\
        --configs-per-run 6 \\
        --cost-model .noeris/cost-model.pkl \\
        --no-curated \\
        --output docs/results/adaptive-router-matmul.json

The ``--cost-model`` argument is optional.  When omitted, conditions B, D, and
the router's cost_model arm all behave like baseline/bandit respectively.
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

from research_engine.adaptive_router import AdaptiveRouter, ARM_COST_MODEL, ARM_BANDIT, ARM_BASELINE
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
CONDITION_NAIVE_ENSEMBLE = "naive_ensemble"
CONDITION_ADAPTIVE_ROUTER = "adaptive_router"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_shape_for_operator(operator: str, shape_str: str) -> dict | None:
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


def select_configs_baseline(*, spec, database, hardware, shapes, max_configs, include_curated=True):
    return select_configs_for_operator(
        spec=spec, database=database, hardware=hardware, shapes=shapes,
        max_configs=max_configs, proposed_configs=None, cost_model=None,
        include_curated=include_curated,
    )


def select_configs_cost_model(*, spec, database, hardware, shapes, max_configs, cost_model, include_curated=True):
    return select_configs_for_operator(
        spec=spec, database=database, hardware=hardware, shapes=shapes,
        max_configs=max_configs, proposed_configs=None, cost_model=cost_model,
        include_curated=include_curated,
    )


def select_configs_bandit(*, spec, database, hardware, shapes, max_configs, bandit):
    return bandit.select_configs(
        spec=spec, database=database, hardware=hardware, shapes=shapes,
        max_configs=max_configs, proposed_configs=None,
    )


def select_configs_naive_ensemble(*, spec, database, hardware, shapes, max_configs, cost_model, bandit, include_curated=True):
    return select_configs_ensemble(
        spec=spec, database=database, hardware=hardware, shapes=shapes,
        max_configs=max_configs, proposed_configs=None, cost_model=cost_model,
        bandit_selector=bandit, include_curated=include_curated,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    """Trajectory and summary for a single ablation condition."""

    name: str
    trajectory: list[float] = field(default_factory=list)
    final_best: float = 0.0
    # For the adaptive router: record which arm was chosen per iteration
    arm_choices: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


def run_standard_condition(
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
    is_naive_ensemble: bool = False,
) -> ConditionResult:
    """Run N iterations of a non-adaptive condition."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    result = ConditionResult(name=condition_name)
    try:
        db = ConfigDatabase(path=db_path)
        best_so_far = 0.0

        for iter_idx in range(iterations):
            if is_naive_ensemble:
                configs = select_configs_naive_ensemble(
                    spec=spec, database=db, hardware=gpu, shapes=shapes,
                    max_configs=configs_per_run, cost_model=cost_model,
                    bandit=bandit, include_curated=include_curated,
                )
            elif bandit is not None:
                configs = select_configs_bandit(
                    spec=spec, database=db, hardware=gpu, shapes=shapes,
                    max_configs=configs_per_run, bandit=bandit,
                )
            elif cost_model is not None:
                configs = select_configs_cost_model(
                    spec=spec, database=db, hardware=gpu, shapes=shapes,
                    max_configs=configs_per_run, cost_model=cost_model,
                    include_curated=include_curated,
                )
            else:
                configs = select_configs_baseline(
                    spec=spec, database=db, hardware=gpu, shapes=shapes,
                    max_configs=configs_per_run, include_curated=include_curated,
                )

            metric = _execute_one_iteration(
                spec=spec, session=session, database=db, shapes=shapes,
                configs_per_run=configs_per_run, operator=spec.name,
                gpu=gpu, configs=configs,
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


def run_adaptive_router_condition(
    *,
    spec: Any,
    session: Any,
    shapes: list[dict],
    configs_per_run: int,
    iterations: int,
    gpu: str,
    cost_model: Any = None,
    router_seed: int = 42,
) -> ConditionResult:
    """Run N iterations under the adaptive router.

    The router starts from uniform priors.  After each iteration it observes
    whether the best metric improved and updates the chosen arm's Beta
    posterior.  The arm selection and outcome update are both operator-scoped.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    result = ConditionResult(name=CONDITION_ADAPTIVE_ROUTER)
    try:
        db = ConfigDatabase(path=db_path)

        # Separate BanditSelector dedicated to the router's bandit arm.
        # Fresh instance — it builds its own state from the run's own database.
        bandit = BanditSelector(seed=router_seed)

        router = AdaptiveRouter(seed=router_seed)
        best_so_far = 0.0

        for iter_idx in range(iterations):
            configs, chosen_arm = router.select_configs(
                spec=spec,
                database=db,
                hardware=gpu,
                shapes=shapes,
                max_configs=configs_per_run,
                cost_model=cost_model,
                bandit_selector=bandit,
            )

            metric = _execute_one_iteration(
                spec=spec, session=session, database=db, shapes=shapes,
                configs_per_run=configs_per_run, operator=spec.name,
                gpu=gpu, configs=configs,
            )

            improved = metric > best_so_far
            if improved:
                best_so_far = metric
            result.trajectory.append(best_so_far)
            result.arm_choices.append(chosen_arm)

            # Update the router's posterior for the chosen arm.
            router.record_outcome(
                operator=spec.name, arm=chosen_arm, improvement=improved
            )

            print(
                f"  [{CONDITION_ADAPTIVE_ROUTER}] iter {iter_idx + 1:2d}: "
                f"arm={chosen_arm}, metric={metric:.2f}, "
                f"best_so_far={best_so_far:.2f}"
            )

        result.final_best = max(result.trajectory) if result.trajectory else 0.0
    finally:
        db_path.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------


def compute_delta_pct(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return (candidate - baseline) / baseline * 100.0


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(results: dict[str, ConditionResult], *, baseline_key: str) -> None:
    baseline_final = results[baseline_key].final_best
    print()
    print("=" * 70)
    print(f"{'Condition':<26} {'Final best':>12}  {'vs baseline':>12}")
    print("-" * 70)
    for name, res in results.items():
        delta = compute_delta_pct(res.final_best, baseline_final)
        marker = "" if name == baseline_key else f"  ({delta:+.2f}%)"
        print(f"  {name:<24} {res.final_best:>12.2f}{marker}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Five-way ablation: baseline vs cost-model vs bandit vs "
            "naive-ensemble vs adaptive-router."
        )
    )
    parser.add_argument("--operator", required=True, help="Operator name (e.g. matmul)")
    parser.add_argument("--gpu", default="A100", help="GPU identifier (default: A100)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per condition")
    parser.add_argument("--configs-per-run", type=int, default=6, help="Max configs per iteration")
    parser.add_argument("--cost-model", default=None, help="Path to a trained CostModel .pkl file")
    parser.add_argument(
        "--shapes-set", default="standard", choices=["tiny", "standard", "full"],
        help="Which shape bucket slice to use (default: standard = first 6)",
    )
    parser.add_argument(
        "--no-curated", action="store_true",
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
        print("No cost model — conditions B, D, and router cost_model arm behave like baseline/bandit.")

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
    bandit_c = BanditSelector(seed=42)
    bandit_d = BanditSelector(seed=42)

    # ---- Run all five conditions in a shared Modal session -------------
    all_results: dict[str, ConditionResult] = {}

    with ModalBenchmarkSession(gpu=args.gpu, timeout_seconds=900) as session:
        print(f"Condition A: {CONDITION_BASELINE}")
        all_results[CONDITION_BASELINE] = run_standard_condition(
            condition_name=CONDITION_BASELINE,
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run, iterations=args.iterations,
            gpu=args.gpu, include_curated=include_curated,
        )

        print()
        print(f"Condition B: {CONDITION_COST_MODEL}")
        all_results[CONDITION_COST_MODEL] = run_standard_condition(
            condition_name=CONDITION_COST_MODEL,
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run, iterations=args.iterations,
            gpu=args.gpu, include_curated=include_curated,
            cost_model=cost_model,
        )

        print()
        print(f"Condition C: {CONDITION_BANDIT}")
        all_results[CONDITION_BANDIT] = run_standard_condition(
            condition_name=CONDITION_BANDIT,
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run, iterations=args.iterations,
            gpu=args.gpu, include_curated=True,
            bandit=bandit_c,
        )

        print()
        print(f"Condition D: {CONDITION_NAIVE_ENSEMBLE}")
        all_results[CONDITION_NAIVE_ENSEMBLE] = run_standard_condition(
            condition_name=CONDITION_NAIVE_ENSEMBLE,
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run, iterations=args.iterations,
            gpu=args.gpu, include_curated=include_curated,
            cost_model=cost_model, bandit=bandit_d,
            is_naive_ensemble=True,
        )

        print()
        print(f"Condition E: {CONDITION_ADAPTIVE_ROUTER}")
        all_results[CONDITION_ADAPTIVE_ROUTER] = run_adaptive_router_condition(
            spec=spec, session=session, shapes=shapes,
            configs_per_run=args.configs_per_run, iterations=args.iterations,
            gpu=args.gpu, cost_model=cost_model, router_seed=42,
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
                **({"arm_choices": res.arm_choices} if res.arm_choices else {}),
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
