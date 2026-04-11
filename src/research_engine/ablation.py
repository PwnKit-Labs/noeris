"""Cross-run learning ablation study.

The core novel contribution of Noeris is the shape-indexed cross-run
config database that persists winning configs across sessions and feeds
them back to the LLM proposer. This module measures how much that
matters.

Experiment protocol:

1. **With database**: Run N iterations of triton-iterate with full
   database restoration. Proposer sees accumulated insights, selector
   uses incumbents from prior runs.

2. **Without database**: Run N iterations of triton-iterate with an
   empty database each iteration. Proposer has no priors, selector
   starts from scratch.

3. **Metric**: For each iteration, record the best metric (TFLOPS or
   GB/s) achieved. Compare trajectories: how many iterations does each
   condition need to reach within 5% of the final best?

Hypothesis: Cross-run learning reduces iterations-to-near-optimal by
2-4x compared to stateless search.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AblationCondition:
    """Results for one ablation condition (with or without database)."""

    name: str
    iterations: list[dict[str, Any]] = field(default_factory=list)

    def best_metric_so_far(self) -> list[float]:
        best = 0.0
        trajectory = []
        for it in self.iterations:
            metric = it.get("best_metric", 0)
            if metric > best:
                best = metric
            trajectory.append(best)
        return trajectory

    def iterations_to_threshold(self, target: float) -> int:
        """How many iterations to reach target metric (or infinity)."""
        for i, it in enumerate(self.iterations):
            if it.get("best_metric", 0) >= target:
                return i + 1
        return len(self.iterations) + 1


@dataclass(slots=True)
class MultiTrialReport:
    """Multi-trial ablation results with statistical summary."""

    operator: str
    hardware: str
    iterations_per_condition: int
    with_db_trials: list[list[float]] = field(default_factory=list)
    without_db_trials: list[list[float]] = field(default_factory=list)

    def summary(self) -> dict:
        import statistics as stats

        def _extract_finals(trials: list[list[float]]) -> list[float]:
            return [max(t) if t else 0 for t in trials]

        with_finals = _extract_finals(self.with_db_trials)
        without_finals = _extract_finals(self.without_db_trials)

        def _stats(values: list[float]) -> dict:
            if not values:
                return {"mean": 0, "stdev": 0, "min": 0, "max": 0, "n": 0}
            return {
                "mean": round(stats.mean(values), 2),
                "stdev": round(stats.stdev(values) if len(values) > 1 else 0, 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "n": len(values),
            }

        with_stats = _stats(with_finals)
        without_stats = _stats(without_finals)

        relative_improvement = 0.0
        if without_stats["mean"] > 0:
            relative_improvement = (
                (with_stats["mean"] - without_stats["mean"]) / without_stats["mean"]
            )

        return {
            "operator": self.operator,
            "hardware": self.hardware,
            "trials": len(self.with_db_trials),
            "iterations_per_condition": self.iterations_per_condition,
            "with_database": {
                **with_stats,
                "trial_finals": with_finals,
                "trial_trajectories": self.with_db_trials,
            },
            "without_database": {
                **without_stats,
                "trial_finals": without_finals,
                "trial_trajectories": self.without_db_trials,
            },
            "relative_improvement": round(relative_improvement, 4),
            "relative_improvement_pct": round(relative_improvement * 100, 2),
        }

    def summary_text(self) -> str:
        s = self.summary()
        w = s["with_database"]
        wo = s["without_database"]
        lines = [
            "# Multi-Trial Cross-Run Learning Ablation",
            "",
            f"Operator: `{s['operator']}`",
            f"Hardware: `{s['hardware']}`",
            f"Trials: {s['trials']}",
            f"Iterations per condition: {s['iterations_per_condition']}",
            "",
            "## Statistical Summary",
            "",
            "| Condition | Mean | StdDev | Min | Max | N |",
            "|---|---|---|---|---|---|",
            f"| **with_database**    | {w['mean']} | {w['stdev']} | {w['min']} | {w['max']} | {w['n']} |",
            f"| **without_database** | {wo['mean']} | {wo['stdev']} | {wo['min']} | {wo['max']} | {wo['n']} |",
            "",
            f"## Relative improvement: **{s['relative_improvement_pct']:+.2f}%**",
            "",
            "## Trial-Level Detail",
            "",
            f"- with_database finals: {w['trial_finals']}",
            f"- without_database finals: {wo['trial_finals']}",
        ]
        return "\n".join(lines) + "\n"


def run_multi_trial_ablation(
    *,
    operator: str,
    gpu: str = "A100",
    trials: int = 3,
    iterations: int = 5,
    configs_per_run: int = 6,
    use_llm: bool = True,
    shapes_set: str = "standard",
) -> MultiTrialReport:
    """Run the ablation with multiple independent trials.

    Each trial runs the full with/without comparison. Across trials we
    report mean, stdev, and relative improvement.
    """
    report = MultiTrialReport(
        operator=operator,
        hardware=gpu,
        iterations_per_condition=iterations,
    )

    for trial_idx in range(trials):
        single = run_ablation(
            operator=operator,
            gpu=gpu,
            iterations=iterations,
            configs_per_run=configs_per_run,
            use_llm=use_llm,
            shapes_set=shapes_set,
        )
        report.with_db_trials.append(single.with_database.best_metric_so_far())
        report.without_db_trials.append(single.without_database.best_metric_so_far())

    return report


def run_fast_multi_trial_ablation(
    *,
    operator: str,
    gpu: str = "A100",
    trials: int = 3,
    iterations: int = 5,
    configs_per_run: int = 6,
    use_llm: bool = True,
    shapes_set: str = "standard",
) -> MultiTrialReport:
    """Fast multi-trial ablation using a single persistent Modal session.

    All 2*trials*iterations benchmark calls happen inside ONE Modal
    ``with app.run()`` context, so the container stays warm. Per-call
    overhead drops from ~10-15s (subprocess spawn + modal orchestration)
    to ~1-3s (just the .remote() round trip).
    """
    from .modal_session import ModalBenchmarkSession
    from .triton_operators import REGISTRY, select_configs_for_operator
    from .triton_kernels import ConfigDatabase

    spec = REGISTRY.get(operator)

    if shapes_set == "tiny":
        shapes = spec.shape_buckets[:2]
    elif shapes_set == "full":
        shapes = spec.shape_buckets
    else:
        shapes = spec.shape_buckets[:6]

    llm_client = None
    if use_llm:
        try:
            from .llm import LlmConfigurationError, ResponsesApiClient
            llm_client = ResponsesApiClient.from_environment()
        except Exception:
            llm_client = None

    report = MultiTrialReport(
        operator=operator,
        hardware=gpu,
        iterations_per_condition=iterations,
    )

    with ModalBenchmarkSession(gpu=gpu) as session:
        for trial_idx in range(trials):
            # --- without_database condition: fresh DB each iteration ---
            without_trajectory = []
            best_so_far_without = 0.0
            for iter_idx in range(iterations):
                db = ConfigDatabase(path=Path(f"/tmp/ablation-nodb-{trial_idx}-{iter_idx}.json"))
                metric = _run_one_session_iteration(
                    spec=spec,
                    session=session,
                    database=db,
                    shapes=shapes,
                    configs_per_run=configs_per_run,
                    llm_client=llm_client,
                    operator=operator,
                    gpu=gpu,
                )
                if metric > best_so_far_without:
                    best_so_far_without = metric
                without_trajectory.append(best_so_far_without)
                try:
                    db.path.unlink(missing_ok=True)
                except Exception:
                    pass
            report.without_db_trials.append(without_trajectory)

            # --- with_database condition: persistent DB across iterations ---
            shared_db_path = Path(f"/tmp/ablation-withdb-{trial_idx}.json")
            shared_db_path.unlink(missing_ok=True)
            shared_db = ConfigDatabase(path=shared_db_path)
            with_trajectory = []
            best_so_far_with = 0.0
            for iter_idx in range(iterations):
                metric = _run_one_session_iteration(
                    spec=spec,
                    session=session,
                    database=shared_db,
                    shapes=shapes,
                    configs_per_run=configs_per_run,
                    llm_client=llm_client,
                    operator=operator,
                    gpu=gpu,
                )
                if metric > best_so_far_with:
                    best_so_far_with = metric
                with_trajectory.append(best_so_far_with)
            report.with_db_trials.append(with_trajectory)
            shared_db_path.unlink(missing_ok=True)

    return report


def _run_one_session_iteration(
    *,
    spec,
    session,
    database,
    shapes: list,
    configs_per_run: int,
    llm_client,
    operator: str,
    gpu: str,
) -> float:
    """Run one iteration within a persistent Modal session.

    Performs: LLM proposal → select configs → send to warm Modal function →
    parse results → record to database → return best metric.
    """
    from .triton_operators import select_configs_for_operator

    proposed_configs: list[dict[str, int]] = []
    if llm_client is not None:
        try:
            from .cli import _propose_operator_configs
            proposal_result = _propose_operator_configs(
                spec=spec,
                proposer=llm_client,
                database=database,
                hardware=gpu,
                target_shapes=shapes,
            )
            proposed_configs = proposal_result.get("configs", [])
        except Exception:
            proposed_configs = []

    configs = select_configs_for_operator(
        spec=spec,
        database=database,
        hardware=gpu,
        shapes=shapes,
        max_configs=configs_per_run,
        proposed_configs=proposed_configs,
    )
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
                operator,
                shape_result.get("shape", ""),
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


def _parse_shape_for_operator(operator: str, shape_str: str) -> dict | None:
    """Parse operator-specific shape strings from benchmark output."""
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
    except (ValueError, IndexError):
        return None
    return None


@dataclass(slots=True)
class AblationReport:
    operator: str
    hardware: str
    with_database: AblationCondition
    without_database: AblationCondition

    def summary(self) -> dict:
        with_best = max(
            (it.get("best_metric", 0) for it in self.with_database.iterations),
            default=0,
        )
        without_best = max(
            (it.get("best_metric", 0) for it in self.without_database.iterations),
            default=0,
        )
        target = max(with_best, without_best) * 0.95

        return {
            "operator": self.operator,
            "hardware": self.hardware,
            "iterations_per_condition": len(self.with_database.iterations),
            "with_database": {
                "final_best": with_best,
                "trajectory": self.with_database.best_metric_so_far(),
                "iterations_to_95pct": self.with_database.iterations_to_threshold(target),
            },
            "without_database": {
                "final_best": without_best,
                "trajectory": self.without_database.best_metric_so_far(),
                "iterations_to_95pct": self.without_database.iterations_to_threshold(target),
            },
            "target_95pct": round(target, 2),
            "speedup_to_convergence": (
                self.without_database.iterations_to_threshold(target)
                / max(1, self.with_database.iterations_to_threshold(target))
            ),
        }

    def summary_text(self) -> str:
        s = self.summary()
        lines = [
            "# Cross-Run Learning Ablation Report",
            "",
            f"Operator: `{s['operator']}`",
            f"Hardware: `{s['hardware']}`",
            f"Iterations per condition: {s['iterations_per_condition']}",
            f"95% threshold: {s['target_95pct']:.2f}",
            "",
            "## Condition Comparison",
            "",
            "| Condition | Final best | Iterations to 95% | Trajectory |",
            "|---|---|---|---|",
            (
                f"| **with_database** | {s['with_database']['final_best']:.2f} | "
                f"{s['with_database']['iterations_to_95pct']} | "
                f"{s['with_database']['trajectory']} |"
            ),
            (
                f"| **without_database** | {s['without_database']['final_best']:.2f} | "
                f"{s['without_database']['iterations_to_95pct']} | "
                f"{s['without_database']['trajectory']} |"
            ),
            "",
            f"## Result: {s['speedup_to_convergence']:.2f}x speedup to convergence",
            "",
            "Cross-run learning reaches 95% of the final best in "
            f"{s['with_database']['iterations_to_95pct']} iterations vs "
            f"{s['without_database']['iterations_to_95pct']} without.",
        ]
        return "\n".join(lines) + "\n"


def run_ablation(
    *,
    operator: str,
    gpu: str = "A100",
    iterations: int = 5,
    configs_per_run: int = 8,
    use_llm: bool = True,
    shapes_set: str = "standard",
    warm_up_database: bool = False,
    warm_up_iterations: int = 3,
) -> AblationReport:
    """Run the ablation: with-database vs without-database.

    Args:
        operator: Which operator to ablate on.
        gpu: Modal GPU type.
        iterations: Number of search iterations per condition.
        configs_per_run: Configs to benchmark per iteration.
        use_llm: Use the LLM proposer (which reads database insights).
        shapes_set: tiny / standard / full.
        warm_up_database: If True, run warm_up_iterations first without
            measuring, to seed the database before the with-database
            condition.
    """
    import argparse
    from .cli import _run_triton_iterate

    def _run_one_iteration(db_path: Path) -> dict:
        """Run one iteration and return the best metric."""
        args = argparse.Namespace(
            operator=operator,
            configs_per_run=configs_per_run,
            gpu=gpu,
            llm=use_llm,
            local=False,
            shapes=shapes_set,
            db_path=str(db_path),
        )
        # Capture stdout by redirecting temporarily
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _run_triton_iterate(args)
        output = buf.getvalue()
        try:
            data = json.loads(output)
            return {
                "best_metric": data.get("best_avg_tflops") or data.get("best_avg_gb_per_s") or 0,
                "best_config_id": data.get("best_config_id", ""),
                "configs_tested": data.get("configs_tested", 0),
                "proposal_source": data.get("proposal", {}).get("source", ""),
            }
        except json.JSONDecodeError:
            return {"best_metric": 0, "error": "parse_error"}

    report = AblationReport(
        operator=operator,
        hardware=gpu,
        with_database=AblationCondition(name="with_database"),
        without_database=AblationCondition(name="without_database"),
    )

    # Without-database condition: fresh empty database each iteration
    for i in range(iterations):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = Path(f.name)
        try:
            result = _run_one_iteration(db_path)
            result["iteration"] = i
            report.without_database.iterations.append(result)
        finally:
            db_path.unlink(missing_ok=True)

    # With-database condition: persistent database across iterations
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        shared_db_path = Path(f.name)
    try:
        # Optional warmup to seed the database
        if warm_up_database:
            for _ in range(warm_up_iterations):
                _run_one_iteration(shared_db_path)

        for i in range(iterations):
            result = _run_one_iteration(shared_db_path)
            result["iteration"] = i
            report.with_database.iterations.append(result)
    finally:
        shared_db_path.unlink(missing_ok=True)

    return report
