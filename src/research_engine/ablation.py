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
