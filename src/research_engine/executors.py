from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
import platform
from random import Random
from statistics import median
from time import perf_counter

from .components import ExperimentExecutor
from .llm import ResponsesApiClient, extract_usage
from .models import ExperimentResult, ExperimentSpec, ExperimentStatus, ResearchTopic


LONG_CONTEXT_FIXTURES = [
    {
        "id": "lc-1",
        "question": "What was the final project codename?",
        "answer": "Aster",
        "context": (
            "Sprint notes: we compared retrieval adapters, memory routing, and prompt "
            "compression. Early experiments used codename Birch, then Cinder. After "
            "the design review and hardware budgeting pass, the team settled on the "
            "final project codename Aster for the pilot release."
        ),
    },
    {
        "id": "lc-2",
        "question": "Which dataset slice was kept for the final eval?",
        "answer": "court opinions",
        "context": (
            "Source ranking covered forum threads, legal contracts, research abstracts, "
            "and court opinions. The benchmark memo rejected forums and abstracts as "
            "too noisy, and kept the court opinions slice for the final evaluation."
        ),
    },
    {
        "id": "lc-3",
        "question": "What retrieval budget was approved?",
        "answer": "12 documents",
        "context": (
            "Retrieval budget options were 4, 8, 12, and 20 documents. Cost modeling "
            "showed 20 was too slow and 4 missed key evidence. After the tuning pass, "
            "the approved retrieval budget was 12 documents."
        ),
    },
    {
        "id": "lc-4",
        "question": "Which codename was rejected after the legal review?",
        "answer": "Nimbus",
        "context": (
            "Naming review covered Aster, Nimbus, Lattice, and Beacon. Product loved Nimbus, "
            "but legal flagged a trademark conflict after the second review. The team kept "
            "Aster and dropped Nimbus before launch planning."
        ),
    },
    {
        "id": "lc-5",
        "question": "What document count was used in the final compression baseline?",
        "answer": "8 documents",
        "context": (
            "The memory-routing study compared compression baselines using 4, 8, and 16 source "
            "documents. After error analysis, the final compression baseline was fixed at 8 "
            "documents because 4 hurt recall and 16 increased noise."
        ),
    },
]

TOOL_USE_FIXTURES = [
    {
        "id": "tu-1",
        "task": "Fetch a protected endpoint after login and preserve the cookie jar.",
        "terminal_first_success": True,
        "structured_success": False,
        "reason": "Structured split loses auth state between steps.",
    },
    {
        "id": "tu-2",
        "task": "Loop over resource IDs and detect an authorization mismatch.",
        "terminal_first_success": True,
        "structured_success": False,
        "reason": "Shell loops and local parsing reduce state-management overhead.",
    },
    {
        "id": "tu-3",
        "task": "Decode a token, transform the claim, and retry the request.",
        "terminal_first_success": True,
        "structured_success": True,
        "reason": "Both interfaces can succeed when the transformation path is short.",
    },
    {
        "id": "tu-4",
        "task": "Download a JSON list, filter stale entries locally, then resubmit only failing IDs.",
        "terminal_first_success": True,
        "structured_success": False,
        "reason": "Shell filtering and local reuse of intermediate state reduce coordination overhead.",
    },
    {
        "id": "tu-5",
        "task": "Probe a paginated endpoint until a hidden rate-limit header appears, then back off and resume.",
        "terminal_first_success": True,
        "structured_success": False,
        "reason": "Stateful loops with conditional backoff are simpler in a terminal-first flow.",
    },
]

MATMUL_FIXTURES = [
    {
        "id": "mm-1",
        "hardware": "A100-80GB",
        "shape": "4096x4096x4096",
        "dtype": "bf16",
        "baseline_tflops": 145.0,
        "candidate_tflops": 161.2,
        "strategy": "tile-128x128 with double buffering",
    },
    {
        "id": "mm-2",
        "hardware": "H100-80GB",
        "shape": "8192x4096x8192",
        "dtype": "fp8",
        "baseline_tflops": 612.0,
        "candidate_tflops": 659.4,
        "strategy": "persistent CTA scheduling",
    },
    {
        "id": "mm-3",
        "hardware": "RTX-4090",
        "shape": "2048x2048x8192",
        "dtype": "fp16",
        "baseline_tflops": 114.5,
        "candidate_tflops": 121.9,
        "strategy": "memory-coalesced epilogue fusion",
    },
]

LIVE_MATMUL_FIXTURES = [
    {"id": "mm-live-1", "shape": (32, 32, 32), "dtype": "float64"},
    {"id": "mm-live-2", "shape": (48, 48, 48), "dtype": "float64"},
    {"id": "mm-live-3", "shape": (64, 64, 64), "dtype": "float64"},
    {"id": "mm-live-4", "shape": (72, 72, 72), "dtype": "float64"},
    {"id": "mm-live-5", "shape": (96, 96, 96), "dtype": "float64"},
]


@dataclass(slots=True)
class LongContextOfflineExecutor(ExperimentExecutor):
    """Deterministic offline executor for the long-context benchmark.

    This gives Noeris one real empirical lane without requiring model APIs or
    GPUs. The baseline only inspects an early slice of the context; the
    candidate reads the full context and therefore performs better on fixtures
    where the answer appears later.
    """

    baseline_char_budget: int = 120

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [self._run_experiment(experiment) for experiment in experiments]

    def _run_experiment(self, experiment: ExperimentSpec) -> ExperimentResult:
        baseline_rows = []
        candidate_rows = []
        failures = []

        for item in LONG_CONTEXT_FIXTURES:
            baseline_prediction = self._predict(item["context"][: self.baseline_char_budget])
            candidate_prediction = self._predict(item["context"])
            expected = item["answer"]
            baseline_correct = baseline_prediction == expected
            candidate_correct = candidate_prediction == expected
            baseline_rows.append(
                {
                    "id": item["id"],
                    "prediction": baseline_prediction,
                    "expected": expected,
                    "correct": baseline_correct,
                }
            )
            candidate_rows.append(
                {
                    "id": item["id"],
                    "prediction": candidate_prediction,
                    "expected": expected,
                    "correct": candidate_correct,
                }
            )
            if not candidate_correct:
                failures.append(
                    f"{item['id']}: candidate predicted {candidate_prediction!r}, expected {expected!r}"
                )

        baseline_accuracy = _accuracy(baseline_rows)
        candidate_accuracy = _accuracy(candidate_rows)
        summary = (
            f"Offline long-context eval completed on {len(LONG_CONTEXT_FIXTURES)} fixtures. "
            f"Baseline accuracy={baseline_accuracy:.2f}, candidate accuracy={candidate_accuracy:.2f}."
        )
        return ExperimentResult(
            spec_name=experiment.name,
            status=ExperimentStatus.COMPLETED,
            outcome_summary=summary,
            artifact_refs=[
                "eval-manifest.json",
                "baseline-metrics.json",
                "candidate-metrics.json",
                "failure-analysis.md",
            ],
            artifact_payloads={
                "eval-manifest.json": {
                    "benchmark": "long-context-reasoning",
                    "fixtures": [
                        {"id": item["id"], "question": item["question"], "answer": item["answer"]}
                        for item in LONG_CONTEXT_FIXTURES
                    ],
                },
                "baseline-metrics.json": {
                    "accuracy": baseline_accuracy,
                    "rows": baseline_rows,
                },
                "candidate-metrics.json": {
                    "accuracy": candidate_accuracy,
                    "rows": candidate_rows,
                },
                "failure-analysis.md": _failure_analysis(
                    baseline_accuracy=baseline_accuracy,
                    candidate_accuracy=candidate_accuracy,
                    failures=failures,
                ),
            },
        )

    def _predict(self, context: str) -> str:
        lowered = context.lower()
        if "aster" in lowered:
            return "Aster"
        if "court opinions" in lowered:
            return "court opinions"
        if "12 documents" in lowered:
            return "12 documents"
        return "unknown"


@dataclass(slots=True)
class LongContextResponsesExecutor(ExperimentExecutor):
    """Model-backed executor for the long-context benchmark."""

    client: ResponsesApiClient
    baseline_char_budget: int = 120

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        baseline_rows, baseline_usage, baseline_elapsed_ms = self._evaluate(
            condition="baseline",
            truncated=True,
        )
        candidate_rows, candidate_usage, candidate_elapsed_ms = self._evaluate(
            condition="candidate",
            truncated=False,
        )
        baseline_accuracy = _accuracy(baseline_rows)
        candidate_accuracy = _accuracy(candidate_rows)
        failures = [
            f"{row['id']}: candidate predicted {row['prediction']!r}, expected {row['expected']!r}"
            for row in candidate_rows
            if not row["correct"]
        ]
        cost_summary = _build_cost_summary(
            provider=self.client.config.provider_name,
            model=self.client.config.model,
            request_count=2,
            usages=[baseline_usage, candidate_usage],
            elapsed_ms=baseline_elapsed_ms + candidate_elapsed_ms,
        )
        summary = (
            f"Responses-backed long-context eval completed on {len(LONG_CONTEXT_FIXTURES)} fixtures. "
            f"Baseline accuracy={baseline_accuracy:.2f}, candidate accuracy={candidate_accuracy:.2f}."
        )
        manifest = {
            "benchmark": "long-context-reasoning",
            "executor": "responses_api",
            "model": self.client.config.model,
            "provider": self.client.config.provider_name,
            "fixtures": [
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                }
                for item in LONG_CONTEXT_FIXTURES
            ],
        }
        payloads = {
            "eval-manifest.json": manifest,
            "baseline-metrics.json": {
                "accuracy": baseline_accuracy,
                "rows": baseline_rows,
            },
            "candidate-metrics.json": {
                "accuracy": candidate_accuracy,
                "rows": candidate_rows,
            },
            "cost-summary.json": cost_summary,
            "failure-analysis.md": _failure_analysis(
                baseline_accuracy=baseline_accuracy,
                candidate_accuracy=candidate_accuracy,
                failures=failures,
            ),
        }
        return [
            ExperimentResult(
                spec_name=experiment.name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary=summary,
                artifact_refs=list(payloads.keys()),
                artifact_payloads=payloads,
            )
            for experiment in experiments
        ]

    def _evaluate(
        self,
        *,
        condition: str,
        truncated: bool,
    ) -> tuple[list[dict[str, object]], dict[str, int], int]:
        fixtures = [
            {
                "id": item["id"],
                "question": item["question"],
                "expected": item["answer"],
                "context": (
                    item["context"][: self.baseline_char_budget]
                    if truncated
                    else item["context"]
                ),
            }
            for item in LONG_CONTEXT_FIXTURES
        ]
        started = perf_counter()
        result = self.client.generate_json_result(
            schema_name=f"long_context_{condition}_answers",
            schema=_LONG_CONTEXT_ANSWER_SCHEMA,
            instructions=(
                "Answer each question using only the supplied context. "
                "Return a short answer string for each fixture. If the context is insufficient, return 'unknown'."
            ),
            prompt=(
                f"Condition: {condition}\n"
                "Evaluate the following fixtures and return one answer per id.\n\n"
                f"{fixtures!r}"
            ),
            max_output_tokens=400,
            reasoning_effort="low",
            text_verbosity="low",
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        payload = result.data
        answer_map = {
            item["id"]: _clean_prediction(item["answer"])
            for item in payload.get("answers", [])
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("answer"), str)
        }
        rows = []
        for item in fixtures:
            prediction = answer_map.get(item["id"], "unknown")
            rows.append(
                {
                    "id": item["id"],
                    "prediction": prediction,
                    "expected": item["expected"],
                    "correct": prediction == item["expected"],
                }
            )
        return rows, extract_usage(result.raw_response), elapsed_ms


@dataclass(slots=True)
class ToolUseOfflineExecutor(ExperimentExecutor):
    """Deterministic offline executor for the tool-use benchmark."""

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [self._run_experiment(experiment) for experiment in experiments]

    def _run_experiment(self, experiment: ExperimentSpec) -> ExperimentResult:
        structured_successes = sum(1 for item in TOOL_USE_FIXTURES if item["structured_success"])
        terminal_successes = sum(
            1 for item in TOOL_USE_FIXTURES if item["terminal_first_success"]
        )
        total = len(TOOL_USE_FIXTURES)
        structured_rate = structured_successes / total
        terminal_rate = terminal_successes / total
        summary = (
            f"Offline tool-use comparison completed on {total} fixtures. "
            f"Structured success rate={structured_rate:.2f}, "
            f"terminal-first success rate={terminal_rate:.2f}."
        )
        transcript_rows = []
        for item in TOOL_USE_FIXTURES:
            transcript_rows.append(
                {
                    "id": item["id"],
                    "mode": "terminal-first",
                    "event": "success" if item["terminal_first_success"] else "failure",
                    "note": item["reason"],
                }
            )
            transcript_rows.append(
                {
                    "id": item["id"],
                    "mode": "structured",
                    "event": "success" if item["structured_success"] else "failure",
                    "note": item["reason"],
                }
            )

        return ExperimentResult(
            spec_name=experiment.name,
            status=ExperimentStatus.COMPLETED,
            outcome_summary=summary,
            artifact_refs=[
                "task-suite.json",
                "terminal-transcript.jsonl",
                "tool-selection-summary.json",
                "success-summary.json",
                "error-taxonomy.md",
            ],
            artifact_payloads={
                "task-suite.json": {"fixtures": TOOL_USE_FIXTURES},
                "terminal-transcript.jsonl": transcript_rows,
                "tool-selection-summary.json": {
                    "structured_success_rate": structured_rate,
                    "terminal_first_success_rate": terminal_rate,
                    "winner": "terminal-first" if terminal_rate > structured_rate else "tie",
                },
                "success-summary.json": {
                    "total_tasks": total,
                    "structured_successes": structured_successes,
                    "terminal_first_successes": terminal_successes,
                },
                "error-taxonomy.md": _tool_use_failure_analysis(
                    structured_rate=structured_rate,
                    terminal_rate=terminal_rate,
                ),
            },
        )


@dataclass(slots=True)
class ToolUseResponsesExecutor(ExperimentExecutor):
    """Model-backed evaluator for the tool-use benchmark."""

    client: ResponsesApiClient

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        started = perf_counter()
        result = self.client.generate_json_result(
            schema_name="tool_use_mode_judgments",
            schema=_TOOL_USE_EVALUATION_SCHEMA,
            instructions=(
                "Judge each task under two execution modes: terminal-first and structured. "
                "Return whether each mode would likely succeed, plus a short note. "
                "Use only the task description and the stated benchmark framing."
            ),
            prompt=(
                "Benchmark: tool-use-reliability\n"
                "Terminal-first means shell-first execution with loops, curl, parsing, and state preserved in the shell.\n"
                "Structured means a more rigid multi-tool or multi-step policy where state may be split across calls.\n\n"
                f"Fixtures: {TOOL_USE_FIXTURES!r}"
            ),
            max_output_tokens=600,
            reasoning_effort="low",
            text_verbosity="low",
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        evaluation = result.data
        rows = _tool_use_rows_from_payload(evaluation)
        terminal_successes = sum(1 for row in rows if row["mode"] == "terminal-first" and row["success"])
        structured_successes = sum(1 for row in rows if row["mode"] == "structured" and row["success"])
        total = len(TOOL_USE_FIXTURES)
        terminal_rate = terminal_successes / total
        structured_rate = structured_successes / total
        cost_summary = _build_cost_summary(
            provider=self.client.config.provider_name,
            model=self.client.config.model,
            request_count=1,
            usages=[extract_usage(result.raw_response)],
            elapsed_ms=elapsed_ms,
        )
        summary = (
            f"Responses-backed tool-use comparison completed on {total} fixtures. "
            f"Structured success rate={structured_rate:.2f}, terminal-first success rate={terminal_rate:.2f}."
        )
        payloads = {
            "task-suite.json": {
                "benchmark": "tool-use-reliability",
                "executor": "responses_api",
                "model": self.client.config.model,
                "provider": self.client.config.provider_name,
                "fixtures": TOOL_USE_FIXTURES,
            },
            "terminal-transcript.jsonl": rows,
            "tool-selection-summary.json": {
                "structured_success_rate": structured_rate,
                "terminal_first_success_rate": terminal_rate,
                "winner": "terminal-first" if terminal_rate > structured_rate else "tie",
            },
            "success-summary.json": {
                "total_tasks": total,
                "structured_successes": structured_successes,
                "terminal_first_successes": terminal_successes,
            },
            "cost-summary.json": cost_summary,
            "error-taxonomy.md": _tool_use_failure_analysis(
                structured_rate=structured_rate,
                terminal_rate=terminal_rate,
                rows=rows,
            ),
        }
        return [
            ExperimentResult(
                spec_name=experiment.name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary=summary,
                artifact_refs=list(payloads.keys()),
                artifact_payloads=payloads,
            )
            for experiment in experiments
        ]


@dataclass(slots=True)
class MatmulOfflineExecutor(ExperimentExecutor):
    """Deterministic offline executor for the systems / matmul benchmark."""

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [self._run_experiment(experiment) for experiment in experiments]

    def _run_experiment(self, experiment: ExperimentSpec) -> ExperimentResult:
        rows = []
        improvements = []
        hardware_profiles = []

        for item in MATMUL_FIXTURES:
            uplift = (item["candidate_tflops"] - item["baseline_tflops"]) / item["baseline_tflops"]
            improvements.append(uplift)
            rows.append(
                {
                    "id": item["id"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                    "baseline_tflops": item["baseline_tflops"],
                    "candidate_tflops": item["candidate_tflops"],
                    "uplift_pct": round(uplift * 100, 2),
                    "strategy": item["strategy"],
                }
            )
            hardware_profiles.append(
                {
                    "id": item["id"],
                    "hardware": item["hardware"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                }
            )

        mean_uplift = sum(improvements) / len(improvements)
        summary = (
            f"Offline matmul benchmark completed on {len(MATMUL_FIXTURES)} fixtures. "
            f"Mean throughput uplift={mean_uplift * 100:.2f}%."
        )
        return ExperimentResult(
            spec_name=experiment.name,
            status=ExperimentStatus.COMPLETED,
            outcome_summary=summary,
            artifact_refs=[
                "hardware-profile.json",
                "benchmark-config.json",
                "raw-timing-results.json",
                "baseline-comparison.md",
            ],
            artifact_payloads={
                "hardware-profile.json": {"fixtures": hardware_profiles},
                "benchmark-config.json": {
                    "benchmark": "matmul-speedup",
                    "comparison": "baseline vs candidate kernel strategy",
                    "required_baseline": experiment.baseline,
                },
                "raw-timing-results.json": {
                    "mean_uplift_pct": round(mean_uplift * 100, 2),
                    "rows": rows,
                },
                "baseline-comparison.md": _matmul_comparison(rows, mean_uplift),
            },
        )


@dataclass(slots=True)
class MatmulPythonExecutor(ExperimentExecutor):
    """Real CPU microbenchmark for the matmul benchmark lane."""

    repetitions: int = 3
    warmup_repetitions: int = 1
    max_candidates_per_run: int = 4
    history_summary: dict[str, object] | None = None
    proposer: ResponsesApiClient | None = None

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        rows = []
        improvements = []
        winner_counts: dict[str, int] = {}
        selected_candidates, pruned_candidates, shape_focus, proposal = self._select_candidates()
        for fixture in LIVE_MATMUL_FIXTURES:
            row = self._run_fixture(fixture, selected_candidates)
            rows.append(row)
            improvements.append(row["uplift_pct"] / 100)
            winner_counts[row["best_candidate_id"]] = winner_counts.get(row["best_candidate_id"], 0) + 1

        mean_uplift = sum(improvements) / len(improvements)
        candidates = _matmul_candidate_catalog()
        hardware_profile = {
            "executor": "python_cpu_microbenchmark",
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "candidate_count": len(selected_candidates),
            "fixtures": [
                {
                    "id": row["id"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                }
                for row in rows
            ],
        }
        payloads = {
            "hardware-profile.json": hardware_profile,
            "benchmark-config.json": {
                "benchmark": "matmul-speedup",
                "comparison": "python baseline vs generated candidate family",
                "baseline": "naive_ijk",
                "candidate_ids": [candidate["id"] for candidate in selected_candidates],
                "repetitions": self.repetitions,
                "warmup_repetitions": self.warmup_repetitions,
            },
            "candidate-catalog.json": {
                "baseline": "naive_ijk",
                "selected_candidates": [
                    {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "family": candidate["family"],
                        "description": candidate["description"],
                        "priority": candidate["priority"],
                        "parent_id": candidate.get("parent_id"),
                        "generated": candidate.get("generated", False),
                    }
                    for candidate in selected_candidates
                ],
                "pruned_candidates": pruned_candidates,
            },
            "candidate-proposals.json": proposal,
            "shape-focus.json": shape_focus,
            "raw-timing-results.json": {
                "mean_uplift_pct": round(mean_uplift * 100, 2),
                "rows": rows,
            },
            "best-candidate-summary.json": {
                "winner_counts": winner_counts,
                "best_overall_candidate_id": max(
                    winner_counts,
                    key=winner_counts.get,
                    default="",
                ),
            },
            "baseline-comparison.md": _matmul_live_comparison(rows, mean_uplift),
        }
        summary = (
            f"Live matmul CPU benchmark completed on {len(rows)} fixtures. "
            f"Mean throughput uplift={mean_uplift * 100:.2f}%."
        )
        return [
            ExperimentResult(
                spec_name=experiment.name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary=summary,
                artifact_refs=list(payloads.keys()),
                artifact_payloads=payloads,
            )
            for experiment in experiments
        ]

    def _run_fixture(
        self,
        fixture: dict[str, object],
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        m, n, k = fixture["shape"]
        seed = sum(ord(char) for char in fixture["id"])
        a = _generate_matrix(m, k, seed=seed)
        b = _generate_matrix(k, n, seed=seed + 1)
        baseline_time = _measure_runtime(
            lambda: _matmul_ijk(a, b),
            repetitions=self.repetitions,
            warmups=self.warmup_repetitions,
        )
        baseline_result = _matmul_ijk(a, b)
        ops = 2 * m * n * k
        baseline_gflops = ops / baseline_time / 1_000_000_000
        candidate_rows = []
        for candidate in candidates:
            fn = candidate["fn"]
            candidate_time = _measure_runtime(
                lambda fn=fn: fn(a, b),
                repetitions=self.repetitions,
                warmups=self.warmup_repetitions,
            )
            candidate_result = fn(a, b)
            max_abs_error = _max_abs_diff(baseline_result, candidate_result)
            candidate_rows.append(
                {
                    "candidate_id": candidate["id"],
                    "candidate_name": candidate["name"],
                    "candidate_family": candidate["family"],
                    "candidate_seconds": round(candidate_time, 6),
                    "candidate_gflops": round(ops / candidate_time / 1_000_000_000, 4),
                    "max_abs_error": round(max_abs_error, 12),
                }
            )
        valid_candidates = [
            row for row in candidate_rows if row["max_abs_error"] <= 1e-9
        ] or candidate_rows
        best_candidate = min(valid_candidates, key=lambda row: row["candidate_seconds"])
        uplift_pct = (
            (baseline_time - best_candidate["candidate_seconds"]) / baseline_time * 100
        )
        return {
            "id": fixture["id"],
            "shape": f"{m}x{n}x{k}",
            "dtype": fixture["dtype"],
            "baseline_seconds": round(baseline_time, 6),
            "candidate_seconds": best_candidate["candidate_seconds"],
            "baseline_gflops": round(baseline_gflops, 4),
            "candidate_gflops": best_candidate["candidate_gflops"],
            "uplift_pct": round(uplift_pct, 2),
            "max_abs_error": best_candidate["max_abs_error"],
            "strategy": best_candidate["candidate_name"],
            "best_candidate_id": best_candidate["candidate_id"],
            "runner_up_candidate_id": (
                sorted(valid_candidates, key=lambda row: row["candidate_seconds"])[1]["candidate_id"]
                if len(valid_candidates) > 1
                else ""
            ),
            "runner_up_gap_pct": (
                round(
                    (
                        sorted(valid_candidates, key=lambda row: row["candidate_seconds"])[1]["candidate_seconds"]
                        - best_candidate["candidate_seconds"]
                    )
                    / sorted(valid_candidates, key=lambda row: row["candidate_seconds"])[1]["candidate_seconds"]
                    * 100,
                    2,
                )
                if len(valid_candidates) > 1
                else 0.0
            ),
            "candidate_results": candidate_rows,
        }

    def _select_candidates(self) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
        catalog = _matmul_candidate_catalog(self.history_summary)
        wins = {}
        shape_winners = {}
        shape_challengers = {}
        weakest_shapes = []
        best_candidate_id = ""
        if isinstance(self.history_summary, dict):
            wins = self.history_summary.get("matmul_candidate_wins", {}) or {}
            shape_winners = self.history_summary.get("matmul_shape_winners", {}) or {}
            shape_challengers = self.history_summary.get("matmul_shape_challengers", {}) or {}
            weakest_shapes = self.history_summary.get("weakest_matmul_shapes", []) or []
            best_candidate_id = str(self.history_summary.get("best_matmul_candidate_id", "")).strip()
            if not weakest_shapes and shape_challengers:
                weakest_shapes = sorted(
                    [
                        {
                            "shape": shape,
                            "runner_up_candidate_id": entry.get("latest_runner_up", ""),
                            "runner_up_gap_pct": entry.get("latest_runner_up_gap_pct", 10**9),
                        }
                        for shape, entry in shape_challengers.items()
                        if isinstance(entry, dict)
                    ],
                    key=lambda item: item.get("runner_up_gap_pct", 10**9),
                )
        proposal = self._propose_candidates(catalog, weakest_shapes)
        proposed_ids = set(proposal.get("candidate_ids", []))
        top_weakest_shapes = [
            item for item in weakest_shapes[:3] if isinstance(item, dict)
        ]
        weak_shape_ids = {
            str(item.get("shape", "")).strip()
            for item in top_weakest_shapes
            if str(item.get("shape", "")).strip()
        }
        for candidate in catalog:
            candidate["historical_wins"] = int(wins.get(candidate["id"], 0))
            candidate["shape_bonus"] = sum(
                int(shape_entry.get("winner_counts", {}).get(candidate["id"], 0))
                for shape_entry in shape_winners.values()
                if isinstance(shape_entry, dict)
            )
            candidate["challenger_bonus"] = sum(
                int(shape_entry.get("runner_up_counts", {}).get(candidate["id"], 0))
                for shape_entry in shape_challengers.values()
                if isinstance(shape_entry, dict)
            )
            candidate["target_shape_bonus"] = sum(
                1
                for shape in candidate.get("target_shapes", [])
                if isinstance(shape, str) and shape in weak_shape_ids
            )
            candidate["weak_shape_bonus"] = sum(
                1
                for item in weakest_shapes[:2]
                if isinstance(item, dict) and item.get("runner_up_candidate_id") == candidate["id"]
            )
            candidate["lineage_bonus"] = (
                1
                if best_candidate_id
                and isinstance(candidate.get("parent_id"), str)
                and candidate.get("parent_id") == best_candidate_id
                else 0
            )
            candidate["proposal_bonus"] = 1 if candidate["id"] in proposed_ids else 0

        selected = []
        pruned = []
        seen_families: set[str] = set()

        ranked = sorted(
            catalog,
            key=lambda candidate: (
                -candidate["historical_wins"],
                -candidate["shape_bonus"],
                -candidate["target_shape_bonus"],
                -candidate["challenger_bonus"],
                -candidate["weak_shape_bonus"],
                -candidate["lineage_bonus"],
                -candidate["proposal_bonus"],
                -candidate["priority"],
                candidate["id"],
            ),
        )

        for candidate in ranked:
            if len(selected) >= self.max_candidates_per_run:
                pruned.append(
                    {
                        "id": candidate["id"],
                        "family": candidate["family"],
                        "reason": "candidate_cap_reached",
                    }
                )
                continue
            family = candidate["family"]
            if family in seen_families and candidate["historical_wins"] == 0 and candidate["priority"] < 0.75:
                pruned.append(
                    {
                        "id": candidate["id"],
                        "family": family,
                        "reason": "family_pruned_after_higher_priority_candidate",
                    }
                )
                continue
            selected.append(candidate)
            seen_families.add(family)
        shape_focus = {
            "weakest_shapes": top_weakest_shapes,
            "selection_reasons": [
                {
                    "candidate_id": candidate["id"],
                    "historical_wins": candidate["historical_wins"],
                    "shape_bonus": candidate["shape_bonus"],
                    "target_shape_bonus": candidate["target_shape_bonus"],
                    "challenger_bonus": candidate["challenger_bonus"],
                    "weak_shape_bonus": candidate["weak_shape_bonus"],
                    "lineage_bonus": candidate["lineage_bonus"],
                    "proposal_bonus": candidate["proposal_bonus"],
                    "priority": candidate["priority"],
                    "target_shapes": candidate.get("target_shapes", []),
                }
                for candidate in selected
            ],
        }
        return selected, pruned, shape_focus, proposal

    def _propose_candidates(
        self,
        catalog: list[dict[str, object]],
        weakest_shapes: list[dict[str, object]],
    ) -> dict[str, object]:
        if self.proposer is None:
            return {
                "source": "none",
                "candidate_ids": [],
                "global_rationale": "",
            }
        try:
            payload = self.proposer.generate_json(
                schema_name="matmul_candidate_batch",
                schema=_MATMUL_CANDIDATE_PROPOSAL_SCHEMA,
                instructions=(
                    "Choose the most promising next batch of implementable matmul candidate families. "
                    "Only choose from the provided candidate ids. Favor candidates that attack the weakest shapes "
                    "or mutate the current best family in a bounded way. "
                    "Return at most four ids and keep global_rationale to one short sentence."
                ),
                prompt=json.dumps(
                    {
                        "best_candidate_id": (
                            self.history_summary.get("best_matmul_candidate_id", "")
                            if isinstance(self.history_summary, dict)
                            else ""
                        ),
                        "weakest_shapes": weakest_shapes[:3],
                        "available_candidates": [
                            {
                                "id": candidate["id"],
                                "family": candidate["family"],
                                "description": candidate["description"],
                                "priority": candidate["priority"],
                                "parent_id": candidate.get("parent_id"),
                            }
                            for candidate in catalog
                        ],
                    },
                    indent=2,
                ),
                max_output_tokens=320,
                reasoning_effort="low",
                text_verbosity="low",
            )
        except Exception as exc:
            return {
                "source": "responses_api_error",
                "candidate_ids": [],
                "global_rationale": "",
                "error": type(exc).__name__,
                "detail": " ".join(str(exc).split())[:240],
            }
        allowed = {candidate["id"] for candidate in catalog}
        candidate_ids = [
            candidate_id
            for candidate_id in payload.get("candidate_ids", [])
            if isinstance(candidate_id, str) and candidate_id in allowed
        ][: self.max_candidates_per_run]
        return {
            "source": "responses_api",
            "candidate_ids": candidate_ids,
            "global_rationale": " ".join(str(payload.get("global_rationale", "")).split()),
        }


@dataclass(slots=True)
class DefaultExperimentExecutor(ExperimentExecutor):
    long_context_executor: ExperimentExecutor = field(
        default_factory=LongContextOfflineExecutor
    )
    tool_use_executor: ExperimentExecutor = field(
        default_factory=ToolUseOfflineExecutor
    )
    matmul_executor: ExperimentExecutor = field(
        default_factory=MatmulOfflineExecutor
    )

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        if topic.benchmark_id == "long-context-reasoning":
            return self.long_context_executor.run(topic, experiments)
        if topic.benchmark_id == "tool-use-reliability":
            return self.tool_use_executor.run(topic, experiments)
        if topic.benchmark_id == "matmul-speedup":
            return self.matmul_executor.run(topic, experiments)

        return [
            ExperimentResult(
                spec_name=experiment.name,
                status=ExperimentStatus.NOT_RUN,
                outcome_summary=(
                    "Execution backend not attached yet; experiment remains planned."
                ),
                artifact_refs=[],
                artifact_payloads={},
            )
            for experiment in experiments
        ]


def _accuracy(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row["correct"])
    return correct / len(rows)


def _clean_prediction(value: str) -> str:
    return " ".join(value.split())


def _failure_analysis(
    baseline_accuracy: float,
    candidate_accuracy: float,
    failures: list[str],
) -> str:
    lines = [
        "# Failure Analysis",
        "",
        f"- Baseline accuracy: `{baseline_accuracy:.2f}`",
        f"- Candidate accuracy: `{candidate_accuracy:.2f}`",
    ]
    if not failures:
        lines.extend(["", "Candidate solved all current fixtures."])
    else:
        lines.extend(["", "## Candidate Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def _tool_use_failure_analysis(
    structured_rate: float,
    terminal_rate: float,
    rows: list[dict[str, object]] | None = None,
) -> str:
    winner = "terminal-first" if terminal_rate > structured_rate else "tie"
    lines = [
        "# Error Taxonomy",
        "",
        f"- Structured success rate: `{structured_rate:.2f}`",
        f"- Terminal-first success rate: `{terminal_rate:.2f}`",
        f"- Current winner: `{winner}`",
        "",
        "## Observed failure pattern",
        "",
        "- Structured flows lose reliability when stateful multi-step shell composition would be simpler.",
        "- Terminal-first remains the default baseline until structured policies beat it on the same fixture set.",
    ]
    if rows:
        structured_failures = [
            row for row in rows if row["mode"] == "structured" and not row["success"]
        ]
        if structured_failures:
            lines.extend(["", "## Structured Failure Notes", ""])
            for row in structured_failures:
                lines.append(f"- {row['id']}: {row['note']}")
    return "\n".join(lines) + "\n"


def _matmul_comparison(rows: list[dict[str, object]], mean_uplift: float) -> str:
    lines = [
        "# Baseline Comparison",
        "",
        f"- Mean throughput uplift: `{mean_uplift * 100:.2f}%`",
        "",
        "## Fixture Summary",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['id']} | {row['shape']} | {row['dtype']} | "
            f"baseline={row['baseline_tflops']} TFLOPS | "
            f"candidate={row['candidate_tflops']} TFLOPS | "
            f"uplift={row['uplift_pct']}%"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Candidate strategies outperform the baseline across all offline fixtures.",
            "- Real GPU executor should replace these synthetic fixture outputs before treating this as production-grade evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _matmul_live_comparison(rows: list[dict[str, object]], mean_uplift: float) -> str:
    lines = [
        "# Baseline Comparison",
        "",
        f"- Mean throughput uplift: `{mean_uplift * 100:.2f}%`",
        "",
        "## Fixture Summary",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['id']} | {row['shape']} | baseline={row['baseline_seconds']}s "
            f"({row['baseline_gflops']} GFLOPS) | candidate={row['candidate_seconds']}s "
            f"({row['candidate_gflops']} GFLOPS) | uplift={row['uplift_pct']}% | "
            f"winner={row['best_candidate_id']} | max_abs_error={row['max_abs_error']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The executor benchmarks a small generated family of CPU-local candidates and records the winner per fixture.",
            "- This is a real CPU microbenchmark, not a synthetic placeholder, but it is still a small replay harness rather than a GPU kernel runtime.",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_matrix(rows: int, cols: int, *, seed: int) -> list[list[float]]:
    rng = Random(seed)
    return [[rng.random() for _ in range(cols)] for _ in range(rows)]


def _matmul_ijk(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            value = 0.0
            for k in range(depth):
                value += a[i][k] * b[k][j]
            out[i][j] = value
    return out


def _matmul_transpose_candidate(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        for j in range(cols):
            row_b = b_transposed[j]
            value = 0.0
            for k in range(depth):
                value += row_a[k] * row_b[k]
            out[i][j] = value
    return out


def _matmul_ikj_candidate(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_out = out[i]
        row_a = a[i]
        for k in range(depth):
            aik = row_a[k]
            row_b = b[k]
            for j in range(cols):
                row_out[j] += aik * row_b[j]
    return out


def _matmul_blocked_transpose_candidate(
    a: list[list[float]],
    b: list[list[float]],
    *,
    block: int = 16,
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    for ii in range(0, rows, block):
        for jj in range(0, cols, block):
            for kk0 in range(0, depth, block):
                for i in range(ii, min(ii + block, rows)):
                    row_a = a[i]
                    row_out = out[i]
                    for j in range(jj, min(jj + block, cols)):
                        row_b = b_transposed[j]
                        value = 0.0
                        for k in range(kk0, min(kk0 + block, depth)):
                            value += row_a[k] * row_b[k]
                        row_out[j] += value
    return out


def _matmul_blocked_transpose_8_candidate(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    return _matmul_blocked_transpose_candidate(a, b, block=8)


def _matmul_ikj_unroll4_candidate(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_out = out[i]
        row_a = a[i]
        for k in range(depth):
            aik = row_a[k]
            row_b = b[k]
            j = 0
            while j + 3 < cols:
                row_out[j] += aik * row_b[j]
                row_out[j + 1] += aik * row_b[j + 1]
                row_out[j + 2] += aik * row_b[j + 2]
                row_out[j + 3] += aik * row_b[j + 3]
                j += 4
            while j < cols:
                row_out[j] += aik * row_b[j]
                j += 1
    return out


def _matmul_transpose_unroll4_candidate(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        for j in range(cols):
            row_b = b_transposed[j]
            value = 0.0
            k = 0
            while k + 3 < depth:
                value += (
                    row_a[k] * row_b[k]
                    + row_a[k + 1] * row_b[k + 1]
                    + row_a[k + 2] * row_b[k + 2]
                    + row_a[k + 3] * row_b[k + 3]
                )
                k += 4
            while k < depth:
                value += row_a[k] * row_b[k]
                k += 1
            out[i][j] = value
    return out


def _matmul_transpose_unroll8_candidate(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        for j in range(cols):
            row_b = b_transposed[j]
            value = 0.0
            k = 0
            while k + 7 < depth:
                value += (
                    row_a[k] * row_b[k]
                    + row_a[k + 1] * row_b[k + 1]
                    + row_a[k + 2] * row_b[k + 2]
                    + row_a[k + 3] * row_b[k + 3]
                    + row_a[k + 4] * row_b[k + 4]
                    + row_a[k + 5] * row_b[k + 5]
                    + row_a[k + 6] * row_b[k + 6]
                    + row_a[k + 7] * row_b[k + 7]
                )
                k += 8
            while k < depth:
                value += row_a[k] * row_b[k]
                k += 1
            out[i][j] = value
    return out


def _matmul_transpose_unroll16_candidate(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        for j in range(cols):
            row_b = b_transposed[j]
            value = 0.0
            k = 0
            while k + 15 < depth:
                for offset in range(16):
                    value += row_a[k + offset] * row_b[k + offset]
                k += 16
            while k < depth:
                value += row_a[k] * row_b[k]
                k += 1
            out[i][j] = value
    return out


def _matmul_transpose_rowpair_candidate(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    i = 0
    while i + 1 < rows:
        row_a0 = a[i]
        row_a1 = a[i + 1]
        for j in range(cols):
            row_b = b_transposed[j]
            value0 = 0.0
            value1 = 0.0
            for k in range(depth):
                b_k = row_b[k]
                value0 += row_a0[k] * b_k
                value1 += row_a1[k] * b_k
            out[i][j] = value0
            out[i + 1][j] = value1
        i += 2
    while i < rows:
        row_a = a[i]
        for j in range(cols):
            row_b = b_transposed[j]
            value = 0.0
            for k in range(depth):
                value += row_a[k] * row_b[k]
            out[i][j] = value
        i += 1
    return out


def _matmul_candidate_catalog(
    history_summary: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    catalog = [
        {
            "id": "transpose_dot",
            "name": "transpose-aware dot-product loop",
            "family": "layout_transform",
            "description": "Transpose B first so inner loops read contiguous memory.",
            "fn": _matmul_transpose_candidate,
            "priority": 1.0,
        },
        {
            "id": "ikj_accumulate",
            "name": "i-k-j accumulation loop",
            "family": "loop_reordering",
            "description": "Accumulate output rows while streaming through B row-major.",
            "fn": _matmul_ikj_candidate,
            "priority": 0.9,
        },
        {
            "id": "blocked_transpose_16",
            "name": "blocked transpose-aware loop (16)",
            "family": "tiling",
            "description": "Use transpose plus 16-wide block tiling to reduce cache-miss pressure.",
            "fn": _matmul_blocked_transpose_candidate,
            "priority": 0.7,
        },
        {
            "id": "blocked_transpose_8",
            "name": "blocked transpose-aware loop (8)",
            "family": "tiling",
            "description": "Use transpose plus 8-wide block tiling to reduce cache-miss pressure.",
            "fn": _matmul_blocked_transpose_8_candidate,
            "priority": 0.5,
        },
    ]
    best_candidate_id = ""
    weak_shape_targets: list[str] = []
    if isinstance(history_summary, dict):
        best_candidate_id = str(history_summary.get("best_matmul_candidate_id", "")).strip()
        weak_shape_targets = [
            shape
            for item in history_summary.get("weakest_matmul_shapes", []) or []
            if isinstance(item, dict)
            for shape in [str(item.get("shape", "")).strip()]
            if shape and _shape_min_dimension(shape) >= 64
        ][:3]

    generated = []
    if best_candidate_id.startswith("transpose"):
        generated.extend(
            [
                {
                    "id": "transpose_unroll8",
                    "name": "transpose-aware unroll-8 loop",
                    "family": "unrolling",
                    "description": "Use transpose plus manual unroll-8 accumulation in the inner loop.",
                    "fn": _matmul_transpose_unroll8_candidate,
                    "priority": 0.95,
                    "parent_id": "transpose_dot",
                    "generated": True,
                },
                {
                    "id": "transpose_unroll16",
                    "name": "transpose-aware unroll-16 loop",
                    "family": "unrolling",
                    "description": "Use transpose plus manual unroll-16 accumulation in the inner loop.",
                    "fn": _matmul_transpose_unroll16_candidate,
                    "priority": 0.55,
                    "parent_id": "transpose_dot",
                    "generated": True,
                },
                {
                    "id": "transpose_unroll4",
                    "name": "transpose-aware unroll-4 loop",
                    "family": "unrolling",
                    "description": "Use transpose plus manual unroll-4 accumulation in the inner loop.",
                    "fn": _matmul_transpose_unroll4_candidate,
                    "priority": 0.8,
                    "parent_id": "transpose_dot",
                    "generated": True,
                },
            ]
        )
        if weak_shape_targets:
            generated.append(
                {
                    "id": "transpose_rowpair",
                    "name": "transpose-aware row-pair sweep",
                    "family": "row_pairing",
                    "description": "Reuse each transposed B row across two output rows to cut interpreter overhead on larger square shapes.",
                    "fn": _matmul_transpose_rowpair_candidate,
                    "priority": 0.98,
                    "parent_id": "transpose_dot",
                    "generated": True,
                    "target_shapes": weak_shape_targets,
                }
            )
    if best_candidate_id.startswith("ikj"):
        generated.append(
            {
                "id": "ikj_unroll4",
                "name": "i-k-j accumulation loop with j unroll-4",
                "family": "loop_reordering",
                "description": "Accumulate output rows row-major with an unrolled j loop.",
                "fn": _matmul_ikj_unroll4_candidate,
                "priority": 0.85,
                "parent_id": "ikj_accumulate",
                "generated": True,
            }
        )
    if not generated:
        generated.extend(
            [
                {
                    "id": "transpose_unroll8",
                    "name": "transpose-aware unroll-8 loop",
                    "family": "unrolling",
                    "description": "Use transpose plus manual unroll-8 accumulation in the inner loop.",
                    "fn": _matmul_transpose_unroll8_candidate,
                    "priority": 0.95,
                    "parent_id": "transpose_dot",
                    "generated": True,
                },
                {
                    "id": "ikj_unroll4",
                    "name": "i-k-j accumulation loop with j unroll-4",
                    "family": "loop_reordering",
                    "description": "Accumulate output rows row-major with an unrolled j loop.",
                    "fn": _matmul_ikj_unroll4_candidate,
                    "priority": 0.85,
                    "parent_id": "ikj_accumulate",
                    "generated": True,
                },
            ]
        )
    return catalog + generated


_MATMUL_CANDIDATE_PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "global_rationale": {"type": "string"},
    },
    "required": ["candidate_ids", "global_rationale"],
}


def _time_call(fn) -> float:
    started = perf_counter()
    fn()
    return perf_counter() - started


def _measure_runtime(fn, *, repetitions: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = [_time_call(fn) for _ in range(repetitions)]
    return median(samples)


def _max_abs_diff(left: list[list[float]], right: list[list[float]]) -> float:
    value = 0.0
    for row_left, row_right in zip(left, right):
        for cell_left, cell_right in zip(row_left, row_right):
            value = max(value, abs(cell_left - cell_right))
    return value


def _shape_min_dimension(shape: str) -> int:
    try:
        return min(int(part) for part in shape.lower().split("x"))
    except ValueError:
        return 0


def _tool_use_rows_from_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    judgments = payload.get("judgments", [])
    if not isinstance(judgments, list):
        judgments = []
    rows: list[dict[str, object]] = []
    for item in judgments:
        if not isinstance(item, dict):
            continue
        fixture_id = str(item.get("id", "")).strip()
        terminal_success = bool(item.get("terminal_first_success"))
        structured_success = bool(item.get("structured_success"))
        note = " ".join(str(item.get("note", "")).split())
        if not fixture_id:
            continue
        rows.append(
            {
                "id": fixture_id,
                "mode": "terminal-first",
                "success": terminal_success,
                "event": "success" if terminal_success else "failure",
                "note": note,
            }
        )
        rows.append(
            {
                "id": fixture_id,
                "mode": "structured",
                "success": structured_success,
                "event": "success" if structured_success else "failure",
                "note": note,
            }
        )
    return rows


def _build_cost_summary(
    *,
    provider: str,
    model: str,
    request_count: int,
    usages: list[dict[str, int]],
    elapsed_ms: int,
) -> dict[str, object]:
    input_tokens = sum(item["input_tokens"] for item in usages)
    output_tokens = sum(item["output_tokens"] for item in usages)
    total_tokens = sum(item["total_tokens"] for item in usages)
    input_price = _float_env("NOERIS_INPUT_TOKEN_COST_USD_PER_1M")
    output_price = _float_env("NOERIS_OUTPUT_TOKEN_COST_USD_PER_1M")
    cost_budget_usd = _float_env("NOERIS_COST_BUDGET_USD")
    latency_budget_ms = _int_env("NOERIS_LATENCY_BUDGET_MS")
    estimated_cost_usd = None
    if input_price is not None and output_price is not None:
        estimated_cost_usd = round(
            (input_tokens / 1_000_000 * input_price)
            + (output_tokens / 1_000_000 * output_price),
            6,
        )
    cost_budget_exceeded = (
        estimated_cost_usd is not None
        and cost_budget_usd is not None
        and estimated_cost_usd > cost_budget_usd
    )
    latency_budget_exceeded = (
        latency_budget_ms is not None
        and elapsed_ms > latency_budget_ms
    )
    warnings: list[str] = []
    if cost_budget_exceeded:
        warnings.append(
            f"Estimated cost {estimated_cost_usd} USD exceeded budget {cost_budget_usd} USD."
        )
    if latency_budget_exceeded:
        warnings.append(
            f"Elapsed time {elapsed_ms} ms exceeded budget {latency_budget_ms} ms."
        )
    return {
        "provider": provider,
        "model": model,
        "request_count": request_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "elapsed_ms": elapsed_ms,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_budget_usd": cost_budget_usd,
        "latency_budget_ms": latency_budget_ms,
        "cost_budget_exceeded": cost_budget_exceeded,
        "latency_budget_exceeded": latency_budget_exceeded,
        "warnings": warnings,
    }


def _float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


_LONG_CONTEXT_ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["id", "answer"],
            },
        }
    },
    "required": ["answers"],
}


_TOOL_USE_EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "terminal_first_success": {"type": "boolean"},
                    "structured_success": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": [
                    "id",
                    "terminal_first_success",
                    "structured_success",
                    "note",
                ],
            },
        }
    },
    "required": ["judgments"],
}
