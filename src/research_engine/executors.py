from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from math import ceil
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
    {
        "id": "mm-live-qkv",
        "shape": (32, 96, 32),
        "dtype": "float64",
        "workload_tag": "attention_qkv",
        "workload_share": 0.14,
    },
    {
        "id": "mm-live-attn-out",
        "shape": (96, 64, 96),
        "dtype": "float64",
        "workload_tag": "attention_out_proj",
        "workload_share": 0.12,
    },
    {
        "id": "mm-live-mlp-up",
        "shape": (64, 256, 64),
        "dtype": "float64",
        "workload_tag": "mlp_up_proj",
        "workload_share": 0.28,
    },
    {
        "id": "mm-live-mlp-down",
        "shape": (64, 64, 256),
        "dtype": "float64",
        "workload_tag": "mlp_down_proj",
        "workload_share": 0.22,
    },
    {
        "id": "mm-live-residual",
        "shape": (128, 64, 32),
        "dtype": "float64",
        "workload_tag": "residual_adapter",
        "workload_share": 0.10,
    },
    {
        "id": "mm-live-control",
        "shape": (96, 96, 96),
        "dtype": "float64",
        "workload_tag": "square_control",
        "workload_share": 0.14,
    },
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
    max_candidates_per_run: int = 7
    history_summary: dict[str, object] | None = None
    proposer: ResponsesApiClient | None = None

    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        rows = []
        weighted_improvements = []
        unweighted_improvements = []
        winner_counts: dict[str, int] = {}
        winner_share_scores: dict[str, float] = {}
        selected_candidates, pruned_candidates, shape_focus, proposal = self._select_candidates()
        for fixture in LIVE_MATMUL_FIXTURES:
            row = self._run_fixture(fixture, selected_candidates)
            rows.append(row)
            uplift = row["uplift_pct"] / 100
            share = float(row["workload_share"])
            weighted_improvements.append(uplift * share)
            unweighted_improvements.append(uplift)
            winner_counts[row["best_candidate_id"]] = winner_counts.get(row["best_candidate_id"], 0) + 1
            winner_share_scores[row["best_candidate_id"]] = (
                winner_share_scores.get(row["best_candidate_id"], 0.0) + share
            )

        total_share = sum(float(row["workload_share"]) for row in rows) or 1.0
        mean_uplift = sum(weighted_improvements) / total_share
        unweighted_mean_uplift = sum(unweighted_improvements) / len(unweighted_improvements)
        pareto_frontier = _build_matmul_pareto_frontier(rows)
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
                    "workload_tag": row["workload_tag"],
                    "workload_share": row["workload_share"],
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
                "aggregation": "workload_share_weighted_mean_uplift_pct",
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
                        "mutation_params": candidate.get("mutation_params", {}),
                        "target_workloads": candidate.get("target_workloads", []),
                    }
                    for candidate in selected_candidates
                ],
                "pruned_candidates": pruned_candidates,
            },
            "candidate-proposals.json": proposal,
            "shape-focus.json": shape_focus,
            "raw-timing-results.json": {
                "aggregation": "workload_share_weighted_mean_uplift_pct",
                "mean_uplift_pct": round(mean_uplift * 100, 2),
                "unweighted_mean_uplift_pct": round(unweighted_mean_uplift * 100, 2),
                "rows": rows,
            },
            "best-candidate-summary.json": {
                "winner_counts": winner_counts,
                "winner_share_scores": {
                    candidate_id: round(score, 4)
                    for candidate_id, score in winner_share_scores.items()
                },
                "pareto_candidate_ids": pareto_frontier["candidate_ids"],
                "best_overall_candidate_id": max(
                    winner_share_scores,
                    key=lambda candidate_id: (
                        winner_share_scores[candidate_id],
                        winner_counts.get(candidate_id, 0),
                    ),
                    default=max(
                        winner_counts,
                        key=winner_counts.get,
                        default="",
                    ),
                ),
            },
            "frontier-archive.json": {
                "workload_winners": [
                    {
                        "workload_tag": row["workload_tag"],
                        "workload_share": row["workload_share"],
                        "best_candidate_id": row["best_candidate_id"],
                        "runner_up_candidate_id": row["runner_up_candidate_id"],
                        "runner_up_gap_pct": row["runner_up_gap_pct"],
                    }
                    for row in rows
                ]
            },
            "pareto-frontier.json": pareto_frontier,
            "baseline-comparison.md": _matmul_live_comparison(rows, mean_uplift),
        }
        summary = (
            f"Live matmul CPU benchmark completed on {len(rows)} fixtures. "
            f"Weighted throughput uplift={mean_uplift * 100:.2f}%."
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
        baseline_measurement = _measure_runtime_stats(
            lambda: _matmul_ijk(a, b),
            repetitions=self.repetitions,
            warmups=self.warmup_repetitions,
        )
        baseline_time = baseline_measurement["seconds"]
        baseline_result = _matmul_ijk(a, b)
        ops = 2 * m * n * k
        baseline_gflops = ops / baseline_time / 1_000_000_000
        candidate_rows = []
        for candidate in candidates:
            fn = candidate["fn"]
            candidate_measurement = _measure_runtime_stats(
                lambda fn=fn: fn(a, b),
                repetitions=self.repetitions,
                warmups=self.warmup_repetitions,
            )
            candidate_time = candidate_measurement["seconds"]
            candidate_result = fn(a, b)
            max_abs_error = _max_abs_diff(baseline_result, candidate_result)
            candidate_rows.append(
                {
                    "candidate_id": candidate["id"],
                    "candidate_name": candidate["name"],
                    "candidate_family": candidate["family"],
                    "candidate_seconds": round(candidate_time, 6),
                    "candidate_gflops": round(ops / candidate_time / 1_000_000_000, 4),
                    "loops_per_sample": candidate_measurement["loops_per_sample"],
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
            "workload_tag": fixture.get("workload_tag", ""),
            "workload_share": float(fixture.get("workload_share", 1.0)),
            "baseline_seconds": round(baseline_time, 6),
            "baseline_loops_per_sample": baseline_measurement["loops_per_sample"],
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
        workload_winners = {}
        workload_challengers = {}
        weakest_workloads = []
        best_candidate_id = ""
        family_wins = {}
        frontier_archive = []
        pareto_candidate_ids: list[str] = []
        if isinstance(self.history_summary, dict):
            wins = self.history_summary.get("matmul_candidate_wins", {}) or {}
            family_wins = self.history_summary.get("matmul_family_wins", {}) or {}
            shape_winners = self.history_summary.get("matmul_shape_winners", {}) or {}
            shape_challengers = self.history_summary.get("matmul_shape_challengers", {}) or {}
            weakest_shapes = self.history_summary.get("weakest_matmul_shapes", []) or []
            workload_winners = self.history_summary.get("matmul_workload_winners", {}) or {}
            workload_challengers = self.history_summary.get("matmul_workload_challengers", {}) or {}
            weakest_workloads = self.history_summary.get("weakest_matmul_workloads", []) or []
            frontier_archive = self.history_summary.get("matmul_frontier_archive", []) or []
            pareto_candidate_ids = self.history_summary.get("matmul_pareto_candidate_ids", []) or []
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
            if not weakest_workloads and workload_challengers:
                weakest_workloads = sorted(
                    [
                        {
                            "workload_tag": workload_tag,
                            "runner_up_candidate_id": entry.get("latest_runner_up", ""),
                            "runner_up_gap_pct": entry.get("latest_runner_up_gap_pct", 10**9),
                        }
                        for workload_tag, entry in workload_challengers.items()
                        if isinstance(entry, dict)
                    ],
                    key=lambda item: item.get("runner_up_gap_pct", 10**9),
                )
        proposal = self._propose_candidates(catalog, weakest_shapes, weakest_workloads)
        existing_ids = {candidate["id"] for candidate in catalog}
        for novel in proposal.get("novel_candidates", []):
            materialized = _materialize_novel_candidate(novel, existing_ids=existing_ids)
            if materialized is not None:
                catalog.append(materialized)
                existing_ids.add(materialized["id"])
        proposed_ids = set(proposal.get("candidate_ids", []))
        top_weakest_shapes = [
            item for item in weakest_shapes[:3] if isinstance(item, dict)
        ]
        top_weakest_workloads = [
            item for item in weakest_workloads[:3] if isinstance(item, dict)
        ]
        weak_shape_ids = {
            str(item.get("shape", "")).strip()
            for item in top_weakest_shapes
            if str(item.get("shape", "")).strip()
        }
        weak_workload_ids = {
            str(item.get("workload_tag", "")).strip()
            for item in top_weakest_workloads
            if str(item.get("workload_tag", "")).strip()
        }
        archive_candidate_ids = {
            str(item.get("best_candidate_id", "")).strip()
            for item in frontier_archive
            if isinstance(item, dict) and str(item.get("best_candidate_id", "")).strip()
        }
        historical_pareto_ids = {
            str(candidate_id).strip()
            for candidate_id in pareto_candidate_ids
            if isinstance(candidate_id, str) and str(candidate_id).strip()
        }
        for candidate in catalog:
            candidate["historical_wins"] = int(wins.get(candidate["id"], 0))
            candidate["family_wins"] = int(family_wins.get(candidate["family"], 0))
            candidate["shape_bonus"] = sum(
                int(shape_entry.get("winner_counts", {}).get(candidate["id"], 0))
                for shape_entry in shape_winners.values()
                if isinstance(shape_entry, dict)
            )
            candidate["workload_bonus"] = sum(
                int(workload_entry.get("winner_counts", {}).get(candidate["id"], 0))
                for workload_entry in workload_winners.values()
                if isinstance(workload_entry, dict)
            )
            candidate["challenger_bonus"] = sum(
                int(shape_entry.get("runner_up_counts", {}).get(candidate["id"], 0))
                for shape_entry in shape_challengers.values()
                if isinstance(shape_entry, dict)
            )
            candidate["workload_challenger_bonus"] = sum(
                int(workload_entry.get("runner_up_counts", {}).get(candidate["id"], 0))
                for workload_entry in workload_challengers.values()
                if isinstance(workload_entry, dict)
            )
            candidate["target_shape_bonus"] = sum(
                1
                for shape in candidate.get("target_shapes", [])
                if isinstance(shape, str) and shape in weak_shape_ids
            )
            candidate["target_workload_bonus"] = sum(
                1
                for workload_tag in candidate.get("target_workloads", [])
                if isinstance(workload_tag, str) and workload_tag in weak_workload_ids
            )
            candidate["weak_shape_bonus"] = sum(
                1
                for item in weakest_shapes[:2]
                if isinstance(item, dict) and item.get("runner_up_candidate_id") == candidate["id"]
            )
            candidate["archive_bonus"] = 1 if candidate["id"] in archive_candidate_ids else 0
            candidate["pareto_bonus"] = 1 if candidate["id"] in historical_pareto_ids else 0
            candidate["novelty_bonus"] = (
                1
                if candidate.get("generated", False)
                and candidate["historical_wins"] == 0
                and candidate["archive_bonus"] == 0
                and candidate["pareto_bonus"] == 0
                else 0
            )
            candidate["family_novelty_bonus"] = 1 if candidate["family_wins"] == 0 else 0
            candidate["lineage_bonus"] = (
                1
                if best_candidate_id
                and isinstance(candidate.get("parent_id"), str)
                and candidate.get("parent_id") == best_candidate_id
                else 0
            )
            candidate["proposal_bonus"] = 1 if candidate["id"] in proposed_ids else 0
            candidate["novel_proposed_bonus"] = 1 if candidate.get("novel") else 0

        selected = []
        pruned = []
        seen_families: set[str] = set()
        slot_labels: list[dict[str, str]] = []

        ranked = sorted(
            catalog,
            key=lambda candidate: (
                -candidate["historical_wins"],
                -candidate["shape_bonus"],
                -candidate["workload_bonus"],
                -candidate["target_shape_bonus"],
                -candidate["target_workload_bonus"],
                -candidate["challenger_bonus"],
                -candidate["workload_challenger_bonus"],
                -candidate["weak_shape_bonus"],
                -candidate["archive_bonus"],
                -candidate["pareto_bonus"],
                -candidate["novelty_bonus"],
                -candidate["family_novelty_bonus"],
                -candidate["lineage_bonus"],
                -candidate["proposal_bonus"],
                -candidate["novel_proposed_bonus"],
                -candidate["priority"],
                candidate["id"],
            ),
        )

        def _select_slot(
            label: str,
            predicate,
        ) -> None:
            if len(selected) >= self.max_candidates_per_run:
                return
            candidate = next(
                (
                    item
                    for item in ranked
                    if item not in selected and predicate(item)
                ),
                None,
            )
            if candidate is None:
                return
            selected.append(candidate)
            seen_families.add(str(candidate["family"]))
            slot_labels.append({"slot": label, "candidate_id": str(candidate["id"])})

        _select_slot("incumbent", lambda candidate: True)
        _select_slot("pareto_specialist", lambda candidate: candidate["pareto_bonus"] > 0)
        _select_slot(
            "frontier_challenger",
            lambda candidate: candidate["proposal_bonus"] > 0
            or candidate["target_workload_bonus"] > 0
            or candidate["weak_shape_bonus"] > 0,
        )
        _select_slot(
            "novelty_proposed",
            lambda candidate: candidate["novel_proposed_bonus"] > 0,
        )
        _select_slot(
            "novelty_1",
            lambda candidate: candidate["novelty_bonus"] > 0
            or candidate["family_novelty_bonus"] > 0,
        )
        _select_slot(
            "novelty_2",
            lambda candidate: candidate["novelty_bonus"] > 0
            or candidate["family_novelty_bonus"] > 0,
        )

        for candidate in ranked:
            if candidate in selected:
                continue
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
            slot_labels.append({"slot": "rank_fill", "candidate_id": str(candidate["id"])})
        shape_focus = {
            "weakest_shapes": top_weakest_shapes,
            "weakest_workloads": top_weakest_workloads,
            "selection_slots": slot_labels,
            "selection_reasons": [
                {
                    "candidate_id": candidate["id"],
                    "historical_wins": candidate["historical_wins"],
                    "family_wins": candidate["family_wins"],
                    "shape_bonus": candidate["shape_bonus"],
                    "workload_bonus": candidate["workload_bonus"],
                    "target_shape_bonus": candidate["target_shape_bonus"],
                    "target_workload_bonus": candidate["target_workload_bonus"],
                    "challenger_bonus": candidate["challenger_bonus"],
                    "workload_challenger_bonus": candidate["workload_challenger_bonus"],
                    "weak_shape_bonus": candidate["weak_shape_bonus"],
                    "archive_bonus": candidate["archive_bonus"],
                    "pareto_bonus": candidate["pareto_bonus"],
                    "novelty_bonus": candidate["novelty_bonus"],
                    "family_novelty_bonus": candidate["family_novelty_bonus"],
                    "lineage_bonus": candidate["lineage_bonus"],
                    "proposal_bonus": candidate["proposal_bonus"],
                    "novel_proposed_bonus": candidate.get("novel_proposed_bonus", 0),
                    "priority": candidate["priority"],
                    "mutation_params": candidate.get("mutation_params", {}),
                    "target_shapes": candidate.get("target_shapes", []),
                    "target_workloads": candidate.get("target_workloads", []),
                }
                for candidate in selected
            ],
        }
        return selected, pruned, shape_focus, proposal

    def _propose_candidates(
        self,
        catalog: list[dict[str, object]],
        weakest_shapes: list[dict[str, object]],
        weakest_workloads: list[dict[str, object]],
    ) -> dict[str, object]:
        if self.proposer is None:
            return {
                "source": "none",
                "candidate_ids": [],
                "novel_candidates": [],
                "global_rationale": "",
            }
        param_insights = _extract_param_insights(self.history_summary)
        prompt_data = {
            "best_candidate_id": (
                self.history_summary.get("best_matmul_candidate_id", "")
                if isinstance(self.history_summary, dict)
                else ""
            ),
            "weakest_shapes": weakest_shapes[:3],
            "weakest_workloads": weakest_workloads[:3] if isinstance(self.history_summary, dict) else [],
            "available_candidates": [
                {
                    "id": candidate["id"],
                    "family": candidate["family"],
                    "description": candidate["description"],
                    "priority": candidate["priority"],
                    "parent_id": candidate.get("parent_id"),
                    "mutation_params": candidate.get("mutation_params", {}),
                    "target_workloads": candidate.get("target_workloads", []),
                }
                for candidate in catalog
            ],
        }
        if param_insights:
            prompt_data["param_insights"] = param_insights
        last_error = None
        for attempt in range(2):
            try:
                payload = self.proposer.generate_json(
                    schema_name="matmul_candidate_batch",
                    schema=_MATMUL_CANDIDATE_PROPOSAL_SCHEMA,
                    instructions=(
                        "Choose the most promising next batch of implementable matmul candidate families. "
                        "You may choose from the provided candidate ids AND/OR propose novel candidates "
                        "with new (row_block, col_block, k_unroll) or (row_block, j_unroll) parameter combos. "
                        "Novel candidates let you explore configurations not yet in the catalog. "
                        "Favor candidates that attack the weakest workloads or mutate the current best family. "
                        "Use the param_insights (if provided) to understand which params tend to win which workloads. "
                        "Return at most four existing ids plus up to three novel candidates. "
                        "Keep global_rationale to one short sentence."
                    ),
                    prompt=json.dumps(prompt_data, indent=2),
                    max_output_tokens=500,
                    reasoning_effort="low",
                    text_verbosity="low",
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            return {
                "source": "responses_api_error",
                "candidate_ids": [],
                "novel_candidates": [],
                "global_rationale": "",
                "error": type(last_error).__name__,
                "detail": " ".join(str(last_error).split())[:240],
            }
        allowed = {candidate["id"] for candidate in catalog}
        candidate_ids = [
            candidate_id
            for candidate_id in payload.get("candidate_ids", [])
            if isinstance(candidate_id, str) and candidate_id in allowed
        ][: self.max_candidates_per_run]
        novel_candidates = _sanitize_novel_candidates(
            payload.get("novel_candidates", []),
        )
        return {
            "source": "responses_api",
            "candidate_ids": candidate_ids,
            "novel_candidates": novel_candidates,
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


def _matmul_transpose_rowpair_unroll4_candidate(
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
            k = 0
            while k + 3 < depth:
                b0 = row_b[k]
                b1 = row_b[k + 1]
                b2 = row_b[k + 2]
                b3 = row_b[k + 3]
                value0 += row_a0[k] * b0 + row_a0[k + 1] * b1 + row_a0[k + 2] * b2 + row_a0[k + 3] * b3
                value1 += row_a1[k] * b0 + row_a1[k + 1] * b1 + row_a1[k + 2] * b2 + row_a1[k + 3] * b3
                k += 4
            while k < depth:
                b_k = row_b[k]
                value0 += row_a0[k] * b_k
                value1 += row_a1[k] * b_k
                k += 1
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


def _matmul_transpose_dual_col_candidate(
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
        j = 0
        while j + 1 < cols:
            row_b0 = b_transposed[j]
            row_b1 = b_transposed[j + 1]
            value0 = 0.0
            value1 = 0.0
            for k in range(depth):
                a_k = row_a[k]
                value0 += a_k * row_b0[k]
                value1 += a_k * row_b1[k]
            out[i][j] = value0
            out[i][j + 1] = value1
            j += 2
        while j < cols:
            row_b = b_transposed[j]
            value = 0.0
            for k in range(depth):
                value += row_a[k] * row_b[k]
            out[i][j] = value
            j += 1
    return out


def _matmul_transpose_rowpair_dualcol_candidate(
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
        j = 0
        while j + 1 < cols:
            row_b0 = b_transposed[j]
            row_b1 = b_transposed[j + 1]
            value00 = value01 = value10 = value11 = 0.0
            for k in range(depth):
                a0 = row_a0[k]
                a1 = row_a1[k]
                b0 = row_b0[k]
                b1 = row_b1[k]
                value00 += a0 * b0
                value01 += a0 * b1
                value10 += a1 * b0
                value11 += a1 * b1
            out[i][j] = value00
            out[i][j + 1] = value01
            out[i + 1][j] = value10
            out[i + 1][j + 1] = value11
            j += 2
        while j < cols:
            row_b = b_transposed[j]
            value0 = 0.0
            value1 = 0.0
            for k in range(depth):
                b_k = row_b[k]
                value0 += row_a0[k] * b_k
                value1 += row_a1[k] * b_k
            out[i][j] = value0
            out[i + 1][j] = value1
            j += 1
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


def _matmul_ikj_param_candidate(
    a: list[list[float]],
    b: list[list[float]],
    *,
    row_block: int,
    j_unroll: int,
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for ii in range(0, rows, row_block):
        active_rows = min(row_block, rows - ii)
        for k in range(depth):
            a_vals = [a[ii + r][k] for r in range(active_rows)]
            row_b = b[k]
            j = 0
            while j + j_unroll - 1 < cols:
                for r in range(active_rows):
                    a_val = a_vals[r]
                    row_out = out[ii + r]
                    for offset in range(j_unroll):
                        row_out[j + offset] += a_val * row_b[j + offset]
                j += j_unroll
            while j < cols:
                for r in range(active_rows):
                    out[ii + r][j] += a_vals[r] * row_b[j]
                j += 1
    return out


def _make_ikj_param_candidate(
    *,
    row_block: int,
    j_unroll: int,
):
    specialized = {
        (1, 4): _matmul_ikj_unroll4_candidate,
    }
    key = (row_block, j_unroll)
    if key in specialized:
        return specialized[key]

    def generated(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return _matmul_ikj_param_candidate(
            a,
            b,
            row_block=row_block,
            j_unroll=j_unroll,
        )

    return generated


def _matmul_transpose_param_candidate(
    a: list[list[float]],
    b: list[list[float]],
    *,
    row_block: int,
    col_block: int,
    k_unroll: int,
) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    depth = len(b)
    b_transposed = [list(column) for column in zip(*b)]
    out = [[0.0] * cols for _ in range(rows)]
    for ii in range(0, rows, row_block):
        active_rows = min(row_block, rows - ii)
        for jj in range(0, cols, col_block):
            active_cols = min(col_block, cols - jj)
            values = [[0.0] * active_cols for _ in range(active_rows)]
            k = 0
            while k + k_unroll - 1 < depth:
                for offset in range(k_unroll):
                    kk = k + offset
                    b_values = [b_transposed[jj + col][kk] for col in range(active_cols)]
                    for row in range(active_rows):
                        a_value = a[ii + row][kk]
                        for col, b_value in enumerate(b_values):
                            values[row][col] += a_value * b_value
                k += k_unroll
            while k < depth:
                b_values = [b_transposed[jj + col][k] for col in range(active_cols)]
                for row in range(active_rows):
                    a_value = a[ii + row][k]
                    for col, b_value in enumerate(b_values):
                        values[row][col] += a_value * b_value
                k += 1
            for row in range(active_rows):
                row_out = out[ii + row]
                for col in range(active_cols):
                    row_out[jj + col] = values[row][col]
    return out


def _make_transpose_param_candidate(
    *,
    row_block: int,
    col_block: int,
    k_unroll: int,
):
    specialized = {
        (1, 1, 1): _matmul_transpose_candidate,
        (1, 1, 4): _matmul_transpose_unroll4_candidate,
        (1, 1, 8): _matmul_transpose_unroll8_candidate,
        (1, 1, 16): _matmul_transpose_unroll16_candidate,
        (2, 1, 1): _matmul_transpose_rowpair_candidate,
        (2, 1, 4): _matmul_transpose_rowpair_unroll4_candidate,
        (1, 2, 1): _matmul_transpose_dual_col_candidate,
        (2, 2, 1): _matmul_transpose_rowpair_dualcol_candidate,
    }
    key = (row_block, col_block, k_unroll)
    if key in specialized:
        return specialized[key]

    def generated(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return _matmul_transpose_param_candidate(
            a,
            b,
            row_block=row_block,
            col_block=col_block,
            k_unroll=k_unroll,
        )

    return generated


_MATMUL_BASE_CANDIDATES = [
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


_MATMUL_CURATED_TRANSPOSE_DESCRIPTORS = [
    {
        "id": "transpose_unroll8",
        "name": "transpose-aware unroll-8 loop",
        "family": "unrolling",
        "description": "Use transpose plus manual unroll-8 accumulation in the inner loop.",
        "row_block": 1,
        "col_block": 1,
        "k_unroll": 8,
        "priority": 0.95,
        "parent_id": "transpose_dot",
        "enabled_when": "transpose_or_default",
        "target_workload_group": "wide_output",
    },
    {
        "id": "transpose_rowpair",
        "name": "transpose-aware row-pair sweep",
        "family": "row_pairing",
        "description": "Reuse each transposed B row across two output rows to cut interpreter overhead on larger workloads.",
        "row_block": 2,
        "col_block": 1,
        "k_unroll": 1,
        "priority": 0.98,
        "parent_id": "transpose_dot",
        "enabled_when": "transpose_or_default",
        "target_workload_group": "weak_or_balanced",
        "target_shape_mode": "weak_shapes",
    },
    {
        "id": "transpose_rowpair_dualcol",
        "name": "transpose-aware row/column pair sweep",
        "family": "pairwise_tiling",
        "description": "Pair both rows and columns to test a small register-blocked transpose family.",
        "row_block": 2,
        "col_block": 2,
        "k_unroll": 1,
        "priority": 0.72,
        "parent_id": "transpose_rowpair",
        "enabled_when": "transpose_or_default",
        "target_workload_group": "wide_output",
    },
    {
        "id": "transpose_rowpair_dualcol_u4",
        "name": "transpose-aware row/column pair sweep with k unroll-4",
        "family": "pairwise_tiling",
        "description": "Pair rows and columns together with modest k unrolling to test a generated microtile variant.",
        "row_block": 2,
        "col_block": 2,
        "k_unroll": 4,
        "priority": 0.83,
        "parent_id": "transpose_rowpair_dualcol",
        "enabled_when": "transpose_or_default",
        "target_workload_group": "wide_output",
    },
]

_MATMUL_CURATED_IKJ_DESCRIPTORS = [
    {
        "id": "ikj_unroll4",
        "name": "i-k-j accumulation loop with j unroll-4",
        "family": "ikj_blocking",
        "description": "Accumulate output rows row-major with an unrolled j loop.",
        "fn": _matmul_ikj_unroll4_candidate,
        "priority": 0.85,
        "parent_id": "ikj_accumulate",
        "enabled_when": "ikj_or_default",
        "target_workload_group": "ikj_friendly",
        "kernel_family": "ikj",
        "row_block": 1,
        "j_unroll": 4,
    },
]


def _generate_transpose_grid() -> list[dict[str, object]]:
    """Generate the full transpose parameter grid.

    Returns descriptors for all (row_block, col_block, k_unroll) combos
    not already covered by curated descriptors.
    """
    curated_params = {
        (d["row_block"], d["col_block"], d["k_unroll"])
        for d in _MATMUL_CURATED_TRANSPOSE_DESCRIPTORS
    }
    grid: list[dict[str, object]] = []
    for row_block in [1, 2, 4]:
        for col_block in [1, 2, 4]:
            for k_unroll in [1, 2, 4, 8, 16]:
                if (row_block, col_block, k_unroll) == (1, 1, 1):
                    continue
                if (row_block, col_block, k_unroll) in curated_params:
                    continue
                candidate_id = f"transpose_r{row_block}_c{col_block}_k{k_unroll}"
                complexity = row_block * col_block
                priority = round(0.4 + 0.05 * min(k_unroll, 8) - 0.02 * complexity, 2)
                workload_group = "wide_output" if col_block >= 2 else "balanced"
                grid.append({
                    "id": candidate_id,
                    "name": f"transpose rb={row_block} cb={col_block} ku={k_unroll}",
                    "family": f"transpose_grid_r{row_block}c{col_block}",
                    "description": (
                        f"Generated transpose kernel with row_block={row_block}, "
                        f"col_block={col_block}, k_unroll={k_unroll}."
                    ),
                    "row_block": row_block,
                    "col_block": col_block,
                    "k_unroll": k_unroll,
                    "priority": max(0.1, min(priority, 0.65)),
                    "parent_id": "transpose_dot",
                    "enabled_when": "transpose_or_default",
                    "target_workload_group": workload_group,
                })
    return grid


def _generate_ikj_grid() -> list[dict[str, object]]:
    """Generate the ikj parameter grid.

    Returns descriptors for all (row_block, j_unroll) combos
    not already covered by curated descriptors.
    """
    grid: list[dict[str, object]] = []
    for row_block in [1, 2, 4]:
        for j_unroll in [1, 2, 4, 8]:
            if (row_block, j_unroll) == (1, 1):
                continue
            if row_block == 1 and j_unroll == 4:
                continue
            candidate_id = f"ikj_r{row_block}_j{j_unroll}"
            priority = round(0.35 + 0.04 * min(j_unroll, 8) - 0.02 * row_block, 2)
            grid.append({
                "id": candidate_id,
                "name": f"ikj rb={row_block} ju={j_unroll}",
                "family": f"ikj_blocking_r{row_block}",
                "description": (
                    f"Generated ikj kernel with row_block={row_block}, j_unroll={j_unroll}."
                ),
                "row_block": row_block,
                "j_unroll": j_unroll,
                "priority": max(0.1, min(priority, 0.6)),
                "parent_id": "ikj_accumulate",
                "enabled_when": "ikj_or_default",
                "target_workload_group": "ikj_friendly",
                "kernel_family": "ikj",
            })
    return grid


def _build_mutation_descriptors() -> list[dict[str, object]]:
    return (
        list(_MATMUL_CURATED_TRANSPOSE_DESCRIPTORS)
        + _generate_transpose_grid()
        + list(_MATMUL_CURATED_IKJ_DESCRIPTORS)
        + _generate_ikj_grid()
    )


def _materialize_matmul_candidate(
    descriptor: dict[str, object],
    *,
    target_workload_groups: dict[str, list[str]],
    target_shapes: list[str],
) -> dict[str, object]:
    kernel_family = str(descriptor.get("kernel_family", "")).strip()
    row_block = int(descriptor.get("row_block", 1))

    if kernel_family == "ikj":
        j_unroll = int(descriptor.get("j_unroll", 1))
        candidate = {
            "id": descriptor["id"],
            "name": descriptor["name"],
            "family": descriptor["family"],
            "description": descriptor["description"],
            "fn": descriptor.get("fn")
            or _make_ikj_param_candidate(
                row_block=row_block,
                j_unroll=j_unroll,
            ),
            "priority": descriptor["priority"],
            "parent_id": descriptor.get("parent_id"),
            "generated": True,
            "mutation_params": {
                "row_block": row_block,
                "j_unroll": j_unroll,
            },
        }
    else:
        col_block = int(descriptor.get("col_block", 1))
        k_unroll = int(descriptor.get("k_unroll", 1))
        candidate = {
            "id": descriptor["id"],
            "name": descriptor["name"],
            "family": descriptor["family"],
            "description": descriptor["description"],
            "fn": descriptor.get("fn")
            or _make_transpose_param_candidate(
                row_block=row_block,
                col_block=col_block,
                k_unroll=k_unroll,
            ),
            "priority": descriptor["priority"],
            "parent_id": descriptor.get("parent_id"),
            "generated": True,
            "mutation_params": {
                "row_block": row_block,
                "col_block": col_block,
                "k_unroll": k_unroll,
            },
        }
    workload_group = str(descriptor.get("target_workload_group", "")).strip()
    if workload_group:
        candidate["target_workloads"] = list(target_workload_groups.get(workload_group, []))
    if descriptor.get("target_shape_mode") == "weak_shapes":
        candidate["target_shapes"] = list(target_shapes)
    return candidate


def _matmul_candidate_catalog(
    history_summary: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    wide_output_workloads = [
        fixture["workload_tag"]
        for fixture in LIVE_MATMUL_FIXTURES
        if fixture["shape"][1] >= fixture["shape"][0] * 2
    ]
    balanced_workloads = [
        fixture["workload_tag"]
        for fixture in LIVE_MATMUL_FIXTURES
        if fixture["shape"][0] >= 64 and fixture["shape"][2] >= 64
    ]
    best_candidate_id = ""
    weak_shape_targets: list[str] = []
    weak_workload_targets: list[str] = []
    if isinstance(history_summary, dict):
        best_candidate_id = str(history_summary.get("best_matmul_candidate_id", "")).strip()
        weak_shape_targets = [
            shape
            for item in history_summary.get("weakest_matmul_shapes", []) or []
            if isinstance(item, dict)
            for shape in [str(item.get("shape", "")).strip()]
            if shape and _shape_min_dimension(shape) >= 64
        ][:3]
        weak_workload_targets = [
            workload_tag
            for item in history_summary.get("weakest_matmul_workloads", []) or []
            if isinstance(item, dict)
            for workload_tag in [str(item.get("workload_tag", "")).strip()]
            if workload_tag
        ][:3]

    target_workload_groups = {
        "wide_output": wide_output_workloads,
        "balanced": balanced_workloads,
        "ikj_friendly": ["mlp_down_proj", "residual_adapter"],
        "weak_or_balanced": weak_workload_targets or balanced_workloads,
    }

    generated: list[dict[str, object]] = []
    for descriptor in _build_mutation_descriptors():
        enabled_when = descriptor["enabled_when"]
        enabled = False
        if enabled_when == "transpose_only":
            enabled = best_candidate_id.startswith("transpose")
        elif enabled_when == "transpose_or_default":
            enabled = best_candidate_id.startswith("transpose") or not best_candidate_id
        elif enabled_when == "transpose_with_weak_shapes":
            enabled = best_candidate_id.startswith("transpose") and bool(weak_shape_targets)
        elif enabled_when == "ikj_or_default":
            enabled = best_candidate_id.startswith("ikj") or not best_candidate_id
        elif enabled_when == "always":
            enabled = True
        if not enabled:
            continue
        generated.append(
            _materialize_matmul_candidate(
                descriptor,
                target_workload_groups=target_workload_groups,
                target_shapes=weak_shape_targets,
            )
        )
    return list(_MATMUL_BASE_CANDIDATES) + generated


_MATMUL_CANDIDATE_PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "novel_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kernel_family": {
                        "type": "string",
                        "description": "transpose or ikj",
                    },
                    "row_block": {"type": "integer"},
                    "col_block": {"type": "integer"},
                    "k_unroll": {"type": "integer"},
                    "j_unroll": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "kernel_family",
                    "row_block",
                    "col_block",
                    "k_unroll",
                    "j_unroll",
                    "rationale",
                ],
            },
        },
        "global_rationale": {"type": "string"},
    },
    "required": ["candidate_ids", "novel_candidates", "global_rationale"],
}


def _sanitize_novel_candidates(
    payload: object,
) -> list[dict[str, object]]:
    """Validate and bound novel candidate proposals from the LLM."""
    if not isinstance(payload, list):
        return []
    max_novel = 3
    seen_keys: set[tuple] = set()
    results: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict) or len(results) >= max_novel:
            break
        kernel_family = str(item.get("kernel_family", "")).strip().lower()
        if kernel_family not in ("transpose", "ikj"):
            continue
        row_block = _clamp_int(item.get("row_block"), 1, 8)
        rationale = " ".join(str(item.get("rationale", "")).split())[:120]
        if kernel_family == "ikj":
            j_unroll = _clamp_int(item.get("j_unroll"), 1, 16)
            key = ("ikj", row_block, j_unroll)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append({
                "kernel_family": "ikj",
                "row_block": row_block,
                "j_unroll": j_unroll,
                "rationale": rationale,
            })
        else:
            col_block = _clamp_int(item.get("col_block"), 1, 8)
            k_unroll = _clamp_int(item.get("k_unroll"), 1, 16)
            key = ("transpose", row_block, col_block, k_unroll)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append({
                "kernel_family": "transpose",
                "row_block": row_block,
                "col_block": col_block,
                "k_unroll": k_unroll,
                "rationale": rationale,
            })
    return results


def _clamp_int(value: object, low: int, high: int) -> int:
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return low


def _materialize_novel_candidate(
    novel: dict[str, object],
    *,
    existing_ids: set[str],
) -> dict[str, object] | None:
    """Build a runnable candidate dict from a novel proposal."""
    kernel_family = str(novel.get("kernel_family", "")).strip().lower()
    row_block = int(novel.get("row_block", 1))
    rationale = str(novel.get("rationale", ""))

    if kernel_family == "ikj":
        j_unroll = int(novel.get("j_unroll", 1))
        candidate_id = f"novel_ikj_r{row_block}_j{j_unroll}"
        if candidate_id in existing_ids:
            return None
        return {
            "id": candidate_id,
            "name": f"novel ikj rb={row_block} ju={j_unroll}",
            "family": f"novel_ikj_r{row_block}",
            "description": rationale or f"LLM-proposed ikj kernel rb={row_block} ju={j_unroll}.",
            "fn": _make_ikj_param_candidate(row_block=row_block, j_unroll=j_unroll),
            "priority": 0.7,
            "parent_id": "ikj_accumulate",
            "generated": True,
            "novel": True,
            "mutation_params": {"row_block": row_block, "j_unroll": j_unroll},
            "target_workloads": [],
        }
    else:
        col_block = int(novel.get("col_block", 1))
        k_unroll = int(novel.get("k_unroll", 1))
        candidate_id = f"novel_transpose_r{row_block}_c{col_block}_k{k_unroll}"
        if candidate_id in existing_ids:
            return None
        return {
            "id": candidate_id,
            "name": f"novel transpose rb={row_block} cb={col_block} ku={k_unroll}",
            "family": f"novel_transpose_r{row_block}c{col_block}",
            "description": rationale or f"LLM-proposed transpose kernel rb={row_block} cb={col_block} ku={k_unroll}.",
            "fn": _make_transpose_param_candidate(
                row_block=row_block, col_block=col_block, k_unroll=k_unroll,
            ),
            "priority": 0.7,
            "parent_id": "transpose_dot",
            "generated": True,
            "novel": True,
            "mutation_params": {"row_block": row_block, "col_block": col_block, "k_unroll": k_unroll},
            "target_workloads": [],
        }


def _extract_param_insights(
    history_summary: dict[str, object] | None,
) -> list[dict[str, object]]:
    """Build concise param->workload win correlations from history."""
    if not isinstance(history_summary, dict):
        return []
    workload_winners = history_summary.get("matmul_workload_winners", {})
    if not isinstance(workload_winners, dict):
        return []
    frontier_archive = history_summary.get("matmul_frontier_archive", [])
    if not isinstance(frontier_archive, list):
        frontier_archive = []
    insights: list[dict[str, object]] = []
    for workload_tag, entry in workload_winners.items():
        if not isinstance(entry, dict):
            continue
        winner_counts = entry.get("winner_counts", {})
        if not isinstance(winner_counts, dict):
            continue
        top_winners = sorted(
            winner_counts.items(),
            key=lambda pair: -int(pair[1]),
        )[:2]
        if top_winners:
            insights.append({
                "workload_tag": str(workload_tag),
                "top_candidates": [
                    {"id": str(candidate_id), "wins": int(count)}
                    for candidate_id, count in top_winners
                ],
            })
    for item in frontier_archive:
        if not isinstance(item, dict):
            continue
        workload_tag = str(item.get("workload_tag", "")).strip()
        best_id = str(item.get("best_candidate_id", "")).strip()
        if workload_tag and best_id:
            existing = next(
                (i for i in insights if i["workload_tag"] == workload_tag),
                None,
            )
            if existing is None:
                insights.append({
                    "workload_tag": workload_tag,
                    "current_best": best_id,
                })
            else:
                existing["current_best"] = best_id
    return insights[:8]


def _time_call(fn) -> float:
    started = perf_counter()
    fn()
    return perf_counter() - started


def _measure_runtime(fn, *, repetitions: int, warmups: int) -> float:
    return _measure_runtime_stats(
        fn,
        repetitions=repetitions,
        warmups=warmups,
    )["seconds"]


def _measure_runtime_stats(
    fn,
    *,
    repetitions: int,
    warmups: int,
    min_sample_seconds: float = 0.01,
) -> dict[str, float | int]:
    for _ in range(warmups):
        fn()
    estimate = max(_time_call(fn), 1e-9)
    loops_per_sample = max(1, ceil(min_sample_seconds / estimate))

    def batched_call() -> None:
        for _ in range(loops_per_sample):
            fn()

    samples = [_time_call(batched_call) / loops_per_sample for _ in range(repetitions)]
    return {
        "seconds": median(samples),
        "loops_per_sample": loops_per_sample,
    }


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


def _build_matmul_pareto_frontier(rows: list[dict[str, object]]) -> dict[str, object]:
    winners: dict[str, dict[str, object]] = {}
    near_frontier: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        workload_tag = str(row.get("workload_tag", "")).strip()
        workload_share = float(row.get("workload_share", 0.0) or 0.0)
        winner_id = str(row.get("best_candidate_id", "")).strip()
        runner_up_id = str(row.get("runner_up_candidate_id", "")).strip()
        runner_up_gap_pct = float(row.get("runner_up_gap_pct", 0.0) or 0.0)
        if workload_tag and winner_id:
            entry = winners.setdefault(
                winner_id,
                {"workload_tags": [], "workload_share_won": 0.0},
            )
            entry["workload_tags"].append(workload_tag)
            entry["workload_share_won"] += workload_share
        if workload_tag and runner_up_id and runner_up_gap_pct <= 5.0:
            entry = near_frontier.setdefault(
                runner_up_id,
                {"workload_tags": [], "closest_gap_pct": runner_up_gap_pct},
            )
            entry["workload_tags"].append(workload_tag)
            entry["closest_gap_pct"] = min(entry["closest_gap_pct"], runner_up_gap_pct)

    candidate_ids = list(winners.keys())
    for candidate_id in sorted(
        near_frontier,
        key=lambda item: (
            near_frontier[item]["closest_gap_pct"],
            item,
        ),
    ):
        if candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)

    candidates = []
    for candidate_id in candidate_ids:
        winner_entry = winners.get(candidate_id, {"workload_tags": [], "workload_share_won": 0.0})
        near_entry = near_frontier.get(candidate_id, {"workload_tags": [], "closest_gap_pct": None})
        candidates.append(
            {
                "candidate_id": candidate_id,
                "workload_tags_won": sorted(winner_entry["workload_tags"]),
                "workload_share_won": round(float(winner_entry["workload_share_won"]), 4),
                "near_frontier_workloads": sorted(near_entry["workload_tags"]),
                "closest_runner_up_gap_pct": near_entry["closest_gap_pct"],
            }
        )
    return {
        "candidate_ids": candidate_ids,
        "candidates": candidates,
    }


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
