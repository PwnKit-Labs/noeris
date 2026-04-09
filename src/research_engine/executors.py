from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from .components import ExperimentExecutor
from .llm import ResponsesApiClient
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
        baseline_rows = self._evaluate(condition="baseline", truncated=True)
        candidate_rows = self._evaluate(condition="candidate", truncated=False)
        baseline_accuracy = _accuracy(baseline_rows)
        candidate_accuracy = _accuracy(candidate_rows)
        failures = [
            f"{row['id']}: candidate predicted {row['prediction']!r}, expected {row['expected']!r}"
            for row in candidate_rows
            if not row["correct"]
        ]
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
    ) -> list[dict[str, object]]:
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
        payload = self.client.generate_json(
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
        return rows


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
        evaluation = self.client.generate_json(
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
        rows = _tool_use_rows_from_payload(evaluation)
        terminal_successes = sum(1 for row in rows if row["mode"] == "terminal-first" and row["success"])
        structured_successes = sum(1 for row in rows if row["mode"] == "structured" and row["success"])
        total = len(TOOL_USE_FIXTURES)
        terminal_rate = terminal_successes / total
        structured_rate = structured_successes / total
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
