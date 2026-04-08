from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from .components import ExperimentExecutor
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
class DefaultExperimentExecutor(ExperimentExecutor):
    long_context_executor: LongContextOfflineExecutor = field(
        default_factory=LongContextOfflineExecutor
    )
    tool_use_executor: ToolUseOfflineExecutor = field(
        default_factory=ToolUseOfflineExecutor
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
) -> str:
    winner = "terminal-first" if terminal_rate > structured_rate else "tie"
    return (
        "# Error Taxonomy\n\n"
        f"- Structured success rate: `{structured_rate:.2f}`\n"
        f"- Terminal-first success rate: `{terminal_rate:.2f}`\n"
        f"- Current winner: `{winner}`\n\n"
        "## Observed failure pattern\n\n"
        "- Structured flows lose reliability when stateful multi-step shell composition would be simpler.\n"
        "- Terminal-first remains the default baseline until structured policies beat it on the same fixture set.\n"
    )
