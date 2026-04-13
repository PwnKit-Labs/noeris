from __future__ import annotations

from ..benchmarks import benchmark_from_topic_constraints, get_benchmark
from .components import (
    ExperimentExecutor,
    ExperimentPlanner,
    HypothesisPlanner,
    MemoWriter,
    ResearchMemory,
    SourceProvider,
    Verifier,
)
from ..models import (
    Claim,
    Contradiction,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    Hypothesis,
    ResearchContext,
    ResearchCycle,
    ResearchMemo,
    ResearchSource,
    ResearchTopic,
    SourceAssessment,
    VerificationReport,
)


class SeedSourceProvider(SourceProvider):
    def collect(self, topic: ResearchTopic) -> list[ResearchSource]:
        return [
            ResearchSource(
                identifier="seed://bootstrap",
                kind="seed",
                title=f"Seed source for {topic.name}",
                locator="seed://bootstrap",
                excerpt=(
                    "Placeholder source until live paper, repo, and benchmark "
                    "ingestion are connected."
                ),
            )
        ]


class SeedResearchMemory(ResearchMemory):
    def build_context(
        self,
        topic: ResearchTopic,
        sources: list[ResearchSource],
    ) -> ResearchContext:
        claims = _claims_from_sources(topic=topic, sources=sources)
        return ResearchContext(
            topic=topic.name,
            sources=sources,
            source_assessments=[
                SourceAssessment(
                    source_id=source.identifier,
                    confidence="low",
                    rationale="Seed source placeholder until live evidence ranking is attached.",
                )
                for source in sources
            ],
            claims=claims,
            open_questions=[
                "Which bounded experiment is cheap enough to run first?",
                "Which paper or repo claims are most likely to be underexplored?",
            ],
            contradictions=[],
        )


class SeedHypothesisPlanner(HypothesisPlanner):
    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
    ) -> list[Hypothesis]:
        benchmark = _topic_benchmark(topic)
        novelty_reason = (
            "The engine should eventually derive this from claim gaps and "
            "contradictions, not from a static template."
        )
        if benchmark is not None:
            novelty_reason = (
                f"Tailor the search to the {benchmark.name} benchmark and avoid "
                "broad speculative rewrites."
            )
        return [
            Hypothesis(
                title=f"Test a focused intervention for {topic.name}",
                rationale=(
                    "Start from a bounded, testable change rather than a broad "
                    "architectural rewrite."
                ),
                novelty_reason=novelty_reason,
                expected_signal=topic.objective,
                supporting_claims=[claim.title for claim in context.claims],
                priority_score=1.0,
                ranking_rationale="Default seed hypothesis with no comparative ranking signal.",
            )
        ]


class SeedExperimentPlanner(ExperimentPlanner):
    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
        hypotheses: list[Hypothesis],
    ) -> list[ExperimentSpec]:
        del context
        benchmark = _topic_benchmark(topic)
        plans: list[ExperimentSpec] = []
        for index, hypothesis in enumerate(hypotheses, start=1):
            plans.append(
                ExperimentSpec(
                    name=_experiment_name(index=index, topic=topic),
                    benchmark_id=benchmark.benchmark_id if benchmark else None,
                    hypothesis_title=hypothesis.title,
                    success_metric=(
                        benchmark.success_metric
                        if benchmark is not None
                        else "task-specific benchmark improvement"
                    ),
                    budget="small",
                    baseline=_baseline(topic=topic, benchmark=benchmark),
                    required_artifacts=_required_artifacts(benchmark),
                    evaluation_notes=_evaluation_notes(benchmark),
                    protocol=_protocol(topic=topic, benchmark=benchmark),
                )
            )
        return plans


class SeedExperimentExecutor(ExperimentExecutor):
    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [
            ExperimentResult(
                spec_name=experiment.name,
                status=ExperimentStatus.NOT_RUN,
                outcome_summary=(
                    "Execution backend not attached yet; experiment remains planned."
                ),
                artifact_refs=[],
            )
            for experiment in experiments
        ]


class CycleVerifier(Verifier):
    def verify(self, cycle: ResearchCycle) -> VerificationReport:
        checks: list[str] = []
        blockers: list[str] = []

        if cycle.context.sources:
            checks.append("sources_present")
        else:
            blockers.append("no_sources")

        if cycle.context.claims:
            checks.append("claims_present")
        else:
            blockers.append("no_claims")

        if cycle.hypotheses:
            checks.append("hypotheses_present")
        else:
            blockers.append("no_hypotheses")

        if cycle.experiments:
            checks.append("experiments_present")
        else:
            blockers.append("no_experiments")

        if cycle.results:
            checks.append("results_recorded")
        else:
            blockers.append("no_results")

        if any(result.status != ExperimentStatus.NOT_RUN for result in cycle.results):
            checks.append("execution_backends_attached")
        else:
            blockers.append("no_empirical_execution")

        return VerificationReport(
            passed=not blockers,
            checks=checks,
            blockers=blockers,
        )


class SeedMemoWriter(MemoWriter):
    def render(
        self,
        cycle: ResearchCycle,
        verification: VerificationReport,
    ) -> ResearchMemo:
        empirical_execution = any(
            result.status != ExperimentStatus.NOT_RUN for result in cycle.results
        )
        live_sources_attached = any(source.kind != "seed" for source in cycle.context.sources)
        live_responses_execution = any(
            (
                isinstance(result.artifact_payloads.get("eval-manifest.json"), dict)
                and result.artifact_payloads["eval-manifest.json"].get("executor") == "responses_api"
            )
            or (
                isinstance(result.artifact_payloads.get("task-suite.json"), dict)
                and result.artifact_payloads["task-suite.json"].get("executor") == "responses_api"
            )
            for result in cycle.results
        )
        live_runtime_execution = any(
            (
                isinstance(result.artifact_payloads.get("hardware-profile.json"), dict)
                and result.artifact_payloads["hardware-profile.json"].get("executor")
                == "python_cpu_microbenchmark"
            )
            for result in cycle.results
        )
        summary = (
            "Initial cycle scaffolded from the topic objective. "
            "Real source ingestion, ranking, and experiment execution are not "
            "wired in yet."
        )
        next_actions = [
            "Attach a live paper and repository ingestion backend.",
            "Persist a prior-art graph with contradiction tracking.",
            "Replace the seed executor with a reproducible runtime.",
        ]
        if live_sources_attached:
            summary = (
                "Research cycle executed with live source discovery and structured "
                "planning, but no experiment executor is attached yet."
            )
            next_actions[0] = "Attach or select an experiment executor for this topic."
        if empirical_execution:
            summary = (
                "Initial cycle scaffolded from the topic objective and executed "
                "through the current deterministic benchmark lane. Source "
                "ingestion and ranking still rely on seed components, so this "
                "remains a bounded offline harness."
            )
            if live_sources_attached:
                summary = (
                    "Research cycle executed with live source discovery, model-backed "
                    "planning, and the current deterministic benchmark lane. This "
                    "remains a bounded offline harness until a live experiment runtime "
                    "is attached."
                )
                next_actions[0] = "Attach or select an experiment executor for this topic."
            if live_responses_execution:
                summary = (
                    "Research cycle executed with live source discovery, model-backed "
                    "planning, and model-backed benchmark execution."
                )
                next_actions = [
                    "Persist contradiction tracking and source confidence across runs.",
                    "Provide pricing inputs and budget thresholds for estimated live-run cost accounting.",
                    "Broaden the benchmark fixture set beyond the current small replay harness.",
                ]
            elif live_runtime_execution:
                summary = (
                    "Research cycle executed with a real benchmark runtime and artifact-backed measurement."
                )
                next_actions = [
                    "Broaden the benchmark fixture set beyond the current small replay harness.",
                    "Add stronger measurement statistics such as warmup control and percentile summaries.",
                    "Move from CPU microbenchmarks to a richer hardware-backed matmul runtime when available.",
                ]
            else:
                next_actions[-1] = (
                    "Replace the current synthetic offline executor with a "
                    "benchmark-specific live runtime."
                )
        if verification.blockers:
            next_actions.insert(
                0,
                "Resolve verification blockers before treating this cycle as evidence-backed.",
            )

        return ResearchMemo(
            topic=cycle.topic.name,
            summary=summary,
            sources=cycle.context.sources,
            source_assessments=cycle.context.source_assessments,
            claims=cycle.context.claims,
            contradictions=cycle.context.contradictions,
            hypotheses=cycle.hypotheses,
            experiments=cycle.experiments,
            results=cycle.results,
            next_actions=next_actions,
            risks=verification.blockers,
        )


def _topic_benchmark(topic: ResearchTopic):
    if topic.benchmark_id:
        return get_benchmark(topic.benchmark_id)
    return benchmark_from_topic_constraints(topic.constraints)


def _claims_from_sources(
    *,
    topic: ResearchTopic,
    sources: list[ResearchSource],
) -> list[Claim]:
    if not sources:
        return [
            Claim(
                title=f"Known work related to {topic.name}",
                source="seed",
                summary=(
                    "This placeholder claim represents the future literature, "
                    "repo ingestion, and prior-art graph."
                ),
                evidence_refs=[],
            )
        ]

    claims: list[Claim] = []
    for source in sources[:3]:
        if source.kind == "paper":
            claims.append(
                Claim(
                    title=f"{source.title} is relevant to {topic.name}",
                    source=source.identifier,
                    summary=(
                        "Derived from the source title and excerpt; replace with "
                        "richer claim extraction when a stronger memory layer is attached."
                    ),
                    evidence_refs=[source.identifier],
                    evidence_kind="source-derived",
                )
            )
        else:
            claims.append(
                Claim(
                    title=f"{source.title} may contain useful implementation evidence",
                    source=source.identifier,
                    summary=(
                        "Repository source identified as potentially relevant to the topic."
                    ),
                    evidence_refs=[source.identifier],
                    evidence_kind="source-derived",
                )
            )
    return claims


def _experiment_name(index: int, topic: ResearchTopic) -> str:
    if topic.benchmark_id:
        return f"{topic.benchmark_id}-exp-{index}"
    return f"exp-{index}"


def _baseline(topic: ResearchTopic, benchmark) -> str:
    if benchmark is None:
        return "Establish a named baseline and holdout evaluation before implementation."
    return benchmark.baseline_guidance


def _required_artifacts(benchmark) -> list[str]:
    if benchmark is None:
        return [
            "experiment-config.json",
            "result-summary.json",
            "comparison-notes.md",
        ]
    return benchmark.required_artifacts


def _evaluation_notes(benchmark) -> list[str]:
    if benchmark is None:
        return [
            "Record baseline and candidate outputs separately.",
            "Keep evaluation deterministic enough to replay.",
        ]
    return [
        f"CI lane: {benchmark.ci_lane}.",
        "Do not treat the run as evidence-backed until the required artifacts exist.",
    ]


def _protocol(topic: ResearchTopic, benchmark) -> list[str]:
    if benchmark is None:
        return [
            "Define baseline and holdout evaluation.",
            "Implement the smallest viable intervention.",
            "Run a bounded experiment and compare against baseline.",
            "Record artifacts, traces, and regression risks.",
            f"Check whether the result advances {topic.objective}.",
        ]

    if benchmark.benchmark_id == "matmul-speedup":
        return [
            "Fix hardware, tensor shapes, dtypes, and baseline kernel path.",
            "Implement the smallest kernel or scheduling intervention worth testing.",
            "Run bounded throughput and latency measurements against the baseline.",
            "Capture raw timings, hardware profile, and comparison notes.",
            f"Check whether the result advances {topic.objective}.",
        ]

    if benchmark.benchmark_id == "long-context-reasoning":
        return [
            "Fix the model, eval set, and long-context baseline configuration.",
            "Implement the smallest retrieval, memory, or context intervention worth testing.",
            "Run the candidate on the same eval slice as the baseline.",
            "Capture eval manifests, metrics, and a short failure analysis.",
            f"Check whether the result advances {topic.objective}.",
        ]

    if benchmark.benchmark_id == "tool-use-reliability":
        return [
            "Fix the task suite and define a terminal-first baseline plus one structured-tool comparison policy.",
            "Implement the smallest planner, memory, recovery, or interface intervention worth testing.",
            "Run the candidate and the baseline on the same task slice.",
            "Capture terminal transcripts, tool-selection summaries, success summaries, and an error taxonomy.",
            f"Check whether the result advances {topic.objective}.",
        ]

    return [
        "Define baseline and holdout evaluation.",
        "Implement the smallest viable intervention.",
        "Run a bounded experiment and compare against baseline.",
        "Record artifacts, traces, and regression risks.",
        f"Check whether the result advances {topic.objective}.",
    ]
