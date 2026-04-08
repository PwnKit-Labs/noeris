from __future__ import annotations

from .components import (
    ExperimentExecutor,
    ExperimentPlanner,
    HypothesisPlanner,
    MemoWriter,
    ResearchMemory,
    SourceProvider,
    Verifier,
)
from .models import (
    Claim,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    Hypothesis,
    ResearchContext,
    ResearchCycle,
    ResearchMemo,
    ResearchSource,
    ResearchTopic,
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
        return ResearchContext(
            topic=topic.name,
            sources=sources,
            claims=[
                Claim(
                    title=f"Known work related to {topic.name}",
                    source="seed",
                    summary=(
                        "This placeholder claim represents the future literature, "
                        "repo ingestion, and prior-art graph."
                    ),
                    evidence_refs=[source.identifier for source in sources],
                )
            ],
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
        return [
            Hypothesis(
                title=f"Test a focused intervention for {topic.name}",
                rationale=(
                    "Start from a bounded, testable change rather than a broad "
                    "architectural rewrite."
                ),
                novelty_reason=(
                    "The engine should eventually derive this from claim gaps and "
                    "contradictions, not from a static template."
                ),
                expected_signal=topic.objective,
                supporting_claims=[claim.title for claim in context.claims],
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
        plans: list[ExperimentSpec] = []
        for index, hypothesis in enumerate(hypotheses, start=1):
            plans.append(
                ExperimentSpec(
                    name=f"exp-{index}",
                    hypothesis_title=hypothesis.title,
                    success_metric="task-specific benchmark improvement",
                    budget="small",
                    protocol=[
                        "Define baseline and holdout evaluation.",
                        "Implement the smallest viable intervention.",
                        "Run a bounded experiment and compare against baseline.",
                        "Record artifacts, traces, and regression risks.",
                        f"Check whether the result advances {topic.objective}.",
                    ],
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
        next_actions = [
            "Attach a live paper and repository ingestion backend.",
            "Persist a prior-art graph with contradiction tracking.",
            "Replace the seed executor with a reproducible runtime.",
        ]
        if verification.blockers:
            next_actions.insert(
                0,
                "Resolve verification blockers before treating this cycle as evidence-backed.",
            )

        return ResearchMemo(
            topic=cycle.topic.name,
            summary=(
                "Initial cycle scaffolded from the topic objective. "
                "Real source ingestion, ranking, and experiment execution are not "
                "wired in yet."
            ),
            sources=cycle.context.sources,
            claims=cycle.context.claims,
            hypotheses=cycle.hypotheses,
            experiments=cycle.experiments,
            results=cycle.results,
            next_actions=next_actions,
            risks=verification.blockers,
        )
