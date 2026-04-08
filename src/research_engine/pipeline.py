from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from .components import (
    ExperimentExecutor,
    ExperimentPlanner,
    HypothesisPlanner,
    MemoWriter,
    ResearchMemory,
    SourceProvider,
    Verifier,
)
from .defaults import (
    CycleVerifier,
    SeedExperimentExecutor,
    SeedExperimentPlanner,
    SeedHypothesisPlanner,
    SeedMemoWriter,
    SeedResearchMemory,
    SeedSourceProvider,
)
from .models import ResearchCycle, ResearchMemo, ResearchRunRecord, ResearchTopic


class ResearchPipeline:
    """Composable scaffold for the empirical research loop.

    The default implementation still uses in-memory seed components, but the
    pipeline now models the explicit boundaries needed for a real system:
    source collection, research memory, hypothesis planning, experiment
    planning, execution, verification, and memo writing.
    """

    def __init__(
        self,
        source_provider: SourceProvider | None = None,
        research_memory: ResearchMemory | None = None,
        hypothesis_planner: HypothesisPlanner | None = None,
        experiment_planner: ExperimentPlanner | None = None,
        experiment_executor: ExperimentExecutor | None = None,
        verifier: Verifier | None = None,
        memo_writer: MemoWriter | None = None,
    ) -> None:
        self.source_provider = source_provider or SeedSourceProvider()
        self.research_memory = research_memory or SeedResearchMemory()
        self.hypothesis_planner = hypothesis_planner or SeedHypothesisPlanner()
        self.experiment_planner = experiment_planner or SeedExperimentPlanner()
        self.experiment_executor = experiment_executor or SeedExperimentExecutor()
        self.verifier = verifier or CycleVerifier()
        self.memo_writer = memo_writer or SeedMemoWriter()

    def build_cycle(self, topic: ResearchTopic) -> ResearchCycle:
        sources = self.source_provider.collect(topic)
        context = self.research_memory.build_context(topic, sources)
        hypotheses = self.hypothesis_planner.plan(topic, context)
        experiments = self.experiment_planner.plan(topic, context, hypotheses)
        results = self.experiment_executor.run(topic, experiments)
        return ResearchCycle(
            topic=topic,
            context=context,
            hypotheses=hypotheses,
            experiments=experiments,
            results=results,
        )

    def run_cycle(self, topic: ResearchTopic) -> ResearchMemo:
        cycle = self.build_cycle(topic)
        verification = self.verifier.verify(cycle)
        return self.memo_writer.render(cycle, verification)

    def run_record(self, topic: ResearchTopic) -> ResearchRunRecord:
        return self.run_record_for(topic=topic, benchmark_id=None)

    def run_record_for(
        self,
        topic: ResearchTopic,
        benchmark_id: str | None,
    ) -> ResearchRunRecord:
        cycle = self.build_cycle(topic)
        verification = self.verifier.verify(cycle)
        memo = self.memo_writer.render(cycle, verification)
        return ResearchRunRecord(
            run_id=uuid4().hex[:12],
            created_at=datetime.now(UTC).isoformat(),
            benchmark_id=benchmark_id,
            cycle=cycle,
            verification=verification,
            memo=memo,
        )

    def run_cycle_dict(self, topic: ResearchTopic) -> dict:
        return asdict(self.run_cycle(topic))

    def run_record_dict(self, topic: ResearchTopic) -> dict:
        return asdict(self.run_record(topic))
