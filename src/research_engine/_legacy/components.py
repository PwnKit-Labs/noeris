from __future__ import annotations

from typing import Protocol

from ..models import (
    ExperimentResult,
    ExperimentSpec,
    Hypothesis,
    ResearchContext,
    ResearchCycle,
    ResearchMemo,
    ResearchSource,
    ResearchTopic,
    VerificationReport,
)


class SourceProvider(Protocol):
    def collect(self, topic: ResearchTopic) -> list[ResearchSource]:
        """Return papers, repos, benchmarks, or other sources for a topic."""


class ResearchMemory(Protocol):
    def build_context(
        self,
        topic: ResearchTopic,
        sources: list[ResearchSource],
    ) -> ResearchContext:
        """Build topic-local claims, contradictions, and open questions."""


class HypothesisPlanner(Protocol):
    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
    ) -> list[Hypothesis]:
        """Generate and rank candidate hypotheses."""


class ExperimentPlanner(Protocol):
    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
        hypotheses: list[Hypothesis],
    ) -> list[ExperimentSpec]:
        """Create bounded experiments from ranked hypotheses."""


class ExperimentExecutor(Protocol):
    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        """Execute or simulate experiments and record outcomes."""


class Verifier(Protocol):
    def verify(self, cycle: ResearchCycle) -> VerificationReport:
        """Check whether a cycle produced evidence-backed output."""


class MemoWriter(Protocol):
    def render(
        self,
        cycle: ResearchCycle,
        verification: VerificationReport,
    ) -> ResearchMemo:
        """Produce the user-facing research memo."""
