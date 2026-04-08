from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(slots=True)
class ResearchTopic:
    name: str
    objective: str
    constraints: list[str] = field(default_factory=list)
    benchmark_id: str | None = None


@dataclass(slots=True)
class Claim:
    title: str
    source: str
    summary: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchSource:
    identifier: str
    kind: str
    title: str
    locator: str
    excerpt: str


@dataclass(slots=True)
class ResearchContext:
    topic: str
    sources: list[ResearchSource]
    claims: list[Claim]
    open_questions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Hypothesis:
    title: str
    rationale: str
    novelty_reason: str
    expected_signal: str
    supporting_claims: list[str] = field(default_factory=list)


class ExperimentStatus(StrEnum):
    PLANNED = "planned"
    COMPLETED = "completed"
    REJECTED = "rejected"
    NOT_RUN = "not_run"


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    benchmark_id: str | None
    hypothesis_title: str
    success_metric: str
    budget: str
    baseline: str
    protocol: list[str]
    required_artifacts: list[str] = field(default_factory=list)
    evaluation_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentResult:
    spec_name: str
    status: ExperimentStatus
    outcome_summary: str
    artifact_refs: list[str] = field(default_factory=list)
    artifact_payloads: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchCycle:
    topic: ResearchTopic
    context: ResearchContext
    hypotheses: list[Hypothesis]
    experiments: list[ExperimentSpec]
    results: list[ExperimentResult]


@dataclass(slots=True)
class ResearchMemo:
    topic: str
    summary: str
    sources: list[ResearchSource]
    claims: list[Claim]
    hypotheses: list[Hypothesis]
    experiments: list[ExperimentSpec]
    results: list[ExperimentResult]
    next_actions: list[str]
    risks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VerificationReport:
    passed: bool
    checks: list[str]
    blockers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchRunRecord:
    run_id: str
    created_at: str
    benchmark_id: str | None
    cycle: ResearchCycle
    verification: VerificationReport
    memo: ResearchMemo


@dataclass(slots=True)
class BenchmarkGoal:
    benchmark_id: str
    name: str
    category: str
    goal: str
    success_metric: str
    why_it_matters: str
    baseline_guidance: str = ""
    required_artifacts: list[str] = field(default_factory=list)
    ci_lane: str = "scheduled-benchmark"
    starter_topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchAgendaItem:
    area_id: str
    name: str
    category: str
    recommended_mode: str
    priority: str
    why_it_matters: str
    benchmark_fit: str
    starter_questions: list[str] = field(default_factory=list)
