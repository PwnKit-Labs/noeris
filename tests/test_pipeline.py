import unittest

from tests import _pathfix  # noqa: F401

from research_engine.cli import ARCHITECTURE, THESIS
from research_engine.components import (
    ExperimentExecutor,
    ExperimentPlanner,
    HypothesisPlanner,
    MemoWriter,
    ResearchMemory,
    SourceProvider,
    Verifier,
)
from research_engine.models import (
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
from research_engine.pipeline import ResearchPipeline


class StubSourceProvider(SourceProvider):
    def collect(self, topic: ResearchTopic) -> list[ResearchSource]:
        return [
            ResearchSource(
                identifier="src-1",
                kind="paper",
                title=f"{topic.name} source",
                locator="memory://src-1",
                excerpt="stub source",
            )
        ]


class StubResearchMemory(ResearchMemory):
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
                    title="stub claim",
                    source=sources[0].identifier,
                    summary="stub summary",
                    evidence_refs=[sources[0].identifier],
                )
            ],
            open_questions=["what next?"],
        )


class StubHypothesisPlanner(HypothesisPlanner):
    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
    ) -> list[Hypothesis]:
        return [
            Hypothesis(
                title=f"hypothesis for {topic.name}",
                rationale="because",
                novelty_reason="gap",
                expected_signal=topic.objective,
                supporting_claims=[claim.title for claim in context.claims],
            )
        ]


class StubExperimentPlanner(ExperimentPlanner):
    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
        hypotheses: list[Hypothesis],
    ) -> list[ExperimentSpec]:
        del topic
        del context
        return [
            ExperimentSpec(
                name="exp-stub",
                benchmark_id=None,
                hypothesis_title=hypotheses[0].title,
                success_metric="score",
                budget="tiny",
                baseline="stub baseline",
                required_artifacts=["artifact.json"],
                evaluation_notes=["note"],
                protocol=["run"],
            )
        ]


class StubExperimentExecutor(ExperimentExecutor):
    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [
            ExperimentResult(
                spec_name=experiments[0].name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary="improved",
                artifact_refs=["artifact://result"],
            )
        ]


class StubLiveExperimentExecutor(ExperimentExecutor):
    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [
            ExperimentResult(
                spec_name=experiments[0].name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary="live benchmark run",
                artifact_refs=["eval-manifest.json"],
                artifact_payloads={
                    "eval-manifest.json": {
                        "benchmark": "long-context-reasoning",
                        "executor": "responses_api",
                    }
                },
            )
        ]


class StubLiveToolUseExecutor(ExperimentExecutor):
    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [
            ExperimentResult(
                spec_name=experiments[0].name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary="live tool-use benchmark run",
                artifact_refs=["task-suite.json"],
                artifact_payloads={
                    "task-suite.json": {
                        "benchmark": "tool-use-reliability",
                        "executor": "responses_api",
                    }
                },
            )
        ]


class StubVerifier(Verifier):
    def verify(self, cycle: ResearchCycle) -> VerificationReport:
        return VerificationReport(
            passed=True,
            checks=[
                cycle.context.sources[0].identifier,
                cycle.hypotheses[0].supporting_claims[0],
            ],
            blockers=[],
        )


class StubMemoWriter(MemoWriter):
    def render(
        self,
        cycle: ResearchCycle,
        verification: VerificationReport,
    ) -> ResearchMemo:
        return ResearchMemo(
            topic=cycle.topic.name,
            summary="custom memo",
            sources=cycle.context.sources,
            claims=cycle.context.claims,
            hypotheses=cycle.hypotheses,
            experiments=cycle.experiments,
            results=cycle.results,
            next_actions=verification.checks,
            risks=verification.blockers,
        )


class ResearchPipelineTests(unittest.TestCase):
    def test_thesis_mentions_research_engine(self) -> None:
        self.assertIn("autonomous R&D engine", THESIS)

    def test_architecture_mentions_stateful_components(self) -> None:
        self.assertIn("Stateful research loop", ARCHITECTURE)

    def test_run_cycle_returns_expected_sections(self) -> None:
        pipeline = ResearchPipeline()
        memo = pipeline.run_cycle(
            ResearchTopic(
                name="long-context reasoning",
                objective="improve benchmark performance",
                constraints=["small compute budget"],
            )
        )

        self.assertEqual(memo.topic, "long-context reasoning")
        self.assertTrue(memo.sources)
        self.assertTrue(memo.claims)
        self.assertTrue(memo.hypotheses)
        self.assertTrue(memo.experiments)
        self.assertTrue(memo.results)
        self.assertTrue(memo.next_actions)
        self.assertTrue(memo.risks)
        self.assertEqual(memo.experiments[0].benchmark_id, None)
        self.assertTrue(memo.experiments[0].required_artifacts)

    def test_pipeline_accepts_component_overrides(self) -> None:
        pipeline = ResearchPipeline(
            source_provider=StubSourceProvider(),
            research_memory=StubResearchMemory(),
            hypothesis_planner=StubHypothesisPlanner(),
            experiment_planner=StubExperimentPlanner(),
            experiment_executor=StubExperimentExecutor(),
            verifier=StubVerifier(),
            memo_writer=StubMemoWriter(),
        )

        memo = pipeline.run_cycle(
            ResearchTopic(
                name="agentic memory",
                objective="improve factual retention",
            )
        )

        self.assertEqual(memo.sources[0].identifier, "src-1")
        self.assertEqual(memo.hypotheses[0].supporting_claims, ["stub claim"])
        self.assertEqual(memo.results[0].status, ExperimentStatus.COMPLETED)
        self.assertEqual(memo.next_actions, ["src-1", "stub claim"])
        self.assertEqual(memo.risks, [])

    def test_benchmark_topic_changes_seed_experiment_shape(self) -> None:
        pipeline = ResearchPipeline()
        memo = pipeline.run_cycle(
            ResearchTopic(
                name="matrix multiplication speedup",
                objective="discover validated kernel-level speedups",
                benchmark_id="matmul-speedup",
                constraints=["benchmark_id:matmul-speedup"],
            )
        )

        experiment = memo.experiments[0]
        self.assertEqual(experiment.benchmark_id, "matmul-speedup")
        self.assertIn("hardware-profile.json", experiment.required_artifacts)
        self.assertIn("baseline kernel", experiment.baseline)
        self.assertIn("throughput and latency", " ".join(experiment.protocol))

    def test_long_context_benchmark_run_produces_empirical_result(self) -> None:
        pipeline = ResearchPipeline()
        record = pipeline.run_record_for(
            topic=ResearchTopic(
                name="long-context reasoning",
                objective="improve long-context eval quality",
                benchmark_id="long-context-reasoning",
                constraints=["benchmark_id:long-context-reasoning"],
            ),
            benchmark_id="long-context-reasoning",
        )

        self.assertTrue(record.verification.passed)
        self.assertIn("execution_backends_attached", record.verification.checks)
        self.assertEqual(record.cycle.results[0].status, ExperimentStatus.COMPLETED)
        self.assertNotIn("not wired in yet", record.memo.summary)
        self.assertIn("deterministic benchmark lane", record.memo.summary)
        self.assertIn("synthetic offline executor", " ".join(record.memo.next_actions))
        self.assertIn(
            "eval-manifest.json",
            record.cycle.results[0].artifact_refs,
        )

    def test_tool_use_benchmark_run_produces_empirical_result(self) -> None:
        pipeline = ResearchPipeline()
        record = pipeline.run_record_for(
            topic=ResearchTopic(
                name="tool-use reliability",
                objective="increase correctness and recovery",
                benchmark_id="tool-use-reliability",
                constraints=["benchmark_id:tool-use-reliability"],
            ),
            benchmark_id="tool-use-reliability",
        )

        self.assertTrue(record.verification.passed)
        self.assertEqual(record.cycle.results[0].status, ExperimentStatus.COMPLETED)
        self.assertIn(
            "tool-selection-summary.json",
            record.cycle.results[0].artifact_refs,
        )

    def test_matmul_benchmark_run_produces_empirical_result(self) -> None:
        pipeline = ResearchPipeline()
        record = pipeline.run_record_for(
            topic=ResearchTopic(
                name="matrix multiplication speedup",
                objective="discover validated kernel-level speedups",
                benchmark_id="matmul-speedup",
                constraints=["benchmark_id:matmul-speedup"],
            ),
            benchmark_id="matmul-speedup",
        )

        self.assertTrue(record.verification.passed)
        self.assertEqual(record.cycle.results[0].status, ExperimentStatus.COMPLETED)
        self.assertIn(
            "raw-timing-results.json",
            record.cycle.results[0].artifact_refs,
        )

    def test_non_benchmark_run_keeps_planning_only_summary(self) -> None:
        pipeline = ResearchPipeline()
        record = pipeline.run_record(
            ResearchTopic(
                name="memory routing",
                objective="improve measurable ML/LLM performance",
            )
        )

        self.assertFalse(record.verification.passed)
        self.assertIn("not wired in yet", record.memo.summary)
        self.assertIn("Replace the seed executor", " ".join(record.memo.next_actions))

    def test_live_source_planning_summary_mentions_missing_executor(self) -> None:
        pipeline = ResearchPipeline(source_provider=StubSourceProvider())
        record = pipeline.run_record(
            ResearchTopic(
                name="long-context reasoning",
                objective="improve benchmark performance",
            )
        )

        self.assertFalse(record.verification.passed)
        self.assertIn("live source discovery", record.memo.summary)
        self.assertIn("Attach or select an experiment executor", record.memo.next_actions[1])

    def test_live_execution_summary_mentions_model_backed_execution(self) -> None:
        pipeline = ResearchPipeline(
            source_provider=StubSourceProvider(),
            experiment_executor=StubLiveExperimentExecutor(),
        )
        record = pipeline.run_record_for(
            topic=ResearchTopic(
                name="long-context reasoning",
                objective="improve benchmark performance",
                benchmark_id="long-context-reasoning",
                constraints=["benchmark_id:long-context-reasoning"],
            ),
            benchmark_id="long-context-reasoning",
        )

        self.assertTrue(record.verification.passed)
        self.assertIn("model-backed benchmark execution", record.memo.summary)

    def test_live_matmul_execution_summary_mentions_real_runtime(self) -> None:
        pipeline = ResearchPipeline(
            experiment_executor=StubLiveMatmulExecutor(),
        )
        record = pipeline.run_record_for(
            topic=ResearchTopic(
                name="matrix multiplication speedup",
                objective="discover validated kernel-level speedups",
                benchmark_id="matmul-speedup",
                constraints=["benchmark_id:matmul-speedup"],
            ),
            benchmark_id="matmul-speedup",
        )

        self.assertTrue(record.verification.passed)
        self.assertIn("real benchmark runtime", record.memo.summary)


class StubLiveMatmulExecutor(ExperimentExecutor):
    def run(
        self,
        topic: ResearchTopic,
        experiments: list[ExperimentSpec],
    ) -> list[ExperimentResult]:
        del topic
        return [
            ExperimentResult(
                spec_name=experiments[0].name,
                status=ExperimentStatus.COMPLETED,
                outcome_summary="live matmul benchmark run",
                artifact_refs=["hardware-profile.json"],
                artifact_payloads={
                    "hardware-profile.json": {
                        "benchmark": "matmul-speedup",
                        "executor": "python_cpu_microbenchmark",
                    }
                },
            )
        ]
        self.assertIn("estimated live-run cost accounting", " ".join(record.memo.next_actions))

    def test_live_tool_use_execution_summary_mentions_model_backed_execution(self) -> None:
        pipeline = ResearchPipeline(
            source_provider=StubSourceProvider(),
            experiment_executor=StubLiveToolUseExecutor(),
        )
        record = pipeline.run_record_for(
            topic=ResearchTopic(
                name="tool-use reliability",
                objective="improve task success",
                benchmark_id="tool-use-reliability",
                constraints=["benchmark_id:tool-use-reliability"],
            ),
            benchmark_id="tool-use-reliability",
        )

        self.assertTrue(record.verification.passed)
        self.assertIn("model-backed benchmark execution", record.memo.summary)


if __name__ == "__main__":
    unittest.main()
