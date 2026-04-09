from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.cli import build_pipeline
from research_engine.executors import (
    DefaultExperimentExecutor,
    LongContextResponsesExecutor,
    ToolUseResponsesExecutor,
)
from research_engine.llm import ResponsesApiClient, ResponsesProviderConfig
from research_engine.models import ExperimentSpec, ResearchTopic


class _FakeResponsesClient:
    def __init__(self) -> None:
        self.config = ResponsesProviderConfig(
            provider_name="azure",
            api_key="azure-key",
            base_url="https://example-resource.openai.azure.com/openai/v1",
            model="gpt-5.4",
        )
        self.calls: list[dict[str, object]] = []

    def generate_json(self, **kwargs) -> dict[str, object]:
        self.calls.append(kwargs)
        if "baseline" in kwargs["schema_name"]:
            return {
                "answers": [
                    {"id": "lc-1", "answer": "unknown"},
                    {"id": "lc-2", "answer": "court opinions"},
                    {"id": "lc-3", "answer": "unknown"},
                ]
            }
        if "tool_use" in kwargs["schema_name"]:
            return {
                "judgments": [
                    {
                        "id": "tu-1",
                        "terminal_first_success": True,
                        "structured_success": False,
                        "note": "Structured policy loses auth state.",
                    },
                    {
                        "id": "tu-2",
                        "terminal_first_success": True,
                        "structured_success": False,
                        "note": "Shell loops preserve local state.",
                    },
                    {
                        "id": "tu-3",
                        "terminal_first_success": True,
                        "structured_success": True,
                        "note": "Both paths are short enough to succeed.",
                    },
                ]
            }
        return {
            "answers": [
                {"id": "lc-1", "answer": "Aster"},
                {"id": "lc-2", "answer": "court opinions"},
                {"id": "lc-3", "answer": "12 documents"},
            ]
        }


class ExecutorTests(unittest.TestCase):
    def test_long_context_responses_executor_emits_expected_artifacts(self) -> None:
        client = _FakeResponsesClient()
        executor = LongContextResponsesExecutor(client=client)

        results = executor.run(
            ResearchTopic(
                name="long-context reasoning",
                objective="improve quality",
                benchmark_id="long-context-reasoning",
            ),
            [
                ExperimentSpec(
                    name="exp-1",
                    benchmark_id="long-context-reasoning",
                    hypothesis_title="Hypothesis",
                    success_metric="accuracy",
                    budget="small",
                    baseline="baseline",
                    protocol=["run"],
                )
            ],
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(results[0].artifact_payloads["baseline-metrics.json"]["accuracy"], 1 / 3)
        self.assertEqual(results[0].artifact_payloads["candidate-metrics.json"]["accuracy"], 1.0)
        self.assertEqual(
            results[0].artifact_payloads["eval-manifest.json"]["executor"],
            "responses_api",
        )

    def test_tool_use_responses_executor_emits_expected_artifacts(self) -> None:
        client = _FakeResponsesClient()
        executor = ToolUseResponsesExecutor(client=client)

        results = executor.run(
            ResearchTopic(
                name="tool-use reliability",
                objective="improve quality",
                benchmark_id="tool-use-reliability",
            ),
            [
                ExperimentSpec(
                    name="exp-1",
                    benchmark_id="tool-use-reliability",
                    hypothesis_title="Hypothesis",
                    success_metric="success rate",
                    budget="small",
                    baseline="baseline",
                    protocol=["run"],
                )
            ],
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            results[0].artifact_payloads["task-suite.json"]["executor"],
            "responses_api",
        )
        self.assertEqual(
            results[0].artifact_payloads["success-summary.json"]["terminal_first_successes"],
            3,
        )
        self.assertEqual(
            results[0].artifact_payloads["success-summary.json"]["structured_successes"],
            1,
        )

    def test_build_pipeline_uses_live_long_context_executor(self) -> None:
        fake_client = _FakeResponsesClient()
        with unittest.mock.patch(
            "research_engine.cli.ResponsesApiClient.from_environment",
            return_value=fake_client,
        ):
            pipeline = build_pipeline(
                use_llm=False,
                max_results=1,
                live_execution=True,
            )

        self.assertIsInstance(pipeline.experiment_executor, DefaultExperimentExecutor)
        self.assertIsInstance(
            pipeline.experiment_executor.long_context_executor,
            LongContextResponsesExecutor,
        )
        self.assertIsInstance(
            pipeline.experiment_executor.tool_use_executor,
            ToolUseResponsesExecutor,
        )


if __name__ == "__main__":
    unittest.main()
