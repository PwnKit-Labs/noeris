from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.cli import build_pipeline
from research_engine.executors import (
    DefaultExperimentExecutor,
    LONG_CONTEXT_FIXTURES,
    LongContextResponsesExecutor,
    LIVE_MATMUL_FIXTURES,
    MatmulPythonExecutor,
    TOOL_USE_FIXTURES,
    ToolUseResponsesExecutor,
)
from research_engine.llm import ResponsesJsonResult, ResponsesProviderConfig
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
        return self.generate_json_result(**kwargs).data

    def generate_json_result(self, **kwargs) -> ResponsesJsonResult:
        self.calls.append(kwargs)
        if "baseline" in kwargs["schema_name"]:
            data = {
                "answers": [
                    {"id": "lc-1", "answer": "unknown"},
                    {"id": "lc-2", "answer": "court opinions"},
                    {"id": "lc-3", "answer": "unknown"},
                    {"id": "lc-4", "answer": "unknown"},
                    {"id": "lc-5", "answer": "unknown"},
                ]
            }
            raw = {"usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}}
            return ResponsesJsonResult(data=data, raw_response=raw)
        if "tool_use" in kwargs["schema_name"]:
            data = {
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
                    {
                        "id": "tu-4",
                        "terminal_first_success": True,
                        "structured_success": False,
                        "note": "Local filtering is simpler in a shell pipeline.",
                    },
                    {
                        "id": "tu-5",
                        "terminal_first_success": True,
                        "structured_success": False,
                        "note": "Rate-limit aware loops are easier in terminal-first mode.",
                    },
                ]
            }
            raw = {"usage": {"input_tokens": 150, "output_tokens": 25, "total_tokens": 175}}
            return ResponsesJsonResult(data=data, raw_response=raw)
        data = {
            "answers": [
                {"id": "lc-1", "answer": "Aster"},
                {"id": "lc-2", "answer": "court opinions"},
                {"id": "lc-3", "answer": "12 documents"},
                {"id": "lc-4", "answer": "Nimbus"},
                {"id": "lc-5", "answer": "8 documents"},
            ]
        }
        raw = {"usage": {"input_tokens": 120, "output_tokens": 12, "total_tokens": 132}}
        return ResponsesJsonResult(data=data, raw_response=raw)


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
        self.assertEqual(results[0].artifact_payloads["baseline-metrics.json"]["accuracy"], 1 / 5)
        self.assertEqual(results[0].artifact_payloads["candidate-metrics.json"]["accuracy"], 1.0)
        self.assertEqual(
            results[0].artifact_payloads["eval-manifest.json"]["executor"],
            "responses_api",
        )
        self.assertEqual(
            results[0].artifact_payloads["cost-summary.json"]["request_count"],
            2,
        )
        self.assertGreaterEqual(
            results[0].artifact_payloads["cost-summary.json"]["total_tokens"],
            0,
        )

    def test_cost_summary_tracks_thresholds_when_pricing_env_is_set(self) -> None:
        client = _FakeResponsesClient()
        executor = LongContextResponsesExecutor(client=client)

        with unittest.mock.patch.dict(
            "os.environ",
            {
                "NOERIS_INPUT_TOKEN_COST_USD_PER_1M": "1.0",
                "NOERIS_OUTPUT_TOKEN_COST_USD_PER_1M": "2.0",
                "NOERIS_COST_BUDGET_USD": "0.0001",
                "NOERIS_LATENCY_BUDGET_MS": "1",
            },
            clear=False,
        ):
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

        cost_summary = results[0].artifact_payloads["cost-summary.json"]
        self.assertIsNotNone(cost_summary["estimated_cost_usd"])
        self.assertTrue(cost_summary["cost_budget_exceeded"])
        self.assertTrue(cost_summary["warnings"])
        self.assertIn("Estimated cost", cost_summary["warnings"][0])

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
            len(TOOL_USE_FIXTURES),
        )
        self.assertEqual(
            results[0].artifact_payloads["success-summary.json"]["structured_successes"],
            1,
        )
        self.assertEqual(
            results[0].artifact_payloads["cost-summary.json"]["request_count"],
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
                benchmark_id="long-context-reasoning",
            )

        self.assertIsInstance(pipeline.experiment_executor, DefaultExperimentExecutor)
        self.assertIsInstance(
            pipeline.experiment_executor.long_context_executor,
            LongContextResponsesExecutor,
        )

    def test_matmul_python_executor_emits_expected_artifacts(self) -> None:
        executor = MatmulPythonExecutor(repetitions=1)

        results = executor.run(
            ResearchTopic(
                name="matrix multiplication speedup",
                objective="improve quality",
                benchmark_id="matmul-speedup",
            ),
            [
                ExperimentSpec(
                    name="exp-1",
                    benchmark_id="matmul-speedup",
                    hypothesis_title="Hypothesis",
                    success_metric="throughput",
                    budget="small",
                    baseline="baseline",
                    protocol=["run"],
                )
            ],
        )

        self.assertEqual(
            results[0].artifact_payloads["hardware-profile.json"]["executor"],
            "python_cpu_microbenchmark",
        )
        self.assertEqual(
            len(results[0].artifact_payloads["raw-timing-results.json"]["rows"]),
            len(LIVE_MATMUL_FIXTURES),
        )
        self.assertEqual(
            len(results[0].artifact_payloads["candidate-catalog.json"]["selected_candidates"]),
            5,
        )
        self.assertEqual(results[0].artifact_payloads["candidate-proposals.json"]["source"], "none")
        self.assertIn("shape-focus.json", results[0].artifact_refs)
        self.assertTrue(results[0].artifact_payloads["candidate-catalog.json"]["pruned_candidates"])
        self.assertTrue(
            any(
                candidate.get("generated")
                for candidate in results[0].artifact_payloads["candidate-catalog.json"]["selected_candidates"]
            )
        )
        self.assertIn("selection_reasons", results[0].artifact_payloads["shape-focus.json"])
        self.assertIn("runner_up_candidate_id", results[0].artifact_payloads["raw-timing-results.json"]["rows"][0])
        self.assertIn(
            "baseline_loops_per_sample",
            results[0].artifact_payloads["raw-timing-results.json"]["rows"][0],
        )
        self.assertIn(
            "loops_per_sample",
            results[0].artifact_payloads["raw-timing-results.json"]["rows"][0]["candidate_results"][0],
        )
        self.assertIn(
            "workload_tag",
            results[0].artifact_payloads["raw-timing-results.json"]["rows"][0],
        )
        self.assertIn(
            "workload_share",
            results[0].artifact_payloads["raw-timing-results.json"]["rows"][0],
        )
        self.assertIn("weakest_workloads", results[0].artifact_payloads["shape-focus.json"])
        self.assertIn("frontier-archive.json", results[0].artifact_refs)
        self.assertTrue(
            results[0].artifact_payloads["best-candidate-summary.json"]["best_overall_candidate_id"]
        )

    def test_build_pipeline_uses_live_matmul_executor(self) -> None:
        with unittest.mock.patch(
            "research_engine.cli.JsonFileRunStore.summarize_history",
            return_value={"best_matmul_candidate_id": "transpose_dot"},
        ):
            pipeline = build_pipeline(
                use_llm=False,
                max_results=1,
                live_execution=True,
                benchmark_id="matmul-speedup",
            )

        self.assertIsInstance(pipeline.experiment_executor, DefaultExperimentExecutor)
        self.assertIsInstance(
            pipeline.experiment_executor.matmul_executor,
            MatmulPythonExecutor,
        )
        self.assertEqual(
            pipeline.experiment_executor.matmul_executor.history_summary["best_matmul_candidate_id"],
            "transpose_dot",
        )
        self.assertIsNone(pipeline.experiment_executor.matmul_executor.proposer)

    def test_matmul_executor_uses_history_to_seed_generated_candidates(self) -> None:
        executor = MatmulPythonExecutor(
            repetitions=1,
            history_summary={
                "best_matmul_candidate_id": "transpose_dot",
                "matmul_candidate_wins": {"transpose_dot": 3},
                "weakest_matmul_shapes": [
                    {
                        "shape": "96x96x96",
                        "runner_up_candidate_id": "transpose_unroll8",
                        "runner_up_gap_pct": 5.0,
                    }
                ],
                "matmul_shape_challengers": {
                    "64x64x64": {
                        "runner_up_counts": {"transpose_unroll8": 2},
                        "latest_runner_up": "transpose_unroll8",
                        "latest_runner_up_gap_pct": 6.0,
                    }
                },
            },
        )

        selected, _, shape_focus, proposal = executor._select_candidates()
        selected_ids = {candidate["id"] for candidate in selected}

        self.assertIn("transpose_unroll8", selected_ids)
        self.assertIn("transpose_rowpair", selected_ids)
        self.assertNotIn("transpose_unroll16", selected_ids)
        self.assertEqual(shape_focus["weakest_shapes"][0]["shape"], "96x96x96")
        self.assertEqual(proposal["source"], "none")

    def test_matmul_executor_includes_llm_proposals_when_available(self) -> None:
        fake_client = _FakeResponsesClient()

        def fake_proposal(**kwargs):
            return ResponsesJsonResult(
                data={
                    "candidate_ids": ["transpose_unroll8", "transpose_unroll4"],
                    "global_rationale": "Best challenger on weak shapes.",
                },
                raw_response={"usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            )

        fake_client.generate_json_result = fake_proposal
        executor = MatmulPythonExecutor(
            repetitions=1,
            history_summary={"best_matmul_candidate_id": "transpose_dot"},
            proposer=fake_client,
        )

        selected, _, _, proposal = executor._select_candidates()

        self.assertEqual(proposal["source"], "responses_api")
        self.assertIn("transpose_unroll8", proposal["candidate_ids"])
        self.assertIn("Best challenger", proposal["global_rationale"])
        self.assertTrue(any(candidate["proposal_bonus"] for candidate in selected))


if __name__ == "__main__":
    unittest.main()
