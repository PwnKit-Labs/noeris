from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.models import Contradiction, ResearchTopic
from research_engine.pipeline import ResearchPipeline
from research_engine.store import JsonFileRunStore


class RunStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        pipeline = ResearchPipeline()
        record = pipeline.run_record(
            ResearchTopic(
                name="long-context reasoning",
                objective="improve benchmark performance",
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            path = store.save(record)
            loaded = store.load(record.run_id)

        self.assertTrue(path.name.endswith(".json"))
        self.assertEqual(loaded.run_id, record.run_id)
        self.assertEqual(loaded.cycle.topic.name, "long-context reasoning")
        self.assertEqual(loaded.memo.topic, "long-context reasoning")

    def test_list_runs_returns_summary(self) -> None:
        pipeline = ResearchPipeline()
        record = pipeline.run_record(
            ResearchTopic(
                name="tool use",
                objective="improve success rate",
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            store.save(record)
            summaries = store.list_runs()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["run_id"], record.run_id)
        self.assertEqual(summaries[0]["topic"], "tool use")

    def test_summarize_history_reports_new_and_dropped_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            older = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="tool-use reliability",
                    objective="improve task success",
                    benchmark_id="tool-use-reliability",
                    constraints=["benchmark_id:tool-use-reliability"],
                ),
                benchmark_id="tool-use-reliability",
            )
            older.memo.claims[0].title = "Older claim"
            older.memo.source_assessments[0].confidence = "low"
            store.save(older)

            newer = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="tool-use reliability",
                    objective="improve task success",
                    benchmark_id="tool-use-reliability",
                    constraints=["benchmark_id:tool-use-reliability"],
                ),
                benchmark_id="tool-use-reliability",
            )
            newer.memo.claims[0].title = "Newer claim"
            newer.memo.claims[0].evidence_kind = "llm-extracted"
            newer.memo.source_assessments[0].confidence = "high"
            store.save(newer)

            summary = store.summarize_history(benchmark_id="tool-use-reliability")

        self.assertEqual(summary["run_count"], 2)
        self.assertIn("Newer claim", summary["new_claim_titles"])
        self.assertIn("Older claim", summary["dropped_claim_titles"])
        self.assertEqual(summary["confidence_changes"][0]["latest_confidence"], "high")
        self.assertEqual(summary["shared_source_ids"], ["seed://bootstrap"])

    def test_summarize_history_reports_evidence_kind_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            older = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="tool-use reliability",
                    objective="improve task success",
                    benchmark_id="tool-use-reliability",
                    constraints=["benchmark_id:tool-use-reliability"],
                ),
                benchmark_id="tool-use-reliability",
            )
            older.memo.claims[0].title = "Shared claim"
            older.memo.claims[0].evidence_kind = "source-derived"
            store.save(older)

            newer = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="tool-use reliability",
                    objective="improve task success",
                    benchmark_id="tool-use-reliability",
                    constraints=["benchmark_id:tool-use-reliability"],
                ),
                benchmark_id="tool-use-reliability",
            )
            newer.memo.claims[0].title = "Shared claim"
            newer.memo.claims[0].evidence_kind = "llm-extracted"
            store.save(newer)

            summary = store.summarize_history(benchmark_id="tool-use-reliability")

        self.assertEqual(
            summary["evidence_kind_changes"][0]["latest_evidence_kind"],
            "llm-extracted",
        )

    def test_summarize_history_reports_source_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            record = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="long-context reasoning",
                    objective="improve benchmark performance",
                    benchmark_id="long-context-reasoning",
                    constraints=["benchmark_id:long-context-reasoning"],
                ),
                benchmark_id="long-context-reasoning",
            )
            record.memo.sources[0].updated_at = "2026-04-10T12:00:00Z"
            store.save(record)

            summary = store.summarize_history(benchmark_id="long-context-reasoning")
            brief = store.render_history_brief(benchmark_id="long-context-reasoning")

        self.assertEqual(summary["source_freshness"]["source_count_with_timestamps"], 1)
        self.assertEqual(summary["source_freshness"]["newest_source_id"], "seed://bootstrap")
        self.assertEqual(summary["source_freshness"]["bucket_counts"]["fresh"], 1)
        self.assertEqual(summary["ranked_sources"][0]["source_id"], "seed://bootstrap")
        self.assertEqual(summary["ranked_sources"][0]["staleness_label"], "fresh")
        self.assertIn("## Source Freshness", brief)
        self.assertIn("## Ranked Sources", brief)
        self.assertIn("Buckets", brief)
        self.assertIn("2026-04-10T12:00:00Z", brief)

    def test_ranked_sources_penalize_contradicted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            record = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="long-context reasoning",
                    objective="improve benchmark performance",
                    benchmark_id="long-context-reasoning",
                    constraints=["benchmark_id:long-context-reasoning"],
                ),
                benchmark_id="long-context-reasoning",
            )
            record.memo.sources = [
                record.memo.sources[0],
                type(record.memo.sources[0])(
                    identifier="source-2",
                    kind="paper",
                    title="Fresh supporting paper",
                    locator="https://example.com/source-2",
                    excerpt="supporting evidence",
                    updated_at="2026-04-11T12:00:00Z",
                ),
            ]
            record.memo.source_assessments = [
                type(record.memo.source_assessments[0])(
                    source_id="seed://bootstrap",
                    confidence="high",
                    rationale="good but disputed",
                ),
                type(record.memo.source_assessments[0])(
                    source_id="source-2",
                    confidence="high",
                    rationale="good and clean",
                ),
            ]
            record.memo.claims = [
                type(record.memo.claims[0])(
                    title="Disputed claim",
                    source="seed://bootstrap",
                    summary="summary",
                    evidence_refs=["seed://bootstrap"],
                    evidence_kind="source-derived",
                ),
                type(record.memo.claims[0])(
                    title="Clean claim",
                    source="source-2",
                    summary="summary",
                    evidence_refs=["source-2"],
                    evidence_kind="source-derived",
                ),
            ]
            record.memo.contradictions = [
                Contradiction(
                    title="Disagreement",
                    summary="disputed",
                    claim_titles=["Disputed claim"],
                    severity="medium",
                )
            ]
            store.save(record)

            summary = store.summarize_history(benchmark_id="long-context-reasoning")

        self.assertEqual(summary["ranked_sources"][0]["source_id"], "source-2")
        self.assertEqual(summary["ranked_sources"][1]["source_id"], "seed://bootstrap")
        self.assertEqual(summary["ranked_sources"][1]["contradiction_count"], 1)

    def test_summarize_history_reports_contradiction_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            older = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="long-context reasoning",
                    objective="improve benchmark performance",
                    benchmark_id="long-context-reasoning",
                    constraints=["benchmark_id:long-context-reasoning"],
                ),
                benchmark_id="long-context-reasoning",
            )
            older.memo.contradictions = []
            store.save(older)

            newer = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="long-context reasoning",
                    objective="improve benchmark performance",
                    benchmark_id="long-context-reasoning",
                    constraints=["benchmark_id:long-context-reasoning"],
                ),
                benchmark_id="long-context-reasoning",
            )
            newer.memo.contradictions = [
                Contradiction(
                    title="Context disagreement",
                    summary="Two sources imply different retention behavior.",
                    claim_titles=[newer.memo.claims[0].title],
                    severity="medium",
                )
            ]
            store.save(newer)

            summary = store.summarize_history(benchmark_id="long-context-reasoning")
            brief = store.render_history_brief(benchmark_id="long-context-reasoning")

        self.assertEqual(summary["new_contradiction_titles"], ["Context disagreement"])
        self.assertEqual(summary["dropped_contradiction_titles"], [])
        self.assertIn("## Contradiction Changes", brief)
        self.assertIn("Context disagreement", brief)

    def test_summarize_history_reports_matmul_candidate_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))
            record = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="matrix multiplication speedup",
                    objective="discover validated kernel-level speedups",
                    benchmark_id="matmul-speedup",
                    constraints=["benchmark_id:matmul-speedup"],
                ),
                benchmark_id="matmul-speedup",
            )
            record.memo.results[0].artifact_payloads["best-candidate-summary.json"] = {
                "winner_counts": {"transpose_dot": 5, "ikj_accumulate": 0},
                "winner_share_scores": {"transpose_dot": 0.7, "ikj_accumulate": 0.3},
                "best_overall_candidate_id": "transpose_dot",
            }
            record.memo.results[0].artifact_payloads["frontier-archive.json"] = {
                "workload_winners": [
                    {
                        "workload_tag": "mlp_up_proj",
                        "workload_share": 0.28,
                        "best_candidate_id": "transpose_dot",
                        "runner_up_candidate_id": "ikj_accumulate",
                        "runner_up_gap_pct": 3.2,
                    }
                ]
            }
            record.memo.results[0].artifact_payloads["raw-timing-results.json"] = {
                "rows": [
                    {
                        "shape": "32x32x32",
                        "workload_tag": "mlp_up_proj",
                        "workload_share": 0.28,
                        "best_candidate_id": "transpose_dot",
                        "uplift_pct": 12.3,
                        "runner_up_candidate_id": "ikj_accumulate",
                        "runner_up_gap_pct": 3.2,
                        "candidate_results": [
                            {
                                "candidate_id": "transpose_dot",
                                "candidate_family": "layout_transform",
                            },
                            {
                                "candidate_id": "ikj_accumulate",
                                "candidate_family": "loop_reordering",
                            },
                        ],
                    }
                ]
            }
            store.save(record)

            summary = store.summarize_history(benchmark_id="matmul-speedup")

        self.assertEqual(summary["best_matmul_candidate_id"], "transpose_dot")
        self.assertEqual(summary["matmul_candidate_wins"]["transpose_dot"], 5)
        self.assertEqual(summary["matmul_family_wins"]["layout_transform"], 1)
        self.assertIn("32x32x32", summary["matmul_shape_winners"])
        self.assertEqual(summary["weakest_matmul_shapes"][0]["shape"], "32x32x32")
        self.assertEqual(summary["weakest_matmul_workloads"][0]["workload_tag"], "mlp_up_proj")
        self.assertEqual(summary["matmul_frontier_archive"][0]["workload_tag"], "mlp_up_proj")

    def test_summarize_history_keeps_latest_shape_winner_from_newest_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonFileRunStore(Path(temp_dir))

            older = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="matrix multiplication speedup",
                    objective="discover validated kernel-level speedups",
                    benchmark_id="matmul-speedup",
                    constraints=["benchmark_id:matmul-speedup"],
                ),
                benchmark_id="matmul-speedup",
            )
            older.created_at = "2026-04-10T09:00:00Z"
            older.memo.results[0].artifact_payloads["best-candidate-summary.json"] = {
                "winner_counts": {"transpose_dot": 1},
                "best_overall_candidate_id": "transpose_dot",
            }
            older.memo.results[0].artifact_payloads["raw-timing-results.json"] = {
                "rows": [
                    {
                        "shape": "96x96x96",
                        "workload_tag": "square_control",
                        "workload_share": 0.14,
                        "best_candidate_id": "transpose_dot",
                        "uplift_pct": 24.5,
                        "runner_up_candidate_id": "transpose_unroll8",
                        "runner_up_gap_pct": 5.9,
                        "candidate_results": [
                            {
                                "candidate_id": "transpose_dot",
                                "candidate_family": "layout_transform",
                            },
                            {
                                "candidate_id": "transpose_unroll8",
                                "candidate_family": "unrolling",
                            },
                        ],
                    }
                ]
            }
            store.save(older)

            newer = ResearchPipeline().run_record_for(
                topic=ResearchTopic(
                    name="matrix multiplication speedup",
                    objective="discover validated kernel-level speedups",
                    benchmark_id="matmul-speedup",
                    constraints=["benchmark_id:matmul-speedup"],
                ),
                benchmark_id="matmul-speedup",
            )
            newer.created_at = "2026-04-10T10:00:00Z"
            newer.memo.results[0].artifact_payloads["best-candidate-summary.json"] = {
                "winner_counts": {"transpose_rowpair": 1},
                "best_overall_candidate_id": "transpose_rowpair",
            }
            newer.memo.results[0].artifact_payloads["raw-timing-results.json"] = {
                "rows": [
                    {
                        "shape": "96x96x96",
                        "workload_tag": "square_control",
                        "workload_share": 0.14,
                        "best_candidate_id": "transpose_rowpair",
                        "uplift_pct": 26.2,
                        "runner_up_candidate_id": "transpose_dot",
                        "runner_up_gap_pct": 2.6,
                        "candidate_results": [
                            {
                                "candidate_id": "transpose_rowpair",
                                "candidate_family": "row_pairing",
                            },
                            {
                                "candidate_id": "transpose_dot",
                                "candidate_family": "layout_transform",
                            },
                        ],
                    }
                ]
            }
            store.save(newer)

            summary = store.summarize_history(benchmark_id="matmul-speedup")

        self.assertEqual(
            summary["matmul_shape_winners"]["96x96x96"]["latest_winner"],
            "transpose_rowpair",
        )
        self.assertEqual(
            summary["matmul_shape_challengers"]["96x96x96"]["latest_runner_up"],
            "transpose_dot",
        )


if __name__ == "__main__":
    unittest.main()
