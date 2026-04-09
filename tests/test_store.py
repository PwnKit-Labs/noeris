from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.models import ResearchTopic
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
            newer.memo.source_assessments[0].confidence = "high"
            store.save(newer)

            summary = store.summarize_history(benchmark_id="tool-use-reliability")

        self.assertEqual(summary["run_count"], 2)
        self.assertIn("Newer claim", summary["new_claim_titles"])
        self.assertIn("Older claim", summary["dropped_claim_titles"])
        self.assertEqual(summary["confidence_changes"][0]["latest_confidence"], "high")


if __name__ == "__main__":
    unittest.main()
