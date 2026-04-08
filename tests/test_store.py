from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
