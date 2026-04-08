from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_engine.export import export_run_bundle
from research_engine.models import ResearchTopic
from research_engine.pipeline import ResearchPipeline


class ExportBundleTests(unittest.TestCase):
    def test_export_run_bundle_writes_expected_files(self) -> None:
        record = ResearchPipeline().run_record_for(
            topic=ResearchTopic(
                name="tool use reliability",
                objective="improve task success",
            ),
            benchmark_id="tool-use-reliability",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = export_run_bundle(record, Path(temp_dir))
            files = sorted(path.name for path in bundle_dir.iterdir())

        self.assertEqual(
            files,
            ["memo.json", "run.json", "summary.md", "verification.json"],
        )


if __name__ == "__main__":
    unittest.main()
