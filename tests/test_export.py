from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

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
            summary = (bundle_dir / "summary.md").read_text(encoding="utf-8")
            lineage = (bundle_dir / "claim-lineage.json").read_text(encoding="utf-8")
            brief = (bundle_dir / "research-brief.md").read_text(encoding="utf-8")

        self.assertEqual(
            files,
            ["claim-lineage.json", "memo.json", "research-brief.md", "run.json", "summary.md", "verification.json"],
        )
        self.assertIn("Source Confidence", summary)
        self.assertIn("claims", lineage)
        self.assertIn("linked_sources", lineage)
        self.assertIn("assessment", lineage)
        self.assertIn("updated_at", lineage)
        self.assertIn("## Claims", brief)
        self.assertIn("## Hypotheses", brief)

    def test_long_context_export_writes_required_artifacts(self) -> None:
        record = ResearchPipeline().run_record_for(
            topic=ResearchTopic(
                name="long-context reasoning",
                objective="improve long-context eval quality",
                benchmark_id="long-context-reasoning",
                constraints=["benchmark_id:long-context-reasoning"],
            ),
            benchmark_id="long-context-reasoning",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = export_run_bundle(record, Path(temp_dir))
            files = sorted(path.name for path in bundle_dir.iterdir())

        self.assertIn("eval-manifest.json", files)
        self.assertIn("baseline-metrics.json", files)
        self.assertIn("candidate-metrics.json", files)
        self.assertIn("failure-analysis.md", files)

    def test_tool_use_export_writes_required_artifacts(self) -> None:
        record = ResearchPipeline().run_record_for(
            topic=ResearchTopic(
                name="tool-use reliability",
                objective="improve task success",
                benchmark_id="tool-use-reliability",
                constraints=["benchmark_id:tool-use-reliability"],
            ),
            benchmark_id="tool-use-reliability",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = export_run_bundle(record, Path(temp_dir))
            files = sorted(path.name for path in bundle_dir.iterdir())

        self.assertIn("task-suite.json", files)
        self.assertIn("terminal-transcript.jsonl", files)
        self.assertIn("tool-selection-summary.json", files)
        self.assertIn("success-summary.json", files)
        self.assertIn("error-taxonomy.md", files)

    def test_matmul_export_writes_required_artifacts(self) -> None:
        record = ResearchPipeline().run_record_for(
            topic=ResearchTopic(
                name="matrix multiplication speedup",
                objective="discover validated kernel-level speedups",
                benchmark_id="matmul-speedup",
                constraints=["benchmark_id:matmul-speedup"],
            ),
            benchmark_id="matmul-speedup",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = export_run_bundle(record, Path(temp_dir))
            files = sorted(path.name for path in bundle_dir.iterdir())

        self.assertIn("hardware-profile.json", files)
        self.assertIn("benchmark-config.json", files)
        self.assertIn("raw-timing-results.json", files)
        self.assertIn("baseline-comparison.md", files)


if __name__ == "__main__":
    unittest.main()
