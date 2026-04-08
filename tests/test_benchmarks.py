from __future__ import annotations

import unittest

from research_engine.benchmarks import get_benchmark


class BenchmarkRegistryTests(unittest.TestCase):
    def test_matmul_benchmark_has_artifact_requirements(self) -> None:
        benchmark = get_benchmark("matmul-speedup")
        self.assertEqual(benchmark.ci_lane, "manual-expensive")
        self.assertIn("hardware-profile.json", benchmark.required_artifacts)
        self.assertIn("baseline kernel", benchmark.baseline_guidance)

    def test_long_context_benchmark_is_scheduled(self) -> None:
        benchmark = get_benchmark("long-context-reasoning")
        self.assertEqual(benchmark.ci_lane, "scheduled-benchmark")
        self.assertIn("eval-manifest.json", benchmark.required_artifacts)

    def test_tool_use_benchmark_tracks_terminal_first_comparison(self) -> None:
        benchmark = get_benchmark("tool-use-reliability")
        self.assertIn("terminal-first", benchmark.baseline_guidance)
        self.assertIn("terminal-transcript.jsonl", benchmark.required_artifacts)


if __name__ == "__main__":
    unittest.main()
