from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_layout_runtime_integration_token_loop_benchmark import _to_md  # noqa: E402


class Fp8LayoutRuntimeIntegrationTokenLoopBenchmarkTests(unittest.TestCase):
    def test_markdown_contains_mode_comparison_rows(self) -> None:
        payload = {
            "generated_at_utc": "2026-05-01T00:00:00+00:00",
            "mode_results": [
                {
                    "mode": "auto_no_cache",
                    "prepack_cache_enabled": False,
                    "runtime_dispatch_total_ms": 1.2,
                    "runtime_dispatch_avg_ms_per_token": 0.02,
                    "runtime_prepack_ops": 48,
                    "runtime_cache_hits": 0,
                    "runtime_cache_misses": 0,
                    "vs_baseline_dispatch_total": 1.0,
                },
                {
                    "mode": "auto_with_cache",
                    "prepack_cache_enabled": True,
                    "runtime_dispatch_total_ms": 0.7,
                    "runtime_dispatch_avg_ms_per_token": 0.0117,
                    "runtime_prepack_ops": 48,
                    "runtime_cache_hits": 42,
                    "runtime_cache_misses": 6,
                    "vs_baseline_dispatch_total": 0.5833,
                },
            ],
        }
        md = _to_md(payload)
        self.assertIn("auto_no_cache", md)
        self.assertIn("auto_with_cache", md)
        self.assertIn("cache hits", md)


if __name__ == "__main__":
    unittest.main()
