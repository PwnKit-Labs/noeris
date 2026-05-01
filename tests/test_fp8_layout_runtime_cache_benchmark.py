from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_layout_runtime_cache_benchmark import _to_md  # noqa: E402


class Fp8LayoutRuntimeCacheBenchmarkTests(unittest.TestCase):
    def test_markdown_table_contains_modes(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "results": {
                "H100": {
                    "config_results": [
                        {
                            "scenario": {"name": "s1024_reuse1_unique"},
                            "results": [
                                {"mode": "force_kn", "avg_ms": 0.05, "vs_best": 1.2, "cache_hits": 0, "cache_misses": 0},
                                {"mode": "auto_policy_cache", "avg_ms": 0.04, "vs_best": 1.0, "cache_hits": 10, "cache_misses": 2},
                            ],
                        }
                    ]
                }
            },
        }
        md = _to_md(payload)
        self.assertIn("s1024_reuse1_unique", md)
        self.assertIn("force_kn", md)
        self.assertIn("auto_policy_cache", md)


if __name__ == "__main__":
    unittest.main()
