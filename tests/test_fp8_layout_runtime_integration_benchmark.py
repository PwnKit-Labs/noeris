from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_layout_runtime_integration_benchmark import _to_md  # noqa: E402


class Fp8LayoutRuntimeIntegrationBenchmarkTests(unittest.TestCase):
    def test_markdown_contains_auto_vs_best(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "results": {
                "H100": {
                    "config_results": [
                        {
                            "shape_name": "fp8_mm_1024",
                            "expected_reuse": 2,
                            "kn_ms": 0.03,
                            "nk_ms": 0.02,
                            "policy_kn_total_ms": 0.06,
                            "policy_nk_total_ms": 0.05,
                            "auto_effective_ms": 0.05,
                            "auto_layout": "nk",
                            "kernel_best_layout": "nk",
                            "policy_best_layout": "nk",
                            "auto_vs_kernel_best_ratio": 1.0,
                            "auto_vs_policy_best_ratio": 1.0,
                        }
                    ]
                }
            },
        }
        md = _to_md(payload)
        self.assertIn("auto/policy", md)
        self.assertIn("fp8_mm_1024", md)
        self.assertIn("| nk | nk | nk | 1.0000 |", md)


if __name__ == "__main__":
    unittest.main()
