from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_triton_matmul_autotune_h100_v3 import _summarize, _to_md  # noqa: E402


class Fp8TritonMatmulAutotuneV3Tests(unittest.TestCase):
    def test_summarize_carries_layout(self) -> None:
        config_results = [
            {
                "config_id": "kn_cfg",
                "config": {"layout": "kn"},
                "results": [
                    {"shape_name": "fp8_mm_1024", "correct": True, "tflops": 62.0, "ms": 0.034, "max_err": 0.125}
                ],
            },
            {
                "config_id": "nk_cfg",
                "config": {"layout": "nk"},
                "results": [
                    {"shape_name": "fp8_mm_1024", "correct": True, "tflops": 65.0, "ms": 0.032, "max_err": 0.125}
                ],
            },
        ]
        fp16_baseline = [{"shape_name": "fp8_mm_1024", "correct": True, "tflops": 100.0, "ms": 0.02}]

        best = _summarize(config_results, fp16_baseline)
        self.assertEqual(best["fp8_mm_1024"]["config_id"], "nk_cfg")
        self.assertEqual(best["fp8_mm_1024"]["layout"], "nk")
        self.assertAlmostEqual(best["fp8_mm_1024"]["fp8_vs_fp16_speedup"], 0.65)

    def test_markdown_mentions_layout_variants(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "best_by_shape": {
                "fp8_mm_1024": {
                    "layout": "nk",
                    "config_id": "nk_cfg",
                    "tflops": 65.0,
                    "fp16_tflops": 100.0,
                    "fp8_vs_fp16_speedup": 0.65,
                    "ms": 0.032,
                    "max_err": 0.125,
                }
            },
        }
        md = _to_md(payload)
        self.assertIn("layout", md)
        self.assertIn("prepacked NxK", md)
        self.assertIn("0.650x", md)


if __name__ == "__main__":
    unittest.main()
