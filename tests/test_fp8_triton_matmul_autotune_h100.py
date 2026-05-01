from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_triton_matmul_autotune_h100 import _summarize, _to_md  # noqa: E402


class Fp8TritonMatmulAutotuneTests(unittest.TestCase):
    def test_summarize_includes_fp16_speedup(self) -> None:
        config_results = [
            {
                "config_id": "cfg_slow",
                "results": [
                    {"shape_name": "fp8_mm_1024", "correct": True, "tflops": 40.0, "ms": 0.05, "max_err": 0.125}
                ],
            },
            {
                "config_id": "cfg_fast",
                "results": [
                    {"shape_name": "fp8_mm_1024", "correct": True, "tflops": 50.0, "ms": 0.04, "max_err": 0.125}
                ],
            },
        ]
        fp16_baseline = [
            {"shape_name": "fp8_mm_1024", "correct": True, "tflops": 25.0, "ms": 0.08}
        ]

        best = _summarize(config_results, fp16_baseline)
        self.assertIn("fp8_mm_1024", best)
        self.assertEqual(best["fp8_mm_1024"]["config_id"], "cfg_fast")
        self.assertAlmostEqual(best["fp8_mm_1024"]["fp16_tflops"], 25.0)
        self.assertAlmostEqual(best["fp8_mm_1024"]["fp8_vs_fp16_speedup"], 2.0)

    def test_markdown_includes_fp16_columns(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-21T00:00:00+00:00",
            "best_by_shape": {
                "fp8_mm_1024": {
                    "config_id": "cfg_fast",
                    "tflops": 50.0,
                    "fp16_tflops": 25.0,
                    "fp8_vs_fp16_speedup": 2.0,
                    "ms": 0.04,
                    "max_err": 0.125,
                }
            },
        }

        md = _to_md(payload)
        self.assertIn("best FP8 TFLOPS", md)
        self.assertIn("FP16 TFLOPS", md)
        self.assertIn("FP8/FP16", md)
        self.assertIn("2.000x", md)


if __name__ == "__main__":
    unittest.main()
