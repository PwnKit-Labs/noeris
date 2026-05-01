from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_triton_matmul_autotune_h100_v2 import _summarize, _to_md  # noqa: E402


class Fp8TritonMatmulAutotuneV2Tests(unittest.TestCase):
    def test_summarize_prefers_faster_config(self) -> None:
        config_results = [
            {
                "config_id": "cfg_1",
                "results": [
                    {"shape_name": "fp8_mm_4096x4096x4096", "correct": True, "tflops": 150.0, "ms": 0.9, "max_err": 0.125}
                ],
            },
            {
                "config_id": "cfg_2",
                "results": [
                    {"shape_name": "fp8_mm_4096x4096x4096", "correct": True, "tflops": 170.0, "ms": 0.8, "max_err": 0.125}
                ],
            },
        ]
        fp16_baseline = [
            {"shape_name": "fp8_mm_4096x4096x4096", "correct": True, "tflops": 220.0, "ms": 0.62}
        ]

        best = _summarize(config_results, fp16_baseline)
        self.assertEqual(best["fp8_mm_4096x4096x4096"]["config_id"], "cfg_2")
        self.assertAlmostEqual(best["fp8_mm_4096x4096x4096"]["fp8_vs_fp16_speedup"], 170.0 / 220.0)

    def test_markdown_mentions_v2_lane(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-21T00:00:00+00:00",
            "best_by_shape": {
                "fp8_mm_4096x4096x4096": {
                    "config_id": "cfg_2",
                    "tflops": 170.0,
                    "fp16_tflops": 220.0,
                    "fp8_vs_fp16_speedup": 170.0 / 220.0,
                    "ms": 0.8,
                    "max_err": 0.125,
                }
            },
        }
        md = _to_md(payload)
        self.assertIn("H100, v2", md)
        self.assertIn("FP8/FP16", md)
        self.assertIn("grouped launch ordering", md)


if __name__ == "__main__":
    unittest.main()
