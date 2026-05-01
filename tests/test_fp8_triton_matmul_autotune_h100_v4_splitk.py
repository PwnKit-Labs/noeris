from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_triton_matmul_autotune_h100_v4_splitk import _summarize, _to_md  # noqa: E402


class Fp8TritonMatmulAutotuneV4SplitKTests(unittest.TestCase):
    def test_summarize_tracks_split_k(self) -> None:
        config_results = [
            {
                "config_id": "cfg_sk1",
                "config": {"SPLIT_K": 1},
                "results": [
                    {"shape_name": "fp8_mm_2048x2048x8192", "correct": True, "tflops": 150.0, "ms": 0.44, "max_err": 0.25}
                ],
            },
            {
                "config_id": "cfg_sk4",
                "config": {"SPLIT_K": 4},
                "results": [
                    {"shape_name": "fp8_mm_2048x2048x8192", "correct": True, "tflops": 190.0, "ms": 0.35, "max_err": 0.25}
                ],
            },
        ]
        fp16 = [{"shape_name": "fp8_mm_2048x2048x8192", "correct": True, "tflops": 300.0, "ms": 0.22}]

        best = _summarize(config_results, fp16)
        self.assertEqual(best["fp8_mm_2048x2048x8192"]["config_id"], "cfg_sk4")
        self.assertEqual(best["fp8_mm_2048x2048x8192"]["split_k"], 4)
        self.assertAlmostEqual(best["fp8_mm_2048x2048x8192"]["fp8_vs_fp16_speedup"], 190.0 / 300.0)

    def test_markdown_mentions_split_k(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "best_by_shape": {
                "fp8_mm_2048x2048x8192": {
                    "split_k": 4,
                    "config_id": "cfg_sk4",
                    "tflops": 190.0,
                    "fp16_tflops": 300.0,
                    "fp8_vs_fp16_speedup": 190.0 / 300.0,
                    "ms": 0.35,
                    "max_err": 0.25,
                }
            },
        }
        md = _to_md(payload)
        self.assertIn("split-K", md)
        self.assertIn("4", md)
        self.assertIn("0.633x", md)


if __name__ == "__main__":
    unittest.main()
