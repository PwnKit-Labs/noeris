from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_prepack_amortization_h100 import _best_ms_by_shape, _merge_amortization, _to_md  # noqa: E402


class Fp8PrepackAmortizationTests(unittest.TestCase):
    def test_best_ms_by_shape_filters_layout(self) -> None:
        config_results = [
            {
                "config": {"layout": "kn"},
                "results": [{"shape_name": "s1", "correct": True, "ms": 0.07}],
            },
            {
                "config": {"layout": "kn"},
                "results": [{"shape_name": "s1", "correct": True, "ms": 0.06}],
            },
            {
                "config": {"layout": "nk"},
                "results": [{"shape_name": "s1", "correct": True, "ms": 0.04}],
            },
        ]
        kn = _best_ms_by_shape(config_results, layout="kn")
        nk = _best_ms_by_shape(config_results, layout="nk")
        self.assertAlmostEqual(kn["s1"], 0.06)
        self.assertAlmostEqual(nk["s1"], 0.04)

    def test_merge_amortization_computes_break_even(self) -> None:
        v3_payload = {
            "config_results": [
                {"config": {"layout": "kn"}, "results": [{"shape_name": "s1", "correct": True, "ms": 0.07}]},
                {"config": {"layout": "nk"}, "results": [{"shape_name": "s1", "correct": True, "ms": 0.04}]},
            ]
        }
        prepack_rows = [{"shape_name": "s1", "prepack_ms": 0.15}]
        rows = _merge_amortization(v3_payload, prepack_rows)
        self.assertEqual(rows[0]["break_even_runs"], 5)

    def test_markdown_contains_break_even_table(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "amortization": [
                {
                    "shape_name": "s1",
                    "kn_ms": 0.07,
                    "nk_ms": 0.04,
                    "delta_ms_per_run": 0.03,
                    "prepack_ms": 0.15,
                    "break_even_runs": 5,
                }
            ],
        }
        md = _to_md(payload)
        self.assertIn("break-even runs", md)
        self.assertIn("| s1 |", md)
        self.assertIn("| 5 |", md)


if __name__ == "__main__":
    unittest.main()
