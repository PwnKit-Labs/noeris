from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fp8_layout_reuse_policy_h100 import _best_ms_by_shape, _policy_rows, _to_md  # noqa: E402


class Fp8LayoutReusePolicyTests(unittest.TestCase):
    def test_best_ms_filters_layout(self) -> None:
        config_results = [
            {"config": {"layout": "kn"}, "results": [{"shape_name": "s", "correct": True, "ms": 0.08}]},
            {"config": {"layout": "kn"}, "results": [{"shape_name": "s", "correct": True, "ms": 0.05}]},
            {"config": {"layout": "nk"}, "results": [{"shape_name": "s", "correct": True, "ms": 0.03}]},
        ]
        self.assertEqual(_best_ms_by_shape(config_results, "kn")["s"], 0.05)
        self.assertEqual(_best_ms_by_shape(config_results, "nk")["s"], 0.03)

    def test_policy_rows_winner_changes_with_reuse(self) -> None:
        v3_payload = {
            "config_results": [
                {"config": {"layout": "kn"}, "results": [{"shape_name": "s", "correct": True, "ms": 0.09}]},
                {"config": {"layout": "nk"}, "results": [{"shape_name": "s", "correct": True, "ms": 0.05}]},
            ]
        }
        prepack_payload = {"prepack_results": [{"shape_name": "s", "prepack_ms": 0.06}]}
        rows = _policy_rows(v3_payload, prepack_payload, [1, 2])
        self.assertEqual(rows[0]["decisions"][0]["winner"], "kn")
        self.assertEqual(rows[0]["decisions"][1]["winner"], "nk")

    def test_markdown_contains_reuse_columns(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "reuse_counts": [1, 2],
            "policy": [
                {
                    "shape_name": "s",
                    "kn_ms": 0.09,
                    "nk_ms": 0.05,
                    "prepack_ms": 0.06,
                    "decisions": [
                        {"reuse_count": 1, "kn_total_ms": 0.09, "nk_total_ms": 0.11, "winner": "kn"},
                        {"reuse_count": 2, "kn_total_ms": 0.18, "nk_total_ms": 0.16, "winner": "nk"},
                    ],
                }
            ],
        }
        md = _to_md(payload)
        self.assertIn("reuse=1", md)
        self.assertIn("reuse=2", md)
        self.assertIn("| s | kn | nk |", md)


if __name__ == "__main__":
    unittest.main()
