from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.fp8_layout_policy import choose_layout_from_policy


class Fp8LayoutPolicyRuntimeTests(unittest.TestCase):
    def test_exact_reuse_match(self) -> None:
        payload = {
            "policy": [
                {
                    "shape_name": "s",
                    "decisions": [
                        {"reuse_count": 1, "winner": "kn", "kn_total_ms": 1.0, "nk_total_ms": 2.0},
                        {"reuse_count": 2, "winner": "nk", "kn_total_ms": 2.0, "nk_total_ms": 1.5},
                    ],
                }
            ]
        }
        d = choose_layout_from_policy(payload, "s", 2)
        self.assertEqual(d.winner, "nk")
        self.assertEqual(d.reuse_count, 2)

    def test_fallback_to_nearest_lower_reuse(self) -> None:
        payload = {
            "policy": [
                {
                    "shape_name": "s",
                    "decisions": [
                        {"reuse_count": 1, "winner": "kn", "kn_total_ms": 1.0, "nk_total_ms": 2.0},
                        {"reuse_count": 4, "winner": "nk", "kn_total_ms": 4.0, "nk_total_ms": 3.0},
                    ],
                }
            ]
        }
        d = choose_layout_from_policy(payload, "s", 3)
        self.assertEqual(d.reuse_count, 1)
        self.assertEqual(d.winner, "kn")

    def test_unknown_shape_raises(self) -> None:
        payload = {"policy": []}
        with self.assertRaises(KeyError):
            choose_layout_from_policy(payload, "missing", 1)


if __name__ == "__main__":
    unittest.main()
