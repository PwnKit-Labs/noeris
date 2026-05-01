from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.fp8_runtime import resolve_fp8_layout


class Fp8RuntimeLayoutResolveTests(unittest.TestCase):
    def test_force_modes(self) -> None:
        self.assertEqual(resolve_fp8_layout(prefer="kn", shape_name="s", expected_reuse_count=1), "kn")
        self.assertEqual(resolve_fp8_layout(prefer="nk", shape_name="s", expected_reuse_count=1), "nk")

    def test_auto_uses_policy_when_available(self) -> None:
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
        self.assertEqual(resolve_fp8_layout(prefer="auto", shape_name="s", expected_reuse_count=2, policy_payload=payload), "nk")

    def test_auto_fallback_heuristic_without_policy(self) -> None:
        self.assertEqual(resolve_fp8_layout(prefer="auto", shape_name="s", expected_reuse_count=1), "kn")
        self.assertEqual(resolve_fp8_layout(prefer="auto", shape_name="s", expected_reuse_count=2), "nk")

    def test_auto_fallback_when_policy_missing_shape(self) -> None:
        payload = {"policy": [{"shape_name": "other", "decisions": []}]}
        self.assertEqual(
            resolve_fp8_layout(
                prefer="auto",
                shape_name="missing",
                expected_reuse_count=3,
                policy_payload=payload,
            ),
            "nk",
        )

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_fp8_layout(prefer="bad", shape_name="s", expected_reuse_count=1)


if __name__ == "__main__":
    unittest.main()
