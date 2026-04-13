"""Tests for the co-evolving world model.

Coverage:
- Hypothesis creation and field defaults
- Bayesian confidence update (evidence_for / evidence_against)
- Context matching (operator, hardware, shape)
- Config proposal generation ordered by confidence
- Hypothesis update from benchmark results
- Discovery of new hypotheses from synthetic data
- Serialization round-trip (save/load)
- Built-in hypotheses are loaded by default
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine._legacy.world_model import (
    ConfigHypothesis,
    WorldModel,
    _BUILTIN_HYPOTHESES,
    _value_matches,
)


class TestConfigHypothesis(unittest.TestCase):
    """Unit tests for the ConfigHypothesis dataclass."""

    def test_creation_defaults(self):
        h = ConfigHypothesis(
            description="test hypothesis",
            conditions={"operator": "matmul"},
            predicted_effect="faster",
        )
        self.assertEqual(h.evidence_for, 0)
        self.assertEqual(h.evidence_against, 0)
        self.assertAlmostEqual(h.confidence, 0.5)
        self.assertEqual(h.source, "manual")

    def test_update_increases_confidence(self):
        h = ConfigHypothesis(
            description="test",
            conditions={},
            predicted_effect="faster",
        )
        # Start at 0.5
        self.assertAlmostEqual(h.confidence, 0.5)
        # Three positive updates: (0+1)/(0+0+2)=0.5 -> (1+1)/(1+0+2)=0.667
        h.update(matched=True)
        self.assertAlmostEqual(h.confidence, 2 / 3)
        h.update(matched=True)
        self.assertAlmostEqual(h.confidence, 3 / 4)
        h.update(matched=True)
        self.assertAlmostEqual(h.confidence, 4 / 5)

    def test_update_decreases_confidence(self):
        h = ConfigHypothesis(
            description="test",
            conditions={},
            predicted_effect="faster",
            evidence_for=10,
            evidence_against=0,
        )
        h.confidence = 11 / 12  # consistent initial state
        h.update(matched=False)
        self.assertEqual(h.evidence_against, 1)
        self.assertAlmostEqual(h.confidence, 11 / 13)

    def test_matches_context_operator(self):
        h = ConfigHypothesis(
            description="test",
            conditions={"operator": "matmul"},
            predicted_effect="faster",
        )
        self.assertTrue(h.matches_context(operator="matmul"))
        self.assertFalse(h.matches_context(operator="attention"))
        # None operator -> still matches (no filtering)
        self.assertTrue(h.matches_context(operator=None))

    def test_matches_context_hardware_list(self):
        h = ConfigHypothesis(
            description="test",
            conditions={"hardware": ["Tesla T4", "A100"]},
            predicted_effect="faster",
        )
        self.assertTrue(h.matches_context(hardware="Tesla T4"))
        self.assertTrue(h.matches_context(hardware="A100"))
        self.assertFalse(h.matches_context(hardware="H100"))

    def test_matches_context_shape(self):
        h = ConfigHypothesis(
            description="test",
            conditions={"head_dim": 512, "operator": "attention"},
            predicted_effect="smaller tiles",
        )
        self.assertTrue(
            h.matches_context(operator="attention", shape={"head_dim": 512})
        )
        self.assertFalse(
            h.matches_context(operator="attention", shape={"head_dim": 128})
        )

    def test_round_trip_dict(self):
        h = ConfigHypothesis(
            description="round trip",
            conditions={"operator": "matmul", "num_warps": [1, 2]},
            predicted_effect="faster",
            evidence_for=5,
            evidence_against=2,
            confidence=0.75,
            source="builtin",
        )
        d = h.to_dict()
        h2 = ConfigHypothesis.from_dict(d)
        self.assertEqual(h.description, h2.description)
        self.assertEqual(h.conditions, h2.conditions)
        self.assertEqual(h.evidence_for, h2.evidence_for)
        self.assertAlmostEqual(h.confidence, h2.confidence)


class TestWorldModel(unittest.TestCase):
    """Integration tests for the WorldModel."""

    def test_builtins_loaded_by_default(self):
        wm = WorldModel()
        self.assertGreaterEqual(len(wm), len(_BUILTIN_HYPOTHESES))

    def test_builtins_excluded(self):
        wm = WorldModel(include_builtins=False)
        self.assertEqual(len(wm), 0)

    def test_propose_configs_returns_matching(self):
        wm = WorldModel(include_builtins=False, hypotheses=[
            ConfigHypothesis(
                description="low warps on T4",
                conditions={"hardware": "Tesla T4", "num_warps": 2},
                predicted_effect="higher throughput",
                confidence=0.9,
            ),
            ConfigHypothesis(
                description="high warps on A100",
                conditions={"hardware": "A100", "num_warps": 8},
                predicted_effect="higher throughput",
                confidence=0.85,
            ),
        ])
        proposals = wm.propose_configs(
            operator="matmul", shape={"M": 1024}, hardware="Tesla T4", n=5
        )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["num_warps"], 2)

    def test_propose_configs_ordered_by_confidence(self):
        wm = WorldModel(include_builtins=False, hypotheses=[
            ConfigHypothesis(
                description="low conf",
                conditions={"hardware": "T4", "BLOCK_SIZE": 64},
                predicted_effect="ok",
                confidence=0.3,
            ),
            ConfigHypothesis(
                description="high conf",
                conditions={"hardware": "T4", "BLOCK_SIZE": 256},
                predicted_effect="great",
                confidence=0.95,
            ),
        ])
        proposals = wm.propose_configs(
            operator="rmsnorm", shape={}, hardware="T4", n=5
        )
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0]["BLOCK_SIZE"], 256)
        self.assertEqual(proposals[1]["BLOCK_SIZE"], 64)

    def test_update_from_result(self):
        h = ConfigHypothesis(
            description="split-k helps deep K",
            conditions={"operator": "matmul", "SPLIT_K": [2, 4]},
            predicted_effect="higher throughput",
            evidence_for=0,
            evidence_against=0,
            confidence=0.5,
        )
        wm = WorldModel(include_builtins=False, hypotheses=[h])
        # A top result that matches the hypothesis conditions.
        wm.update_from_result(
            config={"SPLIT_K": 4, "BLOCK_SIZE_M": 128},
            shape={"M": 64, "N": 64, "K": 4096},
            hardware="A100",
            metric=150.0,
            operator="matmul",
            is_top=True,
        )
        self.assertEqual(h.evidence_for, 1)
        self.assertGreater(h.confidence, 0.5)

    def test_discover_new_hypotheses(self):
        """Synthetic data where num_warps=4 dominates top performers."""
        wm = WorldModel(include_builtins=False)
        records = []
        # 100 records: 20 with num_warps=4 (high throughput), 80 with num_warps=8 (low)
        # Top 10% = 10 records, all num_warps=4 -> rate_top=1.0, rate_full=0.2 -> ratio=5x
        for i in range(20):
            records.append({
                "operator": "rmsnorm",
                "hardware": "A100",
                "config": {"num_warps": 4, "BLOCK_SIZE": 128},
                "throughput": 200.0 + i * 0.1,
            })
        for i in range(80):
            records.append({
                "operator": "rmsnorm",
                "hardware": "A100",
                "config": {"num_warps": 8, "BLOCK_SIZE": 128},
                "throughput": 50.0 + i * 0.1,
            })
        new = wm.discover_new_hypotheses(records, top_fraction=0.10, min_records=5)
        # Should discover that num_warps=4 is over-represented in top configs.
        self.assertGreater(len(new), 0)
        found_warps = any(
            "num_warps" in h.conditions and h.conditions["num_warps"] == 4
            for h in new
        )
        self.assertTrue(found_warps, "Should discover num_warps=4 hypothesis")
        # Check it was appended to the model.
        self.assertGreater(len(wm), 0)

    def test_serialization_round_trip(self):
        wm = WorldModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world_model.json"
            wm.save(path)
            wm2 = WorldModel.load(path)
            self.assertEqual(len(wm), len(wm2))
            for h1, h2 in zip(wm.hypotheses, wm2.hypotheses):
                self.assertEqual(h1.description, h2.description)
                self.assertAlmostEqual(h1.confidence, h2.confidence)

    def test_discover_skips_small_groups(self):
        wm = WorldModel(include_builtins=False)
        records = [
            {"operator": "matmul", "hardware": "T4",
             "config": {"num_warps": 4}, "throughput": 100.0}
        ]
        new = wm.discover_new_hypotheses(records, min_records=10)
        self.assertEqual(len(new), 0)


class TestValueMatches(unittest.TestCase):
    def test_scalar(self):
        self.assertTrue(_value_matches(4, 4))
        self.assertFalse(_value_matches(4, 8))

    def test_list(self):
        self.assertTrue(_value_matches(2, [1, 2, 4]))
        self.assertFalse(_value_matches(3, [1, 2, 4]))


if __name__ == "__main__":
    unittest.main()
