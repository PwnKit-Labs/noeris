from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine._legacy.agenda import DEFAULT_RESEARCH_AGENDA


class ResearchAgendaTests(unittest.TestCase):
    def test_long_context_is_high_priority(self) -> None:
        item = next(entry for entry in DEFAULT_RESEARCH_AGENDA if entry.area_id == "long-context")
        self.assertEqual(item.priority, "high")
        self.assertEqual(item.recommended_mode, "post-training+evals")

    def test_pretraining_is_not_a_v1_focus(self) -> None:
        item = next(
            entry for entry in DEFAULT_RESEARCH_AGENDA if entry.area_id == "base-model-scaling"
        )
        self.assertEqual(item.priority, "low")
        self.assertEqual(item.benchmark_fit, "weak")


if __name__ == "__main__":
    unittest.main()
