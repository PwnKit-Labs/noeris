"""Tests for src/research_engine/kernelbench_l4.py.

Covers:
  * L4 problem definitions (count, skip BigBird/Reformer)
  * NoerisOpSubstitutor (Linear, LayerNorm, GELU, Conv1D, non-target preservation)
  * Benchmark script generation (compiles, required symbols, pip install,
    allclose check, cuda_event timing)

The module under test is GPU-free at import time.  Torch-dependent tests
(substitutor integration) are skipped when torch is not installed.
"""

from __future__ import annotations

import ast
import unittest

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tests import _pathfix  # noqa: F401

from research_engine.kernelbench_l4 import (
    L4_PROBLEMS,
    L4_SKIPPED_IDS,
    L4Problem,
    NoerisOpSubstitutor,
    generate_l4_benchmark_script,
    get_l4_attack_problems,
    get_l4_problems,
)


# ---------------------------------------------------------------------------
# Problem definition tests
# ---------------------------------------------------------------------------


class TestL4Problems(unittest.TestCase):
    """L4 problem catalogue integrity."""

    def test_l4_problems_defined(self) -> None:
        """At least 15 addressable problems exist."""
        problems = get_l4_problems()
        self.assertGreaterEqual(len(problems), 15)

    def test_l4_problems_skip_bigbird_reformer(self) -> None:
        """No problem name should contain BigBird or Reformer."""
        for p in get_l4_problems():
            low = (p.model_name + p.arch_family + p.filename).lower()
            self.assertNotIn("bigbird", low, f"BigBird found in problem #{p.problem_id}")
            self.assertNotIn("reformer", low, f"Reformer found in problem #{p.problem_id}")

    def test_skipped_ids_excluded(self) -> None:
        """Skipped IDs (5, 9, 10, 13, 15) are not in the problem set."""
        defined_ids = {p.problem_id for p in get_l4_problems()}
        for sid in L4_SKIPPED_IDS:
            self.assertNotIn(sid, defined_ids)

    def test_attack_order_subset(self) -> None:
        """Attack-order problems are a subset of all problems."""
        all_ids = {p.problem_id for p in get_l4_problems()}
        attack = get_l4_attack_problems()
        self.assertEqual(len(attack), 5)
        for p in attack:
            self.assertIn(p.problem_id, all_ids)

    def test_each_problem_has_expected_ops(self) -> None:
        """Every problem defines at least one expected op."""
        for p in get_l4_problems():
            self.assertGreater(len(p.expected_ops), 0, f"Problem #{p.problem_id} has no expected_ops")


# ---------------------------------------------------------------------------
# Substitutor tests (require torch)
# ---------------------------------------------------------------------------


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestNoerisOpSubstitutor(unittest.TestCase):
    """NoerisOpSubstitutor replacement logic."""

    def test_substitutor_replaces_linear(self) -> None:
        """Linear modules are replaced by NoerisLinearWrapper."""
        from research_engine.kernelbench_l4 import _make_wrapper_classes
        wrappers = _make_wrapper_classes()
        model = nn.Sequential(nn.Linear(10, 10))
        sub = NoerisOpSubstitutor()
        counts = sub.substitute(model)
        self.assertIsInstance(model[0], wrappers["NoerisLinearWrapper"])
        self.assertGreaterEqual(counts.get("Linear", 0), 1)

    def test_substitutor_replaces_layernorm(self) -> None:
        """LayerNorm modules are replaced by NoerisLayerNormWrapper."""
        from research_engine.kernelbench_l4 import _make_wrapper_classes
        wrappers = _make_wrapper_classes()
        model = nn.Sequential(nn.LayerNorm(10))
        sub = NoerisOpSubstitutor()
        counts = sub.substitute(model)
        self.assertIsInstance(model[0], wrappers["NoerisLayerNormWrapper"])
        self.assertGreaterEqual(counts.get("LayerNorm", 0), 1)

    def test_substitutor_replaces_gelu(self) -> None:
        """GELU modules are replaced by NoerisGELUWrapper."""
        from research_engine.kernelbench_l4 import _make_wrapper_classes
        wrappers = _make_wrapper_classes()
        model = nn.Sequential(nn.GELU())
        sub = NoerisOpSubstitutor()
        counts = sub.substitute(model)
        self.assertIsInstance(model[0], wrappers["NoerisGELUWrapper"])
        self.assertGreaterEqual(counts.get("GELU", 0), 1)

    def test_substitutor_preserves_non_target_modules(self) -> None:
        """nn.Dropout and other non-target modules are NOT replaced."""
        model = nn.Sequential(nn.Dropout(0.1), nn.ReLU(), nn.Embedding(100, 10))
        sub = NoerisOpSubstitutor()
        counts = sub.substitute(model)
        self.assertIsInstance(model[0], nn.Dropout)
        self.assertIsInstance(model[1], nn.ReLU)
        self.assertIsInstance(model[2], nn.Embedding)
        self.assertEqual(sum(counts.values()), 0)

    def test_substitutor_handles_gpt2_conv1d(self) -> None:
        """Conv1D detection works when transformers is installed."""
        try:
            from transformers.pytorch_utils import Conv1D
        except ImportError:
            self.skipTest("transformers not installed")

        from research_engine.kernelbench_l4 import _make_wrapper_classes
        wrappers = _make_wrapper_classes()
        conv = Conv1D(20, 10)  # nf=20, nx=10
        model = nn.Sequential(conv)
        sub = NoerisOpSubstitutor()
        counts = sub.substitute(model)
        self.assertIsInstance(model[0], wrappers["NoerisConv1DWrapper"])
        self.assertGreaterEqual(counts.get("Conv1D", 0), 1)

    def test_substitutor_exact_type_match(self) -> None:
        """Only exact type matches trigger replacement (no isinstance)."""

        class MyLinear(nn.Linear):
            pass

        model = nn.Sequential(MyLinear(10, 10))
        sub = NoerisOpSubstitutor()
        counts = sub.substitute(model)
        # MyLinear is a subclass — should NOT be replaced.
        self.assertIsInstance(model[0], MyLinear)
        self.assertEqual(counts.get("Linear", 0), 0)


# ---------------------------------------------------------------------------
# Benchmark script generation tests
# ---------------------------------------------------------------------------


class TestBenchmarkScript(unittest.TestCase):
    """Generated benchmark script validity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = generate_l4_benchmark_script()

    def test_benchmark_script_compiles(self) -> None:
        """Generated script is syntactically valid Python."""
        ast.parse(self.script)

    def test_benchmark_script_has_required_functions(self) -> None:
        """Script contains NoerisOpSubstitutor, benchmark_l4_problem, main."""
        self.assertIn("NoerisOpSubstitutor", self.script)
        self.assertIn("benchmark_l4_problem", self.script)
        self.assertIn("def main()", self.script)

    def test_benchmark_script_installs_transformers(self) -> None:
        """Script contains pip install transformers."""
        self.assertIn("pip", self.script)
        self.assertIn("install", self.script)
        self.assertIn("transformers", self.script)

    def test_benchmark_script_has_allclose_check(self) -> None:
        """Correctness comparison via allclose is present."""
        self.assertIn("allclose", self.script)
        self.assertIn("atol=5e-3", self.script)

    def test_benchmark_script_has_cuda_event_timing(self) -> None:
        """Uses CUDA event timing."""
        self.assertIn("cuda.Event", self.script)
        self.assertIn("enable_timing", self.script)
        self.assertIn("noeris_time", self.script)

    def test_benchmark_script_has_l2_flush(self) -> None:
        """L2 cache flush is present in timing helper."""
        self.assertIn("l2_flush", self.script)

    def test_benchmark_script_reports_json(self) -> None:
        """Script outputs JSON results."""
        self.assertIn("json.dumps", self.script)

    def test_benchmark_script_custom_problems(self) -> None:
        """Script generation accepts a custom problem list."""
        problems = [get_l4_problems()[0]]
        script = generate_l4_benchmark_script(problems)
        ast.parse(script)
        self.assertIn(problems[0].model_name, script)


if __name__ == "__main__":
    unittest.main()
