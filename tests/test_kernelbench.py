"""Tests for src/research_engine/kernelbench.py.

Covers:
  * fast_p strict-> semantics (Task 5)
  * External H100 Modal baseline loader (Task 3)
  * Static checker integration (Task 6)
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.kernelbench import (
    KernelBenchReport,
    ProblemResult,
)


def _make_result(pid: str, speedup: float, level: int = 1, correct: bool = True) -> ProblemResult:
    return ProblemResult(
        problem_id=pid,
        operator="matmul",
        level=level,
        shape={},
        our_best_metric=1.0,
        our_best_config_id="c",
        pytorch_baseline_metric=1.0,
        compile_baseline_metric=0.0,
        speedup=speedup,
        compile_speedup=0.0,
        correct=correct,
    )


class TestFastPStrictInequality(unittest.TestCase):
    """fast_p must use strict > (upstream) not >= (old Noeris)."""

    def test_tie_is_not_counted(self) -> None:
        # A speedup exactly equal to the threshold must NOT pass fast_p=1.0.
        # This is what the upstream score.py does: `speedup > p`.
        report = KernelBenchReport(results=[_make_result("a", speedup=1.0)])
        report.compute_fast_p()
        row = report.fast_p_scores["vs_eager"][1.0]
        self.assertEqual(row["overall"], 0.0, "speedup==p must not pass strict >")

    def test_above_threshold_passes(self) -> None:
        report = KernelBenchReport(results=[_make_result("a", speedup=1.01)])
        report.compute_fast_p()
        self.assertEqual(report.fast_p_scores["vs_eager"][1.0]["overall"], 1.0)

    def test_mixed_tie_and_above(self) -> None:
        results = [
            _make_result("a", speedup=1.0),   # ties threshold — excluded
            _make_result("b", speedup=1.5),   # above — included
            _make_result("c", speedup=2.01),  # above — included
        ]
        report = KernelBenchReport(results=results)
        report.compute_fast_p()
        # fast_1.0: 2 of 3 (a excluded on strict >)
        self.assertAlmostEqual(report.fast_p_scores["vs_eager"][1.0]["overall"], round(2 / 3, 3))
        # fast_1.5: 1 of 3 (b ties -> excluded strict, only c > 1.5)
        self.assertAlmostEqual(report.fast_p_scores["vs_eager"][1.5]["overall"], round(1 / 3, 3))

    def test_incorrect_kernel_never_passes(self) -> None:
        report = KernelBenchReport(results=[
            _make_result("fast_but_wrong", speedup=100.0, correct=False),
        ])
        report.compute_fast_p()
        self.assertEqual(report.fast_p_scores["vs_eager"][1.0]["overall"], 0.0)


class TestExternalH100Loader(unittest.TestCase):
    """Task 3: reference H100 Modal baselines loader."""

    def test_eager_baseline_file_exists(self) -> None:
        path = Path(__file__).resolve().parents[1] / "docs" / "results" / "external" / "kernelbench_h100_modal_baseline_eager.json"
        self.assertTrue(path.exists(), f"Missing: {path}")

    def test_loader_returns_float_for_known_problems(self) -> None:
        from research_engine.kernelbench import load_external_h100_modal_baseline

        known = [
            "1_Square_matrix_multiplication_.py",
            "2_Standard_matrix_multiplication_.py",
            "3_Batched_matrix_multiplication.py",
            "4_Matrix_vector_multiplication_.py",
            "6_Matmul_with_large_K_dimension_.py",
        ]
        for pname in known:
            t = load_external_h100_modal_baseline(pname, level="level1", variant="eager")
            self.assertIsInstance(t, float)
            self.assertGreater(t, 0.0, f"{pname} should have positive baseline ms")

    def test_loader_compile_variant(self) -> None:
        from research_engine.kernelbench import load_external_h100_modal_baseline

        t = load_external_h100_modal_baseline(
            "1_Square_matrix_multiplication_.py", level="level1", variant="compile"
        )
        self.assertIsInstance(t, float)
        self.assertGreater(t, 0.0)

    def test_loader_missing_problem_returns_none(self) -> None:
        from research_engine.kernelbench import load_external_h100_modal_baseline

        result = load_external_h100_modal_baseline("does_not_exist.py", level="level1")
        self.assertIsNone(result)


class TestStaticCheckerIntegration(unittest.TestCase):
    """Task 6: kernel_static_checker is shipped and callable."""

    def test_triton_checker_accepts_simple_kernel(self) -> None:
        from research_engine.kernel_static_checker import validate_kernel_static

        code = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def k(x, y, N: tl.constexpr):\n"
            "    i = tl.program_id(0)\n"
            "    tl.store(y + i, tl.load(x + i))\n"
        )
        valid, errors, warnings = validate_kernel_static(code, backend="triton")
        self.assertTrue(valid, f"Unexpected errors: {errors}")

    def test_checker_flags_try_except_bypass(self) -> None:
        from research_engine.kernel_static_checker import validate_kernel_static

        code = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def k(x): tl.store(x, 0)\n"
            "try:\n"
            "    k(0)\n"
            "except Exception:\n"
            "    pass\n"
        )
        valid, errors, warnings = validate_kernel_static(code, backend="triton")
        self.assertFalse(valid, "try/except fallback must be flagged")


if __name__ == "__main__":
    unittest.main()
