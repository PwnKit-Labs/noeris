"""Tests for the fused QK-RMSNorm + RoPE backward pass Triton operator.

This is the backward kernel for the Gemma 3/4 attention prologue — the
training-time counterpart to the 10-13x inference fusion. No framework
fuses this backward pass.
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_qk_norm_rope import (
    QK_NORM_ROPE_CURATED_CONFIGS,
    QK_NORM_ROPE_SHAPE_BUCKETS,
)
from research_engine.triton_qk_norm_rope_bwd import (
    generate_qk_norm_rope_bwd_benchmark_script,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """qk_norm_rope_bwd must be discoverable via the shared REGISTRY."""
        self.assertIn("qk_norm_rope_bwd", REGISTRY.names())


class TestShapeBucketsMatchForward(unittest.TestCase):
    def test_shape_buckets_match_forward(self) -> None:
        """Backward uses the same shape buckets as the forward pass."""
        bwd_spec = REGISTRY.get("qk_norm_rope_bwd")
        fwd_spec = REGISTRY.get("qk_norm_rope")
        self.assertEqual(bwd_spec.shape_buckets, fwd_spec.shape_buckets)


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_qk_norm_rope_bwd_benchmark_script(
            configs=[QK_NORM_ROPE_CURATED_CONFIGS[0]],
            shapes=[QK_NORM_ROPE_SHAPE_BUCKETS[0]],
        )

    def test_benchmark_script_compiles(self) -> None:
        script = self._make_script()
        try:
            compile(script, "<bench_bwd>", "exec")
        except SyntaxError as exc:
            self.fail(f"Backward benchmark script has a syntax error: {exc}")

    def test_benchmark_script_has_backward_kernel(self) -> None:
        script = self._make_script()
        self.assertIn("qk_norm_rope_bwd_kernel", script)

    def test_benchmark_script_has_dscale_accumulation(self) -> None:
        """Backward kernel must use atomic_add for dscale accumulation."""
        script = self._make_script()
        self.assertIn("atomic_add", script)

    def test_benchmark_script_has_rope_inverse(self) -> None:
        """Backward must invert the RoPE rotation (transpose rotation matrix)."""
        script = self._make_script()
        # Inverse rotation: dout_even * cos + dout_odd * sin for even component
        # and -dout_even * sin + dout_odd * cos for odd component.
        # The key signature is the negation in the odd inverse.
        self.assertIn("-dout_even", script)
        # Also check the positive cross-term
        self.assertIn("dout_odd * sn", script)

    def test_benchmark_script_reports_backward_fusion_speedup(self) -> None:
        script = self._make_script()
        self.assertIn("backward_fusion_speedup", script)

    def test_benchmark_script_has_recompute_forward(self) -> None:
        """Backward recomputes x_norm from x (FlashAttention-style), not loads saved."""
        script = self._make_script()
        # The recompute path loads x and computes rstd in the backward kernel
        self.assertIn("x_norm_even = x_even * rstd", script)
        self.assertIn("x_norm_odd = x_odd * rstd", script)
        # Should NOT have a "load x_norm" — it recomputes, not loads
        self.assertNotIn("load(x_norm_ptr", script)


if __name__ == "__main__":
    unittest.main()
