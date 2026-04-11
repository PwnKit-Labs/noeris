"""Tests for Triton RMSNorm spec: Gemma-mode affine and stale bucket fixes.

Covers issues #46 (Gemma (1+w) affine mode) and #45 (stale Gemma 4 hidden_dim).
These tests are pure-python: they check shape metadata, classifier routing,
and that the generated benchmark script string contains the expected
AFFINE_MODE branch. They do NOT launch the Triton kernel (no GPU needed).
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_rmsnorm import (
    RMSNORM_CURATED_CONFIGS,
    RMSNORM_SHAPE_BUCKETS,
    generate_rmsnorm_benchmark_script,
    rmsnorm_shape_bucket_key,
)


class TestAffineModeInKernelSource(unittest.TestCase):
    """The generated benchmark script must contain both AFFINE_MODE branches."""

    def _script(self):
        return generate_rmsnorm_benchmark_script(
            configs=[RMSNORM_CURATED_CONFIGS[0]],
            shapes=[b for b in RMSNORM_SHAPE_BUCKETS if b["name"].startswith("gemma4_")],
        )

    def test_affine_mode_0_matches_standard_formula(self):
        script = self._script()
        # Standard path unchanged
        self.assertIn("AFFINE_MODE == 0", script)
        self.assertIn("y = x * rstd * w", script)

    def test_affine_mode_1_matches_gemma_formula(self):
        script = self._script()
        # Gemma (1 + w) path present
        self.assertIn("y = x * rstd * (1.0 + w)", script)

    def test_kernel_accepts_affine_mode_constexpr(self):
        script = self._script()
        self.assertIn("AFFINE_MODE: tl.constexpr", script)

    def test_launcher_passes_affine_mode(self):
        script = self._script()
        self.assertIn("AFFINE_MODE=affine_mode", script)

    def test_pytorch_reference_branches_on_affine_mode(self):
        script = self._script()
        # reference path for gemma includes (1.0 + w) multiplication
        self.assertIn("(1.0 + w)", script)

    def test_script_compiles_as_python(self):
        script = self._script()
        compile(script, "<bench_rmsnorm_affine>", "exec")


class TestGemma4BucketMetadata(unittest.TestCase):
    """Gemma 4 buckets must carry affine_mode=1 and correct HF hidden_dim."""

    def _bucket(self, name):
        for b in RMSNORM_SHAPE_BUCKETS:
            if b["name"] == name:
                return b
        raise KeyError(name)

    def test_gemma4_buckets_default_to_affine_mode_1(self):
        for b in RMSNORM_SHAPE_BUCKETS:
            if b["name"].startswith("gemma4_"):
                self.assertEqual(
                    b.get("affine_mode"),
                    1,
                    f"Gemma bucket {b['name']} must set affine_mode=1",
                )

    def test_non_gemma_buckets_affine_mode_0(self):
        for b in RMSNORM_SHAPE_BUCKETS:
            if not b["name"].startswith("gemma4_"):
                self.assertEqual(
                    b.get("affine_mode"),
                    0,
                    f"Non-Gemma bucket {b['name']} must set affine_mode=0",
                )

    def test_gemma4_e2b_hidden_1536(self):
        # Issue #45: HF config.json hidden_size for gemma-4-E2B-it is 1536
        self.assertEqual(self._bucket("gemma4_e2b")["hidden_dim"], 1536)

    def test_gemma4_e4b_hidden_2560(self):
        self.assertEqual(self._bucket("gemma4_e4b")["hidden_dim"], 2560)

    def test_gemma4_26b_hidden_2816(self):
        # Issue #45: HF config.json hidden_size for gemma-4-26B-A4B-it is 2816
        self.assertEqual(self._bucket("gemma4_26b")["hidden_dim"], 2816)

    def test_gemma4_31b_hidden_5376(self):
        self.assertEqual(self._bucket("gemma4_31b")["hidden_dim"], 5376)


class TestShapeBucketKeyRoutesGemma4Correctly(unittest.TestCase):
    """Classifier routes real Gemma 4 hidden_dims to the right bucket."""

    def test_shape_bucket_key_routes_gemma4_correctly(self):
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 2048, "hidden_dim": 1536}),
            "gemma4_e2b",
        )
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 2048, "hidden_dim": 2560}),
            "gemma4_e4b",
        )
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 4096, "hidden_dim": 2816}),
            "gemma4_26b",
        )
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 4096, "hidden_dim": 5376}),
            "gemma4_31b",
        )


if __name__ == "__main__":
    unittest.main()
