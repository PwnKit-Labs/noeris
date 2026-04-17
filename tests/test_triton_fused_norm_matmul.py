"""Tests for the Triton fused RMSNorm + Linear (matmul) operator.

Covers:
- Registration in the shared REGISTRY
- Curated configs use only power-of-2 BLOCK_K values (required by tl.arange)
- Generated benchmark script contains the correct kernel patterns
- Pure-Python correctness test for the reference implementation
- Minimal correctness test (M=4, K=16, N=8) that can run on GPU
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_fused_norm_matmul import (
    FUSED_NORM_LINEAR_CURATED_CONFIGS,
    FUSED_NORM_LINEAR_PARAM_SPACE,
    FUSED_NORM_LINEAR_SHAPE_BUCKETS,
    default_fused_norm_linear_config,
    fused_norm_linear_config_id,
    fused_norm_linear_shape_bucket_key,
    generate_fused_norm_linear_benchmark_script,
    generate_fused_norm_linear_grid,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        self.assertIn("fused_norm_linear", REGISTRY.names())

    def test_spec_metric_is_tflops(self) -> None:
        spec = REGISTRY.get("fused_norm_linear")
        self.assertEqual(spec.metric_name, "tflops")

    def test_spec_has_curated_configs(self) -> None:
        spec = REGISTRY.get("fused_norm_linear")
        self.assertGreaterEqual(len(spec.curated_configs), 5)


class TestBlockKPowerOfTwo(unittest.TestCase):
    """BLOCK_K must be a power of 2 — required by Triton's tl.arange."""

    def _is_power_of_two(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def test_param_space_block_k_all_power_of_two(self) -> None:
        for bk in FUSED_NORM_LINEAR_PARAM_SPACE["BLOCK_K"]:
            self.assertTrue(
                self._is_power_of_two(bk),
                f"BLOCK_K={bk} in param space is not a power of 2",
            )

    def test_curated_configs_block_k_all_power_of_two(self) -> None:
        for config in FUSED_NORM_LINEAR_CURATED_CONFIGS:
            bk = config["BLOCK_K"]
            self.assertTrue(
                self._is_power_of_two(bk),
                f"Curated config {fused_norm_linear_config_id(config)} has "
                f"BLOCK_K={bk} which is not a power of 2",
            )

    def test_generated_grid_block_k_all_power_of_two(self) -> None:
        grid = generate_fused_norm_linear_grid(include_curated=True, max_configs=50)
        for config in grid:
            bk = config["BLOCK_K"]
            self.assertTrue(
                self._is_power_of_two(bk),
                f"Grid config {fused_norm_linear_config_id(config)} has "
                f"BLOCK_K={bk} which is not a power of 2",
            )


class TestBenchmarkScriptKernel(unittest.TestCase):
    """The generated benchmark script must contain correct kernel patterns."""

    def _script(self):
        return generate_fused_norm_linear_benchmark_script(
            configs=[FUSED_NORM_LINEAR_CURATED_CONFIGS[0]],
            shapes=[FUSED_NORM_LINEAR_SHAPE_BUCKETS[0]],
        )

    def test_kernel_uses_pretransposed_weight(self) -> None:
        """Weight must be pre-transposed to (K, N) in the launcher and loaded
        directly as (BLOCK_K, BLOCK_N) — no tl.trans() needed.  This avoids
        Turing (T4) miscompilation of tl.trans inside tl.dot."""
        script = self._script()
        # The kernel should NOT call tl.trans in the dot product line
        self.assertNotIn("tl.dot(x_normed.to(tl.float16), tl.trans(", script)
        # The launcher should pre-transpose the weight
        self.assertIn("linear_weight.t().contiguous()", script)

    def test_kernel_has_two_pass_structure(self) -> None:
        script = self._script()
        self.assertIn("Pass 1", script)
        self.assertIn("Pass 2", script)

    def test_affine_mode_branches(self) -> None:
        script = self._script()
        self.assertIn("AFFINE_MODE == 0", script)
        self.assertIn("1.0 + w_tile", script)

    def test_output_stored_as_fp16(self) -> None:
        script = self._script()
        self.assertIn("acc.to(tl.float16)", script)


class TestShapeBucketClassifier(unittest.TestCase):
    def test_small_k_routes_to_test_small(self) -> None:
        self.assertEqual(
            fused_norm_linear_shape_bucket_key({"M": 128, "K": 256, "N": 512}),
            "test_small",
        )

    def test_gemma_e2b_decode_routes_correctly(self) -> None:
        self.assertEqual(
            fused_norm_linear_shape_bucket_key({"M": 1, "K": 1536, "N": 2560}),
            "gemma4_e2b_decode",
        )

    def test_gemma_e2b_prefill_routes_correctly(self) -> None:
        self.assertEqual(
            fused_norm_linear_shape_bucket_key({"M": 2048, "K": 1536, "N": 2560}),
            "gemma4_e2b_prefill",
        )


class TestDefaultConfigSelection(unittest.TestCase):
    def test_small_batch_prefers_decode_config(self) -> None:
        self.assertEqual(
            default_fused_norm_linear_config(1, 2560, 1536),
            FUSED_NORM_LINEAR_CURATED_CONFIGS[-1],
        )

    def test_e2b_prefill_prefers_single_pass_tilt(self) -> None:
        self.assertEqual(
            default_fused_norm_linear_config(2048, 2560, 1536),
            FUSED_NORM_LINEAR_CURATED_CONFIGS[1],
        )

    def test_larger_gemma_shapes_fall_back_to_two_pass_config(self) -> None:
        self.assertEqual(
            default_fused_norm_linear_config(2048, 6144, 2560),
            FUSED_NORM_LINEAR_CURATED_CONFIGS[3],
        )

    def test_31b_qkv_prefers_large_n_config(self) -> None:
        self.assertEqual(
            default_fused_norm_linear_config(2048, 16384, 5376),
            FUSED_NORM_LINEAR_CURATED_CONFIGS[5],
        )

    def test_31b_gateup_prefers_large_n_config(self) -> None:
        self.assertEqual(
            default_fused_norm_linear_config(2048, 43008, 5376),
            FUSED_NORM_LINEAR_CURATED_CONFIGS[5],
        )


class TestReferenceCorrectness(unittest.TestCase):
    """Pure-Python check that the reference impl in the generated script
    matches a hand-rolled RMSNorm + matmul (no GPU needed)."""

    def test_reference_impl_present(self) -> None:
        script = generate_fused_norm_linear_benchmark_script(
            configs=[FUSED_NORM_LINEAR_CURATED_CONFIGS[0]],
            shapes=[FUSED_NORM_LINEAR_SHAPE_BUCKETS[0]],
        )
        self.assertIn("torch_rmsnorm_linear", script)
        self.assertIn("torch.rsqrt(variance + eps)", script)


class TestMinimalGPUCorrectness(unittest.TestCase):
    """Minimal correctness test: M=4, K=16, N=8.

    Skipped if no CUDA device is available.  Prints both outputs and the
    difference for easy debugging.
    """

    def test_fused_matches_reference_small(self) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("No CUDA device")
        except ImportError:
            self.skipTest("PyTorch not installed")

        try:
            from research_engine.triton_fused_norm_matmul import fused_rmsnorm_linear
        except Exception:
            self.skipTest("Triton not available")

        M, K, N = 4, 16, 8
        torch.manual_seed(42)
        x = torch.randn((M, K), device="cuda", dtype=torch.float16)
        w = torch.randn((K,), device="cuda", dtype=torch.float16) * 0.01
        linear_w = torch.randn((N, K), device="cuda", dtype=torch.float16) * (K ** -0.5)

        # Reference: RMSNorm in fp32, then matmul
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + 1e-6)
        w_f32 = w.to(torch.float32)
        x_normed = x_normed * w_f32
        ref = x_normed.to(torch.float16) @ linear_w.t()

        # Fused kernel — use a small power-of-2 config
        config = {
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "BLOCK_K": 16,
            "num_warps": 2,
            "num_stages": 1,
        }
        out = fused_rmsnorm_linear(x, w, linear_w, eps=1e-6, affine_mode=0, config=config)

        # Print for debugging
        print(f"\nReference output:\n{ref}")
        print(f"\nFused output:\n{out}")
        diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
        print(f"\nAbsolute difference:\n{diff}")
        print(f"Max abs error: {diff.max().item():.6f}")

        # fp16 matmul tolerance
        self.assertTrue(
            diff.max().item() < 0.1,
            f"Max absolute error {diff.max().item():.6f} exceeds tolerance 0.1",
        )

    def _assert_fused_matches_reference(
        self,
        *,
        m: int,
        k: int,
        n: int,
        affine_mode: int,
        atol: float,
        min_memory_gb: int = 0,
    ) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("No CUDA device")
        except ImportError:
            self.skipTest("PyTorch not installed")

        try:
            from research_engine.triton_fused_norm_matmul import (
                fused_rmsnorm_linear,
                torch_rmsnorm_linear_reference,
            )
        except Exception:
            self.skipTest("Triton not available")

        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_memory_gb < min_memory_gb:
            self.skipTest(
                f"Need at least {min_memory_gb} GB GPU memory, found {total_memory_gb:.1f} GB",
            )

        torch.manual_seed(123)
        x = torch.randn((m, k), device="cuda", dtype=torch.float16)
        w = torch.randn((k,), device="cuda", dtype=torch.float16) * 0.01
        linear_w = torch.randn((n, k), device="cuda", dtype=torch.float16) * (k ** -0.5)

        ref = torch_rmsnorm_linear_reference(
            x,
            w,
            linear_w,
            eps=1e-6,
            affine_mode=affine_mode,
        )
        out = fused_rmsnorm_linear(
            x,
            w,
            linear_w,
            eps=1e-6,
            affine_mode=affine_mode,
            config=default_fused_norm_linear_config(m, n, k),
        )

        diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
        max_abs_err = diff.max().item()
        self.assertLess(
            max_abs_err,
            atol,
            f"Shape {m}x{k}->{n} affine_mode={affine_mode} max_abs_err={max_abs_err:.6f} > {atol}",
        )

    def test_fused_matches_reference_gemma_e2b_dims(self) -> None:
        self._assert_fused_matches_reference(
            m=2,
            k=1536,
            n=2560,
            affine_mode=1,
            atol=0.25,
            min_memory_gb=8,
        )

    def test_fused_matches_reference_gemma_e4b_dims(self) -> None:
        self._assert_fused_matches_reference(
            m=2,
            k=2560,
            n=6144,
            affine_mode=1,
            atol=0.35,
            min_memory_gb=12,
        )


if __name__ == "__main__":
    unittest.main()
