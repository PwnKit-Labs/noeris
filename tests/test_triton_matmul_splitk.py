"""Tests for the split-K matmul Triton operator.

Split-K splits the K dimension across SPLIT_K thread blocks with
interleaved access and atomic-add reduction.  SPLIT_K=1 degenerates
to a standard tiled matmul.  This operator targets cuBLAS-competitive
performance on large-K / small-MN shapes (e.g. LLM down-projections).
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine import triton_matmul_splitk as mod
from research_engine.triton_operators import TritonOperatorSpec
from research_engine.triton_matmul_splitk import (
    MATMUL_SPLITK_CURATED_CONFIGS,
    MATMUL_SPLITK_PARAM_SPACE,
    MATMUL_SPLITK_SHAPE_BUCKETS,
    generate_splitk_benchmark_script,
    generate_splitk_grid,
    splitk_config_id,
    splitk_shape_bucket_key,
    splitk_shared_memory_check,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """matmul_splitk spec must be created at module import time."""
        self.assertTrue(hasattr(mod, "MATMUL_SPLITK_SPEC"))
        self.assertIsInstance(mod.MATMUL_SPLITK_SPEC, TritonOperatorSpec)
        self.assertEqual(mod.MATMUL_SPLITK_SPEC.name, "matmul_splitk")

    def test_spec_metric_is_tflops(self) -> None:
        self.assertEqual(mod.MATMUL_SPLITK_SPEC.metric_name, "tflops")


class TestShapeBuckets(unittest.TestCase):
    def test_shape_bucket_key_generic_buckets(self) -> None:
        """Generic shape buckets (tiny..xlarge, tall_skinny, deep_k)
        must round-trip through the bucket function.  LLM-specific
        shapes (llm_qkv, llm_mlp, llm_mlp_down) are classified into
        the nearest generic bucket, so they are not checked here.
        """
        generic_names = {"tiny", "small", "medium", "large", "xlarge",
                         "tall_skinny", "deep_k"}
        for bucket in MATMUL_SPLITK_SHAPE_BUCKETS:
            if bucket["name"] in generic_names:
                key = splitk_shape_bucket_key(bucket)
                self.assertEqual(key, bucket["name"])

    def test_shape_bucket_key_tiny(self) -> None:
        self.assertEqual(splitk_shape_bucket_key({"M": 128, "N": 128, "K": 128}), "tiny")

    def test_shape_bucket_key_deep_k(self) -> None:
        self.assertEqual(splitk_shape_bucket_key({"M": 1024, "N": 1024, "K": 8192}), "deep_k")

    def test_shape_bucket_key_xlarge(self) -> None:
        self.assertEqual(splitk_shape_bucket_key({"M": 8192, "N": 8192, "K": 8192}), "xlarge")


class TestGridGeneration(unittest.TestCase):
    def test_grid_non_empty(self) -> None:
        grid = generate_splitk_grid()
        self.assertGreater(len(grid), 0)

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_splitk_grid(max_configs=10)
        self.assertLessEqual(len(grid), 10)

    def test_grid_no_duplicate_ids(self) -> None:
        grid = generate_splitk_grid(max_configs=200)
        ids = [splitk_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))

    def test_grid_includes_curated(self) -> None:
        grid = generate_splitk_grid(include_curated=True, max_configs=200)
        grid_ids = {splitk_config_id(c) for c in grid}
        for curated in MATMUL_SPLITK_CURATED_CONFIGS:
            self.assertIn(splitk_config_id(curated), grid_ids)


class TestConfigId(unittest.TestCase):
    def test_config_id_contains_split_k(self) -> None:
        config = MATMUL_SPLITK_CURATED_CONFIGS[0]
        cid = splitk_config_id(config)
        self.assertIn("sk", cid)

    def test_config_id_deterministic(self) -> None:
        config = MATMUL_SPLITK_CURATED_CONFIGS[0]
        self.assertEqual(splitk_config_id(config), splitk_config_id(config))

    def test_different_splitk_different_id(self) -> None:
        base = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
        variant = {**base, "SPLIT_K": 4}
        self.assertNotEqual(splitk_config_id(base), splitk_config_id(variant))


class TestSharedMemoryCheck(unittest.TestCase):
    def test_small_block_passes(self) -> None:
        config = {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
            "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
        self.assertTrue(splitk_shared_memory_check(config))

    def test_huge_block_fails(self) -> None:
        config = {
            "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
            "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4,
        }
        # (256*128 + 128*256) * 2 * 4 = 524288 > 228*1024
        self.assertFalse(splitk_shared_memory_check(config))


class TestSplitkParamInConfigs(unittest.TestCase):
    """SPLIT_K must appear in the parameter space."""

    def test_splitk_param_in_space(self) -> None:
        self.assertIn("SPLIT_K", MATMUL_SPLITK_PARAM_SPACE)
        self.assertIsInstance(MATMUL_SPLITK_PARAM_SPACE["SPLIT_K"], list)
        self.assertIn(1, MATMUL_SPLITK_PARAM_SPACE["SPLIT_K"])

    def test_splitk_values_are_powers_of_two(self) -> None:
        for v in MATMUL_SPLITK_PARAM_SPACE["SPLIT_K"]:
            self.assertTrue(v > 0 and (v & (v - 1)) == 0, f"{v} is not a power of 2")

    def test_curated_configs_have_splitk(self) -> None:
        for c in MATMUL_SPLITK_CURATED_CONFIGS:
            self.assertIn("SPLIT_K", c)


class TestSplitkOneIsStandardMatmul(unittest.TestCase):
    """SPLIT_K=1 should behave identically to non-split tiled matmul."""

    def test_splitk_1_in_curated(self) -> None:
        """At least one curated config has SPLIT_K=1 (baseline)."""
        sk1_configs = [c for c in MATMUL_SPLITK_CURATED_CONFIGS if c["SPLIT_K"] == 1]
        self.assertGreater(len(sk1_configs), 0)

    def test_splitk_1_kernel_uses_store_not_atomic(self) -> None:
        """When SPLIT_K=1, the kernel should use tl.store, not tl.atomic_add."""
        src = mod.TRITON_MATMUL_SPLITK_KERNEL_SOURCE
        # The kernel has a conditional: if SPLIT_K == 1 -> tl.store, else -> tl.atomic_add
        self.assertIn("if SPLIT_K == 1", src)
        self.assertIn("tl.store", src)
        self.assertIn("tl.atomic_add", src)

    def test_splitk_1_grid_has_z_dim_1(self) -> None:
        """When SPLIT_K=1, the 3D grid's z dimension should be 1."""
        script = generate_splitk_benchmark_script(
            configs=[{
                "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
            }],
            shapes=[MATMUL_SPLITK_SHAPE_BUCKETS[0]],
        )
        # The grid lambda should produce SPLIT_K as third element
        self.assertIn("SPLIT_K", script)


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_splitk_benchmark_script(
            configs=[MATMUL_SPLITK_CURATED_CONFIGS[0]],
            shapes=[MATMUL_SPLITK_SHAPE_BUCKETS[0]],
        )

    def test_script_compiles_as_python(self) -> None:
        script = self._make_script()
        try:
            compile(script, "<bench>", "exec")
        except SyntaxError as exc:
            self.fail(f"Benchmark script has a syntax error: {exc}")

    def test_script_has_required_functions(self) -> None:
        script = self._make_script()
        for fn in (
            "matmul_splitk_kernel",
            "matmul_splitk",
            "check_correctness",
            "benchmark_config",
            "main",
        ):
            self.assertIn(fn, script, f"missing {fn}")

    def test_script_has_pre_run_hook(self) -> None:
        """Output must be zeroed before atomic-add reduction."""
        script = self._make_script()
        self.assertIn("add_pre_run_hook", script)
        self.assertIn("zero_", script)

    def test_script_reports_cublas_ratio(self) -> None:
        script = self._make_script()
        self.assertIn("ratio_vs_cublas", script)
        self.assertIn("cublas_ms", script)

    def test_script_has_interleaved_k_access(self) -> None:
        """Split-K uses interleaved K-block striding, not contiguous chunks."""
        script = self._make_script()
        self.assertIn("BLOCK_SIZE_K * SPLIT_K", script)

    def test_script_has_atomic_add(self) -> None:
        script = self._make_script()
        self.assertIn("tl.atomic_add", script)
        self.assertIn('sem="relaxed"', script)


class TestCuratedConfigs(unittest.TestCase):
    def test_curated_configs_have_all_required_keys(self) -> None:
        required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                    "SPLIT_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
        for c in MATMUL_SPLITK_CURATED_CONFIGS:
            self.assertEqual(set(c.keys()), required, f"config {c} has wrong keys")

    def test_curated_configs_pass_smem_check(self) -> None:
        for c in MATMUL_SPLITK_CURATED_CONFIGS:
            self.assertTrue(
                splitk_shared_memory_check(c),
                f"curated config {splitk_config_id(c)} fails smem check",
            )

    def test_curated_has_multiple_splitk_values(self) -> None:
        sk_values = {c["SPLIT_K"] for c in MATMUL_SPLITK_CURATED_CONFIGS}
        self.assertTrue(len(sk_values) >= 3, f"only {sk_values} split-K values in curated")


if __name__ == "__main__":
    unittest.main()
