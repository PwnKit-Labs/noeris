"""Tests for the grouped GEMM Triton operator (MoE expert FFN w1).

This is the second half of the Gemma 4 26B-A4B MoE path. The headline
claim it supports is that 128 separate matmul launches collapse into 1
fused grouped GEMM via vLLM's sort-free A-dispatch trick.
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine import triton_grouped_gemm as mod
from research_engine.triton_operators import TritonOperatorSpec
from research_engine.triton_grouped_gemm import (
    GROUPED_GEMM_CURATED_CONFIGS,
    GROUPED_GEMM_SHAPE_BUCKETS,
    generate_grouped_gemm_benchmark_script,
    generate_grouped_gemm_grid,
    grouped_gemm_config_id,
    grouped_gemm_shape_bucket_key,
    grouped_gemm_shared_memory_check,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """grouped_gemm spec must be created at module import time.

        We do not touch ``__init__.py`` (parallel agent is editing it),
        so we verify the module's own ``register_operator(...)`` call
        produced a valid ``TritonOperatorSpec`` rather than asserting
        membership in the package-wide REGISTRY.
        """
        self.assertTrue(hasattr(mod, "GROUPED_GEMM_SPEC"))
        self.assertIsInstance(mod.GROUPED_GEMM_SPEC, TritonOperatorSpec)
        self.assertEqual(mod.GROUPED_GEMM_SPEC.name, "grouped_gemm")


class TestShapeBuckets(unittest.TestCase):
    def test_shape_bucket_key_all_buckets(self) -> None:
        for bucket in GROUPED_GEMM_SHAPE_BUCKETS:
            key = grouped_gemm_shape_bucket_key(bucket)
            self.assertEqual(key, bucket["name"])

    def test_shape_bucket_key_small(self) -> None:
        self.assertEqual(
            grouped_gemm_shape_bucket_key({"num_tokens": 1024}),
            "gemma4_26b_a4b_w1_small",
        )

    def test_shape_bucket_key_xlong(self) -> None:
        self.assertEqual(
            grouped_gemm_shape_bucket_key({"num_tokens": 16384}),
            "gemma4_26b_a4b_w1_xlong",
        )


class TestGridGeneration(unittest.TestCase):
    def test_grid_generation_non_empty(self) -> None:
        grid = generate_grouped_gemm_grid()
        self.assertGreater(len(grid), 0)

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_grouped_gemm_grid(max_configs=10)
        self.assertLessEqual(len(grid), 10)

    def test_grid_has_no_duplicate_ids(self) -> None:
        grid = generate_grouped_gemm_grid(max_configs=200)
        ids = [grouped_gemm_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))


class TestSharedMemoryCheck(unittest.TestCase):
    def test_small_block_passes(self) -> None:
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                  "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
        self.assertTrue(grouped_gemm_shared_memory_check(config))


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_grouped_gemm_benchmark_script(
            configs=[GROUPED_GEMM_CURATED_CONFIGS[0]],
            shapes=[GROUPED_GEMM_SHAPE_BUCKETS[0]],
        )

    def test_benchmark_script_compiles_as_python(self) -> None:
        script = self._make_script()
        try:
            compile(script, "<bench>", "exec")
        except SyntaxError as exc:
            self.fail(f"Benchmark script has a syntax error: {exc}")

    def test_benchmark_script_has_required_functions(self) -> None:
        script = self._make_script()
        for fn in (
            "grouped_gemm_kernel",
            "grouped_gemm",
            "torch_grouped_gemm",
            "separated_grouped_gemm",
            "benchmark_one",
            "main",
        ):
            self.assertIn(fn, script, f"missing {fn}")

    def test_benchmark_script_has_sort_free_dispatch(self) -> None:
        script = self._make_script()
        self.assertIn("sorted_token_ids", script)
        self.assertIn("// TOP_K", script)

    def test_benchmark_script_has_per_tile_expert_selection(self) -> None:
        script = self._make_script()
        self.assertIn("expert_ids", script)

    def test_benchmark_script_has_mul_routed_weight_constexpr(self) -> None:
        script = self._make_script()
        self.assertIn("MUL_ROUTED_WEIGHT", script)
        # w1 pass: MUL_ROUTED_WEIGHT must be set False
        self.assertIn("MUL_ROUTED_WEIGHT=False", script)

    def test_benchmark_script_reports_fusion_speedup(self) -> None:
        script = self._make_script()
        self.assertIn("fusion_speedup", script)


class TestCuratedConfigs(unittest.TestCase):
    def test_curated_configs_contain_block_size_m_k_n(self) -> None:
        for c in GROUPED_GEMM_CURATED_CONFIGS:
            self.assertIn("BLOCK_SIZE_M", c)
            self.assertIn("BLOCK_SIZE_N", c)
            self.assertIn("BLOCK_SIZE_K", c)
            self.assertIn("GROUP_SIZE_M", c)
            self.assertIn("num_warps", c)
            self.assertIn("num_stages", c)


if __name__ == "__main__":
    unittest.main()
