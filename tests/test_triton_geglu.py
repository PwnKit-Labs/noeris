"""Tests for the parameterized fused GeGLU Triton operator (triton_geglu.py)."""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_geglu import (
    GEGLU_CURATED_CONFIGS,
    GEGLU_SHAPE_BUCKETS,
    geglu_config_id,
    geglu_shape_bucket_key,
    geglu_shared_memory_check,
    generate_geglu_benchmark_script,
    generate_geglu_grid,
)


class TestRegistration(unittest.TestCase):
    def test_operator_is_in_registry(self) -> None:
        """geglu must be discoverable via the shared REGISTRY."""
        self.assertIn("geglu", REGISTRY.names())

    def test_spec_metric_is_gb_per_s(self) -> None:
        spec = REGISTRY.get("geglu")
        self.assertEqual(spec.metric_name, "gb_per_s")

    def test_spec_has_curated_configs(self) -> None:
        spec = REGISTRY.get("geglu")
        self.assertGreaterEqual(len(spec.curated_configs), 8)


class TestConfigId(unittest.TestCase):
    def test_config_id_is_stable(self) -> None:
        config = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1}
        self.assertEqual(geglu_config_id(config), "bs1024_w4_s1")

    def test_all_curated_configs_have_unique_ids(self) -> None:
        ids = [geglu_config_id(c) for c in GEGLU_CURATED_CONFIGS]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate config IDs in GEGLU_CURATED_CONFIGS")


class TestShapeBuckets(unittest.TestCase):
    def test_small_test_shape(self) -> None:
        self.assertEqual(
            geglu_shape_bucket_key({"n_rows": 512, "ffn_dim": 1024}),
            "test_small",
        )

    def test_gemma4_e2b(self) -> None:
        self.assertEqual(
            geglu_shape_bucket_key({"n_rows": 2048, "ffn_dim": 5632}),
            "gemma4_e2b",
        )

    def test_gemma4_e4b(self) -> None:
        self.assertEqual(
            geglu_shape_bucket_key({"n_rows": 2048, "ffn_dim": 14336}),
            "gemma4_e4b",
        )

    def test_gemma4_26b(self) -> None:
        self.assertEqual(
            geglu_shape_bucket_key({"n_rows": 2048, "ffn_dim": 16384}),
            "gemma4_26b",
        )

    def test_gemma4_31b(self) -> None:
        self.assertEqual(
            geglu_shape_bucket_key({"n_rows": 2048, "ffn_dim": 24576}),
            "gemma4_31b",
        )

    def test_large_ffn_falls_to_31b_bucket(self) -> None:
        # Any ffn_dim above 24576 should map to the 31b bucket
        self.assertEqual(
            geglu_shape_bucket_key({"n_rows": 4096, "ffn_dim": 32768}),
            "gemma4_31b",
        )

    def test_all_shape_buckets_are_reachable(self) -> None:
        """Every defined shape bucket name must be hit by the classifier."""
        bucket_names = {b["name"] for b in GEGLU_SHAPE_BUCKETS}
        hit = {geglu_shape_bucket_key({"n_rows": b["n_rows"], "ffn_dim": b["ffn_dim"]})
               for b in GEGLU_SHAPE_BUCKETS}
        self.assertEqual(bucket_names, hit)


class TestSharedMemoryCheck(unittest.TestCase):
    def test_small_block_passes(self) -> None:
        config = {"BLOCK_SIZE": 256, "num_warps": 2, "num_stages": 1}
        self.assertTrue(geglu_shared_memory_check(config))

    def test_large_block_single_stage_passes(self) -> None:
        config = {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 1}
        self.assertTrue(geglu_shared_memory_check(config))

    def test_oversized_config_fails(self) -> None:
        # 4096 * 2 bytes * 2 tensors * 20 stages >> 192 KB
        config = {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 20}
        self.assertFalse(geglu_shared_memory_check(config))


class TestGridGenerator(unittest.TestCase):
    def test_grid_has_at_least_ten_configs(self) -> None:
        grid = generate_geglu_grid()
        self.assertGreaterEqual(len(grid), 10)

    def test_grid_has_no_duplicate_ids(self) -> None:
        grid = generate_geglu_grid(max_configs=200)
        ids = [geglu_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate config IDs in generated grid")

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_geglu_grid(max_configs=15)
        self.assertLessEqual(len(grid), 15)

    def test_grid_without_curated_starts_with_systematic_configs(self) -> None:
        # include_curated=False means curated configs are NOT pre-inserted at
        # the front — the systematic loop still produces configs with the same
        # parameter values if they fall in the search space, but they are not
        # guaranteed to appear first.
        grid_with = generate_geglu_grid(include_curated=True, max_configs=200)
        grid_without = generate_geglu_grid(include_curated=False, max_configs=200)
        # Without curated pre-insertion, the first element should NOT be the
        # first curated config (since curated are not injected at head).
        first_curated_id = geglu_config_id(GEGLU_CURATED_CONFIGS[0])
        if grid_without:
            self.assertNotEqual(
                geglu_config_id(grid_without[0]),
                first_curated_id,
                "With include_curated=False the first config should come from the systematic loop",
            )

    def test_grid_all_configs_pass_shmem_check(self) -> None:
        grid = generate_geglu_grid(max_configs=200)
        for config in grid:
            self.assertTrue(
                geglu_shared_memory_check(config),
                f"Config {geglu_config_id(config)} fails shared memory check",
            )


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_geglu_benchmark_script(
            configs=[GEGLU_CURATED_CONFIGS[0]],
            shapes=[{"name": "test_small", "n_rows": 512, "ffn_dim": 1024}],
        )

    def test_script_compiles_as_valid_python(self) -> None:
        script = self._make_script()
        try:
            compile(script, "<bench>", "exec")
        except SyntaxError as exc:
            self.fail(f"Benchmark script has a syntax error: {exc}")

    def test_script_contains_kernel_definition(self) -> None:
        script = self._make_script()
        self.assertIn("def geglu_kernel(", script)

    def test_script_contains_tanh_gelu_approximation(self) -> None:
        script = self._make_script()
        # The tanh approximation constant sqrt(2/pi) ≈ 0.7978845608
        self.assertIn("0.7978845608", script)
        # And the cubic correction coefficient
        self.assertIn("0.044715", script)

    def test_script_contains_torch_reference(self) -> None:
        script = self._make_script()
        # PyTorch reference using gelu with tanh approximation
        self.assertIn('approximate="tanh"', script)

    def test_script_uses_gb_per_s_metric(self) -> None:
        script = self._make_script()
        self.assertIn("gb_per_s", script)

    def test_script_has_correct_bytes_formula(self) -> None:
        # The formula should account for 3 tensors (gate, up, out)
        script = self._make_script()
        self.assertIn("3 * n_rows * ffn_dim * 2", script)

    def test_script_correctness_threshold_is_1e2(self) -> None:
        script = self._make_script()
        self.assertIn("1e-2", script)


if __name__ == "__main__":
    unittest.main()
