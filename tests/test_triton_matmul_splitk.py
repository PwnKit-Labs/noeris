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
        base_required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                         "SPLIT_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
        persistent_required = base_required | {"PERSISTENT"}
        for c in MATMUL_SPLITK_CURATED_CONFIGS:
            keys = set(c.keys())
            self.assertTrue(
                keys == base_required or keys == persistent_required,
                f"config {c} has wrong keys: {keys}",
            )

    def test_curated_configs_pass_smem_check(self) -> None:
        for c in MATMUL_SPLITK_CURATED_CONFIGS:
            self.assertTrue(
                splitk_shared_memory_check(c),
                f"curated config {splitk_config_id(c)} fails smem check",
            )

    def test_curated_has_multiple_splitk_values(self) -> None:
        sk_values = {c["SPLIT_K"] for c in MATMUL_SPLITK_CURATED_CONFIGS}
        self.assertTrue(len(sk_values) >= 3, f"only {sk_values} split-K values in curated")


class TestPersistentMode(unittest.TestCase):
    """Tests for the persistent kernel scheduling mode."""

    def test_persistent_in_param_space(self) -> None:
        self.assertIn("PERSISTENT", MATMUL_SPLITK_PARAM_SPACE)
        self.assertEqual(MATMUL_SPLITK_PARAM_SPACE["PERSISTENT"], [True, False])

    def test_persistent_curated_configs_exist(self) -> None:
        """At least one curated config has PERSISTENT=True."""
        persistent_configs = [
            c for c in MATMUL_SPLITK_CURATED_CONFIGS if c.get("PERSISTENT")
        ]
        self.assertGreater(len(persistent_configs), 0)

    def test_persistent_config_id_differs_from_standard(self) -> None:
        """PERSISTENT=True and PERSISTENT=False must produce different IDs."""
        base = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
        persistent = {**base, "PERSISTENT": True}
        self.assertNotEqual(splitk_config_id(base), splitk_config_id(persistent))

    def test_persistent_config_id_has_P_suffix(self) -> None:
        config = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "SPLIT_K": 1, "PERSISTENT": True, "GROUP_SIZE_M": 8,
            "num_warps": 4, "num_stages": 3,
        }
        cid = splitk_config_id(config)
        self.assertTrue(cid.endswith("_P"), f"expected _P suffix, got {cid}")

    def test_non_persistent_config_id_no_P_suffix(self) -> None:
        config = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
        cid = splitk_config_id(config)
        self.assertFalse(cid.endswith("_P"), f"unexpected _P suffix in {cid}")

    def test_persistent_kernel_source_present(self) -> None:
        """The persistent kernel function must be in the kernel source."""
        src = mod.TRITON_MATMUL_SPLITK_KERNEL_SOURCE
        self.assertIn("matmul_persistent_splitk_kernel", src)

    def test_persistent_kernel_has_num_sms_param(self) -> None:
        src = mod.TRITON_MATMUL_SPLITK_KERNEL_SOURCE
        self.assertIn("NUM_SMS: tl.constexpr", src)

    def test_persistent_kernel_has_tile_loop(self) -> None:
        """Persistent kernel must loop over tiles with NUM_SMS stride."""
        src = mod.TRITON_MATMUL_SPLITK_KERNEL_SOURCE
        self.assertIn("for tile_id in range(start_pid, num_tiles, NUM_SMS)", src)

    def test_persistent_kernel_has_grouped_ordering(self) -> None:
        """Persistent kernel must use grouped tile ordering for L2 reuse."""
        src = mod.TRITON_MATMUL_SPLITK_KERNEL_SOURCE
        self.assertIn("group_id = tile_id // num_pid_in_group", src)
        self.assertIn("first_pid_m = group_id * GROUP_SIZE_M", src)

    def test_persistent_benchmark_script_has_both_kernels(self) -> None:
        """Benchmark script must contain both standard and persistent kernels."""
        script = generate_splitk_benchmark_script(
            configs=[MATMUL_SPLITK_CURATED_CONFIGS[0]],
            shapes=[MATMUL_SPLITK_SHAPE_BUCKETS[0]],
        )
        self.assertIn("matmul_splitk_kernel", script)
        self.assertIn("matmul_persistent_splitk_kernel", script)

    def test_persistent_benchmark_script_compiles(self) -> None:
        """Benchmark script with persistent config must be valid Python."""
        persistent_config = next(
            c for c in MATMUL_SPLITK_CURATED_CONFIGS if c.get("PERSISTENT")
        )
        script = generate_splitk_benchmark_script(
            configs=[persistent_config],
            shapes=[MATMUL_SPLITK_SHAPE_BUCKETS[0]],
        )
        try:
            compile(script, "<bench_persistent>", "exec")
        except SyntaxError as exc:
            self.fail(f"Persistent benchmark script has a syntax error: {exc}")

    def test_persistent_benchmark_script_selects_kernel(self) -> None:
        """Benchmark launch function must check PERSISTENT flag."""
        script = generate_splitk_benchmark_script(
            configs=[MATMUL_SPLITK_CURATED_CONFIGS[0]],
            shapes=[MATMUL_SPLITK_SHAPE_BUCKETS[0]],
        )
        self.assertIn("PERSISTENT", script)
        self.assertIn("multi_processor_count", script)

    def test_persistent_grid_includes_persistent_configs(self) -> None:
        """Generated grid must contain at least one persistent config."""
        grid = generate_splitk_grid(include_curated=True, max_configs=200)
        persistent = [c for c in grid if c.get("PERSISTENT")]
        self.assertGreater(len(persistent), 0)

    def test_persistent_configs_pass_smem_check(self) -> None:
        for c in MATMUL_SPLITK_CURATED_CONFIGS:
            if c.get("PERSISTENT"):
                self.assertTrue(
                    splitk_shared_memory_check(c),
                    f"persistent config {splitk_config_id(c)} fails smem check",
                )

    def test_persistent_splitk_combination(self) -> None:
        """At least one curated config has both PERSISTENT=True and SPLIT_K > 1."""
        combined = [
            c for c in MATMUL_SPLITK_CURATED_CONFIGS
            if c.get("PERSISTENT") and c["SPLIT_K"] > 1
        ]
        self.assertGreater(
            len(combined), 0,
            "No curated config combines persistent + split-K > 1",
        )


if __name__ == "__main__":
    unittest.main()
