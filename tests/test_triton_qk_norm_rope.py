"""Tests for the fused QK-RMSNorm + RoPE Triton operator.

This is the Gemma 3/4 attention prologue kernel. The headline claim it
supports is that vLLM launches 4 separate kernels (Q-RMSNorm, K-RMSNorm,
Q-RoPE, K-RoPE) while Noeris fuses them into a single pass per tensor.
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_qk_norm_rope import (
    QK_NORM_ROPE_CURATED_CONFIGS,
    QK_NORM_ROPE_SHAPE_BUCKETS,
    generate_qk_norm_rope_benchmark_script,
    generate_qk_norm_rope_grid,
    qk_norm_rope_config_id,
    qk_norm_rope_shape_bucket_key,
    qk_norm_rope_shared_memory_check,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """qk_norm_rope must be discoverable via the shared REGISTRY."""
        self.assertIn("qk_norm_rope", REGISTRY.names())

    def test_spec_metric_is_gb_per_s(self) -> None:
        spec = REGISTRY.get("qk_norm_rope")
        self.assertEqual(spec.metric_name, "gb_per_s")

    def test_spec_has_curated_configs(self) -> None:
        spec = REGISTRY.get("qk_norm_rope")
        self.assertGreaterEqual(len(spec.curated_configs), 5)

    def test_spec_has_all_six_gemma_buckets(self) -> None:
        spec = REGISTRY.get("qk_norm_rope")
        self.assertEqual(len(spec.shape_buckets), 6)


class TestConfigId(unittest.TestCase):
    def test_config_id_is_stable(self) -> None:
        config = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2}
        self.assertEqual(qk_norm_rope_config_id(config), "bs128_w4_s2")

    def test_all_curated_configs_have_unique_ids(self) -> None:
        ids = [qk_norm_rope_config_id(c) for c in QK_NORM_ROPE_CURATED_CONFIGS]
        self.assertEqual(len(ids), len(set(ids)))


class TestShapeBucketKey(unittest.TestCase):
    def test_shape_bucket_key_gemma3_local_1024(self) -> None:
        shape = {"batch": 1, "heads": 16, "num_kv_heads": 16, "seq": 4096, "head_dim": 256}
        self.assertEqual(qk_norm_rope_shape_bucket_key(shape), "gemma3_local_1024")

    def test_shape_bucket_key_gemma4_e2b_local(self) -> None:
        shape = {"batch": 1, "heads": 8, "num_kv_heads": 1, "seq": 4096, "head_dim": 256}
        self.assertEqual(qk_norm_rope_shape_bucket_key(shape), "gemma4_e2b_local")

    def test_shape_bucket_key_gemma4_26b_a4b_local(self) -> None:
        shape = {"batch": 1, "heads": 16, "num_kv_heads": 8, "seq": 4096, "head_dim": 256}
        self.assertEqual(qk_norm_rope_shape_bucket_key(shape), "gemma4_26b_a4b_local")

    def test_shape_bucket_key_gemma4_26b_a4b_global(self) -> None:
        shape = {"batch": 1, "heads": 16, "num_kv_heads": 2, "seq": 4096, "head_dim": 512}
        self.assertEqual(qk_norm_rope_shape_bucket_key(shape), "gemma4_26b_a4b_global")

    def test_shape_bucket_key_gemma4_31b_local(self) -> None:
        shape = {"batch": 1, "heads": 32, "num_kv_heads": 16, "seq": 4096, "head_dim": 256}
        self.assertEqual(qk_norm_rope_shape_bucket_key(shape), "gemma4_31b_local")

    def test_shape_bucket_key_gemma4_31b_global(self) -> None:
        shape = {"batch": 1, "heads": 32, "num_kv_heads": 4, "seq": 4096, "head_dim": 512}
        self.assertEqual(qk_norm_rope_shape_bucket_key(shape), "gemma4_31b_global")

    def test_all_shape_buckets_are_reachable(self) -> None:
        bucket_names = {b["name"] for b in QK_NORM_ROPE_SHAPE_BUCKETS}
        hit = {qk_norm_rope_shape_bucket_key(b) for b in QK_NORM_ROPE_SHAPE_BUCKETS}
        self.assertEqual(bucket_names, hit)


class TestSharedMemoryCheck(unittest.TestCase):
    def test_small_block_passes(self) -> None:
        config = {"BLOCK_SIZE": 32, "num_warps": 2, "num_stages": 1}
        self.assertTrue(qk_norm_rope_shared_memory_check(config))

    def test_large_block_passes(self) -> None:
        config = {"BLOCK_SIZE": 256, "num_warps": 8, "num_stages": 2}
        self.assertTrue(qk_norm_rope_shared_memory_check(config))


class TestGridGeneration(unittest.TestCase):
    def test_grid_generation_is_non_empty(self) -> None:
        grid = generate_qk_norm_rope_grid()
        self.assertGreater(len(grid), 0)

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_qk_norm_rope_grid(max_configs=10)
        self.assertLessEqual(len(grid), 10)

    def test_grid_has_no_duplicate_ids(self) -> None:
        grid = generate_qk_norm_rope_grid(max_configs=200)
        ids = [qk_norm_rope_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_qk_norm_rope_benchmark_script(
            configs=[QK_NORM_ROPE_CURATED_CONFIGS[0]],
            shapes=[QK_NORM_ROPE_SHAPE_BUCKETS[0]],
        )

    def test_benchmark_script_compiles(self) -> None:
        script = self._make_script()
        try:
            compile(script, "<bench>", "exec")
        except SyntaxError as exc:
            self.fail(f"Benchmark script has a syntax error: {exc}")

    def test_benchmark_script_has_required_functions(self) -> None:
        script = self._make_script()
        for fn in (
            "qk_norm_rope_kernel",
            "apply_qk_norm_rope",
            "torch_qk_norm_rope",
            "separated_baseline",
            "benchmark_one",
            "main",
        ):
            self.assertIn(fn, script, f"missing {fn}")

    def test_script_uses_gemma_mode_affine(self) -> None:
        # (1 + scale) not scale
        script = self._make_script()
        self.assertIn("1.0 + s_even", script)
        self.assertIn("1.0 + s_odd", script)

    def test_script_reports_fusion_speedup(self) -> None:
        script = self._make_script()
        self.assertIn("fusion_speedup", script)
        self.assertIn("separated_ms", script)

    def test_script_uses_gb_per_s_metric(self) -> None:
        script = self._make_script()
        self.assertIn("gb_per_s", script)

    def test_script_uses_correct_eps(self) -> None:
        script = self._make_script()
        self.assertIn("1e-6", script)


if __name__ == "__main__":
    unittest.main()
