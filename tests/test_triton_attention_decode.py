"""Tests for the decode-time paged-KV attention Triton operator.

This kernel implements single-query decode against a paged KV cache.
vLLM's PagedAttention is CUDA-only -- this is a from-scratch Triton
implementation with page-table indirection, online softmax, GQA, and
sliding-window support.
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 -- patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_attention_decode import (
    ATTENTION_DECODE_CURATED_CONFIGS,
    ATTENTION_DECODE_SHAPE_BUCKETS,
    generate_attention_decode_benchmark_script,
    generate_attention_decode_grid,
    attention_decode_config_id,
    attention_decode_shape_bucket_key,
    attention_decode_shared_memory_check,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """attention_decode must be discoverable via the shared REGISTRY."""
        self.assertIn("attention_decode", REGISTRY.names())

    def test_spec_metric_is_gb_per_s(self) -> None:
        spec = REGISTRY.get("attention_decode")
        self.assertEqual(spec.metric_name, "gb_per_s")

    def test_spec_has_curated_configs(self) -> None:
        spec = REGISTRY.get("attention_decode")
        self.assertGreaterEqual(len(spec.curated_configs), 4)

    def test_spec_has_four_shape_buckets(self) -> None:
        spec = REGISTRY.get("attention_decode")
        self.assertEqual(len(spec.shape_buckets), 4)


class TestConfigId(unittest.TestCase):
    def test_config_id_is_stable(self) -> None:
        config = {"BLOCK_KV": 64, "num_warps": 4, "num_stages": 3}
        self.assertEqual(attention_decode_config_id(config), "bkv64_w4_s3")

    def test_all_curated_configs_have_unique_ids(self) -> None:
        ids = [attention_decode_config_id(c) for c in ATTENTION_DECODE_CURATED_CONFIGS]
        self.assertEqual(len(ids), len(set(ids)))


class TestShapeBucketKey(unittest.TestCase):
    def test_shape_bucket_key_all_buckets(self) -> None:
        """Every shape bucket must map to its own name."""
        for bucket in ATTENTION_DECODE_SHAPE_BUCKETS:
            expected = bucket["name"]
            got = attention_decode_shape_bucket_key(bucket)
            self.assertEqual(got, expected, f"bucket {expected} mapped to {got}")

    def test_shape_bucket_key_llama(self) -> None:
        shape = {"head_dim": 128, "context_len": 8192, "window_size": -1, "num_kv_heads": 8}
        self.assertEqual(attention_decode_shape_bucket_key(shape), "llama3_70b_decode")

    def test_shape_bucket_key_gemma_local(self) -> None:
        shape = {"head_dim": 256, "context_len": 4096, "window_size": 1024, "num_kv_heads": 16}
        self.assertEqual(attention_decode_shape_bucket_key(shape), "gemma4_31b_decode_local")

    def test_shape_bucket_key_gemma_global(self) -> None:
        shape = {"head_dim": 512, "context_len": 4096, "window_size": -1, "num_kv_heads": 4}
        self.assertEqual(attention_decode_shape_bucket_key(shape), "gemma4_31b_decode_global")

    def test_shape_bucket_key_gemma_256k(self) -> None:
        shape = {"head_dim": 512, "context_len": 262144, "window_size": -1, "num_kv_heads": 4}
        self.assertEqual(attention_decode_shape_bucket_key(shape), "gemma4_31b_decode_256k")


class TestSharedMemoryCheck(unittest.TestCase):
    def test_small_config_passes(self) -> None:
        config = {"BLOCK_KV": 16, "num_warps": 2, "num_stages": 2}
        self.assertTrue(attention_decode_shared_memory_check(config))

    def test_large_config_passes(self) -> None:
        config = {"BLOCK_KV": 128, "num_warps": 8, "num_stages": 3}
        self.assertTrue(attention_decode_shared_memory_check(config))


class TestGridGeneration(unittest.TestCase):
    def test_grid_generation_non_empty(self) -> None:
        grid = generate_attention_decode_grid()
        self.assertGreater(len(grid), 0)

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_attention_decode_grid(max_configs=5)
        self.assertLessEqual(len(grid), 5)

    def test_grid_has_no_duplicate_ids(self) -> None:
        grid = generate_attention_decode_grid(max_configs=200)
        ids = [attention_decode_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))

    def test_grid_includes_curated_by_default(self) -> None:
        grid = generate_attention_decode_grid()
        curated_ids = {attention_decode_config_id(c) for c in ATTENTION_DECODE_CURATED_CONFIGS}
        grid_ids = {attention_decode_config_id(c) for c in grid}
        self.assertTrue(curated_ids.issubset(grid_ids))


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_attention_decode_benchmark_script(
            configs=[ATTENTION_DECODE_CURATED_CONFIGS[0]],
            shapes=[ATTENTION_DECODE_SHAPE_BUCKETS[0]],
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
            "paged_attention_decode_kernel",
            "paged_attention_decode",
            "torch_paged_attention_decode",
            "benchmark_one",
            "main",
        ):
            self.assertIn(fn, script, f"missing function: {fn}")

    def test_benchmark_script_has_page_table_indirection(self) -> None:
        """The kernel must use page_table and page_idx for indirection."""
        script = self._make_script()
        self.assertIn("page_table", script)
        self.assertIn("page_idx", script)

    def test_benchmark_script_has_gqa_support(self) -> None:
        """The kernel must support grouped-query attention."""
        script = self._make_script()
        has_gqa = "NUM_KV_HEADS" in script or "num_kv_heads" in script
        self.assertTrue(has_gqa, "missing GQA support (NUM_KV_HEADS or num_kv_heads)")

    def test_benchmark_script_has_sliding_window(self) -> None:
        """The kernel must support sliding-window attention."""
        script = self._make_script()
        self.assertIn("WINDOW_SIZE", script)

    def test_benchmark_script_has_online_softmax(self) -> None:
        """The kernel must use online softmax (not materialized attention)."""
        script = self._make_script()
        self.assertIn("m_prev", script)
        self.assertIn("l_prev", script)

    def test_benchmark_script_has_context_lens(self) -> None:
        """The kernel must handle variable-length contexts."""
        script = self._make_script()
        self.assertIn("context_len", script)


if __name__ == "__main__":
    unittest.main()
