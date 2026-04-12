"""Tests for triton_attention_v2.py — official FA2 tutorial kernel + Noeris extensions.

Tests cover:
- Operator spec registration under ``attention_v2``
- Shape buckets match v1 (identical workload coverage)
- Benchmark script compiles as valid Python
- Sliding-window support (WINDOW_SIZE parameter in kernel source)
- GQA support (GROUP_SIZE / NUM_KV_HEADS in kernel source)
"""

from __future__ import annotations

import ast
import unittest

from tests import _pathfix  # noqa: F401

from research_engine.triton_attention_v2 import (
    ATTENTION_V2_CURATED_CONFIGS,
    ATTENTION_V2_SHAPE_BUCKETS,
    attention_v2_config_id,
    attention_v2_shape_bucket_key,
    generate_attention_v2_benchmark_script,
    generate_attention_v2_grid,
    flash_attn_v2,
)
from research_engine.triton_operators import REGISTRY


class TestV2SpecRegistration(unittest.TestCase):
    """test_v2_spec_registration: attention_v2 is registered in REGISTRY."""

    def test_v2_spec_registration(self):
        self.assertIn("attention_v2", REGISTRY.names())
        spec = REGISTRY.get("attention_v2")
        self.assertEqual(spec.name, "attention_v2")
        self.assertEqual(spec.metric_name, "tflops")
        # Must have the key functions
        self.assertIsNotNone(spec.config_id_fn)
        self.assertIsNotNone(spec.shape_bucket_fn)
        self.assertIsNotNone(spec.benchmark_script_fn)
        self.assertIsNotNone(spec.grid_generator_fn)


class TestV2ShapeBucketsMatchV1(unittest.TestCase):
    """test_v2_shape_buckets_match_v1: v2 covers the same bucket names as v1."""

    def test_v2_shape_buckets_match_v1(self):
        from research_engine.triton_attention import ATTENTION_SHAPE_BUCKETS as V1_BUCKETS

        v1_names = {s["name"] for s in V1_BUCKETS}
        v2_names = {s["name"] for s in ATTENTION_V2_SHAPE_BUCKETS}
        self.assertEqual(v1_names, v2_names,
                         f"v2 missing: {v1_names - v2_names}, extra: {v2_names - v1_names}")

    def test_v2_bucket_routing_matches_v1(self):
        """Same shapes route to the same bucket names in both v1 and v2."""
        from research_engine.triton_attention import attention_shape_bucket_key as v1_key

        for shape in ATTENTION_V2_SHAPE_BUCKETS:
            with self.subTest(shape=shape["name"]):
                self.assertEqual(
                    attention_v2_shape_bucket_key(shape),
                    v1_key(shape),
                    f"Bucket mismatch for shape {shape['name']}",
                )


class TestV2BenchmarkScriptCompiles(unittest.TestCase):
    """test_v2_benchmark_script_compiles: generated script is valid Python."""

    def test_v2_benchmark_script_compiles(self):
        configs = ATTENTION_V2_CURATED_CONFIGS[:2]
        shapes = ATTENTION_V2_SHAPE_BUCKETS[:2]
        script = generate_attention_v2_benchmark_script(configs, shapes)
        self.assertIsInstance(script, str)
        self.assertTrue(len(script) > 100)
        # Must parse as valid Python (no syntax errors)
        try:
            ast.parse(script)
        except SyntaxError as e:
            self.fail(f"Generated benchmark script has syntax error: {e}")

    def test_v2_benchmark_script_contains_operator_name(self):
        configs = ATTENTION_V2_CURATED_CONFIGS[:1]
        shapes = ATTENTION_V2_SHAPE_BUCKETS[:1]
        script = generate_attention_v2_benchmark_script(configs, shapes)
        self.assertIn('"attention_v2"', script)


class TestV2HasSlidingWindowSupport(unittest.TestCase):
    """test_v2_has_sliding_window_support: kernel and launcher support WINDOW_SIZE."""

    def test_v2_has_sliding_window_support(self):
        configs = ATTENTION_V2_CURATED_CONFIGS[:1]
        shapes = [s for s in ATTENTION_V2_SHAPE_BUCKETS if s.get("window_size", -1) > 0]
        self.assertTrue(len(shapes) > 0, "No sliding-window shapes found")
        script = generate_attention_v2_benchmark_script(configs, shapes[:1])
        self.assertIn("WINDOW_SIZE", script)

    def test_v2_window_shapes_in_buckets(self):
        """Sliding-window shapes exist and route correctly."""
        window_shapes = [s for s in ATTENTION_V2_SHAPE_BUCKETS
                         if s.get("window_size", -1) > 0]
        self.assertGreaterEqual(len(window_shapes), 3)
        for shape in window_shapes:
            bucket = attention_v2_shape_bucket_key(shape)
            self.assertEqual(bucket, shape["name"],
                             f"Window shape {shape['name']} routed to {bucket}")

    def test_v2_launcher_accepts_window_size(self):
        """flash_attn_v2 function signature accepts window_size parameter."""
        import inspect
        sig = inspect.signature(flash_attn_v2)
        self.assertIn("window_size", sig.parameters)


class TestV2HasGQASupport(unittest.TestCase):
    """test_v2_has_gqa_support: kernel supports grouped-query attention."""

    def test_v2_has_gqa_support(self):
        configs = ATTENTION_V2_CURATED_CONFIGS[:1]
        gqa_shapes = [s for s in ATTENTION_V2_SHAPE_BUCKETS
                       if s.get("num_kv_heads", s["heads"]) < s["heads"]]
        self.assertTrue(len(gqa_shapes) > 0, "No GQA shapes found")
        script = generate_attention_v2_benchmark_script(configs, gqa_shapes[:1])
        self.assertIn("GROUP_SIZE", script)
        self.assertIn("NUM_KV_HEADS", script)

    def test_v2_gqa_shapes_route_correctly(self):
        """GQA shapes route to GQA-specific buckets."""
        gqa_shapes = [s for s in ATTENTION_V2_SHAPE_BUCKETS
                       if s.get("num_kv_heads", s["heads"]) < s["heads"]]
        self.assertGreaterEqual(len(gqa_shapes), 4)
        for shape in gqa_shapes:
            bucket = attention_v2_shape_bucket_key(shape)
            self.assertEqual(bucket, shape["name"],
                             f"GQA shape {shape['name']} routed to {bucket}")

    def test_v2_launcher_accepts_num_kv_heads(self):
        """flash_attn_v2 function signature accepts num_kv_heads parameter."""
        import inspect
        sig = inspect.signature(flash_attn_v2)
        self.assertIn("num_kv_heads", sig.parameters)

    def test_v2_launcher_accepts_shared_kv(self):
        """flash_attn_v2 supports YOCO shared_kv parameter."""
        import inspect
        sig = inspect.signature(flash_attn_v2)
        self.assertIn("shared_kv", sig.parameters)


class TestV2GridGenerator(unittest.TestCase):
    """Additional coverage for grid generation and config IDs."""

    def test_grid_includes_curated(self):
        grid = generate_attention_v2_grid(include_curated=True, max_configs=50)
        curated_ids = {attention_v2_config_id(c) for c in ATTENTION_V2_CURATED_CONFIGS}
        grid_ids = {attention_v2_config_id(c) for c in grid}
        self.assertTrue(curated_ids.issubset(grid_ids))

    def test_grid_respects_max_configs(self):
        grid = generate_attention_v2_grid(include_curated=False, max_configs=5)
        self.assertLessEqual(len(grid), 5)

    def test_config_id_format(self):
        config = {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 3}
        cid = attention_v2_config_id(config)
        self.assertEqual(cid, "m64_n32_w4_s3")


if __name__ == "__main__":
    unittest.main()
