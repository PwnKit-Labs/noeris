"""Tests for the fused MoE router Triton operator (Gemma 4 26B-A4B).

vLLM today launches 4 separate kernels per MoE router: matmul, softmax,
topk, and renormalize. Noeris fuses them into a single Triton program per
BLOCK_M tokens. These tests exercise Python-level correctness (spec
registration, shape buckets, grid generation, and the generated benchmark
script compiles and contains the expected machinery).
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_moe_router import (
    MOE_ROUTER_CURATED_CONFIGS,
    MOE_ROUTER_SHAPE_BUCKETS,
    generate_moe_router_benchmark_script,
    generate_moe_router_grid,
    moe_router_config_id,
    moe_router_shape_bucket_key,
    moe_router_shared_memory_check,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """moe_router must be discoverable via the shared REGISTRY."""
        self.assertIn("moe_router", REGISTRY.names())

    def test_spec_metric_is_tflops(self) -> None:
        spec = REGISTRY.get("moe_router")
        self.assertEqual(spec.metric_name, "tflops")

    def test_spec_has_curated_configs(self) -> None:
        spec = REGISTRY.get("moe_router")
        self.assertGreaterEqual(len(spec.curated_configs), 5)


class TestConfigId(unittest.TestCase):
    def test_config_id_is_stable(self) -> None:
        config = {"BLOCK_M": 128, "num_warps": 4, "num_stages": 2}
        self.assertEqual(moe_router_config_id(config), "bm128_w4_s2")

    def test_all_curated_configs_have_unique_ids(self) -> None:
        ids = [moe_router_config_id(c) for c in MOE_ROUTER_CURATED_CONFIGS]
        self.assertEqual(len(ids), len(set(ids)))


class TestShapeBucketKey(unittest.TestCase):
    def test_shape_bucket_key_all_buckets(self) -> None:
        """Every declared shape bucket must route back to its own name."""
        for bucket in MOE_ROUTER_SHAPE_BUCKETS:
            self.assertEqual(
                moe_router_shape_bucket_key(bucket),
                bucket["name"],
                f"bucket {bucket['name']!r} did not route to itself",
            )

    def test_shape_bucket_key_small(self) -> None:
        shape = {"num_tokens": 512, "hidden_dim": 2816, "num_experts": 128, "top_k": 8}
        self.assertEqual(moe_router_shape_bucket_key(shape), "gemma4_26b_a4b_router_small")

    def test_shape_bucket_key_xlong(self) -> None:
        shape = {"num_tokens": 32768, "hidden_dim": 2816, "num_experts": 128, "top_k": 8}
        self.assertEqual(moe_router_shape_bucket_key(shape), "gemma4_26b_a4b_router_xlong")


class TestSharedMemoryCheck(unittest.TestCase):
    def test_curated_configs_shared_memory_permissive(self) -> None:
        """Learned-feasibility is on — the shmem check should accept all
        curated configs (and generally be permissive)."""
        for config in MOE_ROUTER_CURATED_CONFIGS:
            self.assertTrue(
                moe_router_shared_memory_check(config),
                f"shmem check rejected curated config {config}",
            )

    def test_large_block_passes(self) -> None:
        config = {"BLOCK_M": 256, "num_warps": 8, "num_stages": 3}
        self.assertTrue(moe_router_shared_memory_check(config))


class TestGridGeneration(unittest.TestCase):
    def test_grid_generation_non_empty(self) -> None:
        grid = generate_moe_router_grid()
        self.assertGreater(len(grid), 0)

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_moe_router_grid(max_configs=12)
        self.assertLessEqual(len(grid), 12)

    def test_grid_has_no_duplicate_ids(self) -> None:
        grid = generate_moe_router_grid(max_configs=200)
        ids = [moe_router_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_moe_router_benchmark_script(
            configs=[MOE_ROUTER_CURATED_CONFIGS[0]],
            shapes=[MOE_ROUTER_SHAPE_BUCKETS[0]],
        )

    def test_benchmark_script_compiles_as_python(self) -> None:
        script = self._make_script()
        try:
            compile(script, "<bench>", "exec")
        except SyntaxError as exc:
            self.fail(f"benchmark script has a syntax error: {exc}")

    def test_benchmark_script_has_required_functions(self) -> None:
        script = self._make_script()
        for fn in (
            "moe_router_kernel",
            "moe_router",
            "torch_moe_router",
            "separated_moe_router",
            "benchmark_one",
            "main",
        ):
            self.assertIn(fn, script, f"missing {fn}")

    def test_benchmark_script_has_topk_logic(self) -> None:
        script = self._make_script()
        self.assertIn("TOP_K", script)
        # iterative top-k via tl.argmax over the expert axis
        self.assertIn("tl.argmax", script)

    def test_benchmark_script_has_softmax(self) -> None:
        script = self._make_script()
        # numerically stable softmax: subtract max then exp
        self.assertIn("max_logit", script)
        self.assertIn("tl.exp", script)

    def test_benchmark_script_reports_fusion_speedup(self) -> None:
        script = self._make_script()
        self.assertIn("fusion_speedup", script)
        self.assertIn("separated_ms", script)


if __name__ == "__main__":
    unittest.main()
