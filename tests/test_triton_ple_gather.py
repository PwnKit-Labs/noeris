"""Tests for the PLE gather+add Triton operator.

Gemma 4 E2B/E4B per-layer embedding kernel: gathers from a
(vocab_size, num_layers, ple_dim) table and adds into the residual stream.
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_operators import REGISTRY
from research_engine.triton_ple_gather import (
    PLE_GATHER_CURATED_CONFIGS,
    PLE_GATHER_SHAPE_BUCKETS,
    generate_ple_gather_benchmark_script,
    generate_ple_gather_grid,
    ple_gather_config_id,
    ple_gather_shape_bucket_key,
)


class TestRegistration(unittest.TestCase):
    def test_spec_registration(self) -> None:
        """ple_gather must be discoverable via the shared REGISTRY."""
        self.assertIn("ple_gather", REGISTRY.names())

    def test_spec_metric_is_gb_per_s(self) -> None:
        spec = REGISTRY.get("ple_gather")
        self.assertEqual(spec.metric_name, "gb_per_s")

    def test_spec_has_curated_configs(self) -> None:
        spec = REGISTRY.get("ple_gather")
        self.assertGreaterEqual(len(spec.curated_configs), 5)

    def test_spec_has_two_gemma4_buckets(self) -> None:
        spec = REGISTRY.get("ple_gather")
        self.assertEqual(len(spec.shape_buckets), 2)


class TestShapeBucketKey(unittest.TestCase):
    def test_bucket_key_e2b(self) -> None:
        shape = {"batch": 1, "seq_len": 4096, "hidden_dim": 1536, "ple_dim": 256, "vocab_size": 262144, "num_layers": 35}
        self.assertEqual(ple_gather_shape_bucket_key(shape), "gemma4_e2b_ple")

    def test_bucket_key_e4b(self) -> None:
        shape = {"batch": 1, "seq_len": 4096, "hidden_dim": 2560, "ple_dim": 256, "vocab_size": 262144, "num_layers": 42}
        self.assertEqual(ple_gather_shape_bucket_key(shape), "gemma4_e4b_ple")

    def test_all_shape_buckets_are_reachable(self) -> None:
        bucket_names = {b["name"] for b in PLE_GATHER_SHAPE_BUCKETS}
        hit = {ple_gather_shape_bucket_key(b) for b in PLE_GATHER_SHAPE_BUCKETS}
        self.assertEqual(bucket_names, hit)


class TestGridGeneration(unittest.TestCase):
    def test_grid_generation_is_non_empty(self) -> None:
        grid = generate_ple_gather_grid()
        self.assertGreater(len(grid), 0)

    def test_grid_respects_max_configs(self) -> None:
        grid = generate_ple_gather_grid(max_configs=10)
        self.assertLessEqual(len(grid), 10)

    def test_grid_has_no_duplicate_ids(self) -> None:
        grid = generate_ple_gather_grid(max_configs=200)
        ids = [ple_gather_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))


class TestBenchmarkScript(unittest.TestCase):
    def _make_script(self) -> str:
        return generate_ple_gather_benchmark_script(
            configs=[PLE_GATHER_CURATED_CONFIGS[0]],
            shapes=[PLE_GATHER_SHAPE_BUCKETS[0]],
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
            "ple_gather_kernel",
            "apply_ple_gather",
            "torch_ple_gather",
            "benchmark_one",
            "main",
        ):
            self.assertIn(fn, script, f"missing {fn}")

    def test_script_handles_ple_dim_add(self) -> None:
        """The kernel must add PLE into the first ple_dim channels only."""
        script = self._make_script()
        # The kernel adds ple_val into the residual for the first ple_dim dims
        self.assertIn("ple_dim", script)
        self.assertIn("res_ple + ple_val", script)

    def test_script_uses_gb_per_s_metric(self) -> None:
        script = self._make_script()
        self.assertIn("gb_per_s", script)


if __name__ == "__main__":
    unittest.main()
