from __future__ import annotations

import ast
import unittest

from tests import _pathfix  # noqa: F401

from research_engine.triton_kv_quant_write import (
    KV_QUANT_WRITE_CURATED_CONFIGS,
    KV_QUANT_WRITE_SHAPE_BUCKETS,
    generate_kv_quant_write_benchmark_script,
    generate_kv_quant_write_grid,
    kv_quant_write_config_id,
    kv_quant_write_shape_bucket_key,
    kv_quantize_separated,
)


class KvQuantWriteTests(unittest.TestCase):
    def test_config_id(self) -> None:
        self.assertEqual(
            kv_quant_write_config_id({"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 2}),
            "bs256_w4_s2",
        )

    def test_shape_bucket_key(self) -> None:
        self.assertEqual(
            kv_quant_write_shape_bucket_key({"name": "b1_kv4_d512_t1", "rows": 4, "head_dim": 512}),
            "b1_kv4_d512_t1",
        )
        self.assertEqual(
            kv_quant_write_shape_bucket_key({"rows": 128, "head_dim": 256}),
            "b1_kv16_d256_t8",
        )

    def test_grid_contains_curated(self) -> None:
        grid = generate_kv_quant_write_grid(include_curated=True, max_configs=24)
        curated_ids = {kv_quant_write_config_id(c) for c in KV_QUANT_WRITE_CURATED_CONFIGS}
        grid_ids = {kv_quant_write_config_id(c) for c in grid}
        self.assertTrue(curated_ids.issubset(grid_ids))

    def test_benchmark_script_compiles(self) -> None:
        script = generate_kv_quant_write_benchmark_script(
            KV_QUANT_WRITE_CURATED_CONFIGS[:2],
            KV_QUANT_WRITE_SHAPE_BUCKETS[:2],
        )
        ast.parse(script)
        self.assertIn("kv_quant_write_kernel", script)
        self.assertIn("separated", script)

    def test_separated_quantization_shapes(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        x = torch.randn((8, 256), dtype=torch.float16)
        q, s = kv_quantize_separated(x)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(s.shape, (8,))


if __name__ == "__main__":
    unittest.main()
