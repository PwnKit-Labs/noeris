from __future__ import annotations

import ast
import unittest

from tests import _pathfix  # noqa: F401

from research_engine.triton_spec_decode_verify_accept import (
    VERIFY_ACCEPT_CURATED_CONFIGS,
    VERIFY_ACCEPT_SHAPE_BUCKETS,
    generate_verify_accept_benchmark_script,
    generate_verify_accept_grid,
    verify_accept_config_id,
    verify_accept_reference,
    verify_accept_shape_bucket_key,
)


class SpecDecodeVerifyAcceptTests(unittest.TestCase):
    def test_config_id(self) -> None:
        cid = verify_accept_config_id({"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 2})
        self.assertEqual(cid, "bs128_w2_s2")

    def test_shape_bucket_key(self) -> None:
        self.assertEqual(
            verify_accept_shape_bucket_key({"name": "draft16_vocab128k", "draft_len": 16, "vocab": 128000}),
            "draft16_vocab128k",
        )
        self.assertEqual(
            verify_accept_shape_bucket_key({"draft_len": 32, "vocab": 128000}),
            "draft32_vocab128k",
        )

    def test_generate_grid_includes_curated(self) -> None:
        grid = generate_verify_accept_grid(include_curated=True, max_configs=16)
        self.assertGreaterEqual(len(grid), len(VERIFY_ACCEPT_CURATED_CONFIGS))
        curated_ids = {verify_accept_config_id(c) for c in VERIFY_ACCEPT_CURATED_CONFIGS}
        grid_ids = {verify_accept_config_id(c) for c in grid}
        self.assertTrue(curated_ids.issubset(grid_ids))

    def test_reference_logic(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        target = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64)
        draft = torch.tensor([[1, 2, 9, 4], [5, 6, 7, 8]], dtype=torch.int64)
        accept_len, prefix = verify_accept_reference(target, draft)
        self.assertEqual(accept_len.tolist(), [2, 4])
        self.assertEqual(prefix.tolist(), [[True, True, False, False], [True, True, True, True]])

    def test_benchmark_script_compiles(self) -> None:
        script = generate_verify_accept_benchmark_script(
            VERIFY_ACCEPT_CURATED_CONFIGS[:2],
            VERIFY_ACCEPT_SHAPE_BUCKETS[:2],
        )
        ast.parse(script)
        self.assertIn("verify_accept_kernel", script)
        self.assertIn("verify_accept_fused", script)


if __name__ == "__main__":
    unittest.main()
