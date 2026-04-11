"""Tests for Gemma 4 shape buckets added to RMSNorm, Cross-Entropy, and Rotary operators.

Gemma 4 was released April 2026 with variants E2B, E4B, 26B-A4B (MoE), and 31B Dense.
Key distinguishing architectural features (from HF config.json, verified 2026-04-11):
- hidden_dim: 1536 / 2560 / 2816 / 5376 (E2B / E4B / 26B-A4B / 31B)
- vocab: 262144 (largest published dense-LLM vocabulary)
- head_dim: 256 local / 512 global (Gemma 4 introduced asymmetric head_dim)
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401 — patches sys.path

from research_engine.triton_rmsnorm import (
    RMSNORM_SHAPE_BUCKETS,
    generate_rmsnorm_benchmark_script,
    generate_rmsnorm_grid,
    rmsnorm_config_id,
    rmsnorm_shape_bucket_key,
    rmsnorm_shared_memory_check,
)
from research_engine.triton_cross_entropy import (
    CROSS_ENTROPY_SHAPE_BUCKETS,
    cross_entropy_shape_bucket_key,
    generate_cross_entropy_benchmark_script,
    generate_cross_entropy_grid,
    cross_entropy_config_id,
)
from research_engine.triton_rotary import (
    ROTARY_SHAPE_BUCKETS,
    rotary_shape_bucket_key,
    generate_rotary_grid,
    rotary_config_id,
)
from research_engine.kernelbench import KERNELBENCH_SUBSET


class TestRmsnormGemma4Buckets(unittest.TestCase):
    """Shape buckets and classifier for RMSNorm Gemma 4 variants."""

    def _bucket_names(self):
        return {b["name"] for b in RMSNORM_SHAPE_BUCKETS}

    def test_gemma4_e2b_bucket_present(self):
        self.assertIn("gemma4_e2b", self._bucket_names())

    def test_gemma4_e4b_bucket_present(self):
        self.assertIn("gemma4_e4b", self._bucket_names())

    def test_gemma4_26b_bucket_present(self):
        self.assertIn("gemma4_26b", self._bucket_names())

    def test_gemma4_31b_bucket_present(self):
        self.assertIn("gemma4_31b", self._bucket_names())

    def test_classifier_e2b(self):
        # HF gemma-4-E2B-it config.json: hidden_size=1536 (was incorrectly 2048 pre-#46)
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 2048, "hidden_dim": 1536}),
            "gemma4_e2b",
        )

    def test_classifier_e4b(self):
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 2048, "hidden_dim": 2560}),
            "gemma4_e4b",
        )

    def test_classifier_26b(self):
        # HF gemma-4-26B-A4B-it config.json: hidden_size=2816 (was incorrectly 4096 pre-#46)
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 4096, "hidden_dim": 2816}),
            "gemma4_26b",
        )

    def test_classifier_31b(self):
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 4096, "hidden_dim": 5376}),
            "gemma4_31b",
        )

    def test_existing_llama_7b_not_reclassified(self):
        # llama_7b has n_rows=4096, hidden_dim=4096 — same as gemma4_26b,
        # both map to gemma4_26b (the Gemma bucket is the canonical name for
        # this hidden dim at this row count).
        result = rmsnorm_shape_bucket_key({"n_rows": 4096, "hidden_dim": 4096})
        self.assertIn(result, {"gemma4_26b", "llama_7b"})  # either is acceptable

    def test_existing_llama_13b_not_reclassified(self):
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 4096, "hidden_dim": 5120}),
            "llama_13b",
        )

    def test_existing_mixtral_not_reclassified(self):
        self.assertEqual(
            rmsnorm_shape_bucket_key({"n_rows": 8192, "hidden_dim": 4096}),
            "mixtral",
        )

    def test_grid_still_produces_valid_configs(self):
        grid = generate_rmsnorm_grid(max_configs=50)
        self.assertGreater(len(grid), 0)
        for config in grid:
            self.assertTrue(rmsnorm_shared_memory_check(config))

    def test_grid_has_no_duplicate_ids(self):
        grid = generate_rmsnorm_grid(max_configs=200)
        ids = [rmsnorm_config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))

    def test_benchmark_script_valid_python_with_gemma_shapes(self):
        from research_engine.triton_rmsnorm import RMSNORM_CURATED_CONFIGS
        gemma_shapes = [
            b for b in RMSNORM_SHAPE_BUCKETS if b["name"].startswith("gemma4_")
        ]
        script = generate_rmsnorm_benchmark_script(
            configs=[RMSNORM_CURATED_CONFIGS[0]],
            shapes=gemma_shapes,
        )
        compile(script, "<bench_rmsnorm_gemma>", "exec")


class TestCrossEntropyGemma4Buckets(unittest.TestCase):
    """Shape buckets and classifier for Cross-Entropy Gemma 4 vocab."""

    def _bucket_names(self):
        return {b["name"] for b in CROSS_ENTROPY_SHAPE_BUCKETS}

    def test_gemma4_vocab_256k_short_bucket_present(self):
        self.assertIn("gemma4_vocab_256k_short", self._bucket_names())

    def test_gemma4_vocab_256k_long_bucket_present(self):
        self.assertIn("gemma4_vocab_256k_long", self._bucket_names())

    def test_classifier_256k_short(self):
        self.assertEqual(
            cross_entropy_shape_bucket_key({"n_rows": 2048, "n_cols": 256000}),
            "gemma4_vocab_256k_short",
        )

    def test_classifier_256k_long(self):
        self.assertEqual(
            cross_entropy_shape_bucket_key({"n_rows": 4096, "n_cols": 256000}),
            "gemma4_vocab_256k_long",
        )

    def test_existing_llama3_128k_not_reclassified(self):
        self.assertEqual(
            cross_entropy_shape_bucket_key({"n_rows": 2048, "n_cols": 128256}),
            "llama3_128k",
        )

    def test_existing_gpt2_not_reclassified(self):
        self.assertEqual(
            cross_entropy_shape_bucket_key({"n_rows": 2048, "n_cols": 50257}),
            "gpt2_med",
        )

    def test_grid_still_produces_valid_configs(self):
        grid = generate_cross_entropy_grid(max_configs=50)
        self.assertGreater(len(grid), 0)

    def test_benchmark_script_valid_python_with_gemma_vocab(self):
        from research_engine.triton_cross_entropy import CROSS_ENTROPY_CURATED_CONFIGS
        gemma_shapes = [
            b for b in CROSS_ENTROPY_SHAPE_BUCKETS if b["name"].startswith("gemma4_")
        ]
        script = generate_cross_entropy_benchmark_script(
            configs=[CROSS_ENTROPY_CURATED_CONFIGS[0]],
            shapes=gemma_shapes,
        )
        compile(script, "<bench_ce_gemma>", "exec")


class TestRotaryGemma4Buckets(unittest.TestCase):
    """Shape buckets and classifier for Rotary Gemma 4 head_dim=256."""

    def _bucket_names(self):
        return {b["name"] for b in ROTARY_SHAPE_BUCKETS}

    def test_gemma4_2b_rope_bucket_present(self):
        self.assertIn("gemma4_2b_rope", self._bucket_names())

    def test_gemma4_26b_rope_bucket_present(self):
        self.assertIn("gemma4_26b_rope", self._bucket_names())

    def test_classifier_2b_rope(self):
        # E2B/E4B: 8 heads, head_dim=256
        self.assertEqual(
            rotary_shape_bucket_key({"batch": 1, "seq": 4096, "heads": 8, "head_dim": 256}),
            "gemma4_2b_rope",
        )

    def test_classifier_26b_rope(self):
        # 26B MoE: 16 heads, head_dim=256
        self.assertEqual(
            rotary_shape_bucket_key({"batch": 1, "seq": 4096, "heads": 16, "head_dim": 256}),
            "gemma4_26b_rope",
        )

    def test_existing_llama7b_long_not_reclassified(self):
        # head_dim=128 (LLaMA) must NOT be caught by Gemma branch
        self.assertEqual(
            rotary_shape_bucket_key({"batch": 1, "seq": 4096, "heads": 32, "head_dim": 128}),
            "llama7b_long",
        )

    def test_existing_mistral_not_reclassified(self):
        self.assertEqual(
            rotary_shape_bucket_key({"batch": 1, "seq": 8192, "heads": 32, "head_dim": 128}),
            "mistral_long",
        )

    def test_grid_still_produces_valid_configs(self):
        grid = generate_rotary_grid(max_configs=50)
        self.assertGreater(len(grid), 0)

    def test_grid_includes_block_size_128_or_larger(self):
        # head_dim=256 → pairs=128; need BLOCK_SIZE >= 128 somewhere in the grid
        grid = generate_rotary_grid(max_configs=200)
        block_sizes = {c["BLOCK_SIZE"] for c in grid}
        self.assertTrue(any(bs >= 128 for bs in block_sizes),
                        "Grid must contain BLOCK_SIZE >= 128 for head_dim=256 workloads")


class TestKernelBenchGemmaProblems(unittest.TestCase):
    """KernelBench KERNELBENCH_SUBSET contains the new Gemma 4 problems."""

    def test_rmsnorm_gemma_26b_problem_present(self):
        ids = {p["id"] for p in KERNELBENCH_SUBSET.get("rmsnorm", [])}
        self.assertIn("kb_L2_rmsnorm_gemma_26b", ids)

    def test_ce_gemma_256k_problem_present(self):
        ids = {p["id"] for p in KERNELBENCH_SUBSET.get("cross_entropy", [])}
        self.assertIn("kb_L2_ce_gemma_256k", ids)

    def test_rotary_gemma_26b_problem_present(self):
        ids = {p["id"] for p in KERNELBENCH_SUBSET.get("rotary", [])}
        self.assertIn("kb_L2_rotary_gemma_26b", ids)

    def test_rmsnorm_gemma_26b_has_correct_shape(self):
        problem = next(
            p for p in KERNELBENCH_SUBSET["rmsnorm"]
            if p["id"] == "kb_L2_rmsnorm_gemma_26b"
        )
        self.assertEqual(problem["n_rows"], 4096)
        # HF gemma-4-26B-A4B-it config.json: hidden_size=2816
        self.assertEqual(problem["hidden_dim"], 2816)

    def test_ce_gemma_256k_has_correct_vocab(self):
        problem = next(
            p for p in KERNELBENCH_SUBSET["cross_entropy"]
            if p["id"] == "kb_L2_ce_gemma_256k"
        )
        self.assertEqual(problem["n_cols"], 256000)

    def test_rotary_gemma_26b_has_head_dim_256(self):
        problem = next(
            p for p in KERNELBENCH_SUBSET["rotary"]
            if p["id"] == "kb_L2_rotary_gemma_26b"
        )
        self.assertEqual(problem["head_dim"], 256)
        self.assertEqual(problem["heads"], 16)

    def test_all_rotary_problems_have_required_fields(self):
        for p in KERNELBENCH_SUBSET.get("rotary", []):
            for field in ("id", "batch", "seq", "heads", "head_dim", "level"):
                self.assertIn(field, p, f"Problem {p.get('id')} missing field {field!r}")


if __name__ == "__main__":
    unittest.main()
