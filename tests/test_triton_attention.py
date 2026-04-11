"""Tests for triton_attention.py — sliding-window extension.

Tests cover:
- Sliding-window mask logic (pure Python, no torch dependency)
- window_size=-1 behaves identically to no-window (regression)
- Causal + window interaction (pure logic)
- Tile-pruning math: k_start/k_end range sanity
- Shape bucket key classifier puts window shapes in window buckets
- Grid generator configs don't violate shared-memory limit
- KernelBench problem list includes the 2 new window problems
- Benchmark script is valid Python and embeds WINDOW_SIZE
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tests import _pathfix  # noqa: F401

from research_engine.triton_attention import (
    ATTENTION_SHAPE_BUCKETS,
    attention_shape_bucket_key,
    attention_shared_memory_check,
    generate_attention_benchmark_script,
    generate_attention_grid,
)
from research_engine.kernelbench import KERNELBENCH_SUBSET


# ---------------------------------------------------------------------------
# Pure-Python sliding-window mask logic (mirrors make_sliding_window_mask)
# without any torch dependency so tests run locally too.
# ---------------------------------------------------------------------------

def _mask_entry(i: int, j: int, window_size: int, is_causal: bool) -> bool:
    """Return True if query i attends to key j under sliding-window rules."""
    left_bound = j >= (i - window_size + 1)
    if is_causal:
        return left_bound and j <= i
    return left_bound and (j - i) < window_size


# ---------------------------------------------------------------------------
# Sliding-window mask semantics (pure Python)
# ---------------------------------------------------------------------------

class SlidingWindowMaskLogicTests(unittest.TestCase):
    def test_causal_no_future_keys(self):
        """No query should attend to a key in the future (j > i) under causal masking."""
        for seq in [6, 8, 16]:
            for window in [2, 4, seq]:
                for i in range(seq):
                    for j in range(i + 1, seq):
                        self.assertFalse(
                            _mask_entry(i, j, window, is_causal=True),
                            f"seq={seq} window={window}: mask[{i},{j}] should be False (future)",
                        )

    def test_causal_window_enforces_left_boundary(self):
        """Query i must not attend to keys more than window_size-1 positions back."""
        window = 4
        seq = 12
        for i in range(seq):
            for j in range(max(0, i - window)):  # j < i - window + 1
                self.assertFalse(
                    _mask_entry(i, j, window, is_causal=True),
                    f"mask[{i},{j}] should be False (outside window)",
                )

    def test_causal_window_attends_within_window(self):
        """Every key j in [i - W + 1, i] should be attended to."""
        window = 4
        seq = 12
        for i in range(seq):
            for j in range(max(0, i - window + 1), i + 1):
                self.assertTrue(
                    _mask_entry(i, j, window, is_causal=True),
                    f"mask[{i},{j}] should be True (within window)",
                )

    def test_window_1_is_self_attention_only(self):
        """Window of 1 with causal masking → only self-attention (diagonal)."""
        seq = 8
        for i in range(seq):
            for j in range(seq):
                expected = i == j
                self.assertEqual(_mask_entry(i, j, 1, is_causal=True), expected)

    def test_window_eq_seq_is_full_causal(self):
        """Window equal to sequence length → standard lower-triangular causal mask."""
        seq = 10
        for i in range(seq):
            for j in range(seq):
                expected = j <= i
                self.assertEqual(_mask_entry(i, j, seq, is_causal=True), expected)

    def test_noncausal_window_is_symmetric(self):
        """Non-causal sliding window mask is symmetric: mask[i,j] == mask[j,i]."""
        seq, window = 8, 3
        for i in range(seq):
            for j in range(seq):
                self.assertEqual(
                    _mask_entry(i, j, window, is_causal=False),
                    _mask_entry(j, i, window, is_causal=False),
                    f"mask[{i},{j}] != mask[{j},{i}]",
                )

    def test_noncausal_window_no_out_of_range_keys(self):
        """Non-causal: key j more than window_size-1 from query i is not attended."""
        seq, window = 10, 3
        for i in range(seq):
            for j in range(seq):
                dist = abs(j - i)
                if dist >= window:
                    self.assertFalse(
                        _mask_entry(i, j, window, is_causal=False),
                        f"mask[{i},{j}] should be False (dist={dist} >= window={window})",
                    )


# ---------------------------------------------------------------------------
# window_size=-1 regression (pure Python logic)
# ---------------------------------------------------------------------------

class WindowSizeMinusOneRegressionTests(unittest.TestCase):
    def test_shape_bucket_no_window_key(self):
        """Shapes without window_size key use standard (non-window) buckets."""
        shape = {"seq_len": 4096, "head_dim": 128, "heads": 16}
        bucket = attention_shape_bucket_key(shape)
        self.assertEqual(bucket, "long_128")

    def test_shape_bucket_window_minus1(self):
        """Shapes with window_size=-1 use standard buckets (same as absent)."""
        shape = {"seq_len": 4096, "head_dim": 128, "heads": 16, "window_size": -1}
        bucket = attention_shape_bucket_key(shape)
        self.assertEqual(bucket, "long_128")

    def test_shape_bucket_no_window_short(self):
        shape = {"seq_len": 1024, "head_dim": 128, "heads": 16}
        self.assertEqual(attention_shape_bucket_key(shape), "short_128")

    def test_shape_bucket_no_window_mistral(self):
        shape = {"seq_len": 8192, "head_dim": 128, "heads": 32}
        self.assertEqual(attention_shape_bucket_key(shape), "mistral")


# ---------------------------------------------------------------------------
# Causal + window interaction (pure Python)
# ---------------------------------------------------------------------------

class CausalWindowInteractionTests(unittest.TestCase):
    def test_causal_window_upper_triangle_false(self):
        """Under causal + window, the entire upper triangle is always False."""
        seq, window = 10, 5
        for i in range(seq):
            for j in range(i + 1, seq):
                self.assertFalse(_mask_entry(i, j, window, is_causal=True))

    def test_causal_window_diagonal_always_true(self):
        """Every query should at least attend to itself (window >= 1)."""
        seq, window = 10, 4
        for i in range(seq):
            self.assertTrue(_mask_entry(i, i, window, is_causal=True))

    def test_causal_window_coverage_count(self):
        """Number of True entries per row equals min(i+1, window_size)."""
        seq, window = 12, 4
        for i in range(seq):
            count = sum(1 for j in range(seq) if _mask_entry(i, j, window, is_causal=True))
            expected = min(i + 1, window)
            self.assertEqual(count, expected, f"Row {i}: got {count} True entries, expected {expected}")


# ---------------------------------------------------------------------------
# Tile-pruning math sanity checks (pure Python)
# ---------------------------------------------------------------------------

class TilePruningMathTests(unittest.TestCase):
    """Verify the tile-pruning range math in the kernel comments is correct."""

    def _k_range(self, pid, BLOCK_M, N, W, is_causal):
        """Mimic the kernel's k_start / k_end computation."""
        if W > 0:
            k_start = max(0, pid * BLOCK_M - W + 1)
            if is_causal:
                k_end = min(N, (pid + 1) * BLOCK_M)
            else:
                k_end = min(N, pid * BLOCK_M + W)
        else:
            k_start = 0
            k_end = min(N, (pid + 1) * BLOCK_M) if is_causal else N
        return k_start, k_end

    def test_first_tile_starts_at_zero(self):
        k_start, k_end = self._k_range(0, 64, 4096, 1024, is_causal=True)
        self.assertEqual(k_start, 0)
        self.assertEqual(k_end, 64)

    def test_middle_tile_window_limits_start(self):
        # pid=32, BLOCK_M=64 → query rows 2048..2111
        # W=1024: k_start = max(0, 2048 - 1024 + 1) = 1025
        k_start, k_end = self._k_range(32, 64, 4096, 1024, is_causal=True)
        self.assertEqual(k_start, 2048 - 1024 + 1)
        self.assertEqual(k_end, min(4096, 33 * 64))

    def test_range_always_in_bounds(self):
        """k_start/k_end must always be in [0, N]."""
        N, W, BLOCK_M = 8192, 1024, 128
        for pid in range(N // BLOCK_M):
            k_start, k_end = self._k_range(pid, BLOCK_M, N, W, is_causal=True)
            self.assertGreaterEqual(k_start, 0)
            self.assertLessEqual(k_end, N)
            self.assertLessEqual(k_start, k_end)

    def test_window_reduces_tile_count_vs_full_causal(self):
        """With a window, fewer K-tiles are visited than in full causal mode."""
        N, W, BLOCK_M, BLOCK_N = 4096, 1024, 64, 64
        total_window_tiles = 0
        total_causal_tiles = 0
        for pid in range(N // BLOCK_M):
            k_start_w, k_end_w = self._k_range(pid, BLOCK_M, N, W, is_causal=True)
            k_start_aligned = (k_start_w // BLOCK_N) * BLOCK_N
            for start_n in range(k_start_aligned, k_end_w, BLOCK_N):
                if start_n + BLOCK_N > k_start_w:
                    total_window_tiles += 1
            k_end_c = min(N, (pid + 1) * BLOCK_M)
            total_causal_tiles += max(0, k_end_c // BLOCK_N)
        self.assertLess(total_window_tiles, total_causal_tiles)

    def test_no_window_causal_matches_diagonal(self):
        """With W=-1 (no window) the loop ends at the diagonal — standard causal."""
        N, BLOCK_M = 1024, 64
        for pid in range(N // BLOCK_M):
            k_start, k_end = self._k_range(pid, BLOCK_M, N, -1, is_causal=True)
            self.assertEqual(k_start, 0)
            self.assertEqual(k_end, min(N, (pid + 1) * BLOCK_M))


# ---------------------------------------------------------------------------
# Shape bucket classifier for window shapes
# ---------------------------------------------------------------------------

class WindowShapeBucketTests(unittest.TestCase):
    def test_gemma4_local_1024_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 16, "window_size": 1024}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_local_1024")

    def test_gemma4_local_short_bucket(self):
        shape = {"seq_len": 2048, "head_dim": 128, "heads": 16, "window_size": 1024}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_local_short")

    def test_gemma3_local_bucket(self):
        shape = {"seq_len": 8192, "head_dim": 128, "heads": 16, "window_size": 1024}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma3_local")

    def test_window_shapes_present_in_buckets_list(self):
        """All 3 Gemma local shapes appear in ATTENTION_SHAPE_BUCKETS."""
        names = {s["name"] for s in ATTENTION_SHAPE_BUCKETS}
        self.assertIn("gemma4_local_1024", names)
        self.assertIn("gemma4_local_short", names)
        self.assertIn("gemma3_local", names)

    def test_window_shape_buckets_have_window_size(self):
        """All window shapes in ATTENTION_SHAPE_BUCKETS have window_size > 0."""
        for shape in ATTENTION_SHAPE_BUCKETS:
            if "window" in shape.get("name", ""):
                ws = shape.get("window_size", -1)
                self.assertGreater(ws, 0, f"Shape {shape['name']} missing window_size")


# ---------------------------------------------------------------------------
# Grid generator + shared memory check
# ---------------------------------------------------------------------------

class GridSharedMemoryTests(unittest.TestCase):
    def test_all_grid_configs_pass_shmem(self):
        """Every config produced by the grid generator passes the shmem limit."""
        configs = generate_attention_grid(include_curated=False, max_configs=50)
        for cfg in configs:
            self.assertTrue(
                attention_shared_memory_check(cfg),
                f"Config {cfg} exceeds shmem limit"
            )

    def test_curated_configs_have_valid_structure(self):
        """All curated configs have the required keys."""
        from research_engine.triton_attention import ATTENTION_CURATED_CONFIGS
        for cfg in ATTENTION_CURATED_CONFIGS:
            for key in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages"):
                self.assertIn(key, cfg, f"Curated config missing key {key}: {cfg}")

    def test_large_num_stages_fails_shmem(self):
        """A pathologically large config should fail the shmem check."""
        bad = {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 4}
        self.assertFalse(attention_shared_memory_check(bad))


# ---------------------------------------------------------------------------
# KernelBench problem list
# ---------------------------------------------------------------------------

class KernelBenchNewProblemsTests(unittest.TestCase):
    def _attn_ids(self):
        return {p["id"] for p in KERNELBENCH_SUBSET["attention"]}

    def test_gemma_local_problem_present(self):
        self.assertIn("kb_L3_attn_gemma_local", self._attn_ids())

    def test_gemma_slide_causal_problem_present(self):
        self.assertIn("kb_L3_attn_gemma_slide_causal", self._attn_ids())

    def test_gemma_local_has_correct_window_size(self):
        problems = {p["id"]: p for p in KERNELBENCH_SUBSET["attention"]}
        prob = problems["kb_L3_attn_gemma_local"]
        self.assertEqual(prob.get("window_size"), 1024)
        self.assertTrue(prob.get("is_causal"))
        self.assertEqual(prob.get("level"), 3)

    def test_gemma_slide_causal_has_correct_window_size(self):
        problems = {p["id"]: p for p in KERNELBENCH_SUBSET["attention"]}
        prob = problems["kb_L3_attn_gemma_slide_causal"]
        self.assertEqual(prob.get("window_size"), 1024)
        self.assertTrue(prob.get("is_causal"))
        self.assertEqual(prob.get("level"), 3)

    def test_existing_attn_problems_still_present(self):
        ids = self._attn_ids()
        self.assertIn("kb_L3_attn_llama7b_causal", ids)
        self.assertIn("kb_L2_attn_short_64", ids)


# ---------------------------------------------------------------------------
# Benchmark script validity
# ---------------------------------------------------------------------------

class BenchmarkScriptWindowTests(unittest.TestCase):
    def _make_script(self, shapes):
        configs = [{"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}]
        return generate_attention_benchmark_script(configs, shapes)

    def test_script_with_window_shape_is_valid_python(self):
        shapes = [
            {"name": "gemma4_local_1024", "batch": 1, "heads": 16,
             "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024}
        ]
        script = self._make_script(shapes)
        compile(script, "<benchmark_window>", "exec")

    def test_script_embeds_window_size_constexpr(self):
        shapes = [
            {"name": "test_window", "batch": 1, "heads": 8,
             "seq_len": 512, "head_dim": 64, "is_causal": True, "window_size": 256}
        ]
        script = self._make_script(shapes)
        self.assertIn("WINDOW_SIZE", script)
        self.assertIn("window_size", script)

    def test_script_without_window_is_valid_python(self):
        shapes = [
            {"name": "no_window", "batch": 1, "heads": 8,
             "seq_len": 512, "head_dim": 64, "is_causal": False}
        ]
        script = self._make_script(shapes)
        compile(script, "<benchmark_no_window>", "exec")

    def test_script_with_causal_only_is_valid_python(self):
        shapes = [
            {"name": "causal_only", "batch": 1, "heads": 16,
             "seq_len": 2048, "head_dim": 128, "is_causal": True}
        ]
        script = self._make_script(shapes)
        compile(script, "<benchmark_causal>", "exec")

    def test_script_contains_is_causal_constexpr(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("IS_CAUSAL", script)

    def test_script_contains_tile_pruning_logic(self):
        """The generated script should contain the tile-pruning k_start/k_end logic."""
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("k_start", script)
        self.assertIn("k_end", script)

    def test_script_contains_make_sliding_window_mask(self):
        """benchmark_one() uses make_sliding_window_mask for the PyTorch reference."""
        shapes = [
            {"name": "gemma_local", "batch": 1, "heads": 16,
             "seq_len": 2048, "head_dim": 128, "is_causal": True, "window_size": 1024}
        ]
        script = self._make_script(shapes)
        self.assertIn("make_sliding_window_mask", script)


if __name__ == "__main__":
    unittest.main()
