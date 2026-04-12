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

    def test_mha_shapes_still_route_to_mha_buckets(self):
        """GQA gate must not misroute MHA shapes (explicit or implicit)."""
        shape = {"seq_len": 4096, "head_dim": 128, "heads": 16}
        self.assertEqual(attention_shape_bucket_key(shape), "long_128")
        shape = {"seq_len": 4096, "head_dim": 128, "heads": 16, "num_kv_heads": 16}
        self.assertEqual(attention_shape_bucket_key(shape), "long_128")


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
    """Verify that grid generation no longer filters by a hand-coded shmem
    budget. Feasibility is now learned by the bandit from runtime failures
    (reward=0). The shmem-check function is retained as a no-op for
    backward compatibility with external callers."""

    def test_shmem_check_is_now_a_noop(self):
        """The check function returns True for every config — feasibility is
        learned from runtime, not enforced by a hand-coded formula."""
        for cfg in [
            {"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 1, "num_stages": 1},
            {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 4},
            {"BLOCK_M": 256, "BLOCK_N": 256, "num_warps": 8, "num_stages": 8},
        ]:
            self.assertTrue(attention_shared_memory_check(cfg))

    def test_grid_includes_large_tiles_previously_filtered(self):
        """The grid must now contain BLOCK_M=128, BLOCK_N=128, num_stages>=2
        configs that the old hardcoded head_dim=128 shmem check rejected."""
        configs = generate_attention_grid(include_curated=False, max_configs=500)
        large = [
            c for c in configs
            if c.get("BLOCK_M", 0) >= 128
            and c.get("BLOCK_N", 0) >= 128
            and c.get("num_stages", 0) >= 2
        ]
        self.assertGreater(
            len(large),
            1,
            "Expected multiple BLOCK_M>=128, BLOCK_N>=128 configs after the "
            "shmem filter was removed; got " + str(len(large)),
        )

    def test_grid_size_strictly_larger_without_filter(self):
        """As a sanity check, the post-refactor grid must contain every config
        that the old filter would have admitted, plus extras."""
        configs = generate_attention_grid(include_curated=False, max_configs=500)
        # Recompute what the old hardcoded filter would have admitted.
        def _old_shmem_ok(c):
            bm, bn, ns = c["BLOCK_M"], c["BLOCK_N"], c["num_stages"]
            shmem = (bm * 128 + 2 * bn * 128) * 2 * ns + 2048
            return shmem <= 192_000
        passes_old = sum(1 for c in configs if _old_shmem_ok(c))
        self.assertGreater(len(configs), passes_old)

    def test_curated_configs_have_valid_structure(self):
        """All curated configs have the required keys."""
        from research_engine.triton_attention import ATTENTION_CURATED_CONFIGS
        for cfg in ATTENTION_CURATED_CONFIGS:
            for key in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages"):
                self.assertIn(key, cfg, f"Curated config missing key {key}: {cfg}")


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



# ---------------------------------------------------------------------------
# QK-norm shape bucket routing
# ---------------------------------------------------------------------------

class QKNormShapeBucketTests(unittest.TestCase):
    """USE_QK_NORM=True shapes must route to dedicated gemma4_qknorm* buckets."""

    def test_qknorm_local_bucket(self):
        shape = {
            "seq_len": 4096, "head_dim": 256, "heads": 16,
            "window_size": 1024, "use_qk_norm": True,
        }
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_qknorm")

    def test_qknorm_global_bucket(self):
        shape = {
            "seq_len": 4096, "head_dim": 256, "heads": 16,
            "window_size": -1, "use_qk_norm": True,
        }
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_qknorm_global")

    def test_qknorm_global_bucket_no_window_key(self):
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 16, "use_qk_norm": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_qknorm_global")

    def test_no_qknorm_flag_uses_window_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 16, "window_size": 1024}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_local_1024")

    def test_qknorm_false_uses_window_bucket(self):
        shape = {
            "seq_len": 4096, "head_dim": 256, "heads": 16,
            "window_size": 1024, "use_qk_norm": False,
        }
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_local_1024")

    def test_gemma4_qknorm_buckets_present_in_shape_list(self):
        names = {s["name"] for s in ATTENTION_SHAPE_BUCKETS}
        self.assertIn("gemma4_qknorm", names)
        self.assertIn("gemma4_qknorm_global", names)

    def test_gemma4_qknorm_bucket_has_correct_fields(self):
        bucket = next(s for s in ATTENTION_SHAPE_BUCKETS if s["name"] == "gemma4_qknorm")
        self.assertEqual(bucket["head_dim"], 256)
        self.assertEqual(bucket["window_size"], 1024)
        self.assertTrue(bucket["use_qk_norm"])
        self.assertTrue(bucket["is_causal"])

    def test_gemma4_qknorm_global_bucket_has_correct_fields(self):
        bucket = next(s for s in ATTENTION_SHAPE_BUCKETS if s["name"] == "gemma4_qknorm_global")
        self.assertEqual(bucket["head_dim"], 256)
        self.assertEqual(bucket.get("window_size", -1), -1)
        self.assertTrue(bucket["use_qk_norm"])
        self.assertTrue(bucket["is_causal"])

    def test_use_qk_norm_false_regression_does_not_reroute(self):
        shapes_without_qknorm = [
            {"seq_len": 4096, "head_dim": 256, "heads": 16, "window_size": 1024},
            {"seq_len": 4096, "head_dim": 128, "heads": 16},
            {"seq_len": 8192, "head_dim": 128, "heads": 32},
        ]
        for shape in shapes_without_qknorm:
            key = attention_shape_bucket_key(shape)
            self.assertNotIn("qknorm", key, f"Shape {shape} unexpectedly routed to {key!r}")


# ---------------------------------------------------------------------------
# QK-norm benchmark script validity
# ---------------------------------------------------------------------------

class QKNormBenchmarkScriptTests(unittest.TestCase):

    def _make_script(self, shapes):
        configs = [{"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}]
        return generate_attention_benchmark_script(configs, shapes)

    def test_script_embeds_use_qk_norm_constexpr(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("USE_QK_NORM", script)

    def test_script_with_qknorm_shape_is_valid_python(self):
        shapes = [
            {"name": "gemma4_qknorm", "batch": 1, "heads": 16,
             "seq_len": 4096, "head_dim": 256, "is_causal": True,
             "window_size": 1024, "use_qk_norm": True}
        ]
        script = self._make_script(shapes)
        compile(script, "<benchmark_qknorm>", "exec")

    def test_script_with_global_qknorm_shape_is_valid_python(self):
        shapes = [
            {"name": "gemma4_qknorm_global", "batch": 1, "heads": 16,
             "seq_len": 4096, "head_dim": 256, "is_causal": True,
             "use_qk_norm": True}
        ]
        script = self._make_script(shapes)
        compile(script, "<benchmark_qknorm_global>", "exec")

    def test_script_contains_qknorm_rms_logic(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("q_rstd", script)
        self.assertIn("k_rstd", script)
        self.assertIn("1e-6", script)

    def test_script_benchmark_one_passes_use_qk_norm(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("use_qk_norm", script)

    def test_script_reference_uses_rms_norm_when_qknorm(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("rms_norm", script)

    def test_script_qknorm_false_produces_valid_python(self):
        shapes = [
            {"name": "no_norm", "batch": 1, "heads": 8,
             "seq_len": 512, "head_dim": 64, "use_qk_norm": False}
        ]
        script = self._make_script(shapes)
        compile(script, "<benchmark_no_norm>", "exec")

    def test_script_contains_qscale_kscale_pointers(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("QScale", script)
        self.assertIn("KScale", script)


# ---------------------------------------------------------------------------
# KernelBench new QK-norm problems
# ---------------------------------------------------------------------------

class KernelBenchQKNormProblemsTests(unittest.TestCase):
    def _attn_ids(self):
        return {p["id"] for p in KERNELBENCH_SUBSET["attention"]}

    def test_qknorm_local_problem_present(self):
        self.assertIn("kb_L3_attn_gemma_qknorm_local", self._attn_ids())

    def test_qknorm_global_problem_present(self):
        self.assertIn("kb_L3_attn_gemma_qknorm_global", self._attn_ids())

    def test_qknorm_local_has_correct_fields(self):
        problems = {p["id"]: p for p in KERNELBENCH_SUBSET["attention"]}
        prob = problems["kb_L3_attn_gemma_qknorm_local"]
        self.assertEqual(prob.get("window_size"), 1024)
        self.assertTrue(prob.get("is_causal"))
        self.assertTrue(prob.get("use_qk_norm"))
        self.assertEqual(prob.get("level"), 3)
        self.assertEqual(prob.get("head_dim"), 256)

    def test_qknorm_global_has_correct_fields(self):
        problems = {p["id"]: p for p in KERNELBENCH_SUBSET["attention"]}
        prob = problems["kb_L3_attn_gemma_qknorm_global"]
        self.assertNotEqual(prob.get("window_size", -1), 1024)
        self.assertTrue(prob.get("is_causal"))
        self.assertTrue(prob.get("use_qk_norm"))
        self.assertEqual(prob.get("level"), 3)

    def test_existing_attn_problems_unaffected(self):
        ids = self._attn_ids()
        self.assertIn("kb_L3_attn_gemma_local", ids)
        self.assertIn("kb_L3_attn_gemma_slide_causal", ids)
        self.assertIn("kb_L3_attn_llama7b_causal", ids)

# ---------------------------------------------------------------------------
# GQA shape bucket routing
# ---------------------------------------------------------------------------


class GQAShapeBucketTests(unittest.TestCase):
    def test_gemma4_31b_local_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 32, "num_kv_heads": 16,
                 "window_size": 1024, "use_qk_norm": True, "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_31b_local")

    def test_gemma4_31b_global_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 512, "heads": 32, "num_kv_heads": 4,
                 "window_size": -1, "use_qk_norm": True, "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_31b_global")

    def test_gemma4_26b_a4b_local_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 16, "num_kv_heads": 8,
                 "window_size": 1024, "use_qk_norm": True, "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_26b_a4b_local")

    def test_gemma4_26b_a4b_global_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 512, "heads": 16, "num_kv_heads": 2,
                 "window_size": -1, "use_qk_norm": True, "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_26b_a4b_global")

    def test_llama3_70b_gqa_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 128, "heads": 64, "num_kv_heads": 8,
                 "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "llama3_70b_gqa")

    def test_mistral_gqa_bucket(self):
        shape = {"seq_len": 8192, "head_dim": 128, "heads": 32, "num_kv_heads": 8,
                 "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "mistral_gqa")

    def test_extreme_gqa_num_kv_heads_1(self):
        """Gemma 4 E2B: 8 Q heads, 1 KV head."""
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 8, "num_kv_heads": 1,
                 "window_size": 1024, "use_qk_norm": True, "is_causal": True}
        key = attention_shape_bucket_key(shape)
        # Should route to some GQA gemma bucket.
        self.assertIn("gemma", key)

    def test_all_new_gqa_buckets_present_in_shape_list(self):
        names = {s["name"] for s in ATTENTION_SHAPE_BUCKETS}
        for n in ["gemma4_31b_local", "gemma4_31b_global",
                  "gemma4_26b_a4b_local", "gemma4_26b_a4b_global",
                  "llama3_70b_gqa", "mistral_gqa"]:
            self.assertIn(n, names)

    def test_new_gqa_buckets_have_num_kv_heads_lt_heads(self):
        gqa_names = {"gemma4_31b_local", "gemma4_31b_global",
                     "gemma4_26b_a4b_local", "gemma4_26b_a4b_global",
                     "llama3_70b_gqa", "mistral_gqa"}
        for s in ATTENTION_SHAPE_BUCKETS:
            if s["name"] in gqa_names:
                self.assertLess(s["num_kv_heads"], s["heads"])
                self.assertEqual(s["heads"] % s["num_kv_heads"], 0)

    def test_all_existing_buckets_have_num_kv_heads(self):
        """Every bucket carries an explicit num_kv_heads field (MHA or GQA)."""
        for s in ATTENTION_SHAPE_BUCKETS:
            self.assertIn("num_kv_heads", s, f"Bucket {s['name']} missing num_kv_heads")


# ---------------------------------------------------------------------------
# GQA benchmark-script validity
# ---------------------------------------------------------------------------


class GQABenchmarkScriptTests(unittest.TestCase):
    def _make_script(self, shapes):
        configs = [{"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}]
        return generate_attention_benchmark_script(configs, shapes)

    def test_script_with_gqa_shape_is_valid_python(self):
        shapes = [{
            "name": "gemma4_31b_local", "batch": 1, "heads": 32, "num_kv_heads": 16,
            "seq_len": 4096, "head_dim": 256, "is_causal": True,
            "window_size": 1024, "use_qk_norm": True,
        }]
        script = self._make_script(shapes)
        compile(script, "<benchmark_gqa>", "exec")

    def test_script_with_gqa_global_head_dim_512_valid(self):
        shapes = [{
            "name": "gemma4_31b_global", "batch": 1, "heads": 32, "num_kv_heads": 4,
            "seq_len": 4096, "head_dim": 512, "is_causal": True,
            "window_size": -1, "use_qk_norm": True,
        }]
        script = self._make_script(shapes)
        compile(script, "<benchmark_gqa_global>", "exec")

    def test_script_embeds_num_kv_heads(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "num_kv_heads": 2,
                   "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("NUM_KV_HEADS", script)
        self.assertIn("num_kv_heads", script)

    def test_script_embeds_group_size_constexpr(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "num_kv_heads": 2,
                   "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("GROUP_SIZE", script)

    def test_script_uses_repeat_interleave_for_reference(self):
        shapes = [{"name": "x", "batch": 1, "heads": 8, "num_kv_heads": 2,
                   "seq_len": 512, "head_dim": 64}]
        script = self._make_script(shapes)
        self.assertIn("repeat_interleave", script)


# ---------------------------------------------------------------------------
# Launcher assertions (require torch, not CUDA)
# ---------------------------------------------------------------------------


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LauncherAssertionsTests(unittest.TestCase):
    """The launcher in triton_attention.py only lives inside an auto-generated
    f-string benchmark script, so we can't import it directly. Instead we
    compile the generated script in a sandbox namespace, patch `triton` so we
    never actually launch a kernel, and call `flash_attn` to verify that the
    assertions fire before any kernel launch."""

    def _load_flash_attn(self):
        import types
        configs = [{"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 2, "num_stages": 2}]
        shapes = [{"name": "x", "batch": 1, "heads": 8, "num_kv_heads": 2,
                   "seq_len": 16, "head_dim": 64}]
        script = generate_attention_benchmark_script(configs, shapes)
        # Stub triton so kernel launches (if reached) are no-ops.
        fake_triton = types.ModuleType("triton")
        fake_triton.cdiv = lambda a, b: (a + b - 1) // b
        class _FakeJIT:
            def jit(self, fn):
                class _Launcher:
                    def __getitem__(self, grid):
                        return lambda *a, **kw: None
                return _Launcher()
        fake_triton.jit = _FakeJIT().jit
        fake_lang = types.ModuleType("triton.language")
        fake_lang.constexpr = int
        ns = {
            "__name__": "__sandbox__",
            "triton": fake_triton,
            "tl": fake_lang,
            "torch": torch,
        }
        # Strip the `if __name__ == "__main__": main()` tail so importing
        # the module doesn't actually run the benchmark.
        script_no_main = script.replace(
            'if __name__ == "__main__":\n    main()',
            "",
        )
        # Monkey-patch triton/triton.language imports inside the script by
        # executing in a namespace that already has them.
        import sys
        sys.modules.setdefault("triton", fake_triton)
        sys.modules.setdefault("triton.language", fake_lang)
        exec(compile(script_no_main, "<sandbox_attn>", "exec"), ns)
        return ns["flash_attn"]

    def test_flash_attn_rejects_mismatched_kv_heads(self):
        flash_attn = self._load_flash_attn()
        q = torch.zeros(1, 8, 16, 64, dtype=torch.float16)
        k = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        v = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        cfg = {"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 2, "num_stages": 2}
        with self.assertRaises(AssertionError):
            flash_attn(q, k, v, cfg, num_kv_heads=2)  # k/v have 4 not 2

    def test_flash_attn_rejects_nondivisible(self):
        flash_attn = self._load_flash_attn()
        q = torch.zeros(1, 6, 16, 64, dtype=torch.float16)
        k = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        v = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        cfg = {"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 2, "num_stages": 2}
        with self.assertRaises(AssertionError):
            flash_attn(q, k, v, cfg, num_kv_heads=4)  # 6 % 4 != 0


# ---------------------------------------------------------------------------
# YOCO KV-shared attention (Gemma 4 shared KV cache)
# ---------------------------------------------------------------------------


class YOCOShapeBucketTests(unittest.TestCase):
    """YOCO KV-shared shapes must route to dedicated yoco buckets."""

    def test_yoco_31b_local_bucket_present(self):
        names = {s["name"] for s in ATTENTION_SHAPE_BUCKETS}
        self.assertIn("gemma4_31b_yoco_local", names)

    def test_yoco_31b_global_bucket_present(self):
        names = {s["name"] for s in ATTENTION_SHAPE_BUCKETS}
        self.assertIn("gemma4_31b_yoco_global", names)

    def test_yoco_routes_before_gqa(self):
        """shared_kv=True shapes must NOT fall through to non-YOCO GQA buckets."""
        # Local YOCO shape (same dims as gemma4_31b_local but with shared_kv)
        shape_local = {
            "seq_len": 4096, "head_dim": 256, "heads": 32, "num_kv_heads": 16,
            "window_size": 1024, "use_qk_norm": True, "shared_kv": True,
        }
        self.assertEqual(attention_shape_bucket_key(shape_local), "gemma4_31b_yoco_local")
        # Without shared_kv, same shape routes to non-YOCO GQA bucket
        shape_local_no_yoco = dict(shape_local, shared_kv=False)
        self.assertEqual(attention_shape_bucket_key(shape_local_no_yoco), "gemma4_31b_local")

        # Global YOCO shape
        shape_global = {
            "seq_len": 4096, "head_dim": 512, "heads": 32, "num_kv_heads": 4,
            "window_size": -1, "use_qk_norm": True, "shared_kv": True,
        }
        self.assertEqual(attention_shape_bucket_key(shape_global), "gemma4_31b_yoco_global")
        # Without shared_kv, same shape routes to non-YOCO GQA bucket
        shape_global_no_yoco = dict(shape_global, shared_kv=False)
        self.assertEqual(attention_shape_bucket_key(shape_global_no_yoco), "gemma4_31b_global")

    def test_yoco_buckets_have_shared_kv_flag(self):
        for s in ATTENTION_SHAPE_BUCKETS:
            if "yoco" in s.get("name", ""):
                self.assertTrue(s.get("shared_kv", False),
                                f"YOCO bucket {s['name']} missing shared_kv=True")

    def test_yoco_buckets_have_qk_norm(self):
        for s in ATTENTION_SHAPE_BUCKETS:
            if "yoco" in s.get("name", ""):
                self.assertTrue(s.get("use_qk_norm", False),
                                f"YOCO bucket {s['name']} should have use_qk_norm=True")


class YOCOBenchmarkScriptTests(unittest.TestCase):

    def _make_script(self, shapes):
        configs = [{"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}]
        return generate_attention_benchmark_script(configs, shapes)

    def test_benchmark_script_handles_shared_kv(self):
        """The generated benchmark script must contain shared_kv handling."""
        shapes = [{
            "name": "gemma4_31b_yoco_local", "batch": 1, "heads": 32,
            "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256,
            "is_causal": True, "window_size": 1024, "use_qk_norm": True,
            "shared_kv": True,
        }]
        script = self._make_script(shapes)
        self.assertIn("shared_kv", script)
        compile(script, "<benchmark_yoco>", "exec")

    def test_benchmark_script_yoco_global_valid_python(self):
        shapes = [{
            "name": "gemma4_31b_yoco_global", "batch": 1, "heads": 32,
            "num_kv_heads": 4, "seq_len": 4096, "head_dim": 512,
            "is_causal": True, "window_size": -1, "use_qk_norm": True,
            "shared_kv": True,
        }]
        script = self._make_script(shapes)
        compile(script, "<benchmark_yoco_global>", "exec")

    def test_benchmark_script_yoco_skips_k_norm_in_reference(self):
        """YOCO reference path should skip K-norm (not shared_kv)."""
        shapes = [{
            "name": "gemma4_31b_yoco_local", "batch": 1, "heads": 32,
            "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256,
            "is_causal": True, "window_size": 1024, "use_qk_norm": True,
            "shared_kv": True,
        }]
        script = self._make_script(shapes)
        # The script should contain the shared_kv conditional for K-norm skip
        self.assertIn("shared_kv", script)
        self.assertIn("not shared_kv", script)


if __name__ == "__main__":
    unittest.main()
