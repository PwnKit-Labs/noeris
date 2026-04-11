"""Tests for softmax kernel shape buckets and softcap variant (#39, #40, #43)."""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.triton_softmax import (
    SOFTMAX_SHAPE_BUCKETS,
    generate_softmax_benchmark_script,
    softmax_shape_bucket_key,
)


class SoftmaxShapeBucketsTests(unittest.TestCase):
    def _names(self) -> set[str]:
        return {b["name"] for b in SOFTMAX_SHAPE_BUCKETS}

    def test_gemma4_262k_vocab_bucket_present(self):
        self.assertIn("gemma4_262k_vocab", self._names())

    def test_gemma4_31b_softcap_bucket_present(self):
        self.assertIn("gemma4_31b_softcap", self._names())

    def test_kb_l1_23_huge_bucket_present(self):
        self.assertIn("kb_l1_23_huge", self._names())

    def test_gemma4_31b_softcap_has_softcap_field(self):
        bucket = next(b for b in SOFTMAX_SHAPE_BUCKETS if b["name"] == "gemma4_31b_softcap")
        self.assertEqual(bucket["softcap"], 30.0)

    def test_gemma4_262k_vocab_bucket_has_262k_cols(self):
        bucket = next(b for b in SOFTMAX_SHAPE_BUCKETS if b["name"] == "gemma4_262k_vocab")
        self.assertEqual(bucket["n_cols"], 262144)
        # Non-softcap variant should not set a softcap field (or default to 0)
        self.assertEqual(bucket.get("softcap", 0.0), 0.0)

    def test_kb_l1_23_huge_has_upstream_shape(self):
        bucket = next(b for b in SOFTMAX_SHAPE_BUCKETS if b["name"] == "kb_l1_23_huge")
        self.assertEqual(bucket["n_rows"], 4096)
        self.assertEqual(bucket["n_cols"], 393216)


class SoftmaxShapeBucketKeyTests(unittest.TestCase):
    def test_kb_l1_23_huge_routes_huge(self):
        self.assertEqual(
            softmax_shape_bucket_key({"n_rows": 4096, "n_cols": 393216}),
            "kb_l1_23_huge",
        )

    def test_gemma4_262k_vocab_routes(self):
        # Gemma 4 vocab without softcap
        self.assertEqual(
            softmax_shape_bucket_key({"n_rows": 2048, "n_cols": 262144}),
            "gemma4_262k_vocab",
        )

    def test_gemma4_31b_softcap_routes(self):
        # Same vocab but with softcap=30 (Gemma 4 31B final logits)
        self.assertEqual(
            softmax_shape_bucket_key({"n_rows": 2048, "n_cols": 262144, "softcap": 30.0}),
            "gemma4_31b_softcap",
        )

    def test_softcap_zero_does_not_route_to_softcap_bucket(self):
        # softcap=0.0 must NOT flip routing — it's just the default
        self.assertEqual(
            softmax_shape_bucket_key({"n_rows": 2048, "n_cols": 262144, "softcap": 0.0}),
            "gemma4_262k_vocab",
        )

    def test_existing_small_routes_unchanged(self):
        # Regression: no-softcap non-huge small shapes still classify as before
        self.assertEqual(
            softmax_shape_bucket_key({"n_rows": 1024, "n_cols": 512}),
            "small",
        )

    def test_existing_vocab_llama_routes_unchanged(self):
        self.assertEqual(
            softmax_shape_bucket_key({"n_rows": 2048, "n_cols": 32000}),
            "vocab_llama",
        )


class SoftmaxKernelSoftcapTests(unittest.TestCase):
    """Verify the generated benchmark script contains the softcap codepath."""

    def setUp(self):
        configs = [{"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1}]
        self.script = generate_softmax_benchmark_script(configs, SOFTMAX_SHAPE_BUCKETS[:2])

    def test_kernel_has_use_softcap_constexpr(self):
        self.assertIn("USE_SOFTCAP", self.script)

    def test_kernel_has_softcap_arg(self):
        # The kernel signature should accept a softcap scalar
        self.assertIn("softcap,", self.script)

    def test_kernel_has_tanh_equivalent(self):
        # The softcap branch uses softcap * tanh(x / softcap); we implement
        # tanh via the stable exp form — verify the exp form is present.
        self.assertTrue(
            "tl.exp(-2.0 * x * inv_softcap)" in self.script
            or "tanh" in self.script,
        )

    def test_benchmark_reference_applies_softcap(self):
        # PyTorch reference must match — check it applies softcap before softmax
        self.assertIn("torch.tanh", self.script)

    def test_benchmark_script_compiles_as_python(self):
        compile(self.script, "<softmax-softcap>", "exec")


if __name__ == "__main__":
    unittest.main()
