from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.cross_vendor_transfer import (
    canonicalize_config_for_operator,
    collect_source_bucket_candidates,
    latency_regret,
    parse_record_key,
    spearman_rank_correlation,
    top_k_hit_rate,
)


class CrossVendorTransferHelpersTests(unittest.TestCase):
    def test_parse_record_key(self) -> None:
        self.assertEqual(
            parse_record_key("attention:gemma4_qknorm:NVIDIA A100-SXM4-40GB"),
            ("attention", "gemma4_qknorm", "NVIDIA A100-SXM4-40GB"),
        )
        self.assertEqual(
            parse_record_key("bucket:NVIDIA A100"),
            ("matmul", "bucket", "NVIDIA A100"),
        )
        self.assertIsNone(parse_record_key("invalid"))

    def test_collect_source_bucket_candidates(self) -> None:
        records = {
            "attention:gemma4_qknorm:NVIDIA A100-SXM4-40GB": {
                "shape": {"batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 256},
                "results": [
                    {"config_id": "c1", "config": {"BLOCK_M": 32}, "tflops": 10.0, "correct": True},
                    {"config_id": "c2", "config": {"BLOCK_M": 64}, "tflops": 9.0, "correct": True},
                    {"config_id": "c_bad", "config": {"BLOCK_M": 16}, "tflops": 0.0, "correct": False},
                ],
            },
            "attention:gemma4_qknorm:NVIDIA H100 80GB HBM3": {
                "shape": {"batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 256},
                "results": [
                    {"config_id": "c3", "config": {"BLOCK_M": 32}, "tflops": 11.0, "correct": True},
                ],
            },
            "attention:gemma4_qknorm:NVIDIA A100-SXM4-80GB": {
                "shape": {"batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 256},
                "results": [
                    {"config_id": "c4", "config": {"BLOCK_M": 128}, "tflops": 12.0, "correct": True},
                ],
            },
        }

        out = collect_source_bucket_candidates(
            records=records,
            operator="attention",
            source_hardware_substr="A100",
            max_candidates_per_bucket=4,
        )
        self.assertIn("gemma4_qknorm", out)
        self.assertEqual(len(out["gemma4_qknorm"].configs), 3)
        self.assertEqual(out["gemma4_qknorm"].configs[0]["config_id"], "c4")

    def test_canonicalize_config_for_matmul(self) -> None:
        cfg = canonicalize_config_for_operator(operator="matmul", config={"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32})
        self.assertEqual(cfg["BLOCK_SIZE_M"], 128)
        self.assertEqual(cfg["BLOCK_SIZE_N"], 64)
        self.assertEqual(cfg["BLOCK_SIZE_K"], 32)

    def test_transfer_metrics(self) -> None:
        rho = spearman_rank_correlation([10.0, 9.0, 8.0], [100.0, 90.0, 80.0])
        self.assertAlmostEqual(rho, 1.0)
        hit = top_k_hit_rate(["a", "b", "c"], ["b", "a", "d"], k=2)
        self.assertAlmostEqual(hit, 1.0)
        regret = latency_regret(predicted_best_ms=1.2, measured_best_ms=1.0)
        self.assertAlmostEqual(regret, 0.2)


if __name__ == "__main__":
    unittest.main()
