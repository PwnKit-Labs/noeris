from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cross_vendor_transfer_eval import _to_md, evaluate_transfer  # noqa: E402


class CrossVendorTransferEvalTests(unittest.TestCase):
    def test_evaluate_transfer_metrics(self) -> None:
        prediction = {
            "predictions": {
                "attention": {
                    "bucket_a": {
                        "target_predicted_top": [
                            {"config_id": "c1", "predicted_metric": 10.0},
                            {"config_id": "c2", "predicted_metric": 9.0},
                            {"config_id": "c3", "predicted_metric": 8.0},
                        ]
                    }
                }
            }
        }
        measured = {
            "measured": {
                "attention": {
                    "bucket_a": [
                        {"config_id": "c2", "metric": 95.0, "latency_ms": 1.05},
                        {"config_id": "c1", "metric": 90.0, "latency_ms": 1.10},
                        {"config_id": "c3", "metric": 80.0, "latency_ms": 1.20},
                    ]
                }
            }
        }
        out = evaluate_transfer(prediction=prediction, measured=measured, top_k=2)
        row = out["by_operator"]["attention"]["bucket_a"]
        self.assertGreaterEqual(row["spearman"], 0.5)
        self.assertAlmostEqual(row["topk_hit_rate"], 1.0)
        self.assertGreaterEqual(row["latency_regret"], 0.0)

    def test_markdown_contains_summary_columns(self) -> None:
        report = {
            "generated_at_utc": "2026-05-07T00:00:00+00:00",
            "prediction_artifact": "docs/results/pred.json",
            "measured_artifact": "docs/results/meas.json",
            "operator_summary": {
                "attention": {
                    "bucket_count": 1,
                    "mean_spearman": 0.75,
                    "mean_topk_hit_rate": 1.0,
                    "mean_latency_regret": 0.05,
                }
            },
        }
        md = _to_md(report)
        self.assertIn("mean spearman", md)
        self.assertIn("0.7500", md)


if __name__ == "__main__":
    unittest.main()
