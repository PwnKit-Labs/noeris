from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.cost_model import (
    CostModel,
    extract_features,
    FEATURE_NAMES,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_matmul_features_have_fixed_width(self) -> None:
        features = extract_features(
            shape={"M": 2048, "N": 2048, "K": 2048},
            config={
                "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
            },
            hardware="NVIDIA A100-SXM4-40GB",
            operator="matmul",
        )
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(all(isinstance(f, float) for f in features))

    def test_rmsnorm_features_have_fixed_width(self) -> None:
        features = extract_features(
            shape={"n_rows": 4096, "hidden_dim": 4096},
            config={"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 1},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="rmsnorm",
        )
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_attention_features_include_causal(self) -> None:
        non_causal = extract_features(
            shape={"batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": False},
            config={"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="attention",
        )
        causal = extract_features(
            shape={"batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": True},
            config={"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="attention",
        )
        # The only difference should be the is_causal flag in shape slot 4
        self.assertNotEqual(non_causal, causal)


class CostModelTrainingTests(unittest.TestCase):
    def _make_synthetic_db(self, tmpdir: str) -> Path:
        """Create a small synthetic ConfigDatabase with matmul results."""
        db_path = Path(tmpdir) / "configs.json"
        records = {}
        # Create 30 synthetic matmul records across 3 shape buckets
        for bucket_idx, bucket_name in enumerate(["medium", "large", "xlarge"]):
            shape = {
                "M": 2048 * (bucket_idx + 1),
                "N": 2048 * (bucket_idx + 1),
                "K": 2048 * (bucket_idx + 1),
                "bucket": bucket_name,
            }
            key = f"matmul:{bucket_name}:NVIDIA A100-SXM4-40GB"
            results = []
            for i in range(10):
                config = {
                    "BLOCK_SIZE_M": 64 if i % 2 == 0 else 128,
                    "BLOCK_SIZE_N": 128 if i % 2 == 0 else 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                    "num_warps": 4 if i % 2 == 0 else 8,
                    "num_stages": 3,
                }
                # Synthetic metric — larger configs on larger shapes = better
                metric = 100 + 20 * bucket_idx + 10 * (i % 2) + i
                results.append({
                    "config_id": f"test_{i}",
                    "config": config,
                    "tflops": metric,
                    "ms": 1.0,
                    "correct": True,
                    "run_id": f"test_{i}",
                    "hardware": "NVIDIA A100-SXM4-40GB",
                    "operator": "matmul",
                })
            records[key] = {
                "shape_key": key,
                "shape": shape,
                "best_config_id": "test_0",
                "best_tflops": 200.0,
                "results": results,
            }
        data = {"records": records}
        db_path.write_text(json.dumps(data) + "\n")
        return db_path

    def test_train_insufficient_data_returns_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_db = Path(tmpdir) / "empty.json"
            empty_db.write_text(json.dumps({"records": {}}))
            model = CostModel()
            stats = model.train_from_databases([empty_db])
            self.assertEqual(stats["status"], "insufficient_data")
            self.assertEqual(stats["training_size"], 0)

    def test_train_with_synthetic_db_learns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_synthetic_db(tmpdir)
            model = CostModel()
            stats = model.train_from_databases([db_path])
            self.assertIn(stats["status"], ("trained", "sklearn_unavailable"))
            self.assertGreaterEqual(stats["training_size"], 30)

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_synthetic_db(tmpdir)
            model = CostModel()
            model.train_from_databases([db_path])
            model_path = Path(tmpdir) / "model.pkl"
            model.save(model_path)
            loaded = CostModel.load(model_path)
            self.assertEqual(loaded.training_size, model.training_size)

    def test_predict_fallback_without_regressor(self) -> None:
        model = CostModel(mean_by_operator={0: 123.4})
        pred = model.predict(
            shape={"M": 2048, "N": 2048, "K": 2048},
            config={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="matmul",
        )
        self.assertAlmostEqual(pred, 123.4, places=3)

    def test_rank_configs_returns_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_synthetic_db(tmpdir)
            model = CostModel()
            model.train_from_databases([db_path])
            configs = [
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                 "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                 "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
            ]
            shapes = [{"M": 2048, "N": 2048, "K": 2048}]
            ranked = model.rank_configs(
                configs=configs, shapes=shapes,
                hardware="NVIDIA A100-SXM4-40GB", operator="matmul",
            )
            self.assertEqual(len(ranked), 2)
            self.assertGreaterEqual(ranked[0][1], ranked[1][1])


if __name__ == "__main__":
    unittest.main()
