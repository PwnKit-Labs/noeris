from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.cost_model import (
    CostModel,
    extract_features,
    encode_features,
    FEATURE_NAMES,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_matmul_features_have_fixed_width(self) -> None:
        raw = extract_features(
            shape={"M": 2048, "N": 2048, "K": 2048},
            config={
                "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
            },
            hardware="NVIDIA A100-SXM4-40GB",
            operator="matmul",
        )
        features = encode_features(raw)
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(all(isinstance(f, float) for f in features))

    def test_rmsnorm_features_have_fixed_width(self) -> None:
        raw = extract_features(
            shape={"n_rows": 4096, "hidden_dim": 4096},
            config={"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 1},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="rmsnorm",
        )
        features = encode_features(raw)
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_attention_features_include_causal(self) -> None:
        non_causal = encode_features(extract_features(
            shape={"batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": False},
            config={"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="attention",
        ))
        causal = encode_features(extract_features(
            shape={"batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": True},
            config={"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
            hardware="NVIDIA A100-SXM4-40GB",
            operator="attention",
        ))
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


class MultiHardwareTests(unittest.TestCase):
    """Tests for multi-hardware cost model training (v2)."""

    @staticmethod
    def _make_multi_hw_db(tmpdir: str) -> Path:
        """Create a synthetic DB with both T4 and A100 measurements."""
        db_path = Path(tmpdir) / "multi_hw.json"
        records = {}

        for hw_name, hw_label in [("NVIDIA T4", "t4"), ("NVIDIA A100-SXM4-80GB", "a100")]:
            for bucket_idx, bucket_name in enumerate(["small", "medium", "large"]):
                shape = {
                    "M": 1024 * (bucket_idx + 1),
                    "N": 1024 * (bucket_idx + 1),
                    "K": 1024 * (bucket_idx + 1),
                    "bucket": bucket_name,
                }
                key = f"matmul:{bucket_name}:{hw_name}"
                results = []
                for i in range(12):
                    nw = [1, 2, 4, 8][i % 4]
                    config = {
                        "BLOCK_SIZE_M": 64 if i % 2 == 0 else 128,
                        "BLOCK_SIZE_N": 128 if i % 2 == 0 else 64,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 8,
                        "num_warps": nw,
                        "num_stages": 3,
                    }
                    # T4 prefers low warps, A100 prefers high warps
                    if hw_label == "t4":
                        metric = 50 + 10 * bucket_idx - 5 * nw + i
                    else:
                        metric = 100 + 20 * bucket_idx + 5 * nw + i
                    results.append({
                        "config_id": f"{hw_label}_{bucket_name}_{i}",
                        "config": config,
                        "tflops": max(metric, 1.0),
                        "ms": 1.0,
                        "correct": True,
                        "run_id": f"{hw_label}_{i}",
                        "hardware": hw_name,
                        "operator": "matmul",
                    })
                records[key] = {
                    "shape_key": key,
                    "shape": shape,
                    "best_config_id": f"{hw_label}_{bucket_name}_0",
                    "best_tflops": 200.0,
                    "results": results,
                }

        db_path.write_text(json.dumps({"records": records}) + "\n")
        return db_path

    @staticmethod
    def _make_multi_op_db(tmpdir: str) -> Path:
        """Create a synthetic DB with multiple operators for LOO testing."""
        db_path = Path(tmpdir) / "multi_op.json"
        records = {}

        # matmul records
        for bucket_idx, bucket_name in enumerate(["small", "medium", "large"]):
            shape = {"M": 1024 * (bucket_idx + 1), "N": 1024 * (bucket_idx + 1),
                      "K": 1024 * (bucket_idx + 1), "bucket": bucket_name}
            key = f"matmul:{bucket_name}:NVIDIA A100-SXM4-80GB"
            results = []
            for i in range(10):
                config = {"BLOCK_SIZE_M": 64 + 64 * (i % 2), "BLOCK_SIZE_N": 128,
                          "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
                          "num_warps": 4, "num_stages": 3}
                results.append({"config_id": f"mm_{i}", "config": config,
                                "tflops": 100 + 20 * bucket_idx + i,
                                "ms": 1.0, "correct": True,
                                "run_id": f"mm_{i}", "hardware": "NVIDIA A100-SXM4-80GB",
                                "operator": "matmul"})
            records[key] = {"shape_key": key, "shape": shape,
                           "best_config_id": "mm_0", "best_tflops": 200.0,
                           "results": results}

        # rmsnorm records
        for bucket_idx, bucket_name in enumerate(["small_hidden", "large_hidden"]):
            shape = {"n_rows": 1024 * (bucket_idx + 1), "hidden_dim": 768 * (bucket_idx + 1),
                      "bucket": bucket_name}
            key = f"rmsnorm:{bucket_name}:NVIDIA A100-SXM4-80GB"
            results = []
            for i in range(10):
                config = {"BLOCK_SIZE": 1024 * (1 + i % 3), "num_warps": 4, "num_stages": 1}
                results.append({"config_id": f"rms_{i}", "config": config,
                                "gb_per_s": 200 + 30 * bucket_idx + i * 5,
                                "ms": 0.01, "correct": True,
                                "run_id": f"rms_{i}", "hardware": "NVIDIA A100-SXM4-80GB",
                                "operator": "rmsnorm"})
            records[key] = {"shape_key": key, "shape": shape,
                           "best_config_id": "rms_0", "best_tflops": 300.0,
                           "results": results}

        db_path.write_text(json.dumps({"records": records}) + "\n")
        return db_path

    def test_train_on_multi_hardware_data(self) -> None:
        """Synthetic T4 + A100 data, verify R² > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_multi_hw_db(tmpdir)
            model = CostModel()
            stats = model.train_from_databases([db_path])
            self.assertEqual(stats["status"], "trained")
            self.assertGreater(stats["training_size"], 50)
            # R² > 0 means the model explains some variance
            self.assertGreater(stats["holdout_r2"], 0.0)

    def test_hardware_feature_improves_prediction(self) -> None:
        """Model WITH hardware feature should beat model WITHOUT on multi-hw data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_multi_hw_db(tmpdir)

            # Train full model (with hardware features)
            full_model = CostModel()
            full_stats = full_model.train_from_databases([db_path])

            # Train a "no-hw" model by zeroing out hw one-hot features
            # Load the data manually and zero out columns
            data = json.loads(db_path.read_text())
            records = data.get("records", {})
            X_full, X_nohw, y_all = [], [], []

            for key, record in records.items():
                parts = key.split(":", 2)
                operator, _, hardware = parts
                shape = record.get("shape", {})
                for result in record.get("results", []):
                    if not result.get("correct"):
                        continue
                    config = result.get("config", {})
                    metric = result.get("tflops") or result.get("gb_per_s") or 0
                    if metric <= 0:
                        continue
                    raw = extract_features(
                        shape=shape, config=config,
                        hardware=hardware, operator=operator,
                    )
                    encoded = encode_features(raw)
                    X_full.append(encoded)
                    # Zero out hardware one-hot (positions 6..13 in the encoded vector)
                    nohw = list(encoded)
                    n_ops = 6  # number of operator slots
                    n_hw = 8   # number of hardware slots
                    for j in range(n_ops, n_ops + n_hw):
                        nohw[j] = 0.0
                    X_nohw.append(nohw)
                    y_all.append(float(metric))

            try:
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.model_selection import cross_val_score
                import numpy as np

                gbr_full = GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    min_samples_leaf=3, random_state=42,
                )
                gbr_nohw = GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    min_samples_leaf=3, random_state=42,
                )

                scores_full = cross_val_score(gbr_full, np.array(X_full),
                                               np.array(y_all), cv=3, scoring="r2")
                scores_nohw = cross_val_score(gbr_nohw, np.array(X_nohw),
                                               np.array(y_all), cv=3, scoring="r2")

                # Full model (with hw features) should have higher mean R²
                self.assertGreaterEqual(np.mean(scores_full), np.mean(scores_nohw))
            except ImportError:
                self.skipTest("sklearn not available")

    def test_feature_importance_is_dict(self) -> None:
        """Feature importance returns a dict mapping feature names to floats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_multi_hw_db(tmpdir)
            model = CostModel()
            model.train_from_databases([db_path])

            if model.regressor is None:
                self.skipTest("sklearn not available")

            importances = model.regressor.feature_importances_
            self.assertEqual(len(importances), len(FEATURE_NAMES))

            imp_dict = {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}
            self.assertIsInstance(imp_dict, dict)
            self.assertTrue(all(isinstance(v, float) for v in imp_dict.values()))
            # Sum should be ~1.0
            self.assertAlmostEqual(sum(imp_dict.values()), 1.0, places=3)

    def test_leave_one_operator_out(self) -> None:
        """LOO evaluation should return per-operator results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._make_multi_op_db(tmpdir)

            data = json.loads(db_path.read_text())
            records = data.get("records", {})
            X, y_vals, operators = [], [], []

            for key, record in records.items():
                parts = key.split(":", 2)
                operator, _, hardware = parts
                shape = record.get("shape", {})
                for result in record.get("results", []):
                    if not result.get("correct"):
                        continue
                    config = result.get("config", {})
                    metric = result.get("tflops") or result.get("gb_per_s") or 0
                    if metric <= 0:
                        continue
                    raw = extract_features(
                        shape=shape, config=config,
                        hardware=hardware, operator=operator,
                    )
                    X.append(encode_features(raw))
                    y_vals.append(float(metric))
                    operators.append(operator)

            # Verify we have multiple operators
            unique_ops = set(operators)
            self.assertGreaterEqual(len(unique_ops), 2)

            # Import the LOO function from the training script
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
            from train_cost_model_v2 import leave_one_operator_out

            results = leave_one_operator_out(X, y_vals, operators)
            self.assertIsInstance(results, dict)
            # Should have an entry for each operator
            for op in unique_ops:
                self.assertIn(op, results)
                self.assertIn("r2", results[op])
                self.assertIn("n_test", results[op])


if __name__ == "__main__":
    unittest.main()
