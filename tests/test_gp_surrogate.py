"""Tests for the GP surrogate model for cross-shape config transfer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _pathfix  # noqa: F401

from research_engine.triton_kernels import ConfigDatabase, ShapeRecord
from research_engine.gp_surrogate import (
    GPSurrogate,
    GPGuidedSelector,
    encode_features,
    _encode_rmsnorm,
    _encode_attention,
    _encode_matmul,
    _encode_qk_norm_rope,
)


def _make_database(records_data: dict) -> ConfigDatabase:
    """Create a ConfigDatabase from a dict of records for testing."""
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps({"records": records_data}))
    db = ConfigDatabase(path=tmp)
    tmp.unlink(missing_ok=True)
    return db


def _synthetic_matmul_database(n_shapes: int = 5, configs_per_shape: int = 10) -> ConfigDatabase:
    """Generate a synthetic matmul database with realistic performance patterns.

    Performance model: tflops ~ f(log2(BLOCK_M) * log2(M), ...) + noise.
    Larger blocks on larger shapes => higher throughput (simplified).
    """
    rng = np.random.default_rng(42)
    records: dict[str, dict] = {}

    shapes = [
        {"M": 2 ** (8 + i), "N": 2 ** (8 + i), "K": 512, "bucket": f"shape_{i}"}
        for i in range(n_shapes)
    ]

    block_sizes = [32, 64, 128, 256]
    warp_counts = [2, 4, 8]
    stage_counts = [2, 3, 4]

    for si, shape in enumerate(shapes):
        key = f"matmul:shape_{si}:A100"
        results = []
        for ci in range(configs_per_shape):
            bm = block_sizes[ci % len(block_sizes)]
            bn = block_sizes[(ci + 1) % len(block_sizes)]
            bk = 32
            gm = 8
            nw = warp_counts[ci % len(warp_counts)]
            ns = stage_counts[ci % len(stage_counts)]

            config = {
                "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns,
            }

            # Synthetic throughput: larger blocks + larger shapes = higher tflops
            base = (np.log2(bm) + np.log2(bn)) * np.log2(shape["M"]) * 0.1
            noise = rng.normal(0, 0.5)
            tflops = max(float(base + noise), 0.1)

            config_id = f"bm{bm}_bn{bn}_bk{bk}_gm{gm}_w{nw}_s{ns}"
            results.append({
                "config_id": config_id,
                "config": config,
                "tflops": round(tflops, 3),
                "ms": round(1.0 / max(tflops, 0.01), 3),
                "correct": True,
                "hardware": "A100",
                "operator": "matmul",
            })

        records[key] = {
            "shape_key": key,
            "shape": shape,
            "best_config_id": max(results, key=lambda r: r["tflops"])["config_id"],
            "best_tflops": max(r["tflops"] for r in results),
            "results": results,
        }

    return _make_database(records)


def _synthetic_rmsnorm_database(n_shapes: int = 4, configs_per_shape: int = 8) -> ConfigDatabase:
    """Generate a synthetic rmsnorm database."""
    rng = np.random.default_rng(123)
    records: dict[str, dict] = {}

    shapes = [
        {"hidden_dim": 2 ** (9 + i), "n_rows": 2048, "affine": 1, "bucket": f"rms_{i}"}
        for i in range(n_shapes)
    ]

    for si, shape in enumerate(shapes):
        key = f"rmsnorm:rms_{si}:A100"
        results = []
        for ci in range(configs_per_shape):
            bs = 2 ** (5 + ci % 4)
            nw = [2, 4, 8][ci % 3]
            ns = [2, 3, 4][ci % 3]

            config = {"BLOCK_SIZE": bs, "num_warps": nw, "num_stages": ns}
            base = np.log2(bs) * np.log2(shape["hidden_dim"]) * 0.05
            noise = rng.normal(0, 0.3)
            gb_per_s = max(float(base + noise), 0.1)

            results.append({
                "config_id": f"bs{bs}_w{nw}_s{ns}",
                "config": config,
                "tflops": round(gb_per_s, 3),
                "ms": round(1.0 / max(gb_per_s, 0.01), 3),
                "correct": True,
                "hardware": "A100",
                "operator": "rmsnorm",
            })

        records[key] = {
            "shape_key": key,
            "shape": shape,
            "best_config_id": max(results, key=lambda r: r["tflops"])["config_id"],
            "best_tflops": max(r["tflops"] for r in results),
            "results": results,
        }

    return _make_database(records)


class TestFeatureEncoding(unittest.TestCase):
    """Test feature encoding for each operator."""

    def test_feature_encoding_rmsnorm(self) -> None:
        config = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 3}
        shape = {"hidden_dim": 4096, "n_rows": 2048, "affine": 1}
        features = _encode_rmsnorm(config, shape)
        self.assertEqual(len(features), 6)
        # log2(128) = 7.0
        self.assertAlmostEqual(features[0], 7.0)
        # log2(4096) = 12.0
        self.assertAlmostEqual(features[3], 12.0)

    def test_feature_encoding_attention(self) -> None:
        config = {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2}
        shape = {
            "seq_len": 2048, "head_dim": 128, "num_heads": 32,
            "num_kv_heads": 8, "is_causal": True, "window_size": 0,
        }
        features = _encode_attention(config, shape)
        self.assertEqual(len(features), 10)
        # log2(64) = 6.0
        self.assertAlmostEqual(features[0], 6.0)
        # is_causal = 1.0
        self.assertAlmostEqual(features[8], 1.0)

    def test_feature_encoding_matmul(self) -> None:
        config = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3,
        }
        shape = {"M": 4096, "N": 4096, "K": 512}
        features = _encode_matmul(config, shape)
        self.assertEqual(len(features), 9)
        # log2(128) = 7.0
        self.assertAlmostEqual(features[0], 7.0)
        # log2(4096) = 12.0
        self.assertAlmostEqual(features[6], 12.0)

    def test_feature_encoding_qk_norm_rope(self) -> None:
        config = {"BLOCK_SIZE": 64, "num_warps": 4, "num_stages": 2}
        shape = {"head_dim": 128, "num_heads": 32, "num_kv_heads": 8, "seq_len": 2048}
        features = _encode_qk_norm_rope(config, shape)
        self.assertEqual(len(features), 7)
        # log2(64) = 6.0
        self.assertAlmostEqual(features[0], 6.0)

    def test_encode_features_dispatches_correctly(self) -> None:
        config = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 3}
        shape = {"hidden_dim": 4096, "n_rows": 2048, "affine": 1}
        features = encode_features("rmsnorm", config, shape)
        self.assertEqual(len(features), 6)

        config2 = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
        }
        shape2 = {"M": 2048, "N": 2048, "K": 2048}
        features2 = encode_features("matmul", config2, shape2)
        self.assertEqual(len(features2), 9)


class TestGPSurrogate(unittest.TestCase):
    """Test the GP surrogate model."""

    def test_fit_on_synthetic_data(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        surrogate = GPSurrogate(operator="matmul")
        stats = surrogate.fit(db, hardware="A100")

        self.assertEqual(stats["status"], "trained")
        self.assertEqual(stats["n_samples"], 50)
        self.assertGreater(stats["r_squared"], 0.0)

    def test_predict_returns_mean_and_std(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        surrogate = GPSurrogate(operator="matmul")
        surrogate.fit(db, hardware="A100")

        config = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
        shape = {"M": 2048, "N": 2048, "K": 512}
        mean, std = surrogate.predict(config, shape)

        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)
        self.assertGreater(std, 0.0)
        # Mean should be in a reasonable range (not NaN/inf)
        self.assertFalse(np.isnan(mean))
        self.assertFalse(np.isinf(mean))

    def test_recommend_configs_returns_top_k(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        surrogate = GPSurrogate(operator="matmul")
        surrogate.fit(db, hardware="A100")

        candidates = [
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
            for bm in [32, 64, 128, 256]
            for bn in [32, 64, 128, 256]
        ]

        shape = {"M": 4096, "N": 4096, "K": 512}
        recommended = surrogate.recommend_configs(shape, candidates, top_k=5)

        self.assertEqual(len(recommended), 5)
        # Check that results are sorted by predicted_mean descending
        means = [r["predicted_mean"] for r in recommended]
        self.assertEqual(means, sorted(means, reverse=True))
        # Each result should have predicted_mean and predicted_std
        for r in recommended:
            self.assertIn("predicted_mean", r)
            self.assertIn("predicted_std", r)

    def test_transfer_from_nearest_finds_similar_shape(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        surrogate = GPSurrogate(operator="matmul")
        surrogate.fit(db, hardware="A100")

        # Query a shape close to shape_2 (M=1024, N=1024)
        target_shape = {"M": 1100, "N": 1100, "K": 512}
        transferred = surrogate.transfer_from_nearest(
            target_shape, db, hardware="A100", top_k=3,
        )

        self.assertGreater(len(transferred), 0)
        self.assertLessEqual(len(transferred), 3)
        for r in transferred:
            self.assertIn("predicted_mean", r)

    def test_gp_falls_back_on_insufficient_data(self) -> None:
        """GP should gracefully degrade with fewer than 5 data points."""
        records = {
            "matmul:tiny:A100": {
                "shape_key": "matmul:tiny:A100",
                "shape": {"M": 128, "N": 128, "K": 128, "bucket": "tiny"},
                "best_config_id": "test",
                "best_tflops": 1.0,
                "results": [
                    {
                        "config_id": "test",
                        "config": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
                                   "GROUP_SIZE_M": 8, "num_warps": 2, "num_stages": 2},
                        "tflops": 1.0, "ms": 1.0, "correct": True,
                        "hardware": "A100", "operator": "matmul",
                    }
                ],
            }
        }
        db = _make_database(records)
        surrogate = GPSurrogate(operator="matmul")
        stats = surrogate.fit(db, hardware="A100")

        self.assertEqual(stats["status"], "insufficient_data")
        self.assertEqual(stats["n_samples"], 1)

        # predict should return fallback
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                  "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
        shape = {"M": 256, "N": 256, "K": 256}
        mean, std = surrogate.predict(config, shape)
        self.assertEqual(std, float("inf"))

    def test_fit_stats_contain_r_squared(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        surrogate = GPSurrogate(operator="matmul")
        stats = surrogate.fit(db, hardware="A100")

        self.assertIn("r_squared", stats)
        self.assertIn("n_samples", stats)
        self.assertIn("marginal_log_likelihood", stats)
        self.assertIn("status", stats)
        self.assertIsInstance(stats["r_squared"], float)

    def test_evaluate_transfer_on_synthetic(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        surrogate = GPSurrogate(operator="matmul")
        # Note: evaluate_transfer fits its own sub-GPs, so we don't need to
        # fit the outer surrogate first.
        result = surrogate.evaluate_transfer(db, hardware="A100")

        self.assertIn("status", result)
        self.assertIn("mean_spearman_rho", result)
        self.assertIn("mean_top5_overlap", result)
        self.assertIn("mean_predicted_rank_of_actual_best", result)
        self.assertIn("per_shape", result)

        if result["status"] == "evaluated":
            self.assertGreater(len(result["per_shape"]), 0)
            # With synthetic data the GP should do reasonably well
            self.assertIsInstance(result["mean_spearman_rho"], float)

    def test_rmsnorm_surrogate(self) -> None:
        """Verify GP works on a non-matmul operator."""
        db = _synthetic_rmsnorm_database(n_shapes=4, configs_per_shape=8)
        surrogate = GPSurrogate(operator="rmsnorm")
        stats = surrogate.fit(db, hardware="A100")

        self.assertEqual(stats["status"], "trained")
        self.assertEqual(stats["n_samples"], 32)

        config = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 3}
        shape = {"hidden_dim": 4096, "n_rows": 2048, "affine": 1}
        mean, std = surrogate.predict(config, shape)
        self.assertFalse(np.isnan(mean))


class TestGPGuidedSelector(unittest.TestCase):
    """Test the GP-guided config selector."""

    def test_gp_guided_selector_produces_configs(self) -> None:
        db = _synthetic_matmul_database(n_shapes=5, configs_per_shape=10)
        selector = GPGuidedSelector(operator="matmul", top_k=5)
        fit_stats = selector.fit(db, hardware="A100")

        self.assertEqual(fit_stats["status"], "trained")

        candidates = [
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
            for bm in [32, 64, 128, 256]
            for bn in [32, 64, 128, 256]
        ]

        shapes = [
            {"M": 4096, "N": 4096, "K": 512},
            {"M": 2048, "N": 2048, "K": 1024},
        ]

        proposals = selector.propose_configs(
            shapes=shapes,
            candidate_configs=candidates,
            top_k=5,
        )

        self.assertEqual(len(proposals), 5)
        # Each proposal should be a config dict
        for p in proposals:
            self.assertIn("BLOCK_SIZE_M", p)

    def test_gp_guided_selector_empty_when_unfitted(self) -> None:
        selector = GPGuidedSelector(operator="matmul")
        proposals = selector.propose_configs(
            shapes=[{"M": 1024, "N": 1024, "K": 1024}],
            candidate_configs=[{"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
                                "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
                                "num_warps": 4, "num_stages": 3}],
        )
        self.assertEqual(len(proposals), 0)


if __name__ == "__main__":
    unittest.main()
