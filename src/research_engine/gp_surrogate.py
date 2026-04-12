"""Gaussian Process surrogate for cross-shape config transfer.

This is Noeris's second headline contribution: no published autotuning system
transfers configs across operator shapes via a learned surrogate model.  Every
prior system (Triton autotuner, AutoKernel, KernelSkill) starts fresh for each
new shape.  The GP surrogate trains on the ConfigDatabase of
(config, shape, hardware) -> metric triples and predicts performance for unseen
shapes, enabling zero-shot config transfer without GPU evaluation.

The GP uses an RBF + WhiteKernel on StandardScaler-normalised features.  For
each operator we define a custom feature encoding that captures the key
performance-relevant dimensions (block sizes, warp counts, shape dimensions)
with log2 scaling to normalise across orders of magnitude.

Falls back to the existing GBR cost model if the GP fails (numerical
instability on large datasets, insufficient data, etc.).

Dependencies: scikit-learn (GaussianProcessRegressor, StandardScaler).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------

def _safe_log2(x: float | int) -> float:
    """log2 with clamping for non-positive values."""
    return math.log2(max(float(x), 1.0))


def _encode_rmsnorm(config: dict, shape: dict) -> list[float]:
    """Encode rmsnorm (config, shape) pair to feature vector."""
    return [
        _safe_log2(config.get("BLOCK_SIZE", 1)),
        float(config.get("num_warps", 1)),
        float(config.get("num_stages", 1)),
        _safe_log2(shape.get("hidden_dim", 1)),
        _safe_log2(shape.get("n_rows", 1)),
        float(shape.get("affine", 1)),  # affine mode: 0 or 1
    ]


def _encode_attention(config: dict, shape: dict) -> list[float]:
    """Encode attention (config, shape) pair to feature vector."""
    window = shape.get("window_size", 0)
    seq_len = shape.get("seq_len", 1)
    window_norm = float(window) / max(float(seq_len), 1.0) if window else 0.0
    return [
        _safe_log2(config.get("BLOCK_M", 1)),
        _safe_log2(config.get("BLOCK_N", 1)),
        float(config.get("num_warps", 1)),
        float(config.get("num_stages", 1)),
        _safe_log2(shape.get("seq_len", 1)),
        _safe_log2(shape.get("head_dim", 1)),
        float(shape.get("num_heads", shape.get("heads", 1))),
        float(shape.get("num_kv_heads", shape.get("heads", 1))),
        1.0 if shape.get("is_causal", False) else 0.0,
        window_norm,
    ]


def _encode_qk_norm_rope(config: dict, shape: dict) -> list[float]:
    """Encode qk_norm_rope (config, shape) pair to feature vector."""
    return [
        _safe_log2(config.get("BLOCK_SIZE", 1)),
        float(config.get("num_warps", 1)),
        float(config.get("num_stages", 1)),
        _safe_log2(shape.get("head_dim", 1)),
        float(shape.get("num_heads", shape.get("heads", 1))),
        float(shape.get("num_kv_heads", shape.get("heads", 1))),
        _safe_log2(shape.get("seq_len", 1)),
    ]


def _encode_matmul(config: dict, shape: dict) -> list[float]:
    """Encode matmul (config, shape) pair to feature vector."""
    return [
        _safe_log2(config.get("BLOCK_SIZE_M", config.get("BLOCK_M", 1))),
        _safe_log2(config.get("BLOCK_SIZE_N", config.get("BLOCK_N", 1))),
        _safe_log2(config.get("BLOCK_SIZE_K", config.get("BLOCK_K", 1))),
        float(config.get("GROUP_SIZE_M", 1)),
        float(config.get("num_warps", 1)),
        float(config.get("num_stages", 1)),
        _safe_log2(shape.get("M", 1)),
        _safe_log2(shape.get("N", 1)),
        _safe_log2(shape.get("K", 1)),
    ]


_ENCODERS: dict[str, Any] = {
    "rmsnorm": _encode_rmsnorm,
    "attention": _encode_attention,
    "qk_norm_rope": _encode_qk_norm_rope,
    "matmul": _encode_matmul,
}

# Stable ordering for operator one-hot encoding
_OPERATOR_LIST: list[str] = sorted(_ENCODERS.keys())


def encode_features(operator: str, config: dict, shape: dict) -> list[float]:
    """Encode (operator, config, shape) into a feature vector.

    Uses operator-specific encoding with log2 scaling for dimensional features.
    Falls back to matmul encoding for unknown operators.
    """
    encoder = _ENCODERS.get(operator, _encode_matmul)
    return encoder(config, shape)


# ---------------------------------------------------------------------------
# Cross-operator feature encoding (fixed-width)
# ---------------------------------------------------------------------------

# Maximum feature count across all per-operator encoders.  Vectors shorter
# than this are zero-padded so that a single model can consume any operator.
_MAX_OPERATOR_FEATURES = max(
    6,   # rmsnorm
    10,  # attention
    7,   # qk_norm_rope
    9,   # matmul
)


def _encode_cross_operator(operator: str, config: dict, shape: dict) -> list[float]:
    """Fixed-width feature vector that works across ALL operators.

    Layout: [operator one-hot | config params (log2 block sizes, warps,
    stages) | shape params (log2 dims)] — zero-padded to a common width so
    the same GBR model handles every operator.
    """
    # Operator one-hot
    one_hot = [0.0] * len(_OPERATOR_LIST)
    if operator in _OPERATOR_LIST:
        one_hot[_OPERATOR_LIST.index(operator)] = 1.0

    # Per-operator features, zero-padded
    raw = encode_features(operator, config, shape)
    padded = raw + [0.0] * (_MAX_OPERATOR_FEATURES - len(raw))

    return one_hot + padded


# ---------------------------------------------------------------------------
# Ranking Surrogate (cross-operator GBR)
# ---------------------------------------------------------------------------

@dataclass
class RankingSurrogate:
    """Cross-operator ranking surrogate using GradientBoostingRegressor.

    Instead of predicting absolute throughput per-operator (which fails at
    R² < 0 due to low intra-operator variance), this model pools data across
    ALL operators and predicts *relative* performance.  A single GBR is
    trained on (operator_onehot, config_features, shape_features) -> metric,
    and at inference time we rank candidate configs by predicted score.

    The key insight: within one operator on one GPU, configs differ by ±20%,
    but across operators variance spans 22-250 GB/s.  A cross-operator model
    learns transferable patterns ("larger BLOCK_SIZE helps bandwidth-bound
    ops") that a per-operator GP cannot.
    """

    model: Any = None
    scaler_X: Any = None
    _is_fitted: bool = False
    _n_features: int = 0
    _known_operators: list[str] = field(default_factory=list)

    def fit(
        self,
        database: Any,
        hardware: str,
        *,
        operators: list[str] | None = None,
    ) -> dict:
        """Train the GBR on ALL operators for the given hardware.

        Args:
            database: ConfigDatabase instance.
            hardware: Hardware identifier to filter on.
            operators: Optional whitelist; defaults to all operators found.

        Returns:
            Dict with fit stats: n_samples, r_squared, status.
        """
        X_raw: list[list[float]] = []
        y_raw: list[float] = []

        for key, record in database.records.items():
            parts = key.split(":")
            if len(parts) == 3:
                rec_op, _, rec_hw = parts
            elif len(parts) == 2:
                rec_op, rec_hw = "matmul", parts[1]
            else:
                continue
            if hardware and rec_hw != hardware:
                continue
            if operators and rec_op not in operators:
                continue

            shape = record.shape if isinstance(record.shape, dict) else record.get("shape", {})
            results = record.results if isinstance(record.results, list) else record.get("results", [])

            for result in results:
                if not result.get("correct"):
                    continue
                metric = result.get("tflops") or result.get("gb_per_s") or 0.0
                if metric <= 0:
                    continue
                config = result.get("config", {})
                features = _encode_cross_operator(rec_op, config, shape)
                X_raw.append(features)
                y_raw.append(float(metric))

        n_samples = len(X_raw)
        if n_samples < 10:
            self._is_fitted = False
            return {"status": "insufficient_data", "n_samples": n_samples, "r_squared": 0.0}

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self._is_fitted = False
            return {"status": "sklearn_unavailable", "n_samples": n_samples, "r_squared": 0.0}

        X = np.array(X_raw, dtype=np.float64)
        y = np.array(y_raw, dtype=np.float64)

        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        self._n_features = X.shape[1]

        # Collect known operators
        seen_ops: set[str] = set()
        for key in database.records:
            parts = key.split(":")
            if len(parts) == 3:
                seen_ops.add(parts[0])
            elif len(parts) == 2:
                seen_ops.add("matmul")
        self._known_operators = sorted(seen_ops)

        # R² on training data
        y_pred = self.model.predict(X_scaled)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

        return {
            "status": "trained",
            "n_samples": n_samples,
            "r_squared": round(float(r_squared), 4),
            "n_operators": len(self._known_operators),
        }

    def predict_score(self, operator: str, config: dict, shape: dict) -> float:
        """Predict a relative performance score for one (operator, config, shape).

        The score is on the same scale as the training metric (tflops or GB/s)
        but should only be used for *ranking*, not as an absolute estimate.
        """
        if not self._is_fitted or self.model is None:
            return 0.0
        features = _encode_cross_operator(operator, config, shape)
        x = np.array([features], dtype=np.float64)
        x_scaled = self.scaler_X.transform(x)
        return float(self.model.predict(x_scaled)[0])

    def recommend_configs_by_ranking(
        self,
        operator: str,
        shape: dict,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score all candidate configs and return the top-k by predicted rank.

        Each returned dict includes the original config keys plus a
        ``predicted_score`` field.  Results are in descending score order.
        """
        if not candidates:
            return []

        scored: list[tuple[float, dict]] = []
        for config in candidates:
            score = self.predict_score(operator, config, shape)
            scored.append((score, config))

        scored.sort(key=lambda t: -t[0])

        results: list[dict] = []
        for score, config in scored[:top_k]:
            results.append({**config, "predicted_score": round(score, 4)})
        return results

    def evaluate_ranking(
        self,
        database: Any,
        hardware: str,
        *,
        n_splits: int = 5,
    ) -> dict:
        """K-fold cross-validated ranking evaluation.

        Splits the dataset into *n_splits* folds, trains on k-1, and
        measures Spearman rho between predicted and actual ordering on the
        held-out fold.

        Returns dict with per-fold rho and mean_spearman_rho.
        """
        from scipy.stats import spearmanr

        # Collect all data points
        X_raw: list[list[float]] = []
        y_raw: list[float] = []

        for key, record in database.records.items():
            parts = key.split(":")
            if len(parts) == 3:
                rec_op, _, rec_hw = parts
            elif len(parts) == 2:
                rec_op, rec_hw = "matmul", parts[1]
            else:
                continue
            if hardware and rec_hw != hardware:
                continue

            shape = record.shape if isinstance(record.shape, dict) else record.get("shape", {})
            results = record.results if isinstance(record.results, list) else record.get("results", [])

            for result in results:
                if not result.get("correct"):
                    continue
                metric = result.get("tflops") or result.get("gb_per_s") or 0.0
                if metric <= 0:
                    continue
                config = result.get("config", {})
                features = _encode_cross_operator(
                    rec_op if len(key.split(":")) == 3 else "matmul",
                    config, shape,
                )
                X_raw.append(features)
                y_raw.append(float(metric))

        n = len(X_raw)
        if n < n_splits * 2:
            return {"status": "insufficient_data", "n_samples": n, "mean_spearman_rho": 0.0, "folds": []}

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"status": "sklearn_unavailable", "n_samples": n, "mean_spearman_rho": 0.0, "folds": []}

        X = np.array(X_raw, dtype=np.float64)
        y = np.array(y_raw, dtype=np.float64)

        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        fold_size = n // n_splits

        fold_results: list[dict] = []
        for fold_i in range(n_splits):
            start = fold_i * fold_size
            end = start + fold_size if fold_i < n_splits - 1 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train = y[train_idx]
            y_test = y[test_idx]

            gbr = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            gbr.fit(X_train, y_train)
            y_pred = gbr.predict(X_test)

            if len(y_test) >= 2:
                rho, _ = spearmanr(y_test, y_pred)
                rho = float(rho) if not np.isnan(rho) else 0.0
            else:
                rho = 0.0

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

            fold_results.append({
                "fold": fold_i,
                "n_test": len(test_idx),
                "spearman_rho": round(rho, 4),
                "r_squared": round(float(r2), 4),
            })

        mean_rho = float(np.mean([f["spearman_rho"] for f in fold_results]))
        mean_r2 = float(np.mean([f["r_squared"] for f in fold_results]))

        return {
            "status": "evaluated",
            "n_samples": n,
            "n_splits": n_splits,
            "mean_spearman_rho": round(mean_rho, 4),
            "mean_r_squared": round(mean_r2, 4),
            "folds": fold_results,
        }


# ---------------------------------------------------------------------------
# GP Surrogate Model
# ---------------------------------------------------------------------------

@dataclass
class GPSurrogate:
    """Gaussian Process surrogate for cross-shape config transfer.

    Trained on (config_features, shape_features) -> metric pairs from the
    ConfigDatabase. Predicts performance for unseen shapes, enabling
    config transfer without GPU evaluation.

    Uses sklearn's GaussianProcessRegressor with an RBF + WhiteKernel.
    Falls back to the existing GBR cost model if GP fails (numerical
    instability on large datasets).
    """

    operator: str
    gp: Any = None
    scaler_X: Any = None
    scaler_y: Any = None
    _fallback_mean: float = 0.0
    _is_fitted: bool = False
    _n_features: int = 0

    def fit(self, database: Any, hardware: str) -> dict:
        """Train the GP on all results for this operator+hardware.

        Args:
            database: ConfigDatabase instance with .records dict.
            hardware: Hardware identifier to filter on.

        Returns:
            Dict with fit stats: n_samples, r_squared, marginal_log_likelihood, status.
        """
        X_raw: list[list[float]] = []
        y_raw: list[float] = []

        for key, record in database.records.items():
            parts = key.split(":")
            if len(parts) == 3:
                rec_op, _, rec_hw = parts
            elif len(parts) == 2:
                rec_op, rec_hw = "matmul", parts[1]
            else:
                continue

            if rec_op != self.operator:
                continue
            if hardware and rec_hw != hardware:
                continue

            shape = record.shape if isinstance(record.shape, dict) else record.get("shape", {})

            results = record.results if isinstance(record.results, list) else record.get("results", [])
            for result in results:
                if not result.get("correct"):
                    continue
                metric = result.get("tflops") or result.get("gb_per_s") or 0.0
                if metric <= 0:
                    continue
                config = result.get("config", {})
                features = encode_features(self.operator, config, shape)
                X_raw.append(features)
                y_raw.append(float(metric))

        n_samples = len(X_raw)
        if n_samples < 5:
            self._is_fitted = False
            self._fallback_mean = float(np.mean(y_raw)) if y_raw else 0.0
            return {
                "status": "insufficient_data",
                "n_samples": n_samples,
                "r_squared": 0.0,
                "marginal_log_likelihood": 0.0,
            }

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self._is_fitted = False
            self._fallback_mean = float(np.mean(y_raw))
            return {
                "status": "sklearn_unavailable",
                "n_samples": n_samples,
                "r_squared": 0.0,
                "marginal_log_likelihood": 0.0,
            }

        X = np.array(X_raw, dtype=np.float64)
        y = np.array(y_raw, dtype=np.float64)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        kernel = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(
            noise_level=0.1, noise_level_bounds=(1e-5, 1.0)
        )

        # Cap dataset size for GP (cubic complexity); subsample if needed.
        max_gp_samples = 500
        if n_samples > max_gp_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, size=max_gp_samples, replace=False)
            X_scaled = X_scaled[idx]
            y_scaled = y_scaled[idx]
            n_samples_used = max_gp_samples
        else:
            n_samples_used = n_samples

        try:
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=3,
                normalize_y=False,  # we already scaled
                alpha=1e-6,
                random_state=42,
            )
            self.gp.fit(X_scaled, y_scaled)
            self._is_fitted = True
            self._n_features = X.shape[1]
            self._fallback_mean = float(np.mean(y_raw))

            # Compute R^2 on training data
            y_pred_scaled = self.gp.predict(X_scaled)
            ss_res = np.sum((y_scaled - y_pred_scaled) ** 2)
            ss_tot = np.sum((y_scaled - np.mean(y_scaled)) ** 2)
            r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

            mll = self.gp.log_marginal_likelihood_value_

            return {
                "status": "trained",
                "n_samples": n_samples,
                "n_samples_used": n_samples_used,
                "r_squared": round(float(r_squared), 4),
                "marginal_log_likelihood": round(float(mll), 4),
            }

        except Exception as exc:
            LOGGER.warning("GP fit failed: %s — falling back to mean", exc)
            self.gp = None
            self._is_fitted = False
            self._fallback_mean = float(np.mean(y_raw))
            return {
                "status": "fit_failed",
                "n_samples": n_samples,
                "r_squared": 0.0,
                "marginal_log_likelihood": 0.0,
                "error": str(exc),
            }

    def predict(self, config: dict, shape: dict) -> tuple[float, float]:
        """Predict (mean, std) for an unseen (config, shape) pair.

        Returns the predicted throughput and its uncertainty.  If the GP is
        not fitted, returns (fallback_mean, inf).
        """
        if not self._is_fitted or self.gp is None:
            return (self._fallback_mean, float("inf"))

        features = encode_features(self.operator, config, shape)
        x = np.array([features], dtype=np.float64)
        x_scaled = self.scaler_X.transform(x)

        y_mean_scaled, y_std_scaled = self.gp.predict(x_scaled, return_std=True)

        # Inverse-transform the prediction
        y_mean = self.scaler_y.inverse_transform(
            y_mean_scaled.reshape(-1, 1)
        ).ravel()[0]
        # Scale std by the target scaler's scale
        y_std = float(y_std_scaled[0]) * float(self.scaler_y.scale_[0])

        return (float(y_mean), float(y_std))

    def recommend_configs(
        self,
        shape: dict,
        candidate_configs: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rank candidate configs for a new shape by predicted performance.

        Returns the top-k configs sorted by predicted mean throughput
        (descending). Each returned dict includes the original config plus
        'predicted_mean' and 'predicted_std' keys.
        """
        if not candidate_configs:
            return []

        scored: list[tuple[float, float, dict]] = []
        for config in candidate_configs:
            mean, std = self.predict(config, shape)
            scored.append((mean, std, config))

        # Sort by predicted mean descending
        scored.sort(key=lambda t: -t[0])

        results = []
        for mean, std, config in scored[:top_k]:
            results.append({
                **config,
                "predicted_mean": round(mean, 4),
                "predicted_std": round(std, 4),
            })
        return results

    def transfer_from_nearest(
        self,
        target_shape: dict,
        database: Any,
        hardware: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Find the best configs from the nearest known shape and predict
        their performance on the target shape.

        Strategy:
        1. Collect all unique shapes from the database for this operator+hw.
        2. Compute L2 distance in feature space to find the nearest shape.
        3. Gather the best configs from that shape.
        4. Predict their performance on the target shape via the GP.

        Returns a list of dicts with config params + predicted_mean + predicted_std.
        """
        # Collect known shapes and their best configs
        shape_configs: dict[str, tuple[dict, list[dict]]] = {}

        for key, record in database.records.items():
            parts = key.split(":")
            if len(parts) == 3:
                rec_op, _, rec_hw = parts
            elif len(parts) == 2:
                rec_op, rec_hw = "matmul", parts[1]
            else:
                continue
            if rec_op != self.operator:
                continue
            if hardware and rec_hw != hardware:
                continue

            shape = record.shape if isinstance(record.shape, dict) else record.get("shape", {})
            shape_key = key

            # Collect correct results sorted by metric
            correct_results = []
            results = record.results if isinstance(record.results, list) else record.get("results", [])
            for r in results:
                if r.get("correct"):
                    metric = r.get("tflops") or r.get("gb_per_s") or 0.0
                    if metric > 0:
                        correct_results.append((metric, r.get("config", {})))

            if correct_results:
                correct_results.sort(key=lambda t: -t[0])
                configs = [c for _, c in correct_results[:top_k]]
                shape_configs[shape_key] = (shape, configs)

        if not shape_configs:
            return []

        # Encode target shape features (use a dummy config for distance calc)
        dummy_config = {}
        target_features = encode_features(self.operator, dummy_config, target_shape)
        # Only use shape-related features (skip config features)
        # We compute distance on full feature vectors but with zeroed config
        target_vec = np.array(target_features, dtype=np.float64)

        # Find nearest shape
        best_dist = float("inf")
        best_configs: list[dict] = []

        for shape_key, (shape, configs) in shape_configs.items():
            known_features = encode_features(self.operator, dummy_config, shape)
            known_vec = np.array(known_features, dtype=np.float64)
            dist = float(np.linalg.norm(target_vec - known_vec))
            if dist < best_dist:
                best_dist = dist
                best_configs = configs

        if not best_configs:
            return []

        # Predict performance of nearest-shape configs on target shape
        return self.recommend_configs(target_shape, best_configs, top_k=top_k)

    def evaluate_transfer(
        self,
        database: Any,
        hardware: str,
    ) -> dict:
        """Leave-one-shape-out evaluation of cross-shape transfer quality.

        For each shape bucket:
        1. Hold out all results for that shape.
        2. Train the GP on remaining shapes.
        3. Predict the best config for the held-out shape.
        4. Compare predicted ranking vs actual ranking.

        Returns:
            Dict with per-shape results and aggregate metrics:
            - mean_spearman_rho
            - mean_top5_overlap
            - mean_predicted_rank_of_actual_best
            - per_shape: list of per-shape dicts
        """
        from scipy.stats import spearmanr

        # Collect shape buckets
        shape_keys: list[str] = []
        for key in database.records:
            parts = key.split(":")
            if len(parts) == 3:
                rec_op, _, rec_hw = parts
            elif len(parts) == 2:
                rec_op, rec_hw = "matmul", parts[1]
            else:
                continue
            if rec_op != self.operator:
                continue
            if hardware and rec_hw != hardware:
                continue
            shape_keys.append(key)

        if len(shape_keys) < 2:
            return {
                "status": "insufficient_shapes",
                "n_shapes": len(shape_keys),
                "mean_spearman_rho": 0.0,
                "mean_top5_overlap": 0.0,
                "mean_predicted_rank_of_actual_best": 0.0,
                "per_shape": [],
            }

        per_shape_results: list[dict] = []

        for holdout_key in shape_keys:
            # Build a temporary database without the held-out shape
            from research_engine.triton_kernels import ConfigDatabase, ShapeRecord
            import tempfile
            import json
            from pathlib import Path

            # Serialize remaining records
            remaining = {}
            for k, rec in database.records.items():
                if k == holdout_key:
                    continue
                remaining[k] = {
                    "shape_key": rec.shape_key if hasattr(rec, "shape_key") else k,
                    "shape": rec.shape if isinstance(rec.shape, dict) else rec.get("shape", {}),
                    "best_config_id": rec.best_config_id if hasattr(rec, "best_config_id") else "",
                    "best_tflops": rec.best_tflops if hasattr(rec, "best_tflops") else 0.0,
                    "results": rec.results if isinstance(rec.results, list) else rec.get("results", []),
                }

            tmp_path = Path(tempfile.mktemp(suffix=".json"))
            tmp_path.write_text(json.dumps({"records": remaining}))

            try:
                tmp_db = ConfigDatabase(path=tmp_path)
                surrogate = GPSurrogate(operator=self.operator)
                fit_result = surrogate.fit(tmp_db, hardware)

                if fit_result["status"] != "trained":
                    continue

                # Get held-out shape's actual results
                holdout_record = database.records[holdout_key]
                holdout_shape = holdout_record.shape if isinstance(holdout_record.shape, dict) else holdout_record.get("shape", {})
                holdout_results = holdout_record.results if isinstance(holdout_record.results, list) else holdout_record.get("results", [])

                actual_configs: list[tuple[float, dict]] = []
                for r in holdout_results:
                    if r.get("correct"):
                        metric = r.get("tflops") or r.get("gb_per_s") or 0.0
                        if metric > 0:
                            actual_configs.append((metric, r.get("config", {})))

                if len(actual_configs) < 2:
                    continue

                actual_configs.sort(key=lambda t: -t[0])
                actual_metrics = [m for m, _ in actual_configs]
                configs_only = [c for _, c in actual_configs]

                # Predict for each config
                predicted: list[tuple[float, int]] = []
                for idx, config in enumerate(configs_only):
                    mean, _ = surrogate.predict(config, holdout_shape)
                    predicted.append((mean, idx))

                predicted.sort(key=lambda t: -t[0])
                predicted_order = [idx for _, idx in predicted]

                # Metrics
                actual_order = list(range(len(configs_only)))

                # Spearman rho
                if len(actual_order) >= 2:
                    rho, _ = spearmanr(actual_order, predicted_order)
                    rho = float(rho) if not np.isnan(rho) else 0.0
                else:
                    rho = 0.0

                # Predicted rank of actual best (0-indexed)
                actual_best_idx = 0  # already sorted best-first
                predicted_rank = predicted_order.index(actual_best_idx) if actual_best_idx in predicted_order else len(predicted_order)

                # Top-5 overlap
                top5_actual = set(actual_order[:5])
                top5_predicted = set(predicted_order[:5])
                overlap = len(top5_actual & top5_predicted) / max(len(top5_actual), 1)

                per_shape_results.append({
                    "shape_key": holdout_key,
                    "spearman_rho": round(rho, 4),
                    "predicted_rank_of_actual_best": predicted_rank,
                    "top5_overlap": round(overlap, 4),
                    "n_configs": len(configs_only),
                })
            finally:
                tmp_path.unlink(missing_ok=True)

        if not per_shape_results:
            return {
                "status": "no_valid_shapes",
                "n_shapes": len(shape_keys),
                "mean_spearman_rho": 0.0,
                "mean_top5_overlap": 0.0,
                "mean_predicted_rank_of_actual_best": 0.0,
                "per_shape": [],
            }

        mean_rho = float(np.mean([r["spearman_rho"] for r in per_shape_results]))
        mean_overlap = float(np.mean([r["top5_overlap"] for r in per_shape_results]))
        mean_rank = float(np.mean([r["predicted_rank_of_actual_best"] for r in per_shape_results]))

        return {
            "status": "evaluated",
            "n_shapes": len(shape_keys),
            "n_evaluated": len(per_shape_results),
            "mean_spearman_rho": round(mean_rho, 4),
            "mean_top5_overlap": round(mean_overlap, 4),
            "mean_predicted_rank_of_actual_best": round(mean_rank, 2),
            "per_shape": per_shape_results,
        }


# ---------------------------------------------------------------------------
# GP-Guided Config Selector (integrates with BanditSelector)
# ---------------------------------------------------------------------------

@dataclass
class GPGuidedSelector:
    """Uses GP surrogate to warm-start the bandit with high-quality proposals.

    Fits the GP on the current ConfigDatabase, then for each target shape
    uses recommend_configs() to rank candidates. The top-ranked configs are
    passed to the bandit as warm proposals, replacing or augmenting the
    existing cost-model-based proposals.
    """

    operator: str
    top_k: int = 5
    _surrogate: GPSurrogate | None = field(default=None, init=False)
    _fit_stats: dict = field(default_factory=dict, init=False)

    def fit(self, database: Any, hardware: str) -> dict:
        """Fit the GP surrogate on the database."""
        self._surrogate = GPSurrogate(operator=self.operator)
        self._fit_stats = self._surrogate.fit(database, hardware)
        return self._fit_stats

    def propose_configs(
        self,
        *,
        shapes: list[dict],
        candidate_configs: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Generate warm proposals for each target shape.

        Returns a de-duplicated list of the top-k configs across all target
        shapes, ranked by average predicted performance.
        """
        if self._surrogate is None or not self._surrogate._is_fitted:
            return []

        k = top_k or self.top_k
        if not shapes or not candidate_configs:
            return []

        # Score each candidate across all target shapes
        config_scores: dict[int, tuple[float, dict]] = {}
        for i, config in enumerate(candidate_configs):
            total_mean = 0.0
            for shape in shapes:
                mean, _ = self._surrogate.predict(config, shape)
                total_mean += mean
            avg_mean = total_mean / len(shapes)
            config_scores[i] = (avg_mean, config)

        # Sort by average predicted performance
        ranked = sorted(config_scores.values(), key=lambda t: -t[0])
        return [config for _, config in ranked[:k]]

    @property
    def fit_stats(self) -> dict:
        return self._fit_stats
