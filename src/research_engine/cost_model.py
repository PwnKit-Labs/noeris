"""Learned cost model for predicting kernel runtime from (shape, config, hardware).

The key technical novelty of Noeris is pairing an LLM proposer with a learned
cost model trained on the shape-indexed cross-run database. The LLM handles
novelty (proposing configs outside the training distribution, reasoning about
hardware characteristics). The cost model handles filtering (eliminating
configs unlikely to be winners before they consume GPU time).

## Training data

Every benchmark result in ConfigDatabase is a training point:

    features = {shape_dims, config_params, hardware_id, operator_id}
    target = tflops_or_gb_per_s

After accumulating ~1000 results across operators we can train a small
gradient-boosted regressor that generalizes well.

## Usage

    model = CostModel()
    model.train_from_databases(["/path/to/db.json"])
    model.save("model.pkl")

    # At selection time:
    model = CostModel.load("model.pkl")
    predictions = model.predict_many([
        (shape, config, hardware, operator)
        for ...
    ])

## Dependencies

Uses scikit-learn GradientBoostingRegressor (ships with sklearn, no extra
install). Falls back to a mean-baseline if sklearn is unavailable.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HARDWARE_IDS = {
    "none": 0,
    "NVIDIA A100-SXM4-40GB": 1,
    "NVIDIA A100-SXM4-80GB": 2,
    "NVIDIA A100": 3,
    "NVIDIA H100": 4,
    "NVIDIA H100 80GB HBM3": 5,
    "NVIDIA T4": 6,
    "NVIDIA A10G": 7,
}

OPERATOR_IDS = {
    "matmul": 0,
    "rmsnorm": 1,
    "softmax": 2,
    "layernorm": 3,
    "cross_entropy": 4,
    "attention": 5,
}


def _hardware_id(hw_name: str) -> int:
    return HARDWARE_IDS.get(hw_name, 0)


def _operator_id(op_name: str) -> int:
    return OPERATOR_IDS.get(op_name, -1)


# Canonical feature vector layout. The cost model uses a fixed-width vector
# so it can be a single regressor rather than per-operator models.
FEATURE_NAMES = [
    "operator_id",
    "hardware_id",
    # Shape dims (5 slots, operator-specific semantics, padded with 0)
    "shape_0", "shape_1", "shape_2", "shape_3", "shape_4",
    "log_shape_prod",   # log of all non-zero shape dims multiplied
    "log_shape_max",
    "log_shape_min",
    # Config parameters (8 slots, operator-specific semantics, padded with 0)
    "config_0", "config_1", "config_2", "config_3",
    "config_4", "config_5", "config_6", "config_7",
    # Derived features
    "log_tile_area",
    "log_num_warps_total",
    "log_num_stages",
]


def extract_features(
    *,
    shape: dict,
    config: dict,
    hardware: str,
    operator: str,
) -> list[float]:
    """Extract a fixed-width feature vector from (shape, config, hardware, operator)."""
    import math

    op_id = _operator_id(operator)
    hw_id = _hardware_id(hardware)

    # Shape dims per operator
    shape_vals: list[int] = []
    if operator == "matmul":
        shape_vals = [shape.get("M", 0), shape.get("N", 0), shape.get("K", 0)]
    elif operator in ("rmsnorm", "layernorm"):
        shape_vals = [shape.get("n_rows", 0), shape.get("hidden_dim", 0)]
    elif operator in ("softmax", "cross_entropy"):
        shape_vals = [shape.get("n_rows", 0), shape.get("n_cols", 0)]
    elif operator == "attention":
        shape_vals = [
            shape.get("batch", 0),
            shape.get("heads", 0),
            shape.get("seq_len", 0),
            shape.get("head_dim", 0),
            1 if shape.get("is_causal", False) else 0,
        ]

    shape_padded = (shape_vals + [0] * 5)[:5]
    nonzero = [v for v in shape_padded if v > 0]
    log_prod = math.log(1 + math.prod(nonzero)) if nonzero else 0.0
    log_max = math.log(1 + max(shape_padded)) if shape_padded else 0.0
    log_min = math.log(1 + min([v for v in shape_padded if v > 0] or [0])) if nonzero else 0.0

    # Config params per operator. Canonicalized: BLOCK_SIZE_M, BLOCK_SIZE_N,
    # BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages, BLOCK_SIZE, j_unroll.
    config_vals: list[int] = [0] * 8
    if operator == "matmul":
        config_vals[0] = int(config.get("BLOCK_SIZE_M", 0))
        config_vals[1] = int(config.get("BLOCK_SIZE_N", 0))
        config_vals[2] = int(config.get("BLOCK_SIZE_K", 0))
        config_vals[3] = int(config.get("GROUP_SIZE_M", 0))
        config_vals[4] = int(config.get("num_warps", 0))
        config_vals[5] = int(config.get("num_stages", 0))
    elif operator == "attention":
        config_vals[0] = int(config.get("BLOCK_M", 0))
        config_vals[1] = int(config.get("BLOCK_N", 0))
        config_vals[4] = int(config.get("num_warps", 0))
        config_vals[5] = int(config.get("num_stages", 0))
    else:
        # memory-bound: BLOCK_SIZE + num_warps + num_stages
        config_vals[6] = int(config.get("BLOCK_SIZE", 0))
        config_vals[4] = int(config.get("num_warps", 0))
        config_vals[5] = int(config.get("num_stages", 0))

    bm = max(config_vals[0], 1)
    bn = max(config_vals[1], 1)
    bs = max(config_vals[6], 1)
    nw = max(config_vals[4], 1)
    ns = max(config_vals[5], 1)

    log_tile_area = math.log(bm * bn) if operator in ("matmul", "attention") else math.log(bs)
    log_num_warps = math.log(nw)
    log_stages = math.log(ns)

    return [
        float(op_id),
        float(hw_id),
        *[float(v) for v in shape_padded],
        log_prod,
        log_max,
        log_min,
        *[float(v) for v in config_vals],
        log_tile_area,
        log_num_warps,
        log_stages,
    ]


@dataclass
class CostModel:
    """Predicts kernel runtime metric (tflops or gb_per_s) from features.

    Uses sklearn GradientBoostingRegressor if available, otherwise a simple
    per-operator mean baseline.
    """

    regressor: Any = None
    mean_by_operator: dict[int, float] = field(default_factory=dict)
    training_size: int = 0
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def train_from_databases(self, db_paths: list[str | Path]) -> dict:
        """Train the model from one or more ConfigDatabase JSON files."""
        X: list[list[float]] = []
        y: list[float] = []
        op_targets: dict[int, list[float]] = {}

        for db_path in db_paths:
            path = Path(db_path)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            records = data.get("records", {})
            for key, record in records.items():
                # key format: operator:bucket:hardware  OR  bucket:hardware (legacy)
                parts = key.split(":", 2)
                if len(parts) == 3:
                    operator, _, hardware = parts
                elif len(parts) == 2:
                    operator, hardware = "matmul", parts[1]
                else:
                    continue

                shape = record.get("shape", {})
                for result in record.get("results", []):
                    if not result.get("correct"):
                        continue
                    config = result.get("config", {})
                    metric = result.get("tflops") or result.get("gb_per_s") or 0
                    if metric <= 0:
                        continue
                    features = extract_features(
                        shape=shape,
                        config=config,
                        hardware=hardware,
                        operator=operator,
                    )
                    X.append(features)
                    y.append(float(metric))
                    op_targets.setdefault(_operator_id(operator), []).append(float(metric))

        self.training_size = len(X)
        self.mean_by_operator = {
            op_id: (sum(vals) / len(vals)) if vals else 0.0
            for op_id, vals in op_targets.items()
        }

        if len(X) < 20:
            self.regressor = None
            return {
                "status": "insufficient_data",
                "training_size": self.training_size,
                "operator_counts": {op: len(v) for op, v in op_targets.items()},
            }

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self.regressor = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=3,
                random_state=42,
            )
            self.regressor.fit(X, y)

            # Shuffled 80/20 holdout R^2 (avoids ordering leakage when data
            # is grouped by operator or hardware).
            import random as _random
            idx = list(range(len(X)))
            _random.Random(42).shuffle(idx)
            X_shuf = [X[i] for i in idx]
            y_shuf = [y[i] for i in idx]
            n = len(X_shuf)
            split = int(n * 0.8)
            if split < n:
                from sklearn.ensemble import GradientBoostingRegressor as _G
                aux = _G(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=3, random_state=42)
                aux.fit(X_shuf[:split], y_shuf[:split])
                score = aux.score(X_shuf[split:], y_shuf[split:])
            else:
                score = 1.0
            return {
                "status": "trained",
                "training_size": self.training_size,
                "holdout_r2": round(score, 4),
                "operator_counts": {op: len(v) for op, v in op_targets.items()},
            }
        except ImportError:
            self.regressor = None
            return {
                "status": "sklearn_unavailable",
                "training_size": self.training_size,
                "operator_counts": {op: len(v) for op, v in op_targets.items()},
            }

    def predict(
        self,
        *,
        shape: dict,
        config: dict,
        hardware: str,
        operator: str,
    ) -> float:
        features = extract_features(
            shape=shape,
            config=config,
            hardware=hardware,
            operator=operator,
        )
        if self.regressor is not None:
            try:
                return float(self.regressor.predict([features])[0])
            except Exception:
                pass
        # Fallback: operator mean
        op_id = _operator_id(operator)
        return float(self.mean_by_operator.get(op_id, 0.0))

    def predict_many(
        self,
        items: list[tuple[dict, dict, str, str]],
    ) -> list[float]:
        if not items:
            return []
        features = [
            extract_features(shape=s, config=c, hardware=h, operator=o)
            for s, c, h, o in items
        ]
        if self.regressor is not None:
            try:
                return [float(v) for v in self.regressor.predict(features)]
            except Exception:
                pass
        preds = []
        for _, _, _, operator in items:
            op_id = _operator_id(operator)
            preds.append(float(self.mean_by_operator.get(op_id, 0.0)))
        return preds

    def rank_configs(
        self,
        *,
        configs: list[dict],
        shapes: list[dict],
        hardware: str,
        operator: str,
        top_k: int | None = None,
    ) -> list[tuple[dict, float]]:
        """Rank configs by predicted mean metric across target shapes."""
        if not configs:
            return []

        scored: list[tuple[dict, float]] = []
        for config in configs:
            items = [(shape, config, hardware, operator) for shape in shapes]
            preds = self.predict_many(items)
            mean_pred = sum(preds) / len(preds) if preds else 0.0
            scored.append((config, mean_pred))

        scored.sort(key=lambda pair: -pair[1])
        if top_k is not None:
            scored = scored[:top_k]
        return scored
