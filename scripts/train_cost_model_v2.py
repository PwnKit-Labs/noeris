#!/usr/bin/env python3
"""Train cost model v2 on all available benchmark data (multi-hardware).

Loads every ConfigDatabase JSON found under .noeris/ and trains a
GradientBoostingRegressor with hardware one-hot features so the model
learns hardware-specific preferences (e.g. T4 prefers num_warps=1).

Usage:
    python scripts/train_cost_model_v2.py

Outputs:
    models/cost_model_v2.pkl   — trained model
    stdout                     — R², Spearman rho, feature importances, per-op breakdown

LOCAL ONLY. No Modal, no GPU required.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from research_engine.cost_model import (
    CostModel,
    FEATURE_NAMES,
    _encode_categoricals,
    _operator_id,
    extract_features,
)


# ---------------------------------------------------------------------------
# 1. Discover and load all databases
# ---------------------------------------------------------------------------

NOERIS_DIR = REPO / ".noeris"

# Canonical DB paths + any other JSONs that look like config databases
DB_CANDIDATES = [
    NOERIS_DIR / "colab-configs.json",       # T4 Colab data
    NOERIS_DIR / "triton-configs.json",       # CI A100 runs
    NOERIS_DIR / "cost-model-training.json",  # Original collection run
]

# Also scan for any other top-level JSONs in .noeris/ that have a "records" key
if NOERIS_DIR.exists():
    for p in sorted(NOERIS_DIR.glob("*.json")):
        if p not in DB_CANDIDATES:
            try:
                data = json.loads(p.read_text())
                if "records" in data:
                    DB_CANDIDATES.append(p)
            except Exception:
                pass


def load_all_data(
    db_paths: list[Path],
) -> tuple[list[list[float]], list[float], list[str], list[str]]:
    """Load (X, y, operators, hardwares) from all databases."""
    X: list[list[float]] = []
    y: list[float] = []
    operators: list[str] = []
    hardwares: list[str] = []

    seen_paths: set[str] = set()
    for db_path in db_paths:
        if not db_path.exists():
            continue
        rp = str(db_path.resolve())
        if rp in seen_paths:
            continue
        seen_paths.add(rp)

        try:
            data = json.loads(db_path.read_text())
        except Exception:
            print(f"  [WARN] Could not parse {db_path}")
            continue

        records = data.get("records", {})
        n_added = 0
        for key, record in records.items():
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

                raw = extract_features(
                    shape=shape,
                    config=config,
                    hardware=hardware,
                    operator=operator,
                )
                X.append(_encode_categoricals(raw))
                y.append(float(metric))
                operators.append(operator)
                hardwares.append(hardware)
                n_added += 1

        if n_added > 0:
            print(f"  Loaded {n_added:>5d} points from {db_path.name}")

    return X, y, operators, hardwares


# ---------------------------------------------------------------------------
# 2. Training with cross-validation
# ---------------------------------------------------------------------------

def train_with_cv(
    X: list[list[float]],
    y: list[float],
    n_folds: int = 5,
) -> tuple:
    """Train GBR with k-fold cross-validation. Returns (model, cv_scores, oof_preds)."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    X_arr = np.array(X)
    y_arr = np.array(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_r2_scores = []
    oof_preds = np.zeros(len(y_arr))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
        fold_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=3,
            random_state=42,
        )
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])
        score = fold_model.score(X_arr[val_idx], y_arr[val_idx])
        cv_r2_scores.append(score)
        oof_preds[val_idx] = fold_model.predict(X_arr[val_idx])

    # Train final model on all data
    final_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=3,
        random_state=42,
    )
    final_model.fit(X_arr, y_arr)

    return final_model, cv_r2_scores, oof_preds


# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------

def compute_spearman(y_true, y_pred):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    rho, pval = spearmanr(y_true, y_pred)
    return rho, pval


def compute_r2(y_true, y_pred):
    """R-squared."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# 4. Feature importance analysis
# ---------------------------------------------------------------------------

def get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Return sorted feature importance dict."""
    importances = model.feature_importances_
    result = {}
    for name, imp in zip(feature_names, importances):
        result[name] = float(imp)
    return dict(sorted(result.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# 5. Leave-one-operator-out evaluation
# ---------------------------------------------------------------------------

def leave_one_operator_out(
    X: list[list[float]],
    y: list[float],
    operators: list[str],
) -> dict[str, dict[str, float]]:
    """Train on all operators except one, predict on held-out operator."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor

    X_arr = np.array(X)
    y_arr = np.array(y)
    ops_arr = np.array(operators)

    unique_ops = sorted(set(operators))
    results = {}

    for held_out_op in unique_ops:
        train_mask = ops_arr != held_out_op
        test_mask = ops_arr == held_out_op
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_test < 5 or n_train < 20:
            results[held_out_op] = {
                "r2": float("nan"),
                "spearman_rho": float("nan"),
                "n_test": n_test,
                "n_train": n_train,
                "note": "insufficient data",
            }
            continue

        m = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=3,
            random_state=42,
        )
        m.fit(X_arr[train_mask], y_arr[train_mask])
        preds = m.predict(X_arr[test_mask])

        r2 = compute_r2(y_arr[test_mask], preds)
        try:
            rho, _ = compute_spearman(y_arr[test_mask], preds)
        except Exception:
            rho = float("nan")

        results[held_out_op] = {
            "r2": round(float(r2), 4),
            "spearman_rho": round(float(rho), 4),
            "n_test": n_test,
            "n_train": n_train,
        }

    return results


# ---------------------------------------------------------------------------
# 6. Compare with original model
# ---------------------------------------------------------------------------

def compare_with_original(
    new_model,
    X: list[list[float]],
    y: list[float],
    original_path: Path,
) -> dict | None:
    """Compare new model vs original on the same data."""
    if not original_path.exists():
        return None
    try:
        old = CostModel.load(original_path)
        if old.regressor is None:
            return {"status": "original_has_no_regressor"}
    except Exception as e:
        return {"status": f"load_error: {e}"}

    import numpy as np
    X_arr = np.array(X)
    y_arr = np.array(y)

    old_preds = old.regressor.predict(X_arr)
    new_preds = new_model.predict(X_arr)

    old_r2 = compute_r2(y_arr, old_preds)
    new_r2 = compute_r2(y_arr, new_preds)

    try:
        old_rho, _ = compute_spearman(y_arr, old_preds)
        new_rho, _ = compute_spearman(y_arr, new_preds)
    except Exception:
        old_rho = new_rho = float("nan")

    return {
        "original_r2": round(float(old_r2), 4),
        "new_r2": round(float(new_r2), 4),
        "original_spearman": round(float(old_rho), 4),
        "new_spearman": round(float(new_rho), 4),
        "original_training_size": old.training_size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import numpy as np

    print("=" * 70)
    print("Noeris Cost Model v2 — Multi-Hardware Training")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading databases...")
    X, y, operators, hardwares = load_all_data(DB_CANDIDATES)
    print(f"\n  Total: {len(X)} measurements")

    if len(X) < 20:
        print("\n  ERROR: Insufficient data for training (need >= 20 points).")
        sys.exit(1)

    # Summary stats
    hw_counts: dict[str, int] = defaultdict(int)
    op_counts: dict[str, int] = defaultdict(int)
    for hw in hardwares:
        hw_counts[hw] += 1
    for op in operators:
        op_counts[op] += 1

    print("\n  Hardware breakdown:")
    for hw, cnt in sorted(hw_counts.items(), key=lambda x: -x[1]):
        print(f"    {hw}: {cnt}")
    print("\n  Operator breakdown:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op}: {cnt}")

    # 2. Cross-validated training
    print("\n[2/6] Training GBR with 5-fold cross-validation...")
    model, cv_scores, oof_preds = train_with_cv(X, y)
    mean_cv_r2 = float(np.mean(cv_scores))
    std_cv_r2 = float(np.std(cv_scores))
    print(f"  CV R² = {mean_cv_r2:.4f} +/- {std_cv_r2:.4f}")
    print(f"  Per-fold: {[round(s, 4) for s in cv_scores]}")

    # OOF Spearman
    try:
        oof_rho, _ = compute_spearman(y, oof_preds)
        print(f"  OOF Spearman rho = {oof_rho:.4f}")
    except Exception:
        oof_rho = float("nan")
        print("  OOF Spearman: could not compute (scipy missing?)")

    # Full-data R²
    full_preds = model.predict(np.array(X))
    full_r2 = compute_r2(y, full_preds)
    print(f"  Full-data R² = {full_r2:.4f} (resubstitution, for reference)")

    # 3. Feature importances
    print("\n[3/6] Feature importances (top 10):")
    importances = get_feature_importances(model, FEATURE_NAMES)
    for rank, (name, imp) in enumerate(list(importances.items())[:10], 1):
        bar = "#" * int(imp * 200)
        print(f"  {rank:>2d}. {name:<25s} {imp:.4f}  {bar}")

    # 4. Leave-one-operator-out
    print("\n[4/6] Leave-one-operator-out evaluation:")
    loo_results = leave_one_operator_out(X, y, operators)
    for op, res in sorted(loo_results.items()):
        r2_str = f"R²={res['r2']:.4f}" if not np.isnan(res["r2"]) else "R²=N/A"
        rho_str = f"ρ={res['spearman_rho']:.4f}" if not np.isnan(res["spearman_rho"]) else "ρ=N/A"
        note = f"  ({res.get('note', '')})" if "note" in res else ""
        print(f"  {op:<16s} {r2_str}  {rho_str}  (n={res['n_test']}){note}")

    # 5. Compare with original
    print("\n[5/6] Comparing with original model...")
    original_paths = [
        NOERIS_DIR / "cost-model.pkl",
        NOERIS_DIR / "cost-model-a100-only.pkl",
    ]
    compared = False
    for orig_path in original_paths:
        comparison = compare_with_original(model, X, y, orig_path)
        if comparison and "original_r2" in comparison:
            print(f"  Original ({orig_path.name}):")
            print(f"    R² = {comparison['original_r2']:.4f}  (trained on {comparison['original_training_size']} points)")
            print(f"    Spearman = {comparison['original_spearman']:.4f}")
            print(f"  New model:")
            print(f"    R² = {comparison['new_r2']:.4f}  (trained on {len(X)} points)")
            print(f"    Spearman = {comparison['new_spearman']:.4f}")
            delta_r2 = comparison["new_r2"] - comparison["original_r2"]
            print(f"  Delta R² = {delta_r2:+.4f}")
            compared = True
        elif comparison:
            print(f"  {orig_path.name}: {comparison.get('status', 'unknown')}")
    if not compared:
        print("  No original model found for comparison.")

    # 6. Save
    print("\n[6/6] Saving model...")
    out_dir = REPO / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cost_model_v2.pkl"

    cost_model = CostModel()
    cost_model.regressor = model
    cost_model.training_size = len(X)
    # Compute mean_by_operator
    op_targets: dict[int, list[float]] = defaultdict(list)
    for op, metric in zip(operators, y):
        op_targets[_operator_id(op)].append(metric)
    cost_model.mean_by_operator = {
        op_id: sum(vals) / len(vals)
        for op_id, vals in op_targets.items()
    }
    cost_model.save(out_path)
    print(f"  Saved to {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Training points:  {len(X)}")
    print(f"  Hardware types:   {len(hw_counts)} ({', '.join(hw_counts.keys())})")
    print(f"  Operators:        {len(op_counts)} ({', '.join(op_counts.keys())})")
    print(f"  CV R²:            {mean_cv_r2:.4f} +/- {std_cv_r2:.4f}")
    if not np.isnan(oof_rho):
        print(f"  OOF Spearman ρ:   {oof_rho:.4f}")
    print(f"  Top feature:      {list(importances.keys())[0]}")
    print(f"  Model saved to:   {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
