# Cost Model Analysis

**Date:** 2026-04-11  
**Training data:** `.noeris/cost-model-training.json` (A100 run, 384 points)  
**Operators:** rmsnorm, softmax, layernorm, cross_entropy (4 ops × 5 shapes × ~19 configs avg)

---

## 1. Current State

### R² on 80/20 holdout

| Model | R² |
|---|---|
| GradientBoostingRegressor (current, 200 trees, depth=5) | **0.9702** |

The 0.535 previously reported was on an older 144-point set. With 384 points the same hyperparameters reach 0.97, suggesting the model architecture is sound and the bottleneck was data volume.

### Feature importances (trained model)

| Feature | Importance |
|---|---|
| `log_shape_min` | 0.4326 |
| `log_shape_max` | 0.1256 |
| `shape_1` (hidden_dim / n_cols) | 0.0988 |
| `log_shape_prod` | 0.0967 |
| `shape_0` (n_rows) | 0.0704 |
| `config_4` (num_warps, raw int) | 0.0685 |
| `log_num_warps_total` | 0.0528 |
| `operator_id` | 0.0340 |
| `config_6` (BLOCK_SIZE) | 0.0116 |
| `log_tile_area` | 0.0089 |
| All others (stages, hardware_id, config_{0-3,5,7}) | < 0.0002 total |

Key observation: hardware_id importance is essentially 0.0000 — the training set is single-hardware (all A100-SXM4-80GB), so this feature carries no signal now but will matter at multi-hardware scale.

### Per-operator breakdown

**In-distribution R² (80/20 within each operator):**

| Operator | R² |
|---|---|
| rmsnorm | 0.9980 |
| cross_entropy | 0.9988 |
| layernorm | 0.9973 |
| softmax | 0.9866 |

**Leave-one-operator-out R² (train on 3, predict on held-out 1):**

| Held-out operator | Transfer R² |
|---|---|
| rmsnorm | 0.9541 |
| layernorm | 0.8496 |
| softmax | 0.8086 |
| cross_entropy | **−0.2611** |

Cross-entropy transfer is badly negative. Root cause: cross_entropy `n_cols` values (32000–128256) are far outside the training distribution of the other three operators (rmsnorm/layernorm hidden_dim ≤ 5120; softmax n_cols ≤ 32000, with the 32000 case only appearing as one softmax bucket). The model learns that large `shape_1` → high throughput for softmax but cross_entropy at 128256 n_cols has a complex non-monotone response to BLOCK_SIZE (because the block must fit in shared memory and changes tiling factor), which the model cannot extrapolate to.

---

## 2. Proposed Improvements (ranked by expected impact)

### Rank 1 — Per-operator models (expected +0.08–0.15 R² on new operators)

**What:** Train a separate GBR per operator instead of one unified model.

**Why:** In-distribution per-operator R² is 0.997–0.999 vs 0.970 unified. The single model spends capacity on an `operator_id` split that a tree must re-learn at every node. Separate models also eliminate the cross_entropy extrapolation failure entirely (−0.26 → 0.999 for in-distribution).

**How:** Keep a `dict[str, GBR]` keyed by operator name. At `predict` time, dispatch on `operator`. Fall back to the unified model for unknown operators. Training data requirement per operator drops to ~80 samples, easily met after one production run.

**Cost:** Minimal — 4 × 200-tree GBRs fit in <10 MB combined.

---

### Rank 2 — Add `log_tile_coverage` = log(BLOCK_SIZE / n_cols) feature (expected +0.05–0.10 R² on transfer)

**What:** Add a single derived feature `log_tile_coverage = log(BLOCK_SIZE / max(n_cols, hidden_dim, 1))` clamped to [−10, 0].

**Why:** Feature importances show raw BLOCK_SIZE (config_6) ranks 9th (0.0116), while shape_1 (n_cols) ranks 3rd (0.0988). The model must infer their ratio implicitly from two separate features with very different scales. The ratio directly encodes how many passes the kernel takes over the reduction dimension: 1 pass (BLOCK_SIZE ≥ n_cols, fully in SRAM) vs multiple-pass (BLOCK_SIZE < n_cols, memory-bound). This is the primary performance predictor for all memory-bound reduction kernels. For cross_entropy with n_cols=128256 and BLOCK_SIZE≤32768, `log_tile_coverage ≈ −1.36`; for softmax with n_cols=512 and BLOCK_SIZE=1024, `log_tile_coverage = 0` (one pass) — a critical distinction that the model currently cannot represent cleanly. Implementation: one line in `extract_features`, no schema change.

---

### Rank 3 — Replace integer `operator_id` / `hardware_id` with one-hot encoding (expected +0.01–0.03 R² unified; larger benefit when hardware diversity grows)

**What:** Replace the two integer columns (operator_id ∈ {0..5}, hardware_id ∈ {0..7}) with 6 + 8 = 14 binary columns.

**Why:** GBR splits on thresholds. An integer `operator_id` forces the tree to partition {0,1,2,3,4,5} with cuts like `op_id < 2.5`, implying ordinal relationships between operators that don't exist (there is no reason rmsnorm=1 and softmax=2 should be "more similar" than rmsnorm=1 and matmul=0). One-hot encoding directly tested gives 0.9717 vs 0.9702 — marginal now with 4 operators, but cross_entropy transfer R² stays at −0.26, suggesting it doesn't fix the shape extrapolation problem. The real payoff comes when 5+ hardware types are in the training set (e.g., A100 + H100 + T4): hardware_id currently has 0.0000 importance because all data is single-hardware, but incorrect ordinality will hurt badly in multi-hardware training.

---

### Additional proposals (not in top 3 but worth noting)

**Log-transform target:** Using log(tflops) as the regression target improves R² in log-space (0.9813 vs 0.9702) but *slightly hurts* raw-space R² (0.9652) because the exp() amplifies errors on high-tflops points. Only beneficial for ranking tasks where relative ordering matters more than absolute magnitude.

**XGBoost / LightGBM:** Would likely yield +0.01–0.02 R² via better regularization and native categorical support, but per-operator models (Rank 1) already push in-distribution R² above 0.997, leaving little headroom. Propose only if hardware-generalization becomes the bottleneck.

**LambdaRank (learning-to-rank):** Directly optimizes NDCG, beneficial when the use case is selecting the top-3 configs rather than predicting absolute tflops. Worth implementing after per-operator models, since current R² ≥ 0.99 per-operator already yields near-perfect ranking within a shape bucket.

---

## 3. Priority-Ordered Implementation Plan

### Step 1: Per-operator models (1–2 hours)

In `cost_model.py`, change `CostModel` to store `self.regressors: dict[str, GBR]` (one per operator) and fall back to the unified model for unknown operators. Update `train_from_databases` to split `X, y` by operator and fit each model separately. Update `predict` / `predict_many` to dispatch on `operator` string. Add `per_operator_r2` to the training return dict. Expected outcome: in-distribution R² > 0.997 for all four current operators.

### Step 2: Add `log_tile_coverage` feature (30 minutes)

In `extract_features`, after computing `config_vals`, add:

```python
n_cols_dim = shape.get("n_cols", shape.get("hidden_dim", 0))
block_size_for_tile = config_vals[6] if operator not in ("matmul", "attention") else max(config_vals[0], config_vals[1])
log_tile_coverage = math.log(block_size_for_tile / max(n_cols_dim, 1)) if n_cols_dim > 0 else 0.0
log_tile_coverage = max(log_tile_coverage, -10.0)
```

Append to the feature vector and add `"log_tile_coverage"` to `FEATURE_NAMES`. This requires retraining — saved `.pkl` models must be regenerated. Expected outcome: cross_entropy leave-one-out transfer improves from −0.26 toward ≥ 0.5.

### Step 3: One-hot categorical encoding (1 hour, before multi-hardware data arrives)

Add a helper `_encode_categoricals(features: list[float]) -> list[float]` that replaces positions 0 (operator_id) and 1 (hardware_id) with one-hot segments. Keep the integer IDs in `extract_features` for backward compatibility (so the pkl format stays stable) and apply the one-hot transform inside `train_from_databases` and `predict`. Update `FEATURE_NAMES` to list all one-hot column names. Estimated net benefit: ~+0.002 R² now, +0.05–0.10 when H100/T4 data is mixed in.
