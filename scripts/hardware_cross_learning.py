#!/usr/bin/env python3
"""Hardware cross-learning experiment.

Tests whether a cost model trained on source-GPU data (e.g. A100) can make
useful predictions on a different target GPU (e.g. H100).

This is a core generalization test: if the model transfers, the approach
scales to new hardware for free. If not, we need per-hardware models.

Steps:
  1. Train a cost model on existing .noeris/cost-model-training.json filtered
     to source-GPU records only. Save as .noeris/cost-model-a100-only.pkl.
  2. Collect a small H100 test set via Modal (~16 configs × 4 shapes per op).
  3. Compute predictions from the source-trained model for target test points.
  4. Report per-operator cross-hardware R² and Spearman rank correlation.
  5. Report top-K ranking agreement: how often does A100 model pick H100's top-K?

Usage:
    python3.11 scripts/hardware_cross_learning.py \\
        --source-gpu A100 \\
        --target-gpu H100 \\
        --operators rmsnorm softmax layernorm cross_entropy \\
        --configs-per-op 16 \\
        --shapes-per-op 4 \\
        --output docs/results/hardware-cross-learning-a100-to-h100.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_engine.cost_model import (
    CostModel,
    HARDWARE_IDS,
    _hardware_id,
    extract_features,
    _encode_categoricals,
)
from research_engine.modal_session import ModalBenchmarkSession
from research_engine.triton_operators import REGISTRY


# ---------------------------------------------------------------------------
# Helper: parse shape string back to dict
# ---------------------------------------------------------------------------

def _parse_shape(operator: str, shape_str: str) -> dict | None:
    parts = shape_str.split("x")
    try:
        if operator in ("rmsnorm", "layernorm"):
            return {"n_rows": int(parts[0]), "hidden_dim": int(parts[1])}
        if operator in ("softmax", "cross_entropy"):
            return {"n_rows": int(parts[0]), "n_cols": int(parts[1])}
        if operator == "matmul":
            return {"M": int(parts[0]), "N": int(parts[1]), "K": int(parts[2])}
        if operator == "attention":
            return {
                "batch": int(parts[0]),
                "heads": int(parts[1]),
                "seq_len": int(parts[2]),
                "head_dim": int(parts[3]),
            }
    except (ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Step 1: Train source-GPU-only cost model
# ---------------------------------------------------------------------------

def _infer_hardware_family(hw_name: str, gpu_arg: str) -> bool:
    """Return True if hw_name belongs to the source GPU family."""
    hw_lower = hw_name.lower()
    gpu_lower = gpu_arg.lower()

    # Map CLI arg to canonical substring
    family_map = {
        "a100": "a100",
        "h100": "h100",
        "t4": "t4",
        "a10g": "a10g",
    }
    family = family_map.get(gpu_lower, gpu_lower)
    return family in hw_lower


def train_source_model(
    db_path: Path,
    source_gpu: str,
    model_save_path: Path,
) -> dict:
    """Filter training DB to source-GPU records and train a CostModel."""
    import json as _json
    import pickle

    data = _json.loads(db_path.read_text())
    records = data.get("records", {})

    # Build a filtered DB dict with only source-GPU records
    filtered_records = {}
    source_count = 0
    skipped_count = 0

    for key, record in records.items():
        parts = key.split(":", 2)
        if len(parts) == 3:
            operator, bucket, hardware = parts
        elif len(parts) == 2:
            operator, hardware = "matmul", parts[1]
        else:
            continue

        if _infer_hardware_family(hardware, source_gpu):
            filtered_records[key] = record
            source_count += len(record.get("results", []))
        else:
            skipped_count += len(record.get("results", []))

    print(f"  Source-GPU records: {source_count} points ({skipped_count} skipped from other GPUs)")

    # Write temp file for CostModel to consume
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp.write(_json.dumps({"records": filtered_records}))
        tmp_path = Path(tmp.name)

    try:
        model = CostModel()
        result = model.train_from_databases([tmp_path])
    finally:
        tmp_path.unlink(missing_ok=True)

    model.save(model_save_path)
    print(f"  Model saved to {model_save_path}")
    print(f"  Training result: {result}")
    return result


# ---------------------------------------------------------------------------
# Step 2: Collect target-GPU test points via Modal
# ---------------------------------------------------------------------------

def collect_target_test_set(
    *,
    operators: list[str],
    target_gpu: str,
    configs_per_op: int,
    shapes_per_op: int,
    session: ModalBenchmarkSession,
) -> dict[str, list[dict]]:
    """Run benchmark grid on target GPU and return raw results per operator.

    Returns:
        {operator: [{"shape": dict, "config": dict, "hardware": str,
                     "metric": float, "metric_name": str}]}
    """
    results: dict[str, list[dict]] = {}

    for op_name in operators:
        try:
            spec = REGISTRY.get(op_name)
        except KeyError:
            print(f"  [{op_name}] unknown operator, skipping")
            continue

        shapes = spec.shape_buckets[:shapes_per_op]
        grid = spec.grid_generator_fn(
            include_curated=True,
            max_configs=configs_per_op,
        )[:configs_per_op]

        print(f"  [{op_name}] running {len(grid)} configs × {len(shapes)} shapes on {target_gpu}...")

        script = spec.benchmark_script_fn(grid, shapes)
        batch = session.run_batch(script)

        if not batch.success:
            print(f"  [{op_name}] FAILED: {batch.error[:200]}")
            results[op_name] = []
            continue

        hw_name = batch.hardware.get("gpu", target_gpu)
        op_results: list[dict] = []

        for config_result in batch.config_results:
            config = config_result.get("config", {})
            for sr in config_result.get("results", []):
                if not sr.get("correct"):
                    continue
                metric_val = sr.get("tflops") or sr.get("gb_per_s") or 0.0
                if metric_val <= 0:
                    continue
                parsed_shape = _parse_shape(op_name, sr.get("shape", ""))
                if parsed_shape is None:
                    continue
                metric_name = "tflops" if "tflops" in sr and sr["tflops"] else "gb_per_s"
                op_results.append({
                    "shape": parsed_shape,
                    "config": config,
                    "hardware": hw_name,
                    "metric": float(metric_val),
                    "metric_name": metric_name,
                    "shape_str": sr.get("shape", ""),
                })

        print(f"  [{op_name}] collected {len(op_results)} points (hardware={hw_name})")
        results[op_name] = op_results

    return results


# ---------------------------------------------------------------------------
# Step 3 & 4: Evaluate cross-hardware predictions
# ---------------------------------------------------------------------------

def _spearman_r(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return float("nan")

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        for rank_idx, (orig_idx, _) in enumerate(indexed):
            ranks[orig_idx] = float(rank_idx + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """Compute R² (coefficient of determination)."""
    n = len(y_true)
    if n < 2:
        return float("nan")
    mean_true = sum(y_true) / n
    ss_tot = sum((v - mean_true) ** 2 for v in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def _top_k_agreement(
    configs: list[dict],
    y_measured: list[float],
    y_predicted: list[float],
    k: int,
) -> float:
    """Fraction of actual top-K that appear in predicted top-K.

    Returns a value in [0, 1]. Random baseline = k/n.
    """
    n = len(configs)
    if n == 0 or k >= n:
        return float("nan")

    actual_top_k = set(
        i for i, _ in sorted(enumerate(y_measured), key=lambda t: -t[1])[:k]
    )
    pred_top_k = set(
        i for i, _ in sorted(enumerate(y_predicted), key=lambda t: -t[1])[:k]
    )
    intersection = len(actual_top_k & pred_top_k)
    return intersection / k


def evaluate_cross_hardware(
    *,
    operator: str,
    target_gpu_data: list[dict],
    source_model: CostModel,
    target_gpu: str,
    source_gpu: str,
    top_k: int = 5,
) -> dict:
    """Evaluate source-model predictions against target-GPU measurements.

    Groups by shape bucket, computes per-bucket metrics and aggregates.
    Returns a rich eval dict.
    """
    if not target_gpu_data:
        return {
            "operator": operator,
            "n_points": 0,
            "error": "no_data",
        }

    # Predict using source-GPU hardware label (to test pure cross-hw transfer)
    # AND using target-GPU hardware label (to test if 1-hot helps)
    y_measured = [pt["metric"] for pt in target_gpu_data]

    # --- Prediction with source hardware encoding (pure transfer) ---
    y_pred_source_hw = []
    y_pred_target_hw = []
    for pt in target_gpu_data:
        # Using source hardware label — what A100 model "thinks" about these configs
        pred_src = source_model.predict(
            shape=pt["shape"],
            config=pt["config"],
            hardware=pt["hardware"].replace(
                pt["hardware"], _hardware_canonical(source_gpu)
            ),
            operator=operator,
        )
        y_pred_source_hw.append(pred_src)

        # Using actual target hardware label (H100 one-hot will be "unknown"
        # if model never saw H100, so hw_id=0 via HARDWARE_IDS.get fallback)
        pred_tgt = source_model.predict(
            shape=pt["shape"],
            config=pt["config"],
            hardware=pt["hardware"],
            operator=operator,
        )
        y_pred_target_hw.append(pred_tgt)

    r2_src = _r_squared(y_measured, y_pred_source_hw)
    rho_src = _spearman_r(y_measured, y_pred_source_hw)
    topk_src = _top_k_agreement(
        [pt["config"] for pt in target_gpu_data],
        y_measured,
        y_pred_source_hw,
        k=min(top_k, len(y_measured) // 2),
    )

    r2_tgt = _r_squared(y_measured, y_pred_target_hw)
    rho_tgt = _spearman_r(y_measured, y_pred_target_hw)
    topk_tgt = _top_k_agreement(
        [pt["config"] for pt in target_gpu_data],
        y_measured,
        y_pred_target_hw,
        k=min(top_k, len(y_measured) // 2),
    )

    n = len(y_measured)
    random_baseline = min(top_k, n // 2) / n if n > 0 else float("nan")

    # Per-shape bucket breakdown
    buckets: dict[str, list] = {}
    for pt in target_gpu_data:
        bk = pt.get("shape_str", str(pt["shape"]))
        buckets.setdefault(bk, []).append(pt)

    per_bucket = []
    for bk, pts in buckets.items():
        bm = [p["metric"] for p in pts]
        bp_src = [
            source_model.predict(
                shape=p["shape"], config=p["config"],
                hardware=_hardware_canonical(source_gpu), operator=operator
            )
            for p in pts
        ]
        per_bucket.append({
            "shape_str": bk,
            "n": len(pts),
            "mean_measured": round(sum(bm) / len(bm), 3) if bm else 0,
            "r2": round(_r_squared(bm, bp_src), 4) if len(bm) >= 2 else None,
            "spearman": round(_spearman_r(bm, bp_src), 4) if len(bm) >= 2 else None,
        })

    return {
        "operator": operator,
        "n_points": n,
        "source_gpu": source_gpu,
        "target_gpu": target_gpu,
        # Using source hw encoding (A100 label used for prediction)
        "r2_source_hw_encoding": round(r2_src, 4),
        "spearman_source_hw_encoding": round(rho_src, 4),
        "topk_agreement_source_hw_encoding": round(topk_src, 4) if not math.isnan(topk_src) else None,
        # Using target hw encoding (H100 label, maps to hw_id=0 if unseen)
        "r2_target_hw_encoding": round(r2_tgt, 4),
        "spearman_target_hw_encoding": round(rho_tgt, 4),
        "topk_agreement_target_hw_encoding": round(topk_tgt, 4) if not math.isnan(topk_tgt) else None,
        "random_topk_baseline": round(random_baseline, 4) if not math.isnan(random_baseline) else None,
        "top_k": min(top_k, n // 2),
        "per_bucket": per_bucket,
        "sample_predictions": [
            {
                "shape": pt["shape"],
                "config": pt["config"],
                "measured": round(pt["metric"], 3),
                "predicted_source_hw": round(y_pred_source_hw[i], 3),
                "predicted_target_hw": round(y_pred_target_hw[i], 3),
            }
            for i, pt in enumerate(target_gpu_data[:10])
        ],
    }


def _hardware_canonical(gpu_arg: str) -> str:
    """Return a canonical hardware string matching HARDWARE_IDS for the CLI arg."""
    mapping = {
        "a100": "NVIDIA A100-SXM4-80GB",
        "h100": "NVIDIA H100 80GB HBM3",
        "t4": "NVIDIA T4",
        "a10g": "NVIDIA A10G",
    }
    return mapping.get(gpu_arg.lower(), gpu_arg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardware cross-learning: train on source GPU, evaluate on target GPU."
    )
    parser.add_argument("--source-gpu", default="A100", help="Source GPU (training data)")
    parser.add_argument("--target-gpu", default="H100", help="Target GPU (evaluation)")
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["rmsnorm", "softmax", "layernorm", "cross_entropy"],
    )
    parser.add_argument(
        "--configs-per-op",
        type=int,
        default=16,
        help="Number of configs to benchmark per operator on target GPU",
    )
    parser.add_argument(
        "--shapes-per-op",
        type=int,
        default=4,
        help="Number of shape buckets per operator",
    )
    parser.add_argument(
        "--training-db",
        default=".noeris/cost-model-training.json",
        help="Path to the existing training database",
    )
    parser.add_argument(
        "--model-save-path",
        default=".noeris/cost-model-a100-only.pkl",
        help="Where to save the source-only trained model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="K for top-K ranking agreement metric",
    )
    parser.add_argument(
        "--output",
        default="docs/results/hardware-cross-learning-a100-to-h100.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / args.training_db
    model_save_path = root / args.model_save_path
    output_path = root / args.output

    if not db_path.exists():
        print(f"Training DB not found: {db_path}")
        return 1

    print("=" * 60)
    print(f"Hardware Cross-Learning: {args.source_gpu} -> {args.target_gpu}")
    print(f"Operators: {args.operators}")
    print(f"Configs/op: {args.configs_per_op}, Shapes/op: {args.shapes_per_op}")
    print("=" * 60)
    print()

    # Step 1: Train source-only model
    print("Step 1: Training source-GPU-only cost model...")
    train_info = train_source_model(db_path, args.source_gpu, model_save_path)
    source_model = CostModel.load(model_save_path)
    print(f"  Model loaded: {source_model.training_size} training points")
    print()

    # Step 2: Collect target-GPU test set
    print(f"Step 2: Collecting {args.target_gpu} test data via Modal...")
    print()

    all_target_data: dict[str, list[dict]] = {}
    with ModalBenchmarkSession(gpu=args.target_gpu, timeout_seconds=900) as session:
        all_target_data = collect_target_test_set(
            operators=args.operators,
            target_gpu=args.target_gpu,
            configs_per_op=args.configs_per_op,
            shapes_per_op=args.shapes_per_op,
            session=session,
        )

    total_target_points = sum(len(v) for v in all_target_data.values())
    print()
    print(f"Total target-GPU points collected: {total_target_points}")
    print()

    # Step 3 & 4: Evaluate predictions
    print("Step 3/4: Evaluating cross-hardware predictions...")
    print()

    per_operator_results = []
    for op_name in args.operators:
        op_data = all_target_data.get(op_name, [])
        eval_result = evaluate_cross_hardware(
            operator=op_name,
            target_gpu_data=op_data,
            source_model=source_model,
            target_gpu=args.target_gpu,
            source_gpu=args.source_gpu,
            top_k=args.top_k,
        )
        per_operator_results.append(eval_result)

        n = eval_result.get("n_points", 0)
        r2 = eval_result.get("r2_source_hw_encoding", float("nan"))
        rho = eval_result.get("spearman_source_hw_encoding", float("nan"))
        topk = eval_result.get("topk_agreement_source_hw_encoding")
        rand = eval_result.get("random_topk_baseline")

        print(f"  [{op_name}] n={n}, R²={r2:.3f}, Spearman rho={rho:.3f}", end="")
        if topk is not None and rand is not None:
            print(f", top-{eval_result['top_k']} agreement={topk:.3f} (random={rand:.3f})", end="")
        print()

    # Build summary stats across all operators
    valid_r2 = [
        r["r2_source_hw_encoding"]
        for r in per_operator_results
        if "r2_source_hw_encoding" in r and not math.isnan(r["r2_source_hw_encoding"])
    ]
    valid_rho = [
        r["spearman_source_hw_encoding"]
        for r in per_operator_results
        if "spearman_source_hw_encoding" in r and not math.isnan(r["spearman_source_hw_encoding"])
    ]
    valid_topk = [
        r["topk_agreement_source_hw_encoding"]
        for r in per_operator_results
        if r.get("topk_agreement_source_hw_encoding") is not None
    ]
    rand_topk_baselines = [
        r["random_topk_baseline"]
        for r in per_operator_results
        if r.get("random_topk_baseline") is not None
    ]

    mean_r2 = sum(valid_r2) / len(valid_r2) if valid_r2 else float("nan")
    mean_rho = sum(valid_rho) / len(valid_rho) if valid_rho else float("nan")
    mean_topk = sum(valid_topk) / len(valid_topk) if valid_topk else float("nan")
    mean_rand = sum(rand_topk_baselines) / len(rand_topk_baselines) if rand_topk_baselines else float("nan")

    print()
    print("=" * 60)
    print(f"SUMMARY: {args.source_gpu} -> {args.target_gpu} cross-hardware transfer")
    print(f"  Mean R²:            {mean_r2:.3f}")
    print(f"  Mean Spearman rho:  {mean_rho:.3f}")
    if not math.isnan(mean_topk):
        print(f"  Mean top-K agree:   {mean_topk:.3f} (random baseline: {mean_rand:.3f})")
    print("=" * 60)

    # Qualitative verdict
    if mean_r2 >= 0.7:
        verdict = "strong_transfer"
        verdict_text = "Strong transfer: A100-trained model predicts H100 performance well."
    elif mean_r2 >= 0.4:
        verdict = "partial_transfer"
        verdict_text = "Partial transfer: ranking is preserved but absolute values drift."
    elif mean_r2 >= 0.0:
        verdict = "weak_transfer"
        verdict_text = "Weak transfer: slight positive correlation but not reliably useful."
    else:
        verdict = "no_transfer"
        verdict_text = "No transfer: predictions anticorrelated or worse than mean baseline."

    topk_verdict = "above_random"
    if not math.isnan(mean_topk) and not math.isnan(mean_rand):
        topk_verdict = "above_random" if mean_topk > mean_rand * 1.1 else "at_or_below_random"

    print()
    print(f"Verdict: {verdict_text}")
    print(f"Top-K verdict: {topk_verdict}")

    report = {
        "experiment": "hardware_cross_learning",
        "source_gpu": args.source_gpu,
        "target_gpu": args.target_gpu,
        "operators": args.operators,
        "configs_per_op": args.configs_per_op,
        "shapes_per_op": args.shapes_per_op,
        "top_k": args.top_k,
        "source_model": {
            "path": str(model_save_path),
            "training_size": source_model.training_size,
            "training_info": train_info,
        },
        "total_target_points": total_target_points,
        "summary": {
            "mean_r2_source_hw": round(mean_r2, 4) if not math.isnan(mean_r2) else None,
            "mean_spearman_source_hw": round(mean_rho, 4) if not math.isnan(mean_rho) else None,
            "mean_topk_agreement": round(mean_topk, 4) if not math.isnan(mean_topk) else None,
            "random_topk_baseline": round(mean_rand, 4) if not math.isnan(mean_rand) else None,
            "verdict": verdict,
            "verdict_text": verdict_text,
            "topk_verdict": topk_verdict,
        },
        "per_operator": per_operator_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print()
    print(f"Report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
