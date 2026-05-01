#!/usr/bin/env python3
"""Fail CI when exported history contains blocking regressions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ALIGNMENT_METRICS = [
    "reuse_1_kn_rate",
    "reuse_2_4_nk_rate",
    "reuse_5_plus_nk_rate",
]


def _load_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("history-regressions payload must be a JSON object")
    return data


def _blocking_regressions(payload: dict, benchmark_id: str) -> list[str]:
    if str(payload.get("benchmark_id", "")) != benchmark_id:
        return []
    rows = payload.get("fp8_policy_regressions", [])
    if not isinstance(rows, list):
        return []
    return [str(item) for item in rows if str(item).strip()]


def _load_threshold_overrides() -> dict[str, dict[str, float]]:
    raw = os.getenv("NOERIS_FP8_DROP_THRESHOLDS_JSON", "").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for lane, lane_values in payload.items():
        if not isinstance(lane, str) or not isinstance(lane_values, dict):
            continue
        lane_out: dict[str, float] = {}
        for metric, value in lane_values.items():
            if not isinstance(metric, str) or not isinstance(value, (int, float)):
                continue
            lane_out[metric] = float(value)
        out[lane] = lane_out
    return out


def _effective_thresholds(*, benchmark_id: str, default_drop_threshold: float) -> dict[str, float]:
    thresholds = {metric: float(default_drop_threshold) for metric in ALIGNMENT_METRICS}
    overrides = _load_threshold_overrides()
    for metric, value in overrides.get("default", {}).items():
        if metric in thresholds:
            thresholds[metric] = float(value)
    for metric, value in overrides.get(benchmark_id, {}).items():
        if metric in thresholds:
            thresholds[metric] = float(value)
    return thresholds


def _load_summary_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("history-summary payload must be a JSON object")
    return data


def _metric_value(mapping: dict, key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _blocking_threshold_regressions(
    summary_payload: dict | None,
    *,
    benchmark_id: str,
    thresholds: dict[str, float],
) -> list[str]:
    if not isinstance(summary_payload, dict):
        return []
    if str(summary_payload.get("benchmark_id", "")) != benchmark_id:
        return []
    latest = summary_payload.get("fp8_latest_alignment", {})
    previous = summary_payload.get("fp8_previous_alignment", {})
    if not isinstance(latest, dict) or not isinstance(previous, dict):
        return []

    rows: list[str] = []
    for metric in ALIGNMENT_METRICS:
        threshold = float(thresholds.get(metric, 0.2))
        latest_value = _metric_value(latest, metric)
        previous_value = _metric_value(previous, metric)
        if latest_value is None or previous_value is None:
            continue
        drop = previous_value - latest_value
        if drop >= threshold:
            rows.append(
                f"{metric} dropped by {drop:.4f} (prev={previous_value:.4f}, latest={latest_value:.4f}, threshold={threshold:.4f})"
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=".noeris/history/history-regressions.json",
        help="path to exported history-regressions.json",
    )
    parser.add_argument(
        "--benchmark-id",
        default="matmul-speedup",
        help="benchmark id for gating",
    )
    parser.add_argument(
        "--summary-path",
        default=".noeris/history/history-summary.json",
        help="path to exported history-summary.json",
    )
    parser.add_argument(
        "--default-drop-threshold",
        type=float,
        default=0.20,
        help="default max allowed drop for FP8 policy-alignment rates",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="fail when the regressions file is missing",
    )
    args = parser.parse_args()

    path = Path(args.path)
    payload = _load_payload(path)
    summary_payload = _load_summary_payload(Path(args.summary_path))
    thresholds = _effective_thresholds(
        benchmark_id=args.benchmark_id,
        default_drop_threshold=args.default_drop_threshold,
    )
    if payload is None:
        print(
            json.dumps(
                {
                    "status": "missing",
                    "path": str(path),
                    "benchmark_id": args.benchmark_id,
                    "blocking_regressions": [],
                    "blocking_threshold_regressions": [],
                    "thresholds": thresholds,
                },
                indent=2,
            )
        )
        return 1 if args.fail_on_missing else 0

    blocking = _blocking_regressions(payload, benchmark_id=args.benchmark_id)
    blocking_threshold = _blocking_threshold_regressions(
        summary_payload,
        benchmark_id=args.benchmark_id,
        thresholds=thresholds,
    )
    all_blocking = blocking + blocking_threshold
    print(
        json.dumps(
            {
                "status": "ok" if not all_blocking else "regressions_found",
                "path": str(path),
                "summary_path": str(args.summary_path),
                "benchmark_id": args.benchmark_id,
                "blocking_regressions": blocking,
                "blocking_threshold_regressions": blocking_threshold,
                "thresholds": thresholds,
            },
            indent=2,
        )
    )
    return 0 if not all_blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
