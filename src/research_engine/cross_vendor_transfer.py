"""Cross-vendor zero-shot transfer helpers.

Scaffold utilities for predicting target-hardware config rankings without
target-device benchmark data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass
class BucketEntry:
    shape: dict[str, Any]
    bucket: str
    configs: list[dict[str, Any]]


def canonicalize_config_for_operator(*, operator: str, config: dict[str, Any]) -> dict[str, Any]:
    """Map equivalent config field names to model-expected keys."""
    out = dict(config)
    if operator == "matmul":
        if "BLOCK_SIZE_M" not in out and "BLOCK_M" in out:
            out["BLOCK_SIZE_M"] = out["BLOCK_M"]
        if "BLOCK_SIZE_N" not in out and "BLOCK_N" in out:
            out["BLOCK_SIZE_N"] = out["BLOCK_N"]
        if "BLOCK_SIZE_K" not in out and "BLOCK_K" in out:
            out["BLOCK_SIZE_K"] = out["BLOCK_K"]
    return out


def parse_record_key(key: str) -> tuple[str, str, str] | None:
    parts = key.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        # Legacy format bucket:hardware -> assume matmul
        return "matmul", parts[0], parts[1]
    return None


def collect_source_bucket_candidates(
    *,
    records: dict[str, Any],
    operator: str,
    source_hardware_substr: str,
    max_candidates_per_bucket: int = 40,
) -> dict[str, BucketEntry]:
    """Collect observed configs from source hardware grouped by bucket."""
    source_norm = source_hardware_substr.lower()
    out: dict[str, BucketEntry] = {}
    rows_by_bucket: dict[str, list[dict[str, Any]]] = {}

    for key, record in records.items():
        parsed = parse_record_key(key)
        if parsed is None:
            continue
        op, bucket, hardware = parsed
        if op != operator:
            continue
        if source_norm not in hardware.lower():
            continue

        shape = record.get("shape", {})
        rows = []
        for result in record.get("results", []):
            if not result.get("correct"):
                continue
            metric = float(result.get("tflops") or result.get("gb_per_s") or 0.0)
            if metric <= 0:
                continue
            rows.append(
                {
                    "config_id": result.get("config_id", ""),
                    "config": canonicalize_config_for_operator(
                        operator=operator,
                        config=result.get("config", {}),
                    ),
                    "metric": metric,
                }
            )

        if not rows:
            continue

        if bucket not in out:
            out[bucket] = BucketEntry(shape=shape, bucket=bucket, configs=[])
            rows_by_bucket[bucket] = []
        rows_by_bucket[bucket].extend(rows)

    for bucket, bucket_rows in rows_by_bucket.items():
        bucket_rows.sort(key=lambda r: r["metric"], reverse=True)
        dedup: dict[str, dict[str, Any]] = {}
        for row in bucket_rows:
            cid = row["config_id"]
            if cid:
                if cid not in dedup:
                    dedup[cid] = row
                continue
            anon_key = repr(sorted(row.get("config", {}).items()))
            if anon_key not in dedup:
                dedup[anon_key] = row
        out[bucket].configs = list(dedup.values())[:max_candidates_per_bucket]

    return out


def _rank_desc(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def spearman_rank_correlation(predicted: list[float], measured: list[float]) -> float:
    if len(predicted) != len(measured) or len(predicted) < 2:
        return 0.0
    pr = _rank_desc(predicted)
    mr = _rank_desc(measured)
    n = len(pr)
    mean_p = sum(pr) / n
    mean_m = sum(mr) / n
    cov = sum((pr[i] - mean_p) * (mr[i] - mean_m) for i in range(n))
    var_p = sum((v - mean_p) ** 2 for v in pr)
    var_m = sum((v - mean_m) ** 2 for v in mr)
    denom = sqrt(var_p * var_m)
    if denom <= 0.0:
        return 0.0
    return cov / denom


def top_k_hit_rate(predicted_ids: list[str], measured_ids: list[str], *, k: int) -> float:
    if k <= 0:
        return 0.0
    p = set(predicted_ids[:k])
    m = set(measured_ids[:k])
    if not p or not m:
        return 0.0
    return len(p & m) / float(k)


def latency_regret(predicted_best_ms: float, measured_best_ms: float) -> float:
    if measured_best_ms <= 0.0:
        return 0.0
    return max((predicted_best_ms - measured_best_ms) / measured_best_ms, 0.0)
