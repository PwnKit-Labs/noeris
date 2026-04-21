"""Cross-vendor zero-shot transfer helpers.

Scaffold utilities for predicting target-hardware config rankings without
target-device benchmark data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BucketEntry:
    shape: dict[str, Any]
    bucket: str
    configs: list[dict[str, Any]]


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
                    "config": result.get("config", {}),
                    "metric": metric,
                }
            )

        if not rows:
            continue

        rows.sort(key=lambda r: r["metric"], reverse=True)
        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            cid = row["config_id"]
            if cid and cid not in dedup:
                dedup[cid] = row
        top = list(dedup.values())[:max_candidates_per_bucket]
        out[bucket] = BucketEntry(shape=shape, bucket=bucket, configs=top)

    return out
