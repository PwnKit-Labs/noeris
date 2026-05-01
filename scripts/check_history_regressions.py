#!/usr/bin/env python3
"""Fail CI when exported history contains blocking regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        "--fail-on-missing",
        action="store_true",
        help="fail when the regressions file is missing",
    )
    args = parser.parse_args()

    path = Path(args.path)
    payload = _load_payload(path)
    if payload is None:
        print(
            json.dumps(
                {
                    "status": "missing",
                    "path": str(path),
                    "benchmark_id": args.benchmark_id,
                    "blocking_regressions": [],
                },
                indent=2,
            )
        )
        return 1 if args.fail_on_missing else 0

    blocking = _blocking_regressions(payload, benchmark_id=args.benchmark_id)
    print(
        json.dumps(
            {
                "status": "ok" if not blocking else "regressions_found",
                "path": str(path),
                "benchmark_id": args.benchmark_id,
                "blocking_regressions": blocking,
            },
            indent=2,
        )
    )
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
