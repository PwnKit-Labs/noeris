#!/usr/bin/env python3
"""Probe FP8 capability and basic matmul behavior on Hopper via Modal."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_engine.modal_session import ModalBenchmarkSession


PROBE_SCRIPT = r'''
import json
import platform
import torch

probe = {
    "hardware": {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cuda_version": torch.version.cuda or "unknown",
        "python": platform.python_version(),
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    },
    "fp8": {},
}

fp8_dtypes = [
    ("float8_e4m3fn", "float8_e4m3fn"),
    ("float8_e5m2", "float8_e5m2"),
]

for label, attr in fp8_dtypes:
    entry = {"dtype_available": hasattr(torch, attr)}
    if not entry["dtype_available"]:
        probe["fp8"][label] = entry
        continue

    dtype = getattr(torch, attr)
    try:
        a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16).to(dtype)
        b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16).to(dtype)
        c = torch.matmul(a, b)
        entry["matmul_supported"] = True
        entry["result_dtype"] = str(c.dtype)
    except Exception as exc:  # noqa: BLE001
        entry["matmul_supported"] = False
        entry["matmul_error"] = str(exc)[:300]

    probe["fp8"][label] = entry

print(json.dumps({
    "hardware": probe["hardware"],
    "config_results": [],
    "probe": probe,
}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-cost-usd", type=float, default=1.5)
    parser.add_argument("--output", default="docs/results/fp8-hopper-probe.json")
    args = parser.parse_args()

    with ModalBenchmarkSession(
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        local_source_dir=str(Path("src/research_engine").resolve()),
    ) as session:
        result = session.run_script(PROBE_SCRIPT)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": result.success,
        "error": result.error,
        "hardware": result.hardware,
        "config_results": result.config_results,
        "extra": result.extra,
    }

    if result.success:
        payload["probe"] = result.extra.get("probe", {})
    else:
        payload["probe"] = {
            "stderr_tail": result.stderr[-2000:],
            "stdout_tail": result.stdout[-2000:],
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
