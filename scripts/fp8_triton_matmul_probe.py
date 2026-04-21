#!/usr/bin/env python3
"""Probe Triton FP8 matmul kernel viability on Hopper."""

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


SCRIPT = r'''
import json
import platform
import torch
import triton
import triton.language as tl

@triton.jit
def fp8_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def bench_ms(fn, warmup=25, trials=100):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        vals.append(st.elapsed_time(en))
    vals.sort()
    return float(vals[len(vals)//2])


def run_probe(M=1024, N=1024, K=1024):
    if not hasattr(torch, "float8_e4m3fn"):
        return {"ok": False, "reason": "torch.float8_e4m3fn unavailable"}

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def launch():
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        fp8_mm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )

    launch()
    ref = (a.float() @ b.float()).half()
    max_err = (c - ref).abs().max().item()
    ms = bench_ms(launch)
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    return {
        "ok": True,
        "shape": [M, N, K],
        "ms": round(ms, 4),
        "tflops": round(float(tflops), 4),
        "max_err": round(float(max_err), 6),
    }


probe = run_probe()
print(json.dumps({
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda or "unknown",
        "python": platform.python_version(),
    },
    "config_results": [],
    "probe": probe,
}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--output-json", default="docs/results/fp8-triton-matmul-probe-h100.json")
    parser.add_argument("--output-md", default="docs/results/fp8-triton-matmul-probe-h100.md")
    args = parser.parse_args()

    with ModalBenchmarkSession(
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        local_source_dir=str((ROOT / "src/research_engine").resolve()),
    ) as session:
        result = session.run_script(SCRIPT)

    if not result.success:
        raise RuntimeError(f"FP8 Triton probe failed: {result.error}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": result.hardware,
        "probe": result.extra.get("probe", {}),
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# FP8 Triton Matmul Probe (H100)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Probe success: `{payload['probe'].get('ok', False)}`",
        f"Shape: `{payload['probe'].get('shape', [])}`",
        f"Latency: `{payload['probe'].get('ms', 'n/a')} ms`",
        f"Throughput: `{payload['probe'].get('tflops', 'n/a')} TFLOPS`",
        f"Max abs error vs fp32-acc ref: `{payload['probe'].get('max_err', 'n/a')}`",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
