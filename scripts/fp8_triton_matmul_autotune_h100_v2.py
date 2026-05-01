#!/usr/bin/env python3
"""Run an expanded FP8 Triton matmul autotune sweep on H100 (v2 lane)."""

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

CONFIGS = [
    {"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32,  "num_warps": 4, "num_stages": 2, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64,  "num_warps": 8, "num_stages": 3, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 3, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
    {"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
    {"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8},
]

SHAPES = [
    {"name": "fp8_mm_1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "fp8_mm_2048x1024x2048", "M": 2048, "N": 1024, "K": 2048},
    {"name": "fp8_mm_4096x4096x4096", "M": 4096, "N": 4096, "K": 4096},
]


@triton.jit
def fp8_mm_kernel_grouped(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k0 * BLOCK_K + offs_k
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offs[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def bench_ms(fn, warmup=30, trials=120):
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


def run_one(shape, cfg):
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    torch.manual_seed(0)
    if not hasattr(torch, "float8_e4m3fn"):
        return {
            "shape_name": shape["name"],
            "correct": False,
            "error": "torch.float8_e4m3fn unavailable",
        }

    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def launch():
        grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
        fp8_mm_kernel_grouped[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg["GROUP_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    try:
        launch()
        ref = (a.float() @ b.float()).half()
        max_err = (c - ref).abs().max().item()
        ms = bench_ms(launch)
        tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
        return {
            "shape_name": shape["name"],
            "shape": f"{M}x{N}x{K}",
            "correct": bool(max_err <= 0.5),
            "max_err": round(float(max_err), 6),
            "ms": round(float(ms), 4),
            "tflops": round(float(tflops), 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "shape_name": shape["name"],
            "shape": f"{M}x{N}x{K}",
            "correct": False,
            "error": str(exc)[:300],
        }


def run_fp16_baseline(shape):
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    torch.manual_seed(0)
    a16 = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b16 = torch.randn((K, N), device="cuda", dtype=torch.float16)

    def launch_fp16():
        _ = a16 @ b16

    try:
        ms = bench_ms(launch_fp16)
        tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
        return {
            "shape_name": shape["name"],
            "shape": f"{M}x{N}x{K}",
            "correct": True,
            "ms": round(float(ms), 4),
            "tflops": round(float(tflops), 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "shape_name": shape["name"],
            "shape": f"{M}x{N}x{K}",
            "correct": False,
            "error": str(exc)[:300],
        }


def main():
    out = []
    fp16_baseline = [run_fp16_baseline(shape) for shape in SHAPES]
    for cfg in CONFIGS:
        cid = (
            f"bm{cfg['BLOCK_M']}_bn{cfg['BLOCK_N']}_bk{cfg['BLOCK_K']}_"
            f"w{cfg['num_warps']}_s{cfg['num_stages']}_g{cfg['GROUP_M']}"
        )
        rows = [run_one(shape, cfg) for shape in SHAPES]
        out.append({"config_id": cid, "config": cfg, "results": rows})

    print(json.dumps({
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        },
        "config_results": out,
        "fp16_baseline": fp16_baseline,
    }, indent=2))


if __name__ == "__main__":
    main()
'''


def _summarize(config_results: list[dict], fp16_baseline: list[dict]) -> dict:
    fp16_by_shape = {
        row.get("shape_name", ""): row
        for row in fp16_baseline
        if row.get("shape_name") and row.get("correct")
    }
    best: dict[str, dict] = {}
    for cfg in config_results:
        cid = cfg.get("config_id", "")
        for row in cfg.get("results", []):
            if not row.get("correct"):
                continue
            name = row.get("shape_name", "")
            prev = best.get(name)
            if prev is None or float(row.get("tflops", 0.0)) > float(prev.get("tflops", 0.0)):
                best[name] = {
                    "config_id": cid,
                    "tflops": float(row.get("tflops", 0.0)),
                    "ms": float(row.get("ms", 0.0)),
                    "max_err": float(row.get("max_err", 0.0)),
                }

    for shape_name, fp8 in best.items():
        fp16 = fp16_by_shape.get(shape_name)
        if fp16:
            fp16_tflops = float(fp16.get("tflops", 0.0))
            fp8["fp16_tflops"] = fp16_tflops
            fp8["fp8_vs_fp16_speedup"] = (
                float(fp8["tflops"]) / fp16_tflops if fp16_tflops > 0.0 else 0.0
            )
    return best


def _to_md(payload: dict) -> str:
    lines = [
        "# FP8 Triton Matmul Autotune (H100, v2)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Shape | best FP8 TFLOPS | FP16 TFLOPS | FP8/FP16 | FP8 ms | max err | config |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for shape in sorted(payload["best_by_shape"].keys()):
        row = payload["best_by_shape"][shape]
        fp16_tflops = float(row.get("fp16_tflops", 0.0))
        speedup = float(row.get("fp8_vs_fp16_speedup", 0.0))
        lines.append(
            f"| {shape} | {row['tflops']:.4f} | {fp16_tflops:.4f} | {speedup:.3f}x | {row['ms']:.4f} | {row['max_err']:.6f} | {row['config_id']} |"
        )
    lines += [
        "",
        "v2 adds grouped launch ordering and an expanded BLOCK_K/BLOCK_N search lane.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-cost-usd", type=float, default=4.0)
    parser.add_argument("--output-json", default="docs/results/fp8-triton-matmul-autotune-h100-v2.json")
    parser.add_argument("--output-md", default="docs/results/fp8-triton-matmul-autotune-h100-v2.md")
    args = parser.parse_args()

    with ModalBenchmarkSession(
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        local_source_dir=str((ROOT / "src/research_engine").resolve()),
    ) as session:
        out = session.run_script(SCRIPT)
    if not out.success:
        raise RuntimeError(f"FP8 autotune sweep v2 failed: {out.error}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": out.hardware,
        "config_results": out.config_results,
        "fp16_baseline": out.extra.get("fp16_baseline", []),
        "best_by_shape": _summarize(out.config_results, out.extra.get("fp16_baseline", [])),
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
