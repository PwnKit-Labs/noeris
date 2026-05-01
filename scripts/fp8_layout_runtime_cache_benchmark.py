#!/usr/bin/env python3
"""Benchmark FP8 runtime layout with prepack cache hit/miss workloads."""

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


SCRIPT_TEMPLATE = r'''
import json
import platform
import torch
import triton
import triton.language as tl

from research_engine.fp8_prepack_cache import Fp8PrepackCache
from research_engine.fp8_runtime import resolve_fp8_layout

POLICY_PAYLOAD = __POLICY_JSON__

CFG = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8}

SCENARIOS = [
    {
        "name": "s1024_reuse1_unique",
        "shape_name": "fp8_mm_1024",
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "expected_reuse": 1,
        "ops": 64,
        "keys": [f"k{i}" for i in range(64)],
    },
    {
        "name": "s1024_reuse8_hotset",
        "shape_name": "fp8_mm_1024",
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "expected_reuse": 8,
        "ops": 64,
        "keys": [f"k{i % 8}" for i in range(64)],
    },
    {
        "name": "s2048_reuse2",
        "shape_name": "fp8_mm_2048x1024x2048",
        "M": 2048,
        "N": 1024,
        "K": 2048,
        "expected_reuse": 2,
        "ops": 64,
        "keys": [f"k{i % 32}" for i in range(64)],
    },
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


def launch_kn(a, b_kn, c):
    M, K = a.shape
    _, N = b_kn.shape
    grid = (triton.cdiv(M, CFG["BLOCK_M"]) * triton.cdiv(N, CFG["BLOCK_N"]),)
    fp8_mm_kernel_grouped[grid](
        a, b_kn, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_kn.stride(0), b_kn.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=CFG["BLOCK_M"],
        BLOCK_N=CFG["BLOCK_N"],
        BLOCK_K=CFG["BLOCK_K"],
        GROUP_M=CFG["GROUP_M"],
        num_warps=CFG["num_warps"],
        num_stages=CFG["num_stages"],
    )


def launch_nk(a, b_nk, c):
    M, K = a.shape
    N, _ = b_nk.shape
    grid = (triton.cdiv(M, CFG["BLOCK_M"]) * triton.cdiv(N, CFG["BLOCK_N"]),)
    fp8_mm_kernel_grouped[grid](
        a, b_nk, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_nk.stride(1), b_nk.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=CFG["BLOCK_M"],
        BLOCK_N=CFG["BLOCK_N"],
        BLOCK_K=CFG["BLOCK_K"],
        GROUP_M=CFG["GROUP_M"],
        num_warps=CFG["num_warps"],
        num_stages=CFG["num_stages"],
    )


def run_mode(scenario, mode):
    M = int(scenario["M"])
    N = int(scenario["N"])
    K = int(scenario["K"])
    ops = int(scenario["ops"])
    keys = list(scenario["keys"])

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Pre-create KN weights by logical key.
    unique_keys = sorted(set(keys))
    weights_kn = {
        key: torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        for key in unique_keys
    }

    cache = Fp8PrepackCache(max_items=32)
    auto_layout = None
    if mode == "auto_policy_cache":
        auto_layout = resolve_fp8_layout(
            prefer="auto",
            shape_name=scenario["shape_name"],
            expected_reuse_count=int(scenario["expected_reuse"]),
            policy_payload=POLICY_PAYLOAD,
        )

    # Warmup a few operations.
    for i in range(min(8, ops)):
        key = keys[i]
        b_kn = weights_kn[key]
        if mode == "force_kn":
            launch_kn(a, b_kn, c)
        elif mode == "force_nk_no_cache":
            b_nk = b_kn.t().contiguous()
            launch_nk(a, b_nk, c)
        elif mode == "force_nk_cache":
            b_nk, _ = cache.get_or_create((scenario["shape_name"], key), lambda: b_kn.t().contiguous())
            launch_nk(a, b_nk, c)
        else:
            layout = auto_layout
            if layout == "kn":
                launch_kn(a, b_kn, c)
            else:
                b_nk, _ = cache.get_or_create((scenario["shape_name"], key), lambda: b_kn.t().contiguous())
                launch_nk(a, b_nk, c)

    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for i in range(ops):
        key = keys[i]
        b_kn = weights_kn[key]
        if mode == "force_kn":
            launch_kn(a, b_kn, c)
        elif mode == "force_nk_no_cache":
            b_nk = b_kn.t().contiguous()
            launch_nk(a, b_nk, c)
        elif mode == "force_nk_cache":
            b_nk, _ = cache.get_or_create((scenario["shape_name"], key), lambda: b_kn.t().contiguous())
            launch_nk(a, b_nk, c)
        else:
            layout = auto_layout
            if layout == "kn":
                launch_kn(a, b_kn, c)
            else:
                b_nk, _ = cache.get_or_create((scenario["shape_name"], key), lambda: b_kn.t().contiguous())
                launch_nk(a, b_nk, c)
    en.record()
    torch.cuda.synchronize()
    total_ms = st.elapsed_time(en)
    avg_ms = total_ms / float(ops)

    stats = cache.stats()
    return {
        "mode": mode,
        "avg_ms": round(float(avg_ms), 4),
        "total_ms": round(float(total_ms), 4),
        "cache_hits": int(stats.hits),
        "cache_misses": int(stats.misses),
        "cache_evictions": int(stats.evictions),
    }


def run_scenario(scenario):
    modes = ["force_kn", "force_nk_no_cache", "force_nk_cache", "auto_policy_cache"]
    rows = [run_mode(scenario, m) for m in modes]
    best = min(rows, key=lambda r: float(r["avg_ms"]))
    for r in rows:
        r["vs_best"] = round(float(r["avg_ms"]) / float(best["avg_ms"]), 4) if best["avg_ms"] > 0 else 0.0
    return {
        "scenario": scenario,
        "results": rows,
        "best_mode": best["mode"],
    }


def main():
    rows = [run_scenario(s) for s in SCENARIOS]
    print(json.dumps({
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        },
        "config_results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
'''


def _to_md(payload: dict) -> str:
    lines = [
        "# FP8 Runtime Cache Integration (H100)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Scenario | mode | avg ms/op | vs best | cache hits | cache misses |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["results"]["H100"]["config_results"]:
        scenario_name = row["scenario"]["name"]
        for mode in row["results"]:
            lines.append(
                f"| {scenario_name} | {mode['mode']} | {mode['avg_ms']:.4f} | {mode['vs_best']:.4f} | "
                f"{mode['cache_hits']} | {mode['cache_misses']} |"
            )
    lines += [
        "",
        "`auto_policy_cache` combines policy layout choice with prepack cache reuse.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-seconds", type=int, default=3000)
    parser.add_argument("--max-cost-usd", type=float, default=4.0)
    parser.add_argument("--policy-json", default="docs/results/fp8-layout-reuse-policy-h100.json")
    parser.add_argument("--output-json", default="docs/results/fp8-layout-runtime-cache-integration-h100.json")
    parser.add_argument("--output-md", default="docs/results/fp8-layout-runtime-cache-integration-h100.md")
    args = parser.parse_args()

    policy_payload = json.loads((ROOT / args.policy_json).read_text(encoding="utf-8"))
    script = SCRIPT_TEMPLATE.replace("__POLICY_JSON__", json.dumps(policy_payload))

    with ModalBenchmarkSession(
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        local_source_dir=str((ROOT / "src/research_engine").resolve()),
    ) as session:
        out = session.run_script(script)
    if not out.success:
        raise RuntimeError(f"{args.gpu} FP8 runtime cache benchmark failed: {out.error}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": [args.gpu],
        "policy_json": args.policy_json,
        "results": {
            args.gpu: {
                "hardware": out.hardware,
                "config_results": out.config_results,
            }
        },
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
