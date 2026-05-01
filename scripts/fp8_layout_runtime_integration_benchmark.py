#!/usr/bin/env python3
"""Benchmark FP8 runtime layout integration using policy-driven auto mode."""

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

from research_engine.fp8_runtime import resolve_fp8_layout
from research_engine.fp8_layout_policy import choose_layout_from_policy

POLICY_PAYLOAD = __POLICY_JSON__

SHAPES = [
    {"name": "fp8_mm_1024", "M": 1024, "N": 1024, "K": 1024, "expected_reuse": 1},
    {"name": "fp8_mm_2048x1024x2048", "M": 2048, "N": 1024, "K": 2048, "expected_reuse": 2},
    {"name": "fp8_mm_4096x4096x4096", "M": 4096, "N": 4096, "K": 4096, "expected_reuse": 1},
]

CFG = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4, "GROUP_M": 8}


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


def median_ms(fn, warmup=25, trials=100):
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


def _launch(a, b, c):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, CFG["BLOCK_M"]) * triton.cdiv(N, CFG["BLOCK_N"]),)
    fp8_mm_kernel_grouped[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=CFG["BLOCK_M"],
        BLOCK_N=CFG["BLOCK_N"],
        BLOCK_K=CFG["BLOCK_K"],
        GROUP_M=CFG["GROUP_M"],
        num_warps=CFG["num_warps"],
        num_stages=CFG["num_stages"],
    )


def run_shape(shape):
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    reuse = int(shape["expected_reuse"])
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b_kn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b_nk = b_kn.t().contiguous()
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def run_kn():
        _launch(a, b_kn, c)

    def run_nk():
        # Pass b_nk with transposed strides to match kernel's KxN interpretation.
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

    run_kn()
    ref = (a.float() @ b_kn.float()).half()
    kn_err = (c - ref).abs().max().item()

    run_nk()
    nk_err = (c - ref).abs().max().item()

    kn_ms = median_ms(run_kn)
    nk_ms = median_ms(run_nk)

    policy_decision = choose_layout_from_policy(POLICY_PAYLOAD, shape["name"], reuse)

    auto_layout = resolve_fp8_layout(
        prefer="auto",
        shape_name=shape["name"],
        expected_reuse_count=reuse,
        policy_payload=POLICY_PAYLOAD,
    )
    auto_effective_ms = policy_decision.nk_total_ms if auto_layout == "nk" else policy_decision.kn_total_ms

    kernel_best_ms = min(kn_ms, nk_ms)
    kernel_best_layout = "kn" if kn_ms <= nk_ms else "nk"
    policy_best_ms = min(policy_decision.kn_total_ms, policy_decision.nk_total_ms)
    policy_best_layout = "kn" if policy_decision.kn_total_ms <= policy_decision.nk_total_ms else "nk"

    return {
        "shape_name": shape["name"],
        "expected_reuse": reuse,
        "correct": bool(kn_err <= 0.5 and nk_err <= 0.5),
        "kn_ms": round(kn_ms, 4),
        "nk_ms": round(nk_ms, 4),
        "policy_kn_total_ms": round(policy_decision.kn_total_ms, 4),
        "policy_nk_total_ms": round(policy_decision.nk_total_ms, 4),
        "auto_effective_ms": round(auto_effective_ms, 4),
        "auto_layout": auto_layout,
        "kernel_best_layout": kernel_best_layout,
        "policy_best_layout": policy_best_layout,
        "auto_vs_kernel_best_ratio": round((nk_ms if auto_layout == "nk" else kn_ms) / kernel_best_ms, 4) if kernel_best_ms > 0 else 0.0,
        "auto_vs_policy_best_ratio": round(auto_effective_ms / policy_best_ms, 4) if policy_best_ms > 0 else 0.0,
        "max_err": round(float(max(kn_err, nk_err)), 6),
    }


def main():
    rows = [run_shape(s) for s in SHAPES]
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
        "# FP8 Layout Runtime Integration (H100)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Shape | reuse | kn ms | nk ms | policy kn total | policy nk total | auto layout | kernel best | policy best | auto/policy |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for row in payload["results"]["H100"]["config_results"]:
        lines.append(
            f"| {row['shape_name']} | {row['expected_reuse']} | {row['kn_ms']:.4f} | {row['nk_ms']:.4f} | "
            f"{row['policy_kn_total_ms']:.4f} | {row['policy_nk_total_ms']:.4f} | {row['auto_layout']} | "
            f"{row['kernel_best_layout']} | {row['policy_best_layout']} | {row['auto_vs_policy_best_ratio']:.4f} |"
        )
    lines += [
        "",
        "Kernel best compares raw kernel latency only; policy best includes prepack amortization.",
        "auto/policy near 1.0 indicates runtime auto follows the policy decision exactly.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-seconds", type=int, default=3000)
    parser.add_argument("--max-cost-usd", type=float, default=3.5)
    parser.add_argument("--policy-json", default="docs/results/fp8-layout-reuse-policy-h100.json")
    parser.add_argument("--output-json", default="docs/results/fp8-layout-runtime-integration-h100.json")
    parser.add_argument("--output-md", default="docs/results/fp8-layout-runtime-integration-h100.md")
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
        raise RuntimeError(f"{args.gpu} FP8 runtime layout benchmark failed: {out.error}")

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
