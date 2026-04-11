"""Modal-based GPU execution backend for Triton kernel benchmarks.

Sends self-contained benchmark scripts to Modal for execution on A100/H100
GPUs. Results stream back as JSON to stdout.

Usage from CLI:
    python -m research_engine.modal_runner --config '{"BLOCK_SIZE_M": 128, ...}'

Usage from CI:
    The triton-iterate command calls run_benchmark() directly.

Requires:
    pip install modal
    modal setup  (one-time auth)
    MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars for CI
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .triton_kernels import (
    MATMUL_SHAPE_BUCKETS,
    config_id,
    generate_triton_benchmark_script,
)

LOGGER = logging.getLogger(__name__)

MODAL_GPU_TYPES = {
    "T4": "t4",
    "A10G": "a10g",
    "A100": "a100",
    "A100-80GB": "a100-80gb",
    "H100": "h100",
}

DEFAULT_GPU = "A100"


@dataclass(slots=True)
class ModalBenchmarkResult:
    config: dict[str, int]
    config_id: str
    hardware: dict[str, str]
    results: list[dict]
    raw_output: str
    success: bool
    error: str = ""


def generate_batched_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, int]],
) -> str:
    """Generate a single script that benchmarks ALL configs on ALL shapes.

    One GPU invocation, no per-config cold starts. This is how you want
    to run it — a single Modal call that tests everything.
    """
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated batched Triton matmul benchmark."""

import json
import platform
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_offset in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_offset * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, config):
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"], GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
    return c


def benchmark_one(M, N, K, config, dtype=torch.float16):
    try:
        a = torch.randn((M, K), device="cuda", dtype=dtype)
        b = torch.randn((K, N), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)
        out = matmul(a, b, config)
        max_err = (out - ref).abs().max().item()
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        correct = max_err <= atol
        if not correct:
            return {{"correct": False, "max_err": max_err, "ms": None, "tflops": None}}
        ms = triton.testing.do_bench(lambda: matmul(a, b, config), warmup=25, rep=100)
        flops = 2.0 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        return {{"correct": True, "max_err": round(max_err, 8), "ms": round(ms, 4), "tflops": round(tflops, 2)}}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "tflops": None}}


def main():
    configs = {configs_json}
    shapes = {shapes_json}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bm{{}}_bn{{}}_bk{{}}_gm{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"],
            config["GROUP_SIZE_M"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            M, N, K = shape["M"], shape["N"], shape["K"]
            result = benchmark_one(M, N, K, config)
            result["shape"] = f"{{M}}x{{N}}x{{K}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "hardware": {{
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        }},
        "configs_tested": len(configs),
        "config_results": all_results,
    }}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
'''


def build_modal_app_script(
    benchmark_script: str,
    *,
    gpu: str = DEFAULT_GPU,
    timeout_seconds: int = 600,
) -> str:
    """Wrap a benchmark script in a Modal app for remote GPU execution."""
    gpu_type = MODAL_GPU_TYPES.get(gpu, "a100")
    # Escape the benchmark script for embedding
    escaped = benchmark_script.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    return f'''#!/usr/bin/env python3
"""Modal wrapper for Noeris Triton benchmark. Auto-generated."""
import modal

app = modal.App("noeris-triton-bench")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "triton",
)

@app.function(gpu="{gpu_type}", image=image, timeout={timeout_seconds})
def run_benchmark():
    import subprocess, sys, tempfile
    script = """{escaped}"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout={timeout_seconds - 60})
    return {{"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}}

@app.local_entrypoint()
def main():
    import json
    result = run_benchmark.remote()
    if result["returncode"] != 0:
        print(json.dumps({{"error": result["stderr"][:2000], "returncode": result["returncode"]}}))
    else:
        print(result["stdout"])
'''


def run_benchmark_local(
    config: dict[str, int],
    *,
    shapes: list[dict[str, int]] | None = None,
) -> ModalBenchmarkResult:
    """Run a benchmark locally (requires GPU and triton installed)."""
    if shapes is None:
        shapes = MATMUL_SHAPE_BUCKETS[:4]

    script = generate_triton_benchmark_script(config=config, shapes=shapes)
    cid = config_id(config)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return ModalBenchmarkResult(
                config=config,
                config_id=cid,
                hardware={},
                results=[],
                raw_output=result.stderr,
                success=False,
                error=result.stderr[:500],
            )
        data = json.loads(result.stdout)
        return ModalBenchmarkResult(
            config=data.get("config", config),
            config_id=data.get("config_id", cid),
            hardware=data.get("hardware", {}),
            results=data.get("results", []),
            raw_output=result.stdout,
            success=True,
        )
    except subprocess.TimeoutExpired:
        return ModalBenchmarkResult(
            config=config,
            config_id=cid,
            hardware={},
            results=[],
            raw_output="",
            success=False,
            error="Benchmark timed out after 300 seconds",
        )
    except json.JSONDecodeError as exc:
        return ModalBenchmarkResult(
            config=config,
            config_id=cid,
            hardware={},
            results=[],
            raw_output=result.stdout if 'result' in dir() else "",
            success=False,
            error=f"JSON decode error: {exc}",
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


def _extract_json_object(text: str, required_key: str) -> dict | None:
    """Find and parse a JSON object in mixed output containing required_key.

    Modal's output includes log lines, progress bars, and the actual
    benchmark JSON. This scans for a balanced brace pair containing the
    required key and tries to parse it.
    """
    # Try each { as a potential JSON start
    positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in positions:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start:i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict) and required_key in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


@dataclass(slots=True)
class BatchBenchmarkResult:
    hardware: dict[str, str]
    config_results: list[dict]
    raw_output: str
    success: bool
    error: str = ""


def run_benchmark_batch_modal(
    configs: list[dict[str, int]],
    *,
    shapes: list[dict[str, int]] | None = None,
    gpu: str = DEFAULT_GPU,
) -> BatchBenchmarkResult:
    """Run ALL configs in a single Modal GPU call. One cold start, minimal cost."""
    if shapes is None:
        shapes = MATMUL_SHAPE_BUCKETS[:6]

    benchmark_script = generate_batched_benchmark_script(configs, shapes)
    modal_script = build_modal_app_script(benchmark_script, gpu=gpu)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
    ) as f:
        f.write(modal_script)
        modal_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "modal", "run", modal_path],
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ},
        )
        if result.returncode != 0:
            return BatchBenchmarkResult(
                hardware={"gpu": gpu},
                config_results=[],
                raw_output=result.stderr,
                success=False,
                error=result.stderr[:500],
            )

        stdout = result.stdout
        data = _extract_json_object(stdout, "config_results")
        if data is None:
            return BatchBenchmarkResult(
                hardware={"gpu": gpu},
                config_results=[],
                raw_output=stdout[:2000],
                success=False,
                error=f"No valid JSON found. Output preview: {stdout[:500]}",
            )
        return BatchBenchmarkResult(
            hardware=data.get("hardware", {"gpu": gpu}),
            config_results=data.get("config_results", []),
            raw_output=stdout[-2000:],
            success=True,
        )
    except FileNotFoundError:
        return BatchBenchmarkResult(
            hardware={"gpu": gpu},
            config_results=[],
            raw_output="",
            success=False,
            error="modal not found. Install with: pip install modal && modal setup",
        )
    except subprocess.TimeoutExpired:
        return BatchBenchmarkResult(
            hardware={"gpu": gpu},
            config_results=[],
            raw_output="",
            success=False,
            error="Modal execution timed out",
        )
    except json.JSONDecodeError as exc:
        return BatchBenchmarkResult(
            hardware={"gpu": gpu},
            config_results=[],
            raw_output="",
            success=False,
            error=f"JSON decode error: {exc}",
        )
    finally:
        Path(modal_path).unlink(missing_ok=True)


def run_benchmark_modal(
    config: dict[str, int],
    *,
    shapes: list[dict[str, int]] | None = None,
    gpu: str = DEFAULT_GPU,
) -> ModalBenchmarkResult:
    """Run a single config on Modal. Prefer run_benchmark_batch_modal for multiple configs."""
    batch = run_benchmark_batch_modal([config], shapes=shapes, gpu=gpu)
    if not batch.success:
        return ModalBenchmarkResult(
            config=config,
            config_id=config_id(config),
            hardware=batch.hardware,
            results=[],
            raw_output=batch.raw_output,
            success=False,
            error=batch.error,
        )
    config_result = batch.config_results[0] if batch.config_results else {}
    return ModalBenchmarkResult(
        config=config_result.get("config", config),
        config_id=config_result.get("config_id", config_id(config)),
        hardware=batch.hardware,
        results=config_result.get("results", []),
        raw_output=batch.raw_output,
        success=True,
    )
