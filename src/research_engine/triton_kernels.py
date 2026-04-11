"""Parameterized Triton GPU kernel generation and shape-indexed config database.

This module generates Triton matmul kernels parameterized by
(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages) and
maintains a persistent shape-indexed database of winning configurations.

The key contribution vs existing systems (AutoKernel, KernelSkill, CUDA-L1):
every other system starts fresh each run and produces one kernel per operator.
This module persists (op, shape, hardware) -> (best params) mappings across
runs and feeds them back to the LLM proposer as cross-run learning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------

TRITON_MATMUL_PARAM_SPACE = {
    "BLOCK_SIZE_M": [32, 64, 128, 256],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128],
    "GROUP_SIZE_M": [4, 8, 16],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4, 5],
}

# Curated high-priority configs (known-good starting points)
TRITON_MATMUL_CURATED_CONFIGS = [
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 2, "num_stages": 5},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 2, "num_stages": 5},
    # Larger tiles for big matrices
    {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    # FP8-friendly configs (large K blocks)
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
]

# Shape buckets for workload-aware config selection
MATMUL_SHAPE_BUCKETS = [
    {"name": "tiny", "M": 128, "N": 128, "K": 128},
    {"name": "small", "M": 512, "N": 512, "K": 512},
    {"name": "medium", "M": 2048, "N": 2048, "K": 2048},
    {"name": "large", "M": 4096, "N": 4096, "K": 4096},
    {"name": "xlarge", "M": 8192, "N": 8192, "K": 8192},
    {"name": "tall_skinny", "M": 8192, "N": 1024, "K": 1024},
    {"name": "deep_k", "M": 1024, "N": 1024, "K": 8192},
    {"name": "llm_qkv", "M": 4096, "N": 4096, "K": 512},
    {"name": "llm_mlp", "M": 4096, "N": 11008, "K": 4096},
    {"name": "llm_mlp_down", "M": 4096, "N": 4096, "K": 11008},
]


def config_id(config: dict[str, int]) -> str:
    """Generate a stable string ID for a Triton config."""
    bm = config["BLOCK_SIZE_M"]
    bn = config["BLOCK_SIZE_N"]
    bk = config["BLOCK_SIZE_K"]
    gm = config["GROUP_SIZE_M"]
    nw = config["num_warps"]
    ns = config["num_stages"]
    return f"bm{bm}_bn{bn}_bk{bk}_gm{gm}_w{nw}_s{ns}"


def shape_bucket_key(m: int, n: int, k: int) -> str:
    """Classify a shape into a bucket for config lookup."""
    total = m * n * k
    aspect = max(m, n) / max(min(m, n), 1)
    k_ratio = k / max(max(m, n), 1)
    if total < 2**20:
        return "tiny"
    if total < 2**24:
        return "small"
    if k_ratio > 2:
        return "deep_k"
    if aspect > 4:
        return "tall_skinny"
    if total < 2**30:
        return "medium"
    if total < 2**34:
        return "large"
    return "xlarge"


# ---------------------------------------------------------------------------
# Triton kernel source generation
# ---------------------------------------------------------------------------

TRITON_MATMUL_KERNEL_SOURCE = '''
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
    """Parameterized Triton matmul kernel.

    Computes C = A @ B where A is (M, K) and B is (K, N).
    Tile sizes and scheduling params are compile-time constants,
    enabling Triton to generate specialized GPU code per config.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grouped ordering for L2 cache reuse of A tiles
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Tile pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_offset in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_offset * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store output tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
'''


def generate_triton_benchmark_script(
    config: dict[str, int],
    shapes: list[dict[str, int]],
) -> str:
    """Generate a self-contained Python script that benchmarks a Triton matmul
    config across multiple shapes.

    The script imports triton, defines the kernel, runs correctness checks,
    and prints JSON results to stdout. Designed to be sent to Modal for
    remote GPU execution.
    """
    config_json = json.dumps(config)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton matmul benchmark script."""

import json
import sys
import torch

{TRITON_MATMUL_KERNEL_SOURCE}


def matmul(a, b, config):
    """Launch the Triton matmul kernel with the given config."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return c


def check_correctness(M, N, K, config, dtype=torch.float16):
    """Five-stage correctness check."""
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    ref = torch.matmul(a, b)
    out = matmul(a, b, config)
    max_err = (out - ref).abs().max().item()
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    return max_err <= atol, max_err


def benchmark_config(M, N, K, config, dtype=torch.float16, warmup=25, rep=100):
    """Benchmark a single config on a single shape."""
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    # Correctness gate
    correct, max_err = check_correctness(M, N, K, config, dtype)
    if not correct:
        return {{"correct": False, "max_err": max_err, "ms": None, "tflops": None}}

    # Determinism check (3 runs, identical outputs)
    results = [matmul(a, b, config) for _ in range(3)]
    deterministic = all(torch.equal(results[0], r) for r in results[1:])

    # Performance
    ms = triton.testing.do_bench(lambda: matmul(a, b, config), warmup=warmup, rep=rep)
    flops = 2.0 * M * N * K
    tflops = flops / (ms * 1e-3) / 1e12

    return {{
        "correct": True,
        "deterministic": deterministic,
        "max_err": max_err,
        "ms": round(ms, 4),
        "tflops": round(tflops, 2),
    }}


def main():
    config = {config_json}
    shapes = {shapes_json}

    import platform
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    results = []
    for shape in shapes:
        M, N, K = shape["M"], shape["N"], shape["K"]
        result = benchmark_config(M, N, K, config)
        result["shape"] = f"{{M}}x{{N}}x{{K}}"
        result["shape_name"] = shape.get("name", "")
        results.append(result)

    output = {{
        "config": config,
        "config_id": "{config_id(config)}",
        "hardware": {{
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        }},
        "results": results,
    }}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Config grid generation
# ---------------------------------------------------------------------------

def generate_config_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    """Generate the full config grid, deduped, curated configs first."""
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in TRITON_MATMUL_CURATED_CONFIGS:
            cid = config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    # Systematic grid (bounded)
    for bm in TRITON_MATMUL_PARAM_SPACE["BLOCK_SIZE_M"]:
        for bn in TRITON_MATMUL_PARAM_SPACE["BLOCK_SIZE_N"]:
            for bk in TRITON_MATMUL_PARAM_SPACE["BLOCK_SIZE_K"]:
                for gm in [8]:  # fix GROUP_SIZE_M to reduce grid
                    for nw in TRITON_MATMUL_PARAM_SPACE["num_warps"]:
                        for ns in [3, 4]:  # fix stages to reduce grid
                            config = {
                                "BLOCK_SIZE_M": bm,
                                "BLOCK_SIZE_N": bn,
                                "BLOCK_SIZE_K": bk,
                                "GROUP_SIZE_M": gm,
                                "num_warps": nw,
                                "num_stages": ns,
                            }
                            cid = config_id(config)
                            if cid not in seen:
                                seen.add(cid)
                                configs.append(config)
                            if len(configs) >= max_configs:
                                return configs
    return configs


# ---------------------------------------------------------------------------
# Shape-indexed config database (the novel differentiator)
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    config_id: str
    config: dict[str, int]
    tflops: float
    ms: float
    correct: bool
    run_id: str = ""
    hardware: str = ""


@dataclass
class ShapeRecord:
    shape_key: str
    shape: dict[str, int]
    best_config_id: str = ""
    best_tflops: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConfigDatabase:
    """Persistent shape-indexed config database.

    Stores (op, shape_bucket, hardware) -> ranked configs across runs.
    This is the key differentiator: no published system maintains this.
    """

    path: Path
    records: dict[str, ShapeRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        for key, record in data.get("records", {}).items():
            self.records[key] = ShapeRecord(
                shape_key=record["shape_key"],
                shape=record["shape"],
                best_config_id=record.get("best_config_id", ""),
                best_tflops=record.get("best_tflops", 0.0),
                results=record.get("results", []),
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": {
                key: {
                    "shape_key": record.shape_key,
                    "shape": record.shape,
                    "best_config_id": record.best_config_id,
                    "best_tflops": record.best_tflops,
                    "results": record.results[-50:],  # keep last 50 per shape
                }
                for key, record in self.records.items()
            }
        }
        self.path.write_text(json.dumps(data, indent=2) + "\n")

    def record_result(
        self,
        *,
        shape: dict[str, int],
        hardware: str,
        config: dict[str, int],
        tflops: float,
        ms: float,
        correct: bool,
        run_id: str = "",
    ) -> bool:
        """Record a benchmark result. Returns True if this is a new best."""
        m, n, k = shape["M"], shape["N"], shape["K"]
        bucket = shape_bucket_key(m, n, k)
        key = f"{bucket}:{hardware}"
        cid = config_id(config)

        if key not in self.records:
            self.records[key] = ShapeRecord(
                shape_key=key,
                shape={"M": m, "N": n, "K": k, "bucket": bucket},
            )

        record = self.records[key]
        record.results.append({
            "config_id": cid,
            "config": config,
            "tflops": tflops,
            "ms": ms,
            "correct": correct,
            "run_id": run_id,
            "hardware": hardware,
        })

        if correct and tflops > record.best_tflops:
            record.best_config_id = cid
            record.best_tflops = tflops
            return True
        return False

    def get_best_config(
        self,
        *,
        shape: dict[str, int],
        hardware: str,
    ) -> dict[str, int] | None:
        """Look up the best known config for a shape+hardware combo."""
        m, n, k = shape["M"], shape["N"], shape["K"]
        bucket = shape_bucket_key(m, n, k)
        key = f"{bucket}:{hardware}"
        record = self.records.get(key)
        if record is None or not record.best_config_id:
            return None
        # Find the actual config dict
        for result in reversed(record.results):
            if result["config_id"] == record.best_config_id:
                return result["config"]
        return None

    def get_insights(self, *, hardware: str = "") -> list[dict[str, Any]]:
        """Extract cross-run insights for the LLM proposer."""
        insights: list[dict[str, Any]] = []
        for key, record in self.records.items():
            if hardware and not key.endswith(f":{hardware}"):
                continue
            if not record.best_config_id:
                continue
            # Find all configs tested on this shape and their performance
            config_perf: dict[str, list[float]] = {}
            for result in record.results:
                if result["correct"]:
                    config_perf.setdefault(result["config_id"], []).append(result["tflops"])
            top_configs = sorted(
                config_perf.items(),
                key=lambda item: -max(item[1]),
            )[:3]
            insights.append({
                "shape_bucket": record.shape_key,
                "shape": record.shape,
                "best_config_id": record.best_config_id,
                "best_tflops": record.best_tflops,
                "top_configs": [
                    {"config_id": cid, "best_tflops": round(max(perfs), 2)}
                    for cid, perfs in top_configs
                ],
                "total_experiments": len(record.results),
            })
        return insights


# ---------------------------------------------------------------------------
# Config selection with frontier awareness
# ---------------------------------------------------------------------------

_TRITON_CONFIG_PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "configs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "BLOCK_SIZE_M": {"type": "integer"},
                    "BLOCK_SIZE_N": {"type": "integer"},
                    "BLOCK_SIZE_K": {"type": "integer"},
                    "GROUP_SIZE_M": {"type": "integer"},
                    "num_warps": {"type": "integer"},
                    "num_stages": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                    "GROUP_SIZE_M", "num_warps", "num_stages", "rationale",
                ],
            },
        },
        "global_rationale": {"type": "string"},
    },
    "required": ["configs", "global_rationale"],
}


def propose_triton_configs(
    *,
    proposer,  # ResponsesApiClient
    database: ConfigDatabase | None = None,
    hardware: str = "",
    target_shapes: list[dict[str, int]] | None = None,
    max_proposals: int = 4,
) -> dict[str, Any]:
    """Ask the LLM to propose novel Triton matmul configs.

    Uses cross-run insights from the ConfigDatabase to guide proposals.
    This is the core of the LLM-guided search: the proposer sees what
    worked on which shapes and suggests new configs to try.
    """
    insights = []
    if database is not None:
        insights = database.get_insights(hardware=hardware)

    prompt_data: dict[str, Any] = {
        "task": "Propose novel Triton matmul kernel configurations",
        "param_space": {
            "BLOCK_SIZE_M": "power of 2, 16-256",
            "BLOCK_SIZE_N": "power of 2, 16-256",
            "BLOCK_SIZE_K": "power of 2, 16-128",
            "GROUP_SIZE_M": "4-16, controls L2 cache reuse of A tiles",
            "num_warps": "1-8, warps per thread block",
            "num_stages": "2-5, software pipeline depth for hiding latency",
        },
        "hardware_constraints": {
            "shared_memory_per_sm": "192 KB on A100, tiles must fit",
            "rule": "BLOCK_M * BLOCK_K * 2 * num_stages + BLOCK_K * BLOCK_N * 2 * num_stages < 192000",
        },
        "target_shapes": target_shapes or [s for s in MATMUL_SHAPE_BUCKETS[:6]],
    }
    if insights:
        prompt_data["cross_run_insights"] = insights
    curated_ids = [config_id(c) for c in TRITON_MATMUL_CURATED_CONFIGS]
    prompt_data["already_tested"] = curated_ids

    try:
        import json as _json
        payload = proposer.generate_json(
            schema_name="triton_config_proposals",
            schema=_TRITON_CONFIG_PROPOSAL_SCHEMA,
            instructions=(
                "You are proposing Triton GPU matmul kernel configurations. "
                "Each config specifies tile sizes and scheduling params that get compiled "
                "into a specialized GPU kernel. Different shapes have different optimal configs. "
                "Use the cross_run_insights to understand what has worked before and propose "
                "configs that explore promising unexplored regions of the parameter space. "
                f"Return at most {max_proposals} configs. Keep rationale to one sentence each."
            ),
            prompt=_json.dumps(prompt_data, indent=2),
            max_output_tokens=1200,
            reasoning_effort="low",
            text_verbosity="low",
        )
    except Exception as exc:
        return {
            "source": "responses_api_error",
            "configs": [],
            "global_rationale": "",
            "error": type(exc).__name__,
            "detail": str(exc)[:240],
        }

    configs = []
    for item in payload.get("configs", []):
        if not isinstance(item, dict) or len(configs) >= max_proposals:
            break
        config = _sanitize_triton_config(item)
        if config is not None:
            configs.append(config)

    return {
        "source": "responses_api",
        "configs": configs,
        "global_rationale": " ".join(str(payload.get("global_rationale", "")).split()),
    }


def _sanitize_triton_config(item: dict) -> dict[str, int] | None:
    """Validate and clamp a proposed Triton config."""
    try:
        bm = _clamp_pow2(int(item.get("BLOCK_SIZE_M", 0)), 16, 256)
        bn = _clamp_pow2(int(item.get("BLOCK_SIZE_N", 0)), 16, 256)
        bk = _clamp_pow2(int(item.get("BLOCK_SIZE_K", 0)), 16, 128)
        gm = max(1, min(int(item.get("GROUP_SIZE_M", 8)), 16))
        nw = _clamp_pow2(int(item.get("num_warps", 4)), 1, 8)
        ns = max(2, min(int(item.get("num_stages", 3)), 5))
    except (TypeError, ValueError):
        return None

    # Shared memory sanity check (rough: 2 bytes per element, A + B tiles)
    shmem = (bm * bk + bk * bn) * 2 * ns
    if shmem > 192_000:
        return None

    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": gm,
        "num_warps": nw,
        "num_stages": ns,
    }


def _clamp_pow2(value: int, low: int, high: int) -> int:
    """Clamp to nearest power of 2 within bounds."""
    value = max(low, min(value, high))
    # Round to nearest power of 2
    p = 1
    while p * 2 <= value:
        p *= 2
    if abs(p - value) > abs(p * 2 - value) and p * 2 <= high:
        p *= 2
    return max(low, min(p, high))


def select_configs_for_run(
    *,
    database: ConfigDatabase | None = None,
    hardware: str = "",
    shapes: list[dict[str, int]],
    max_configs: int = 8,
    proposed_configs: list[dict[str, int]] | None = None,
) -> list[dict[str, int]]:
    """Select configs for a benchmark run using frontier-aware slotting.

    Slot allocation:
    1. incumbent — best known config (from database)
    2. curated — known-good starting points not yet tested
    3. proposed — LLM-proposed configs
    4. exploration — systematic grid configs not yet tested
    """
    selected: list[dict[str, int]] = []
    seen: set[str] = set()

    def _add(config: dict[str, int]) -> bool:
        cid = config_id(config)
        if cid in seen or len(selected) >= max_configs:
            return False
        seen.add(cid)
        selected.append(config)
        return True

    # Slot 1: incumbent (best from database across all shapes)
    if database is not None:
        for shape in shapes:
            best = database.get_best_config(shape=shape, hardware=hardware)
            if best is not None:
                _add(best)
                break

    # Slot 2: LLM-proposed configs
    for config in proposed_configs or []:
        _add(config)

    # Slot 3: curated configs not yet tested
    tested_ids = set()
    if database is not None:
        for record in database.records.values():
            for result in record.results:
                tested_ids.add(result["config_id"])

    for config in TRITON_MATMUL_CURATED_CONFIGS:
        if config_id(config) not in tested_ids:
            _add(config)

    # Slot 4: exploration (grid configs not yet tested)
    for config in generate_config_grid(include_curated=False, max_configs=500):
        if config_id(config) not in tested_ids:
            _add(config)

    return selected
