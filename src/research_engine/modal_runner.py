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
DEFAULT_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"


@dataclass(slots=True)
class ModalBenchmarkResult:
    config: dict[str, int]
    config_id: str
    hardware: dict[str, str]
    results: list[dict]
    raw_output: str
    success: bool
    error: str = ""


def _build_modal_script(
    benchmark_script: str,
    *,
    gpu: str = DEFAULT_GPU,
    timeout_seconds: int = 300,
) -> str:
    """Wrap a benchmark script in a Modal app that runs it on a GPU."""
    gpu_type = MODAL_GPU_TYPES.get(gpu, "a100")
    return f'''#!/usr/bin/env python3
"""Modal wrapper for Triton kernel benchmark. Auto-generated."""

import modal

app = modal.App("noeris-triton-bench")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "triton",
)


@app.function(gpu="{gpu_type}", image=image, timeout={timeout_seconds})
def run_benchmark():
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    script = """{benchmark_script.replace(chr(92), chr(92)+chr(92)).replace('"', chr(92)+'"')}"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run(
            [sys.executable, f.name],
            capture_output=True,
            text=True,
            timeout={timeout_seconds - 30},
        )
    return {{
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }}


@app.local_entrypoint()
def main():
    import json
    result = run_benchmark.remote()
    if result["returncode"] != 0:
        print(json.dumps({{"error": result["stderr"], "returncode": result["returncode"]}}))
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


def run_benchmark_modal(
    config: dict[str, int],
    *,
    shapes: list[dict[str, int]] | None = None,
    gpu: str = DEFAULT_GPU,
) -> ModalBenchmarkResult:
    """Run a benchmark on Modal (requires modal CLI and auth)."""
    if shapes is None:
        shapes = MATMUL_SHAPE_BUCKETS[:6]

    benchmark_script = generate_triton_benchmark_script(config=config, shapes=shapes)
    modal_script = _build_modal_script(benchmark_script, gpu=gpu)
    cid = config_id(config)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
    ) as f:
        f.write(modal_script)
        modal_path = f.name

    try:
        result = subprocess.run(
            ["modal", "run", modal_path],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )
        if result.returncode != 0:
            return ModalBenchmarkResult(
                config=config,
                config_id=cid,
                hardware={"gpu": gpu},
                results=[],
                raw_output=result.stderr,
                success=False,
                error=result.stderr[:500],
            )

        # Parse the JSON output from the benchmark
        stdout = result.stdout.strip()
        # Modal may print extra lines; find the JSON
        json_start = stdout.find("{")
        if json_start < 0:
            return ModalBenchmarkResult(
                config=config,
                config_id=cid,
                hardware={"gpu": gpu},
                results=[],
                raw_output=stdout,
                success=False,
                error="No JSON found in Modal output",
            )
        data = json.loads(stdout[json_start:])
        return ModalBenchmarkResult(
            config=data.get("config", config),
            config_id=data.get("config_id", cid),
            hardware=data.get("hardware", {"gpu": gpu}),
            results=data.get("results", []),
            raw_output=stdout,
            success=True,
        )
    except FileNotFoundError:
        return ModalBenchmarkResult(
            config=config,
            config_id=cid,
            hardware={"gpu": gpu},
            results=[],
            raw_output="",
            success=False,
            error="modal CLI not found. Install with: pip install modal && modal setup",
        )
    except subprocess.TimeoutExpired:
        return ModalBenchmarkResult(
            config=config,
            config_id=cid,
            hardware={"gpu": gpu},
            results=[],
            raw_output="",
            success=False,
            error="Modal execution timed out after 600 seconds",
        )
    except json.JSONDecodeError as exc:
        return ModalBenchmarkResult(
            config=config,
            config_id=cid,
            hardware={"gpu": gpu},
            results=[],
            raw_output=stdout if 'stdout' in dir() else "",
            success=False,
            error=f"JSON decode error: {exc}",
        )
    finally:
        Path(modal_path).unlink(missing_ok=True)
