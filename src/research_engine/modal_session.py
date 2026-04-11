"""Persistent Modal session runner for fast multi-iteration workflows.

The existing modal_runner.run_benchmark_batch_modal_generic() spawns a new
`modal run` subprocess per invocation. Each subprocess:

1. Imports modal
2. Creates an app from scratch
3. Builds/reuses image
4. Connects to Modal control plane
5. Runs the function
6. Tears down the app

Steps 1-4 and 6 are the overhead (~5-10 seconds per call). For a multi-trial
ablation that does 30 sequential calls, that's 2.5-5 minutes of pure overhead.

This module keeps a single Modal app alive across many calls by using
modal.App.run() as a Python context manager and invoking the function
programmatically via .remote(). The container stays warm between calls.

Usage:
    with ModalBenchmarkSession(gpu="A100") as session:
        for script in my_scripts:
            result = session.run_script(script)

Each .run_script() is ~0.5-2s in the warm path vs ~10-20s for subprocess spawn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .modal_runner import (
    BatchBenchmarkResult,
    MODAL_GPU_TYPES,
    _extract_json_object,
)


@dataclass(slots=True)
class SessionResult:
    """Result of one script invocation within a Modal session."""

    success: bool
    hardware: dict
    config_results: list[dict]
    extra: dict
    stdout: str
    stderr: str
    error: str = ""


class ModalBenchmarkSession:
    """Persistent Modal session that runs multiple benchmark scripts.

    Each call to .run_script(script) sends the script to a warm Modal
    container that executes it and returns the stdout/stderr/returncode.
    The caller parses the output as JSON.

    Must be used as a context manager:

        with ModalBenchmarkSession(gpu="A100") as session:
            result = session.run_script(my_script)
    """

    def __init__(self, *, gpu: str = "A100", timeout_seconds: int = 600) -> None:
        self.gpu = gpu
        self.timeout_seconds = timeout_seconds
        self._app = None
        self._run_function = None
        self._app_ctx = None

    def __enter__(self) -> "ModalBenchmarkSession":
        import modal

        gpu_type = MODAL_GPU_TYPES.get(self.gpu, "a100")
        app = modal.App("noeris-session-bench")
        image = modal.Image.debian_slim(python_version="3.11").pip_install(
            "torch", "triton",
        )

        timeout_s = self.timeout_seconds

        @app.function(gpu=gpu_type, image=image, timeout=timeout_s, serialized=True)
        def run_benchmark_script(script: str) -> dict:
            import subprocess
            import sys
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
            ) as f:
                f.write(script)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s - 60,
                )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        self._app = app
        self._run_function = run_benchmark_script
        self._app_ctx = app.run()
        self._app_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._app_ctx is not None:
            try:
                self._app_ctx.__exit__(exc_type, exc_val, exc_tb)
            finally:
                self._app_ctx = None
                self._app = None
                self._run_function = None

    def run_script(self, script: str) -> SessionResult:
        """Run a benchmark script on the warm GPU container."""
        if self._run_function is None:
            raise RuntimeError(
                "ModalBenchmarkSession must be used as a context manager"
            )
        try:
            result = self._run_function.remote(script)
        except Exception as exc:
            return SessionResult(
                success=False,
                hardware={"gpu": self.gpu},
                config_results=[],
                extra={},
                stdout="",
                stderr="",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        returncode = result.get("returncode", -1)

        if returncode != 0:
            return SessionResult(
                success=False,
                hardware={"gpu": self.gpu},
                config_results=[],
                extra={},
                stdout=stdout,
                stderr=stderr,
                error=stderr[:500] if stderr else f"returncode={returncode}",
            )

        data = _extract_json_object(stdout, "config_results")
        if data is None:
            return SessionResult(
                success=False,
                hardware={"gpu": self.gpu},
                config_results=[],
                extra={},
                stdout=stdout,
                stderr=stderr,
                error=f"No JSON object found in stdout. Preview: {stdout[:300]}",
            )

        extras = {k: v for k, v in data.items() if k not in ("hardware", "config_results")}
        return SessionResult(
            success=True,
            hardware=data.get("hardware", {"gpu": self.gpu}),
            config_results=data.get("config_results", []),
            extra=extras,
            stdout=stdout,
            stderr=stderr,
        )

    def run_batch(self, script: str) -> BatchBenchmarkResult:
        """Run a script and return in the legacy BatchBenchmarkResult shape.

        Convenience for callers that already consume BatchBenchmarkResult.
        """
        sess = self.run_script(script)
        return BatchBenchmarkResult(
            hardware=sess.hardware,
            config_results=sess.config_results,
            raw_output=sess.stdout[-2000:] if sess.stdout else "",
            success=sess.success,
            error=sess.error,
            extra=sess.extra,
        )
