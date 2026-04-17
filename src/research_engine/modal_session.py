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
from pathlib import Path

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


GPU_COST_PER_SEC = {
    "A100": 2.10 / 3600,   # $2.10/hr
    "H100": 3.90 / 3600,   # $3.90/hr
    "T4":   0.59 / 3600,   # $0.59/hr
}


class ModalBudgetExceeded(RuntimeError):
    """Raised when a ModalBenchmarkSession exceeds its per-session cost cap."""


class ModalBenchmarkSession:
    """Persistent Modal session that runs multiple benchmark scripts.

    Each call to .run_script(script) sends the script to a warm Modal
    container that executes it and returns the stdout/stderr/returncode.
    The caller parses the output as JSON.

    **Cost guardrail**: ``max_cost_usd`` (default $1.00) caps the
    cumulative estimated GPU cost within a single session. If a
    ``.run_script()`` call would push the session over this cap,
    ``ModalBudgetExceeded`` is raised and no further GPU work runs.
    Set ``max_cost_usd=0`` to disable the cap (not recommended).

    Must be used as a context manager:

        with ModalBenchmarkSession(gpu="A100", max_cost_usd=0.50) as session:
            result = session.run_script(my_script)
    """

    def __init__(
        self,
        *,
        gpu: str = "A100",
        timeout_seconds: int = 600,
        max_cost_usd: float = 1.00,
        local_source_dir: str | None = None,
    ) -> None:
        self.gpu = gpu
        self.timeout_seconds = timeout_seconds
        self.max_cost_usd = max_cost_usd
        self.local_source_dir = local_source_dir
        self._cost_per_sec = GPU_COST_PER_SEC.get(gpu, GPU_COST_PER_SEC["A100"])
        self._cumulative_cost = 0.0
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
        remote_pythonpath = ""
        if self.local_source_dir:
            source_dir = Path(self.local_source_dir).resolve()
            image = image.add_local_dir(str(source_dir), remote_path="/root/src/research_engine")
            remote_pythonpath = "/root/src"

        timeout_s = self.timeout_seconds

        @app.function(gpu=gpu_type, image=image, timeout=timeout_s, serialized=True)
        def run_benchmark_script(script: str) -> dict:
            import os
            import subprocess
            import sys
            import tempfile

            env = os.environ.copy()
            if remote_pythonpath:
                current = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    f"{remote_pythonpath}:{current}" if current else remote_pythonpath
                )

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
                    env=env,
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
        """Run a benchmark script on the warm GPU container.

        Raises ModalBudgetExceeded if the cumulative session cost would
        exceed ``max_cost_usd`` after this call.
        """
        if self._run_function is None:
            raise RuntimeError(
                "ModalBenchmarkSession must be used as a context manager"
            )
        # Pre-flight budget check: estimate this call will take at most
        # timeout_seconds of GPU time.  If that would blow the cap, bail.
        if self.max_cost_usd > 0:
            worst_case = self._cumulative_cost + self.timeout_seconds * self._cost_per_sec
            if worst_case > self.max_cost_usd:
                raise ModalBudgetExceeded(
                    f"Session budget cap ${self.max_cost_usd:.2f} would be "
                    f"exceeded (cumulative ${self._cumulative_cost:.3f} + "
                    f"worst-case ${self.timeout_seconds * self._cost_per_sec:.3f} "
                    f"for a {self.timeout_seconds}s timeout on {self.gpu}). "
                    f"Raise max_cost_usd or reduce timeout_seconds."
                )

        import time as _time
        _t0 = _time.monotonic()

        try:
            result = self._run_function.remote(script)
        except Exception as exc:
            _elapsed = _time.monotonic() - _t0
            self._cumulative_cost += _elapsed * self._cost_per_sec
            return SessionResult(
                success=False,
                hardware={"gpu": self.gpu},
                config_results=[],
                extra={},
                stdout="",
                stderr="",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )

        # Track actual wall-clock cost after the call completes
        _elapsed = _time.monotonic() - _t0
        self._cumulative_cost += _elapsed * self._cost_per_sec

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
