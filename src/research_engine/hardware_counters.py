"""Hardware counter (NCU) integration infrastructure.

Provides structured parsing of NVIDIA Nsight Compute (NCU) profiling output,
feature extraction for the ranking surrogate, and NCU-augmented benchmark
script generation.  All code paths gracefully degrade when NCU is unavailable.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. HardwareCounters dataclass
# ---------------------------------------------------------------------------

@dataclass
class HardwareCounters:
    """Hardware performance counters from NCU profiling."""

    # Utilization percentages (0-100)
    compute_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    l2_cache_hit_rate: float = 0.0
    achieved_occupancy: float = 0.0
    sm_efficiency: float = 0.0

    # Absolute throughputs
    dram_throughput_gb_s: float = 0.0
    compute_throughput_tflops: float = 0.0

    # Stall breakdown (% of cycles)
    stall_memory_dependency: float = 0.0
    stall_execution_dependency: float = 0.0
    stall_not_selected: float = 0.0

    # Raw NCU metrics dict for extensibility
    raw_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON storage."""
        return {
            "compute_utilization": self.compute_utilization,
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization,
            "l2_cache_hit_rate": self.l2_cache_hit_rate,
            "achieved_occupancy": self.achieved_occupancy,
            "sm_efficiency": self.sm_efficiency,
            "dram_throughput_gb_s": self.dram_throughput_gb_s,
            "compute_throughput_tflops": self.compute_throughput_tflops,
            "stall_memory_dependency": self.stall_memory_dependency,
            "stall_execution_dependency": self.stall_execution_dependency,
            "stall_not_selected": self.stall_not_selected,
            "raw_metrics": dict(self.raw_metrics),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HardwareCounters":
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        raw = d.get("raw_metrics", {})
        return cls(
            compute_utilization=float(d.get("compute_utilization", 0)),
            memory_bandwidth_utilization=float(d.get("memory_bandwidth_utilization", 0)),
            l2_cache_hit_rate=float(d.get("l2_cache_hit_rate", 0)),
            achieved_occupancy=float(d.get("achieved_occupancy", 0)),
            sm_efficiency=float(d.get("sm_efficiency", 0)),
            dram_throughput_gb_s=float(d.get("dram_throughput_gb_s", 0)),
            compute_throughput_tflops=float(d.get("compute_throughput_tflops", 0)),
            stall_memory_dependency=float(d.get("stall_memory_dependency", 0)),
            stall_execution_dependency=float(d.get("stall_execution_dependency", 0)),
            stall_not_selected=float(d.get("stall_not_selected", 0)),
            raw_metrics={k: float(v) for k, v in raw.items()},
        )


# ---------------------------------------------------------------------------
# NCU metric name mapping
# ---------------------------------------------------------------------------

# Maps NCU metric names -> HardwareCounters field names.
_NCU_METRIC_MAP: dict[str, str] = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_utilization",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_bandwidth_utilization",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_bandwidth_utilization",
    "l1tex__t_sector_hit_rate.pct": "l2_cache_hit_rate",
    "lts__t_sector_hit_rate.pct": "l2_cache_hit_rate",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy",
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed": "sm_efficiency",
    "dram__bytes.sum.per_second": "dram_throughput_gb_s",
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed": "compute_throughput_tflops",
    # Stall reasons
    "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct": "stall_memory_dependency",
    "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct": "stall_execution_dependency",
    "smsp__warps_issue_stalled_not_selected_per_warp_active.pct": "stall_not_selected",
}

# Metrics that need unit conversion (bytes/s -> GB/s)
_BYTES_TO_GB = {"dram__bytes.sum.per_second"}


# ---------------------------------------------------------------------------
# 2. NCU metric extraction
# ---------------------------------------------------------------------------

def extract_ncu_metrics(kernel_name: str, ncu_output: str) -> HardwareCounters:
    """Parse NCU CLI output (CSV format) into structured metrics.

    Supports the ``--csv`` output mode of ``ncu``.  Each row has columns:
    ``"Kernel Name","Metric Name","Metric Unit","Metric Value"`` (at minimum).
    We filter rows matching *kernel_name* (substring match) and map known
    metrics to :class:`HardwareCounters` fields.
    """
    counters = HardwareCounters()
    raw: dict[str, float] = {}

    reader = csv.reader(io.StringIO(ncu_output))
    header: list[str] | None = None

    for row in reader:
        if not row:
            continue
        # Skip comment lines that NCU prefixes with ==
        if row[0].startswith("=="):
            continue
        # Detect header row
        if header is None:
            lower = [c.strip().lower().replace('"', '') for c in row]
            if "metric name" in lower or "metric_name" in lower:
                header = lower
                continue
            # Also accept if first row looks like header with "Kernel Name"
            if any("kernel" in c.lower() for c in row):
                header = lower
                continue
            continue

        # Build a dict for this row
        row_dict: dict[str, str] = {}
        for i, col in enumerate(header):
            if i < len(row):
                row_dict[col] = row[i].strip().strip('"')

        # Filter by kernel name (substring match)
        row_kernel = row_dict.get("kernel name", row_dict.get("kernel_name", ""))
        if kernel_name and kernel_name not in row_kernel:
            continue

        metric_name = row_dict.get("metric name", row_dict.get("metric_name", ""))
        metric_value_str = row_dict.get("metric value", row_dict.get("metric_value", ""))

        if not metric_name or not metric_value_str:
            continue

        # Parse numeric value (strip commas, handle scientific notation)
        try:
            value = float(metric_value_str.replace(",", ""))
        except ValueError:
            continue

        raw[metric_name] = value

        # Map to structured field
        field_name = _NCU_METRIC_MAP.get(metric_name)
        if field_name:
            if metric_name in _BYTES_TO_GB:
                value = value / 1e9  # bytes/s -> GB/s
            setattr(counters, field_name, value)

    counters.raw_metrics = raw
    return counters


# ---------------------------------------------------------------------------
# 3. NCU-augmented benchmark script generator
# ---------------------------------------------------------------------------

def generate_ncu_benchmark_snippet(kernel_launch_code: str) -> str:
    """Wrap a kernel launch with NCU profiling.

    Returns a Python script snippet that:
    1. Runs the kernel once without profiling (warmup)
    2. Runs with NCU profiling via ``torch.cuda.nvtx`` range markers
    3. Parses NCU CSV output
    4. Returns :class:`HardwareCounters` alongside the timing result

    The generated script assumes ``ncu`` is on PATH and the caller has
    permission to profile.  It falls back gracefully if NCU is unavailable.
    """
    # Indent the user's kernel launch code for embedding
    indented = textwrap.indent(kernel_launch_code.rstrip(), "    ")

    snippet = textwrap.dedent("""\
        import subprocess, json, sys, os, time

        def _run_kernel():
        {kernel_code}

        # --- warmup (no profiling) ---
        _run_kernel()
        if hasattr(__import__('torch').cuda, 'synchronize'):
            __import__('torch').cuda.synchronize()

        # --- timed run (wall-clock) ---
        _t0 = time.perf_counter()
        _run_kernel()
        if hasattr(__import__('torch').cuda, 'synchronize'):
            __import__('torch').cuda.synchronize()
        _elapsed_ms = (time.perf_counter() - _t0) * 1000

        # --- NCU profiling (optional) ---
        _ncu_counters = None
        _ncu_bin = "ncu"
        if os.popen(f"which {{_ncu_bin}} 2>/dev/null").read().strip():
            _ncu_script = os.path.abspath(__file__) if '__file__' in dir() else 'ncu_target.py'
            _ncu_cmd = [
                _ncu_bin,
                "--csv",
                "--target-processes", "all",
                "--metrics",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
                "lts__t_sector_hit_rate.pct,"
                "sm__warps_active.avg.pct_of_peak_sustained_active,"
                "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,"
                "dram__bytes.sum.per_second,"
                "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct,"
                "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct,"
                "smsp__warps_issue_stalled_not_selected_per_warp_active.pct",
                sys.executable, "-c",
                "exec(open('" + _ncu_script + "').read())"
            ]
            try:
                _result = subprocess.run(_ncu_cmd, capture_output=True, text=True, timeout=120)
                _ncu_output = _result.stdout
            except Exception:
                _ncu_output = ""
        else:
            _ncu_output = ""

        print(json.dumps({{"elapsed_ms": _elapsed_ms, "ncu_output": _ncu_output}}))
    """).format(kernel_code=indented)

    return snippet


# ---------------------------------------------------------------------------
# 4. Feature integration with the ranking surrogate
# ---------------------------------------------------------------------------

def counters_to_features(counters: HardwareCounters) -> list[float]:
    """Convert hardware counters to an 8-element feature vector.

    Returns ``[compute_util, mem_bw_util, l2_hit, occupancy, sm_eff,
    stall_mem, stall_exec, stall_not_sel]`` -- 8 additional features that
    :class:`RankingSurrogate` can use to predict which configs will be fast
    on similar workloads.
    """
    return [
        counters.compute_utilization,
        counters.memory_bandwidth_utilization,
        counters.l2_cache_hit_rate,
        counters.achieved_occupancy,
        counters.sm_efficiency,
        counters.stall_memory_dependency,
        counters.stall_execution_dependency,
        counters.stall_not_selected,
    ]


# ---------------------------------------------------------------------------
# 5. NCU availability check
# ---------------------------------------------------------------------------

def is_ncu_available() -> bool:
    """Check whether the ``ncu`` CLI binary is available on this system.

    Returns ``False`` gracefully if the binary is not found or is not
    executable.  Does NOT require a GPU to be present.
    """
    return shutil.which("ncu") is not None


# ---------------------------------------------------------------------------
# 6. ConfigDatabase extension — record_result_with_counters
# ---------------------------------------------------------------------------

def record_result_with_counters(
    db: Any,
    *,
    shape: dict[str, int],
    hardware: str,
    config: dict[str, int],
    tflops: float,
    ms: float,
    correct: bool,
    run_id: str = "",
    operator: str = "matmul",
    bucket: str | None = None,
    config_id_str: str | None = None,
    counters: HardwareCounters | None = None,
) -> bool:
    """Record a benchmark result with optional hardware counters.

    Delegates to ``db.record_result()`` and then, if *counters* is provided,
    injects the serialised counters into the ``extra`` field of the most
    recently appended result dict.

    Parameters
    ----------
    db : ConfigDatabase
        The config database instance.
    counters : HardwareCounters | None
        Optional NCU-derived hardware counters.  When ``None`` the call
        behaves identically to ``db.record_result()``.

    Returns
    -------
    bool
        ``True`` if this result is a new best for the shape bucket.
    """
    is_new_best = db.record_result(
        shape=shape,
        hardware=hardware,
        config=config,
        tflops=tflops,
        ms=ms,
        correct=correct,
        run_id=run_id,
        operator=operator,
        bucket=bucket,
        config_id_str=config_id_str,
    )

    if counters is not None:
        # Find the record and attach counters to the last result entry.
        # We walk the records to find the one we just appended to.
        for record in db.records.values():
            if record.results and record.results[-1].get("run_id") == run_id:
                record.results[-1]["extra"] = {
                    "hardware_counters": counters.to_dict(),
                }
                break

    return is_new_best
