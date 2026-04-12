"""Tests for hardware_counters module."""
from __future__ import annotations

from tests import _pathfix  # noqa: F401 — add src/ to sys.path
import ast
import textwrap

import pytest

from research_engine.hardware_counters import (
    HardwareCounters,
    counters_to_features,
    extract_ncu_metrics,
    generate_ncu_benchmark_snippet,
    is_ncu_available,
    record_result_with_counters,
)


# ---------------------------------------------------------------------------
# Sample NCU CSV output (representative of real ncu --csv output)
# ---------------------------------------------------------------------------

SAMPLE_NCU_CSV = textwrap.dedent("""\
    "Kernel Name","Metric Name","Metric Unit","Metric Value"
    "my_triton_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","72.5"
    "my_triton_kernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","55.3"
    "my_triton_kernel","lts__t_sector_hit_rate.pct","%","88.1"
    "my_triton_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","63.7"
    "my_triton_kernel","smsp__cycles_active.avg.pct_of_peak_sustained_elapsed","%","70.2"
    "my_triton_kernel","dram__bytes.sum.per_second","byte/second","450000000000"
    "my_triton_kernel","smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct","%","12.4"
    "my_triton_kernel","smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct","%","8.1"
    "my_triton_kernel","smsp__warps_issue_stalled_not_selected_per_warp_active.pct","%","5.6"
""")


# ---------------------------------------------------------------------------
# 1. test_hardware_counters_dataclass
# ---------------------------------------------------------------------------

def test_hardware_counters_dataclass():
    """Fields exist and have correct types."""
    c = HardwareCounters(
        compute_utilization=72.5,
        memory_bandwidth_utilization=55.3,
        l2_cache_hit_rate=88.1,
        achieved_occupancy=63.7,
        sm_efficiency=70.2,
        dram_throughput_gb_s=450.0,
        compute_throughput_tflops=12.5,
        stall_memory_dependency=12.4,
        stall_execution_dependency=8.1,
        stall_not_selected=5.6,
        raw_metrics={"foo": 1.0},
    )
    assert isinstance(c.compute_utilization, float)
    assert isinstance(c.memory_bandwidth_utilization, float)
    assert isinstance(c.l2_cache_hit_rate, float)
    assert isinstance(c.achieved_occupancy, float)
    assert isinstance(c.sm_efficiency, float)
    assert isinstance(c.dram_throughput_gb_s, float)
    assert isinstance(c.compute_throughput_tflops, float)
    assert isinstance(c.stall_memory_dependency, float)
    assert isinstance(c.stall_execution_dependency, float)
    assert isinstance(c.stall_not_selected, float)
    assert isinstance(c.raw_metrics, dict)


# ---------------------------------------------------------------------------
# 2. test_extract_ncu_metrics_from_sample_output
# ---------------------------------------------------------------------------

def test_extract_ncu_metrics_from_sample_output():
    """Parse a hardcoded NCU CSV sample correctly."""
    c = extract_ncu_metrics("my_triton_kernel", SAMPLE_NCU_CSV)
    assert c.compute_utilization == pytest.approx(72.5)
    assert c.memory_bandwidth_utilization == pytest.approx(55.3)
    assert c.l2_cache_hit_rate == pytest.approx(88.1)
    assert c.achieved_occupancy == pytest.approx(63.7)
    assert c.sm_efficiency == pytest.approx(70.2)
    # dram__bytes.sum.per_second = 450e9 bytes/s -> 450 GB/s
    assert c.dram_throughput_gb_s == pytest.approx(450.0)
    assert c.stall_memory_dependency == pytest.approx(12.4)
    assert c.stall_execution_dependency == pytest.approx(8.1)
    assert c.stall_not_selected == pytest.approx(5.6)
    # raw_metrics should contain all 9 parsed metrics
    assert len(c.raw_metrics) == 9


# ---------------------------------------------------------------------------
# 3. test_counters_to_features_length
# ---------------------------------------------------------------------------

def test_counters_to_features_length():
    """Feature vector has exactly 8 elements."""
    c = HardwareCounters()
    features = counters_to_features(c)
    assert len(features) == 8


# ---------------------------------------------------------------------------
# 4. test_counters_to_features_range
# ---------------------------------------------------------------------------

def test_counters_to_features_range():
    """All features within reasonable ranges (0-100 for percentages)."""
    c = HardwareCounters(
        compute_utilization=72.5,
        memory_bandwidth_utilization=55.3,
        l2_cache_hit_rate=88.1,
        achieved_occupancy=63.7,
        sm_efficiency=70.2,
        stall_memory_dependency=12.4,
        stall_execution_dependency=8.1,
        stall_not_selected=5.6,
    )
    features = counters_to_features(c)
    for f in features:
        assert 0.0 <= f <= 100.0, f"Feature {f} out of [0, 100] range"


# ---------------------------------------------------------------------------
# 5. test_is_ncu_available_returns_bool
# ---------------------------------------------------------------------------

def test_is_ncu_available_returns_bool():
    """is_ncu_available returns a bool and does not crash."""
    result = is_ncu_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 6. test_ncu_benchmark_snippet_valid_python
# ---------------------------------------------------------------------------

def test_ncu_benchmark_snippet_valid_python():
    """Generated benchmark snippet is valid Python (compiles)."""
    snippet = generate_ncu_benchmark_snippet("y = x + 1")
    # Should parse without SyntaxError
    ast.parse(snippet)


# ---------------------------------------------------------------------------
# 7. test_record_result_with_counters_roundtrip
# ---------------------------------------------------------------------------

def test_record_result_with_counters_roundtrip():
    """Store and retrieve hardware counters from ConfigDatabase."""
    from research_engine.triton_kernels import ConfigDatabase
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ConfigDatabase(path=pathlib.Path(tmpdir) / "db.json")

        counters = HardwareCounters(
            compute_utilization=72.5,
            memory_bandwidth_utilization=55.3,
            l2_cache_hit_rate=88.1,
            achieved_occupancy=63.7,
            sm_efficiency=70.2,
            dram_throughput_gb_s=450.0,
            compute_throughput_tflops=12.5,
            stall_memory_dependency=12.4,
            stall_execution_dependency=8.1,
            stall_not_selected=5.6,
        )

        record_result_with_counters(
            db,
            shape={"M": 1024, "N": 1024, "K": 1024},
            hardware="A100",
            config={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            tflops=150.0,
            ms=1.5,
            correct=True,
            run_id="test-run-1",
            operator="matmul",
            counters=counters,
        )

        # Retrieve the stored result and check counters
        found = False
        for record in db.records.values():
            for result in record.results:
                if result.get("run_id") == "test-run-1":
                    assert "extra" in result
                    hw = result["extra"]["hardware_counters"]
                    restored = HardwareCounters.from_dict(hw)
                    assert restored.compute_utilization == pytest.approx(72.5)
                    assert restored.l2_cache_hit_rate == pytest.approx(88.1)
                    assert restored.dram_throughput_gb_s == pytest.approx(450.0)
                    found = True
                    break
        assert found, "Could not find the recorded result with counters"


# ---------------------------------------------------------------------------
# 8. test_hardware_counters_serialization_roundtrip
# ---------------------------------------------------------------------------

def test_hardware_counters_serialization_roundtrip():
    """to_dict / from_dict roundtrip preserves values."""
    original = HardwareCounters(
        compute_utilization=72.5,
        memory_bandwidth_utilization=55.3,
        l2_cache_hit_rate=88.1,
        achieved_occupancy=63.7,
        sm_efficiency=70.2,
        dram_throughput_gb_s=450.0,
        compute_throughput_tflops=12.5,
        stall_memory_dependency=12.4,
        stall_execution_dependency=8.1,
        stall_not_selected=5.6,
        raw_metrics={"custom_metric": 42.0},
    )
    d = original.to_dict()
    restored = HardwareCounters.from_dict(d)
    assert restored.compute_utilization == original.compute_utilization
    assert restored.dram_throughput_gb_s == original.dram_throughput_gb_s
    assert restored.raw_metrics == original.raw_metrics


# ---------------------------------------------------------------------------
# 9. test_extract_ncu_metrics_ignores_other_kernels
# ---------------------------------------------------------------------------

def test_extract_ncu_metrics_ignores_other_kernels():
    """Metrics for unrelated kernels are excluded."""
    csv_with_two_kernels = textwrap.dedent("""\
        "Kernel Name","Metric Name","Metric Unit","Metric Value"
        "target_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","80.0"
        "other_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","20.0"
    """)
    c = extract_ncu_metrics("target_kernel", csv_with_two_kernels)
    assert c.compute_utilization == pytest.approx(80.0)
