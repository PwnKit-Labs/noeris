"""MAP-Elites quality-diversity archive for Triton kernel config exploration.

Instead of keeping only the best config per shape bucket (as ConfigDatabase
does), this module maintains a 2D archive of diverse high-performing configs
indexed by two behavioral dimensions:

  Dimension 1 — Memory intensity: ratio of memory bytes to compute flops.
    Configs with high memory intensity prefer bandwidth-optimized tile sizes;
    compute-bound configs prefer compute-optimized tiles.

  Dimension 2 — Parallelism level: num_warps * effective block size.
    High-parallelism configs saturate more SMs but may have higher register
    pressure.

Each cell in the 2D archive holds the best-performing config that falls into
that behavioral bin. The archive enables quality-diversity search: when
selecting configs for benchmarking, we draw from under-explored bins rather
than only exploiting the best-known region.

Reference: Mouret & Clune (2015), "Illuminating search spaces by mapping
elites". KernelFoundry (March 2026) applies MAP-Elites to kernel code
structure; using it for Triton *config* search with hardware-aware behavioral
dimensions is unexplored.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

# ── Operator-type classification ──────────────────────────────────────────

_MEMORY_BOUND_OPS = frozenset({
    "rmsnorm", "layernorm", "softmax", "geglu", "cross_entropy",
})
_COMPUTE_BOUND_OPS = frozenset({
    "matmul", "grouped_gemm",
})
_MIXED_OPS = frozenset({
    "attention", "attention_decode", "qk_norm_rope",
})


def _op_category(operator: str) -> str:
    """Return 'memory', 'compute', or 'mixed' for an operator name."""
    op = operator.lower().strip()
    if op in _MEMORY_BOUND_OPS:
        return "memory"
    if op in _COMPUTE_BOUND_OPS:
        return "compute"
    if op in _MIXED_OPS:
        return "mixed"
    # Default: treat unknown operators as memory-bound (safer assumption for
    # element-wise / reduction kernels).
    return "memory"


# ── Behavioral dimension extraction ──────────────────────────────────────

def memory_intensity(config: dict, operator: str) -> float:
    """Compute a scalar memory-intensity proxy from config parameters.

    Higher values mean the config is more memory-bound.
    """
    category = _op_category(operator)

    if category == "compute":
        # matmul / grouped_gemm:
        # (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) / (BLOCK_M*BLOCK_N)
        bm = config.get("BLOCK_SIZE_M", config.get("BLOCK_M", 64))
        bn = config.get("BLOCK_SIZE_N", config.get("BLOCK_N", 64))
        bk = config.get("BLOCK_SIZE_K", config.get("BLOCK_K", 32))
        loads = bm * bk + bk * bn
        compute = bm * bn
        return loads / max(compute, 1)

    if category == "mixed":
        # attention: BLOCK_N / BLOCK_M — higher means loading more K/V
        bm = config.get("BLOCK_M", config.get("BLOCK_SIZE_M", 64))
        bn = config.get("BLOCK_N", config.get("BLOCK_SIZE_N", 64))
        return bn / max(bm, 1)

    # memory-bound ops: BLOCK_SIZE (larger = more memory per program)
    return float(config.get("BLOCK_SIZE", config.get("BLOCK_SIZE_M", 256)))


def parallelism_level(config: dict, operator: str) -> float:
    """Compute a scalar parallelism proxy from config parameters.

    Higher values mean the config occupies more SMs / has more threads.
    """
    num_warps = config.get("num_warps", 4)
    category = _op_category(operator)

    if category == "compute":
        # matmul / grouped_gemm: num_warps * GROUP_SIZE_M
        group = config.get("GROUP_SIZE_M", 1)
        return float(num_warps * group)

    # memory-bound & mixed: just num_warps
    return float(num_warps)


# ── Archive cell ─────────────────────────────────────────────────────────

@dataclass
class _ArchiveCell:
    """Contents of one bin in the MAP-Elites archive."""
    config: dict
    metric: float
    hardware_counters: dict = field(default_factory=dict)


# ── MAP-Elites archive ──────────────────────────────────────────────────

@dataclass
class MAPElitesArchive:
    """Quality-diversity archive for kernel configs.

    Maintains a 2D grid indexed by (memory_intensity_bin, parallelism_bin).
    Each cell holds the highest-throughput config that exhibits that behavior.

    Attributes:
        operator: Operator name (e.g. "matmul", "rmsnorm").
        shape_bucket: Shape bucket string from the operator spec.
        n_memory_bins: Number of bins along the memory-intensity axis.
        n_parallelism_bins: Number of bins along the parallelism axis.
        memory_range: (lo, hi) range for memory-intensity values. Configs
            outside this range are clamped.
        parallelism_range: (lo, hi) range for parallelism values.
    """

    operator: str
    shape_bucket: str
    n_memory_bins: int = 5
    n_parallelism_bins: int = 5
    memory_range: tuple[float, float] = (0.0, 1.0)
    parallelism_range: tuple[float, float] = (1.0, 64.0)

    def __post_init__(self) -> None:
        # archive: (mem_bin, par_bin) -> _ArchiveCell
        self.archive: dict[tuple[int, int], _ArchiveCell] = {}
        # Auto-detect ranges from operator type if using defaults
        self._init_ranges()

    def _init_ranges(self) -> None:
        """Set sensible default ranges based on operator category."""
        category = _op_category(self.operator)
        if category == "compute":
            # memory intensity for matmul: (BM*BK + BK*BN)/(BM*BN)
            # typical range ~0.25 to ~4.0
            self.memory_range = (0.25, 4.0)
            # parallelism: num_warps * GROUP_SIZE_M, typical 4..64
            self.parallelism_range = (4.0, 64.0)
        elif category == "mixed":
            # attention: BLOCK_N/BLOCK_M, typical 0.25..4.0
            self.memory_range = (0.25, 4.0)
            self.parallelism_range = (1.0, 16.0)
        else:
            # memory-bound: BLOCK_SIZE, typical 256..8192
            self.memory_range = (256.0, 8192.0)
            self.parallelism_range = (1.0, 16.0)

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def _bin_index(self, value: float, lo: float, hi: float, n_bins: int) -> int:
        """Map a continuous value to a discrete bin index in [0, n_bins)."""
        if hi <= lo:
            return 0
        # Use log-scale for ranges spanning more than one order of magnitude
        if hi / max(lo, 1e-9) > 10:
            value = math.log1p(value)
            lo = math.log1p(lo)
            hi = math.log1p(hi)
        clamped = max(lo, min(value, hi))
        frac = (clamped - lo) / (hi - lo)
        idx = int(frac * n_bins)
        return min(idx, n_bins - 1)

    def classify_config(self, config: dict) -> tuple[int, int]:
        """Map a config to its (memory_bin, parallelism_bin) behavioral bin."""
        mi = memory_intensity(config, self.operator)
        pl = parallelism_level(config, self.operator)
        mem_bin = self._bin_index(mi, *self.memory_range, self.n_memory_bins)
        par_bin = self._bin_index(pl, *self.parallelism_range, self.n_parallelism_bins)
        return (mem_bin, par_bin)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(
        self,
        config: dict,
        metric: float,
        hardware_counters: dict | None = None,
    ) -> bool:
        """Insert config if it is the best for its behavioral bin.

        Returns True if the config was inserted (new bin or better metric).
        """
        bin_key = self.classify_config(config)
        existing = self.archive.get(bin_key)
        if existing is not None and existing.metric >= metric:
            return False
        self.archive[bin_key] = _ArchiveCell(
            config=dict(config),
            metric=metric,
            hardware_counters=dict(hardware_counters or {}),
        )
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_diverse_candidates(self, top_k: int = 10) -> list[dict]:
        """Return configs from diverse bins, sorted by metric descending.

        Returns one config per occupied cell, up to top_k.
        """
        cells = sorted(
            self.archive.values(),
            key=lambda c: -c.metric,
        )
        return [cell.config for cell in cells[:top_k]]

    def best_config(self) -> dict | None:
        """Return the single highest-metric config across all bins."""
        if not self.archive:
            return None
        best = max(self.archive.values(), key=lambda c: c.metric)
        return best.config

    def coverage(self) -> float:
        """Fraction of bins occupied. Higher = more diverse exploration."""
        total_bins = self.n_memory_bins * self.n_parallelism_bins
        return len(self.archive) / total_bins

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the archive for storage in ConfigDatabase."""
        cells = {}
        for (mb, pb), cell in self.archive.items():
            key = f"{mb},{pb}"
            cells[key] = {
                "config": cell.config,
                "metric": cell.metric,
                "hardware_counters": cell.hardware_counters,
            }
        return {
            "operator": self.operator,
            "shape_bucket": self.shape_bucket,
            "n_memory_bins": self.n_memory_bins,
            "n_parallelism_bins": self.n_parallelism_bins,
            "memory_range": list(self.memory_range),
            "parallelism_range": list(self.parallelism_range),
            "cells": cells,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MAPElitesArchive:
        """Reconstruct an archive from its serialized form."""
        archive = cls(
            operator=data["operator"],
            shape_bucket=data["shape_bucket"],
            n_memory_bins=data.get("n_memory_bins", 5),
            n_parallelism_bins=data.get("n_parallelism_bins", 5),
        )
        archive.memory_range = tuple(data["memory_range"])  # type: ignore[assignment]
        archive.parallelism_range = tuple(data["parallelism_range"])  # type: ignore[assignment]
        for key, cell_data in data.get("cells", {}).items():
            mb, pb = (int(x) for x in key.split(","))
            archive.archive[(mb, pb)] = _ArchiveCell(
                config=cell_data["config"],
                metric=cell_data["metric"],
                hardware_counters=cell_data.get("hardware_counters", {}),
            )
        return archive

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for logging / CLI display."""
        metrics = [c.metric for c in self.archive.values()]
        return {
            "operator": self.operator,
            "shape_bucket": self.shape_bucket,
            "coverage": round(self.coverage(), 3),
            "occupied_bins": len(self.archive),
            "total_bins": self.n_memory_bins * self.n_parallelism_bins,
            "best_metric": round(max(metrics), 4) if metrics else None,
            "mean_metric": round(sum(metrics) / len(metrics), 4) if metrics else None,
        }


# ── MAP-Elites selector (integration with bandit) ───────────────────────

@dataclass
class MAPElitesSelector:
    """Config selector that uses MAP-Elites archives for diversity-aware search.

    Maintains one archive per (operator, shape_bucket, hardware) triple.
    When selecting configs for benchmarking, draws from under-explored bins
    to encourage full coverage of the behavioral config space.
    """

    n_memory_bins: int = 5
    n_parallelism_bins: int = 5
    # Fraction of selection budget allocated to exploration of empty bins
    exploration_fraction: float = 0.3

    def __post_init__(self) -> None:
        # (operator, bucket, hardware) -> MAPElitesArchive
        self._archives: dict[tuple[str, str, str], MAPElitesArchive] = {}

    def _get_or_create_archive(
        self,
        operator: str,
        shape_bucket: str,
        hardware: str,
    ) -> MAPElitesArchive:
        key = (operator, shape_bucket, hardware)
        if key not in self._archives:
            self._archives[key] = MAPElitesArchive(
                operator=operator,
                shape_bucket=shape_bucket,
                n_memory_bins=self.n_memory_bins,
                n_parallelism_bins=self.n_parallelism_bins,
            )
        return self._archives[key]

    def ingest_result(
        self,
        *,
        operator: str,
        shape_bucket: str,
        hardware: str,
        config: dict,
        metric: float,
        hardware_counters: dict | None = None,
    ) -> bool:
        """Record a benchmark result into the appropriate archive.

        Returns True if the config was inserted (new bin or better metric).
        """
        archive = self._get_or_create_archive(operator, shape_bucket, hardware)
        return archive.insert(config, metric, hardware_counters)

    def select_configs(
        self,
        *,
        operator: str,
        shape_bucket: str,
        hardware: str,
        candidate_pool: list[dict],
        max_configs: int = 8,
    ) -> list[dict]:
        """Select configs emphasizing diversity via under-explored bins.

        Strategy:
        1. Allocate ``exploration_fraction`` of the budget to candidates
           that fall into unoccupied or sparsely-occupied bins.
        2. Fill the remaining budget with the best-performing known configs
           from diverse bins (one per bin, sorted by metric).
        3. Any leftover slots are filled from the candidate pool in order.
        """
        archive = self._get_or_create_archive(operator, shape_bucket, hardware)
        selected: list[dict] = []
        seen_bins: set[tuple[int, int]] = set()

        n_explore = max(1, int(max_configs * self.exploration_fraction))
        n_exploit = max_configs - n_explore

        # ── Exploit: best known configs from diverse bins ──
        diverse = archive.get_diverse_candidates(top_k=n_exploit)
        for cfg in diverse:
            if len(selected) >= n_exploit:
                break
            b = archive.classify_config(cfg)
            if b not in seen_bins:
                seen_bins.add(b)
                selected.append(cfg)

        # ── Explore: candidates in unoccupied bins ──
        occupied = set(archive.archive.keys())
        explore_candidates = []
        for cfg in candidate_pool:
            b = archive.classify_config(cfg)
            if b not in occupied and b not in seen_bins:
                explore_candidates.append((b, cfg))

        for b, cfg in explore_candidates[:n_explore]:
            seen_bins.add(b)
            selected.append(cfg)

        # ── Fill remaining slots from candidate pool ──
        if len(selected) < max_configs:
            selected_set = {id(c) for c in selected}
            for cfg in candidate_pool:
                if id(cfg) not in selected_set:
                    selected.append(cfg)
                    selected_set.add(id(cfg))
                if len(selected) >= max_configs:
                    break

        return selected[:max_configs]

    def coverage_report(self) -> list[dict[str, Any]]:
        """Return a summary of all archives for logging."""
        return [
            archive.summary()
            for archive in self._archives.values()
        ]

    def to_dict(self) -> dict:
        """Serialize all archives."""
        return {
            "n_memory_bins": self.n_memory_bins,
            "n_parallelism_bins": self.n_parallelism_bins,
            "exploration_fraction": self.exploration_fraction,
            "archives": {
                f"{op}:{bucket}:{hw}": arch.to_dict()
                for (op, bucket, hw), arch in self._archives.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> MAPElitesSelector:
        """Reconstruct a selector from its serialized form."""
        selector = cls(
            n_memory_bins=data.get("n_memory_bins", 5),
            n_parallelism_bins=data.get("n_parallelism_bins", 5),
            exploration_fraction=data.get("exploration_fraction", 0.3),
        )
        for key, arch_data in data.get("archives", {}).items():
            parts = key.split(":")
            if len(parts) == 3:
                op, bucket, hw = parts
            else:
                continue
            selector._archives[(op, bucket, hw)] = MAPElitesArchive.from_dict(arch_data)
        return selector
