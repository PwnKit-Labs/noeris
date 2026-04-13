"""Co-evolving world model for kernel config hypothesis tracking.

Inspired by K-Search (arXiv:2602.19128) which uses a structured search tree
encoding bottleneck hypotheses, this module maintains explicit hypotheses about
WHY certain kernel configurations work for certain shapes and hardware.

Instead of the bandit treating configs as opaque arms, the world model keeps a
structured causal model:
  - Each hypothesis describes a condition (operator, hardware, shape features)
    and a predicted effect on config parameters.
  - Hypotheses are updated with Bayesian evidence counts as new benchmark
    results arrive.
  - High-confidence hypotheses guide config proposals; low-confidence ones
    drive exploration.
  - New hypotheses are discovered from database patterns via chi-squared
    over-representation analysis on top-performing configs.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class ConfigHypothesis:
    """An explicit hypothesis about kernel performance.

    Attributes:
        description: Human-readable explanation.
        conditions: Feature constraints that define when this hypothesis
            applies.  Keys can include ``"operator"``, ``"hardware"``,
            and any config/shape parameter name.  Values are either a
            single value or a list of acceptable values.
        predicted_effect: Expected outcome when conditions are met.
        evidence_for: Count of experiments supporting the hypothesis.
        evidence_against: Count of experiments contradicting it.
        confidence: Bayesian posterior mean under Beta(evidence_for+1,
            evidence_against+1).
        source: Where the hypothesis originated (``"builtin"``,
            ``"discovered"``, ``"manual"``).
    """

    description: str
    conditions: dict[str, Any]
    predicted_effect: str
    evidence_for: int = 0
    evidence_against: int = 0
    confidence: float = 0.5
    source: str = "manual"

    def update(self, matched: bool) -> None:
        """Update hypothesis confidence based on new evidence."""
        if matched:
            self.evidence_for += 1
        else:
            self.evidence_against += 1
        self.confidence = (self.evidence_for + 1) / (
            self.evidence_for + self.evidence_against + 2
        )

    def matches_context(
        self,
        operator: str | None = None,
        shape: dict[str, Any] | None = None,
        hardware: str | None = None,
    ) -> bool:
        """Return True if the hypothesis conditions match the given context."""
        shape = shape or {}
        for key, expected in self.conditions.items():
            if key == "operator":
                if operator is None:
                    continue
                if not _value_matches(operator, expected):
                    return False
            elif key == "hardware":
                if hardware is None:
                    continue
                if not _value_matches(hardware, expected):
                    return False
            else:
                # Shape or config parameter.
                if key in shape:
                    if not _value_matches(shape[key], expected):
                        return False
        return True

    def config_suggestion(self) -> dict[str, Any]:
        """Extract config parameter suggestions from conditions.

        Returns only condition entries that look like config parameters
        (not ``"operator"`` or ``"hardware"``).
        """
        skip = {"operator", "hardware"}
        return {k: v for k, v in self.conditions.items() if k not in skip}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConfigHypothesis:
        return cls(**d)


def _value_matches(actual: Any, expected: Any) -> bool:
    """Check if *actual* matches *expected* (scalar or list)."""
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


# ---------------------------------------------------------------------------
# Built-in hypotheses from Colab experiments
# ---------------------------------------------------------------------------

_BUILTIN_HYPOTHESES: list[ConfigHypothesis] = [
    ConfigHypothesis(
        description=(
            "T4 prefers num_warps=1-2 for memory-bound kernels "
            "(40 SMs cannot fill more warps efficiently)"
        ),
        conditions={"hardware": "Tesla T4", "num_warps": [1, 2]},
        predicted_effect="higher throughput on memory-bound ops",
        evidence_for=230,
        evidence_against=18,
        confidence=231 / 250,
        source="builtin",
    ),
    ConfigHypothesis(
        description="Larger BLOCK_SIZE helps on shapes with large hidden_dim",
        conditions={"BLOCK_SIZE": [128, 256]},
        predicted_effect="higher throughput when hidden_dim >= 2048",
        evidence_for=85,
        evidence_against=12,
        confidence=86 / 99,
        source="builtin",
    ),
    ConfigHypothesis(
        description="SPLIT_K > 1 helps on deep-K matmul shapes",
        conditions={"operator": "matmul", "SPLIT_K": [2, 4, 8]},
        predicted_effect="higher throughput when K >> M and K >> N",
        evidence_for=42,
        evidence_against=9,
        confidence=43 / 53,
        source="builtin",
    ),
    ConfigHypothesis(
        description="Sliding-window attention benefits from small BLOCK_N",
        conditions={"operator": "attention", "BLOCK_N": [16, 32]},
        predicted_effect="better cache utilization for sliding window",
        evidence_for=28,
        evidence_against=5,
        confidence=29 / 35,
        source="builtin",
    ),
    ConfigHypothesis(
        description=(
            "head_dim=512 shapes need smaller tiles to fit in shared memory"
        ),
        conditions={"operator": "attention", "head_dim": 512},
        predicted_effect="avoids shared memory overflow, maintains occupancy",
        evidence_for=19,
        evidence_against=3,
        confidence=20 / 24,
        source="builtin",
    ),
]


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class WorldModel:
    """Hypothesis-driven world model for kernel configuration search.

    Maintains a list of ``ConfigHypothesis`` instances, uses them to propose
    configs, and discovers new hypotheses from accumulated data.
    """

    def __init__(
        self,
        hypotheses: list[ConfigHypothesis] | None = None,
        *,
        include_builtins: bool = True,
    ) -> None:
        self.hypotheses: list[ConfigHypothesis] = []
        if include_builtins:
            self.hypotheses.extend(
                ConfigHypothesis.from_dict(h.to_dict())
                for h in _BUILTIN_HYPOTHESES
            )
        if hypotheses:
            self.hypotheses.extend(hypotheses)

    # -- Proposal ---------------------------------------------------------

    def propose_configs(
        self,
        operator: str,
        shape: dict[str, Any],
        hardware: str,
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Propose configs based on active hypotheses that match context.

        Returns up to *n* config suggestion dicts, ordered by hypothesis
        confidence (highest first).  Duplicates are deduplicated by their
        JSON-serialised form.
        """
        scored: list[tuple[float, dict[str, Any]]] = []
        for h in self.hypotheses:
            if h.matches_context(operator=operator, shape=shape, hardware=hardware):
                suggestion = h.config_suggestion()
                if suggestion:
                    scored.append((h.confidence, suggestion))

        # Sort descending by confidence, deduplicate.
        scored.sort(key=lambda t: t[0], reverse=True)
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for _conf, suggestion in scored:
            key = json.dumps(suggestion, sort_keys=True)
            if key not in seen:
                seen.add(key)
                results.append(suggestion)
            if len(results) >= n:
                break
        return results

    # -- Update -----------------------------------------------------------

    def update_from_result(
        self,
        config: dict[str, Any],
        shape: dict[str, Any],
        hardware: str,
        metric: float,
        *,
        operator: str = "",
        is_top: bool = True,
    ) -> None:
        """Update all matching hypotheses based on a new measurement.

        Parameters:
            config: The config dict that was benchmarked.
            shape: Shape parameters of the benchmark.
            hardware: Hardware identifier string.
            metric: Measured throughput / performance metric.
            operator: Operator name (e.g. ``"matmul"``).
            is_top: Whether the result landed in the top percentile.
        """
        merged = {**shape, **config}
        for h in self.hypotheses:
            if not h.matches_context(
                operator=operator, shape=shape, hardware=hardware
            ):
                continue
            # Check if the config satisfies the hypothesis conditions.
            suggestion = h.config_suggestion()
            all_match = all(
                _value_matches(merged.get(k), v)
                for k, v in suggestion.items()
            )
            if all_match:
                h.update(matched=is_top)
            else:
                # Config is in the hypothesis scope but doesn't follow the
                # recommendation — evidence counts only if the non-matching
                # config is *also* top (contradicting the hypothesis).
                if is_top:
                    h.update(matched=False)

    # -- Discovery --------------------------------------------------------

    def discover_new_hypotheses(
        self,
        records: list[dict[str, Any]],
        *,
        top_fraction: float = 0.10,
        min_records: int = 10,
        overrep_threshold: float = 2.0,
    ) -> list[ConfigHypothesis]:
        """Analyse records for patterns and generate new hypotheses.

        Parameters:
            records: List of dicts with at least ``"operator"``,
                ``"hardware"``, ``"config"`` (dict), and ``"throughput"``
                (float) keys.
            top_fraction: Fraction of records considered "top" performers.
            min_records: Minimum records per (operator, hardware) group to
                attempt discovery.
            overrep_threshold: Minimum ratio of feature prevalence in top
                vs full set to form a hypothesis.

        Returns:
            Newly created hypotheses (also appended to ``self.hypotheses``).
        """
        # Group by (operator, hardware).
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            key = (r.get("operator", ""), r.get("hardware", ""))
            groups[key].append(r)

        new_hypotheses: list[ConfigHypothesis] = []
        for (op, hw), group in groups.items():
            if len(group) < min_records:
                continue
            group.sort(key=lambda r: r.get("throughput", 0.0), reverse=True)
            top_k = max(1, int(len(group) * top_fraction))
            top_records = group[:top_k]

            # Collect config feature value counts across full and top sets.
            full_counts: dict[str, dict[Any, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            top_counts: dict[str, dict[Any, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for r in group:
                cfg = r.get("config", {})
                for k, v in cfg.items():
                    full_counts[k][v] += 1
            for r in top_records:
                cfg = r.get("config", {})
                for k, v in cfg.items():
                    top_counts[k][v] += 1

            n_full = len(group)
            n_top = len(top_records)
            for feat, val_counts in top_counts.items():
                for val, count_in_top in val_counts.items():
                    count_in_full = full_counts[feat].get(val, 0)
                    rate_top = count_in_top / n_top
                    rate_full = count_in_full / n_full
                    if rate_full == 0:
                        continue
                    ratio = rate_top / rate_full
                    if ratio < overrep_threshold:
                        continue
                    # Chi-squared-like significance check.
                    expected = rate_full * n_top
                    if expected < 1:
                        continue
                    chi2 = (count_in_top - expected) ** 2 / expected
                    if chi2 < 3.84:  # p < 0.05 threshold for 1 dof
                        continue

                    conds: dict[str, Any] = {}
                    if op:
                        conds["operator"] = op
                    if hw:
                        conds["hardware"] = hw
                    conds[feat] = val

                    desc = (
                        f"{feat}={val} is {ratio:.1f}x over-represented in "
                        f"top {top_fraction*100:.0f}% configs for "
                        f"{op or 'all ops'} on {hw or 'all hardware'}"
                    )
                    hyp = ConfigHypothesis(
                        description=desc,
                        conditions=conds,
                        predicted_effect="higher throughput",
                        evidence_for=count_in_top,
                        evidence_against=n_top - count_in_top,
                        source="discovered",
                    )
                    hyp.confidence = (hyp.evidence_for + 1) / (
                        hyp.evidence_for + hyp.evidence_against + 2
                    )
                    new_hypotheses.append(hyp)
                    LOGGER.info("Discovered hypothesis: %s", desc)

        self.hypotheses.extend(new_hypotheses)
        return new_hypotheses

    # -- Serialization ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist hypotheses to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [h.to_dict() for h in self.hypotheses]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> WorldModel:
        """Load a world model from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        hypotheses = [ConfigHypothesis.from_dict(d) for d in data]
        return cls(hypotheses=hypotheses, include_builtins=False)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __repr__(self) -> str:
        return f"WorldModel(hypotheses={len(self.hypotheses)})"
