"""HuggingFace KernelBench dataset loader.

KernelBench (Ouyang et al., 2025, arXiv:2502.10517) is the standard
benchmark for LLM-driven GPU kernel generation. It has 250 problems
across 4 difficulty levels, hosted on HuggingFace under
``ScalingIntelligence/KernelBench``.

This loader:
  1. Fetches the dataset
  2. Parses each problem's PyTorch reference code
  3. Maps problems to our supported operators via heuristics
  4. Reports coverage (how many problems we can benchmark)
  5. Produces a list of ProblemSpec entries compatible with our
     kernelbench.evaluate_kernelbench() pipeline

Unsupported problems are reported but not silently skipped — reviewers
care about honest coverage numbers.

Dependencies: ``datasets`` (pip install datasets).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Heuristic operator matchers — regex patterns that indicate which
# operator a KernelBench problem uses.
OPERATOR_PATTERNS = {
    "matmul": [
        r"torch\.matmul",
        r"torch\.nn\.functional\.linear",
        r"torch\.nn\.Linear",
        r"@\s*self\.weight",
        r"x\s*@\s*",
    ],
    "softmax": [
        r"torch\.softmax",
        r"F\.softmax",
        r"torch\.nn\.functional\.softmax",
        r"torch\.nn\.Softmax",
    ],
    "layernorm": [
        r"torch\.nn\.functional\.layer_norm",
        r"F\.layer_norm",
        r"torch\.nn\.LayerNorm",
    ],
    "rmsnorm": [
        r"torch\.nn\.functional\.rms_norm",
        r"RMSNorm",
        r"rsqrt\s*\(.*pow\(2\).*mean",
    ],
    "cross_entropy": [
        r"torch\.nn\.functional\.cross_entropy",
        r"F\.cross_entropy",
        r"torch\.nn\.CrossEntropyLoss",
    ],
    "attention": [
        r"torch\.nn\.functional\.scaled_dot_product_attention",
        r"F\.scaled_dot_product_attention",
        r"torch\.nn\.MultiheadAttention",
        r"flash_attn",
    ],
}


@dataclass(slots=True)
class KernelBenchProblem:
    """A single KernelBench problem with reference code and metadata."""

    problem_id: str
    level: int
    name: str
    code: str
    matched_operator: str | None = None
    reason: str = ""

    def is_supported(self) -> bool:
        return self.matched_operator is not None


@dataclass(slots=True)
class CoverageReport:
    total: int = 0
    supported: int = 0
    by_operator: dict[str, int] = field(default_factory=dict)
    by_level: dict[int, int] = field(default_factory=dict)
    unsupported_samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "supported": self.supported,
            "coverage_pct": round(self.supported / max(1, self.total) * 100, 2),
            "by_operator": dict(self.by_operator),
            "by_level": {str(k): v for k, v in self.by_level.items()},
            "unsupported_samples": self.unsupported_samples[:10],
        }


def match_operator(code: str) -> str | None:
    """Heuristic: which operator does this problem's reference code use?

    Returns the most specific matching operator, or None if no match.
    """
    if not code:
        return None

    # Try each operator; pick the first that matches (order matters — more
    # specific operators should come first).
    priority_order = ["attention", "rmsnorm", "layernorm", "softmax", "cross_entropy", "matmul"]
    for operator in priority_order:
        patterns = OPERATOR_PATTERNS.get(operator, [])
        for pattern in patterns:
            if re.search(pattern, code):
                return operator
    return None


def load_kernelbench_problems(
    *,
    split: str = "level_1",
    limit: int | None = None,
) -> list[KernelBenchProblem]:
    """Load KernelBench problems from HuggingFace.

    Args:
        split: Which level split to load (level_1 through level_4).
        limit: Optional cap on number of problems returned.

    Returns:
        List of KernelBenchProblem, annotated with matched_operator.

    Raises:
        ImportError: if ``datasets`` package is not installed.
        RuntimeError: if the dataset fails to load.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace 'datasets' package is required. "
            "Install with: pip install datasets"
        ) from exc

    try:
        ds = load_dataset("ScalingIntelligence/KernelBench", split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load KernelBench split {split}: {exc}") from exc

    problems: list[KernelBenchProblem] = []
    level_num = int(split.split("_")[1]) if "_" in split else 1
    for idx, row in enumerate(ds):
        if limit is not None and len(problems) >= limit:
            break
        # KernelBench rows have fields like: name, code, problem_id, ...
        code = row.get("code", "") or row.get("source", "")
        name = row.get("name", f"kb_{split}_{idx}")
        problem_id = row.get("problem_id") or f"{split}/{idx}"
        op = match_operator(code)
        reason = "matched by regex" if op else "no regex match"
        problems.append(KernelBenchProblem(
            problem_id=str(problem_id),
            level=level_num,
            name=str(name),
            code=str(code),
            matched_operator=op,
            reason=reason,
        ))

    return problems


def compute_coverage(problems: list[KernelBenchProblem]) -> CoverageReport:
    """Summarize how many KernelBench problems we can benchmark."""
    report = CoverageReport()
    for p in problems:
        report.total += 1
        report.by_level[p.level] = report.by_level.get(p.level, 0) + 1
        if p.is_supported():
            report.supported += 1
            report.by_operator[p.matched_operator] = (
                report.by_operator.get(p.matched_operator, 0) + 1
            )
        elif len(report.unsupported_samples) < 20:
            report.unsupported_samples.append(f"{p.problem_id}: {p.name}")
    return report


def problems_to_benchmark_shapes(
    problems: list[KernelBenchProblem],
) -> dict[str, list[dict[str, Any]]]:
    """Convert supported KernelBench problems into benchmark shape dicts.

    This is a BEST-EFFORT mapping. Real KernelBench problems have code
    that defines input shapes; we use regex to extract M/N/K or similar
    dimensions where possible. Problems with unparseable shapes get a
    default shape for their operator.
    """
    by_operator: dict[str, list[dict]] = {}
    for p in problems:
        if not p.is_supported():
            continue
        shape = _extract_shape_from_code(p.code, p.matched_operator)
        if shape is None:
            continue
        entry = {
            "id": f"kb_{p.problem_id}",
            "level": p.level,
            "name": p.name,
            **shape,
        }
        by_operator.setdefault(p.matched_operator, []).append(entry)
    return by_operator


def _extract_shape_from_code(code: str, operator: str) -> dict | None:
    """Best-effort regex extraction of input tensor shapes from problem code.

    KernelBench problems typically define something like
    ``get_inputs()`` that returns tensors with fixed shapes. We scan for
    common shape patterns.
    """
    # Look for torch.randn((a, b, c), ...) or torch.randn(a, b, c, ...)
    patterns = [
        r"torch\.randn\s*\(\s*\(([^)]+)\)",
        r"torch\.randn\s*\(\s*([0-9, ]+)",
        r"torch\.empty\s*\(\s*\(([^)]+)\)",
        r"shape\s*=\s*\(([^)]+)\)",
    ]
    found_shapes: list[list[int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, code):
            dims = match.group(1)
            try:
                nums = [int(n.strip()) for n in dims.split(",") if n.strip().isdigit()]
                if 1 <= len(nums) <= 5 and all(n > 0 for n in nums):
                    found_shapes.append(nums)
            except (ValueError, AttributeError):
                continue
        if found_shapes:
            break

    if not found_shapes:
        return None

    # Map to operator-specific shape dict
    if operator == "matmul":
        # Expect two 2D tensors for matmul
        two_d = [s for s in found_shapes if len(s) == 2]
        if len(two_d) >= 2:
            a = two_d[0]
            b = two_d[1]
            if a[1] == b[0]:
                return {"M": a[0], "N": b[1], "K": a[1]}
        # Fallback to default
        return {"M": 1024, "N": 1024, "K": 1024}
    elif operator in ("rmsnorm", "layernorm"):
        two_d = next((s for s in found_shapes if len(s) == 2), None)
        if two_d:
            return {"n_rows": two_d[0], "hidden_dim": two_d[1]}
        return {"n_rows": 4096, "hidden_dim": 4096}
    elif operator in ("softmax", "cross_entropy"):
        two_d = next((s for s in found_shapes if len(s) == 2), None)
        if two_d:
            return {"n_rows": two_d[0], "n_cols": two_d[1]}
        return {"n_rows": 2048, "n_cols": 1024}
    elif operator == "attention":
        four_d = next((s for s in found_shapes if len(s) == 4), None)
        if four_d:
            return {
                "batch": four_d[0],
                "heads": four_d[1],
                "seq_len": four_d[2],
                "head_dim": four_d[3],
                "is_causal": False,
            }
        return {"batch": 1, "heads": 16, "seq_len": 2048, "head_dim": 64, "is_causal": False}
    return None


def fetch_and_report_coverage(
    *,
    levels: list[int] | None = None,
    limit_per_level: int | None = None,
) -> dict:
    """End-to-end coverage probe: fetch, match, report.

    Returns a dict suitable for printing as JSON.
    """
    levels = levels or [1, 2, 3, 4]
    all_problems: list[KernelBenchProblem] = []
    for level in levels:
        try:
            split = f"level_{level}"
            problems = load_kernelbench_problems(split=split, limit=limit_per_level)
            all_problems.extend(problems)
        except Exception as exc:
            return {"error": f"Failed to load level {level}: {exc}"}

    coverage = compute_coverage(all_problems)
    by_op = problems_to_benchmark_shapes(all_problems)
    return {
        "coverage": coverage.to_dict(),
        "mapped_problems_per_operator": {
            op: len(probs) for op, probs in by_op.items()
        },
    }
