"""Runtime helper for FP8 B-layout selection.

Policy is derived from benchmark artifacts in docs/results and can be applied
at inference runtime based on expected weight reuse count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutDecision:
    shape_name: str
    reuse_count: int
    winner: str
    kn_total_ms: float
    nk_total_ms: float


def choose_layout_from_policy(policy_payload: dict, shape_name: str, expected_reuse_count: int) -> LayoutDecision:
    """Choose `kn` or `nk` from a policy payload for a shape/reuse pair.

    If exact reuse_count is missing in the artifact, this falls back to the
    nearest lower reuse_count available for the shape, then nearest higher.
    """
    if expected_reuse_count <= 0:
        raise ValueError("expected_reuse_count must be >= 1")

    policy_rows = policy_payload.get("policy", [])
    for row in policy_rows:
        if row.get("shape_name") != shape_name:
            continue
        decisions = list(row.get("decisions", []))
        if not decisions:
            break

        exact = next((d for d in decisions if int(d.get("reuse_count", 0)) == expected_reuse_count), None)
        if exact is None:
            lower = [d for d in decisions if int(d.get("reuse_count", 0)) <= expected_reuse_count]
            if lower:
                exact = max(lower, key=lambda d: int(d.get("reuse_count", 0)))
            else:
                exact = min(decisions, key=lambda d: int(d.get("reuse_count", 0)))

        return LayoutDecision(
            shape_name=shape_name,
            reuse_count=int(exact.get("reuse_count", 1)),
            winner=str(exact.get("winner", "kn")),
            kn_total_ms=float(exact.get("kn_total_ms", 0.0)),
            nk_total_ms=float(exact.get("nk_total_ms", 0.0)),
        )

    raise KeyError(f"shape_name not found in policy: {shape_name}")
