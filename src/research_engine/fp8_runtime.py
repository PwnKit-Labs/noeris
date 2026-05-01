"""Runtime layout selection helpers for FP8 matmul paths."""

from __future__ import annotations

from .fp8_layout_policy import choose_layout_from_policy


def resolve_fp8_layout(
    *,
    prefer: str,
    shape_name: str,
    expected_reuse_count: int,
    policy_payload: dict | None = None,
) -> str:
    """Resolve `kn` vs `nk` layout for FP8 matmul runtime dispatch.

    Modes:
    - ``prefer='kn'``: force KxN weight layout
    - ``prefer='nk'``: force NxK prepacked weight layout
    - ``prefer='auto'``: use artifact-driven policy if provided, else fallback
      heuristic (reuse >= 2 => nk, else kn)
    """
    mode = (prefer or "auto").lower()
    if mode not in {"auto", "kn", "nk"}:
        raise ValueError(f"Unknown FP8 layout preference: {prefer}")
    if mode in {"kn", "nk"}:
        return mode

    if policy_payload is not None:
        try:
            decision = choose_layout_from_policy(policy_payload, shape_name, expected_reuse_count)
            return decision.winner
        except (KeyError, ValueError):
            pass

    return "nk" if expected_reuse_count >= 2 else "kn"
