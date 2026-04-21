"""Runtime hook for speculative decode verify+accept.

Selection order:
- ``prefer='flashinfer'``: use FlashInfer callback only
- ``prefer='fused'``: use Noeris Triton fused kernel only
- ``prefer='separated'``: use PyTorch separated reference path
- ``prefer='auto'`` (default): FlashInfer callback if provided, else fused,
  else separated fallback.

The runtime API returns both outputs and the backend used so callers can record
operational provenance in benchmarks.
"""

from __future__ import annotations

from typing import Callable

BackendFn = Callable[[object, object], tuple[object, object]]


def resolve_verify_accept_backend(*, prefer: str, flashinfer_fn: BackendFn | None = None) -> str:
    prefer = (prefer or "auto").lower()
    if prefer not in {"auto", "flashinfer", "fused", "separated"}:
        raise ValueError(f"Unknown backend preference: {prefer}")
    if prefer == "flashinfer":
        return "flashinfer"
    if prefer == "fused":
        return "fused"
    if prefer == "separated":
        return "separated"
    # auto
    if flashinfer_fn is not None:
        return "flashinfer"
    return "fused"


def verify_accept_tokens(
    target_tokens,
    draft_tokens,
    *,
    prefer: str = "auto",
    config: dict[str, int] | None = None,
    flashinfer_fn: BackendFn | None = None,
    allow_fallback: bool = True,
):
    """Run verify+accept from token IDs.

    Returns: ``(accept_len, prefix_mask, backend_used)``.
    """
    from .triton_spec_decode_verify_accept import verify_accept_fused, verify_accept_reference

    backend = resolve_verify_accept_backend(prefer=prefer, flashinfer_fn=flashinfer_fn)

    if backend == "flashinfer":
        if flashinfer_fn is None:
            raise RuntimeError("flashinfer backend requested but no flashinfer_fn provided")
        try:
            accept_len, prefix_mask = flashinfer_fn(target_tokens, draft_tokens)
            return accept_len, prefix_mask, "flashinfer"
        except Exception:  # noqa: BLE001
            if not allow_fallback:
                raise
            # fall back to fused path first
            backend = "fused"

    if backend == "fused":
        try:
            accept_len, prefix_mask = verify_accept_fused(target_tokens, draft_tokens, config=config)
            return accept_len, prefix_mask, "fused"
        except Exception:  # noqa: BLE001
            if not allow_fallback:
                raise
            accept_len, prefix_mask = verify_accept_reference(target_tokens, draft_tokens)
            return accept_len, prefix_mask, "separated"

    accept_len, prefix_mask = verify_accept_reference(target_tokens, draft_tokens)
    return accept_len, prefix_mask, "separated"


def verify_accept_from_logits(
    target_logits,
    draft_tokens,
    *,
    prefer: str = "auto",
    config: dict[str, int] | None = None,
    flashinfer_fn: BackendFn | None = None,
    allow_fallback: bool = True,
):
    """Run verify+accept from target logits and draft tokens.

    Returns: ``(accept_len, prefix_mask, backend_used)``.
    """
    target_tokens = target_logits.argmax(dim=-1)
    return verify_accept_tokens(
        target_tokens,
        draft_tokens,
        prefer=prefer,
        config=config,
        flashinfer_fn=flashinfer_fn,
        allow_fallback=allow_fallback,
    )
