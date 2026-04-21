"""Runtime hook for KV cache quantize-on-write."""

from __future__ import annotations

from typing import Callable

BackendFn = Callable[[object], tuple[object, object]]


def resolve_kv_quant_backend(*, prefer: str, external_fn: BackendFn | None = None) -> str:
    prefer = (prefer or "auto").lower()
    if prefer not in {"auto", "external", "fused", "separated"}:
        raise ValueError(f"Unknown backend preference: {prefer}")
    if prefer == "external":
        return "external"
    if prefer == "fused":
        return "fused"
    if prefer == "separated":
        return "separated"
    # auto
    if external_fn is not None:
        return "external"
    return "fused"


def quantize_kv_write(
    kv_tensor,
    *,
    prefer: str = "auto",
    config: dict[str, int] | None = None,
    external_fn: BackendFn | None = None,
    allow_fallback: bool = True,
):
    """Quantize and return (q_int8, scale, backend_used)."""
    from .triton_kv_quant_write import kv_quantize_separated, kv_quantize_write_fused

    backend = resolve_kv_quant_backend(prefer=prefer, external_fn=external_fn)

    if backend == "external":
        if external_fn is None:
            raise RuntimeError("external backend requested but no external_fn provided")
        try:
            q, s = external_fn(kv_tensor)
            return q, s, "external"
        except Exception:  # noqa: BLE001
            if not allow_fallback:
                raise
            backend = "fused"

    if backend == "fused":
        try:
            q, s = kv_quantize_write_fused(kv_tensor, config=config)
            return q, s, "fused"
        except Exception:  # noqa: BLE001
            if not allow_fallback:
                raise
            q, s = kv_quantize_separated(kv_tensor)
            return q, s, "separated"

    q, s = kv_quantize_separated(kv_tensor)
    return q, s, "separated"


def quantize_kv_pair_write(
    k_tensor,
    v_tensor,
    *,
    prefer: str = "auto",
    config: dict[str, int] | None = None,
    external_fn: BackendFn | None = None,
    allow_fallback: bool = True,
):
    """Quantize K and V cache rows and return backend provenance.

    Returns: (k_q, k_s, v_q, v_s, backend_used)
    """
    k_q, k_s, kb = quantize_kv_write(
        k_tensor,
        prefer=prefer,
        config=config,
        external_fn=external_fn,
        allow_fallback=allow_fallback,
    )
    v_q, v_s, vb = quantize_kv_write(
        v_tensor,
        prefer=prefer,
        config=config,
        external_fn=external_fn,
        allow_fallback=allow_fallback,
    )
    backend = kb if kb == vb else f"mixed({kb},{vb})"
    return k_q, k_s, v_q, v_s, backend
