"""Adaptive-config wrappers around existing Triton kernel launchers.

These thin wrappers call ``get_selector`` to pick the best Triton config
for the current input shape, then delegate to the underlying kernel
launcher.  Drop-in replacements for fixed-config calls.
"""

from __future__ import annotations

from .adaptive_config import get_selector
from .triton_rmsnorm import rmsnorm
from .triton_qk_norm_rope import apply_qk_norm_rope


def adaptive_rmsnorm(x, w, eps=1e-6, affine_mode=0):
    """RMSNorm with adaptive config selection based on hidden_dim."""
    selector = get_selector("rmsnorm")
    config = selector.select(hidden_dim=x.shape[-1], n_rows=x.shape[0])
    return rmsnorm(x, w, config=config, eps=eps, affine_mode=affine_mode)


def adaptive_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, eps=1e-6):
    """QK-RMSNorm+RoPE with adaptive config selection based on head geometry."""
    selector = get_selector("qk_norm_rope")
    config = selector.select(head_dim=q.shape[-1], heads=q.shape[1])
    return apply_qk_norm_rope(
        q, k, cos, sin, q_scale, k_scale, config=config, eps=eps,
    )
