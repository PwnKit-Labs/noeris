"""KernelBench adapter functions that call real Noeris Triton operators.

Each adapter bridges the KernelBench upstream evaluation interface
(``model, init_inputs, fwd_inputs, cfg``) to the actual module-level
launchers in ``triton_rmsnorm``, ``triton_softmax``, etc.

The adapters handle:
  1) fp32 -> fp16 casting (Noeris kernels are fp16-native)
  2) Shape adaptation (upstream problems often use 4D tensors)
  3) Parameter extraction from the upstream Model instance
  4) fp16 -> fp32 casting on output (to match upstream reference dtype)

Usage from kernelbench_upstream.py's generated script:

    from research_engine.noeris_kb_adapters import NOERIS_ADAPTERS

Or in local (non-Modal) evaluation mode, import directly and call
the adapter with the upstream problem's Model and inputs.

Related: issue #41 — wire real Triton kernels into _NOERIS_ADAPTERS.
"""

from __future__ import annotations

from typing import Any, Callable

# Lazy imports: torch and the triton_* modules are only imported when
# an adapter is actually called. This keeps the module importable
# without a GPU and without triton installed (for offline tests).


def _noeris_rmsnorm(model: Any, init_inputs: list, fwd_inputs: list, cfg: dict) -> Any:
    """RMSNorm adapter for upstream problem 36_RMSNorm_.py.

    Upstream shape: (B, C, H, W) normalized along dim=1.
    Noeris expects: (rows, hidden_dim) normalized along last dim.
    """
    import torch
    from .triton_rmsnorm import rmsnorm

    (x,) = fwd_inputs
    B, C, H, W = x.shape
    eps = getattr(model, "eps", 1e-5)
    # Permute C to innermost, flatten to 2D
    rows = x.permute(0, 2, 3, 1).contiguous().view(-1, C).to(torch.float16)
    w = torch.ones((C,), device=x.device, dtype=torch.float16)
    out_rows = rmsnorm(rows, w, cfg, eps=eps, affine_mode=0)
    out = out_rows.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return out.to(torch.float32)


def _noeris_softmax(model: Any, init_inputs: list, fwd_inputs: list, cfg: dict) -> Any:
    """Softmax adapter for upstream problem 23_Softmax.py.

    Upstream shape: (n_rows, n_cols) softmax along dim=-1.
    Noeris expects the same layout.
    """
    import torch
    from .triton_softmax import softmax

    (x,) = fwd_inputs
    x_h = x.to(torch.float16).contiguous()
    out = softmax(x_h, cfg)
    return out.to(torch.float32)


def _noeris_layernorm(model: Any, init_inputs: list, fwd_inputs: list, cfg: dict) -> Any:
    """LayerNorm adapter for upstream problem 40_LayerNorm.py.

    Upstream shape: (B, C, H, W) with nn.LayerNorm over last three dims.
    Noeris expects: (rows, feature_dim) with weight and bias.
    """
    import torch
    from .triton_layernorm import layernorm

    (x,) = fwd_inputs
    B = x.shape[0]
    feat = x.numel() // B
    rows = x.reshape(B, feat).to(torch.float16).contiguous()
    # Pull learned weight/bias from the nn.LayerNorm module
    w = model.ln.weight.reshape(-1).to(torch.float16).contiguous()
    b = model.ln.bias.reshape(-1).to(torch.float16).contiguous()
    out_rows = layernorm(rows, w, b, cfg, eps=model.ln.eps)
    return out_rows.view(*x.shape).to(torch.float32)


def _noeris_cross_entropy(model: Any, init_inputs: list, fwd_inputs: list, cfg: dict) -> Any:
    """Cross-entropy adapter for upstream problem 95_CrossEntropyLoss.py.

    Upstream inputs: (logits, targets) where logits is (n_rows, n_cols) fp32.
    Noeris returns per-row loss; upstream expects scalar mean.
    """
    import torch
    from .triton_cross_entropy import cross_entropy

    logits, targets = fwd_inputs
    logits_h = logits.to(torch.float16).contiguous()
    targets_i64 = targets.to(torch.long).contiguous()
    per_row = cross_entropy(logits_h, targets_i64, cfg)
    return per_row.to(torch.float32).mean()


def _noeris_geglu(model: Any, init_inputs: list, fwd_inputs: list, cfg: dict) -> Any:
    """GeGLU adapter for upstream GELU problems (26, 88).

    Noeris has no standalone GELU, but the geglu kernel computes
    gate * GELU_tanh(up), so we pass gate=ones to recover GELU_tanh(up).
    This matches upstream #88 (tanh-approx GELU) but NOT #26 (exact GELU).
    """
    import torch
    from .triton_geglu import geglu

    (x,) = fwd_inputs
    up = x.to(torch.float16).contiguous()
    gate = torch.ones_like(up)
    out = geglu(gate, up, cfg)
    return out.to(torch.float32)


def _noeris_rotary(model: Any, init_inputs: list, fwd_inputs: list, cfg: dict) -> Any:
    """Rotary embedding adapter.

    Upstream inputs: (x, cos, sin) where x is (batch, seq, heads, head_dim).
    cos/sin are (seq, head_dim // 2).
    """
    import torch
    from .triton_rotary import apply_rotary_emb

    x, cos, sin = fwd_inputs
    x_h = x.to(torch.float16).contiguous()
    cos_h = cos.to(torch.float16).contiguous()
    sin_h = sin.to(torch.float16).contiguous()
    out = apply_rotary_emb(x_h, cos_h, sin_h, cfg)
    return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Adapter registry
#
# Maps operator name -> adapter callable with signature:
#     (model, init_inputs, fwd_inputs, cfg) -> output_tensor
#
# This dict is the canonical source of truth for which operators have
# real Triton kernel adapters (vs. torch-reference fallbacks).
# ---------------------------------------------------------------------------

NOERIS_ADAPTERS: dict[str, Callable] = {
    "rmsnorm":       _noeris_rmsnorm,
    "softmax":       _noeris_softmax,
    "layernorm":     _noeris_layernorm,
    "cross_entropy": _noeris_cross_entropy,
    "geglu":         _noeris_geglu,
    "rotary":        _noeris_rotary,
}


# Per-operator curated configs for KernelBench evaluation. These are
# the first-choice configs imported from each operator module.
def get_curated_config(operator: str) -> dict[str, int]:
    """Return the curated config for the given operator.

    Uses the first entry from each operator's CURATED_CONFIGS list,
    ensuring we always use the real module's config (not a stale copy).
    """
    if operator == "rmsnorm":
        from .triton_rmsnorm import RMSNORM_CURATED_CONFIGS
        return RMSNORM_CURATED_CONFIGS[0]
    elif operator == "softmax":
        from .triton_softmax import SOFTMAX_CURATED_CONFIGS
        return SOFTMAX_CURATED_CONFIGS[0]
    elif operator == "layernorm":
        from .triton_layernorm import LAYERNORM_CURATED_CONFIGS
        return LAYERNORM_CURATED_CONFIGS[0]
    elif operator == "cross_entropy":
        from .triton_cross_entropy import CROSS_ENTROPY_CURATED_CONFIGS
        return CROSS_ENTROPY_CURATED_CONFIGS[0]
    elif operator == "geglu":
        from .triton_geglu import GEGLU_CURATED_CONFIGS
        return GEGLU_CURATED_CONFIGS[0]
    elif operator == "rotary":
        from .triton_rotary import ROTARY_CURATED_CONFIGS
        return ROTARY_CURATED_CONFIGS[0]
    else:
        raise ValueError(f"No curated config for operator: {operator!r}")
