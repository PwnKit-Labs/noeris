"""Drop-in nn.Module replacements for HuggingFace modules.

These modules are signature-compatible with the originals and use
Triton kernels via autograd.Function wrappers for training support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._autograd import TritonRMSNorm, TritonGeGLU, TritonSwiGLU


class NoerisRMSNorm(nn.Module):
    """Drop-in replacement for LlamaRMSNorm / GemmaRMSNorm / etc."""

    def __init__(self, original_module: nn.Module, affine_mode: int = 0):
        super().__init__()
        # Share the weight parameter (not a copy)
        self.weight = original_module.weight
        self.variance_epsilon = getattr(
            original_module, "variance_epsilon",
            getattr(original_module, "eps", 1e-6),
        )
        self.affine_mode = affine_mode
        self._original_class_name = type(original_module).__name__

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(-1, input_shape[-1])
        if not hidden_states_2d.is_contiguous():
            hidden_states_2d = hidden_states_2d.contiguous()
        output = TritonRMSNorm.apply(
            hidden_states_2d, self.weight,
            self.variance_epsilon, self.affine_mode,
        )
        return output.reshape(input_shape)

    def extra_repr(self) -> str:
        return f"noeris_patched={self._original_class_name}, affine_mode={self.affine_mode}"


class NoerisGatedActivation(nn.Module):
    """Wraps the gated activation inside an MLP block.

    Replaces the act_fn + gate/up multiplication with a fused Triton kernel.
    This is injected into the MLP's forward, not as a standalone module replacement.
    """

    def __init__(self, activation: str = "swiglu"):
        super().__init__()
        self.activation = activation

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_2d = gate.reshape(-1, gate.shape[-1])
        up_2d = up.reshape(-1, up.shape[-1])
        if not gate_2d.is_contiguous():
            gate_2d = gate_2d.contiguous()
        if not up_2d.is_contiguous():
            up_2d = up_2d.contiguous()

        if self.activation == "geglu":
            out = TritonGeGLU.apply(gate_2d, up_2d)
        else:
            out = TritonSwiGLU.apply(gate_2d, up_2d)

        return out.reshape(gate.shape)
