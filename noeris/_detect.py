"""Architecture detection for HuggingFace models.

Inspects model.config to determine which kernels can be applied and
locates the modules to patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchInfo:
    """Describes a model architecture for patching."""
    model_type: str
    has_qk_norm: bool = False
    affine_mode: int = 0  # 0 = standard, 1 = Gemma (1+w)
    activation: str = "swiglu"  # "geglu" or "swiglu"
    # Module class names to look for
    norm_classes: list[str] = field(default_factory=list)
    attn_classes: list[str] = field(default_factory=list)
    mlp_classes: list[str] = field(default_factory=list)


# Registry: model_type -> ArchInfo
_ARCH_REGISTRY: dict[str, ArchInfo] = {
    "gemma": ArchInfo(
        model_type="gemma",
        has_qk_norm=True,
        affine_mode=1,
        activation="geglu",
        norm_classes=["GemmaRMSNorm"],
        attn_classes=["GemmaAttention", "GemmaSdpaAttention", "GemmaFlashAttention2"],
        mlp_classes=["GemmaMLP"],
    ),
    "gemma2": ArchInfo(
        model_type="gemma2",
        has_qk_norm=True,
        affine_mode=1,
        activation="geglu",
        norm_classes=["Gemma2RMSNorm"],
        attn_classes=["Gemma2Attention", "Gemma2SdpaAttention", "Gemma2FlashAttention2"],
        mlp_classes=["Gemma2MLP"],
    ),
    "gemma3": ArchInfo(
        model_type="gemma3",
        has_qk_norm=True,
        affine_mode=1,
        activation="geglu",
        norm_classes=["Gemma3RMSNorm"],
        attn_classes=["Gemma3Attention", "Gemma3SdpaAttention"],
        mlp_classes=["Gemma3MLP"],
    ),
    "llama": ArchInfo(
        model_type="llama",
        has_qk_norm=False,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["LlamaRMSNorm"],
        attn_classes=["LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2"],
        mlp_classes=["LlamaMLP"],
    ),
    "mistral": ArchInfo(
        model_type="mistral",
        has_qk_norm=False,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["MistralRMSNorm"],
        attn_classes=["MistralAttention", "MistralSdpaAttention", "MistralFlashAttention2"],
        mlp_classes=["MistralMLP"],
    ),
    "qwen2": ArchInfo(
        model_type="qwen2",
        has_qk_norm=True,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["Qwen2RMSNorm"],
        attn_classes=["Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2"],
        mlp_classes=["Qwen2MLP"],
    ),
    "qwen3": ArchInfo(
        model_type="qwen3",
        has_qk_norm=True,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["Qwen3RMSNorm"],
        attn_classes=["Qwen3Attention", "Qwen3SdpaAttention"],
        mlp_classes=["Qwen3MLP"],
    ),
    "phi3": ArchInfo(
        model_type="phi3",
        has_qk_norm=False,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["Phi3RMSNorm"],
        attn_classes=["Phi3Attention", "Phi3SdpaAttention", "Phi3FlashAttention2"],
        mlp_classes=["Phi3MLP"],
    ),
    "phi": ArchInfo(
        model_type="phi",
        has_qk_norm=False,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["PhiRMSNorm"],
        attn_classes=["PhiAttention", "PhiSdpaAttention"],
        mlp_classes=["PhiMLP"],
    ),
    "olmo2": ArchInfo(
        model_type="olmo2",
        has_qk_norm=True,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["OLMo2RMSNorm", "OlmoRMSNorm"],
        attn_classes=["OLMo2Attention", "OlmoAttention"],
        mlp_classes=["OLMo2MLP", "OlmoMLP"],
    ),
    "falcon": ArchInfo(
        model_type="falcon",
        has_qk_norm=False,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["FalconRMSNorm"],
        attn_classes=["FalconAttention"],
        mlp_classes=["FalconMLP"],
    ),
    "internlm2": ArchInfo(
        model_type="internlm2",
        has_qk_norm=False,
        affine_mode=0,
        activation="swiglu",
        norm_classes=["InternLM2RMSNorm"],
        attn_classes=["InternLM2Attention"],
        mlp_classes=["InternLM2MLP"],
    ),
}


def detect_architecture(model) -> ArchInfo | None:
    """Detect model architecture from model.config.model_type."""
    config = getattr(model, "config", None)
    if config is None:
        return None

    model_type = getattr(config, "model_type", "")

    # Direct match
    if model_type in _ARCH_REGISTRY:
        return _ARCH_REGISTRY[model_type]

    # Fuzzy match: try prefix
    for key, arch in _ARCH_REGISTRY.items():
        if model_type.startswith(key):
            return arch

    return None
