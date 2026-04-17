"""Hardware+shape routing policy for Gemma4 layer benchmark kernels."""

from __future__ import annotations


RMSNORM_CONFIG = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}
QK_NORM_ROPE_CONFIG = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1}
GEGLU_CONFIG = {"BLOCK_SIZE": 128, "num_warps": 16, "num_stages": 1}

FUSED_NORM_LINEAR_CONFIG_PREFILL = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_warps": 8,
    "num_stages": 3,
}
FUSED_NORM_LINEAR_CONFIG_31B = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_warps": 8,
    "num_stages": 3,
}

ATTN_CONFIG_LOCAL = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
ATTN_CONFIG_GLOBAL = {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2}


FUSED_NORM_LINEAR_CONFIG_POLICY = {
    "default": {
        "31b_prefill": FUSED_NORM_LINEAR_CONFIG_31B,
        "prefill": FUSED_NORM_LINEAR_CONFIG_PREFILL,
        "decode": None,
    },
    "a100": {
        "31b_prefill": FUSED_NORM_LINEAR_CONFIG_31B,
        "prefill": FUSED_NORM_LINEAR_CONFIG_PREFILL,
        "decode": None,
    },
    "h100": {
        "31b_prefill": FUSED_NORM_LINEAR_CONFIG_31B,
        "prefill": FUSED_NORM_LINEAR_CONFIG_PREFILL,
        "decode": None,
    },
}

ATTN_CONFIG_POLICY = {
    "default": {
        "31b_global": ATTN_CONFIG_GLOBAL,
        "local": ATTN_CONFIG_LOCAL,
    },
    "a100": {
        "31b_global": ATTN_CONFIG_GLOBAL,
        "local": ATTN_CONFIG_LOCAL,
    },
    "h100": {
        "31b_global": ATTN_CONFIG_GLOBAL,
        "local": ATTN_CONFIG_LOCAL,
    },
}

GEGLU_CONFIG_POLICY = {
    "default": {
        "31b": GEGLU_CONFIG,
        "default": GEGLU_CONFIG,
    },
    "a100": {
        "31b": GEGLU_CONFIG,
        "default": GEGLU_CONFIG,
    },
    "h100": {
        "31b": GEGLU_CONFIG,
        "default": GEGLU_CONFIG,
    },
}


def _copy_config(config: dict[str, int] | None) -> dict[str, int] | None:
    if config is None:
        return None
    return dict(config)


def gpu_family_name(gpu_name: str | None = None) -> str:
    if gpu_name is None:
        import torch

        gpu_name = torch.cuda.get_device_name(0)
    name = gpu_name.upper()
    if "H100" in name:
        return "h100"
    if "A100" in name:
        return "a100"
    return "default"


def fused_norm_linear_profile(m: int, n: int, k: int) -> str:
    if m >= 1024 and k >= 4096:
        return "31b_prefill"
    if m >= 1024:
        return "prefill"
    return "decode"


def attention_profile(head_dim: int, window_size: int) -> str:
    if head_dim >= 512 and window_size <= 0:
        return "31b_global"
    return "local"


def geglu_profile(ffn_dim: int) -> str:
    if ffn_dim >= 21504:
        return "31b"
    return "default"


def fused_norm_linear_config_for_shape(
    m: int,
    n: int,
    k: int,
    *,
    gpu_name: str | None = None,
) -> dict[str, int] | None:
    gpu = gpu_family_name(gpu_name)
    profile = fused_norm_linear_profile(m, n, k)
    policy = FUSED_NORM_LINEAR_CONFIG_POLICY.get(gpu, FUSED_NORM_LINEAR_CONFIG_POLICY["default"])
    return _copy_config(policy.get(profile, None))


def attention_config_for_shape(
    head_dim: int,
    window_size: int,
    *,
    gpu_name: str | None = None,
) -> dict[str, int]:
    gpu = gpu_family_name(gpu_name)
    profile = attention_profile(head_dim, window_size)
    policy = ATTN_CONFIG_POLICY.get(gpu, ATTN_CONFIG_POLICY["default"])
    return _copy_config(policy.get(profile, ATTN_CONFIG_LOCAL)) or dict(ATTN_CONFIG_LOCAL)


def geglu_config_for_ffn_dim(
    ffn_dim: int,
    *,
    gpu_name: str | None = None,
) -> dict[str, int]:
    gpu = gpu_family_name(gpu_name)
    profile = geglu_profile(ffn_dim)
    policy = GEGLU_CONFIG_POLICY.get(gpu, GEGLU_CONFIG_POLICY["default"])
    return _copy_config(policy.get(profile, GEGLU_CONFIG)) or dict(GEGLU_CONFIG)
