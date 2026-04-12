"""Parameterized Triton kernel for fused QK-RMSNorm + RoPE (Gemma 3/4 prologue).

This is the single fused prologue that Gemma 3 and Gemma 4 apply to each
attention block: Q is RMS-normalized, multiplied by a learnable affine of
the form ``(1 + q_scale)`` (Gemma-mode), then rotated by RoPE; K receives
the same treatment with its own ``k_scale``. Under GQA, K has a different
number of heads (``H_kv``) from Q (``H``).

vLLM deep-read (docs/research/vllm-gemma4-kernel-patterns.md) confirms that
the vLLM implementation launches 4+ separate kernels for this prologue:

    1. Q-RMSNorm
    2. K-RMSNorm
    3. Q-RoPE
    4. K-RoPE

Fusing the pair (rmsnorm + rope) into a single pass per tensor means the
Gemma-mode affine, the rstd math, and the RoPE rotation all run on the
resident fp32 register copy of each (b, h, s) row and we pay the HBM
traffic exactly once. This module ships that fused kernel, a PyTorch
reference, and a vLLM-style "separated baseline" so we can report the
fusion speedup ``separated_ms / fused_ms`` as the headline number.

Design:
- Two kernel launches (Q and K) both call a single ``@triton.jit`` entry.
- Grid: ``(B * H * S,)`` or ``(B * H_kv * S,)``. One program per row.
- BLOCK_SIZE constexpr covers ``head_dim // 2`` pairs (the RoPE split-pair).
- Loads even + odd halves, scale[D] affine params, cos[S, D/2], sin[S, D/2].
- fp32 intermediate math, fp16 output. eps = 1e-6.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


QK_NORM_ROPE_PARAM_SPACE = {
    "BLOCK_SIZE": [32, 64, 128, 256, 512],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
}


QK_NORM_ROPE_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 32, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 64, "num_warps": 4, "num_stages": 1},
    {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_SIZE": 256, "num_warps": 8, "num_stages": 2},
    # T4-optimized: 40 SMs, fewer warps reduce register pressure;
    # head_dim=256 -> half=128 pairs, so BLOCK_SIZE=128 covers it exactly
    {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 64, "num_warps": 2, "num_stages": 1},
    # head_dim=512 global layers: half=256 pairs, need BLOCK_SIZE=256
    {"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 1},
]


# Real Gemma 3 / Gemma 4 per-layer shapes (batch, heads, num_kv_heads, seq, head_dim)
QK_NORM_ROPE_SHAPE_BUCKETS = [
    {"name": "gemma3_local_1024", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq": 4096, "head_dim": 256},
    {"name": "gemma4_e2b_local", "batch": 1, "heads": 8, "num_kv_heads": 1, "seq": 4096, "head_dim": 256},
    {"name": "gemma4_26b_a4b_local", "batch": 1, "heads": 16, "num_kv_heads": 8, "seq": 4096, "head_dim": 256},
    {"name": "gemma4_26b_a4b_global", "batch": 1, "heads": 16, "num_kv_heads": 2, "seq": 4096, "head_dim": 512},
    {"name": "gemma4_31b_local", "batch": 1, "heads": 32, "num_kv_heads": 16, "seq": 4096, "head_dim": 256},
    {"name": "gemma4_31b_global", "batch": 1, "heads": 32, "num_kv_heads": 4, "seq": 4096, "head_dim": 512},
    # Non-Gemma architectures — kernel generalizes via affine_mode=0 (standard RMSNorm)
    # for models without QK-norm, and affine_mode=1 for Phi-3 which has QK-norm.
    {"name": "llama3_8b", "batch": 1, "heads": 32, "num_kv_heads": 8, "seq": 4096, "head_dim": 128},
    {"name": "llama3_70b", "batch": 1, "heads": 64, "num_kv_heads": 8, "seq": 4096, "head_dim": 128},
    {"name": "mistral_7b", "batch": 1, "heads": 32, "num_kv_heads": 8, "seq": 4096, "head_dim": 128},
    {"name": "phi3_mini", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq": 4096, "head_dim": 96},
]


def qk_norm_rope_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def qk_norm_rope_shape_bucket_key(shape: dict[str, int]) -> str:
    """Classify a QK-RMSNorm+RoPE shape into a named bucket.

    Discriminators:
    - head_dim: 96 = Phi-3, 128 = LLaMA/Mistral, 256 = Gemma local, 512 = Gemma global
    - heads: 8 / 16 / 32 / 64
    - num_kv_heads: picks the per-variant GQA ratio
    """
    hd = shape.get("head_dim", 0)
    h = shape.get("heads", 0)
    h_kv = shape.get("num_kv_heads", 0)
    name = shape.get("name", "")

    # Exact name match for non-Gemma buckets
    if name in ("llama3_8b", "llama3_70b", "mistral_7b", "phi3_mini"):
        return name

    # Phi-3 mini: head_dim=96, MHA
    if hd == 96:
        return "phi3_mini"

    # LLaMA 3 / Mistral family: head_dim=128
    if hd == 128:
        if h >= 64:
            return "llama3_70b"
        # Both LLaMA 3 8B and Mistral 7B have heads=32, kv_heads=8, head_dim=128.
        # Disambiguate by name if available; default to llama3_8b.
        return "llama3_8b"

    if hd >= 512:
        # Gemma 4 global attention layers
        if h >= 32:
            return "gemma4_31b_global"
        return "gemma4_26b_a4b_global"

    # head_dim == 256 branch (Gemma local attention family)
    if h >= 32:
        return "gemma4_31b_local"
    if h >= 16:
        if h_kv >= 16:
            return "gemma3_local_1024"
        return "gemma4_26b_a4b_local"
    # heads <= 8
    return "gemma4_e2b_local"


def qk_norm_rope_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — feasibility is learned at runtime."""
    return True


def generate_qk_norm_rope_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 100,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in QK_NORM_ROPE_CURATED_CONFIGS:
            cid = qk_norm_rope_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bs in QK_NORM_ROPE_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in QK_NORM_ROPE_PARAM_SPACE["num_warps"]:
            for ns in [1, 2]:
                config = {"BLOCK_SIZE": bs, "num_warps": nw, "num_stages": ns}
                cid = qk_norm_rope_config_id(config)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(config)
                if len(configs) >= max_configs:
                    return configs
    return configs


_triton_qk_available = False
_qk_norm_rope_kernel_compiled = None


def _ensure_triton_qk_norm_rope():
    global _triton_qk_available, _qk_norm_rope_kernel_compiled
    if _qk_norm_rope_kernel_compiled is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _qk_norm_rope_kernel(
            x_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
            row_stride,
            heads, seq_len, head_dim,
            eps,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            s_idx = pid % seq_len
            x_base = x_ptr + pid * row_stride
            out_base = out_ptr + pid * row_stride
            half = head_dim // 2
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < half
            x_even = tl.load(x_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
            x_odd = tl.load(x_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
            sum_sq = tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)
            mean_sq = sum_sq / head_dim
            rstd = 1.0 / tl.sqrt(mean_sq + eps)
            s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
            s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
            n_even = x_even * rstd * (1.0 + s_even)
            n_odd = x_odd * rstd * (1.0 + s_odd)
            cos_row = cos_ptr + s_idx * half
            sin_row = sin_ptr + s_idx * half
            c = tl.load(cos_row + offs, mask=mask, other=1.0).to(tl.float32)
            sn = tl.load(sin_row + offs, mask=mask, other=0.0).to(tl.float32)
            out_even = n_even * c - n_odd * sn
            out_odd = n_even * sn + n_odd * c
            tl.store(out_base + 2 * offs, out_even.to(tl.float16), mask=mask)
            tl.store(out_base + 2 * offs + 1, out_odd.to(tl.float16), mask=mask)

        _qk_norm_rope_kernel_compiled = _qk_norm_rope_kernel
        _triton_qk_available = True
    except ImportError:
        _triton_qk_available = False


def apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config=None, eps=1e-6):
    """Module-level fused QK-RMSNorm+RoPE launcher. Requires CUDA GPU.

    Args:
        q: (B, H, S, D) fp16 tensor.
        k: (B, H_kv, S, D) fp16 tensor.
        cos: (S, D/2) fp32 tensor.
        sin: (S, D/2) fp32 tensor.
        q_scale: (D,) fp32 learnable scale for Q.
        k_scale: (D,) fp32 learnable scale for K.
        config: Triton config dict with BLOCK_SIZE, num_warps, num_stages.
            Defaults to the first curated config.
        eps: Epsilon for RMSNorm.

    Returns:
        (q_out, k_out) tuple of fp16 tensors with same shapes as inputs.
    """
    import torch
    import triton

    _ensure_triton_qk_norm_rope()
    if not _triton_qk_available:
        raise RuntimeError("Triton not available")

    if config is None:
        config = QK_NORM_ROPE_CURATED_CONFIGS[0]

    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_flat = q.reshape(B * H * S, D).contiguous()
    q_out_flat = q_out.reshape(B * H * S, D)
    k_flat = k.reshape(B * H_kv * S, D).contiguous()
    k_out_flat = k_out.reshape(B * H_kv * S, D)

    half = D // 2
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(half))
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    # Q launch
    _qk_norm_rope_kernel_compiled[(B * H * S,)](
        q_flat, q_scale, cos_c, sin_c, q_out_flat,
        D, H, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    # K launch
    _qk_norm_rope_kernel_compiled[(B * H_kv * S,)](
        k_flat, k_scale, cos_c, sin_c, k_out_flat,
        D, H_kv, S, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    return q_out, k_out


def generate_qk_norm_rope_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained Triton QK-RMSNorm+RoPE benchmark script."""
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated fused QK-RMSNorm+RoPE benchmark (Gemma 3/4 prologue)."""

import json
import platform

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}


@triton.jit
def qk_norm_rope_kernel(
    x_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
    row_stride,
    heads, seq_len, head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm (Gemma-mode affine) + split-pair RoPE for one row.

    Grid: (B * H * S,) — one program per (batch, head, seq) row.
    Each row has ``head_dim`` elements; we process the ``head_dim // 2``
    even/odd pairs in parallel within BLOCK_SIZE.
    """
    pid = tl.program_id(0)

    # Recover the sequence index for cos/sin lookup.
    # Layout is (b, h, s, d) contiguous, so pid = ((b * heads) + h) * seq_len + s
    s_idx = pid % seq_len

    x_base = x_ptr + pid * row_stride
    out_base = out_ptr + pid * row_stride

    half = head_dim // 2
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < half

    # Load even / odd halves of the row. fp16 -> fp32 for math.
    x_even = tl.load(x_base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: mean of squares over ALL head_dim elements.
    sum_sq = tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)
    mean_sq = sum_sq / head_dim
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Gemma-mode affine: (1 + scale). scale is [head_dim] fp32 learnable.
    s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    n_even = x_even * rstd * (1.0 + s_even)
    n_odd = x_odd * rstd * (1.0 + s_odd)

    # RoPE lookup — cos/sin rows are [seq_len, half] row-major.
    cos_row = cos_ptr + s_idx * half
    sin_row = sin_ptr + s_idx * half
    c = tl.load(cos_row + offs, mask=mask, other=1.0).to(tl.float32)
    sn = tl.load(sin_row + offs, mask=mask, other=0.0).to(tl.float32)

    out_even = n_even * c - n_odd * sn
    out_odd = n_even * sn + n_odd * c

    tl.store(out_base + 2 * offs, out_even.to(tl.float16), mask=mask)
    tl.store(out_base + 2 * offs + 1, out_odd.to(tl.float16), mask=mask)


def apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config, eps=1e-6):
    """Launch two kernel invocations — one for Q, one for K.

    q: (B, H, S, D) fp16
    k: (B, H_kv, S, D) fp16
    cos/sin: (S, D/2) fp32
    q_scale/k_scale: (D,) fp32
    """
    B, H, S, D = q.shape
    _, H_kv, _, _ = k.shape

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_flat = q.reshape(B * H * S, D).contiguous()
    q_out_flat = q_out.reshape(B * H * S, D)

    k_flat = k.reshape(B * H_kv * S, D).contiguous()
    k_out_flat = k_out.reshape(B * H_kv * S, D)

    half = D // 2
    BLOCK_SIZE = max(config["BLOCK_SIZE"], triton.next_power_of_2(half))

    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    # Q launch
    grid_q = (B * H * S,)
    qk_norm_rope_kernel[grid_q](
        q_flat, q_scale, cos_c, sin_c, q_out_flat,
        D,
        H, S, D,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    # K launch (GQA-aware: H_kv may differ from H)
    grid_k = (B * H_kv * S,)
    qk_norm_rope_kernel[grid_k](
        k_flat, k_scale, cos_c, sin_c, k_out_flat,
        D,
        H_kv, S, D,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    return q_out, k_out


def torch_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, eps=1e-6):
    """PyTorch reference: fused RMSNorm (Gemma-mode) + RoPE."""
    q_var = q.float().pow(2).mean(-1, keepdim=True)
    q_normed = (q.float() * torch.rsqrt(q_var + eps)).to(torch.float16) * (1.0 + q_scale).to(torch.float16)
    k_var = k.float().pow(2).mean(-1, keepdim=True)
    k_normed = (k.float() * torch.rsqrt(k_var + eps)).to(torch.float16) * (1.0 + k_scale).to(torch.float16)

    q_even, q_odd = q_normed[..., 0::2], q_normed[..., 1::2]
    k_even, k_odd = k_normed[..., 0::2], k_normed[..., 1::2]
    c = cos[None, None, :, :].to(torch.float16)
    sn = sin[None, None, :, :].to(torch.float16)
    q_out_even = q_even * c - q_odd * sn
    q_out_odd = q_even * sn + q_odd * c
    k_out_even = k_even * c - k_odd * sn
    k_out_odd = k_even * sn + k_odd * c
    q_out = torch.stack([q_out_even, q_out_odd], dim=-1).reshape(q.shape)
    k_out = torch.stack([k_out_even, k_out_odd], dim=-1).reshape(k.shape)
    return q_out.to(torch.float16), k_out.to(torch.float16)


def separated_baseline(q, k, cos, sin, q_scale, k_scale, eps=1e-6):
    """vLLM-style separated baseline: 4 distinct torch ops, no fusion.

    This mimics what vLLM does per layer:
      1. Q-RMSNorm
      2. K-RMSNorm
      3. Q-RoPE
      4. K-RoPE

    Reporting ``separated_ms / fused_ms`` against this baseline is the
    headline number that justifies the fused kernel.
    """
    D = q.shape[-1]
    # 1. Q-RMSNorm
    q_var = q.float().pow(2).mean(-1, keepdim=True)
    q_normed = (q.float() * torch.rsqrt(q_var + eps)).to(torch.float16) * (1.0 + q_scale).to(torch.float16)
    # 2. K-RMSNorm
    k_var = k.float().pow(2).mean(-1, keepdim=True)
    k_normed = (k.float() * torch.rsqrt(k_var + eps)).to(torch.float16) * (1.0 + k_scale).to(torch.float16)
    # 3. Q-RoPE
    q_even, q_odd = q_normed[..., 0::2], q_normed[..., 1::2]
    c = cos[None, None, :, :].to(torch.float16)
    sn = sin[None, None, :, :].to(torch.float16)
    q_out_even = q_even * c - q_odd * sn
    q_out_odd = q_even * sn + q_odd * c
    q_out = torch.stack([q_out_even, q_out_odd], dim=-1).reshape(q.shape)
    # 4. K-RoPE
    k_even, k_odd = k_normed[..., 0::2], k_normed[..., 1::2]
    k_out_even = k_even * c - k_odd * sn
    k_out_odd = k_even * sn + k_odd * c
    k_out = torch.stack([k_out_even, k_out_odd], dim=-1).reshape(k.shape)
    return q_out.to(torch.float16), k_out.to(torch.float16)


def benchmark_one(batch, heads, num_kv_heads, seq, head_dim, config):
    try:
        q = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, num_kv_heads, seq, head_dim), device="cuda", dtype=torch.float16)
        cos = torch.randn((seq, head_dim // 2), device="cuda", dtype=torch.float32)
        sin = torch.randn((seq, head_dim // 2), device="cuda", dtype=torch.float32)
        q_scale = torch.randn((head_dim,), device="cuda", dtype=torch.float32) * 0.1
        k_scale = torch.randn((head_dim,), device="cuda", dtype=torch.float32) * 0.1

        q_ref, k_ref = torch_qk_norm_rope(q, k, cos, sin, q_scale, k_scale)
        q_out, k_out = apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config)

        q_err = (q_out - q_ref).abs().max().item()
        k_err = (k_out - k_ref).abs().max().item()
        max_err = max(q_err, k_err)
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "gb_per_s": None, "tflops": None}}

        ms = triton.testing.do_bench(
            lambda: apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale, config),
            warmup=25, rep=100,
        )
        sep_ms = triton.testing.do_bench(
            lambda: separated_baseline(q, k, cos, sin, q_scale, k_scale),
            warmup=25, rep=100,
        )
        # Memory bandwidth: read q + k + cos + sin + scales, write q_out + k_out
        q_bytes = batch * heads * seq * head_dim * 2
        k_bytes = batch * num_kv_heads * seq * head_dim * 2
        trig_bytes = 2 * seq * (head_dim // 2) * 4
        scale_bytes = 2 * head_dim * 4
        bytes_moved = 2 * (q_bytes + k_bytes) + trig_bytes + scale_bytes
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        fusion_speedup = sep_ms / ms if ms > 0 else 0.0
        return {{
            "correct": True,
            "max_err": round(max_err, 6),
            "ms": round(ms, 4),
            "separated_ms": round(sep_ms, 4),
            "fusion_speedup": round(fusion_speedup, 3),
            "gb_per_s": round(gb_per_s, 2),
            "tflops": round(gb_per_s, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "gb_per_s": None, "tflops": None}}


def main():
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "bs{{}}_w{{}}_s{{}}".format(
            config["BLOCK_SIZE"], config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            batch = shape["batch"]
            heads = shape["heads"]
            num_kv_heads = shape["num_kv_heads"]
            seq = shape["seq"]
            head_dim = shape["head_dim"]
            result = benchmark_one(batch, heads, num_kv_heads, seq, head_dim, config)
            result["shape"] = f"{{batch}}x{{heads}}x{{num_kv_heads}}x{{seq}}x{{head_dim}}"
            result["shape_name"] = shape.get("name", "")
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "qk_norm_rope",
        "hardware": {{
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        }},
        "configs_tested": len(configs),
        "config_results": all_results,
    }}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
'''


QK_NORM_ROPE_SPEC = register_operator(TritonOperatorSpec(
    name="qk_norm_rope",
    param_space=QK_NORM_ROPE_PARAM_SPACE,
    curated_configs=QK_NORM_ROPE_CURATED_CONFIGS,
    shape_buckets=QK_NORM_ROPE_SHAPE_BUCKETS,
    metric_name="gb_per_s",
    config_id_fn=qk_norm_rope_config_id,
    shape_bucket_fn=qk_norm_rope_shape_bucket_key,
    benchmark_script_fn=generate_qk_norm_rope_benchmark_script,
    grid_generator_fn=generate_qk_norm_rope_grid,
    shared_memory_check_fn=qk_norm_rope_shared_memory_check,
    description=(
        "Fused Gemma 3/4 attention prologue: QK-RMSNorm (1+scale affine) + "
        "split-pair RoPE. Novel vs vLLM which launches 4 separate kernels."
    ),
))
