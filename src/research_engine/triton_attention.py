"""Parameterized Triton FlashAttention kernel and operator spec.

Tiled scaled dot-product attention with online softmax (FlashAttention-style).
Computes: out = softmax(Q @ K.T * scale) @ V per head.

The kernel supports causal masking, sliding-window local attention, and
combinations of both.  It operates on tiles of BLOCK_M queries by BLOCK_N
keys at a time, accumulating the online softmax normalizer so no O(N^2)
intermediate attention matrix is materialized.

Sliding-window attention (WINDOW_SIZE > 0):
  Each query position q_i only attends to key positions in
  [q_i - WINDOW_SIZE + 1, q_i] (causal) or
  [q_i - WINDOW_SIZE + 1, q_i + WINDOW_SIZE - 1] (non-causal, not implemented here).
  Tile pruning skips K-tiles that are entirely outside the window, giving
  true O(N * WINDOW_SIZE) complexity instead of O(N^2).

References:
- FlashAttention-2 (Dao 2023): https://arxiv.org/abs/2307.08691
- FlashAttention-3 (Shah et al 2024): for Hopper/H100 TMA/WGMMA optimizations
- Mistral-7B (Jiang et al 2023): sliding-window attention for efficient long contexts

Search space: BLOCK_M, BLOCK_N, num_warps, num_stages.
Different head dims and sequence lengths prefer different configurations.
"""

from __future__ import annotations

import json
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


ATTENTION_PARAM_SPACE = {
    "BLOCK_M": [32, 64, 128],
    "BLOCK_N": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}


ATTENTION_CURATED_CONFIGS = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    # T4-optimized: short_64 shape (batch=4, heads=32, seq=512, head_dim=64)
    # needs small tiles + fewer warps for 40-SM GPU to avoid underutilization
    {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 2, "num_stages": 3},
    # Large head_dim (512, Gemma 4 global): smaller tiles reduce register pressure
    {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
]


# Shape buckets: (batch, num_heads, seq_len, head_dim)
# These match LLM workload patterns.
ATTENTION_SHAPE_BUCKETS = [
    {"name": "short_64", "batch": 4, "heads": 32, "num_kv_heads": 32, "seq_len": 512, "head_dim": 64, "is_causal": False},
    {"name": "short_128", "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 1024, "head_dim": 128, "is_causal": False},
    {"name": "med_128", "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 2048, "head_dim": 128, "is_causal": False},
    {"name": "long_64", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 64, "is_causal": False},
    {"name": "long_128", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 128, "is_causal": False},
    {"name": "llama7b", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": False},
    {"name": "mistral", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 8192, "head_dim": 128, "is_causal": False},
    # Causal variants (for decoder-only LLMs)
    {"name": "llama7b_causal", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": True},
    {"name": "mistral_causal", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 8192, "head_dim": 128, "is_causal": True},
    {"name": "long_128_causal", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 128, "is_causal": True},
    # Sliding-window local attention (Gemma 3 / 4 style)
    # 5 of every 6 layers use a 1024-token window; full attention on the rest.
    {"name": "gemma4_local_1024", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024},
    {"name": "gemma4_local_short", "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 2048, "head_dim": 128, "is_causal": True, "window_size": 1024},
    {"name": "gemma3_local", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 8192, "head_dim": 128, "is_causal": True, "window_size": 1024},
    # QK-norm variants (Gemma 3 / 4 -- all layers normalise Q and K before dot-product).
    # window_size=1024 -> local layer (5 of 6); window_size=-1 -> global layer (1 of 6).
    {"name": "gemma4_qknorm", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_qknorm_global", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": -1, "use_qk_norm": True},
    # GQA variants (Gemma 4 / Llama 3 / Mistral — grouped-query attention).
    # Gemma 4 31B: 32:16 local, 32:4 global with head_dim=512.
    {"name": "gemma4_31b_local", "batch": 1, "heads": 32, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_31b_global", "batch": 1, "heads": 32, "num_kv_heads": 4, "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1, "use_qk_norm": True},
    # Gemma 4 26B-A4B: 16:8 local, 16:2 global with head_dim=512.
    {"name": "gemma4_26b_a4b_local", "batch": 1, "heads": 16, "num_kv_heads": 8, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_26b_a4b_global", "batch": 1, "heads": 16, "num_kv_heads": 2, "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1, "use_qk_norm": True},
    # Llama 3 70B: 64:8 GQA.
    {"name": "llama3_70b_gqa", "batch": 1, "heads": 64, "num_kv_heads": 8, "seq_len": 4096, "head_dim": 128, "is_causal": True, "window_size": -1},
    # Mistral: 32:8 GQA.
    {"name": "mistral_gqa", "batch": 1, "heads": 32, "num_kv_heads": 8, "seq_len": 8192, "head_dim": 128, "is_causal": True, "window_size": -1},
    # YOCO KV-shared layers (Gemma 4 last-N layers reuse earlier KV, skip K-norm)
    {"name": "gemma4_31b_yoco_local", "batch": 1, "heads": 32, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True, "shared_kv": True},
    {"name": "gemma4_31b_yoco_global", "batch": 1, "heads": 32, "num_kv_heads": 4, "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1, "use_qk_norm": True, "shared_kv": True},
]


def attention_config_id(config: dict[str, int]) -> str:
    return f"m{config['BLOCK_M']}_n{config['BLOCK_N']}_w{config['num_warps']}_s{config['num_stages']}"


def attention_shape_bucket_key(shape: dict[str, int]) -> str:
    seq = shape.get("seq_len", 0)
    hd = shape.get("head_dim", 0)
    heads = shape.get("heads", 0)
    nkv = shape.get("num_kv_heads", heads)  # default = MHA
    ws = shape.get("window_size", -1)
    use_qk_norm = bool(shape.get("use_qk_norm", False))
    shared_kv = bool(shape.get("shared_kv", False))
    is_gqa = nkv > 0 and nkv < heads

    # YOCO KV-shared layers: route BEFORE GQA so shared_kv=True shapes don't
    # fall through to non-YOCO GQA buckets.  Gemma 4 YOCO layers reuse K/V
    # from an earlier layer and skip K-norm (K is already normalized).
    if shared_kv and is_gqa:
        if hd >= 512:
            return "gemma4_31b_yoco_global"
        return "gemma4_31b_yoco_local"

    # GQA buckets take precedence — these are the LLM-shaped routes.
    # Route BEFORE the qk-norm branch so Gemma 4 GQA + QK-norm shapes land here.
    if is_gqa:
        if use_qk_norm:
            # Gemma 4. head_dim=512 signals global, 256 signals local.
            if hd >= 512:
                return "gemma4_31b_global" if heads >= 32 else "gemma4_26b_a4b_global"
            # local layers: distinguish 31B (32:16) vs 26B (16:8) by heads
            return "gemma4_31b_local" if heads >= 32 else "gemma4_26b_a4b_local"
        # Non-QK-norm GQA: Llama 3 / Mistral
        if seq >= 8192:
            return "mistral_gqa"
        if heads >= 64:
            return "llama3_70b_gqa"
        # Fallback: treat as mistral_gqa for now
        return "mistral_gqa"

    # QK-norm shapes get dedicated buckets (Gemma 3/4 with fused QK-RMSNorm).
    if use_qk_norm:
        if ws is not None and ws > 0:
            return "gemma4_qknorm"
        return "gemma4_qknorm_global"

    # Sliding-window shapes get their own buckets
    if ws is not None and ws > 0:
        if seq >= 8192:
            return "gemma3_local"
        if hd >= 256:
            return "gemma4_local_1024"
        return "gemma4_local_short"

    if seq >= 8192:
        return "mistral"
    if seq >= 4096:
        if hd <= 64:
            return "long_64"
        return "llama7b" if heads >= 32 else "long_128"
    if seq >= 2048:
        return "med_128"
    if hd <= 64:
        return "short_64"
    return "short_128"


def attention_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation only — always returns True.

    The system now learns feasibility from runtime failures (recorded as
    reward=0) rather than filtering at grid-generation time. Kept on the
    operator spec for backward compatibility with any external callers.
    """
    return True


def generate_attention_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in ATTENTION_CURATED_CONFIGS:
            cid = attention_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bm in ATTENTION_PARAM_SPACE["BLOCK_M"]:
        for bn in ATTENTION_PARAM_SPACE["BLOCK_N"]:
            for nw in ATTENTION_PARAM_SPACE["num_warps"]:
                for ns in [2, 3]:
                    config = {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "num_warps": nw,
                        "num_stages": ns,
                    }
                    cid = attention_config_id(config)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    configs.append(config)
                    if len(configs) >= max_configs:
                        return configs
    return configs


def make_sliding_window_mask(seq_len: int, window_size: int, is_causal: bool):
    """Build a boolean attention mask for sliding-window (optionally causal) attention.

    Returns a (seq_len, seq_len) boolean tensor where True means "attend to this key".
    Suitable for passing as ``attn_mask`` to
    ``torch.nn.functional.scaled_dot_product_attention``.

    Args:
        seq_len: Sequence length S.
        window_size: Number of key positions each query attends to.  With
            is_causal=True these are the window_size most-recent past tokens
            (including the query itself).  With is_causal=False the window is
            symmetric: [i - window_size + 1, i + window_size - 1].
        is_causal: If True, also apply a causal (lower-triangular) mask.
    """
    import torch
    rows = torch.arange(seq_len).unsqueeze(1)  # (S, 1)
    cols = torch.arange(seq_len).unsqueeze(0)  # (1, S)
    left_bound = cols >= (rows - window_size + 1)
    if is_causal:
        mask = left_bound & (cols <= rows)
    else:
        mask = left_bound & ((cols - rows) < window_size)
    return mask  # dtype=torch.bool


def flash_attn(
    q, k, v, config=None, is_causal=False, sm_scale=None, window_size=-1,
    use_qk_norm=False, q_scale=None, k_scale=None,
    num_kv_heads=None, shared_kv=False,
):
    """Module-level FlashAttention launcher. Requires CUDA GPU.

    Args:
        q: (B, H, S, D) fp16 tensor.
        k: (B, H_kv, S, D) fp16 tensor.
        v: (B, H_kv, S, D) fp16 tensor.
        config: Triton config dict with BLOCK_M, BLOCK_N, num_warps, num_stages.
            Defaults to the first curated config.
        is_causal: Apply causal masking.
        sm_scale: Softmax scale. Defaults to 1/sqrt(D).
        window_size: Sliding-window size. -1 means full attention.
        use_qk_norm: Fuse per-row RMSNorm into Q and K.
        q_scale: (D,) fp32 scale weights for Q-norm.
        k_scale: (D,) fp32 scale weights for K-norm.
        num_kv_heads: Number of KV heads for GQA. Defaults to H (MHA).
        shared_kv: YOCO-style KV sharing.

    Returns:
        (B, H, S, D) fp16 output tensor.
    """
    import torch
    import triton
    import triton.language as tl

    if config is None:
        config = ATTENTION_CURATED_CONFIGS[0]

    @triton.jit
    def _attn_fwd_kernel(
        Q, K, V, Out,
        QScale, KScale,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        B, H, M, N,
        sm_scale,
        HEAD_DIM: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        USE_QK_NORM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H
        off_kvh = off_h // GROUP_SIZE

        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_kvh * stride_kh
        v_base = V + off_b * stride_vb + off_kvh * stride_vh
        o_base = Out + off_b * stride_ob + off_h * stride_oh

        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q_mask = offs_m[:, None] < M
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        if USE_QK_NORM:
            q = q.to(tl.float32)
            q_sq = q * q
            q_var = tl.sum(q_sq, axis=1) / HEAD_DIM
            q_rstd = 1.0 / tl.sqrt(q_var + 1e-6)
            q = q * q_rstd[:, None]
            q_scale_val = tl.load(QScale + offs_k)
            q = q * q_scale_val[None, :]
            q = q.to(tl.float16)

        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - 1.0e30
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        qk_scale = sm_scale * 1.44269504

        for start_n in range(0, N, BLOCK_N):
            curr_n = start_n + offs_n
            k_ptrs = k_base + curr_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
            v_ptrs = v_base + curr_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
            n_mask = curr_n[:, None] < N
            k = tl.load(k_ptrs, mask=n_mask, other=0.0)
            v = tl.load(v_ptrs, mask=n_mask, other=0.0)

            if USE_QK_NORM:
                k = k.to(tl.float32)
                k_sq = k * k
                k_var = tl.sum(k_sq, axis=1) / HEAD_DIM
                k_rstd = 1.0 / tl.sqrt(k_var + 1e-6)
                k = k * k_rstd[:, None]
                k_scale_val = tl.load(KScale + offs_k)
                k = k * k_scale_val[None, :]
                k = k.to(tl.float16)

            qk = tl.dot(q, tl.trans(k))
            qk = qk * qk_scale

            NEG_INF = -1.0e30
            qk = tl.where(curr_n[None, :] < N, qk, NEG_INF)

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= curr_n[None, :]
                qk = tl.where(causal_mask, qk, NEG_INF)

            if WINDOW_SIZE > 0:
                window_floor = offs_m[:, None] - WINDOW_SIZE + 1
                window_mask = curr_n[None, :] >= window_floor
                qk = tl.where(window_mask, qk, NEG_INF)

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_ij = tl.sum(p, axis=1)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_new

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / safe_l[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

    B, H, M, D = q.shape
    _, Hk, N, Dk = k.shape
    _, Hv, Nv, Dv = v.shape
    if num_kv_heads is None:
        num_kv_heads = H
    assert num_kv_heads > 0, "num_kv_heads must be positive"
    assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
    assert Hk == num_kv_heads and Hv == num_kv_heads, (
        f"K/V must have shape[1] == num_kv_heads; got Hk={Hk} Hv={Hv} num_kv_heads={num_kv_heads}"
    )
    assert Dk == D and Dv == D, "head_dim must match across Q/K/V"
    assert Nv == N, "K/V seq_len must match"
    group_size = H // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    if q_scale is None:
        q_scale = torch.ones(D, device=q.device, dtype=torch.float32)
    if k_scale is None:
        k_scale = torch.ones(D, device=k.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    _attn_fwd_kernel[grid](
        q, k, v, out,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N,
        sm_scale,
        HEAD_DIM=D,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE=window_size,
        USE_QK_NORM=use_qk_norm,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def generate_attention_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained script benchmarking all attention configs.

    Uses a minimal Triton FlashAttention kernel with optional sliding-window
    support via the WINDOW_SIZE constexpr.  WINDOW_SIZE=-1 means no window
    (full causal / full non-causal per IS_CAUSAL).  WINDOW_SIZE>0 restricts
    each query to at most WINDOW_SIZE keys, enabling tile pruning to achieve
    O(N * W) instead of O(N^2) complexity.
    """
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton FlashAttention benchmark (with sliding-window support)."""

import json
import platform
# Parse configs and shapes from JSON to avoid Python literal issues (true/false)
CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}

import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd_kernel(
    Q, K, V, Out,
    QScale, KScale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    USE_QK_NORM: tl.constexpr,
):
    """Tiled FlashAttention forward with online softmax and sliding-window support.

    Tile-pruning strategy
    ---------------------
    For each query tile (pid * BLOCK_M .. (pid+1) * BLOCK_M - 1) we compute a
    tight [k_start, k_end) range of key positions that can contribute non-(-inf)
    attention scores, then only iterate K-tiles that overlap that range.

    With IS_CAUSAL=True, WINDOW_SIZE=W:
        k_start = max(0, pid * BLOCK_M - W + 1)   # oldest key in window
        k_end   = min(N, (pid + 1) * BLOCK_M)     # causal: no future keys

    With IS_CAUSAL=False, WINDOW_SIZE=W (symmetric window):
        k_start = max(0, pid * BLOCK_M - W + 1)
        k_end   = min(N, (pid + 1) * BLOCK_M + W - 1)

    With WINDOW_SIZE=-1 (no window):
        k_start = 0 (or diagonal start for causal)
        k_end   = N (or diagonal end for causal)

    A K-tile starting at start_n is skipped if
        start_n + BLOCK_N <= k_start  OR  start_n >= k_end.

    For IS_CAUSAL=True without window this reduces to the usual causal loop
    (loop from 0 to diagonal), matching the original behavior.
    """
    pid = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H                      # Q head index, [0, H)
    # GQA: integer-divide Q head by group size to find KV head.
    # For MHA (GROUP_SIZE=1) this is a no-op.
    off_kvh = off_h // GROUP_SIZE           # KV head index, [0, NUM_KV_HEADS)

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_kvh * stride_kh
    v_base = V + off_b * stride_vb + off_kvh * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Fused QK-norm: per-row RMSNorm on Q before tile loop (epsilon=1e-6).
    if USE_QK_NORM:
        q = q.to(tl.float32)
        q_sq = q * q
        q_var = tl.sum(q_sq, axis=1) / HEAD_DIM
        q_rstd = 1.0 / tl.sqrt(q_var + 1e-6)
        q = q * q_rstd[:, None]
        q_scale = tl.load(QScale + offs_k)
        q = q * q_scale[None, :]
        q = q.to(tl.float16)

    # Initialize online-softmax state.  Start m_i at a large finite
    # negative (not -inf) so that early tiles with all-masked positions
    # produce 0 exp2 contributions rather than NaN when subtracting -inf.
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - 1.0e30
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e) for tl.exp2

    # --- Loop over all K tiles with mask-based window/causal enforcement ---
    # We iterate the full `range(0, N, BLOCK_N)` and enforce causal + window
    # constraints via `tl.where` masks on the raw QK scores.  This pattern
    # compiles reliably under Triton's JIT (no `break`/`continue` on runtime
    # predicates, no tensor-typed `range` bounds).  Cost: sliding-window
    # runs O(N^2) instead of O(N*W).  Correct-first; tile pruning is a
    # follow-up optimization.
    for start_n in range(0, N, BLOCK_N):
        curr_n = start_n + offs_n
        k_ptrs = k_base + curr_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = v_base + curr_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        n_mask = curr_n[:, None] < N
        k = tl.load(k_ptrs, mask=n_mask, other=0.0)
        v = tl.load(v_ptrs, mask=n_mask, other=0.0)

        # Fused QK-norm: per-row RMSNorm on K inside tile loop.
        if USE_QK_NORM:
            k = k.to(tl.float32)
            k_sq = k * k
            k_var = tl.sum(k_sq, axis=1) / HEAD_DIM
            k_rstd = 1.0 / tl.sqrt(k_var + 1e-6)
            k = k * k_rstd[:, None]
            k_scale = tl.load(KScale + offs_k)
            k = k * k_scale[None, :]
            k = k.to(tl.float16)

        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale

        # Mask out-of-bounds keys (past sequence end).  Use a large finite
        # negative instead of -inf so that tiles where every position is
        # masked don't produce NaN in the online-softmax arithmetic.
        NEG_INF = -1.0e30
        qk = tl.where(curr_n[None, :] < N, qk, NEG_INF)

        # Apply causal mask on this tile.
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_n[None, :]
            qk = tl.where(causal_mask, qk, NEG_INF)

        # Apply sliding-window mask: mask positions below the window floor.
        if WINDOW_SIZE > 0:
            window_floor = offs_m[:, None] - WINDOW_SIZE + 1
            window_mask = curr_n[None, :] >= window_floor
            qk = tl.where(window_mask, qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # Handle empty row (no keys attended to) — happens in causal mode or when
    # the window excludes all keys for early query positions.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / safe_l[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def flash_attn(
    q, k, v, config, is_causal=False, sm_scale=None, window_size=-1,
    use_qk_norm=False, q_scale=None, k_scale=None,
    num_kv_heads=None, shared_kv=False,
):
    """q, k, v: (batch, heads, seq, head_dim) float16. Returns out.

    Args:
        window_size: Sliding-window size.  -1 (default) means no window (full
            attention).  >0 restricts each query to at most window_size keys.
        use_qk_norm: Fuse per-row RMSNorm into Q and K (Gemma 3/4, epsilon=1e-6).
        q_scale: Learnable [head_dim] scale weights for Q-norm (float32).
        k_scale: Learnable [head_dim] scale weights for K-norm (float32).
        num_kv_heads: Number of KV heads for grouped-query attention. If None
            (default), equals the number of Q heads (MHA). Must divide `heads`.
        shared_kv: YOCO-style KV-cache sharing (Gemma 4).  When True, K/V
            tensors are reused from an earlier layer and K is already
            normalized.  The launcher still applies QK-norm to Q but skips
            K-norm (passes USE_QK_NORM=True to the kernel — K-norm is
            redundant on already-normalized data and acts as a no-op).
    """
    B, H, M, D = q.shape
    _, Hk, N, Dk = k.shape
    _, Hv, Nv, Dv = v.shape
    if num_kv_heads is None:
        num_kv_heads = H
    assert num_kv_heads > 0, "num_kv_heads must be positive"
    assert H % num_kv_heads == 0, f"H={{H}} not divisible by num_kv_heads={{num_kv_heads}}"
    assert Hk == num_kv_heads and Hv == num_kv_heads, (
        f"K/V must have shape[1] == num_kv_heads; got Hk={{Hk}} Hv={{Hv}} num_kv_heads={{num_kv_heads}}"
    )
    assert Dk == D and Dv == D, "head_dim must match across Q/K/V"
    assert Nv == N, "K/V seq_len must match"
    group_size = H // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    if q_scale is None:
        q_scale = torch.ones(D, device=q.device, dtype=torch.float32)
    if k_scale is None:
        k_scale = torch.ones(D, device=k.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    attn_fwd_kernel[grid](
        q, k, v, out,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N,
        sm_scale,
        HEAD_DIM=D,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE=window_size,
        USE_QK_NORM=use_qk_norm,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def make_sliding_window_mask(seq_len: int, window_size: int, is_causal: bool) -> "torch.Tensor":
    """Build a boolean attention mask for sliding-window (optionally causal) attention.

    Returns a (seq_len, seq_len) boolean tensor where True means "attend".
    Suitable for passing as ``attn_mask`` to
    ``torch.nn.functional.scaled_dot_product_attention``.
    """
    import torch
    rows = torch.arange(seq_len).unsqueeze(1)  # (S, 1)
    cols = torch.arange(seq_len).unsqueeze(0)  # (1, S)
    # Key must be within the window
    in_window = (rows - cols) < window_size  # col >= row - window_size + 1
    left_bound = cols >= (rows - window_size + 1)
    if is_causal:
        # Key position must not be in the future
        mask = left_bound & (cols <= rows)
    else:
        # Symmetric window
        mask = left_bound & ((cols - rows) < window_size)
    return mask  # True = attend


def benchmark_one(batch, heads, seq_len, head_dim, config, is_causal=False, window_size=-1, use_qk_norm=False, num_kv_heads=None, shared_kv=False):
    try:
        if num_kv_heads is None:
            num_kv_heads = heads
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

        # YOCO shared-KV: K/V come pre-normalized from an earlier layer.
        # Pre-normalize K so the reference path can skip K-norm (matching the
        # real YOCO pipeline where earlier layers already applied RMSNorm to K).
        if shared_kv:
            k = torch.nn.functional.rms_norm(k.float(), (head_dim,)).half()

        # PyTorch reference uses repeat_interleave to broadcast K/V across
        # group members (avoids version-dependent SDPA GQA behavior). Reference
        # cost is not benchmarked for speed, so the extra memory is fine.
        group_size = heads // num_kv_heads
        k_ref = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
        v_ref = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v

        # When QK-norm enabled, normalise Q/K before PyTorch reference.
        # YOCO shared-KV: skip K-norm (K already normalized from earlier layer).
        if use_qk_norm:
            q_ref = torch.nn.functional.rms_norm(q.float(), (head_dim,)).half()
            if not shared_kv:
                k_ref = torch.nn.functional.rms_norm(k_ref.float(), (head_dim,)).half()
        else:
            q_ref = q

        if window_size > 0:
            # Build explicit sliding-window mask for PyTorch reference.
            ws_mask = make_sliding_window_mask(seq_len, window_size, is_causal).to(q.device)
            # SDPA expects float mask (0 = attend, -inf = mask out) or bool mask.
            # Pass as bool; SDPA converts internally.
            ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref,
                attn_mask=ws_mask,
                is_causal=False,  # mask already encodes causality
            )
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=is_causal)

        out = flash_attn(q, k, v, config, is_causal=is_causal, window_size=window_size, use_qk_norm=use_qk_norm, num_kv_heads=num_kv_heads, shared_kv=shared_kv)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "tflops": None}}

        ms = triton.testing.do_bench(
            lambda: flash_attn(q, k, v, config, is_causal=is_causal, window_size=window_size, use_qk_norm=use_qk_norm, num_kv_heads=num_kv_heads, shared_kv=shared_kv),
            warmup=10, rep=50,
        )
        # Effective work: causal halves flops; window shrinks further.
        if window_size > 0:
            window_factor = min(window_size, seq_len) / seq_len
            if is_causal:
                window_factor *= 0.5
        else:
            window_factor = 0.5 if is_causal else 1.0
        flops = 4.0 * batch * heads * seq_len * seq_len * head_dim * window_factor
        tflops = flops / (ms * 1e-3) / 1e12
        return {{
            "correct": True,
            "max_err": round(max_err, 5),
            "ms": round(ms, 4),
            "tflops": round(tflops, 2),
        }}
    except Exception as exc:
        return {{"correct": False, "error": str(exc)[:200], "ms": None, "tflops": None}}


def main():
    configs = json.loads(CONFIGS_JSON)
    shapes = json.loads(SHAPES_JSON)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    all_results = []
    for config in configs:
        cid = "m{{}}_n{{}}_w{{}}_s{{}}".format(
            config["BLOCK_M"], config["BLOCK_N"],
            config["num_warps"], config["num_stages"],
        )
        shape_results = []
        for shape in shapes:
            batch = shape["batch"]
            heads = shape["heads"]
            seq_len = shape["seq_len"]
            head_dim = shape["head_dim"]
            is_causal = bool(shape.get("is_causal", False))
            window_size = int(shape.get("window_size", -1))
            use_qk_norm = bool(shape.get("use_qk_norm", False))
            num_kv_heads = int(shape.get("num_kv_heads", heads))
            shared_kv = bool(shape.get("shared_kv", False))
            result = benchmark_one(
                batch, heads, seq_len, head_dim, config,
                is_causal=is_causal, window_size=window_size, use_qk_norm=use_qk_norm,
                num_kv_heads=num_kv_heads, shared_kv=shared_kv,
            )
            result["shape"] = f"{{batch}}x{{heads}}x{{seq_len}}x{{head_dim}}"
            result["shape_name"] = shape.get("name", "")
            result["is_causal"] = is_causal
            result["window_size"] = window_size
            result["use_qk_norm"] = use_qk_norm
            result["num_kv_heads"] = num_kv_heads
            result["shared_kv"] = shared_kv
            shape_results.append(result)
        all_results.append({{
            "config_id": cid,
            "config": config,
            "results": shape_results,
        }})

    output = {{
        "operator": "attention",
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


ATTENTION_SPEC = register_operator(TritonOperatorSpec(
    name="attention",
    param_space=ATTENTION_PARAM_SPACE,
    curated_configs=ATTENTION_CURATED_CONFIGS,
    shape_buckets=ATTENTION_SHAPE_BUCKETS,
    metric_name="tflops",
    config_id_fn=attention_config_id,
    shape_bucket_fn=attention_shape_bucket_key,
    benchmark_script_fn=generate_attention_benchmark_script,
    grid_generator_fn=generate_attention_grid,
    shared_memory_check_fn=attention_shared_memory_check,
    description="FlashAttention-style scaled dot-product attention with tiled online softmax and sliding-window support.",
))
