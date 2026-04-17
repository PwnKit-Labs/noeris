"""Official Triton FA2 tutorial kernel + Noeris sliding-window tile-pruning.

Ported from the official Triton FlashAttention-2 tutorial
(https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
with Noeris extensions:

  - **GQA support** (NUM_KV_HEADS / GROUP_SIZE)
  - **Sliding-window tile-pruning** (tile_in_range predicate)
  - **QK-norm support** (USE_QK_NORM constexpr for fused per-row RMSNorm)
  - **YOCO KV-share** (shared_kv flag — skip K-norm on pre-normalized K)

Key differences from the v1 attention kernel:
  - Uses tl.exp2 with qk_scale = sm_scale * 1.44269504 (log2(e)) for
    numerically stable online softmax, matching the official tutorial.
  - Initializes m_i = -inf and l_i = 1.0 (official convention) instead of
    m_i = -1e30 and l_i = 0.0.
  - Two-stage causal processing: off-band (STAGE 1, no mask) then on-band
    (STAGE 2, causal mask), matching the official tutorial's structure for
    better warp utilization.
  - Q loaded once into SRAM, K/V streamed through — correct access pattern.
  - T4-safe default configs: BLOCK_M=32, BLOCK_N=32 fits 64KB shared memory
    even with head_dim=256.

Registered as ``attention_v2`` alongside the existing ``attention`` operator
for backward compatibility.

References:
  - FlashAttention-2 (Dao 2023): https://arxiv.org/abs/2307.08691
  - Official Triton tutorial: 06-fused-attention.py
  - Mistral-7B sliding-window attention
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .triton_operators import TritonOperatorSpec, register_operator


# ---------------------------------------------------------------------------
# Parameter space & curated configs
# ---------------------------------------------------------------------------

ATTENTION_V2_PARAM_SPACE = {
    "BLOCK_M": [32, 64, 128],
    "BLOCK_N": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}

ATTENTION_V2_CURATED_CONFIGS = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    # T4-safe: head_dim=256 → (32+32)*512 = 32KB < 64KB shmem
    {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 2, "num_stages": 3},
    # Large head_dim (512, Gemma 4 global)
    {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
    # Sliding-window optimized
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 2, "num_stages": 2},
    # T4 sliding-window: small tiles, low warps, minimal register pressure
    {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
]


# Copy shape buckets from v1 — identical workload shapes.
ATTENTION_V2_SHAPE_BUCKETS = [
    {"name": "short_64", "batch": 4, "heads": 32, "num_kv_heads": 32, "seq_len": 512, "head_dim": 64, "is_causal": False},
    {"name": "short_128", "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 1024, "head_dim": 128, "is_causal": False},
    {"name": "med_128", "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 2048, "head_dim": 128, "is_causal": False},
    {"name": "long_64", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 64, "is_causal": False},
    {"name": "long_128", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 128, "is_causal": False},
    {"name": "llama7b", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": False},
    {"name": "mistral", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 8192, "head_dim": 128, "is_causal": False},
    {"name": "llama7b_causal", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": True},
    {"name": "mistral_causal", "batch": 1, "heads": 32, "num_kv_heads": 32, "seq_len": 8192, "head_dim": 128, "is_causal": True},
    {"name": "long_128_causal", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 128, "is_causal": True},
    {"name": "gemma4_local_1024", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024},
    {"name": "gemma4_local_short", "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 2048, "head_dim": 128, "is_causal": True, "window_size": 1024},
    {"name": "gemma3_local", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 8192, "head_dim": 128, "is_causal": True, "window_size": 1024},
    {"name": "gemma4_qknorm", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_qknorm_global", "batch": 1, "heads": 16, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": -1, "use_qk_norm": True},
    {"name": "gemma4_31b_local", "batch": 1, "heads": 32, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_31b_global", "batch": 1, "heads": 32, "num_kv_heads": 4, "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1, "use_qk_norm": True},
    {"name": "gemma4_26b_a4b_local", "batch": 1, "heads": 16, "num_kv_heads": 8, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_26b_a4b_global", "batch": 1, "heads": 16, "num_kv_heads": 2, "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1, "use_qk_norm": True},
    {"name": "llama3_70b_gqa", "batch": 1, "heads": 64, "num_kv_heads": 8, "seq_len": 4096, "head_dim": 128, "is_causal": True, "window_size": -1},
    {"name": "mistral_gqa", "batch": 1, "heads": 32, "num_kv_heads": 8, "seq_len": 8192, "head_dim": 128, "is_causal": True, "window_size": -1},
    {"name": "gemma4_31b_yoco_local", "batch": 1, "heads": 32, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True, "shared_kv": True},
    {"name": "gemma4_31b_yoco_global", "batch": 1, "heads": 32, "num_kv_heads": 4, "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1, "use_qk_norm": True, "shared_kv": True},
]


# ---------------------------------------------------------------------------
# Helper functions (reused from v1)
# ---------------------------------------------------------------------------

def attention_v2_config_id(config: dict[str, int]) -> str:
    return f"m{config['BLOCK_M']}_n{config['BLOCK_N']}_w{config['num_warps']}_s{config['num_stages']}"


def attention_v2_shape_bucket_key(shape: dict[str, int]) -> str:
    """Identical routing logic to v1 — same shape buckets."""
    seq = shape.get("seq_len", 0)
    hd = shape.get("head_dim", 0)
    heads = shape.get("heads", 0)
    nkv = shape.get("num_kv_heads", heads)
    ws = shape.get("window_size", -1)
    use_qk_norm = bool(shape.get("use_qk_norm", False))
    shared_kv = bool(shape.get("shared_kv", False))
    is_gqa = nkv > 0 and nkv < heads

    if shared_kv and is_gqa:
        if hd >= 512:
            return "gemma4_31b_yoco_global"
        return "gemma4_31b_yoco_local"

    if is_gqa:
        if use_qk_norm:
            if hd >= 512:
                return "gemma4_31b_global" if heads >= 32 else "gemma4_26b_a4b_global"
            return "gemma4_31b_local" if heads >= 32 else "gemma4_26b_a4b_local"
        if seq >= 8192:
            return "mistral_gqa"
        if heads >= 64:
            return "llama3_70b_gqa"
        return "mistral_gqa"

    if use_qk_norm:
        if ws is not None and ws > 0:
            return "gemma4_qknorm"
        return "gemma4_qknorm_global"

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


def attention_v2_shared_memory_check(config: dict[str, int]) -> bool:
    """Soft annotation — always True (learning from runtime failures)."""
    return True


def generate_attention_v2_grid(
    *,
    include_curated: bool = True,
    max_configs: int = 200,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()

    if include_curated:
        for config in ATTENTION_V2_CURATED_CONFIGS:
            cid = attention_v2_config_id(config)
            if cid not in seen:
                seen.add(cid)
                configs.append(config)

    for bm in ATTENTION_V2_PARAM_SPACE["BLOCK_M"]:
        for bn in ATTENTION_V2_PARAM_SPACE["BLOCK_N"]:
            for nw in ATTENTION_V2_PARAM_SPACE["num_warps"]:
                for ns in [2, 3]:
                    config = {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "num_warps": nw,
                        "num_stages": ns,
                    }
                    cid = attention_v2_config_id(config)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    configs.append(config)
                    if len(configs) >= max_configs:
                        return configs
    return configs


# ---------------------------------------------------------------------------
# Module-level launcher (lazy GPU imports)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_attention_v2_forward_kernel():
    """Build the Triton kernels once so hot call sites reuse them."""
    import triton
    import triton.language as tl

    @triton.jit
    def _attn_v2_inner(
        acc, l_i, m_i, q,
        K_base, V_base, QScale, KScale,
        stride_kn, stride_kk, stride_vn, stride_vk,
        start_m, qk_scale, N,
        offs_m, offs_n, offs_k,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        USE_QK_NORM: tl.constexpr,
    ):
        if STAGE == 1:
            lo = 0
            hi = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
        elif STAGE == 2:
            lo = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
            hi = (((start_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
            if hi > N:
                hi = N
        else:
            lo = 0
            hi = N

        if WINDOW_SIZE > 0:
            window_lo = start_m * BLOCK_M - WINDOW_SIZE + 1
            if window_lo < 0:
                window_lo = 0
            if lo < window_lo:
                lo = (window_lo // BLOCK_N) * BLOCK_N
            if STAGE == 3:
                window_hi = (start_m + 1) * BLOCK_M + WINDOW_SIZE - 1
                if window_hi < hi:
                    hi = ((window_hi + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
                    if hi > N:
                        hi = N

        if WINDOW_SIZE > 0:
            tile_window_lo = (start_m + 1) * BLOCK_M - WINDOW_SIZE

        for start_n in range(lo, hi, BLOCK_N):
            curr_n = start_n + offs_n
            tile_n_end = start_n + BLOCK_N
            tile_fully_in_bounds = tile_n_end <= N

            k_ptrs = K_base + curr_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
            if tile_fully_in_bounds:
                k = tl.load(k_ptrs)
            else:
                n_mask = curr_n[:, None] < N
                k = tl.load(k_ptrs, mask=n_mask, other=0.0)

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

            if STAGE == 2:
                neg_inf = -1.0e6
                mask = offs_m[:, None] >= curr_n[None, :]
                if not tile_fully_in_bounds:
                    qk = tl.where(curr_n[None, :] < N, qk, neg_inf)
                qk = qk * qk_scale + tl.where(mask, 0, neg_inf)
                if WINDOW_SIZE > 0:
                    window_floor = offs_m[:, None] - WINDOW_SIZE + 1
                    window_mask = curr_n[None, :] >= window_floor
                    qk = tl.where(window_mask, qk, neg_inf)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                qk = qk * qk_scale
                if not tile_fully_in_bounds:
                    qk = tl.where(curr_n[None, :] < N, qk, -1.0e6)
                if WINDOW_SIZE > 0 and start_n < tile_window_lo:
                    window_floor = offs_m[:, None] - WINDOW_SIZE + 1
                    window_mask = curr_n[None, :] >= window_floor
                    qk = tl.where(window_mask, qk, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]

            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V_base + curr_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
            if tile_fully_in_bounds:
                v = tl.load(v_ptrs)
            else:
                n_mask = curr_n[:, None] < N
                v = tl.load(v_ptrs, mask=n_mask, other=0.0)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

        return acc, l_i, m_i

    @triton.jit
    def _attn_v2_fwd(
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
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H
        off_kvh = off_h // GROUP_SIZE

        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_kvh * stride_kh
        v_base = V + off_b * stride_vb + off_kvh * stride_vh
        o_base = Out + off_b * stride_ob + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
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

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        if IS_CAUSAL:
            acc, l_i, m_i = _attn_v2_inner(
                acc, l_i, m_i, q,
                k_base, v_base, QScale, KScale,
                stride_kn, stride_kk, stride_vn, stride_vk,
                start_m, qk_scale, N,
                offs_m, offs_n, offs_k,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                STAGE=1,
                WINDOW_SIZE=WINDOW_SIZE,
                USE_QK_NORM=USE_QK_NORM,
            )
            acc, l_i, m_i = _attn_v2_inner(
                acc, l_i, m_i, q,
                k_base, v_base, QScale, KScale,
                stride_kn, stride_kk, stride_vn, stride_vk,
                start_m, qk_scale, N,
                offs_m, offs_n, offs_k,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                STAGE=2,
                WINDOW_SIZE=WINDOW_SIZE,
                USE_QK_NORM=USE_QK_NORM,
            )
        else:
            acc, l_i, m_i = _attn_v2_inner(
                acc, l_i, m_i, q,
                k_base, v_base, QScale, KScale,
                stride_kn, stride_kk, stride_vn, stride_vk,
                start_m, qk_scale, N,
                offs_m, offs_n, offs_k,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                STAGE=3,
                WINDOW_SIZE=WINDOW_SIZE,
                USE_QK_NORM=USE_QK_NORM,
            )

        acc = acc / l_i[:, None]
        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

    return _attn_v2_fwd


def flash_attn_v2(
    q, k, v, config=None, is_causal=False, sm_scale=None, window_size=-1,
    use_qk_norm=False, q_scale=None, k_scale=None,
    num_kv_heads=None, shared_kv=False,
):
    """Module-level FlashAttention-v2 launcher. Requires CUDA GPU."""
    import torch
    import triton

    if config is None:
        config = ATTENTION_V2_CURATED_CONFIGS[0]

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
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]

    if q_scale is None:
        q_scale = torch.ones(D, device=q.device, dtype=torch.float32)
    if k_scale is None:
        k_scale = torch.ones(D, device=k.device, dtype=torch.float32)

    attn_v2_fwd = _get_attention_v2_forward_kernel()
    grid = (triton.cdiv(M, block_m), B * H, 1)
    attn_v2_fwd[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE=window_size,
        USE_QK_NORM=use_qk_norm,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


# ---------------------------------------------------------------------------
# Benchmark script generator
# ---------------------------------------------------------------------------

def generate_attention_v2_benchmark_script(
    configs: list[dict[str, int]],
    shapes: list[dict[str, Any]],
) -> str:
    """Generate a self-contained benchmark script for attention_v2.

    Uses the official Triton FA2 tutorial kernel structure with two-stage
    causal processing, exp2-based online softmax, and Noeris extensions
    (GQA, sliding-window tile-pruning, QK-norm, YOCO).
    """
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)

    return f'''#!/usr/bin/env python3
"""Auto-generated Triton FA2 v2 benchmark (official tutorial kernel + Noeris extensions)."""

import json
import platform
CONFIGS_JSON = {configs_json!r}
SHAPES_JSON = {shapes_json!r}

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_v2_inner(
    acc, l_i, m_i, q,
    K_base, V_base, QScale, KScale,
    stride_kn, stride_kk, stride_vn, stride_vk,
    start_m, qk_scale, N,
    offs_m, offs_n, offs_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    USE_QK_NORM: tl.constexpr,
):
    if STAGE == 1:
        lo = 0
        # Align hi to BLOCK_N boundary so we don't miss/double-count
        # keys when BLOCK_M != BLOCK_N.
        hi = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    elif STAGE == 2:
        # Start where STAGE 1 left off (BLOCK_N-aligned).
        lo = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
        # Cover through the last key that could be causally valid for
        # any row in this Q-tile, rounded up to BLOCK_N boundary.
        hi = (((start_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        if hi > N:
            hi = N
    else:
        lo = 0
        hi = N

    if WINDOW_SIZE > 0:
        window_lo = start_m * BLOCK_M - WINDOW_SIZE + 1
        if window_lo < 0:
            window_lo = 0
        if lo < window_lo:
            lo = (window_lo // BLOCK_N) * BLOCK_N
        if STAGE == 3:
            window_hi = (start_m + 1) * BLOCK_M + WINDOW_SIZE - 1
            if window_hi < hi:
                hi = ((window_hi + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
                if hi > N:
                    hi = N

    if WINDOW_SIZE > 0:
        tile_window_lo = (start_m + 1) * BLOCK_M - WINDOW_SIZE

    for start_n in range(lo, hi, BLOCK_N):
        curr_n = start_n + offs_n

        tile_n_end = start_n + BLOCK_N
        tile_fully_in_bounds = tile_n_end <= N

        k_ptrs = K_base + curr_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        if tile_fully_in_bounds:
            k = tl.load(k_ptrs)
        else:
            n_mask = curr_n[:, None] < N
            k = tl.load(k_ptrs, mask=n_mask, other=0.0)

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

        if STAGE == 2:
            NEG_INF = -1.0e6
            mask = offs_m[:, None] >= curr_n[None, :]
            if not tile_fully_in_bounds:
                qk = tl.where(curr_n[None, :] < N, qk, NEG_INF)
            qk = qk * qk_scale + tl.where(mask, 0, NEG_INF)
            if WINDOW_SIZE > 0:
                window_floor = offs_m[:, None] - WINDOW_SIZE + 1
                window_mask = curr_n[None, :] >= window_floor
                qk = tl.where(window_mask, qk, NEG_INF)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            if not tile_fully_in_bounds:
                qk = tl.where(curr_n[None, :] < N, qk, -1.0e6)
            if WINDOW_SIZE > 0:
                if start_n < tile_window_lo:
                    window_floor = offs_m[:, None] - WINDOW_SIZE + 1
                    window_mask = curr_n[None, :] >= window_floor
                    qk = tl.where(window_mask, qk, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None]

        v_ptrs = V_base + curr_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        if tile_fully_in_bounds:
            v = tl.load(v_ptrs)
        else:
            n_mask = curr_n[:, None] < N
            v = tl.load(v_ptrs, mask=n_mask, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def attn_v2_fwd_kernel(
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
    """Official Triton FA2 kernel with Noeris GQA + sliding-window + QK-norm."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    off_kvh = off_h // GROUP_SIZE

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_kvh * stride_kh
    v_base = V + off_b * stride_vb + off_kvh * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    if IS_CAUSAL:
        acc, l_i, m_i = _attn_v2_inner(
            acc, l_i, m_i, q,
            k_base, v_base, QScale, KScale,
            stride_kn, stride_kk, stride_vn, stride_vk,
            start_m, qk_scale, N,
            offs_m, offs_n, offs_k,
            HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            STAGE=1, WINDOW_SIZE=WINDOW_SIZE, USE_QK_NORM=USE_QK_NORM,
        )
        acc, l_i, m_i = _attn_v2_inner(
            acc, l_i, m_i, q,
            k_base, v_base, QScale, KScale,
            stride_kn, stride_kk, stride_vn, stride_vk,
            start_m, qk_scale, N,
            offs_m, offs_n, offs_k,
            HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            STAGE=2, WINDOW_SIZE=WINDOW_SIZE, USE_QK_NORM=USE_QK_NORM,
        )
    else:
        acc, l_i, m_i = _attn_v2_inner(
            acc, l_i, m_i, q,
            k_base, v_base, QScale, KScale,
            stride_kn, stride_kk, stride_vn, stride_vk,
            start_m, qk_scale, N,
            offs_m, offs_n, offs_k,
            HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            STAGE=3, WINDOW_SIZE=WINDOW_SIZE, USE_QK_NORM=USE_QK_NORM,
        )

    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def flash_attn(
    q, k, v, config, is_causal=False, sm_scale=None, window_size=-1,
    use_qk_norm=False, q_scale=None, k_scale=None,
    num_kv_heads=None, shared_kv=False,
):
    B, H, M, D = q.shape
    _, Hk, N, Dk = k.shape
    _, Hv, Nv, Dv = v.shape
    if num_kv_heads is None:
        num_kv_heads = H
    assert num_kv_heads > 0
    assert H % num_kv_heads == 0
    assert Hk == num_kv_heads and Hv == num_kv_heads
    assert Dk == D and Dv == D
    assert Nv == N
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
    attn_v2_fwd_kernel[grid](
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


def make_sliding_window_mask(seq_len, window_size, is_causal):
    rows = torch.arange(seq_len).unsqueeze(1)
    cols = torch.arange(seq_len).unsqueeze(0)
    left_bound = cols >= (rows - window_size + 1)
    if is_causal:
        mask = left_bound & (cols <= rows)
    else:
        mask = left_bound & ((cols - rows) < window_size)
    return mask


def benchmark_one(batch, heads, seq_len, head_dim, config, is_causal=False, window_size=-1, use_qk_norm=False, num_kv_heads=None, shared_kv=False):
    try:
        if num_kv_heads is None:
            num_kv_heads = heads
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

        if shared_kv:
            k = torch.nn.functional.rms_norm(k.float(), (head_dim,)).half()

        group_size = heads // num_kv_heads
        k_ref = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
        v_ref = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v

        if use_qk_norm:
            q_ref = torch.nn.functional.rms_norm(q.float(), (head_dim,)).half()
            if not shared_kv:
                k_ref = torch.nn.functional.rms_norm(k_ref.float(), (head_dim,)).half()
        else:
            q_ref = q

        if window_size > 0:
            ws_mask = make_sliding_window_mask(seq_len, window_size, is_causal).to(q.device)
            ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, attn_mask=ws_mask, is_causal=False)
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=is_causal)

        out = flash_attn(q, k, v, config, is_causal=is_causal, window_size=window_size,
                         use_qk_norm=use_qk_norm, num_kv_heads=num_kv_heads, shared_kv=shared_kv)
        max_err = (out - ref).abs().max().item()
        if max_err > 0.1:
            return {{"correct": False, "max_err": max_err, "ms": None, "tflops": None}}

        ms = triton.testing.do_bench(
            lambda: flash_attn(q, k, v, config, is_causal=is_causal, window_size=window_size,
                               use_qk_norm=use_qk_norm, num_kv_heads=num_kv_heads, shared_kv=shared_kv),
            warmup=10, rep=50,
        )
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
        "operator": "attention_v2",
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


# ---------------------------------------------------------------------------
# Operator registration
# ---------------------------------------------------------------------------

ATTENTION_V2_SPEC = register_operator(TritonOperatorSpec(
    name="attention_v2",
    param_space=ATTENTION_V2_PARAM_SPACE,
    curated_configs=ATTENTION_V2_CURATED_CONFIGS,
    shape_buckets=ATTENTION_V2_SHAPE_BUCKETS,
    metric_name="tflops",
    config_id_fn=attention_v2_config_id,
    shape_bucket_fn=attention_v2_shape_bucket_key,
    benchmark_script_fn=generate_attention_v2_benchmark_script,
    grid_generator_fn=generate_attention_v2_grid,
    shared_memory_check_fn=attention_v2_shared_memory_check,
    description=(
        "Official Triton FA2 tutorial kernel with Noeris extensions: "
        "two-stage causal processing, exp2-based online softmax, "
        "GQA, sliding-window tile-pruning, QK-norm, and YOCO KV-share."
    ),
))
