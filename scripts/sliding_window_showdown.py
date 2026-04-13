#!/usr/bin/env python3
"""Sliding-window attention showdown: Noeris Triton tile-pruning vs SDPA/cuDNN.

Targets shapes where Noeris has maximum advantage: long sequences, small windows
(high tile-skip ratio), small head_dim (low register pressure on T4).

cuDNN FlashAttention (via SDPA) always computes the full causal triangle -- it
cannot skip tiles.  Noeris's sliding-window tile-pruning skips ~75-97% of tiles
depending on (N, W), giving a potential wall-clock win.

Usage (Kaggle T4 / Colab):
    !git clone https://github.com/PwnKit-Labs/noeris && cd noeris
    !pip install -e . numpy scikit-learn -q
    !python scripts/sliding_window_showdown.py

Outputs a table + JSON results file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch

print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: No GPU available. Change runtime to T4 GPU.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")

import triton
import triton.language as tl

print(f"Triton {triton.__version__}")

# ============================================================================
# Shapes to test  (long seq + small window = high tile skip ratio)
# ============================================================================

SHAPES = [
    # Long seq, tiny window -- maximum tile skip
    {"name": "n8192_w128",       "B": 1, "H": 8,  "S": 8192, "D": 64,  "W": 128},
    {"name": "n8192_w256",       "B": 1, "H": 8,  "S": 8192, "D": 64,  "W": 256},
    {"name": "n4096_w128",       "B": 1, "H": 8,  "S": 4096, "D": 64,  "W": 128},
    {"name": "n4096_w256",       "B": 1, "H": 8,  "S": 4096, "D": 64,  "W": 256},
    # head_dim=128 variants
    {"name": "n4096_w256_d128",  "B": 1, "H": 8,  "S": 4096, "D": 128, "W": 256},
    {"name": "n8192_w256_d128",  "B": 1, "H": 4,  "S": 8192, "D": 128, "W": 256},
    # More heads for parallelism
    {"name": "n4096_w128_h32",   "B": 1, "H": 32, "S": 4096, "D": 64,  "W": 128},
    {"name": "n8192_w128_h16",   "B": 1, "H": 16, "S": 8192, "D": 64,  "W": 128},
]

# ============================================================================
# Triton configs to sweep (T4-aggressive: small tiles, low warps)
# ============================================================================

CONFIGS = [
    {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 2, "num_stages": 1},
    {"BLOCK_M": 16, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
    # Also try some from curated list
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
]

# ============================================================================
# Timing helper
# ============================================================================

def cuda_event_timer(fn, warmup=5, trials=20):
    """Time a function using CUDA events. Returns median ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


# ============================================================================
# Noeris Triton kernel (inline from triton_attention_v2.py)
# ============================================================================

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
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
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
        tile_window_lo = start_m * BLOCK_M - WINDOW_SIZE + 1

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


# ============================================================================
# Noeris launcher
# ============================================================================

def noeris_sliding_window_attn(q, k, v, config, window_size, is_causal=True):
    """Launch Noeris Triton sliding-window attention."""
    B, H, M, D = q.shape
    _, Hk, N, Dk = k.shape
    num_kv_heads = Hk
    group_size = H // num_kv_heads
    sm_scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    q_scale = torch.ones(D, device=q.device, dtype=torch.float32)
    k_scale = torch.ones(D, device=q.device, dtype=torch.float32)

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
        USE_QK_NORM=False,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


# ============================================================================
# SDPA baselines
# ============================================================================

def make_sliding_window_mask(seq_len, window_size, device):
    """Causal sliding-window boolean mask: j >= i - W + 1 AND j <= i."""
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (cols >= (rows - window_size + 1)) & (cols <= rows)
    return mask


def sdpa_full_causal(q, k, v):
    """SDPA with full causal mask (no window) -- what most engines do."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )


def sdpa_with_window_mask(q, k, v, window_size):
    """SDPA with explicit sliding-window boolean mask."""
    S = q.shape[2]
    mask = make_sliding_window_mask(S, window_size, q.device)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False
    )


# ============================================================================
# Correctness check
# ============================================================================

def check_correctness(q, k, v, config, window_size):
    """Verify Noeris output matches reference within tolerance."""
    ref = sdpa_with_window_mask(q, k, v, window_size)
    out = noeris_sliding_window_attn(q, k, v, config, window_size, is_causal=True)
    max_err = (out - ref).abs().max().item()
    return max_err


# ============================================================================
# Main benchmark
# ============================================================================

def config_id(cfg):
    return f"m{cfg['BLOCK_M']}_n{cfg['BLOCK_N']}_w{cfg['num_warps']}_s{cfg['num_stages']}"


def estimate_skip_ratio(S, W, block_size=32):
    """Estimate fraction of tiles skipped with causal + sliding window."""
    total_causal_tiles = 0
    window_tiles = 0
    num_m_tiles = (S + block_size - 1) // block_size
    num_n_tiles = (S + block_size - 1) // block_size
    for m_tile in range(num_m_tiles):
        q_row = m_tile * block_size
        # Causal: keys 0..q_row => tiles 0..m_tile
        causal_tiles = m_tile + 1
        total_causal_tiles += causal_tiles
        # Window: keys max(0, q_row - W + 1)..q_row
        lo_key = max(0, q_row - W + 1)
        lo_tile = lo_key // block_size
        window_tiles += (m_tile - lo_tile + 1)
    if total_causal_tiles == 0:
        return 0.0
    return 1.0 - window_tiles / total_causal_tiles


def main():
    results_all = []
    print()
    print("=" * 90)
    print("  SLIDING-WINDOW ATTENTION SHOWDOWN: Noeris Triton vs SDPA/cuDNN")
    print("=" * 90)
    print()

    for shape in SHAPES:
        name = shape["name"]
        B, H, S, D, W = shape["B"], shape["H"], shape["S"], shape["D"], shape["W"]
        skip_pct = estimate_skip_ratio(S, W) * 100

        print(f"\n{'─' * 80}")
        print(f"  Shape: {name}  (B={B}, H={H}, S={S}, D={D}, W={W})")
        print(f"  Estimated tile skip ratio: {skip_pct:.1f}%")
        print(f"{'─' * 80}")

        # Allocate tensors
        q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)
        k = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)
        v = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)

        # ---- Baseline A: SDPA full causal (no window) ----
        try:
            ms_sdpa_causal = cuda_event_timer(
                lambda: sdpa_full_causal(q, k, v), warmup=5, trials=20
            )
        except Exception as e:
            print(f"  SDPA full causal FAILED: {e}")
            ms_sdpa_causal = None

        # ---- Baseline B: SDPA with explicit window mask ----
        try:
            ms_sdpa_mask = cuda_event_timer(
                lambda: sdpa_with_window_mask(q, k, v, W), warmup=5, trials=20
            )
        except Exception as e:
            print(f"  SDPA window mask FAILED: {e}")
            ms_sdpa_mask = None

        if ms_sdpa_causal is not None:
            print(f"  SDPA full causal:        {ms_sdpa_causal:8.3f} ms")
        if ms_sdpa_mask is not None:
            print(f"  SDPA explicit window:    {ms_sdpa_mask:8.3f} ms")
        print()

        # ---- Noeris: sweep configs ----
        best_noeris_ms = float("inf")
        best_noeris_cfg = None
        noeris_results = []

        for cfg in CONFIGS:
            cid = config_id(cfg)
            try:
                # Correctness check first (only on first run per config)
                max_err = check_correctness(q, k, v, cfg, W)
                if max_err > 0.1:
                    print(f"    {cid}: INCORRECT (max_err={max_err:.4f})")
                    noeris_results.append({
                        "config": cid, "ms": None, "correct": False,
                        "max_err": round(max_err, 5),
                    })
                    continue

                ms = cuda_event_timer(
                    lambda cfg=cfg: noeris_sliding_window_attn(q, k, v, cfg, W, is_causal=True),
                    warmup=5, trials=20,
                )
                noeris_results.append({
                    "config": cid, "ms": round(ms, 4), "correct": True,
                    "max_err": round(max_err, 5),
                })
                print(f"    Noeris {cid}: {ms:8.3f} ms  (err={max_err:.5f})")
                if ms < best_noeris_ms:
                    best_noeris_ms = ms
                    best_noeris_cfg = cid

            except Exception as e:
                print(f"    Noeris {cid}: FAILED ({e})")
                noeris_results.append({
                    "config": cid, "ms": None, "correct": False,
                    "error": str(e)[:200],
                })

        # ---- Summary for this shape ----
        print()
        best_sdpa = None
        best_sdpa_label = None
        if ms_sdpa_causal is not None:
            best_sdpa = ms_sdpa_causal
            best_sdpa_label = "SDPA-causal"
        if ms_sdpa_mask is not None and (best_sdpa is None or ms_sdpa_mask < best_sdpa):
            best_sdpa = ms_sdpa_mask
            best_sdpa_label = "SDPA-mask"

        wins = best_noeris_ms < best_sdpa if (best_sdpa and best_noeris_ms < float("inf")) else False
        if best_sdpa and best_noeris_ms < float("inf"):
            speedup = best_sdpa / best_noeris_ms
            status = ">>> NOERIS WINS <<<" if wins else "SDPA faster"
            print(f"  BEST Noeris ({best_noeris_cfg}): {best_noeris_ms:.3f} ms")
            print(f"  BEST SDPA   ({best_sdpa_label}):  {best_sdpa:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x  {status}")
        elif best_noeris_ms < float("inf"):
            print(f"  BEST Noeris ({best_noeris_cfg}): {best_noeris_ms:.3f} ms")
            print(f"  SDPA: unavailable")
        else:
            print(f"  All Noeris configs failed for this shape.")

        results_all.append({
            "shape": name,
            "B": B, "H": H, "S": S, "D": D, "W": W,
            "skip_ratio_pct": round(skip_pct, 1),
            "sdpa_causal_ms": round(ms_sdpa_causal, 4) if ms_sdpa_causal else None,
            "sdpa_mask_ms": round(ms_sdpa_mask, 4) if ms_sdpa_mask else None,
            "best_noeris_ms": round(best_noeris_ms, 4) if best_noeris_ms < float("inf") else None,
            "best_noeris_config": best_noeris_cfg,
            "noeris_wins": wins,
            "speedup_vs_sdpa": round(best_sdpa / best_noeris_ms, 3) if (best_sdpa and best_noeris_ms < float("inf")) else None,
            "noeris_configs": noeris_results,
        })

        # Free memory between shapes
        del q, k, v
        torch.cuda.empty_cache()

    # ============================================================================
    # Final summary table
    # ============================================================================
    print("\n")
    print("=" * 100)
    print("  FINAL RESULTS TABLE")
    print("=" * 100)
    hdr = f"{'Shape':<22} {'Skip%':>5} {'SDPA-caus ms':>12} {'SDPA-mask ms':>12} {'Noeris ms':>10} {'Config':<18} {'Speedup':>8} {'Winner':>10}"
    print(hdr)
    print("-" * 100)

    any_win = False
    for r in results_all:
        sdpa_c = f"{r['sdpa_causal_ms']:.3f}" if r['sdpa_causal_ms'] else "N/A"
        sdpa_m = f"{r['sdpa_mask_ms']:.3f}" if r['sdpa_mask_ms'] else "N/A"
        noeris = f"{r['best_noeris_ms']:.3f}" if r['best_noeris_ms'] else "FAIL"
        cfg = r['best_noeris_config'] or "N/A"
        spd = f"{r['speedup_vs_sdpa']:.2f}x" if r['speedup_vs_sdpa'] else "N/A"
        winner = "NOERIS" if r['noeris_wins'] else "SDPA"
        if r['noeris_wins']:
            any_win = True
        print(f"{r['shape']:<22} {r['skip_ratio_pct']:>5.1f} {sdpa_c:>12} {sdpa_m:>12} {noeris:>10} {cfg:<18} {spd:>8} {winner:>10}")

    print("-" * 100)
    if any_win:
        print("\n  *** NOERIS BEATS SDPA on at least one shape! ***")
    else:
        print("\n  SDPA wins all shapes (tile-pruning overhead exceeds skip savings).")
    print()

    # ============================================================================
    # Save JSON
    # ============================================================================
    output = {
        "benchmark": "sliding_window_showdown",
        "gpu": GPU_NAME,
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda or "unknown",
        "any_noeris_win": any_win,
        "results": results_all,
    }

    out_path = REPO / "results" / "sliding_window_showdown.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
