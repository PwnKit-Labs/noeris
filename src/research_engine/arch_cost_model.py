"""Kernel-aware architecture cost model.

Predicts end-to-end transformer layer latency from architecture config
using measured kernel throughputs.  A single decoder layer has ~9 kernel
calls; we estimate bytes or FLOPs for each and divide by the measured
bandwidth/throughput to get time.

Measured throughputs come from .noeris/cost-model-training.json (A100).
For kernels without direct measurements we fall back to hardware-spec
peak values with empirical efficiency factors.

Usage::

    model = ArchitectureCostModel()
    result = model.predict_layer_ms({
        "hidden_dim": 1536, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 6144, "seq_len": 2048,
        "batch_size": 1, "use_qk_norm": True, "window_size": 1024,
    })
    print(result["total_ms"], result["bottleneck"])
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Measured throughputs (A100-80GB, bf16/fp16)
# ---------------------------------------------------------------------------
# RMSNorm: memory-bound.  "tflops" in the DB is really effective GB/s
# because each element is 1 FLOP but 2 bytes read + 2 bytes written.
# Measured best: 1375 "tflops" for llama_7b shape (4096x4096).
# Convert: tflops * elem_bytes * 2 (r+w) ~ actual bandwidth.
# But simpler: we model as bytes / bandwidth directly.
#
# From measurements, rmsnorm achieves ~1200-1400 GB/s effective for large
# shapes on A100 (2039 GB/s peak HBM).  We'll use shape-dependent values.

# Hardcoded bandwidth / throughput values by GPU.
# These are "effective" values derived from our Triton kernel measurements.

HARDWARE_PROFILES = {
    "a100": {
        "hbm_bandwidth_gbps": 2039,      # peak spec
        "rmsnorm_gbps": 1300,             # measured effective (large shapes)
        "rmsnorm_gbps_small": 600,        # measured effective (small shapes, <2048 hidden)
        "qk_norm_rope_gbps": 900,         # fused kernel, measured
        "qk_norm_rope_separate_gbps": 500,  # unfused fallback
        "geglu_gbps": 1200,               # measured effective
        "matmul_tflops": 200,             # bf16 tensor core, measured effective
        "matmul_tflops_small": 50,        # small shapes (<1024 in any dim)
        "attention_tflops": 155,          # FlashAttention effective
        "residual_add_gbps": 1800,        # trivial elementwise, near peak
    },
    "t4": {
        "hbm_bandwidth_gbps": 320,
        "rmsnorm_gbps": 150,
        "rmsnorm_gbps_small": 80,
        "qk_norm_rope_gbps": 80,
        "qk_norm_rope_separate_gbps": 50,
        "geglu_gbps": 150,
        "matmul_tflops": 30,             # fp16 tensor core
        "matmul_tflops_small": 8,
        "attention_tflops": 25,
        "residual_add_gbps": 280,
    },
    "h100": {
        "hbm_bandwidth_gbps": 3350,
        "rmsnorm_gbps": 2400,
        "rmsnorm_gbps_small": 1000,
        "qk_norm_rope_gbps": 1600,
        "qk_norm_rope_separate_gbps": 900,
        "geglu_gbps": 2200,
        "matmul_tflops": 500,
        "matmul_tflops_small": 120,
        "attention_tflops": 400,
        "residual_add_gbps": 3000,
    },
}

# Triton tile sizes that get good occupancy.  Dims that are multiples of
# these avoid wasted work in the tail tile.
TILE_ALIGNED_SIZES = [64, 128, 256]
MATMUL_TILE_SIZE = 128  # typical Triton BLOCK_M/N


def _is_tile_aligned(dim: int, tile: int = MATMUL_TILE_SIZE) -> bool:
    return dim % tile == 0


def _tile_efficiency(dim: int, tile: int = MATMUL_TILE_SIZE) -> float:
    """Fraction of useful work in the last tile row/col.

    E.g. dim=4096, tile=128 -> 1.0 (perfect).
         dim=4000, tile=128 -> last tile is 32/128 = 0.25 partial.
    Overall efficiency = (full_tiles * tile + remainder) / (ceil_tiles * tile).
    """
    n_tiles = math.ceil(dim / tile)
    return dim / (n_tiles * tile)


class ArchitectureCostModel:
    """Predicts layer latency from architecture config using real kernel data."""

    def __init__(self, hardware: str = "a100"):
        if hardware not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware {hardware!r}, choose from {list(HARDWARE_PROFILES)}")
        self.hw = HARDWARE_PROFILES[hardware]
        self.hardware = hardware

    # ------------------------------------------------------------------
    # Per-kernel latency estimators
    # ------------------------------------------------------------------

    def _rmsnorm_ms(self, B: int, S: int, D: int) -> float:
        """RMSNorm: read x, write y, read/write weight (small). ~2*B*S*D bytes."""
        nbytes = 2 * B * S * D * 2  # 2 for read+write, 2 for bf16
        bw = self.hw["rmsnorm_gbps"] if D >= 2048 else self.hw["rmsnorm_gbps_small"]
        return nbytes / (bw * 1e6)  # GB/s -> bytes/ms

    def _qkv_matmul_ms(self, B: int, S: int, D: int, H: int, Hkv: int, Dh: int) -> float:
        """QKV projection: (B*S, D) @ (D, (H+2*Hkv)*Dh)."""
        M = B * S
        K = D
        N = (H + 2 * Hkv) * Dh
        flops = 2 * M * N * K  # matmul FLOPs
        # Tile efficiency penalty
        eff_m = _tile_efficiency(M)
        eff_n = _tile_efficiency(N)
        eff_k = _tile_efficiency(K)
        eff = eff_m * eff_n  # K dimension is reduced, less sensitive
        is_small = min(M, N, K) < 1024
        tflops = self.hw["matmul_tflops_small"] if is_small else self.hw["matmul_tflops"]
        return flops / (tflops * eff * 1e9)  # TFLOPS -> FLOPS/ms

    def _qk_norm_rope_ms(self, B: int, S: int, H: int, Hkv: int, Dh: int,
                          fused: bool = True) -> float:
        """QK-RMSNorm + RoPE: touches Q and K tensors."""
        # Q: B*H*S*Dh, K: B*Hkv*S*Dh, each read+write -> *2
        nbytes = 2 * B * (H + Hkv) * S * Dh * 2  # 2 for r+w, 2 for bf16
        bw = self.hw["qk_norm_rope_gbps"] if fused else self.hw["qk_norm_rope_separate_gbps"]
        return nbytes / (bw * 1e6)

    def _attention_ms(self, B: int, S: int, H: int, Dh: int,
                       window_size: int | None = None) -> float:
        """SDPA (FlashAttention): O(B*H*S^2*Dh) FLOPs for full attention."""
        # With sliding window, effective S_kv = min(S, window_size)
        S_kv = min(S, window_size) if window_size and window_size > 0 else S
        # 2 * B * H * S * S_kv * Dh (Q@K^T + attn@V)
        flops = 2 * 2 * B * H * S * S_kv * Dh
        return flops / (self.hw["attention_tflops"] * 1e9)

    def _output_proj_ms(self, B: int, S: int, D: int, H: int, Dh: int) -> float:
        """Output projection: (B*S, H*Dh) @ (H*Dh, D)."""
        M = B * S
        K = H * Dh
        N = D
        flops = 2 * M * N * K
        eff = _tile_efficiency(M) * _tile_efficiency(N)
        is_small = min(M, N, K) < 1024
        tflops = self.hw["matmul_tflops_small"] if is_small else self.hw["matmul_tflops"]
        return flops / (tflops * eff * 1e9)

    def _residual_add_ms(self, B: int, S: int, D: int) -> float:
        """Residual add: read two tensors, write one. 3*B*S*D*2 bytes."""
        nbytes = 3 * B * S * D * 2
        return nbytes / (self.hw["residual_add_gbps"] * 1e6)

    def _geglu_mlp_ms(self, B: int, S: int, D: int, ffn_dim: int) -> float:
        """GeGLU MLP: gate_up matmul + fused activation + down matmul.

        gate_up: (B*S, D) @ (D, 2*ffn_dim) -- gate and up in one matmul
        activation: fused GELU(gate) * up on 2*B*S*ffn_dim elements
        down: (B*S, ffn_dim) @ (ffn_dim, D)
        """
        M = B * S
        # gate_up matmul
        flops_up = 2 * M * (2 * ffn_dim) * D
        # down matmul
        flops_down = 2 * M * D * ffn_dim

        eff_up = _tile_efficiency(M) * _tile_efficiency(2 * ffn_dim)
        eff_down = _tile_efficiency(M) * _tile_efficiency(D)
        is_small = min(M, D, ffn_dim) < 1024

        tflops = self.hw["matmul_tflops_small"] if is_small else self.hw["matmul_tflops"]
        matmul_ms = (flops_up / (tflops * eff_up * 1e9) +
                     flops_down / (tflops * eff_down * 1e9))

        # Fused GeGLU activation: memory-bound, read gate+up, write output
        act_bytes = (2 + 1) * B * S * ffn_dim * 2  # read 2, write 1, bf16
        act_ms = act_bytes / (self.hw["geglu_gbps"] * 1e6)

        return matmul_ms + act_ms

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict_layer_ms(self, config: dict[str, Any]) -> dict[str, Any]:
        """Predict full decoder layer latency from architecture config.

        Parameters
        ----------
        config : dict
            Required keys: hidden_dim, num_heads, num_kv_heads, head_dim,
            ffn_dim, seq_len, batch_size.
            Optional: use_qk_norm (default True), window_size (default None
            for full attention).

        Returns
        -------
        dict with keys: total_ms, per_kernel (dict of kernel->ms),
        bottleneck (name of slowest kernel), tile_penalties (dict).
        """
        D = config["hidden_dim"]
        H = config["num_heads"]
        Hkv = config["num_kv_heads"]
        Dh = config["head_dim"]
        ffn = config["ffn_dim"]
        S = config["seq_len"]
        B = config.get("batch_size", 1)
        use_qk_norm = config.get("use_qk_norm", True)
        window = config.get("window_size", None)

        per_kernel = {}
        per_kernel["pre_attn_rmsnorm"] = self._rmsnorm_ms(B, S, D)
        per_kernel["qkv_projection"] = self._qkv_matmul_ms(B, S, D, H, Hkv, Dh)
        if use_qk_norm:
            per_kernel["qk_norm_rope"] = self._qk_norm_rope_ms(B, S, H, Hkv, Dh, fused=True)
        else:
            per_kernel["rope_only"] = self._qk_norm_rope_ms(B, S, H, Hkv, Dh, fused=False) * 0.5
        per_kernel["attention"] = self._attention_ms(B, S, H, Dh, window)
        per_kernel["output_projection"] = self._output_proj_ms(B, S, D, H, Dh)
        per_kernel["residual_add_1"] = self._residual_add_ms(B, S, D)
        per_kernel["pre_mlp_rmsnorm"] = self._rmsnorm_ms(B, S, D)
        per_kernel["geglu_mlp"] = self._geglu_mlp_ms(B, S, D, ffn)
        per_kernel["residual_add_2"] = self._residual_add_ms(B, S, D)

        total = sum(per_kernel.values())
        bottleneck = max(per_kernel, key=per_kernel.get)

        # Tile alignment analysis
        tile_penalties = {
            "hidden_dim": {
                "value": D,
                "aligned_128": _is_tile_aligned(D),
                "efficiency": _tile_efficiency(D),
            },
            "ffn_dim": {
                "value": ffn,
                "aligned_128": _is_tile_aligned(ffn),
                "efficiency": _tile_efficiency(ffn),
            },
            "head_dim": {
                "value": Dh,
                "aligned_128": _is_tile_aligned(Dh),
                "efficiency": _tile_efficiency(Dh),
            },
        }

        return {
            "total_ms": total,
            "per_kernel": per_kernel,
            "bottleneck": bottleneck,
            "tile_penalties": tile_penalties,
        }

    def sweep_dimension(self, base_config: dict[str, Any], dim_name: str,
                         values: list[int]) -> list[dict[str, Any]]:
        """Sweep a single dimension and return predictions for each value.

        Useful for finding kernel cliffs at tile boundaries.
        """
        results = []
        for v in values:
            cfg = {**base_config, dim_name: v}
            pred = self.predict_layer_ms(cfg)
            results.append({
                dim_name: v,
                "total_ms": pred["total_ms"],
                "bottleneck": pred["bottleneck"],
                "aligned_128": _is_tile_aligned(v),
            })
        return results
