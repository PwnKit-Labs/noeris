"""FP8 config-space helpers for matmul and grouped-gemm lanes."""

from __future__ import annotations

from itertools import product


def build_fp8_matmul_config_space() -> list[dict[str, int]]:
    block_m = [64, 128, 256]
    block_n = [64, 128]
    block_k = [32, 64, 128]
    num_warps = [4, 8]
    num_stages = [2, 3, 4]
    split_k = [1, 2, 4]
    out: list[dict[str, int]] = []
    for bm, bn, bk, nw, ns, sk in product(block_m, block_n, block_k, num_warps, num_stages, split_k):
        if bm * bn > 32768:
            continue
        out.append(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk,
                "num_warps": nw,
                "num_stages": ns,
                "SPLIT_K": sk,
            }
        )
    return out


def build_fp8_grouped_gemm_config_space() -> list[dict[str, int]]:
    group_m = [4, 8, 16]
    block_m = [64, 128]
    block_n = [64, 128]
    block_k = [32, 64]
    num_warps = [4, 8]
    num_stages = [2, 3]
    out: list[dict[str, int]] = []
    for gm, bm, bn, bk, nw, ns in product(group_m, block_m, block_n, block_k, num_warps, num_stages):
        out.append(
            {
                "GROUP_SIZE_M": gm,
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk,
                "num_warps": nw,
                "num_stages": ns,
            }
        )
    return out
