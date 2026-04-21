"""Triton fused KV cache quantize-on-write kernel.

This kernel fuses per-row symmetric INT8 quantization into cache write:
- compute per-row absmax
- compute scale = absmax / 127 (or 1.0 for all-zero row)
- quantize and store INT8 values
- store per-row scale

Intended usage: quantize K/V activations as they are written to decode-time
KV cache, avoiding a separate read/quantize/write pass.
"""

from __future__ import annotations

import json
from typing import Any


KV_QUANT_WRITE_PARAM_SPACE = {
    "BLOCK_SIZE": [64, 128, 256, 512, 1024],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
}

KV_QUANT_WRITE_CURATED_CONFIGS = [
    {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 256, "num_warps": 2, "num_stages": 2},
    {"BLOCK_SIZE": 512, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8, "num_stages": 2},
]

KV_QUANT_WRITE_SHAPE_BUCKETS = [
    {"name": "b1_kv16_d256_t1", "rows": 16, "head_dim": 256},
    {"name": "b1_kv4_d512_t1", "rows": 4, "head_dim": 512},
    {"name": "b4_kv16_d256_t1", "rows": 64, "head_dim": 256},
    {"name": "b1_kv16_d256_t8", "rows": 128, "head_dim": 256},
    {"name": "b8_kv8_d128_t1", "rows": 64, "head_dim": 128},
]


def kv_quant_write_config_id(config: dict[str, int]) -> str:
    return f"bs{config['BLOCK_SIZE']}_w{config['num_warps']}_s{config['num_stages']}"


def kv_quant_write_shape_bucket_key(shape: dict[str, int]) -> str:
    name = shape.get("name", "")
    if name:
        return name
    d = int(shape.get("head_dim", 0))
    rows = int(shape.get("rows", 0))
    if d >= 512:
        return "b1_kv4_d512_t1"
    if rows >= 128:
        return "b1_kv16_d256_t8"
    if rows >= 64 and d >= 256:
        return "b4_kv16_d256_t1"
    if d >= 256:
        return "b1_kv16_d256_t1"
    return "b8_kv8_d128_t1"


_triton_available = False
_kv_quant_kernel = None


def _ensure_triton_kernel() -> None:
    global _triton_available, _kv_quant_kernel
    if _kv_quant_kernel is not None:
        return
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _kernel(
            x_ptr,
            q_ptr,
            s_ptr,
            n_cols,
            stride_x_row,
            stride_q_row,
            BLOCK_SIZE: tl.constexpr,
        ):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols

            x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
            max_abs = tl.max(tl.abs(x), axis=0)
            scale = tl.where(max_abs > 0.0, max_abs / 127.0, 1.0)
            inv_scale = 1.0 / scale

            y = x * inv_scale
            y = tl.where(y >= 0.0, y + 0.5, y - 0.5)
            y = tl.maximum(-127.0, tl.minimum(127.0, y))
            q = y.to(tl.int8)

            tl.store(q_ptr + row * stride_q_row + offs, q, mask=mask)
            tl.store(s_ptr + row, scale)

        _kv_quant_kernel = _kernel
        _triton_available = True
    except ImportError:
        _triton_available = False


def kv_quantize_separated(x):
    """Reference separated quantization path."""
    import torch

    max_abs = x.abs().amax(dim=1, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float32)


def kv_quantize_write_fused(x, config=None):
    """Fused Triton quantize-on-write path."""
    import torch
    import triton

    _ensure_triton_kernel()
    if not _triton_available:
        raise RuntimeError("Triton not available")
    if not x.is_cuda:
        raise ValueError("kv_quantize_write_fused requires CUDA tensor input")

    if config is None:
        config = KV_QUANT_WRITE_CURATED_CONFIGS[0]

    rows, cols = x.shape
    x_in = x.to(torch.float16).contiguous()
    q = torch.empty((rows, cols), device=x.device, dtype=torch.int8)
    s = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block_size = max(int(config["BLOCK_SIZE"]), int(triton.next_power_of_2(cols)))
    _kv_quant_kernel[(rows,)](
        x_in,
        q,
        s,
        cols,
        x_in.stride(0),
        q.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    return q, s


def generate_kv_quant_write_grid(*, include_curated: bool = True, max_configs: int = 100) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[str] = set()
    if include_curated:
        for c in KV_QUANT_WRITE_CURATED_CONFIGS:
            cid = kv_quant_write_config_id(c)
            if cid not in seen:
                seen.add(cid)
                configs.append(c)
    for bs in KV_QUANT_WRITE_PARAM_SPACE["BLOCK_SIZE"]:
        for nw in KV_QUANT_WRITE_PARAM_SPACE["num_warps"]:
            for ns in KV_QUANT_WRITE_PARAM_SPACE["num_stages"]:
                c = {"BLOCK_SIZE": bs, "num_warps": nw, "num_stages": ns}
                cid = kv_quant_write_config_id(c)
                if cid in seen:
                    continue
                seen.add(cid)
                configs.append(c)
                if len(configs) >= max_configs:
                    return configs
    return configs


def generate_kv_quant_write_benchmark_script(configs: list[dict[str, int]], shapes: list[dict[str, Any]]) -> str:
    configs_json = json.dumps(configs)
    shapes_json = json.dumps(shapes)
    return f'''#!/usr/bin/env python3
"""Auto-generated KV quantize-on-write benchmark."""

import json
import platform
import torch
import triton
import triton.language as tl


@triton.jit
def kv_quant_write_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    n_cols,
    stride_x_row,
    stride_q_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    max_abs = tl.max(tl.abs(x), axis=0)
    scale = tl.where(max_abs > 0.0, max_abs / 127.0, 1.0)
    inv_scale = 1.0 / scale
    y = x * inv_scale
    y = tl.where(y >= 0.0, y + 0.5, y - 0.5)
    y = tl.maximum(-127.0, tl.minimum(127.0, y))
    tl.store(q_ptr + row * stride_q_row + offs, y.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


def fused(x, config):
    rows, cols = x.shape
    q = torch.empty((rows, cols), device=x.device, dtype=torch.int8)
    s = torch.empty((rows,), device=x.device, dtype=torch.float32)
    bs = max(config["BLOCK_SIZE"], triton.next_power_of_2(cols))
    kv_quant_write_kernel[(rows,)](
        x, q, s, cols, x.stride(0), q.stride(0),
        BLOCK_SIZE=bs,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return q, s


def separated(x):
    max_abs = x.abs().amax(dim=1, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float32)


def bench_ms(fn, warmup=25, trials=120):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        vals.append(st.elapsed_time(en))
    vals.sort()
    return float(vals[len(vals)//2])


def run_one(shape, config):
    rows = int(shape["rows"])
    cols = int(shape["head_dim"])
    torch.manual_seed(0)
    x = torch.randn((rows, cols), device="cuda", dtype=torch.float16)
    q_ref, s_ref = separated(x)
    q_out, s_out = fused(x, config)
    # compare in dequantized space
    deq_ref = q_ref.to(torch.float32) * s_ref.unsqueeze(1)
    deq_out = q_out.to(torch.float32) * s_out.unsqueeze(1)
    max_err = (deq_ref - deq_out).abs().max().item()
    correct = bool(max_err <= 1.5)
    f_ms = bench_ms(lambda: fused(x, config))
    s_ms = bench_ms(lambda: separated(x))
    return {{
        "shape": f"{{rows}}x{{cols}}",
        "shape_name": shape.get("name", ""),
        "correct": correct,
        "max_err": round(max_err, 6),
        "ms": round(f_ms, 4),
        "separated_ms": round(s_ms, 4),
        "tflops": round((rows * cols) / f_ms, 4),
        "speedup": round(s_ms / f_ms, 4) if f_ms > 0 else 0.0,
    }}


def main():
    configs = {configs_json}
    shapes = {shapes_json}
    all_results = []
    for cfg in configs:
        cid = f"bs{{cfg['BLOCK_SIZE']}}_w{{cfg['num_warps']}}_s{{cfg['num_stages']}}"
        rows = [run_one(s, cfg) for s in shapes]
        all_results.append({{"config_id": cid, "config": cfg, "results": rows}})
    print(json.dumps({{
        "hardware": {{
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
            "python": platform.python_version(),
        }},
        "config_results": all_results,
    }}, indent=2))


if __name__ == "__main__":
    main()
'''
