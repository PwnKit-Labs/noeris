#!/usr/bin/env python3
"""Self-contained test for the split-k cooperative softmax kernel.

Verifies correctness and measures performance against PyTorch for
KernelBench problem #23: softmax over shape (4096, 393216) fp32.

Can run on any GPU (T4, A100, H100).
"""

import torch
import triton
import triton.language as tl
import time


# ---------- Kernels (copied from kernelbench_upstream.py) ----------

@triton.jit
def noeris_softmax_split_reduce_kernel(
    x_ptr, partial_max_ptr, partial_sumexp_ptr,
    row_stride, n_cols, n_chunks, chunk_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // n_chunks
    chunk_idx = pid % n_chunks
    chunk_col_start = chunk_idx * chunk_cols
    x_base = x_ptr + row_idx * row_stride + chunk_col_start

    m = tl.zeros((1,), dtype=tl.float32) - 1e30
    d = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, chunk_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = (chunk_col_start + offs) < n_cols
        x = tl.load(x_base + offs, mask=mask, other=-1e30).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, tile_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    partial_idx = row_idx * n_chunks + chunk_idx
    tl.store(partial_max_ptr + partial_idx, m)
    tl.store(partial_sumexp_ptr + partial_idx, d)


@triton.jit
def noeris_softmax_split_norm_kernel(
    x_ptr, y_ptr, partial_max_ptr, partial_sumexp_ptr,
    row_stride, n_cols, n_chunks, chunk_cols,
    BLOCK_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // n_chunks
    chunk_idx = pid % n_chunks
    chunk_col_start = chunk_idx * chunk_cols
    x_base = x_ptr + row_idx * row_stride + chunk_col_start
    y_base = y_ptr + row_idx * row_stride + chunk_col_start

    partial_base = row_idx * N_CHUNKS
    global_m = tl.load(partial_max_ptr + partial_base).to(tl.float32)
    global_d = tl.load(partial_sumexp_ptr + partial_base).to(tl.float32)
    for c in range(1, N_CHUNKS):
        cm = tl.load(partial_max_ptr + partial_base + c).to(tl.float32)
        cd = tl.load(partial_sumexp_ptr + partial_base + c).to(tl.float32)
        new_m = tl.maximum(global_m, cm)
        global_d = global_d * tl.exp(global_m - new_m) + cd * tl.exp(cm - new_m)
        global_m = new_m

    for start in range(0, chunk_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = (chunk_col_start + offs) < n_cols
        x = tl.load(x_base + offs, mask=mask, other=-1e30).to(tl.float32)
        y = tl.exp(x - global_m) / global_d
        tl.store(y_base + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


# ---------- Launcher ----------

def splitk_softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    N_CHUNKS = 8
    BLOCK = 8192
    total_tiles = (n_cols + BLOCK - 1) // BLOCK
    tiles_per_chunk = (total_tiles + N_CHUNKS - 1) // N_CHUNKS
    chunk_cols = tiles_per_chunk * BLOCK
    total_programs = n_rows * N_CHUNKS

    partial_max = torch.empty((n_rows, N_CHUNKS), dtype=torch.float32, device=x.device)
    partial_sumexp = torch.empty((n_rows, N_CHUNKS), dtype=torch.float32, device=x.device)

    noeris_softmax_split_reduce_kernel[(total_programs,)](
        x, partial_max, partial_sumexp,
        x.stride(0), n_cols, N_CHUNKS, chunk_cols,
        BLOCK_SIZE=BLOCK,
        num_warps=16,
        num_stages=2,
    )
    noeris_softmax_split_norm_kernel[(total_programs,)](
        x, y, partial_max, partial_sumexp,
        x.stride(0), n_cols, N_CHUNKS, chunk_cols,
        BLOCK_SIZE=BLOCK,
        N_CHUNKS=N_CHUNKS,
        num_warps=16,
        num_stages=2,
    )
    return y


# ---------- Test ----------

def main():
    device = "cuda"
    dtype = torch.float32
    rows, cols = 4096, 393216
    print(f"Shape: ({rows}, {cols})  dtype: {dtype}")

    x = torch.randn(rows, cols, device=device, dtype=dtype)

    # Correctness check
    print("Running correctness check...")
    ref = torch.softmax(x, dim=-1)
    out = splitk_softmax(x)
    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    print(f"  max_abs_diff = {max_diff:.2e}")
    print(f"  mean_abs_diff = {mean_diff:.2e}")
    if max_diff < 1e-4:
        print("  PASS")
    else:
        print("  FAIL - diffs too large!")
        return

    # Warmup
    for _ in range(3):
        splitk_softmax(x)
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Benchmark
    N = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        splitk_softmax(x)
    torch.cuda.synchronize()
    noeris_ms = (time.perf_counter() - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) / N * 1000

    speedup = torch_ms / noeris_ms
    print(f"\nPerformance:")
    print(f"  Noeris split-k: {noeris_ms:.2f} ms")
    print(f"  PyTorch:         {torch_ms:.2f} ms")
    print(f"  Speedup:         {speedup:.2f}x")


if __name__ == "__main__":
    main()
