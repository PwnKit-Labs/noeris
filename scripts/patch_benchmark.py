#!/usr/bin/env python3
"""Benchmark noeris.patch() speedup on a HuggingFace model.

Usage:
    python scripts/patch_benchmark.py                           # default: gemma-4-2b
    python scripts/patch_benchmark.py --model meta-llama/Llama-3.2-1B
    python scripts/patch_benchmark.py --model google/gemma-4-2b --seq-len 512 --batch 2

Measures forward-pass latency before and after noeris.patch() and reports
the speedup ratio.
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark noeris.patch() on a HuggingFace model")
    p.add_argument("--model", type=str, default="google/gemma-4-2b",
                   help="HuggingFace model name or local path")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("--repeats", type=int, default=20, help="Timed iterations")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"],
                   help="Model dtype")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def benchmark_forward(model, input_ids, warmup: int, repeats: int, device: str) -> float:
    """Return median forward-pass time in milliseconds."""
    import torch

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times.sort()
    # Return median
    mid = len(times) // 2
    return times[mid]


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Add project root to path so we can import noeris
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
    import research_engine as noeris

    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}, Seq: {args.seq_len}, Dtype: {args.dtype}")
    print(f"Device: {args.device}")
    print()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    # Create dummy input
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.batch, args.seq_len), device=args.device)

    # Benchmark BEFORE patch
    print("Benchmarking baseline (unpatched)...")
    baseline_ms = benchmark_forward(model, input_ids, args.warmup, args.repeats, args.device)
    print(f"  Baseline: {baseline_ms:.2f} ms")

    # Patch
    print("\nApplying noeris.patch()...")
    counts = noeris.patch(model, device=args.device, verbose=True)
    print(f"  Patched: {counts}")

    # Benchmark AFTER patch
    print("\nBenchmarking patched model...")
    patched_ms = benchmark_forward(model, input_ids, args.warmup, args.repeats, args.device)
    print(f"  Patched: {patched_ms:.2f} ms")

    # Results
    speedup = baseline_ms / patched_ms if patched_ms > 0 else float("inf")
    print(f"\n{'='*50}")
    print(f"  Baseline : {baseline_ms:.2f} ms")
    print(f"  Patched  : {patched_ms:.2f} ms")
    print(f"  Speedup  : {speedup:.3f}x")
    print(f"{'='*50}")

    if speedup < 1.0:
        print("\nWARNING: patched model is SLOWER. This can happen if:")
        print("  - Model is too small for kernel launch overhead to amortize")
        print("  - Running on CPU (Triton kernels need CUDA)")
        print("  - dtype mismatch causing extra casts")


if __name__ == "__main__":
    main()
