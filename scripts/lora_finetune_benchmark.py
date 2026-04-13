#!/usr/bin/env python3
"""Benchmark noeris.patch() speedup on LoRA fine-tuning (forward + backward + optimizer).

Usage:
    python scripts/lora_finetune_benchmark.py
    python scripts/lora_finetune_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python scripts/lora_finetune_benchmark.py --steps 30 --seq-len 256 --batch 2

Measures wall-clock time per training step before and after noeris.patch()
and saves results to lora_finetune_benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning benchmark for noeris.patch()")
    p.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HuggingFace model ID (default: TinyLlama 1.1B)")
    p.add_argument("--batch", type=int, default=2, help="Batch size")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p.add_argument("--steps", type=int, default=20, help="Timed training steps")
    p.add_argument("--warmup", type=int, default=5, help="Warmup training steps")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--output", type=str, default="lora_finetune_benchmark_results.json")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def benchmark_training(model, optimizer, dummy_ids, warmup: int, steps: int, device: str):
    """Return list of per-step times in ms using CUDA event timing."""
    import torch

    model.train()

    # Warmup (not timed)
    for _ in range(warmup):
        out = model(input_ids=dummy_ids, labels=dummy_ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed steps
    times_ms = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        out = model(input_ids=dummy_ids, labels=dummy_ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return times_ms


def main():
    args = parse_args()

    # --- Dependency check ---
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch transformers peft")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU. Skipping.")
        sys.exit(0)

    # --- Import noeris ---
    try:
        import noeris  # noqa: F811
    except ImportError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
        import research_engine as noeris  # type: ignore[no-redef]

    # --- Load model ---
    print(f"Model : {args.model}")
    print(f"Batch : {args.batch}  Seq: {args.seq_len}  LoRA r={args.lora_r}")
    print(f"Steps : {args.steps} timed + {args.warmup} warmup\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
    )

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dummy data ---
    vocab_size = model.config.vocab_size
    dummy_ids = torch.randint(0, vocab_size, (args.batch, args.seq_len), device=args.device)

    # --- Baseline (unpatched) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("\nBenchmarking baseline (unpatched LoRA training)...")
    baseline_times = benchmark_training(model, optimizer, dummy_ids, args.warmup, args.steps, args.device)
    baseline_times.sort()
    baseline_ms = baseline_times[len(baseline_times) // 2]
    print(f"  Baseline median: {baseline_ms:.1f} ms/step")

    # --- Apply noeris.patch ---
    print("\nApplying noeris.patch()...")
    counts = noeris.patch(model, device=args.device, verbose=True)
    print(f"  Patched layers: {counts}")

    # Reset optimizer state to avoid stale momentum from pre-patch params
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Patched ---
    print("\nBenchmarking patched (noeris LoRA training)...")
    patched_times = benchmark_training(model, optimizer, dummy_ids, args.warmup, args.steps, args.device)
    patched_times.sort()
    patched_ms = patched_times[len(patched_times) // 2]
    print(f"  Patched median:  {patched_ms:.1f} ms/step")

    # --- Report ---
    speedup = baseline_ms / patched_ms if patched_ms > 0 else float("inf")
    print(f"\n{'=' * 55}")
    print(f"  Baseline : {baseline_ms:8.1f} ms/step")
    print(f"  Patched  : {patched_ms:8.1f} ms/step")
    print(f"  Speedup  : {speedup:8.2f}x")
    print(f"{'=' * 55}")

    if speedup < 1.0:
        print("\nWARNING: patched model is SLOWER — likely kernel launch")
        print("overhead dominates on this small model / short sequence.")

    # --- Save JSON ---
    results = {
        "model": args.model,
        "batch_size": args.batch,
        "seq_len": args.seq_len,
        "lora_r": args.lora_r,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "baseline_median_ms": round(baseline_ms, 2),
        "patched_median_ms": round(patched_ms, 2),
        "speedup": round(speedup, 3),
        "patched_layer_counts": counts,
        "baseline_all_ms": [round(t, 2) for t in baseline_times],
        "patched_all_ms": [round(t, 2) for t in patched_times],
    }
    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
