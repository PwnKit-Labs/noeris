#!/usr/bin/env python3
"""LLM-guided kernel variant search.

Loads an operator's base kernel, asks an LLM for N variant proposals,
evaluates each for correctness + performance, and reports results.

Works WITHOUT an API key in "dry run" mode (prints prompts only).

Usage:
    # Dry run -- print prompts, no API calls
    python scripts/llm_kernel_search.py --operator rmsnorm --dry-run

    # With Anthropic Claude
    ANTHROPIC_API_KEY=sk-... python scripts/llm_kernel_search.py \
        --operator rmsnorm --provider anthropic --variants 5

    # With OpenAI
    OPENAI_API_KEY=sk-... python scripts/llm_kernel_search.py \
        --operator rmsnorm --provider openai --variants 5

    # Save results
    python scripts/llm_kernel_search.py --operator rmsnorm -o results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import textwrap
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_engine.llm_kernel_proposer import LLMKernelProposer, HARDWARE_PROFILES

# ---------------------------------------------------------------------------
# Base kernel sources (inline -- keeps the script self-contained)
# ---------------------------------------------------------------------------

RMSNORM_BASE_KERNEL = textwrap.dedent("""\
    @triton.jit
    def rmsnorm_kernel(
        x_ptr, w_ptr, y_ptr,
        x_row_stride,
        y_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
        AFFINE_MODE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        x_ptr += row_idx * x_row_stride
        y_ptr += row_idx * y_row_stride
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if AFFINE_MODE == 0:
            y = x * rstd * w
        else:
            y = x * rstd * (1.0 + w)
        tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)
""")

# Test shapes for quick evaluation
RMSNORM_TEST_SHAPES = [
    {"name": "llama_7b", "n_rows": 4096, "hidden_dim": 4096, "affine_mode": 0},
    {"name": "gemma4_e2b", "n_rows": 2048, "hidden_dim": 1536, "affine_mode": 1},
]

# Add more operators here as they're integrated
OPERATOR_KERNELS = {
    "rmsnorm": {
        "source": RMSNORM_BASE_KERNEL,
        "test_shapes": RMSNORM_TEST_SHAPES,
        "performance_history": [
            {"config": "bs1024_w4_s1", "metric_name": "gb_per_s", "metric_value": 245.3},
            {"config": "bs2048_w8_s1", "metric_name": "gb_per_s", "metric_value": 261.7},
            {"config": "bs4096_w16_s1", "metric_name": "gb_per_s", "metric_value": 258.1},
            {"config": "bs512_w2_s2", "metric_name": "gb_per_s", "metric_value": 219.4},
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-guided Triton kernel variant search",
    )
    parser.add_argument(
        "--operator", "-op",
        default="rmsnorm",
        choices=list(OPERATOR_KERNELS.keys()),
        help="Operator to optimise (default: rmsnorm)",
    )
    parser.add_argument(
        "--provider", "-p",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Override the default model ID for the provider",
    )
    parser.add_argument(
        "--variants", "-n",
        type=int,
        default=5,
        help="Number of variants to request (default: 5)",
    )
    parser.add_argument(
        "--hardware", "-hw",
        default="t4",
        choices=list(HARDWARE_PROFILES.keys()),
        help="Target hardware profile (default: t4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling any API",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Actually run variants on GPU (requires CUDA)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Save results as JSON to this path",
    )
    args = parser.parse_args()

    op = OPERATOR_KERNELS[args.operator]
    kernel_source = op["source"]
    perf_history = op["performance_history"]
    test_shapes = op["test_shapes"]

    # Resolve API key
    api_key = None
    if not args.dry_run:
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(f"[warn] No API key found for {args.provider}; falling back to dry run.")
            args.dry_run = True

    proposer = LLMKernelProposer(
        operator_name=args.operator,
        provider=args.provider,
        api_key=api_key if not args.dry_run else None,
        hardware=args.hardware,
        model=args.model,
    )

    results = []
    for i in range(args.variants):
        print(f"\n{'='*60}")
        print(f"  Variant {i+1}/{args.variants}")
        print(f"{'='*60}")

        proposal = proposer.propose_variant(kernel_source, perf_history)

        if args.dry_run:
            print("\n[DRY RUN] Prompt that would be sent:\n")
            print(proposal["prompt"])
            results.append({"variant_id": i + 1, "dry_run": True, "prompt": proposal["prompt"]})
            # Only print the prompt once in dry-run -- they're all identical
            if i == 0:
                print("\n(remaining variants use the same prompt, skipping...)")
            break

        print(f"\n[LLM response excerpt]\n{(proposal['response'] or '')[:500]}...")

        variant_src = proposal.get("variant_source")
        if not variant_src:
            print("  -> Could not extract code block from response.")
            results.append({
                "variant_id": i + 1,
                "response": proposal["response"],
                "error": "no code block extracted",
            })
            continue

        print(f"\n  Extracted variant ({len(variant_src)} chars)")

        eval_result: dict = {"variant_id": i + 1, "variant_source": variant_src}

        if args.evaluate:
            print("  Evaluating on GPU...")
            ev = proposer.evaluate_variant(variant_src, test_shapes, kernel_source)
            eval_result.update(ev)
            status = "PASS" if ev["correct"] else "FAIL"
            print(f"  -> {status}  throughput={ev['throughput']:.1f} GB/s")
            if ev.get("error"):
                print(f"     error: {ev['error'][:200]}")
        else:
            print("  (skipping evaluation -- pass --evaluate to run on GPU)")

        results.append(eval_result)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  Operator:  {args.operator}")
    print(f"  Provider:  {args.provider}")
    print(f"  Hardware:  {args.hardware}")
    print(f"  Variants:  {len(results)}")
    if not args.dry_run and args.evaluate:
        correct = [r for r in results if r.get("correct")]
        print(f"  Correct:   {len(correct)}/{len(results)}")
        if correct:
            best = max(correct, key=lambda r: r.get("throughput", 0))
            print(f"  Best:      {best['throughput']:.1f} GB/s (variant {best['variant_id']})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
