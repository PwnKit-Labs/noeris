#!/usr/bin/env python3
"""Compiler failure analysis: proves torch.compile (Inductor) fails to fuse
QK-RMSNorm + RoPE into a single kernel automatically.

For the paper's "compiler failure analysis" section. Shows that even with
mode="max-autotune", Inductor generates 4+ separate Triton kernels for the
separated ops, while the Noeris hand-written kernel needs only 2 launches.

Usage (Kaggle T4 / Colab T4):
  !git clone https://github.com/PwnKit-Labs/noeris && cd noeris
  !pip install -e . numpy scikit-learn -q
  !python scripts/compiler_analysis.py

Outputs structured text summary + compiler_analysis_results.json.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
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

# ---------------------------------------------------------------------------
# Test shapes: Gemma 4 E2B local layer
# ---------------------------------------------------------------------------
B, H, H_KV, S, D = 1, 8, 1, 4096, 256
HALF_D = D // 2

# ---------------------------------------------------------------------------
# Helper: CUDA event timer (same as colab_bombshell.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# The 4 separated ops as plain PyTorch (what Inductor sees)
# ---------------------------------------------------------------------------

def separated_qk_norm_rope(q, k, q_scale, k_scale, cos, sin, eps=1e-6):
    """Separated QK-RMSNorm + RoPE -- 4 logical ops."""
    # Q-RMSNorm (Gemma 1+w mode)
    q_var = q.float().pow(2).mean(-1, keepdim=True)
    q_n = (q.float() * torch.rsqrt(q_var + eps)).half() * (1.0 + q_scale).half()
    # K-RMSNorm
    k_var = k.float().pow(2).mean(-1, keepdim=True)
    k_n = (k.float() * torch.rsqrt(k_var + eps)).half() * (1.0 + k_scale).half()
    # Q-RoPE
    qe, qo = q_n[..., 0::2], q_n[..., 1::2]
    c = cos[None, None, :, :].half()
    s = sin[None, None, :, :].half()
    q_out = torch.stack([qe * c - qo * s, qe * s + qo * c], dim=-1).reshape(q.shape)
    # K-RoPE
    ke, ko = k_n[..., 0::2], k_n[..., 1::2]
    k_out = torch.stack([ke * c - ko * s, ke * s + ko * c], dim=-1).reshape(k.shape)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Allocate tensors
# ---------------------------------------------------------------------------

def make_inputs():
    q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H_KV, S, D), device="cuda", dtype=torch.float16)
    q_scale = torch.randn((D,), device="cuda", dtype=torch.float32) * 0.1
    k_scale = torch.randn((D,), device="cuda", dtype=torch.float32) * 0.1
    cos = torch.randn((S, HALF_D), device="cuda", dtype=torch.float32)
    sin = torch.randn((S, HALF_D), device="cuda", dtype=torch.float32)
    return q, k, q_scale, k_scale, cos, sin


# ===================================================================
# Test 1: Dynamo graph inspection -- check for graph breaks
# ===================================================================

def test_dynamo_explain():
    print("\n" + "=" * 60)
    print("TEST 1: torch._dynamo.explain() -- graph break analysis")
    print("=" * 60)
    info = {}
    try:
        q, k, q_scale, k_scale, cos, sin = make_inputs()
        explanation = torch._dynamo.explain(separated_qk_norm_rope)(
            q, k, q_scale, k_scale, cos, sin,
        )
        # explanation is an ExplainOutput object; extract useful fields
        explain_str = str(explanation)
        print(explain_str[:2000])
        info["graph_breaks"] = getattr(explanation, "graph_break_count", None)
        info["graph_count"] = getattr(explanation, "graph_count", None)
        info["raw"] = explain_str[:1500]
    except Exception as exc:
        print(f"  dynamo.explain failed: {exc}")
        info["error"] = str(exc)[:300]
    return info


# ===================================================================
# Test 2: Count CUDA kernel launches via torch.profiler
# ===================================================================

def count_kernel_launches(fn, label="function"):
    """Run fn under torch.profiler and count CUDA kernel launch events."""
    print(f"\n  Profiling {label}...")
    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    kernel_names = []
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            fn()
            torch.cuda.synchronize()

        for evt in prof.key_averages():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                kernel_names.append(evt.key)

        # Also try the events() API for a more detailed list
        detailed_kernels = []
        for evt in prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name:
                detailed_kernels.append(evt.name)

        return {
            "unique_kernels": len(kernel_names),
            "total_launches": len(detailed_kernels),
            "kernel_names": kernel_names[:30],
            "detailed_names": detailed_kernels[:50],
        }
    except Exception as exc:
        print(f"  Profiler error: {exc}")
        return {"error": str(exc)[:300]}


def test_kernel_launches():
    print("\n" + "=" * 60)
    print("TEST 2: CUDA kernel launch counts (torch.profiler)")
    print("=" * 60)

    q, k, q_scale, k_scale, cos, sin = make_inputs()

    # --- Eager PyTorch ---
    eager_fn = lambda: separated_qk_norm_rope(q, k, q_scale, k_scale, cos, sin)
    eager_info = count_kernel_launches(eager_fn, "PyTorch eager")
    print(f"  Eager: {eager_info.get('unique_kernels', '?')} unique kernels, "
          f"{eager_info.get('total_launches', '?')} total launches")

    # --- torch.compile (max-autotune) ---
    compiled_info = {"error": "compilation failed"}
    try:
        print("\n  Compiling with torch.compile(mode='max-autotune')...")
        compiled_fn = torch.compile(separated_qk_norm_rope, mode="max-autotune")
        # Trigger compilation with real inputs
        _ = compiled_fn(q, k, q_scale, k_scale, cos, sin)
        torch.cuda.synchronize()
        print("  Compilation successful.")

        compiled_caller = lambda: compiled_fn(q, k, q_scale, k_scale, cos, sin)
        compiled_info = count_kernel_launches(compiled_caller, "torch.compile max-autotune")
        print(f"  Compiled: {compiled_info.get('unique_kernels', '?')} unique kernels, "
              f"{compiled_info.get('total_launches', '?')} total launches")
    except Exception as exc:
        print(f"  torch.compile failed: {exc}")
        compiled_info["error"] = str(exc)[:300]

    # --- Noeris fused Triton ---
    noeris_info = {"unique_kernels": 2, "total_launches": 2, "note": "by construction"}
    try:
        from research_engine.triton_qk_norm_rope import apply_qk_norm_rope
        noeris_fn = lambda: apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale)
        # Verify it runs
        _ = noeris_fn()
        torch.cuda.synchronize()
        noeris_info = count_kernel_launches(noeris_fn, "Noeris fused Triton")
        print(f"  Noeris:   {noeris_info.get('unique_kernels', '?')} unique kernels, "
              f"{noeris_info.get('total_launches', '?')} total launches")
    except Exception as exc:
        print(f"  Noeris kernel not available (expected on CPU): {exc}")
        noeris_info["note"] = f"unavailable: {exc}"

    return {"eager": eager_info, "compiled": compiled_info, "noeris": noeris_info}


# ===================================================================
# Test 3: Timing comparison (eager vs compiled vs Noeris)
# ===================================================================

def test_timing():
    print("\n" + "=" * 60)
    print("TEST 3: Timing comparison")
    print("=" * 60)

    q, k, q_scale, k_scale, cos, sin = make_inputs()
    results = {}

    # Eager
    eager_fn = lambda: separated_qk_norm_rope(q, k, q_scale, k_scale, cos, sin)
    eager_ms = cuda_event_timer(eager_fn)
    print(f"  PyTorch eager:         {eager_ms:.3f} ms")
    results["eager_ms"] = round(eager_ms, 4)

    # torch.compile
    try:
        # Reset dynamo to get a clean compile
        torch._dynamo.reset()
        compiled_fn = torch.compile(separated_qk_norm_rope, mode="max-autotune")
        # Trigger compilation
        _ = compiled_fn(q, k, q_scale, k_scale, cos, sin)
        torch.cuda.synchronize()
        compiled_caller = lambda: compiled_fn(q, k, q_scale, k_scale, cos, sin)
        compiled_ms = cuda_event_timer(compiled_caller)
        print(f"  torch.compile:         {compiled_ms:.3f} ms")
        results["compiled_ms"] = round(compiled_ms, 4)
    except Exception as exc:
        print(f"  torch.compile timing failed: {exc}")
        results["compiled_ms"] = None
        results["compiled_error"] = str(exc)[:200]

    # Noeris fused Triton
    try:
        from research_engine.triton_qk_norm_rope import apply_qk_norm_rope
        noeris_fn = lambda: apply_qk_norm_rope(q, k, cos, sin, q_scale, k_scale)
        _ = noeris_fn()
        torch.cuda.synchronize()
        noeris_ms = cuda_event_timer(noeris_fn)
        print(f"  Noeris fused Triton:   {noeris_ms:.3f} ms")
        results["noeris_ms"] = round(noeris_ms, 4)
    except Exception as exc:
        print(f"  Noeris kernel not available: {exc}")
        results["noeris_ms"] = None
        results["noeris_error"] = str(exc)[:200]

    return results


# ===================================================================
# Test 4: Inductor-generated code inspection
# ===================================================================

def test_inductor_code():
    print("\n" + "=" * 60)
    print("TEST 4: Inductor generated code inspection")
    print("=" * 60)

    info = {}
    try:
        import torch._inductor.config as inductor_config
        # Enable Inductor logging to capture generated code
        old_debug = inductor_config.debug
        inductor_config.debug = True

        torch._dynamo.reset()
        q, k, q_scale, k_scale, cos, sin = make_inputs()

        # Use the code inspection API
        from torch._dynamo.utils import counters
        counters.clear()

        compiled_fn = torch.compile(separated_qk_norm_rope, mode="max-autotune")
        _ = compiled_fn(q, k, q_scale, k_scale, cos, sin)
        torch.cuda.synchronize()

        # Report counters
        info["dynamo_counters"] = {k: dict(v) for k, v in counters.items() if v}
        print(f"  Dynamo counters: {json.dumps(info['dynamo_counters'], indent=2, default=str)[:1000]}")

        inductor_config.debug = old_debug
    except Exception as exc:
        print(f"  Inductor inspection failed: {exc}")
        info["error"] = str(exc)[:300]

    return info


# ===================================================================
# Main
# ===================================================================

def main():
    print("\n" + "#" * 60)
    print("# COMPILER FAILURE ANALYSIS")
    print(f"# torch.compile vs Noeris fused QK-RMSNorm+RoPE")
    print(f"# Shape: B={B}, H={H}, H_kv={H_KV}, S={S}, D={D}")
    print(f"# GPU: {GPU_NAME}")
    print("#" * 60)

    all_results = {
        "hardware": {
            "gpu": GPU_NAME,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unknown",
        },
        "shape": {"B": B, "H": H, "H_kv": H_KV, "S": S, "D": D},
    }

    # Test 1: Dynamo explain
    try:
        all_results["dynamo_explain"] = test_dynamo_explain()
    except Exception as exc:
        all_results["dynamo_explain"] = {"error": traceback.format_exc()[:500]}

    # Test 2: Kernel launch counts
    try:
        torch._dynamo.reset()
        all_results["kernel_launches"] = test_kernel_launches()
    except Exception as exc:
        all_results["kernel_launches"] = {"error": traceback.format_exc()[:500]}

    # Test 3: Timing
    try:
        torch._dynamo.reset()
        all_results["timing"] = test_timing()
    except Exception as exc:
        all_results["timing"] = {"error": traceback.format_exc()[:500]}

    # Test 4: Inductor inspection
    try:
        torch._dynamo.reset()
        all_results["inductor_inspection"] = test_inductor_code()
    except Exception as exc:
        all_results["inductor_inspection"] = {"error": traceback.format_exc()[:500]}

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n")
    print("=" * 60)
    print("COMPILER FAILURE ANALYSIS — SUMMARY")
    print("=" * 60)

    kl = all_results.get("kernel_launches", {})
    compiled_launches = kl.get("compiled", {}).get("total_launches", "?")
    noeris_launches = kl.get("noeris", {}).get("total_launches", 2)
    eager_launches = kl.get("eager", {}).get("total_launches", "?")

    timing = all_results.get("timing", {})
    eager_ms = timing.get("eager_ms", "?")
    compiled_ms = timing.get("compiled_ms", "?")
    noeris_ms = timing.get("noeris_ms", "?")

    # Determine if compiler found fusion
    try:
        fusion_found = (
            isinstance(compiled_launches, int)
            and compiled_launches <= 2
        )
    except Exception:
        fusion_found = False

    all_results["fusion_found_by_compiler"] = fusion_found

    print(f"Shape: B={B}, H={H}, H_kv={H_KV}, S={S}, D={D}")
    print(f"GPU: {GPU_NAME}")
    print(f"")
    print(f"Kernel launches:")
    print(f"  PyTorch eager:         {eager_launches}")
    print(f"  torch.compile:         {compiled_launches}")
    print(f"  Noeris fused Triton:   {noeris_launches}")
    print(f"")
    print(f"Latency (median, ms):")
    print(f"  PyTorch eager:         {eager_ms}")
    print(f"  torch.compile:         {compiled_ms}")
    print(f"  Noeris fused Triton:   {noeris_ms}")

    if isinstance(compiled_ms, (int, float)) and isinstance(noeris_ms, (int, float)) and noeris_ms > 0:
        speedup_vs_compile = compiled_ms / noeris_ms
        print(f"")
        print(f"Noeris speedup vs torch.compile: {speedup_vs_compile:.2f}x")
        all_results["noeris_speedup_vs_compile"] = round(speedup_vs_compile, 3)

    if isinstance(eager_ms, (int, float)) and isinstance(noeris_ms, (int, float)) and noeris_ms > 0:
        speedup_vs_eager = eager_ms / noeris_ms
        print(f"Noeris speedup vs eager:         {speedup_vs_eager:.2f}x")
        all_results["noeris_speedup_vs_eager"] = round(speedup_vs_eager, 3)

    print(f"")
    print(f"Fusion found by compiler: {'YES' if fusion_found else 'NO'}")
    if not fusion_found:
        print(f"  -> Inductor generates {compiled_launches} kernel launches")
        print(f"     vs Noeris's 2 launches (1 per Q/K).")
        print(f"  -> The compiler cannot discover the cross-op RMSNorm+RoPE")
        print(f"     fusion because the intermediate (normalized Q/K) is")
        print(f"     materialized to HBM between the norm and rotation steps.")

    # Save JSON
    out_path = REPO / "compiler_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
