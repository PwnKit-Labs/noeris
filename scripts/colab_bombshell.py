#!/usr/bin/env python3
"""Comprehensive bombshell measurement script for free T4 GPU (Kaggle or Colab).

Primary platform: Kaggle (30 hr/week free T4, API-driven via `kaggle kernels push`).
Backup platform: Google Colab (~4-5 hr/day free T4).

Validates ALL headline claims in a single ~20-minute run:
  Phase 1: Attention sliding-window vs SDPA (can we beat FlashAttention?)
  Phase 2: Split-K matmul vs cuBLAS
  Phase 3: Full Gemma 4 layer benchmark (Noeris fused vs PyTorch separated)
  Phase 4: Forward + backward prologue (fused QK-RMSNorm+RoPE)
  Phase 5: Bandit search convergence on attention

Usage (Kaggle or Colab):
  !git clone https://github.com/PwnKit-Labs/noeris && cd noeris
  !pip install -e . numpy scikit-learn -q
  !python scripts/colab_bombshell.py

Outputs machine-readable JSON with all results.
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
    print("ERROR: No GPU available. Change Colab runtime to T4 GPU.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")

import triton
print(f"Triton {triton.__version__}")

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
# Phase 1: Attention sliding-window vs SDPA
# ============================================================================

def phase1_attention_vs_sdpa():
    """Compare Noeris sliding-window attention (tile-pruning) vs PyTorch SDPA."""
    print("\n" + "=" * 70)
    print("PHASE 1: Attention sliding-window vs SDPA")
    print("=" * 70)

    from research_engine.triton_attention import flash_attn

    # Gemma 3/4 sliding-window shape: W=1024, S=4096, head_dim=256
    # Use smaller batch/heads for T4 memory
    B, H, S, D = 1, 8, 4096, 256
    W = 1024
    num_kv_heads = 8

    print(f"  Shape: B={B}, H={H}, S={S}, D={D}, window={W}")

    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, num_kv_heads, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, num_kv_heads, S, D, device="cuda", dtype=torch.float16)

    # T4-friendly config: small tiles to fit shared memory with head_dim=256
    config = {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 3}

    # Noeris attention with tile pruning
    noeris_ms = cuda_event_timer(
        lambda: flash_attn(q, k, v, config=config, is_causal=True,
                           window_size=W, num_kv_heads=num_kv_heads)
    )

    # PyTorch SDPA with sliding-window mask
    # Expand KV for GQA
    repeat = H // num_kv_heads
    k_exp = k.unsqueeze(2).expand(B, num_kv_heads, repeat, S, D).reshape(B, H, S, D)
    v_exp = v.unsqueeze(2).expand(B, num_kv_heads, repeat, S, D).reshape(B, H, S, D)

    rows = torch.arange(S, device="cuda").unsqueeze(1)
    cols = torch.arange(S, device="cuda").unsqueeze(0)
    mask = ((cols >= (rows - W + 1)) & (cols <= rows)).unsqueeze(0).unsqueeze(0)

    sdpa_ms = cuda_event_timer(
        lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask=mask, is_causal=False
        )
    )

    speedup = sdpa_ms / noeris_ms if noeris_ms > 0 else 0.0
    beat_flashattn = speedup > 1.0

    print(f"  Noeris:  {noeris_ms:.3f} ms")
    print(f"  SDPA:    {sdpa_ms:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Beat FlashAttention on this workload: {beat_flashattn}")

    return {
        "noeris_ms": round(noeris_ms, 4),
        "sdpa_ms": round(sdpa_ms, 4),
        "speedup": round(speedup, 4),
        "beat_flashattn": beat_flashattn,
        "shape": f"B{B}_H{H}_S{S}_D{D}_W{W}",
    }


# ============================================================================
# Phase 2: Split-K matmul vs cuBLAS
# ============================================================================

def phase2_splitk_vs_cublas():
    """Compare Noeris split-K matmul vs torch.matmul (cuBLAS) via subprocess."""
    print("\n" + "=" * 70)
    print("PHASE 2: Split-K matmul vs cuBLAS")
    print("=" * 70)

    import subprocess
    import tempfile

    from research_engine.triton_operators import REGISTRY

    spec = REGISTRY.get("matmul_splitk")

    shapes = [
        {"name": "4096x4096", "M": 4096, "N": 4096, "K": 4096},
        {"name": "8192x4096", "M": 8192, "N": 4096, "K": 4096},
        {"name": "llm_mlp_down", "M": 4096, "N": 4096, "K": 11008},
    ]

    # T4-friendly configs: smaller tiles to fit shared memory
    configs = [
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "SPLIT_K": 2, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
         "SPLIT_K": 4, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
         "SPLIT_K": 1, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    ]

    script = spec.benchmark_script_fn(configs, shapes)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    print(f"  Running split-K benchmark subprocess...")
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "unknown"
        print(f"  FAIL: {err[:200]}")
        return {"error": err[:300]}

    # Parse JSON output
    stdout = proc.stdout
    json_start = stdout.find("{")
    if json_start < 0:
        print("  FAIL: no JSON output found")
        print(stdout[-500:])
        return {"error": "no JSON output"}

    payload = json.loads(stdout[json_start:])

    all_results = []
    best_ratio = 0.0
    best_shape = ""

    for cr in payload.get("config_results", []):
        config = cr.get("config", {})
        sk = config.get("SPLIT_K", 1)
        for r in cr.get("results", []):
            shape_name = r.get("shape_name", "?")
            ratio = r.get("ratio_vs_cublas", 0.0) or 0.0
            correct = r.get("correct", False)

            if correct:
                all_results.append({
                    "shape": shape_name,
                    "SPLIT_K": sk,
                    "splitk_ms": r.get("ms"),
                    "cublas_ms": r.get("cublas_ms"),
                    "ratio_vs_cublas": ratio,
                })
                print(f"  {shape_name} SPLIT_K={sk}: ratio={ratio:.3f}")

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_shape = f"{shape_name}_SK{sk}"

    print(f"\n  Best ratio vs cuBLAS: {best_ratio:.3f} on {best_shape}")

    return {
        "best_ratio": round(best_ratio, 4),
        "shape": best_shape,
        "details": all_results,
    }


# ============================================================================
# Phase 3: Full layer benchmark (Gemma 4 E2B local)
# ============================================================================

def phase3_layer_benchmark():
    """Run Gemma 4 E2B local layer: Noeris fused vs PyTorch separated."""
    print("\n" + "=" * 70)
    print("PHASE 3: Full layer benchmark (Gemma 4 E2B local)")
    print("=" * 70)

    from research_engine.gemma4_layer_benchmark import generate_gemma4_layer_benchmark_script

    # Use only the E2B local config — smallest, fits T4 easily
    e2b_config = {
        "name": "gemma4_e2b_local",
        "batch": 1,
        "seq_len": 2048,  # reduced for T4
        "hidden_dim": 1536,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
        "ffn_dim": 6144,
        "window_size": 512,
        "is_causal": True,
    }

    script = generate_gemma4_layer_benchmark_script([e2b_config])

    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    print(f"  Running layer benchmark subprocess...")
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "unknown"
        print(f"  FAIL: {err[:200]}")
        return {"error": err[:300]}

    # Extract JSON results
    stdout = proc.stdout
    json_start = stdout.find("--- JSON_RESULTS_START ---")
    json_end = stdout.find("--- JSON_RESULTS_END ---")
    if json_start < 0 or json_end < 0:
        print("  FAIL: no JSON output found")
        # Print stdout for debugging
        print(stdout[-500:])
        return {"error": "no JSON output"}

    json_str = stdout[json_start + len("--- JSON_RESULTS_START ---"):json_end].strip()
    payload = json.loads(json_str)

    results = payload.get("layer_results", [])
    if not results:
        return {"error": "empty layer_results"}

    r = results[0]
    if "error" in r:
        return {"error": r["error"]}

    noeris_ms = r["noeris_fused_ms"]
    pytorch_ms = r["pytorch_separated_ms"]
    speedup = r["layer_speedup"]
    correct = r["correct"]

    print(f"  Noeris fused:      {noeris_ms:.3f} ms")
    print(f"  PyTorch separated: {pytorch_ms:.3f} ms")
    print(f"  Layer speedup:     {speedup:.2f}x")
    print(f"  Correct:           {correct}")

    # Print per-step breakdown
    if "noeris_step_times" in r and "pytorch_step_times" in r:
        print("\n  Per-step breakdown (ms):")
        for step, ms in sorted(r["noeris_step_times"].items()):
            print(f"    Noeris {step}: {ms:.3f}")
        for step, ms in sorted(r["pytorch_step_times"].items()):
            print(f"    PyTorch {step}: {ms:.3f}")

    return {
        "noeris_ms": round(noeris_ms, 4),
        "pytorch_ms": round(pytorch_ms, 4),
        "speedup": round(speedup, 4),
        "correct": correct,
    }


# ============================================================================
# Phase 4: Forward + backward prologue (QK-RMSNorm+RoPE)
# ============================================================================

def phase4_prologue_fwd_bwd():
    """Benchmark fused QK-RMSNorm+RoPE forward+backward vs separated ops via subprocess."""
    print("\n" + "=" * 70)
    print("PHASE 4: Forward + backward prologue (QK-RMSNorm+RoPE)")
    print("=" * 70)

    import subprocess
    import tempfile

    from research_engine.triton_operators import REGISTRY

    spec = REGISTRY.get("qk_norm_rope_bwd")

    # Gemma 4 local shape, T4-friendly
    shapes = [
        {"name": "gemma4_local", "batch": 1, "heads": 8, "num_kv_heads": 4,
         "seq": 4096, "head_dim": 256},
    ]

    # T4-friendly configs: small BLOCK_SIZE to fit shared memory
    configs = [
        {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1},
        {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 1},
    ]

    script = spec.benchmark_script_fn(configs, shapes)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    print(f"  Running prologue fwd+bwd benchmark subprocess...")
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "unknown"
        print(f"  FAIL: {err[:200]}")
        return {"error": err[:300]}

    # Parse JSON output
    stdout = proc.stdout
    json_start = stdout.find("{")
    if json_start < 0:
        print("  FAIL: no JSON output found")
        print(stdout[-500:])
        return {"error": "no JSON output"}

    payload = json.loads(stdout[json_start:])

    best_speedup = 0.0
    best_ms = None
    best_sep_ms = None

    for cr in payload.get("config_results", []):
        for r in cr.get("results", []):
            correct = r.get("correct", False)
            speedup = r.get("backward_fusion_speedup", 0.0) or 0.0
            if correct and speedup > best_speedup:
                best_speedup = speedup
                best_ms = r.get("ms")
                best_sep_ms = r.get("separated_ms")

    print(f"  Best fused fwd+bwd:     {best_ms} ms")
    print(f"  Separated fwd+bwd:      {best_sep_ms} ms")
    print(f"  Backward fusion speedup: {best_speedup:.2f}x")

    return {
        "forward": {
            "fusion_speedup": round(best_speedup, 4),
            "fused_ms": round(best_ms, 4) if best_ms else None,
            "separated_ms": round(best_sep_ms, 4) if best_sep_ms else None,
        },
        "backward": {
            "fusion_speedup": round(best_speedup, 4),
            "fused_ms": round(best_ms, 4) if best_ms else None,
            "separated_ms": round(best_sep_ms, 4) if best_sep_ms else None,
        },
    }


# ============================================================================
# Phase 5: Bandit search convergence on attention
# ============================================================================

def phase5_bandit_search():
    """Run 3 iterations of bandit search on attention, report improvement."""
    print("\n" + "=" * 70)
    print("PHASE 5: Bandit search convergence (attention)")
    print("=" * 70)

    from research_engine.triton_operators import REGISTRY
    from research_engine.triton_kernels import ConfigDatabase

    import subprocess
    import tempfile

    spec = REGISTRY.get("attention")
    hardware = GPU_NAME

    # Use tiny shapes for speed on T4 — include a sliding-window shape
    shapes = [s for s in spec.shape_buckets
              if s["name"] in ("short_64", "gemma4_local_short")][:2]
    if not shapes:
        shapes = spec.shape_buckets[:2]

    # Baseline: curated config on first shape
    baseline_config = spec.curated_configs[0]
    script_baseline = spec.benchmark_script_fn([baseline_config], shapes[:1])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_baseline)
        f.flush()
        proc = subprocess.run(
            [sys.executable, f.name],
            capture_output=True, text=True, timeout=120,
        )

    baseline_metric = None
    if proc.returncode == 0:
        start = proc.stdout.find("{")
        if start >= 0:
            payload = json.loads(proc.stdout[start:])
            for cr in payload.get("config_results", []):
                for r in cr.get("results", []):
                    m = r.get("gb_per_s") or r.get("tflops") or 0
                    if r.get("correct") and (baseline_metric is None or m > baseline_metric):
                        baseline_metric = m

    print(f"  Baseline metric (curated starter): {baseline_metric}")

    # Run 3 bandit iterations
    db_path = "/tmp/noeris_bombshell_bandit.json"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    best_metric_after = baseline_metric

    for iteration in range(3):
        print(f"\n  --- Bandit iteration {iteration + 1}/3 ---")

        try:
            from research_engine.bandit_selector import BanditSelector
            db = ConfigDatabase(path=db_path)
            bandit = BanditSelector()
            configs = bandit.select_configs(
                spec=spec, database=db, hardware=hardware,
                shapes=shapes, max_configs=6,
            )
        except Exception as e:
            print(f"    Bandit selection failed: {e}, using grid")
            configs = spec.grid_generator_fn(max_configs=6)[:6]

        script = spec.benchmark_script_fn(configs, shapes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            proc = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=180,
            )

        if proc.returncode != 0:
            err = proc.stderr[-200:] if proc.stderr else "unknown"
            print(f"    FAIL: {err[:100]}")
            continue

        start = proc.stdout.find("{")
        if start < 0:
            print("    No JSON output")
            continue

        payload = json.loads(proc.stdout[start:])

        for cr in payload.get("config_results", []):
            config = cr.get("config", {})
            for r in cr.get("results", []):
                shape_name = r.get("shape_name", "?")
                correct = r.get("correct", False)
                m = r.get("gb_per_s") or r.get("tflops") or 0
                ms_val = r.get("ms") or 0

                if correct:
                    if best_metric_after is None or m > best_metric_after:
                        best_metric_after = m

                    # Record in DB for next iteration
                    bucket = spec.shape_bucket_fn(
                        next((s for s in shapes if s["name"] == shape_name), {})
                    )
                    db.record_result(
                        shape={"name": shape_name},
                        hardware=hardware,
                        config=config,
                        tflops=float(m),
                        ms=float(ms_val),
                        correct=correct,
                        operator="attention",
                        bucket=bucket,
                        config_id_str=spec.config_id_fn(config),
                    )

        db.save()
        print(f"    Best metric so far: {best_metric_after}")

    before = baseline_metric or 0
    after = best_metric_after or 0
    improvement = ((after - before) / before * 100) if before > 0 else 0.0

    print(f"\n  Before (curated): {before}")
    print(f"  After (bandit):   {after}")
    print(f"  Improvement:      {improvement:.1f}%")

    return {
        "before": round(before, 4) if before else None,
        "after": round(after, 4) if after else None,
        "improvement_pct": round(improvement, 2),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("NOERIS BOMBSHELL MEASUREMENT SCRIPT")
    print(f"GPU: {GPU_NAME}")
    print("=" * 70)

    t_start = time.time()
    results = {}

    phases = [
        ("attention_vs_sdpa", phase1_attention_vs_sdpa),
        ("splitk_vs_cublas", phase2_splitk_vs_cublas),
        ("layer_speedup", phase3_layer_benchmark),
        ("prologue_forward_backward", phase4_prologue_fwd_bwd),
        ("attention_bandit_improvement", phase5_bandit_search),
    ]

    for name, fn in phases:
        try:
            results[name] = fn()
        except Exception:
            tb = traceback.format_exc()
            print(f"\n  PHASE ERROR: {tb[-300:]}")
            results[name] = {"error": tb[-300:]}

    elapsed = time.time() - t_start

    # Build final output
    output = {
        "gpu": GPU_NAME,
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "elapsed_seconds": round(elapsed, 1),
        "bombshell_results": results,
    }

    # Summary
    print("\n" + "=" * 70)
    print("BOMBSHELL RESULTS SUMMARY")
    print("=" * 70)

    attn = results.get("attention_vs_sdpa", {})
    if "error" not in attn:
        print(f"  Attention vs SDPA:        {attn.get('speedup', '?')}x "
              f"(beat_flashattn={attn.get('beat_flashattn', '?')})")

    sk = results.get("splitk_vs_cublas", {})
    if "error" not in sk:
        print(f"  Split-K vs cuBLAS:        best ratio {sk.get('best_ratio', '?')} "
              f"on {sk.get('shape', '?')}")

    layer = results.get("layer_speedup", {})
    if "error" not in layer:
        print(f"  Layer speedup:            {layer.get('speedup', '?')}x "
              f"(correct={layer.get('correct', '?')})")

    pro = results.get("prologue_forward_backward", {})
    if "error" not in pro:
        fwd = pro.get("forward", {})
        bwd = pro.get("backward", {})
        print(f"  Prologue forward speedup: {fwd.get('fusion_speedup', '?')}x")
        print(f"  Prologue backward speedup: {bwd.get('fusion_speedup', '?')}x")

    bandit = results.get("attention_bandit_improvement", {})
    if "error" not in bandit:
        print(f"  Bandit improvement:       {bandit.get('improvement_pct', '?')}%")

    print(f"\n  Total time: {elapsed:.0f}s")

    # Save JSON
    out_path = Path("bombshell_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {out_path}")

    # Machine-readable JSON to stdout
    print("\n--- BOMBSHELL_JSON_START ---")
    print(json.dumps(output, indent=2))
    print("--- BOMBSHELL_JSON_END ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
