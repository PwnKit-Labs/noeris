#!/usr/bin/env python3
"""Noeris GPU benchmark on Kaggle T4.

Toggle phases via NOERIS_PHASES env var (comma-separated, e.g. "1,6,8").
Default: all phases. Set NOERIS_PHASES=fast for just validation + new stuff.

Push via: KAGGLE_API_TOKEN=... kaggle kernels push -p scripts/kaggle/
"""
import subprocess
import sys
import os
import shutil

# ============================================================================
# Phase toggle: set NOERIS_PHASES env var to pick which phases run
# Examples:
#   NOERIS_PHASES=1,8,9      -> validation + adaptive + LLM dry run
#   NOERIS_PHASES=fast        -> 1,6,7,8,9 (skip slow bandit/bombshell)
#   NOERIS_PHASES=all         -> everything (default)
#   NOERIS_PHASES=new         -> 8,9 only (just the new stuff)
# ============================================================================
PHASE_PRESETS = {
    "all": {1, 2, 3, 4, 5, 6, 7, 8, 9},
    "fast": {1, 6, 7, 8, 9},
    "new": {8, 9},
    "paper": {1, 4, 5, 6, 7},
    "search": {1, 2, 3},
    "novel": {1, 10, 11, 12, 13, 14, 15},
}

phases_env = os.environ.get("NOERIS_PHASES", "novel").strip().lower()
if phases_env in PHASE_PRESETS:
    ACTIVE_PHASES = PHASE_PRESETS[phases_env]
else:
    ACTIVE_PHASES = {int(p.strip()) for p in phases_env.split(",") if p.strip().isdigit()}

if not ACTIVE_PHASES:
    ACTIVE_PHASES = PHASE_PRESETS["all"]

print(f"Active phases: {sorted(ACTIVE_PHASES)}")

# Clone and install noeris
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/0sec-labs/noeris.git", "/tmp/noeris"],
               check=True)
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "-e", "/tmp/noeris", "numpy", "scikit-learn", "-q"])

# ============================================================================
# Phases
# ============================================================================

if 1 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 1: Validate all operators")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_validate_all.py"])

if 2 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 2: Bandit search on key operators (3 iter x 8 configs)")
    print("=" * 60)
    for op in ["qk_norm_rope", "rmsnorm", "softmax", "geglu", "cross_entropy",
               "layernorm", "rotary", "qk_norm_rope_bwd"]:
        print(f"\n--- {op} ---")
        subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_iterate.py",
                        "--operator", op, "--iterations", "3",
                        "--configs-per-iter", "8", "--shapes", "standard"])

if 3 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 3: Attention decode search")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_iterate.py",
                    "--operator", "attention_decode", "--iterations", "3",
                    "--configs-per-iter", "8", "--shapes", "full"])

if 4 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 4: Bombshell benchmark (full layer + T4 shootout)")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_bombshell.py"])

if 5 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 5: Compiler failure analysis (torch.compile vs Noeris)")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/compiler_analysis.py"])

if 6 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 6: Multi-model fusion benchmark (19 models)")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/multi_model_fusion_benchmark.py"])

if 7 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 7: End-to-end 26-layer Gemma 4 forward pass")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/end_to_end_layer_stack.py"])

if 8 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 8: Adaptive config benchmark")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/adaptive_benchmark.py"])

if 9 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 9: LLM kernel search (dry run)")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/llm_kernel_search.py",
                    "--operator", "rmsnorm", "--dry-run", "--variants", "3"])

if 10 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 10: Sliding-window showdown (Noeris vs SDPA)")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/sliding_window_showdown.py"])

if 11 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 11: Fused norm+matmul validation")
    print("=" * 60)
    subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0,'/tmp/noeris/src'); "
        "from research_engine.triton_operators import REGISTRY; "
        "spec = REGISTRY.get('fused_norm_linear'); "
        "script = spec.benchmark_script_fn(spec.curated_configs[:2], spec.shape_buckets[:2]); "
        "exec(script)"])

if 12 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 12: Bandit convergence experiment")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/convergence_experiment.py"])

if 13 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 13: PLE fusion benchmark")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/ple_fusion_benchmark.py"])

if 14 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 14: K=V shared attention benchmark")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/kv_shared_benchmark.py"])

if 15 in ACTIVE_PHASES:
    print("\n" + "=" * 60)
    print("PHASE 15: Mamba-3 SSM scan benchmark")
    print("=" * 60)
    subprocess.run([sys.executable, "/tmp/noeris/scripts/ssm_scan_benchmark.py"])

# Phase 16 removed: autotune_comparison.py was broken (do_bench L2 flush issue).
# The fair comparison is Phase 12 (convergence_experiment.py).

print("\n" + "=" * 60)
print(f"DONE — Ran phases {sorted(ACTIVE_PHASES)}")
print("=" * 60)

# ============================================================================
# Copy results to Kaggle output
# ============================================================================
RESULT_FILES_DIR = "/tmp/noeris/results"
if os.path.isdir(RESULT_FILES_DIR):
    for f in os.listdir(RESULT_FILES_DIR):
        shutil.copy(os.path.join(RESULT_FILES_DIR, f), f"/kaggle/working/{f}")
        print(f"Saved {f}")

RESULT_FILES = {
    "/tmp/noeris/.noeris/colab-configs.json": "colab-configs.json",
    "/tmp/noeris/colab_validation_results.json": "validation_results.json",
    "/tmp/noeris/bombshell_results.json": "bombshell_results.json",
    "/tmp/noeris/compiler_analysis_results.json": "compiler_analysis_results.json",
    "/tmp/noeris/end_to_end_results.json": "end_to_end_results.json",
    "/tmp/noeris/adaptive_benchmark_results.json": "adaptive_benchmark_results.json",
    "/tmp/noeris/multi_model_results.json": "multi_model_results.json",
}

for src, dst in RESULT_FILES.items():
    if os.path.exists(src):
        shutil.copy(src, f"/kaggle/working/{dst}")
        print(f"Saved {dst}")
