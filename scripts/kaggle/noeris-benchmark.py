#!/usr/bin/env python3
"""Noeris GPU benchmark on Kaggle T4.

Validates all 15 operators + runs bandit search on key operators.
Push via: KAGGLE_API_TOKEN=... kaggle kernels push -p scripts/kaggle/
"""
import subprocess
import sys
import os

# Clone and install noeris
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/PwnKit-Labs/noeris.git", "/tmp/noeris"],
               check=True)
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "-e", "/tmp/noeris", "numpy", "scikit-learn", "-q"])

print("=" * 60)
print("PHASE 1: Validate all 15 operators")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_validate_all.py"])

print("\n" + "=" * 60)
print("PHASE 2: Bandit search on key operators (3 iter × 8 configs)")
print("=" * 60)
for op in ["qk_norm_rope", "rmsnorm", "softmax", "geglu", "cross_entropy",
           "layernorm", "rotary", "qk_norm_rope_bwd"]:
    print(f"\n--- {op} ---")
    subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_iterate.py",
                    "--operator", op, "--iterations", "3",
                    "--configs-per-iter", "8", "--shapes", "standard"])

print("\n" + "=" * 60)
print("PHASE 3: Attention decode search")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_iterate.py",
                "--operator", "attention_decode", "--iterations", "3",
                "--configs-per-iter", "8", "--shapes", "full"])

print("\n" + "=" * 60)
print("PHASE 4: Bombshell benchmark (full layer + T4 shootout)")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_bombshell.py"])

print("\n" + "=" * 60)
print("PHASE 5: Compiler failure analysis (torch.compile vs Noeris)")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/compiler_analysis.py"])

print("\n" + "=" * 60)
print("PHASE 6: Multi-model fusion benchmark (LLaMA/Mistral/Phi-3)")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/multi_model_fusion_benchmark.py"])

print("\n" + "=" * 60)
print("PHASE 7: End-to-end 26-layer Gemma 4 forward pass")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/end_to_end_layer_stack.py"])

print("\n" + "=" * 60)
print("DONE — Results in /tmp/noeris/.noeris/colab-configs.json")
print("=" * 60)

# Copy results to Kaggle output
import shutil
if os.path.exists("/tmp/noeris/.noeris/colab-configs.json"):
    shutil.copy("/tmp/noeris/.noeris/colab-configs.json", "/kaggle/working/colab-configs.json")
    print("Config DB saved to /kaggle/working/colab-configs.json")
if os.path.exists("/tmp/noeris/colab_validation_results.json"):
    shutil.copy("/tmp/noeris/colab_validation_results.json", "/kaggle/working/validation_results.json")
if os.path.exists("/tmp/noeris/bombshell_results.json"):
    shutil.copy("/tmp/noeris/bombshell_results.json", "/kaggle/working/bombshell_results.json")
    print("Bombshell results saved to /kaggle/working/bombshell_results.json")
if os.path.exists("/tmp/noeris/compiler_analysis_results.json"):
    shutil.copy("/tmp/noeris/compiler_analysis_results.json", "/kaggle/working/compiler_analysis_results.json")
    print("Compiler analysis saved to /kaggle/working/compiler_analysis_results.json")
if os.path.exists("/tmp/noeris/end_to_end_results.json"):
    shutil.copy("/tmp/noeris/end_to_end_results.json", "/kaggle/working/end_to_end_results.json")
    print("End-to-end results saved to /kaggle/working/end_to_end_results.json")
if os.path.exists("/tmp/noeris/multi_model_results.json"):
    shutil.copy("/tmp/noeris/multi_model_results.json", "/kaggle/working/multi_model_results.json")
    print("Multi-model results saved to /kaggle/working/multi_model_results.json")
