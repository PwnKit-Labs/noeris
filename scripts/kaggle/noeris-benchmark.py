#!/usr/bin/env python3
"""Noeris GPU benchmark on Kaggle T4.

Validates all 14 operators + runs bandit search on key operators.
Push via: KAGGLE_API_TOKEN=... kaggle kernels push -p scripts/kaggle/
"""
import subprocess
import sys
import os

# Install noeris
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "git+https://github.com/peaktwilight/noeris",
                       "numpy", "scikit-learn", "-q"])

# Clone for scripts access
subprocess.check_call(["git", "clone", "https://github.com/peaktwilight/noeris", "/tmp/noeris"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("=" * 60)
print("PHASE 1: Validate all 14 operators")
print("=" * 60)
subprocess.run([sys.executable, "/tmp/noeris/scripts/colab_validate_all.py"])

print("\n" + "=" * 60)
print("PHASE 2: Bandit search on key operators (5 iter × 10 configs)")
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
print("DONE — Results in /tmp/noeris/.noeris/colab-configs.json")
print("=" * 60)

# Copy results to Kaggle output
import shutil
if os.path.exists("/tmp/noeris/.noeris/colab-configs.json"):
    shutil.copy("/tmp/noeris/.noeris/colab-configs.json", "/kaggle/working/colab-configs.json")
    print("Config DB saved to /kaggle/working/colab-configs.json")
if os.path.exists("/tmp/noeris/colab_validation_results.json"):
    shutil.copy("/tmp/noeris/colab_validation_results.json", "/kaggle/working/validation_results.json")
