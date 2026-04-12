#!/usr/bin/env python3
"""Noeris GPU benchmark on Kaggle T4.

Validates all 14 operators + runs bandit search on key operators.
Push via: KAGGLE_API_TOKEN=... kaggle kernels push -p scripts/kaggle/
"""
import subprocess
import sys
import os

# Install from Kaggle dataset (no internet needed)
import zipfile, glob
# Debug: list what's in the input directory
print("Kaggle input contents:")
for p in glob.glob("/kaggle/input/**/*", recursive=True):
    print(f"  {p}")

# Try multiple possible paths
dataset_zip = None
for candidate in [
    "/kaggle/input/noeris-source-code/noeris-code.zip",
    "/kaggle/input/noeris-source-code.zip",
    "/kaggle/input/noeris-source-code/src",  # auto-extracted
]:
    if os.path.exists(candidate):
        dataset_zip = candidate
        break

if dataset_zip and dataset_zip.endswith(".zip"):
    print("Installing from Kaggle dataset (offline)...")
    with zipfile.ZipFile(dataset_zip, 'r') as z:
        z.extractall("/tmp/noeris")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-e", "/tmp/noeris", "numpy", "scikit-learn", "-q"])
elif dataset_zip and os.path.isdir(dataset_zip):
    # Dataset was auto-extracted by Kaggle
    print(f"Installing from auto-extracted dataset at {dataset_zip}...")
    # The extracted root should contain pyproject.toml
    extract_root = os.path.dirname(dataset_zip)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-e", extract_root, "numpy", "scikit-learn", "-q"])
    os.makedirs("/tmp/noeris", exist_ok=True)
    import shutil
    for item in ["src", "scripts", "pyproject.toml"]:
        src_path = os.path.join(extract_root, item)
        dst_path = os.path.join("/tmp/noeris", item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        elif os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
else:
    # Fallback: clone from GitHub (needs internet/phone verification)
    print(f"Dataset not found at any candidate path, cloning from GitHub...")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/PwnKit-Labs/noeris.git", "/tmp/noeris"],
                   check=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-e", "/tmp/noeris", "numpy", "scikit-learn", "-q"])

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
