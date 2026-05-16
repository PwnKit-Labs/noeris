"""Validate all Noeris operators on a free T4 GPU (Kaggle or Google Colab).

Primary platform: Kaggle (30 hr/week free T4, API-driven via `kaggle kernels push`).
Backup platform: Google Colab (~4-5 hr/day free T4).

Usage on Kaggle:
  1. New Notebook → Settings → GPU T4 x2, enable Internet
  2. !git clone https://github.com/0sec-labs/noeris && cd noeris && pip install -e . numpy scikit-learn -q
  3. !cd noeris && python scripts/colab_validate_all.py

Usage on Colab:
  1. Open https://colab.research.google.com
  2. New notebook → Runtime → Change runtime type → T4 GPU
  3. Same commands as above.

No Modal needed. No billing. Free T4 GPU.
"""
import json
import sys
import time
from pathlib import Path

# Ensure we can import research_engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: No GPU available. Change Colab runtime to T4 GPU.")
    sys.exit(1)

import triton
print(f"Triton {triton.__version__}")

from research_engine.triton_operators import REGISTRY

ops = sorted(REGISTRY.names())
print(f"\n{'='*70}")
print(f"NOERIS OPERATOR VALIDATION — {len(ops)} operators on {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

results = []

for op_name in ops:
    spec = REGISTRY.get(op_name)
    configs = spec.curated_configs[:1]  # 1 config
    shapes = spec.shape_buckets[:1]     # 1 shape (smallest)

    # Generate the benchmark script
    script = spec.benchmark_script_fn(configs, shapes)

    print(f"  {op_name:20s} ... ", end="", flush=True)
    t0 = time.time()

    try:
        # Execute the generated script in a subprocess to isolate Triton JIT state
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            proc = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=120,
            )

        elapsed = time.time() - t0

        if proc.returncode != 0:
            # Check if it's a compile/runtime error
            err = proc.stderr[-300:] if proc.stderr else "unknown"
            print(f"FAIL ({elapsed:.1f}s) — {err[:100]}")
            results.append({"op": op_name, "ok": False, "error": err[:200]})
            continue

        # Parse the JSON output
        stdout = proc.stdout
        start = stdout.find("{")
        if start < 0:
            print(f"FAIL ({elapsed:.1f}s) — no JSON output")
            results.append({"op": op_name, "ok": False, "error": "no JSON"})
            continue

        payload = json.loads(stdout[start:])
        cfg_results = payload.get("config_results", [])

        any_correct = False
        best_metric = None
        best_shape = None
        fusion_speedup = None

        for cfg in cfg_results:
            for r in cfg.get("results", []):
                if r.get("correct"):
                    any_correct = True
                    m = r.get("gb_per_s") or r.get("tflops")
                    if m and (best_metric is None or m > best_metric):
                        best_metric = m
                        best_shape = r.get("shape_name", "?")
                        fusion_speedup = r.get("fusion_speedup")

        if any_correct:
            fs_str = f" fusion={fusion_speedup:.2f}x" if fusion_speedup else ""
            print(f"OK    ({elapsed:.1f}s) — {best_shape}: {best_metric} GB/s{fs_str}")
            results.append({"op": op_name, "ok": True, "metric": best_metric,
                           "shape": best_shape, "fusion_speedup": fusion_speedup})
        else:
            max_err = None
            for cfg in cfg_results:
                for r in cfg.get("results", []):
                    e = r.get("max_err")
                    if e is not None:
                        max_err = e
            err_str = f"max_err={max_err}" if max_err else "unknown correctness failure"
            print(f"FAIL  ({elapsed:.1f}s) — {err_str}")
            results.append({"op": op_name, "ok": False, "error": err_str})

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (120s)")
        results.append({"op": op_name, "ok": False, "error": "timeout"})
    except Exception as e:
        print(f"ERROR — {e}")
        results.append({"op": op_name, "ok": False, "error": str(e)[:200]})

# Summary
passed = sum(1 for r in results if r["ok"])
total = len(results)
print(f"\n{'='*70}")
print(f"RESULTS: {passed}/{total} operators validated on {torch.cuda.get_device_name(0)}")
print(f"{'='*70}")

for r in results:
    status = "OK  " if r["ok"] else "FAIL"
    detail = ""
    if r["ok"]:
        detail = f"{r.get('metric', '?')} GB/s"
        if r.get("fusion_speedup"):
            detail += f" (fusion {r['fusion_speedup']:.2f}x)"
    else:
        detail = r.get("error", "?")[:80]
    print(f"  [{status}] {r['op']:20s} {detail}")

# Save results
out_path = Path("colab_validation_results.json")
out_path.write_text(json.dumps({
    "gpu": torch.cuda.get_device_name(0),
    "pytorch": torch.__version__,
    "triton": triton.__version__,
    "passed": passed,
    "total": total,
    "results": results,
}, indent=2))
print(f"\nResults saved to {out_path}")
