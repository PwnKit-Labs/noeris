#!/usr/bin/env python3
"""Compare @triton.autotune vs Noeris bandit vs fixed curated config.

Uses FAIR measurement: all three methods timed with the SAME cuda_event_timer
that flushes L2 cache between trials (matches KernelBench methodology).
Previous version used triton.testing.do_bench (no L2 flush), producing
inflated numbers (e.g. 867 GB/s rmsnorm on T4 whose peak is 320 GB/s).

Usage::
    python3 scripts/autotune_comparison.py [--output autotune_comparison.json]
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_engine.bandit_selector import BanditSelector
from research_engine.triton_kernels import ConfigDatabase
from research_engine.triton_operators import REGISTRY

OPERATORS = ["rmsnorm", "qk_norm_rope", "geglu"]
BANDIT_ITERATIONS, CONFIGS_PER_ITER, EXHAUSTIVE_MAX = 3, 6, 50

# Shared L2-flushing timer embedded into every subprocess for fairness.
TIMER = '''
def cuda_event_timer(fn, warmup=5, trials=20):
    import torch; torch.cuda.synchronize()
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        _f = torch.empty(40*1024*1024//4, dtype=torch.float32, device="cuda"); _f.zero_(); del _f
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort(); return times[len(times)//2]
'''

# --- Kernel source fragments (shared between autotune + fair-timing scripts) ---
_KERN_RMSNORM = '''
def kern(x_ptr, w_ptr, y_ptr, x_rs, y_rs, n_cols, eps,
         BLOCK_SIZE: tl.constexpr, AFFINE_MODE: tl.constexpr):
    r = tl.program_id(0); x_ptr += r*x_rs; y_ptr += r*y_rs
    o = tl.arange(0, BLOCK_SIZE); m = o < n_cols
    x = tl.load(x_ptr+o, mask=m, other=0.).to(tl.float32)
    rstd = 1./tl.sqrt(tl.sum(x*x,axis=0)/n_cols + eps)
    w = tl.load(w_ptr+o, mask=m, other=0.).to(tl.float32)
    y = x*rstd*w if AFFINE_MODE==0 else x*rstd*(1.+w)
    tl.store(y_ptr+o, y.to(tl.float16), mask=m)
'''
_KERN_GEGLU = '''
def kern(gate_ptr, up_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0); gp = gate_ptr+r*n_cols; up2 = up_ptr+r*n_cols; op = out_ptr+r*n_cols
    o = tl.arange(0, BLOCK_SIZE); m = o < n_cols
    gate = tl.load(gp+o, mask=m, other=0.).to(tl.float32)
    up = tl.load(up2+o, mask=m, other=0.).to(tl.float32)
    inner = 0.7978845608028654*(up + 0.044715*up*up*up)
    gelu_up = 0.5*up*(1.+tl.extra.libdevice.tanh(inner))
    tl.store(op+o, (gate*gelu_up).to(tl.float16), mask=m)
'''
_KERN_QK = '''
def kern(x_ptr, scale_ptr, cos_ptr, sin_ptr, out_ptr,
         row_stride, heads, seq_len, head_dim, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0); s_idx = pid % seq_len
    xb = x_ptr+pid*row_stride; ob = out_ptr+pid*row_stride; half = head_dim//2
    o = tl.arange(0, BLOCK_SIZE); m = o < half
    xe = tl.load(xb+2*o, mask=m, other=0.).to(tl.float32)
    xo = tl.load(xb+2*o+1, mask=m, other=0.).to(tl.float32)
    rstd = 1./tl.sqrt((tl.sum(xe*xe,axis=0)+tl.sum(xo*xo,axis=0))/head_dim + eps)
    se = tl.load(scale_ptr+2*o, mask=m, other=0.).to(tl.float32)
    so = tl.load(scale_ptr+2*o+1, mask=m, other=0.).to(tl.float32)
    ne, no2 = xe*rstd*(1.+se), xo*rstd*(1.+so)
    c = tl.load(cos_ptr+s_idx*half+o, mask=m, other=1.).to(tl.float32)
    sn = tl.load(sin_ptr+s_idx*half+o, mask=m, other=0.).to(tl.float32)
    tl.store(ob+2*o, (ne*c-no2*sn).to(tl.float16), mask=m)
    tl.store(ob+2*o+1, (ne*sn+no2*c).to(tl.float16), mask=m)
'''
_KERNS = {"rmsnorm": _KERN_RMSNORM, "geglu": _KERN_GEGLU, "qk_norm_rope": _KERN_QK}

def _parse_shape(op, s):
    p = s.split("x")
    try:
        if op == "rmsnorm": return {"n_rows": int(p[0]), "hidden_dim": int(p[1])}
        if op == "geglu": return {"n_rows": int(p[0]), "ffn_dim": int(p[1])}
        if op == "qk_norm_rope":
            return {"batch": int(p[0]), "heads": int(p[1]), "num_kv_heads": int(p[2]),
                    "seq": int(p[3]), "head_dim": int(p[4])}
    except (ValueError, IndexError): pass
    return None

def _run_script(script: str, timeout: int = 600) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script); sp = f.name
    try:
        proc = subprocess.run([sys.executable, sp], capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            print(f"    [WARN] subprocess failed: {proc.stderr[:300]}"); return {}
        return json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"    [WARN] error: {e}"); return {}
    finally: Path(sp).unlink(missing_ok=True)

def run_benchmark(spec, configs, shapes):
    script = spec.benchmark_script_fn(configs, shapes)
    data = _run_script(script, timeout=300)
    return data.get("config_results", [])

def record_results(spec, db, config_results, hardware, operator):
    best, best_cfg = 0.0, {}
    for cr in config_results:
        cid, cfg = cr.get("config_id", ""), cr.get("config", {})
        for sr in cr.get("results", []):
            if not sr.get("correct") or not sr.get("tflops"): continue
            parsed = _parse_shape(operator, sr.get("shape", ""))
            if not parsed: continue
            db.record_result(shape=parsed, hardware=hardware, config=cfg, tflops=sr["tflops"],
                ms=sr.get("ms", 0), correct=True, run_id=cid, operator=operator,
                bucket=spec.shape_bucket_fn(parsed), config_id_str=cid)
            if sr["tflops"] > best: best, best_cfg = sr["tflops"], cfg
    db.save(); return best, best_cfg

# ---------------------------------------------------------------------------
# Autotune scripts: trigger autotune selection, then time winner with fair timer
# ---------------------------------------------------------------------------
def _autotune_harness(op, kern_src, configs_json, shapes_json):
    """Generate operator-specific autotune+fair-timing subprocess script."""
    if op == "rmsnorm":
        ac_expr = '''[triton.Config({"BLOCK_SIZE": c["BLOCK_SIZE"], "AFFINE_MODE": 0},
      num_warps=c["num_warps"], num_stages=c["num_stages"]) for c in CONFIGS]'''
        key, setup, launch, bw = '"n_cols"', '''
    nr, hd = sh["n_rows"], sh["hidden_dim"]
    x = torch.randn((nr,hd), device="cuda", dtype=torch.float16)
    w = torch.randn((hd,), device="cuda", dtype=torch.float16); y = torch.empty_like(x)''', \
            'lambda: kern[(nr,)](x, w, y, x.stride(0), y.stride(0), hd, 1e-6)', \
            '2*nr*hd*2+hd*2'
        shape_str = 'f"{nr}x{hd}"'
    elif op == "geglu":
        ac_expr = '''[triton.Config({"BLOCK_SIZE": c["BLOCK_SIZE"]},
      num_warps=c["num_warps"], num_stages=c["num_stages"]) for c in CONFIGS]'''
        key, setup, launch, bw = '"n_cols"', '''
    nr, fd = sh["n_rows"], sh["ffn_dim"]
    gate = torch.randn((nr,fd), device="cuda", dtype=torch.float16)
    up = torch.randn((nr,fd), device="cuda", dtype=torch.float16); out = torch.empty_like(gate)''', \
            'lambda: kern[(nr,)](gate, up, out, fd)', '3*nr*fd*2'
        shape_str = 'f"{nr}x{fd}"'
    else:  # qk_norm_rope
        ac_expr = '''[triton.Config({"BLOCK_SIZE": c["BLOCK_SIZE"]},
      num_warps=c["num_warps"], num_stages=c["num_stages"]) for c in CONFIGS]'''
        key, setup = '"head_dim"', '''
    B,H,Hk,S,D = sh["batch"],sh["heads"],sh["num_kv_heads"],sh["seq"],sh["head_dim"]
    q = torch.randn((B,H,S,D), device="cuda", dtype=torch.float16)
    k = torch.randn((B,Hk,S,D), device="cuda", dtype=torch.float16)
    cos = torch.randn((S,D//2), device="cuda", dtype=torch.float32)
    sin = torch.randn((S,D//2), device="cuda", dtype=torch.float32)
    qs = torch.randn((D,), device="cuda", dtype=torch.float32)*0.1
    ks = torch.randn((D,), device="cuda", dtype=torch.float32)*0.1
    qf, kf = q.reshape(B*H*S,D).contiguous(), k.reshape(B*Hk*S,D).contiguous()
    qo, ko = torch.empty_like(qf), torch.empty_like(kf)
    def _run():
        kern[(B*H*S,)](qf, qs, cos, sin, qo, D, H, S, D, 1e-6)
        kern[(B*Hk*S,)](kf, ks, cos, sin, ko, D, Hk, S, D, 1e-6)'''
        launch = '_run'
        bw = '2*(B*H*S*D*2+B*Hk*S*D*2)+2*S*(D//2)*4+2*D*4'
        shape_str = 'f"{B}x{H}x{Hk}x{S}x{D}"'

    return f'''#!/usr/bin/env python3
import json, time, torch, triton, triton.language as tl
{TIMER}
CONFIGS, SHAPES = {configs_json}, {shapes_json}
ac = {ac_expr}
@triton.autotune(configs=ac, key=[{key}])
@triton.jit
{kern_src}
results = []
for sh in SHAPES:{setup}
    t0 = time.perf_counter()
    ({launch})(); torch.cuda.synchronize()
    tt = time.perf_counter()-t0
    ms = cuda_event_timer({launch})
    gbs = ({bw})/(ms*1e-3)/1e9
    bc = kern.best_config
    sel = f"BS={{bc.kwargs.get('BLOCK_SIZE','?')}}, w={{bc.num_warps}}, s={{bc.num_stages}}"
    results.append({{"shape":{shape_str},"gb_per_s":round(gbs,2),
        "tune_time_s":round(tt,3),"configs_tested":len(CONFIGS),"selected_config":sel}})
print(json.dumps({{"operator":"{op}","results":results}}))
'''

# ---------------------------------------------------------------------------
# Fair timing script: time a SINGLE fixed config with cuda_event_timer
# ---------------------------------------------------------------------------
def _fair_timing_script(op, config, shapes_json):
    """Generate script to time one specific config with L2-flushing timer."""
    kern_src = _KERNS[op]
    if op == "rmsnorm":
        setup = '''
    nr, hd = sh["n_rows"], sh["hidden_dim"]
    x = torch.randn((nr,hd), device="cuda", dtype=torch.float16)
    w = torch.randn((hd,), device="cuda", dtype=torch.float16); y = torch.empty_like(x)'''
        launch = f'lambda: kern[(nr,)](x, w, y, x.stride(0), y.stride(0), hd, 1e-6, BLOCK_SIZE={config["BLOCK_SIZE"]}, AFFINE_MODE=0, num_warps={config.get("num_warps",4)}, num_stages={config.get("num_stages",1)})'
        bw, shape_str = '2*nr*hd*2+hd*2', 'f"{nr}x{hd}"'
    elif op == "geglu":
        setup = '''
    nr, fd = sh["n_rows"], sh["ffn_dim"]
    gate = torch.randn((nr,fd), device="cuda", dtype=torch.float16)
    up = torch.randn((nr,fd), device="cuda", dtype=torch.float16); out = torch.empty_like(gate)'''
        launch = f'lambda: kern[(nr,)](gate, up, out, fd, BLOCK_SIZE={config["BLOCK_SIZE"]}, num_warps={config.get("num_warps",4)}, num_stages={config.get("num_stages",1)})'
        bw, shape_str = '3*nr*fd*2', 'f"{nr}x{fd}"'
    else:  # qk_norm_rope
        setup = '''
    B,H,Hk,S,D = sh["batch"],sh["heads"],sh["num_kv_heads"],sh["seq"],sh["head_dim"]
    q = torch.randn((B,H,S,D), device="cuda", dtype=torch.float16)
    k = torch.randn((B,Hk,S,D), device="cuda", dtype=torch.float16)
    cos = torch.randn((S,D//2), device="cuda", dtype=torch.float32)
    sin = torch.randn((S,D//2), device="cuda", dtype=torch.float32)
    qs = torch.randn((D,), device="cuda", dtype=torch.float32)*0.1
    ks = torch.randn((D,), device="cuda", dtype=torch.float32)*0.1
    qf, kf = q.reshape(B*H*S,D).contiguous(), k.reshape(B*Hk*S,D).contiguous()
    qo, ko = torch.empty_like(qf), torch.empty_like(kf)
    def _run():
        kern[(B*H*S,)](qf, qs, cos, sin, qo, D, H, S, D, 1e-6, BLOCK_SIZE={config["BLOCK_SIZE"]}, num_warps={config.get("num_warps",4)}, num_stages={config.get("num_stages",1)})
        kern[(B*Hk*S,)](kf, ks, cos, sin, ko, D, Hk, S, D, 1e-6, BLOCK_SIZE={config["BLOCK_SIZE"]}, num_warps={config.get("num_warps",4)}, num_stages={config.get("num_stages",1)})'''
        launch = '_run'
        bw = '2*(B*H*S*D*2+B*Hk*S*D*2)+2*S*(D//2)*4+2*D*4'
        shape_str = 'f"{B}x{H}x{Hk}x{S}x{D}"'

    return f'''#!/usr/bin/env python3
import json, torch, triton, triton.language as tl
{TIMER}
SHAPES = {shapes_json}
@triton.jit
{kern_src}
results = []
for sh in SHAPES:{setup}
    ms = cuda_event_timer({launch})
    gbs = ({bw})/(ms*1e-3)/1e9
    results.append({{"shape":{shape_str},"gb_per_s":round(gbs,2)}})
print(json.dumps({{"results":results}}))
'''

def fair_time_config(op, config, shapes):
    """Time a single config with fair L2-flushing timer via subprocess."""
    script = _fair_timing_script(op, config, json.dumps(shapes))
    data = _run_script(script, timeout=300)
    best = 0.0
    for r in data.get("results", []):
        if r.get("gb_per_s", 0) > best: best = r["gb_per_s"]
    return best

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(output_path: str) -> int:
    import torch
    hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "T4"
    print(f"Hardware: {hw}")
    print("Timer: cuda_event_timer with L2 flush (40 MiB memset), median of 20 trials\n")
    all_data: dict[str, Any] = {"hardware": hw, "timer": "cuda_event_timer_l2flush", "operators": {}}

    for operator in OPERATORS:
        spec = REGISTRY.get(operator)
        shapes = spec.shape_buckets[:2]
        snames = [s.get("name", "?") for s in shapes]
        n_grid = len(spec.grid_generator_fn(max_configs=EXHAUSTIVE_MAX))
        print(f"{'='*60}\nOperator: {operator}  |  Shapes: {snames}")

        # --- 1. @triton.autotune: exhaustive search + fair timer ---
        print(f"\n  [1/3] @triton.autotune ({n_grid} configs)...")
        cfgs = spec.grid_generator_fn(max_configs=EXHAUSTIVE_MAX)
        script = _autotune_harness(operator, _KERNS[operator],
                                   json.dumps(cfgs), json.dumps(shapes))
        at_gbps, at_tt, at_sel = 0.0, 0.0, ""
        data = _run_script(script)
        for r in data.get("results", []):
            if r.get("gb_per_s", 0) > at_gbps:
                at_gbps, at_sel = r["gb_per_s"], r.get("selected_config", "")
            at_tt += r.get("tune_time_s", 0)
        print(f"    Best: {at_gbps:.1f} GB/s, tune: {at_tt:.1f}s, config: {at_sel}")

        # --- 2. Noeris bandit: explore via subprocess, then fair-time winner ---
        print(f"\n  [2/3] Noeris bandit ({BANDIT_ITERATIONS}x{CONFIGS_PER_ITER} configs)...")
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        dbp = Path(tmp.name); tmp.close()
        bn_tt, bn_n, bn_cfg, bn_explore_best = 0.0, 0, {}, 0.0
        try:
            db, bandit = ConfigDatabase(path=dbp), BanditSelector(seed=42)
            for it in range(BANDIT_ITERATIONS):
                cfgs = bandit.select_configs(spec=spec, database=db, hardware=hw,
                                             shapes=shapes, max_configs=CONFIGS_PER_ITER)
                t0 = time.perf_counter()
                res = run_benchmark(spec, cfgs, shapes)
                bn_tt += time.perf_counter() - t0; bn_n += len(cfgs)
                b, c = record_results(spec, db, res, hw, operator)
                if b > bn_explore_best:
                    bn_explore_best, bn_cfg = b, c
                print(f"    Iter {it+1}: tested {len(cfgs)} configs")
        finally: dbp.unlink(missing_ok=True)
        # Fair re-timing of bandit's best config
        bn_gbps = fair_time_config(operator, bn_cfg, shapes) if bn_cfg else 0.0
        bn_sel = spec.config_id_fn(bn_cfg) if bn_cfg else "?"
        print(f"    Best: {bn_gbps:.1f} GB/s, tune: {bn_tt:.1f}s, config: {bn_sel}")

        # --- 3. Fixed curated: fair-time only ---
        print(f"\n  [3/3] Fixed curated (1 config)...")
        cc = spec.curated_configs[0]
        cu_gbps = fair_time_config(operator, cc, shapes)
        cu_sel = spec.config_id_fn(cc)
        print(f"    Best: {cu_gbps:.1f} GB/s, config: {cu_sel}")

        # --- Table ---
        ref = max(at_gbps, bn_gbps, cu_gbps, 1e-9)
        hdr = f"  {'Method':<20} | {'Best GB/s':>10} | {'Configs':>7} | {'Tune(s)':>7} | Selected"
        print(f"\n{hdr}\n  {'-'*20}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")
        for lb, g, n, t, s in [("@triton.autotune", at_gbps, n_grid, at_tt, at_sel),
                                ("Noeris bandit", bn_gbps, bn_n, bn_tt, bn_sel),
                                ("Fixed curated", cu_gbps, 1, 0.0, cu_sel)]:
            print(f"  {lb:<20} | {g:>10.1f} | {n:>7} | {t:>7.1f} | {s}")

        all_data["operators"][operator] = {
            "shapes": snames,
            "autotune": {"best_gbps": at_gbps, "configs_tested": n_grid,
                         "tune_time_s": round(at_tt, 2), "selected": at_sel},
            "bandit": {"best_gbps": bn_gbps, "configs_tested": bn_n,
                       "tune_time_s": round(bn_tt, 2), "selected": bn_sel},
            "curated": {"best_gbps": cu_gbps, "configs_tested": 1,
                        "tune_time_s": 0.0, "selected": cu_sel},
        }
        print()

    # --- Global summary ---
    print(f"{'='*70}\nGLOBAL SUMMARY  (cuda_event_timer with L2 flush)\n{'='*70}")
    print(f"  {'Operator':<16} | {'AT GB/s':>8} | {'Bandit':>8} | {'ratio':>6} | {'speedup':>7}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}")
    for op, d in all_data["operators"].items():
        a, b = d["autotune"]["best_gbps"], d["bandit"]["best_gbps"]
        r = b/a*100 if a > 0 else 0
        su = d["autotune"]["tune_time_s"]/d["bandit"]["tune_time_s"] if d["bandit"]["tune_time_s"] > 0 else float("inf")
        print(f"  {op:<16} | {a:>8.1f} | {b:>8.1f} | {r:>5.0f}% | {su:>6.1f}x")
    print(f"\nBandit tests ~{BANDIT_ITERATIONS*CONFIGS_PER_ITER} configs vs autotune's ~{EXHAUSTIVE_MAX}.\n")
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_data, indent=2) + "\n")
    print(f"Saved: {out}"); return 0

def main() -> int:
    ap = argparse.ArgumentParser(description="Autotune vs bandit comparison (fair L2-flush timer)")
    ap.add_argument("--output", default="autotune_comparison.json")
    return run_experiment(ap.parse_args().output)

if __name__ == "__main__":
    sys.exit(main())
