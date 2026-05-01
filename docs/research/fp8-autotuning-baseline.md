# FP8 Hopper Baseline Plan

This note defines the first reproducible baseline pass for issue #50 (FP8 kernel autotuning on Hopper).

## Why now

- Noeris currently has no published FP8 operator benchmark path.
- Hopper-class hardware (H100) is where FP8 is operationally relevant.
- Before autotuning, we need a capability and behavior probe with artifacts.

## Baseline workflow

1. Run the FP8 probe on H100:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_hopper_probe.py
```

2. Save artifact:

- `docs/results/fp8-hopper-probe.json`

3. Confirm:

- Which FP8 dtypes exist (`float8_e4m3fn`, `float8_e5m2`)
- Whether `torch.matmul` accepts them end-to-end in this runtime
- Any runtime error signatures to account for in future kernels

4. Probe Triton-native FP8 matmul viability on H100:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_triton_matmul_probe.py
```

Artifacts:

- `docs/results/fp8-triton-matmul-probe-h100.json`
- `docs/results/fp8-triton-matmul-probe-h100.md`

Interpretation:

- If torch `addmm` is unavailable for float8 but Triton kernel probe passes,
  FP8 autotuning should proceed through Triton kernel paths rather than torch
  eager matmul baselines.

## Next implementation steps

- Add FP8 shape bucket(s) for matmul in a dedicated experiment lane.
- Add a minimal FP8 matmul benchmark script generator (parallel to fp16 path).
- Add A100 vs H100 sanity comparison (expect H100-only viability for serious results).
- Only after baseline correctness: add bandit/autotune loop for FP8 configs.

## First autotune lane artifact

Initial curated FP8 Triton config sweep on H100:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_triton_matmul_autotune_h100.py
```

Outputs:

- `docs/results/fp8-triton-matmul-autotune-h100.json`
- `docs/results/fp8-triton-matmul-autotune-h100.md`

Current best (first sweep):

- config: `bm128_bn64_bk64_w8_s3`
- `fp8_mm_1024`: `56.44 TFLOPS` (`0.585x` vs fp16 baseline)
- `fp8_mm_2048x1024x2048`: `134.89 TFLOPS` (`0.419x` vs fp16 baseline)

Interpretation (updated):

- This first FP8 lane is correctness-stable but not yet throughput-competitive with eager fp16 matmul on H100.
- Next lane should focus on better FP8 tensor-core utilization (layout/blocking/pipeline changes), not just wider config sweeps.

## Second autotune lane (v2): grouped launch + expanded shapes

Expanded sweep with grouped PID ordering and larger BLOCK_K/BLOCK_N search space:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_triton_matmul_autotune_h100_v2.py
```

Outputs:

- `docs/results/fp8-triton-matmul-autotune-h100-v2.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v2.md`

Best per-shape results (v2):

- `fp8_mm_1024`: `66.51 TFLOPS` (`0.609x` vs fp16 baseline)
- `fp8_mm_2048x1024x2048`: `171.20 TFLOPS` (`0.511x` vs fp16 baseline)
- `fp8_mm_4096x4096x4096`: `298.80 TFLOPS` (`0.409x` vs fp16 baseline)

Winning config IDs (v2):

- `fp8_mm_1024`: `bm128_bn64_bk64_w8_s3_g8`
- `fp8_mm_2048x1024x2048`: `bm256_bn64_bk128_w8_s4_g8`
- `fp8_mm_4096x4096x4096`: `bm256_bn64_bk128_w8_s4_g8`

Interpretation (v2):

- v2 materially improves absolute FP8 throughput over v1 but still trails eager fp16 on these shapes.
- The current limiter is likely kernel design (data movement/layout strategy), not only config search coverage.

## Third autotune lane (v3): B layout variants (`kn` vs prepacked `nk`)

Layout-variant sweep with the same grouped launch family, comparing standard KxN B
layout against prepacked NxK B layout:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_triton_matmul_autotune_h100_v3.py
```

Outputs:

- `docs/results/fp8-triton-matmul-autotune-h100-v3.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v3.md`

Best per-shape results (v3):

- `fp8_mm_1024`: `83.57 TFLOPS` (`0.811x` vs fp16 baseline), winner layout `nk`
- `fp8_mm_2048x1024x2048`: `258.36 TFLOPS` (`0.794x` vs fp16 baseline), winner layout `nk`
- `fp8_mm_4096x4096x4096`: `736.57 TFLOPS` (`1.025x` vs fp16 baseline), winner layout `nk`

Interpretation (v3):

- Prepacked `nk` layout is a clear win over `kn` for all tested shapes in this lane.
- At large shape (`4096^3`), FP8 now slightly exceeds eager fp16 in this measurement setup.
- Reported numbers exclude one-time B transpose/prepack cost and should be interpreted
  for inference-style scenarios with reusable static weights.

## Fourth autotune lane (v4): split-K follow-up on `nk`

Split-K sweep on top of `nk`-style kernels, including a larger K stress shape:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_triton_matmul_autotune_h100_v4_splitk.py
```

Outputs:

- `docs/results/fp8-triton-matmul-autotune-h100-v4-splitk.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v4-splitk.md`

Best per-shape results (v4):

- `fp8_mm_1024`: `45.59 TFLOPS` (`0.401x` vs fp16), best `split_k=2`
- `fp8_mm_2048x1024x2048`: `160.64 TFLOPS` (`0.470x` vs fp16), best `split_k=1`
- `fp8_mm_4096x4096x4096`: `410.73 TFLOPS` (`0.565x` vs fp16), best `split_k=1`
- `fp8_mm_2048x2048x8192`: `486.08 TFLOPS` (`0.710x` vs fp16), best `split_k=1`

Interpretation (v4):

- In this implementation, split-K does not improve throughput; non-split (`split_k=1`) dominates for large shapes.
- FP32 atomic accumulation overhead appears to outweigh occupancy gains for this lane.

## Prepack amortization model for v3 (`kn` vs `nk`)

To quantify the practical impact of one-time B prepack cost:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_prepack_amortization_h100.py
```

Outputs:

- `docs/results/fp8-prepack-amortization-h100.json`
- `docs/results/fp8-prepack-amortization-h100.md`

Break-even reuse counts (H100):

- `fp8_mm_1024`: `2` runs
- `fp8_mm_2048x1024x2048`: `2` runs
- `fp8_mm_4096x4096x4096`: `1` run

Interpretation:

- `nk` prepack is quickly amortized in inference-like scenarios with reused weights.
- This supports treating `nk` as the default FP8 runtime layout path when weight reuse is expected.

## Runtime reuse policy artifact (H100)

To make deployment decisions explicit for expected weight reuse counts:

```bash
python3 scripts/fp8_layout_reuse_policy_h100.py
```

Outputs:

- `docs/results/fp8-layout-reuse-policy-h100.json`
- `docs/results/fp8-layout-reuse-policy-h100.md`

Policy summary from current artifacts:

- `fp8_mm_1024`: use `kn` when reuse=`1`, use `nk` when reuse>=`2`
- `fp8_mm_2048x1024x2048`: use `kn` when reuse=`1`, use `nk` when reuse>=`2`
- `fp8_mm_4096x4096x4096`: use `nk` even at reuse=`1`

This gives a concrete runtime rule for layout choice rather than a fixed
one-size-fits-all default.

Reference runtime helper:

- `src/research_engine/fp8_layout_policy.py` provides `choose_layout_from_policy(...)`
  to apply this artifact-driven decision at runtime.

Additional runtime resolve helper:

- `src/research_engine/fp8_runtime.py` provides `resolve_fp8_layout(...)`
  with `prefer={kn,nk,auto}` and policy-backed auto behavior.

## Runtime integration benchmark (policy-backed auto)

Benchmark command:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_layout_runtime_integration_benchmark.py
```

Outputs:

- `docs/results/fp8-layout-runtime-integration-h100.json`
- `docs/results/fp8-layout-runtime-integration-h100.md`

Current result highlights:

- `auto` follows policy exactly (`auto/policy = 1.0` for all tested shapes).
- For `fp8_mm_1024` at reuse=`1`, `auto` selects `kn` per amortization policy even though raw kernel-only latency favors `nk`.
- For larger/reused workloads (`fp8_mm_2048x1024x2048` at reuse=`2`, `fp8_mm_4096x4096x4096` at reuse=`1`), `auto` selects `nk`.

Interpretation:

- Runtime layout choice should be evaluated against *effective* latency (including one-time prepack where relevant), not kernel-only latency.

## Runtime cache integration benchmark

Benchmark command:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/fp8_layout_runtime_cache_benchmark.py
```

Outputs:

- `docs/results/fp8-layout-runtime-cache-integration-h100.json`
- `docs/results/fp8-layout-runtime-cache-integration-h100.md`

Implementation helpers:

- `src/research_engine/fp8_prepack_cache.py` (LRU prepack cache)
- `src/research_engine/fp8_runtime.py` (policy-backed layout resolve)
- `src/research_engine/executors.py` (`MatmulPythonExecutor` now annotates FP8 fixture rows with
  policy-resolved runtime layout decisions)

Current workload findings (H100):

- Unique, no-reuse small-shape workload (`s1024_reuse1_unique`): `auto_policy_cache` is best and effectively matches `force_kn`.
- Hotset high-reuse workload (`s1024_reuse8_hotset`): `force_nk_cache` is best; `auto_policy_cache` is within `1.01x`.
- Moderate-reuse larger workload (`s2048_reuse2`): `auto_policy_cache` is best.

Interpretation:

- The policy+cache runtime path is robust across mixed reuse regimes and avoids catastrophic choices
  (e.g. forcing `nk` cache on unique weights).
- Remaining small overhead vs hand-forced best mode on specific hotset workloads is minimal.

## Executor artifact integration

`MatmulPythonExecutor` now emits `fp8-runtime-layout-summary.json` in experiment artifacts
when FP8 fixtures are included. This provides a run-history-friendly summary of:

- selected FP8 layouts (`kn`/`nk`),
- weighted workload share by layout,
- per-fixture reuse assumptions and chosen layout.

`JsonFileRunStore.summarize_history(...)` now aggregates FP8 trend metrics across runs,
including policy-alignment rates:

- `overall_nk_rate`
- `reuse_1_kn_rate`
- `reuse_2_4_nk_rate`
- `reuse_5_plus_nk_rate`

`history-brief` now also surfaces policy-regression warnings when these rates
drop materially versus the previous run.

`export-history` now emits regression-focused artifacts alongside the standard
summary/brief outputs:

- `history-regressions.json`
- `history-regressions.md`

## Exit criteria for baseline phase

- A reproducible JSON artifact exists in `docs/results/`.
- We have a clear yes/no answer on runtime FP8 matmul viability on H100.
- We have concrete error constraints documented before writing FP8 Triton kernels.
