# Results Index

Canonical latest artifacts (current public references):

- Gemma deeper-fusion full-layer main results:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json`
- Gemma deeper-fusion stability reruns:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune-repeat2.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune-repeat3.json`
- Policy-routing sanity checks:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-policy-routing-sanity.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-policy-routing-sanity.json`

Targeted QK-norm attention reruns:

- `docs/results/bandit-qknorm-attention-a100-v3.json`
- `docs/results/bandit-qknorm-attention-a100-v3.md`

FP8 baseline probes:

- `docs/results/fp8-hopper-probe.json`
- `docs/results/fp8-triton-matmul-probe-h100.json`
- `docs/results/fp8-triton-matmul-probe-h100.md`
- `docs/results/fp8-triton-matmul-autotune-h100.json`
- `docs/results/fp8-triton-matmul-autotune-h100.md`
- `docs/results/fp8-triton-matmul-autotune-h100-v2.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v2.md`
- `docs/results/fp8-triton-matmul-autotune-h100-v3.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v3.md`
- `docs/results/fp8-triton-matmul-autotune-h100-v4-splitk.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v4-splitk.md`
- `docs/results/fp8-prepack-amortization-h100.json`
- `docs/results/fp8-prepack-amortization-h100.md`
- `docs/results/fp8-layout-reuse-policy-h100.json`
- `docs/results/fp8-layout-reuse-policy-h100.md`
- `docs/results/fp8-layout-runtime-integration-h100.json`
- `docs/results/fp8-layout-runtime-integration-h100.md`
- `docs/results/fp8-layout-runtime-cache-integration-h100.json`
- `docs/results/fp8-layout-runtime-cache-integration-h100.md`
- `docs/results/fp8-layout-runtime-integration-token-loop.json`
- `docs/results/fp8-layout-runtime-integration-token-loop.md`

Executor integration note:

- Live matmul benchmark payload now includes `fp8-runtime-layout-summary.json` when FP8 fixtures are present.

Speculative decoding verify+accept baseline:

- `docs/results/spec-decode-verify-accept-baseline.json`
- `docs/results/spec-decode-verify-accept-baseline.md`

Speculative decoding verify+accept fused v1:

- `docs/results/spec-decode-verify-accept-fused-v1.json`
- `docs/results/spec-decode-verify-accept-fused-v1.md`

Speculative decoding runtime-hook integration benchmark:

- `docs/results/spec-decode-verify-accept-runtime-integration.json`
- `docs/results/spec-decode-verify-accept-runtime-integration.md`

KV cache quantize-on-write fused v1:

- `docs/results/kv-quant-write-fused-v1.json`
- `docs/results/kv-quant-write-fused-v1.md`

KV cache quantize-on-write runtime integration:

- `docs/results/kv-quant-write-runtime-integration.json`
- `docs/results/kv-quant-write-runtime-integration.md`

Cross-vendor zero-shot scaffold (MI300X label, no target measurements):

- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.json`
- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.md`

Reproducible benchmark-pack command:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/gemma4_layer_benchmark_pack.py
```

This writes canonical pack outputs:

- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json`
- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md`

Current canonical pack artifacts:

- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json`
- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md`

Notes:

- Historical artifacts in this directory are retained for auditability and timeline context.
- README and paper should reference canonical latest artifacts above unless explicitly discussing historical progression.
