# KV Quantize-on-Write Plan

Issue: `#83` — fuse KV cache quantization into write path.

## Baseline and fused benchmark commands

Fused kernel benchmark (includes separated baseline comparison):

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/kv_quant_write_fused_benchmark.py
```

Runtime integration benchmark (auto/fused/separated):

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/kv_quant_write_runtime_benchmark.py
```

## Artifacts

- `docs/results/kv-quant-write-fused-v1.json`
- `docs/results/kv-quant-write-fused-v1.md`
- `docs/results/kv-quant-write-runtime-integration.json`
- `docs/results/kv-quant-write-runtime-integration.md`

## Scope of v1 kernel

- Per-row symmetric INT8 quantization (`scale = absmax/127`)
- One-pass fused write for quantized values + scale
- Intended for KV write path integration (K and V quantized independently)

## Next steps after v1

- Add packed 4-bit variant lane (FP4/INT4-like storage) once baseline INT8 path is stable.
- Evaluate end-to-end decode throughput impact with paged-KV decode workloads.
- Add calibration option (per-channel or groupwise scales) for quality/perf tradeoffs.
