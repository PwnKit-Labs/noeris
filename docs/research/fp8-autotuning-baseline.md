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

## Next implementation steps

- Add FP8 shape bucket(s) for matmul in a dedicated experiment lane.
- Add a minimal FP8 matmul benchmark script generator (parallel to fp16 path).
- Add A100 vs H100 sanity comparison (expect H100-only viability for serious results).
- Only after baseline correctness: add bandit/autotune loop for FP8 configs.

## Exit criteria for baseline phase

- A reproducible JSON artifact exists in `docs/results/`.
- We have a clear yes/no answer on runtime FP8 matmul viability on H100.
- We have concrete error constraints documented before writing FP8 Triton kernels.
