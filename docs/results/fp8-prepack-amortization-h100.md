# FP8 Prepack Amortization (H100)

Generated: 2026-04-24T12:57:22.991710+00:00

| Shape | best kn ms | best nk ms | delta/run ms | prepack ms | break-even runs |
|---|---:|---:|---:|---:|---:|
| fp8_mm_1024 | 0.0359 | 0.0257 | 0.0102 | 0.0199 | 2 |
| fp8_mm_2048x1024x2048 | 0.0504 | 0.0332 | 0.0172 | 0.0264 | 2 |
| fp8_mm_4096x4096x4096 | 0.4597 | 0.1866 | 0.2731 | 0.1257 | 1 |

Break-even runs means how many repeated matmuls with the same B are needed
for prepacked `nk` to recover one-time transpose+pack cost.
