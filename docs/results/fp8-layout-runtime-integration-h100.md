# FP8 Layout Runtime Integration (H100)

Generated: 2026-04-24T18:21:45.870090+00:00

| Shape | reuse | kn ms | nk ms | policy kn total | policy nk total | auto layout | kernel best | policy best | auto/policy |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| fp8_mm_1024 | 1 | 0.0534 | 0.0324 | 0.0359 | 0.0456 | kn | nk | kn | 1.0000 |
| fp8_mm_2048x1024x2048 | 2 | 0.0785 | 0.0373 | 0.1008 | 0.0928 | nk | nk | nk | 1.0000 |
| fp8_mm_4096x4096x4096 | 1 | 0.8607 | 0.1883 | 0.4597 | 0.3123 | nk | nk | nk | 1.0000 |

Kernel best compares raw kernel latency only; policy best includes prepack amortization.
auto/policy near 1.0 indicates runtime auto follows the policy decision exactly.
