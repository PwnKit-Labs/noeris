# FP8 Runtime Integration Token-Loop Benchmark

Generated: 2026-05-02T19:44:26.327021+00:00

| Mode | cache | dispatch total ms | avg ms/token | prepack ops | cache hits | cache misses | vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| auto_no_cache | false | 0.4496 | 0.0187 | 24 | 0 | 0 | 1.0000 |
| auto_with_cache | true | 0.0855 | 0.0036 | 24 | 21 | 3 | 0.1902 |
| force_kn_no_cache | false | 0.0155 | 0.0006 | 0 | 0 | 0 | 0.0345 |

`auto_with_cache` should show cache hits during repeated FP8 token-loop dispatch.
