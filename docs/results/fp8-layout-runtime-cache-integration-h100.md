# FP8 Runtime Cache Integration (H100)

Generated: 2026-04-27T10:34:17.720041+00:00

| Scenario | mode | avg ms/op | vs best | cache hits | cache misses |
|---|---|---:|---:|---:|---:|
| s1024_reuse1_unique | force_kn | 0.0293 | 1.0034 | 0 | 0 |
| s1024_reuse1_unique | force_nk_no_cache | 0.0339 | 1.1610 | 0 | 0 |
| s1024_reuse1_unique | force_nk_cache | 0.0670 | 2.2945 | 8 | 64 |
| s1024_reuse1_unique | auto_policy_cache | 0.0292 | 1.0000 | 0 | 0 |
| s1024_reuse8_hotset | force_kn | 0.0280 | 1.1429 | 0 | 0 |
| s1024_reuse8_hotset | force_nk_no_cache | 0.0329 | 1.3429 | 0 | 0 |
| s1024_reuse8_hotset | force_nk_cache | 0.0245 | 1.0000 | 64 | 8 |
| s1024_reuse8_hotset | auto_policy_cache | 0.0247 | 1.0082 | 64 | 8 |
| s2048_reuse2 | force_kn | 0.0552 | 1.9301 | 0 | 0 |
| s2048_reuse2 | force_nk_no_cache | 0.0326 | 1.1399 | 0 | 0 |
| s2048_reuse2 | force_nk_cache | 0.0367 | 1.2832 | 40 | 32 |
| s2048_reuse2 | auto_policy_cache | 0.0286 | 1.0000 | 40 | 32 |

`auto_policy_cache` combines policy layout choice with prepack cache reuse.
