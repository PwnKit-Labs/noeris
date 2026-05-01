# FP8 Triton Matmul Autotune (H100)

Generated: 2026-04-21T16:04:12.394797+00:00

| Shape | best FP8 TFLOPS | FP16 TFLOPS | FP8/FP16 | FP8 ms | max err | config |
|---|---:|---:|---:|---:|---:|---|
| fp8_mm_1024 | 56.4414 | 96.4208 | 0.585x | 0.0380 | 0.125000 | bm128_bn64_bk64_w8_s3 |
| fp8_mm_2048x1024x2048 | 134.8922 | 321.8651 | 0.419x | 0.0637 | 0.125000 | bm128_bn64_bk64_w8_s3 |

This is a first FP8 tuning lane scaffold (curated config sweep, not exhaustive search).
