# FP8 Triton Matmul Autotune (H100, v4 split-K)

Generated: 2026-04-24T12:56:46.784095+00:00

| Shape | split-K | best FP8 TFLOPS | FP16 TFLOPS | FP8/FP16 | FP8 ms | max err | config |
|---|---:|---:|---:|---:|---:|---:|---|
| fp8_mm_1024 | 2 | 45.5903 | 113.5514 | 0.401x | 0.0471 | 0.125000 | nk_bm128_bn128_bk128_w8_s4_sk2 |
| fp8_mm_2048x1024x2048 | 1 | 160.6436 | 341.5209 | 0.470x | 0.0535 | 0.125000 | nk_bm128_bn64_bk64_w8_s3_sk1 |
| fp8_mm_2048x2048x8192 | 1 | 486.0760 | 684.3479 | 0.710x | 0.1414 | 0.250000 | nk_bm128_bn128_bk128_w8_s4_sk1 |
| fp8_mm_4096x4096x4096 | 1 | 410.7265 | 726.3601 | 0.565x | 0.3346 | 0.250000 | nk_bm128_bn64_bk64_w8_s3_sk1 |

v4 adds split-K accumulation with fp32 atomic reduction and fp16 cast-out.
