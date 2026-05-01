# FP8 Triton Matmul Autotune (H100, v2)

Generated: 2026-04-24T09:20:47.517827+00:00

| Shape | best FP8 TFLOPS | FP16 TFLOPS | FP8/FP16 | FP8 ms | max err | config |
|---|---:|---:|---:|---:|---:|---|
| fp8_mm_1024 | 66.5103 | 109.2978 | 0.609x | 0.0323 | 0.125000 | bm128_bn64_bk64_w8_s3_g8 |
| fp8_mm_2048x1024x2048 | 171.1961 | 334.7076 | 0.511x | 0.0502 | 0.125000 | bm256_bn64_bk128_w8_s4_g8 |
| fp8_mm_4096x4096x4096 | 298.8011 | 730.8095 | 0.409x | 0.4600 | 0.250000 | bm256_bn64_bk128_w8_s4_g8 |

v2 adds grouped launch ordering and an expanded BLOCK_K/BLOCK_N search lane.
