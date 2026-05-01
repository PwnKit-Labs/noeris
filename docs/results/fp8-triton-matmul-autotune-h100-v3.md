# FP8 Triton Matmul Autotune (H100, v3 layout variants)

Generated: 2026-04-24T12:18:55.498850+00:00

| Shape | layout | best FP8 TFLOPS | FP16 TFLOPS | FP8/FP16 | FP8 ms | max err | config |
|---|---|---:|---:|---:|---:|---:|---|
| fp8_mm_1024 | nk | 83.5727 | 103.0858 | 0.811x | 0.0257 | 0.125000 | nk_bm128_bn64_bk64_w8_s3_g8 |
| fp8_mm_2048x1024x2048 | nk | 258.3594 | 325.3763 | 0.794x | 0.0332 | 0.125000 | nk_bm128_bn64_bk64_w8_s3_g8 |
| fp8_mm_4096x4096x4096 | nk | 736.5747 | 718.5824 | 1.025x | 0.1866 | 0.250000 | nk_bm128_bn128_bk128_w8_s4_g8 |

v3 compares standard KxN B layout (`kn`) against prepacked NxK B layout (`nk`).
Transpose/prepack time is excluded by design (inference-style amortized static weights).
