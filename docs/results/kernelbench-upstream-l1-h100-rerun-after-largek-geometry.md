# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.310 | 8.59x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 1.626 | 0.80x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.976 | 1.778 | 2.24x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.383 | 3.144 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.499 | 1.720 | 1.45x | y | 2.590 |
| 23_Softmax.py | softmax | 8.642 | 6.610 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.277 | 4.497 | 0.95x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.220 | 13.301 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.404 | 1.404 | 5.99x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.619 | 0.193 | 8.37x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.411 | 19.301 | 1.26x | y | 8.450 |
