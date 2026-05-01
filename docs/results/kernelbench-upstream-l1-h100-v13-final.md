# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.663 | 0.362 | 7.36x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.305 | — | — | — | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.034 | 4.363 | 0.93x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.376 | 3.137 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.545 | 4.218 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.653 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.562 | 4.538 | 1.00x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.220 | 14.219 | 1.00x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.226 | 8.226 | 1.00x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.207 | 19.131 | 1.26x | y | 8.450 |
