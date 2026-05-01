# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.364 | 7.31x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 4.697 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.003 | 4.350 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.376 | 3.124 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.500 | 4.222 | 0.59x | y | 2.590 |
| 23_Softmax.py | softmax | 8.659 | 11.276 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.547 | 4.537 | 1.00x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.221 | 44.347 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.218 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.623 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.377 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.217 | 19.252 | 1.26x | y | 8.450 |
