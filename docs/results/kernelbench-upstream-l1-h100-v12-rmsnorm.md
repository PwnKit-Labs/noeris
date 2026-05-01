# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.642 | 0.358 | 7.37x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.297 | 1.774 | 0.73x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.004 | 4.369 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.378 | 3.140 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.534 | 4.213 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.649 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.681 | 4.537 | 1.03x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.201 | 14.202 | 1.00x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.205 | 8.196 | 1.00x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.621 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.252 | 19.249 | 1.26x | y | 8.450 |
