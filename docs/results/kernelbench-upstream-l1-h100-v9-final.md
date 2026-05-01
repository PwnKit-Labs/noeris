# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.642 | 0.363 | 7.27x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.293 | 4.626 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.935 | 4.339 | 0.91x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.413 | 3.161 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.516 | 4.228 | 0.59x | y | 2.590 |
| 23_Softmax.py | softmax | 8.682 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.615 | 4.538 | 1.02x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.202 | 44.695 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.250 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.621 | 0.193 | 8.38x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.377 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.208 | 19.185 | 1.26x | y | 8.450 |
