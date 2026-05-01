# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.311 | 8.54x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 1.419 | 0.92x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.976 | 1.710 | 2.33x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.375 | 3.121 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.536 | 1.717 | 1.48x | y | 2.590 |
| 23_Softmax.py | softmax | 8.646 | 6.604 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.279 | 4.527 | 0.94x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.205 | 13.290 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.199 | 1.395 | 5.88x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.620 | 0.193 | 8.39x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.473 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.620 | 19.146 | 1.29x | y | 8.450 |
