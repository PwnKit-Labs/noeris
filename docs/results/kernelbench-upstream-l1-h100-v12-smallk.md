# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.662 | 0.361 | 7.37x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 1.785 | 0.73x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.003 | 4.362 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.374 | 3.122 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.543 | 4.213 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.669 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.343 | 4.537 | 0.96x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.217 | 14.219 | 1.00x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.222 | 8.227 | 1.00x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.621 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.184 | 19.249 | 1.26x | y | 8.450 |
