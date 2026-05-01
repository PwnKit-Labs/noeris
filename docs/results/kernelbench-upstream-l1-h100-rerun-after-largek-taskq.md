# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.662 | 0.308 | 8.65x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 1.887 | 0.69x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.003 | 1.796 | 2.23x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.377 | 3.158 | 2.02x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.516 | 1.720 | 1.46x | y | 2.590 |
| 23_Softmax.py | softmax | 8.651 | 6.605 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.277 | 4.498 | 0.95x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.228 | 13.307 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.477 | 1.401 | 6.05x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.619 | 0.193 | 8.38x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.171 | 19.354 | 1.25x | y | 8.450 |
