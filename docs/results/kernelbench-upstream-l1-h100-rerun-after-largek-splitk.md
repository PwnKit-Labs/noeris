# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.311 | 8.55x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 1.618 | 0.81x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.977 | 1.796 | 2.21x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.412 | 3.142 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.511 | 1.720 | 1.46x | y | 2.590 |
| 23_Softmax.py | softmax | 8.668 | 11.270 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.489 | 4.538 | 0.99x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.220 | 13.305 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.453 | 1.401 | 6.03x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.620 | 0.193 | 8.41x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.476 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.319 | 19.209 | 1.27x | y | 8.450 |
