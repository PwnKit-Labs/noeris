# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.309 | 8.62x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 1.503 | 0.87x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.028 | 1.776 | 2.27x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.411 | 3.144 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.514 | 1.645 | 1.53x | y | 2.590 |
| 23_Softmax.py | softmax | 8.659 | 6.607 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.277 | 4.511 | 0.95x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.223 | 13.300 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.451 | 1.401 | 6.03x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.620 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.401 | 19.348 | 1.26x | y | 8.450 |
