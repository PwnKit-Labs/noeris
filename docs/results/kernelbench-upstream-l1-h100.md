# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.360 | 7.38x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.307 | 4.703 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.921 | 4.369 | 0.90x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.413 | 3.122 | 2.05x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.526 | 4.217 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.642 | 28.878 | 0.30x | y | 8.520 |
| 26_GELU_.py | geglu | 4.279 | 44.021 | 0.10x | — | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.218 | 55.702 | 0.26x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.391 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.621 | 0.598 | 2.71x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 25.058 | 19.317 | 1.30x | y | 8.450 |
