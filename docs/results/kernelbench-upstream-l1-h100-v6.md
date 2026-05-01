# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.660 | 0.360 | 7.40x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 4.707 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.894 | 4.352 | 0.90x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.412 | 3.124 | 2.05x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.527 | 4.221 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.634 | 11.269 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.613 | 4.537 | 1.02x | — | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.217 | 44.446 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.227 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.621 | 0.193 | 8.39x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.476 | 0.376 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.115 | 19.135 | 1.26x | y | 8.450 |
