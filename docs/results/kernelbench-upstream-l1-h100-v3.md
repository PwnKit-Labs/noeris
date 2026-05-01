# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.361 | 7.37x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 4.700 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.922 | 4.347 | 0.90x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.412 | 3.137 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.525 | 4.212 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.641 | 28.886 | 0.30x | y | 8.520 |
| 26_GELU_.py | geglu | 4.340 | 4.523 | 0.96x | — | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.200 | 55.753 | 0.26x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.466 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.623 | 0.192 | 8.43x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.369 | 19.300 | 1.26x | y | 8.450 |
