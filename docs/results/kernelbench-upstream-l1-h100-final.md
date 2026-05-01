# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.642 | 0.360 | 7.33x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.296 | 4.682 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.921 | 4.344 | 0.90x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.393 | 3.120 | 2.05x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.563 | 4.210 | 0.61x | y | 2.590 |
| 23_Softmax.py | softmax | 8.674 | 11.246 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.681 | 4.535 | 1.03x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.207 | 44.409 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.198 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.623 | 0.193 | 8.41x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.375 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.462 | 19.254 | 1.27x | y | 8.450 |
