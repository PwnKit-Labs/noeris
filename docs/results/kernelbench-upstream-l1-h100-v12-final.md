# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.663 | 1.709 | 1.56x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.304 | — | — | — | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.032 | 3.324 | 1.21x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.384 | 5.098 | 1.25x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.455 | 3.780 | 0.65x | y | 2.590 |
| 23_Softmax.py | softmax | 8.648 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.499 | 4.540 | 0.99x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.211 | 14.201 | 1.00x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.233 | 8.227 | 1.00x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.623 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.178 | 19.287 | 1.25x | y | 8.450 |
