# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.662 | 0.357 | 7.46x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 4.746 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.004 | 4.368 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.376 | 3.138 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.496 | 4.229 | 0.59x | y | 2.590 |
| 23_Softmax.py | softmax | 8.662 | 11.405 | 0.76x | y | 8.520 |
| 26_GELU_.py | geglu | 4.325 | 4.535 | 0.95x | — | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.224 | 44.377 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.234 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.476 | 0.376 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.300 | 19.128 | 1.27x | y | 8.450 |
