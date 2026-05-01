# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.659 | 0.362 | 7.35x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 1.775 | 0.73x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.980 | 4.342 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.406 | 3.138 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.520 | 4.206 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.646 | 11.285 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.632 | 4.537 | 1.02x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.229 | 44.980 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.460 | 8.226 | 1.03x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.41x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.445 | 19.359 | 1.26x | y | 8.450 |
