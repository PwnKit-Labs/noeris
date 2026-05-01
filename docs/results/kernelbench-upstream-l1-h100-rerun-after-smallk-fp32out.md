# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.660 | 0.309 | 8.61x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.304 | 1.769 | 0.74x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.975 | 1.798 | 2.21x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.404 | 3.139 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.544 | 1.720 | 1.48x | y | 2.590 |
| 23_Softmax.py | softmax | 8.666 | 11.286 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.581 | 4.535 | 1.01x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.219 | 13.302 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.416 | 1.399 | 6.02x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.473 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.304 | 19.290 | 1.26x | y | 8.450 |
