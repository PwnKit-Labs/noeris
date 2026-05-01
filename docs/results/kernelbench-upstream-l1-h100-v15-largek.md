# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.361 | 7.37x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.307 | 1.771 | 0.74x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.978 | 4.372 | 0.91x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.376 | 3.143 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.546 | 4.229 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.684 | 11.264 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.609 | 4.539 | 1.02x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.215 | 55.237 | 0.26x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.327 | 1.402 | 5.94x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.621 | 0.193 | 8.41x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.375 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.663 | 19.221 | 1.28x | y | 8.450 |
