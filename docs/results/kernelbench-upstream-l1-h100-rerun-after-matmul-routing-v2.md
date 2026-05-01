# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.311 | 8.55x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 1.754 | 0.74x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.977 | 4.407 | 0.90x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.412 | 3.590 | 1.79x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.511 | 4.354 | 0.58x | y | 2.590 |
| 23_Softmax.py | softmax | 8.649 | 11.271 | 0.77x | y | 8.520 |
| 26_GELU_.py | geglu | 4.679 | 4.537 | 1.03x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.226 | 13.313 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.460 | 1.401 | 6.04x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.476 | 0.376 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.633 | 19.308 | 1.28x | y | 8.450 |
