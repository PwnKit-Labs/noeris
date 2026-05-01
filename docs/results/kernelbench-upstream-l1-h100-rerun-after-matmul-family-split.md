# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.642 | 0.308 | 8.57x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.293 | 1.740 | 0.74x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.948 | 4.425 | 0.89x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.376 | 3.119 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.524 | 4.343 | 0.58x | y | 2.590 |
| 23_Softmax.py | softmax | 8.454 | 11.158 | 0.76x | y | 8.520 |
| 26_GELU_.py | geglu | 4.657 | 4.448 | 1.05x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.084 | 13.170 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.367 | 1.398 | 5.99x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.606 | 0.190 | 8.45x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.476 | 0.373 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.422 | 19.194 | 1.27x | y | 8.450 |
