# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.660 | 0.360 | 7.39x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 1.770 | 0.74x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.976 | 4.359 | 0.91x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.399 | 3.140 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.543 | 4.207 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.471 | 11.157 | 0.76x | y | 8.520 |
| 26_GELU_.py | geglu | 4.585 | 4.448 | 1.03x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.106 | 13.181 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.320 | 1.402 | 5.94x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.608 | 0.189 | 8.49x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.373 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.183 | 19.164 | 1.26x | y | 8.450 |
