# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.310 | 8.58x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.303 | 1.596 | 0.82x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.948 | 1.700 | 2.32x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.375 | 3.124 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.535 | 1.718 | 1.48x | y | 2.590 |
| 23_Softmax.py | softmax | 8.643 | 6.614 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.276 | 4.497 | 0.95x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.222 | 13.301 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.424 | 1.401 | 6.01x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.620 | 0.194 | 8.37x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.339 | 19.315 | 1.26x | y | 8.450 |
