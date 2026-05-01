# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.660 | 0.360 | 7.40x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 1.778 | 0.73x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.003 | 4.361 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.379 | 3.135 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.498 | 4.859 | 0.51x | y | 2.590 |
| 23_Softmax.py | softmax | 8.478 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.245 | 4.449 | 0.95x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.094 | 44.964 | 0.31x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.408 | 8.231 | 1.02x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.605 | 0.190 | 8.47x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.372 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.156 | 19.158 | 1.26x | y | 8.450 |
