# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.361 | 7.38x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.305 | 4.707 | 0.28x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.920 | 4.342 | 0.90x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.414 | 3.140 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.542 | 4.217 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.644 | 12.379 | 0.70x | y | 8.520 |
| 26_GELU_.py | geglu | 4.679 | 4.535 | 1.03x | — | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.222 | 51.464 | 0.28x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.476 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.41x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.377 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.161 | 19.299 | 1.25x | y | 8.450 |
