# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.660 | 0.309 | 8.61x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.302 | 1.607 | 0.81x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.972 | 1.796 | 2.21x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.402 | 3.154 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.512 | 1.722 | 1.46x | y | 2.590 |
| 23_Softmax.py | softmax | 8.658 | 6.641 | 1.30x | y | 8.520 |
| 26_GELU_.py | geglu | 4.290 | 4.550 | 0.94x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.226 | 13.298 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.467 | 1.402 | 6.04x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.40x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.377 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.390 | 19.289 | 1.26x | y | 8.450 |
