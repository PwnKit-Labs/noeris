# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.660 | 0.311 | 8.55x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.304 | 1.617 | 0.81x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.974 | 1.726 | 2.30x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.366 | 3.152 | 2.02x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.525 | 1.720 | 1.47x | y | 2.590 |
| 23_Softmax.py | softmax | 8.645 | 6.606 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.277 | 4.524 | 0.94x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.210 | 13.286 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.406 | 1.399 | 6.01x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.620 | 0.193 | 8.41x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.377 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.541 | 19.199 | 1.28x | y | 8.450 |
