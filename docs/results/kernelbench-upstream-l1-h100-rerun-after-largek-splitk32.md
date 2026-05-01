# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.644 | 0.306 | 8.63x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.293 | 1.404 | 0.92x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.975 | 1.742 | 2.28x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.412 | 3.159 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.559 | 1.729 | 1.48x | y | 2.590 |
| 23_Softmax.py | softmax | 8.451 | 6.456 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.244 | 4.415 | 0.96x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.096 | 13.178 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.428 | 1.401 | 6.02x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.607 | 0.189 | 8.48x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.476 | 0.373 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.141 | 19.294 | 1.25x | y | 8.450 |
