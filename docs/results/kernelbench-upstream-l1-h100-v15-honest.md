# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.642 | 0.361 | 7.33x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.295 | — | — | — | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.005 | 4.360 | 0.92x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.374 | 3.138 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.516 | 4.224 | 0.60x | y | 2.590 |
| 23_Softmax.py | softmax | 8.458 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.581 | 4.449 | 1.03x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.095 | 44.953 | 0.31x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.449 | 1.399 | 6.04x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.608 | 0.190 | 8.47x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.373 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.479 | 19.268 | 1.27x | y | 8.450 |
