# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.364 | 7.31x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.304 | 3.866 | 0.34x | n | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.948 | 4.724 | 0.84x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.373 | 3.138 | 2.03x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.519 | 4.637 | 0.54x | y | 2.590 |
| 23_Softmax.py | softmax | 8.671 | — | — | — | 8.520 |
| 26_GELU_.py | geglu | 4.278 | 4.536 | 0.94x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.220 | 44.393 | 0.32x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.250 | — | — | — | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.622 | 0.193 | 8.39x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.474 | 0.376 | 1.26x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.475 | 19.324 | 1.27x | y | 8.450 |
