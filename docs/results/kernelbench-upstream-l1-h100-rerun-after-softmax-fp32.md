# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H100 80GB HBM3
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.661 | 0.311 | 8.56x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.306 | 1.600 | 0.82x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 3.949 | 1.767 | 2.23x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.410 | 3.136 | 2.04x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.525 | 1.729 | 1.46x | y | 2.590 |
| 23_Softmax.py | softmax | 8.461 | 6.456 | 1.31x | y | 8.520 |
| 26_GELU_.py | geglu | 4.243 | 4.439 | 0.96x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 14.097 | 13.182 | 1.07x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.425 | 1.399 | 6.02x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.606 | 0.190 | 8.46x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.475 | 0.373 | 1.27x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.631 | 19.269 | 1.28x | y | 8.450 |
