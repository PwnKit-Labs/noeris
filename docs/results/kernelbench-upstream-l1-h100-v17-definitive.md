# KernelBench Upstream L1 — Honest Apples-to-Apples

Hardware: NVIDIA H200
Timer: cuda_event (3 warmup + 10 trials, L2 flush, median ms)
Correctness: torch.allclose(rtol=1e-4, atol=1e-4) on fp32
Problems: 12

| Problem | Operator | Upstream (ms) | Noeris (ms) | Speedup | Correct | External H100 (ms) |
|---|---|---|---|---|---|---|
| 1_Square_matrix_multiplication_.py | matmul | 2.692 | 0.326 | 8.27x | y | 2.660 |
| 6_Matmul_with_large_K_dimension_.py | matmul | 1.310 | 1.799 | 0.73x | y | 1.300 |
| 7_Matmul_with_small_K_dimension_.py | matmul | 4.022 | 4.093 | 0.98x | y | 4.060 |
| 8_Matmul_with_irregular_shapes_.py | matmul | 6.464 | 3.159 | 2.05x | y | 6.380 |
| 9_Tall_skinny_matrix_multiplication_.py | matmul | 2.556 | 3.981 | 0.64x | y | 2.590 |
| 23_Softmax.py | softmax | 6.404 | 9.053 | 0.71x | y | 8.520 |
| 26_GELU_.py | geglu | 3.039 | 3.277 | 0.93x | y | 4.300 |
| 36_RMSNorm_.py | rmsnorm | 11.395 | 51.644 | 0.22x | y | 14.200 |
| 40_LayerNorm.py | layernorm | 8.270 | 1.358 | 6.09x | y | 8.120 |
| 88_MinGPTNewGelu.py | geglu | 1.152 | 0.143 | 8.05x | y | 1.650 |
| 95_CrossEntropyLoss.py | cross_entropy | 0.416 | 0.284 | 1.47x | y | 0.477 |
| 97_ScaledDotProductAttention.py | attention | 24.698 | 18.485 | 1.34x | y | 8.450 |
