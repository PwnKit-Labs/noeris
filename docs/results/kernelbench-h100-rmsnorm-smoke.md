# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 8

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 100.0% | 100.0% | 100.0% |
| fast_2.0 | 100.0% | 100.0% | 100.0% |
| fast_3.0 | 100.0% | 100.0% | 100.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 100.0% | 100.0% | 100.0% |
| fast_2.0 | 100.0% | 100.0% | 100.0% |
| fast_3.0 | 12.5% | 33.3% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L1_rmsnorm_gpt2 | rmsnorm | 1 | 443.0 | 83.7 | 5.29x | 122.0 | 3.63x | `bs1024_w4_s1` |
| kb_L1_rmsnorm_bert | rmsnorm | 1 | 1479.1 | 194.5 | 7.60x | 540.3 | 2.74x | `bs256_w1_s1` |
| kb_L1_rmsnorm_gpt_xl | rmsnorm | 1 | 1699.8 | 197.6 | 8.60x | 755.5 | 2.25x | `bs2048_w4_s2` |
| kb_L2_rmsnorm_llama7b | rmsnorm | 2 | 2325.3 | 210.0 | 11.07x | 1127.9 | 2.06x | `bs2048_w8_s1` |
| kb_L2_rmsnorm_llama13b | rmsnorm | 2 | 2420.8 | 214.8 | 11.27x | 1145.9 | 2.11x | `bs4096_w16_s1` |
| kb_L2_rmsnorm_llama70b | rmsnorm | 2 | 2298.8 | 209.5 | 10.97x | 1087.7 | 2.11x | `bs4096_w16_s1` |
| kb_L2_rmsnorm_mixtral | rmsnorm | 2 | 2624.9 | 224.3 | 11.70x | 1167.4 | 2.25x | `bs2048_w8_s1` |
| kb_L2_rmsnorm_gemma_26b | rmsnorm | 2 | 2095.9 | 202.8 | 10.34x | 1020.5 | 2.05x | `bs4096_w8_s2` |
