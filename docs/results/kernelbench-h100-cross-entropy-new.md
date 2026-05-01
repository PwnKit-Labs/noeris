# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 8

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 100.0% | 100.0% | 100.0% |
| fast_2.0 | 87.5% | 100.0% | 75.0% |
| fast_3.0 | 75.0% | 100.0% | 50.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 75.0% | 100.0% | 50.0% |
| fast_1.5 | 75.0% | 100.0% | 50.0% |
| fast_2.0 | 75.0% | 100.0% | 50.0% |
| fast_3.0 | 12.5% | 25.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L1_ce_bert | cross_entropy | 1 | 1912.1 | 355.2 | 5.38x | 799.8 | 2.39x | `bs4096_w4_s1` |
| kb_L1_ce_gpt2 | cross_entropy | 1 | 1346.2 | 292.6 | 4.60x | 620.2 | 2.17x | `bs4096_w16_s1` |
| kb_L1_ce_gpt2_long | cross_entropy | 1 | 1496.0 | 302.5 | 4.95x | 743.3 | 2.01x | `bs32768_w16_s1` |
| kb_L1_ce_llama | cross_entropy | 1 | 2172.7 | 323.1 | 6.73x | 695.5 | 3.12x | `bs16384_w16_s1` |
| kb_L2_ce_mistral | cross_entropy | 2 | 2449.0 | 332.6 | 7.36x | 842.9 | 2.91x | `bs2048_w4_s2` |
| kb_L2_ce_long_llama | cross_entropy | 2 | 2615.8 | 337.2 | 7.76x | 877.0 | 2.98x | `bs4096_w4_s1` |
| kb_L2_ce_llama3_128k | cross_entropy | 2 | 708.5 | 300.6 | 2.36x | 910.0 | 0.78x | `bs4096_w8_s1` |
| kb_L2_ce_gemma_256k | cross_entropy | 2 | 563.4 | 303.2 | 1.86x | 955.1 | 0.59x | `bs32768_w16_s1` |
