# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 8

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 100.0% | 100.0% | 100.0% |
| fast_2.0 | 100.0% | 100.0% | 100.0% |
| fast_3.0 | 50.0% | 25.0% | 75.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 87.5% | 100.0% | 75.0% |
| fast_2.0 | 87.5% | 100.0% | 75.0% |
| fast_3.0 | 37.5% | 75.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L1_softmax_tiny | softmax | 1 | 161.8 | 75.2 | 2.15x | 25.8 | 6.28x | `bs512_w2_s2` |
| kb_L1_softmax_small | softmax | 1 | 307.0 | 131.1 | 2.34x | 72.1 | 4.26x | `bs512_w2_s2` |
| kb_L1_softmax_medium | softmax | 1 | 955.0 | 342.7 | 2.79x | 261.1 | 3.66x | `bs1024_w4_s1` |
| kb_L1_softmax_large | softmax | 1 | 2363.0 | 431.6 | 5.47x | 919.8 | 2.57x | `bs4096_w8_s1` |
| kb_L2_softmax_attn_short | softmax | 2 | 1452.3 | 474.7 | 3.06x | 535.0 | 2.71x | `bs512_w2_s2` |
| kb_L2_softmax_attn_long | softmax | 2 | 1932.9 | 402.5 | 4.80x | 691.4 | 2.80x | `bs2048_w8_s1` |
| kb_L2_softmax_vocab_gpt2 | softmax | 2 | 976.8 | 383.4 | 2.55x | 792.2 | 1.23x | `bs8192_w16_s1` |
| kb_L2_softmax_vocab_llama | softmax | 2 | 2514.8 | 396.1 | 6.35x | 975.9 | 2.58x | `bs8192_w16_s1` |
