# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 6

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 16.7% | 0.0% | 33.3% |
| fast_2.0 | 0.0% | 0.0% | 0.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 100.0% | 100.0% |
| fast_1.5 | 100.0% | 100.0% | 100.0% |
| fast_2.0 | 100.0% | 100.0% | 100.0% |
| fast_3.0 | 16.7% | 33.3% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L1_layernorm_gpt2 | layernorm | 1 | 434.4 | 353.3 | 1.23x | 100.4 | 4.33x | `bs1024_w4_s1` |
| kb_L1_layernorm_bert_base | layernorm | 1 | 1145.0 | 816.0 | 1.40x | 431.3 | 2.66x | `bs1024_w4_s1` |
| kb_L1_layernorm_bert_large | layernorm | 1 | 1385.5 | 984.8 | 1.41x | 521.3 | 2.66x | `bs1024_w4_s1` |
| kb_L2_layernorm_gpt_xl | layernorm | 2 | 1642.1 | 1202.3 | 1.37x | 564.9 | 2.91x | `bs1024_w4_s1` |
| kb_L2_layernorm_neox | layernorm | 2 | 2256.8 | 1672.8 | 1.35x | 1005.6 | 2.24x | `bs2048_w8_s1` |
| kb_L2_layernorm_long_seq | layernorm | 2 | 1620.0 | 1067.2 | 1.52x | 731.0 | 2.22x | `bs512_w2_s2` |
