# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 4

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 100.0% | 0.0% | 100.0% |
| fast_3.0 | 50.0% | 0.0% | 50.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 50.0% | 0.0% | 50.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L2_geglu_gemma4_e2b | geglu | 2 | 2092.5 | 557.2 | 3.76x | 840.2 | 2.49x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma4_e4b | geglu | 2 | 1734.5 | 576.1 | 3.01x | 1042.3 | 1.66x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma4_26b_a4b_expert | geglu | 2 | 1558.1 | 571.3 | 2.73x | 714.9 | 2.18x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma4_31b | geglu | 2 | 1527.5 | 602.2 | 2.54x | 1016.0 | 1.50x | `bs4096_w16_s1` |
