# KernelBench-style Evaluation

Hardware: A100
Problems evaluated: 4

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 100.0% | 0.0% | 100.0% |
| fast_3.0 | 100.0% | 0.0% | 100.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 75.0% | 0.0% | 75.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L2_geglu_gemma2b | geglu | 2 | 1167.6 | 309.8 | 3.77x | 475.9 | 2.45x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma4b | geglu | 2 | 1279.5 | 337.2 | 3.79x | 591.5 | 2.16x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma26b | geglu | 2 | 1351.1 | 339.7 | 3.98x | 575.3 | 2.35x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma31b | geglu | 2 | 1060.0 | 344.4 | 3.08x | 606.4 | 1.75x | `bs4096_w16_s1` |
