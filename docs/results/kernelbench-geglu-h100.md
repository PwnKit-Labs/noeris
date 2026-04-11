# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 4

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 100.0% | 0.0% | 100.0% |
| fast_3.0 | 75.0% | 0.0% | 75.0% |

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
| kb_L2_geglu_gemma2b | geglu | 2 | 1999.0 | 555.7 | 3.60x | 816.0 | 2.45x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma4b | geglu | 2 | 2120.3 | 586.9 | 3.61x | 1025.9 | 2.07x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma26b | geglu | 2 | 2287.3 | 591.7 | 3.87x | 978.1 | 2.34x | `bs4096_w16_s1` |
| kb_L2_geglu_gemma31b | geglu | 2 | 1601.3 | 604.5 | 2.65x | 1051.3 | 1.52x | `bs4096_w16_s1` |
