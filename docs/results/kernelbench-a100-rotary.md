# KernelBench-style Evaluation

Hardware: A100
Problems evaluated: 2

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 100.0% | 0.0% | 100.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 0.0% | 0.0% | 0.0% |
| fast_1.5 | 0.0% | 0.0% | 0.0% |
| fast_2.0 | 0.0% | 0.0% | 0.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L2_rotary_llama7b | rotary | 2 | 324.4 | 156.3 | 2.08x | 549.5 | 0.59x | `bs32_w2_s1` |
| kb_L2_rotary_gemma_26b | rotary | 2 | 425.6 | 162.3 | 2.62x | 539.8 | 0.79x | `bs32_w2_s1` |
