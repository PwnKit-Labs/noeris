# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 2

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 100.0% | 0.0% | 100.0% |
| fast_1.5 | 100.0% | 0.0% | 100.0% |
| fast_2.0 | 50.0% | 0.0% | 50.0% |
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
| kb_L2_rotary_llama7b | rotary | 2 | 555.2 | 325.3 | 1.71x | 1021.6 | 0.54x | `bs32_w2_s1` |
| kb_L2_rotary_gemma_26b | rotary | 2 | 820.0 | 327.4 | 2.50x | 1068.2 | 0.77x | `bs128_w2_s1` |
