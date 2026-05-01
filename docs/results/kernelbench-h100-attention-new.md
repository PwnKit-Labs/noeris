# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 12

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 25.0% | 0.0% | 0.0% |
| fast_1.5 | 25.0% | 0.0% | 0.0% |
| fast_2.0 | 16.7% | 0.0% | 0.0% |
| fast_3.0 | 16.7% | 0.0% | 0.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 50.0% | 0.0% | 100.0% |
| fast_1.5 | 33.3% | 0.0% | 33.3% |
| fast_2.0 | 25.0% | 0.0% | 0.0% |
| fast_3.0 | 16.7% | 0.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L2_attn_short_64 | attention | 2 | 169.5 | 206.7 | 0.82x | 93.3 | 1.82x | `m128_n64_w8_s3` |
| kb_L2_attn_short_128 | attention | 2 | 231.8 | 314.1 | 0.74x | 159.8 | 1.45x | `m64_n64_w4_s3` |
| kb_L2_attn_med_128 | attention | 2 | 414.1 | 583.2 | 0.71x | 367.4 | 1.13x | `m64_n64_w4_s3` |
| kb_L3_attn_long_64 | attention | 3 | 352.4 | 442.6 | 0.80x | 375.9 | 0.94x | `m128_n64_w8_s3` |
| kb_L3_attn_long_128 | attention | 3 | 455.8 | 637.4 | 0.72x | 503.8 | 0.90x | `m64_n64_w4_s3` |
| kb_L3_attn_llama7b | attention | 3 | 471.7 | 655.0 | 0.72x | 533.2 | 0.88x | `m64_n64_w4_s3` |
| kb_L3_attn_llama7b_causal | attention | 3 | 293.2 | 514.7 | 0.57x | 378.3 | 0.78x | `m64_n64_w4_s2` |
| kb_L3_attn_mistral_causal | attention | 3 | 334.4 | 600.3 | 0.56x | 498.4 | 0.67x | `m64_n64_w4_s2` |
| kb_L3_attn_gemma_local | attention | 3 | 133.9 | 30.1 | 4.45x | 27.9 | 4.81x | `m64_n64_w4_s3` |
| kb_L3_attn_gemma_slide_causal | attention | 3 | 142.9 | 73.7 | 1.94x | 62.8 | 2.27x | `m64_n64_w4_s2` |
| kb_L3_attn_gemma_qknorm_local | attention | 3 | 92.9 | 23.7 | 3.92x | 27.2 | 3.42x | `m128_n64_w8_s3` |
| kb_L3_attn_gemma_qknorm_global | attention | 3 | 183.9 | 238.0 | 0.77x | 331.2 | 0.56x | `m128_n64_w8_s3` |
