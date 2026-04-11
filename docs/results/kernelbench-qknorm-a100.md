# KernelBench-style Evaluation

Hardware: A100
Problems evaluated: 12

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 25.0% | 0.0% | 33.3% |
| fast_1.5 | 8.3% | 0.0% | 0.0% |
| fast_2.0 | 8.3% | 0.0% | 0.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 33.3% | 0.0% | 66.7% |
| fast_1.5 | 16.7% | 0.0% | 33.3% |
| fast_2.0 | 8.3% | 0.0% | 0.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L2_attn_short_64 | attention | 2 | 83.7 | 73.7 | 1.14x | 46.4 | 1.80x | `m64_n128_w4_s3` |
| kb_L2_attn_short_128 | attention | 2 | 76.5 | 95.4 | 0.80x | 61.4 | 1.25x | `m64_n128_w4_s3` |
| kb_L2_attn_med_128 | attention | 2 | 121.1 | 150.0 | 0.81x | 148.3 | 0.82x | `m128_n128_w8_s2` |
| kb_L3_attn_long_64 | attention | 3 | 147.1 | 169.0 | 0.87x | 150.7 | 0.98x | `m64_n128_w4_s3` |
| kb_L3_attn_long_128 | attention | 3 | 159.4 | 197.2 | 0.81x | 172.7 | 0.92x | `m128_n64_w8_s3` |
| kb_L3_attn_llama7b | attention | 3 | 160.8 | 201.2 | 0.80x | 177.0 | 0.91x | `m128_n64_w8_s3` |
| kb_L3_attn_llama7b_causal | attention | 3 | 76.4 | 161.0 | 0.47x | 132.4 | 0.58x | `m128_n128_w8_s2` |
| kb_L3_attn_mistral_causal | attention | 3 | 81.5 | 181.6 | 0.45x | 164.6 | 0.49x | `m128_n128_w8_s2` |
| kb_L3_attn_gemma_local | attention | 3 | 9.7 | 9.2 | 1.06x | 7.0 | 1.39x | `m64_n64_w4_s2` |
| kb_L3_attn_gemma_slide_causal | attention | 3 | 31.5 | 14.4 | 2.19x | 13.6 | 2.32x | `m128_n32_w4_s3` |
| kb_L3_attn_gemma_qknorm_local | attention | 3 | 3.6 | 7.3 | 0.49x | 6.8 | 0.53x | `m64_n64_w4_s2` |
| kb_L3_attn_gemma_qknorm_global | attention | 3 | 18.6 | 78.0 | 0.24x | 96.0 | 0.19x | `m64_n64_w4_s2` |
