# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 12

## fast_p vs PyTorch eager

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 16.7% | 0.0% | 0.0% |
| fast_1.5 | 8.3% | 0.0% | 0.0% |
| fast_2.0 | 0.0% | 0.0% | 0.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## fast_p vs torch.compile max-autotune

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 41.7% | 0.0% | 100.0% |
| fast_1.5 | 16.7% | 0.0% | 33.3% |
| fast_2.0 | 0.0% | 0.0% | 0.0% |
| fast_3.0 | 0.0% | 0.0% | 0.0% |

## Per-Problem Results

| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |
|---|---|---|---|---|---|---|---|---|
| kb_L2_attn_short_64 | attention | 2 | 172.2 | 206.7 | 0.83x | 105.3 | 1.64x | `m64_n64_w4_s3` |
| kb_L2_attn_short_128 | attention | 2 | 234.9 | 314.8 | 0.75x | 158.7 | 1.48x | `m64_n64_w4_s3` |
| kb_L2_attn_med_128 | attention | 2 | 420.0 | 583.8 | 0.72x | 388.9 | 1.08x | `m64_n64_w4_s3` |
| kb_L3_attn_long_64 | attention | 3 | 356.3 | 439.6 | 0.81x | 376.0 | 0.95x | `m128_n64_w8_s3` |
| kb_L3_attn_long_128 | attention | 3 | 458.6 | 622.0 | 0.74x | 497.3 | 0.92x | `m64_n64_w4_s3` |
| kb_L3_attn_llama7b | attention | 3 | 454.2 | 628.5 | 0.72x | 526.0 | 0.86x | `m64_n64_w4_s3` |
| kb_L3_attn_llama7b_causal | attention | 3 | 227.5 | 489.9 | 0.46x | 374.8 | 0.61x | `m64_n64_w4_s3` |
| kb_L3_attn_mistral_causal | attention | 3 | 223.0 | 559.7 | 0.40x | 496.0 | 0.45x | `m64_n64_w4_s3` |
| kb_L3_attn_gemma_local | attention | 3 | 49.4 | 29.8 | 1.66x | 28.5 | 1.73x | `m64_n64_w4_s3` |
| kb_L3_attn_gemma_slide_causal | attention | 3 | 90.1 | 74.8 | 1.20x | 62.4 | 1.44x | `m64_n64_w4_s3` |
| kb_L3_attn_gemma_qknorm_local | attention | 3 | 21.4 | 23.8 | 0.90x | 27.3 | 0.78x | `m64_n64_w4_s3` |
| kb_L3_attn_gemma_qknorm_global | attention | 3 | 89.0 | 237.3 | 0.38x | 328.2 | 0.27x | `m64_n64_w4_s3` |
