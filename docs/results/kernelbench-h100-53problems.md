# KernelBench-style Evaluation

Hardware: H100
Problems evaluated: 53

## fast_p Scores

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 56.6% | 58.3% | 61.5% |
| fast_1.5 | 43.4% | 45.8% | 46.2% |
| fast_2.0 | 41.5% | 45.8% | 42.3% |
| fast_3.0 | 30.2% | 33.3% | 30.8% |

## Per-Problem Results

| Problem | Operator | Level | Our Metric | PyTorch | Speedup | Config |
|---------|----------|-------|-----------|---------|---------|--------|
| kb_L1_matmul_128 | matmul | 1 | 0.6 | 0.6 | 0.95x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_256 | matmul | 1 | 4.2 | 4.9 | 0.87x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_512 | matmul | 1 | 26.7 | 34.6 | 0.77x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_1024 | matmul | 1 | 142.1 | 206.9 | 0.69x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_2048 | matmul | 1 | 567.8 | 589.8 | 0.96x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_matmul_4096 | matmul | 1 | 688.3 | 727.2 | 0.95x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_matmul_tall_4k | matmul | 1 | 421.2 | 441.9 | 0.95x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_matmul_tall_8k | matmul | 1 | 494.1 | 556.3 | 0.89x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_matmul_wide | matmul | 1 | 491.4 | 548.6 | 0.90x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_matmul_deep | matmul | 1 | 221.0 | 505.1 | 0.44x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L2_gpt2_qkv | matmul | 2 | 234.1 | 293.7 | 0.80x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L2_gpt2_out | matmul | 2 | 96.6 | 132.9 | 0.73x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L2_gpt2_mlp_up | matmul | 2 | 301.8 | 336.8 | 0.90x | `bm64_bn256_bk32_gm8_w4_s4` |
| kb_L2_gpt2_mlp_down | matmul | 2 | 152.3 | 315.9 | 0.48x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L2_llama7b_qkv | matmul | 2 | 691.9 | 688.3 | 1.01x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L2_llama7b_mlp_up | matmul | 2 | 632.8 | 624.7 | 1.01x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L2_llama7b_mlp_down | matmul | 2 | 654.2 | 684.2 | 0.96x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L2_bert_qkv | matmul | 2 | 194.5 | 254.0 | 0.77x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L2_mistral_mlp | matmul | 2 | 667.2 | 685.0 | 0.97x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_rmsnorm_gpt2 | rmsnorm | 1 | 462.1 | 82.4 | 5.60x | `bs512_w2_s2` |
| kb_L1_rmsnorm_bert | rmsnorm | 1 | 1495.0 | 194.7 | 7.68x | `bs512_w2_s2` |
| kb_L1_rmsnorm_gpt_xl | rmsnorm | 1 | 1733.3 | 198.8 | 8.72x | `bs1024_w4_s1` |
| kb_L2_rmsnorm_llama7b | rmsnorm | 2 | 2340.2 | 210.7 | 11.11x | `bs2048_w8_s1` |
| kb_L2_rmsnorm_llama13b | rmsnorm | 2 | 2418.3 | 215.9 | 11.20x | `bs4096_w16_s1` |
| kb_L2_rmsnorm_llama70b | rmsnorm | 2 | 2292.8 | 210.6 | 10.89x | `bs4096_w16_s1` |
| kb_L2_rmsnorm_mixtral | rmsnorm | 2 | 2625.1 | 225.2 | 11.66x | `bs2048_w8_s1` |
| kb_L1_softmax_tiny | softmax | 1 | 164.8 | 75.7 | 2.18x | `bs1024_w4_s1` |
| kb_L1_softmax_small | softmax | 1 | 311.2 | 135.5 | 2.30x | `bs1024_w4_s1` |
| kb_L1_softmax_medium | softmax | 1 | 979.6 | 340.9 | 2.87x | `bs1024_w4_s1` |
| kb_L1_softmax_large | softmax | 1 | 2347.5 | 429.9 | 5.46x | `bs2048_w8_s1` |
| kb_L2_softmax_attn_short | softmax | 2 | 1278.9 | 455.0 | 2.81x | `bs1024_w4_s1` |
| kb_L2_softmax_attn_long | softmax | 2 | 1956.9 | 404.9 | 4.83x | `bs4096_w8_s1` |
| kb_L2_softmax_vocab_gpt2 | softmax | 2 | 975.0 | 383.1 | 2.55x | `bs4096_w16_s1` |
| kb_L2_softmax_vocab_llama | softmax | 2 | 2526.4 | 396.1 | 6.38x | `bs4096_w16_s1` |
| kb_L1_layernorm_gpt2 | layernorm | 1 | 444.9 | 355.7 | 1.25x | `bs1024_w4_s1` |
| kb_L1_layernorm_bert_base | layernorm | 1 | 1161.3 | 832.1 | 1.40x | `bs1024_w4_s1` |
| kb_L1_layernorm_bert_large | layernorm | 1 | 1407.6 | 1009.0 | 1.40x | `bs1024_w4_s1` |
| kb_L2_layernorm_gpt_xl | layernorm | 2 | 1655.8 | 1233.3 | 1.34x | `bs1024_w4_s1` |
| kb_L2_layernorm_neox | layernorm | 2 | 2263.8 | 1742.9 | 1.30x | `bs2048_w8_s1` |
| kb_L2_layernorm_long_seq | layernorm | 2 | 1666.8 | 1088.4 | 1.53x | `bs512_w2_s2` |
| kb_L1_ce_bert | cross_entropy | 1 | 1552.4 | 281.3 | 5.52x | `bs16384_w16_s1` |
| kb_L1_ce_gpt2 | cross_entropy | 1 | 1219.3 | 235.3 | 5.18x | `bs16384_w16_s1` |
| kb_L1_ce_gpt2_long | cross_entropy | 1 | 1352.3 | 241.9 | 5.59x | `bs16384_w16_s1` |
| kb_L1_ce_llama | cross_entropy | 1 | 2006.9 | 248.1 | 8.09x | `bs16384_w16_s1` |
| kb_L2_ce_mistral | cross_entropy | 2 | 2266.3 | 249.6 | 9.08x | `bs8192_w16_s1` |
| kb_L2_ce_long_llama | cross_entropy | 2 | 2407.1 | 249.5 | 9.65x | `bs32768_w16_s1` |
| kb_L2_ce_llama3_128k | cross_entropy | 2 | 564.9 | 237.5 | 2.38x | `bs4096_w8_s1` |
| kb_L2_attn_short_64 | attention | 2 | 136.6 | 200.1 | 0.68x | `m128_n32_w4_s3` |
| kb_L2_attn_short_128 | attention | 2 | 241.0 | 326.4 | 0.74x | `m64_n64_w4_s3` |
| kb_L2_attn_med_128 | attention | 2 | 434.3 | 578.6 | 0.75x | `m64_n64_w4_s3` |
| kb_L3_attn_long_64 | attention | 3 | 362.8 | 439.3 | 0.83x | `m64_n64_w4_s3` |
| kb_L3_attn_long_128 | attention | 3 | 462.9 | 619.4 | 0.75x | `m64_n64_w4_s3` |
| kb_L3_attn_llama7b | attention | 3 | 468.4 | 598.0 | 0.78x | `m64_n64_w4_s3` |
