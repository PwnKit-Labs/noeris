# KernelBench-style Evaluation

Hardware: A100
Problems evaluated: 53

## fast_p Scores

| Threshold | Overall | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| fast_1.0 | 56.6% | 58.3% | 61.5% |
| fast_1.5 | 41.5% | 45.8% | 42.3% |
| fast_2.0 | 37.7% | 37.5% | 42.3% |
| fast_3.0 | 32.1% | 33.3% | 34.6% |

## Per-Problem Results

| Problem | Operator | Level | Our Metric | PyTorch | Speedup | Config |
|---------|----------|-------|-----------|---------|---------|--------|
| kb_L1_matmul_128 | matmul | 1 | 0.4 | 0.4 | 0.98x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_256 | matmul | 1 | 3.0 | 3.4 | 0.86x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_512 | matmul | 1 | 18.4 | 21.5 | 0.86x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_1024 | matmul | 1 | 77.1 | 95.2 | 0.81x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L1_matmul_2048 | matmul | 1 | 169.6 | 227.4 | 0.75x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L1_matmul_4096 | matmul | 1 | 227.7 | 257.7 | 0.88x | `bm64_bn256_bk32_gm8_w4_s4` |
| kb_L1_matmul_tall_4k | matmul | 1 | 0.0 | 144.5 | FAIL | `` |
| kb_L1_matmul_tall_8k | matmul | 1 | 187.1 | 215.8 | 0.87x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L1_matmul_wide | matmul | 1 | 187.9 | 213.2 | 0.88x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L1_matmul_deep | matmul | 1 | 113.1 | 171.8 | 0.66x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L2_gpt2_qkv | matmul | 2 | 112.0 | 119.2 | 0.94x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L2_gpt2_out | matmul | 2 | 61.4 | 68.5 | 0.90x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L2_gpt2_mlp_up | matmul | 2 | 144.8 | 153.2 | 0.94x | `bm64_bn256_bk32_gm8_w4_s4` |
| kb_L2_gpt2_mlp_down | matmul | 2 | 102.4 | 108.5 | 0.94x | `bm128_bn64_bk32_gm8_w4_s4` |
| kb_L2_llama7b_qkv | matmul | 2 | 229.7 | 253.6 | 0.91x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L2_llama7b_mlp_up | matmul | 2 | 227.6 | 258.4 | 0.88x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L2_llama7b_mlp_down | matmul | 2 | 0.0 | 239.2 | FAIL | `` |
| kb_L2_bert_qkv | matmul | 2 | 111.5 | 98.9 | 1.13x | `bm128_bn128_bk32_gm8_w4_s4` |
| kb_L2_mistral_mlp | matmul | 2 | 227.1 | 241.4 | 0.94x | `bm128_bn256_bk64_gm8_w8_s3` |
| kb_L1_rmsnorm_gpt2 | rmsnorm | 1 | 290.4 | 63.1 | 4.60x | `bs512_w2_s2` |
| kb_L1_rmsnorm_bert | rmsnorm | 1 | 852.2 | 122.8 | 6.94x | `bs512_w2_s2` |
| kb_L1_rmsnorm_gpt_xl | rmsnorm | 1 | 958.7 | 113.8 | 8.43x | `bs1024_w4_s1` |
| kb_L2_rmsnorm_llama7b | rmsnorm | 2 | 1168.1 | 115.6 | 10.11x | `bs2048_w8_s1` |
| kb_L2_rmsnorm_llama13b | rmsnorm | 2 | 1182.2 | 117.8 | 10.03x | `bs4096_w16_s1` |
| kb_L2_rmsnorm_llama70b | rmsnorm | 2 | 1148.6 | 115.2 | 9.97x | `bs2048_w8_s1` |
| kb_L2_rmsnorm_mixtral | rmsnorm | 2 | 1253.8 | 120.2 | 10.43x | `bs2048_w8_s1` |
| kb_L1_softmax_tiny | softmax | 1 | 103.6 | 58.8 | 1.76x | `bs2048_w8_s1` |
| kb_L1_softmax_small | softmax | 1 | 188.4 | 105.2 | 1.79x | `bs2048_w8_s1` |
| kb_L1_softmax_medium | softmax | 1 | 556.6 | 231.4 | 2.41x | `bs2048_w8_s1` |
| kb_L1_softmax_large | softmax | 1 | 1183.9 | 237.5 | 4.98x | `bs1024_w4_s1` |
| kb_L2_softmax_attn_short | softmax | 2 | 788.1 | 302.4 | 2.61x | `bs1024_w4_s1` |
| kb_L2_softmax_attn_long | softmax | 2 | 1100.5 | 226.2 | 4.86x | `bs2048_w8_s1` |
| kb_L2_softmax_vocab_gpt2 | softmax | 2 | 689.2 | 185.4 | 3.72x | `bs4096_w16_s1` |
| kb_L2_softmax_vocab_llama | softmax | 2 | 1294.9 | 192.3 | 6.73x | `bs4096_w16_s1` |
| kb_L1_layernorm_gpt2 | layernorm | 1 | 298.3 | 228.8 | 1.30x | `bs512_w2_s2` |
| kb_L1_layernorm_bert_base | layernorm | 1 | 671.5 | 510.0 | 1.32x | `bs512_w2_s2` |
| kb_L1_layernorm_bert_large | layernorm | 1 | 761.2 | 606.9 | 1.25x | `bs2048_w8_s1` |
| kb_L2_layernorm_gpt_xl | layernorm | 2 | 917.2 | 693.6 | 1.32x | `bs512_w2_s2` |
| kb_L2_layernorm_neox | layernorm | 2 | 1149.2 | 942.6 | 1.22x | `bs2048_w8_s1` |
| kb_L2_layernorm_long_seq | layernorm | 2 | 896.0 | 666.4 | 1.34x | `bs1024_w4_s1` |
| kb_L1_ce_bert | cross_entropy | 1 | 837.5 | 131.9 | 6.35x | `bs16384_w16_s1` |
| kb_L1_ce_gpt2 | cross_entropy | 1 | 674.2 | 117.3 | 5.75x | `bs32768_w16_s1` |
| kb_L1_ce_gpt2_long | cross_entropy | 1 | 764.0 | 118.6 | 6.44x | `bs16384_w16_s1` |
| kb_L1_ce_llama | cross_entropy | 1 | 1086.1 | 123.0 | 8.83x | `bs32768_w16_s1` |
| kb_L2_ce_mistral | cross_entropy | 2 | 1244.6 | 122.3 | 10.17x | `bs16384_w16_s1` |
| kb_L2_ce_long_llama | cross_entropy | 2 | 1336.6 | 122.1 | 10.95x | `bs8192_w16_s1` |
| kb_L2_ce_llama3_128k | cross_entropy | 2 | 269.3 | 118.3 | 2.28x | `bs4096_w8_s1` |
| kb_L2_attn_short_64 | attention | 2 | 94.0 | 89.2 | 1.05x | `m64_n128_w4_s3` |
| kb_L2_attn_short_128 | attention | 2 | 95.5 | 116.4 | 0.82x | `m128_n64_w8_s3` |
| kb_L2_attn_med_128 | attention | 2 | 152.7 | 191.0 | 0.80x | `m128_n64_w8_s3` |
| kb_L3_attn_long_64 | attention | 3 | 148.6 | 169.7 | 0.88x | `m64_n128_w4_s3` |
| kb_L3_attn_long_128 | attention | 3 | 159.9 | 196.7 | 0.81x | `m128_n64_w8_s3` |
| kb_L3_attn_llama7b | attention | 3 | 161.7 | 199.0 | 0.81x | `m128_n64_w8_s3` |
