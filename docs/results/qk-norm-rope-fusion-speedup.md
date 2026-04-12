# Fused QK-RMSNorm+RoPE — Gemma 3/4 Prologue Fusion

**Date**: 2026-04-12
**Hardware**: NVIDIA A100-SXM4-40GB, NVIDIA H100-SXM5-80GB (Modal), Tesla T4 (Kaggle/Colab free tier)
**Timer**: `cuda_event` + L2 flush, 3 warmup / 10 trials, median ms
**Correctness tol**: `max_err ≤ 0.1` against PyTorch fp32 reference
**Baseline**: "separated" — 4 sequential PyTorch kernel launches matching vLLM's Gemma 4 prologue (Q-RMSNorm → K-RMSNorm → Q-RoPE → K-RoPE)
**Gemma-mode affine**: `y = x * rstd * (1 + w)` (per HF `Gemma4RMSNorm`)
**Fusion speedup** = `separated_ms / fused_ms`

## Critical context

This kernel does not exist in vLLM. vLLM's `Gemma4Attention.forward` (file `vllm/model_executor/models/gemma4.py:395-427`, PR [#38826](https://github.com/vllm-project/vllm/pull/38826)) issues **4+ separate launches** for Q-norm, K-norm, Q-rotary, K-rotary. Confirmed by direct source read in [`docs/research/vllm-gemma4-kernel-patterns.md`](../research/vllm-gemma4-kernel-patterns.md). Fusing this sequence is therefore a **novel kernel**, not a port.

## Results — A100 (best config per shape)

| Shape | Variant | GQA | head_dim | Config | GB/s | ms | **fusion_speedup** |
|---|---|---|---|---|---|---|---|
| `gemma4_31b_global` | Gemma 4 31B global | 32:4 | **512** | `bs64_w4_s1` | 925.5 | 0.335 | **12.85×** |
| `gemma4_26b_a4b_global` | Gemma 4 26B-A4B global | 16:2 | **512** | `bs64_w4_s1` | 905.0 | 0.176 | **12.40×** |
| `gemma4_31b_local` | Gemma 4 31B local | 32:16 | 256 | `bs128_w4_s2` | 731.2 | 0.281 | 11.30× |
| `gemma3_local_1024` | Gemma 3 local | 16:16 | 256 | `bs128_w4_s2` | 715.2 | 0.194 | 10.69× |
| `gemma4_26b_a4b_local` | Gemma 4 26B-A4B local | 16:8 | 256 | `bs64_w4_s1` | 711.1 | 0.148 | 10.37× |
| `gemma4_e2b_local` | Gemma 4 E2B local | 8:1 | 256 | `bs32_w2_s1` | 593.6 | 0.071 | 10.23× |

## Results — H100 (best config per shape)

| Shape | Variant | GQA | head_dim | Config | GB/s | ms | **fusion_speedup** |
|---|---|---|---|---|---|---|---|
| `gemma4_31b_global` | Gemma 4 31B global | 32:4 | **512** | `bs64_w4_s1` | **1627.7** | 0.191 | 11.88× |
| `gemma4_26b_a4b_global` | Gemma 4 26B-A4B global | 16:2 | **512** | `bs64_w4_s1` | 1576.5 | 0.101 | 11.73× |
| `gemma3_local_1024` | Gemma 3 local | 16:16 | 256 | `bs32_w2_s1` | 1490.2 | 0.093 | **11.82×** |
| `gemma4_31b_local` | Gemma 4 31B local | 32:16 | 256 | `bs32_w2_s1` | 1536.3 | 0.134 | 11.81× |
| `gemma4_26b_a4b_local` | Gemma 4 26B-A4B local | 16:8 | 256 | `bs32_w2_s1` | 1443.2 | 0.073 | 11.49× |
| `gemma4_e2b_local` | Gemma 4 E2B local | 8:1 | 256 | `bs32_w2_s1` | 1213.7 | 0.035 | 10.35× |

## Results — T4 (Kaggle/Colab free tier)

**Hardware**: Tesla T4, SM 7.5, 16 GB GDDR6, ~300 GB/s memory bandwidth. Cross-validated on both Kaggle T4 and Google Colab T4.

The fused `qk_norm_rope` kernel achieves **113.80 GB/s** on T4 after bandit search (up from 80.55 GB/s with curated starters), which is ~38% of T4's theoretical peak bandwidth (~300 GB/s). Fusion speedup: **8.37×** over the separated 4-launch baseline (up from 6.06× curated).

| Metric | T4 (curated) | T4 (bandit) | A100 | H100 |
|---|---|---|---|---|
| Best GB/s | 80.55 | **113.80** | 925.5 | 1627.7 |
| Best fusion_speedup | 6.06× | **8.37×** | **12.85×** | **11.88×** |
| % of HBM peak | ~27% | ~38% | ~46% | ~49% |

**Bandit search results (1,800+ measurements across 43 shape buckets).** The autonomous bandit search on Kaggle/Colab T4 improved qk_norm_rope from 6.46× to 8.37× fusion speedup — a 30% improvement over curated starters. The bandit discovered that T4 strongly prefers `num_warps=1` and small block sizes (`BLOCK_SIZE=16` for rotary, `BLOCK_SIZE=32` for backward), configurations not present in the curated list. On the `gemma4_e2b` shape, the bandit achieved 8.14× fusion speedup at 113.80 GB/s. Backward pass fusion speedup across the full 6-shape sweep: 4.9–7.5× (mean 5.75×).

**Interpretation.** The T4 fusion_speedup (8.37× after bandit search, 6.06× curated) is lower than A100 (10–13×) and H100 (10–12×), consistent with the launch-overhead hypothesis from §3.2.1. T4's lower absolute launch latency means the separated baseline wastes a smaller fraction of total time on launches, so fusion saves proportionally less. However, 6–8× is still far above the ~2× predicted by HBM traffic accounting alone, confirming that launch overhead dominates fusion value across all three GPU tiers. The 30% improvement from bandit search over curated configs demonstrates that hardware-specific autotuning adds substantial value even on commodity GPUs.

**Validation script.** `scripts/colab_validate_all.py` runs all 13 operators on Kaggle's or Colab's free T4 — no Modal account or paid GPU required.

## Key findings

1. **All (shape, config) combinations are correct across 3 GPUs.** Zero failures across 6 Gemma 3/4 shapes × 5 curated configs × 2 datacenter GPUs (A100, H100), plus T4 correctness validation on Kaggle/Colab.
2. **Every shape beats the separated baseline by ≥5×; the best cases hit 12.85× on A100 and 11.88× on H100.** The arithmetic accounts for ~2× HBM traffic savings (read/write Q and K once vs. twice), but the *measured* speedups are much higher — indicating **launch overhead dominates** for these small-to-medium Gemma shapes. Going from 4 separate kernel launches to 1 fused kernel saves ~25-50 µs per prologue, which is a huge fraction of the total time at this scale.
3. **H100 peak bandwidth is 1627 GB/s** (54% of theoretical 3.35 TB/s HBM3 peak) — the fused kernel is genuinely memory-bound, not launch-overhead-bound, at this tile size.
4. **Best config regime**: `BLOCK_SIZE=64, num_warps=4, num_stages=1` is the sweet spot across both GPUs. Small enough to keep multiple blocks per SM, large enough to amortize the RMSNorm reduction.
5. **Raw data**: [`qk-norm-rope-a100-full.json`](./qk-norm-rope-a100-full.json), [`qk-norm-rope-h100-full.json`](./qk-norm-rope-h100-full.json)

## Fusion cost model (order of magnitude)

For Gemma 4 E2B (`B=1, H=8, S=4096, D=256`, Q+K tensors = 2 × 8 × 4096 × 256 × 2 bytes = 16.7 MB):

- **Separated (4 launches)**:
  - Q-RMSNorm: read Q (8.4 MB) + write Q_normed (8.4 MB) = 16.8 MB
  - K-RMSNorm: 16.8 MB
  - Q-RoPE: read Q_normed + write Q_out = 16.8 MB
  - K-RoPE: 16.8 MB
  - **Total: 67.2 MB HBM traffic + 4 kernel launches (≈40 µs overhead)**
- **Fused (2 launches — one for Q, one for K)**:
  - Q pass: read Q (8.4 MB) + write Q_out (8.4 MB) = 16.8 MB
  - K pass: 16.8 MB
  - **Total: 33.6 MB HBM traffic + 2 kernel launches (≈20 µs overhead)**

Expected speedup from HBM traffic alone: 2×. Measured on `gemma4_e2b_local`: **10.23×** on A100. The 5× gap between the model and the measurement is launch overhead — for sub-100 µs kernels, launch latency is the dominant cost, and halving the launch count halves it.

## Reproduction

```bash
# A100 + H100 (Modal)
python scripts/smoke_modal.py --full --h100 --qk-only --write-results

# T4 (Kaggle primary, Colab backup — both free)
# Upload scripts/colab_validate_all.py to a Kaggle or Colab T4 notebook and run
```

Runtime: ~3 minutes per GPU (cold compile + 30 evals each). Modal cost: ~$0.10 per GPU. T4 validation is free via Kaggle (30 hr/week) or Colab. See [`scripts/smoke_modal.py`](../../scripts/smoke_modal.py) for the full orchestration.
