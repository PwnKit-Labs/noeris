# Competitive Benchmarking: Noeris vs the Field

**Date**: 2026-04-12
**Purpose**: Honest assessment of where Noeris stands against published competitors, to inform submission decisions.

---

## 1. KernelBench Leaderboard Comparison

Noeris evaluates on a curated 53-problem subset (L1-L2, FP16, A100/H100). The published systems evaluate on the full 250-problem KernelBench suite (L1-L3, CUDA backend unless noted).

| System | Problems | Hardware | fast_1.0 | Avg Speedup | Correctness | Backend |
|---|---|---|---|---|---|---|
| **Noeris** | 53 (curated) | A100/H100 | **56.6%** | ~3-4x (mem-bound ops) | 96% (2 FAILs) | Triton |
| [Kernel-Smith-235B-RL](https://arxiv.org/abs/2603.28342) | 250 | NVIDIA (L40S?) | **70%** | **3.70x** | 96% | Triton |
| [KernelSkill](https://arxiv.org/abs/2603.10085) | 250 | A100-80GB | 62% (L1) / 100% (L2) / 82% (L3) | 5.44x (L1), 2.82x (L2), 1.92x (L3) | **100%** | CUDA |
| [CUDA Agent](https://arxiv.org/abs/2602.24286) | 250 | A100 | **100% (L1-L2)**, 92% (L3) | 2.80x geomean vs compile | 100% (L1-L2) | CUDA |
| [CUDA-L1](https://arxiv.org/abs/2507.14111) | 250 | A100 | — | **3.12x** avg, 1.42x median | — | CUDA |

**Verdict: Noeris loses on coverage and aggregate scores.** Our 53-problem curated subset is not comparable to 250-problem full-suite evaluation. CUDA Agent achieves 100% fast_1.0 on L1-L2; Kernel-Smith hits 70% overall. Our 56.6% is measured on an easier subset. On the two upstream L1 problems we tested apples-to-apples (matmul 1.14x, cross-entropy 1.05x), the adapters were still running torch references, not our Triton kernels. This is not a credible KernelBench claim yet.

---

## 2. Per-Operator Head-to-Head: Noeris vs AutoKernel on H100

AutoKernel ([arXiv:2603.21331](https://arxiv.org/abs/2603.21331)) is the closest Triton-on-H100 comparison. Both measure against PyTorch eager.

| Operator | Noeris Best (H100) | AutoKernel Best (H100) | Noeris Advantage | Note |
|---|---|---|---|---|
| **RMSNorm** | **11.66x** (2625 GB/s, Mixtral) | 5.29x (2788 GB/s) | **+120%** speedup ratio | Noeris wins on speedup; AutoKernel's absolute GB/s is comparable due to different shapes |
| **Cross-entropy** | **9.65x** (2407 GB/s, long_llama) | 2.21x (2070 GB/s) | **+337%** speedup ratio | Clear win |
| **Softmax** | **6.38x** (2526 GB/s, vocab_llama) | 2.82x (2800 GB/s) | **+126%** speedup ratio | Win on speedup; AutoKernel reports higher absolute GB/s on some shapes |
| **LayerNorm** | 1.53x (1667 GB/s) | **3.21x** vs compile (~1.07-1.25x vs eager) | **Ambiguous** | AutoKernel's 3.21x is vs torch.compile, not eager. Apples-to-oranges. Noeris likely loses on large shapes. |
| **GeGLU** | **3.87x** (2287 GB/s, gemma26b) | Not reported | N/A | Novel operator not in AutoKernel |
| **Matmul** | 1.01x (692 TFLOPS, llama7b) | Not competitive vs cuBLAS | Tie at ~1x | Neither beats cuBLAS meaningfully |
| **Attention** | 0.68-0.83x of SDPA | Not reported | **Lose** | Our simplified kernel loses to FlashAttention-2/3 |

**Verdict: Noeris clearly wins on RMSNorm, cross-entropy, and softmax against AutoKernel.** The speedup ratios are 2-4x better. However, AutoKernel reports comparable absolute GB/s on some shapes, suggesting part of our advantage comes from a weaker PyTorch eager baseline on our chosen shapes. LayerNorm is our weakest operator and likely loses to AutoKernel.

---

## 3. Fused QK-RMSNorm+RoPE (Practical Implementation)

| Metric | Noeris | Nearest Competitor |
|---|---|---|
| Fusion speedup (A100) | **10.2-12.9x** | vLLM has fusion pass but disabled by default (H100 regression) |
| Fusion speedup (H100) | **10.4-11.9x** | vLLM has fusion pass but disabled by default (H100 regression) |
| Peak bandwidth (H100) | **1628 GB/s** (49% of HBM3 peak) | N/A |

**Competitive landscape check:**
- **vLLM**: Has an experimental `enable_qk_norm_rope_fusion` pass (torch.compile + CUDA kernel) but it is **disabled by default** due to H100 performance regression ([issue #34391](https://github.com/vllm-project/vllm/issues/34391)). With fusion off, vLLM issues 4+ separate launches in `gemma4.py:395-427`. Reported benefit when enabled: 2-3% E2E. ([source](https://github.com/vllm-project/vllm/pull/38826))
- **Liger Kernel**: Has individual RMSNorm, RoPE, GeGLU kernels. No fused QK-norm+RoPE. ([source](https://github.com/linkedin/Liger-Kernel))
- **Unsloth**: Has a 2.3x faster fused QK-RoPE kernel, but it does NOT include RMSNorm in the fusion. Different (smaller) fusion scope.
- **FlashAttention-4**: Fuses prologue into the attention kernel on Blackwell (B200), reaching 1605 TFLOPS. But FA4 targets Blackwell-specific async MMA; it does not exist for A100/H100. ([source](https://www.together.ai/blog/flashattention-4))
- **FlashInfer**: No published fused prologue kernel.

**Verdict: The fusion idea has prior art (vLLM's `enable_qk_norm_rope_fusion`, Flash Normalization arXiv:2407.09577). Our contribution is making it practical.** vLLM's torch.compile+CUDA approach is disabled by default due to H100 regressions. Our parameterized Triton implementation with bandit-tuned configs achieves 10-13x prologue speedup reliably across A100/H100. The 10-13x speedup is real and attributable to launch-overhead elimination (4 launches reduced to 2). FlashAttention-4 subsumes this on Blackwell but does not exist for prior hardware. Unsloth's QK-RoPE fusion is the nearest work but excludes normalization. The novelty is the SYSTEM (autonomous search + parameterized kernels), not the fusion idea itself.

---

## 4. Honest Gaps

### Where Noeris definitively loses

1. **KernelBench coverage**: 53 curated problems vs 250 full suite. CUDA Agent hits 100% on L1-L2. Kernel-Smith averages 3.70x across all 250. We cannot claim KernelBench competitiveness.

2. **Matmul**: 0.44-1.01x of cuBLAS on H100. Triton matmul cannot beat hand-tuned cuBLAS/CUTLASS. DeepGEMM reaches 970 TFLOPS FP8 on H100; our best is 692 TFLOPS FP16. This is an expected and permanent gap.

3. **Attention**: 0.68-0.83x of SDPA/FlashAttention-2/3 on H100. Our 120-line reference kernel is not competitive with production FlashAttention. FlashAttention-4 on Blackwell hits 1605 TFLOPS. We are not in this game.

4. **LayerNorm**: 1.25-1.53x vs eager, while AutoKernel claims 3.21x vs compile. Our Welford implementation is not optimized. This is fixable but currently a gap.

5. **Cross-run learning**: The central architectural claim (persistent shape-indexed database improves search) showed no statistically significant effect in ablation. -2.16% on matmul, -1.25% on rmsnorm, both within noise. The cost model and bandit selectors show value (+45-133% on matmul), but the cross-run persistence story is empirically unproven.

6. **Upstream KernelBench integration**: The 2 problems we tested apples-to-apples used torch reference adapters, not real Triton kernels. The claimed speedups on our curated suite do not transfer to upstream problem formats without adapter work (e.g., RMSNorm 4D reshape kills our 11.66x claim).

### Where Noeris wins

1. **Practical fused kernel**: 10-13x on QK-RMSNorm+RoPE is a real, reproducible result. Prior art exists (vLLM's `enable_qk_norm_rope_fusion`, SGLang's `fused_qknorm_rope`) but is disabled or limited. Our Triton implementation with autotuning makes the fusion practical on A100/H100 and supports Gemma's `(1+w)` affine mode.
2. **Memory-bound operators**: RMSNorm (11.66x), cross-entropy (9.65x), softmax (6.38x) beat AutoKernel by 2-4x on speedup ratios.
3. **Cost**: Full evaluation costs $1.44. Reproduction is trivial.
4. **Search infrastructure**: Cost model (+45% matmul) and bandit selector (+133% matmul) demonstrably outperform random grid search when the parameter space is large.

---

## 5. Venue Assessment

### What tier can Noeris target?

**MLSys / NeurIPS Systems track: Workshop paper or poster, not oral.**

The fused QK-RMSNorm+RoPE kernel is a legitimate systems contribution, but a single fused kernel for one model family (Gemma 3/4) is narrow scope for a top venue oral. The search infrastructure (cost model + bandit) is interesting but the ablation is negative.

### Likely reviewer criticisms

1. "The 10-13x speedup is just launch overhead elimination for small tensors. The HBM model predicts 2x; the rest is PyTorch overhead, not algorithmic innovation."
   - **Rebuttal**: Launch overhead IS the bottleneck at these shapes. vLLM attempted the fusion (`enable_qk_norm_rope_fusion`) but had to disable it due to H100 regressions — our Triton approach avoids this. The kernel reaches 49% of HBM peak, proving it is not just overhead avoidance. The practical impact on Gemma 3/4 inference is real regardless of the source of the speedup.

2. "KernelBench evaluation on 53 curated problems is not comparable to the standard 250-problem suite."
   - **Rebuttal**: Acknowledged. We do not claim KernelBench SOTA. The curated suite tests our Triton templates on shapes that matter for LLM inference. Full-suite evaluation is future work.

3. "The cross-run learning ablation is negative. The paper's central claim is unsubstantiated."
   - **Rebuttal**: This is the hardest criticism. The honest answer is that curated starters are too strong for 5-iteration budgets. The cost model and bandit DO show value (Table in three-way-summary). Reframing: the contribution is the search infrastructure (cost model + bandit + persistent database), not cross-run learning alone. The ablation is reported transparently as a negative result.

4. "AutoKernel reports comparable absolute GB/s. The speedup ratio advantage may reflect a weaker baseline."
   - **Rebuttal**: Partially valid. Different input shapes yield different eager baselines. Head-to-head on identical shapes is needed. We can run AutoKernel's exact shapes.

### Strongest submission angle

**"Fused operator kernels for Gemma-family inference + a search infrastructure that discovers hardware-specific configs across GPU tiers."** Lead with the practical fused kernel (10-13x prologue speedup, three GPUs; prior art exists but is disabled/limited — vLLM reports 2-3% E2E), support with the cost model/bandit ablation, and present the per-operator results as secondary evidence. Acknowledge matmul and attention losses honestly.

**Recommended venue**: MLSys 2027 (systems track), NeurIPS 2026 workshop on efficient ML, or ISCA/MICRO workshop on GPU optimization. A full MLSys/NeurIPS main-track paper would need either (a) the cross-run learning ablation to go positive with longer budgets, or (b) upstream KernelBench full-suite results that are competitive with Kernel-Smith/CUDA Agent.

---

## Sources

- [KernelBench Leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)
- [Kernel-Smith (arXiv:2603.28342)](https://arxiv.org/abs/2603.28342)
- [KernelSkill (arXiv:2603.10085)](https://arxiv.org/abs/2603.10085)
- [AutoKernel (arXiv:2603.21331)](https://arxiv.org/abs/2603.21331)
- [CUDA Agent (arXiv:2602.24286)](https://arxiv.org/abs/2602.24286)
- [CUDA-L1 (arXiv:2507.14111)](https://arxiv.org/abs/2507.14111)
- [KernelEvolve (arXiv:2512.23236)](https://arxiv.org/abs/2512.23236) -- Meta, ISCA 2026, 100% on 250 KernelBench problems
- [FlashAttention-4](https://www.together.ai/blog/flashattention-4) -- Blackwell-only, fuses prologue
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) -- No fused QK-norm+RoPE
- [vLLM Gemma 4 (PR #38826)](https://github.com/vllm-project/vllm/pull/38826) -- 4+ separate launches confirmed
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) -- 970 TFLOPS FP8 grouped GEMM on H100
- [vLLM Triton Attention Backend](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)
