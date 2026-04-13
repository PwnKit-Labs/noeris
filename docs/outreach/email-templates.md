# Outreach Email Templates

---

## Template 1: Open-Source Maintainers (vLLM / FlashInfer / SGLang)

**Subject:** Fused QK-RMSNorm+RoPE Triton kernel — 40 launches down to 1, works on H100

Hi [Name],

I'm Doruk Tan Ozturk, an independent researcher / solo developer working on Noeris, an open-source autonomous GPU kernel optimization system. I wanted to flag something that might be relevant to [project] — specifically [issue/PR ref, e.g., vLLM's `enable_qk_norm_rope_fusion` which is disabled by default (#34391) / FlashInfer's lack of QK-norm+RoPE fusion (#2971)].

I built a fused QK-RMSNorm+RoPE Triton kernel that collapses PyTorch eager's 40 kernel launches (torch.compile gets it to 9) down to 1. I've benchmarked it across 19 models spanning 13 architecture families (LLaMA 3/4, Qwen 3, Mistral, Mixtral, Phi-3/4, Falcon 3, DBRX, OLMo 2, InternLM 3, Gemma 4):

- **6.5-8.9x** prologue speedup on T4; **10.2-12.9x** on A100, **10.4-11.9x** on H100
- **1.18x** end-to-end on a 26-layer Gemma 4 E2B forward pass (322.7ms to 274.4ms)
- **4.9-7.5x** backward pass fusion — usable for training, not just inference
- Handles Gemma's `(1+w)` affine RMSNorm mode (silently wrong output if you use standard form with Gemma weights — no NaN, no crash, just ~10x magnitude error)
- Peak 1,627 GB/s on H100 (49% of HBM3 theoretical peak)

The kernel was found through an autonomous bandit search over parameterized Triton templates — the bandit discovers hardware-specific configs that beat hand-tuned starting points by 22-167%. On H100 specifically, it learned that packing multiple heads per warp is critical, which is likely why vLLM's 1-head-per-warp approach regresses there.

Everything is MIT-licensed: [github.com/PwnKit-Labs/noeris](https://github.com/PwnKit-Labs/noeris)

I'd be happy to contribute a PR, adapt the kernel to your codebase, or just discuss the approach. No expectations — if it's useful, great.

Best,
Doruk Tan Ozturk
GitHub: [@peaktwilight](https://github.com/peaktwilight)

---

## Template 2: Academic Researchers (Tri Dao, Song Han, etc.)

**Subject:** Autonomous kernel search for fused attention prologues — seeking feedback on approach

Dear Professor [Name],

I'm Doruk Tan Ozturk, an independent researcher working on Noeris, an autonomous GPU kernel optimization system. Your work on [specific paper, e.g., FlashAttention / TinyChat / SparseGPT] has been a major influence on this project, and I'd genuinely value your perspective.

There are two contributions I'd like your feedback on:

**1. A fused QK-RMSNorm+RoPE kernel that makes this fusion practical across hardware.** This isn't a new idea — vLLM has `enable_qk_norm_rope_fusion` (disabled by default due to H100 regression, #34391), SGLang has `fused_qknorm_rope`, and Liao et al. describe algebraic optimizations in "Flash Normalization." What's new is a parameterized Triton implementation with bandit-tuned configs that delivers the fusion reliably:

- **10.2-12.9x** prologue speedup on A100, **10.4-11.9x** on H100, across 19 models spanning 13 architecture families (LLaMA, Qwen, Mistral, Phi, Falcon, DBRX, OLMo, InternLM, Gemma, and others)
- **1.18x** end-to-end speedup on a 26-layer Gemma 4 forward pass — 6-9x more E2E impact than vLLM's reported 2-3%
- **4.9-7.5x** backward pass fusion speedup, making this usable for training (to my knowledge, no existing framework fuses the Gemma prologue backward pass)
- Handles Gemma's `(1+w)` affine RMSNorm mode — a correctness subtlety that breaks existing fusions
- Compiler analysis shows `torch.compile` emits 9 launches where Noeris emits 1 (it splits at the RMSNorm reduction boundary because Inductor can't assume the row fits in registers)

**2. An autonomous bandit search system** that finds kernel configs 22-167% better than hand-tuned baselines. The system uses a shape-indexed cross-run config database, a learned cost model (R^2 = 0.94), and a multi-armed bandit with meta-bandit routing. A100-trained rankings transfer to H100 with Spearman rho = 0.967. The real contribution is this search infrastructure — the fused kernel is its headline result.

I'm preparing an arXiv preprint and would be grateful for any feedback on the positioning — even brief comments would be genuinely helpful. This is a solo project, so outside perspective is especially valuable.

Full system is open-source under MIT: [github.com/PwnKit-Labs/noeris](https://github.com/PwnKit-Labs/noeris)

Respectfully,
Doruk Tan Ozturk
Independent researcher / solo developer
GitHub: [@peaktwilight](https://github.com/peaktwilight)

---

## Template 3: Industry Engineers (NVIDIA, Meta, etc.)

**Subject:** Fused the attention prologue from 40 launches down to 1 — works on H100, handles Gemma affine

Hi [Name],

I'm Doruk Tan Ozturk, an independent developer behind Noeris, an open-source autonomous kernel optimization system. I saw [team/project]'s work on [specific topic, e.g., fused attention prologues / RoPE performance / kernel launch overhead] and wanted to share something that might be relevant.

The problem: the attention prologue (QK-RMSNorm + RoPE) is 40 CUDA launches in PyTorch eager. `torch.compile` gets it to 9 but still splits at the RMSNorm reduction boundary — 4 separate Triton kernels. Noeris fuses it to 1 kernel, 1 launch.

Results across 19 models / 13 architecture families:

- **10.2-12.9x** prologue speedup on A100, **10.4-11.9x** on H100
- **6.5-8.9x** on T4 (best: Qwen 3 32B at 8.9x / 120 GB/s)
- **1.18x** end-to-end on a 26-layer Gemma 4 forward pass
- **4.9-7.5x** backward pass fusion — training-applicable, not inference-only
- Handles Gemma's `(1+w)` affine RMSNorm mode, which silently breaks other fusions
- Peak 1,627 GB/s on H100 (49% HBM3 peak)

The kernel configs are discovered by an autonomous bandit search over parameterized Triton templates — no hand-tuning. The bandit finds configs 22-167% better than curated starting points, and learned rankings transfer across hardware (Spearman 0.967 from A100 to H100). On H100, it discovered that multi-head-per-warp packing is critical — a config choice that explains why vLLM's `enable_qk_norm_rope_fusion` regresses on H100 and remains disabled (#34391).

MIT-licensed, reproducible for ~$0.20 on Modal: [github.com/PwnKit-Labs/noeris](https://github.com/PwnKit-Labs/noeris)

If this overlaps with anything your team is exploring, I'd welcome a conversation. Happy to adapt the kernel or contribute directly.

Best,
Doruk Tan Ozturk
Independent researcher / solo developer
GitHub: [@peaktwilight](https://github.com/peaktwilight)
