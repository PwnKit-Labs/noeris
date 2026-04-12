# Strategic Next Steps — Noeris

*Written 2026-04-12. Brutally honest assessment for 2-week sprint planning.*

---

## 1. What Is Actually Novel (and What Is Not)

**Genuinely novel:**

- **Fused QK-RMSNorm+RoPE forward+backward.** 10-13x fwd, 4.9-7.5x bwd vs separated launches. vLLM has an experimental fusion (`enable_qk_norm_rope_fusion`) but it is disabled by default due to H100 perf regression. Our Triton implementation with bandit-tuned configs makes the fusion practical. The backward pass remains novel — vLLM's fusion is inference-only.
- **From-scratch Triton paged-KV decode attention.** vLLM's equivalent is CUDA-only (`paged_attention_v1.cu`). A pure-Triton implementation is a genuine contribution to the Triton ecosystem.
- **2300+ measurement config database on free T4.** Nobody else runs autonomous search on Colab. The persistent (op, shape, hardware) index with cross-run accumulation has no published open-source equivalent.
- **MAP-Elites quality-diversity archive for config tuning.** KernelFoundry uses MAP-Elites for kernel code structure; using it for Triton config space with hardware-behavioral dimensions is distinct.

**Not novel (and reviewers will know):**

- **Thompson-sampling bandit.** Standard technique. The adaptive meta-router is mildly interesting but +0.5% over best fixed selector is not a paper contribution.
- **GBR cost model.** R^2=0.94 cross-operator is competent engineering, not a research contribution. Per-operator GP failed (negative R^2). The cost model is a system component, not a headline.
- **LLM proposer.** Marginal value demonstrated. Every competing system has one. The cross-run insights injection is the only differentiating aspect, and it lacks ablation data showing it matters.
- **14 operator coverage.** Breadth is good for a system paper but is not novel per se. Liger Kernel covers more operators.

**Honest positioning:** Noeris has one strong kernel result (fused prologue — prior art exists in vLLM's disabled `enable_qk_norm_rope_fusion`, but our Triton implementation makes it practical), one interesting system property (persistent cross-run shape-indexed learning), and a lot of solid engineering. The paper needs to make those two things undeniable and stop trying to claim novelty where there is none. The novelty is the SYSTEM, not the fusion idea itself.

---

## 2. Top 3 Things to Build for Maximum Paper Impact

### (1) End-to-end Gemma 4 layer timing breakdown — HIGH URGENCY

The paper's biggest weakness right now: the 10-13x claim is on the prologue in isolation. A reviewer will immediately ask "what fraction of total layer time is this?" Without answering that, the paper is dead on arrival.

**What to do:** Profile one full Gemma 4 31B decoder layer (prefill, seq_len=2048) on A100 using `torch.cuda.Event` or NSight. Decompose into: QKV projection, QK-RMSNorm+RoPE (prologue), attention, output projection, MLP/MoE, LayerNorm+residual. Report the prologue as % of total. Then report actual end-to-end layer speedup from the fusion.

**Estimated outcome:** Based on the vLLM kernel pattern audit, the prologue (kernels 3-5) is ~4-8% of a prefill-dominated layer (the QKV GEMM and attention dominate). For decode (memory-bound, tiny matmuls), prologue fraction rises to ~10-20% because GEMMs shrink. Honest claim: "1.04-1.08x layer speedup on prefill, 1.10-1.20x on decode." That is still publishable if presented honestly alongside the 10-13x micro-kernel number.

### (2) Shape-transfer ablation with real cross-run evidence — THE PAPER THESIS

The cross-run shape-indexed learning is the system's theoretical differentiator, but there is no controlled experiment proving it works. The GP surrogate failed. The GBR cost model transfers cross-hardware (rho=0.967) but that is cross-hardware, not cross-shape.

**What to do:** Design a clean ablation: (a) cold-start search on 10 unseen shape buckets, (b) warm-start search using the config database from 100+ prior shapes. Measure iterations-to-best and final throughput. If warm-start converges 3-5x faster, that is the paper's second headline. If it does not, drop the cross-run learning claim and reposition around the fused kernel + open Triton ecosystem.

**Risk:** This experiment might show the config database helps minimally (configs are operator-specific enough that cross-shape transfer is weak). Better to know now than have a reviewer discover it.

### (3) Backward pass kernel correctness on A100/H100 — COMPLETE THE STORY

The backward pass (4.9-7.5x on T4) is currently only validated on Colab T4. Running it on A100/H100 with the same rigor as the forward pass (cuda_event timing, L2 flush, all 6 shape buckets) turns the paper from "inference micro-optimization" into "training-applicable fusion." This is a much stronger story for MLSys/NeurIPS reviewers who care about training.

**Estimated cost:** ~$0.40 Modal. One afternoon of work.

---

## 3. The "10x Claim" — Honest End-to-End Assessment

The prologue fuses kernels 3-5 from the vLLM audit (Q-RMSNorm, K-RMSNorm, rotary). Here is the arithmetic:

**Prefill (seq_len=2048, Gemma 4 31B, A100):**
- QKV projection (cuBLAS GEMM): ~40-50% of layer time
- Attention (FlashAttention): ~25-35%
- Output projection + MLP GEMMs: ~15-20%
- Prologue (QK-norm + RoPE): ~3-7% (sub-100us kernels, dominated by launch overhead)
- LayerNorm + residuals: ~2-5%

With 10x prologue speedup, layer speedup = 1 / (1 - 0.05 + 0.05/10) = ~1.047x.

**Decode (batch=1, seq_len=1):**
- GEMMs shrink dramatically (memory-bound at batch=1)
- Prologue fraction rises to ~10-20% because everything is memory-bound and small
- Layer speedup = 1 / (1 - 0.15 + 0.15/10) = ~1.16x

**Honest framing for the paper:** "The fused prologue kernel achieves 10-13x over the separated baseline on the prologue operation. In a full Gemma 4 decoder layer, this translates to 1.04-1.08x prefill speedup and 1.10-1.20x decode latency improvement per layer. The value is primarily in eliminating kernel launch overhead for latency-sensitive decode paths." This is defensible. Claiming "10x" without context is not.

**The real opportunity:** Stack multiple Noeris-optimized operators (prologue + GeGLU + cross-entropy + MoE router) for a combined 1.15-1.25x end-to-end claim. That requires the layer timing breakdown from item (1) above.

---

## 4. Should Noeris Pivot to Generating Kernel Code?

**No. Not now. Maybe later.**

Arguments for staying template-based:
- **Reproducibility.** Every result maps to a typed config tuple. A reviewer can reproduce any number by running one command with one config dict. Code-generating systems produce non-deterministic outputs.
- **Correctness guarantees.** The template is verified once; only the config varies. Code-gen systems have 60-90% pass rates. Noeris has 0 correctness failures across 60 (shape, config) pairs.
- **The config space is underexplored.** 2300 T4 measurements across 110 shape buckets is a start, but the full grid for one operator (e.g., matmul with 6 config dimensions) has 10^4+ points. There is still signal to extract from template-based search.
- **Differentiator.** Every new system (Kernel-Smith, AutoKernel, KernelEvolve, K-Search) generates code. Noeris occupying the "disciplined template search with persistent learning" niche is a cleaner story than being the 8th code-gen system.

Arguments for eventually adding code gen:
- Template-based search hits a ceiling. Once the best config for a shape is found, there is no way to improve further without changing the kernel structure (e.g., different tiling strategy, different memory access pattern).
- The fused QK-RMSNorm+RoPE kernel was itself a manual template creation. Automating that process (identifying fusion opportunities, generating fused templates) is a natural extension.

**Recommendation:** Ship the current paper as a template-search system. Add code-gen as future work. If the paper gets strong reviews, the follow-up paper is "Noeris v2: from config search to template synthesis."

---

## 5. The Colab Angle

**It is a real contribution, not a gimmick.** Here is why:

- **Democratization claim is credible.** AutoKernel runs on H100. KernelEvolve runs on Meta's fleet. K-Search needs A100+. Noeris's bandit discovered configs on free T4 that improved fusion speedup from 6.06x to 8.37x (30% improvement over curated starters). That is a meaningful result obtained at zero dollar cost.
- **Reproducibility.** Any reviewer can replicate the T4 results without a cloud account. This is a genuine advantage for peer review.
- **Hardware-diversity data.** T4 + A100 + H100 results in the same paper show the system generalizes across 3 GPU generations. The cross-hardware transfer (Spearman rho=0.967) is validated on real data, not simulated.

**How to frame it:** "Noeris is, to our knowledge, the first autonomous kernel optimization system that runs its full search loop on free-tier consumer GPUs." One sentence, factual, not oversold.

**Caveat:** Do not make Colab the headline. It is a supporting claim that makes the system accessible. The headline is the fused kernel and the cross-run learning.

---

## 6. Concrete 2-Week Sprint Plan

### Week 1: Make the paper defensible

**Day 1-2: End-to-end layer timing breakdown.**
- Profile Gemma 4 31B decoder layer on A100 (Modal). Decompose into the 6+ kernel categories from the vLLM audit.
- Produce Table: "Kernel category | Baseline time (us) | With Noeris fusion | Speedup".
- This is the most important missing artifact.

**Day 3: Backward pass on A100/H100.**
- Run `smoke_modal.py` equivalent for backward pass on both GPUs.
- Produce the same table format as forward (6 shapes x best config, fusion speedup).
- Cost: ~$0.40. Blocks: nothing.

**Day 4-5: Shape-transfer ablation.**
- Select 10 shape buckets held out from training. Run cold-start (uniform random) vs warm-start (database-seeded) search, 50 iterations each, 3 seeds.
- Measure: iterations to reach 90% of best known, final throughput delta.
- If positive: this becomes Section 4.X of the paper. If negative: drop the cross-run learning claim and reframe around the config database as a reproducibility artifact.

**Day 6-7: Paper revision.**
- Integrate layer timing into Section 4 (evaluation).
- Add backward pass results to Section 3.2.1 (fused kernel).
- Write honest "end-to-end impact" subsection with Amdahl's law accounting.
- Cut any overclaiming about cost model or LLM proposer novelty.

### Week 2: Strengthen the story

**Day 8-9: Multi-operator stacking experiment.**
- Deploy fused prologue + best GeGLU config + best cross-entropy config in a simulated Gemma 4 layer.
- Measure combined speedup vs all-PyTorch-eager baseline.
- Target: 1.15-1.25x combined layer speedup (prefill).

**Day 10-11: Paged-KV decode attention benchmarks.**
- The from-scratch Triton paged-KV decode exists but lacks published A100/H100 numbers.
- Run full shape sweep. Compare against vLLM's CUDA paged attention (if accessible via Python API) or against FlashDecoding.
- This is the second novel kernel claim.

**Day 12: MAP-Elites behavioral dimension analysis.**
- Visualize the MAP-Elites archive: plot config diversity along (memory-bound vs compute-bound, tile size, occupancy) axes.
- Produce one figure showing the quality-diversity archive discovers configs that grid search misses.

**Day 13-14: Final paper polish + artifact preparation.**
- Ensure every claim has a reproduction command.
- Package Colab notebook for T4 validation.
- Write related work section positioning against the 7 new April 2026 systems.
- Prepare supplementary material with full benchmark tables.

### Key milestones

| Day | Deliverable | Paper impact |
|---|---|---|
| 2 | Layer timing breakdown | Answers the #1 reviewer question |
| 3 | Backward A100/H100 numbers | Training applicability claim |
| 5 | Shape-transfer ablation | Validates or kills the system thesis |
| 9 | Multi-operator stacking | End-to-end speedup claim |
| 11 | Paged-KV decode numbers | Second novel kernel |
| 14 | Submission-ready draft | — |

### What NOT to build

- Do not add NCU profiling integration. It is 2-3 weeks of work and the paper does not need it for submission.
- Do not attempt code generation. Wrong time horizon.
- Do not chase FP8. Needs H100 access patterns that are not in the current infrastructure.
- Do not try to beat KernelBench SOTA. The fast_1.0 = 56.6% number is fine as a system sanity check, not the headline.

---

*Bottom line: The paper has one undeniable result (fused prologue, 10-13x) and one unproven thesis (cross-run shape-indexed learning). Week 1 proves or disproves the thesis and makes the kernel claim end-to-end honest. Week 2 stacks evidence. Ship by day 14.*
