# Gemma Family Architecture Analysis for Noeris

**Date:** 2026-04-11  
**Scope:** Google Gemma 1 through Gemma 4; implications for Noeris GPU kernel optimization  

---

## 1. What Is the Latest Gemma Model?

As of April 2026, **Gemma 4 is the current generation**, released on April 2, 2026. The family has progressed: Gemma 1 (Feb 2024) → Gemma 2 (Jun 2024) → Gemma 3 (Mar 2025) → Gemma 4 (Apr 2026).

**Gemma 4 model family:**

| Variant | Total Params | Active Params | Context | Key New Feature |
|---------|-------------|---------------|---------|-----------------|
| E2B | 5.1B total | 2.3B | 128K | Per-Layer Embeddings (PLE) |
| E4B | 8B total | 4.5B | 128K | Per-Layer Embeddings (PLE) |
| 26B A4B | 26B total | ~3.8B active | 256K | MoE (128 experts, top-2 routing) |
| 31B Dense | 31B | 31B | 256K | Proportional RoPE, dense |

All Gemma 4 models are multimodal (text + vision + audio), released under Apache 2.0.

### Gemma 3 (March 2025) — the previous generation

Gemma 3 remains widely deployed for text-only fine-tuning. Key specs:

| Model | Non-emb Params | Train Tokens | Context |
|-------|---------------|--------------|---------|
| 1B | 698M | 2T | 32K |
| 4B | 3.2B | 4T | 128K |
| 12B | 10.8B | 12T | 128K |
| 27B | 25.6B | 14T | 128K |

**Gemma 3 27B architectural parameters** (from HuggingFace config):
- Hidden size: 5376
- Layers: 62
- Attention heads: 32
- KV heads: 16 (GQA ratio 2:1)
- Head dim: 128
- Intermediate (MLP): 21,504
- Vocab: 262,144 (Gemini 2.0 SentencePiece tokenizer)

**Gemma 2 27B** (for comparison, fully documented in the Gemma 2 tech report):
- Layers: 46, Hidden: 4608, Heads: 32, KV heads: 16, Head dim: 128, MLP: 73,728, Vocab: 256,128

Sources: [Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/abs/2503.19786), [Gemma 4 HuggingFace blog](https://huggingface.co/blog/gemma4), [Gemma 2 arXiv:2408.00118](https://arxiv.org/html/2408.00118v1), [Gemma 4 architecture breakdown](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture)

---

## 2. What Kernels Dominate Gemma's Compute?

No Gemma-specific GPU profiling paper or tech report was found (confirmed absence — see note below). However, well-established transformer compute analysis provides reliable ballparks.

### FLOP breakdown by operation (general transformer, Gemma applicable)

The dominant compute insight from scaling literature ([JAX Scaling Book](https://jax-ml.github.io/scaling-book/transformers/), [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)):

**Rule of thumb:** Total training FLOPs ≈ 6 × num_params × num_tokens. Almost all of those FLOPs live in matmuls.

| Operation | Approx. % of Training FLOPs | Notes |
|-----------|---------------------------|-------|
| MLP matmuls (up/gate/down) | ~55–65% | Largest single block; scales as O(BTD²) |
| Attention projection matmuls (Q, K, V, O) | ~15–25% | Also O(BTD²) |
| Attention score computation (QKᵀ, AV) | ~5–10% at seq 2048; up to ~20% at 8K | Scales O(BT²DH) — grows with seq length |
| RMSNorm / LayerNorm | < 1% | Memory-bandwidth bound but tiny FLOP count |
| Softmax (within attention) | < 1% | Folded into flash attention; negligible standalone |
| Rotary embeddings (RoPE) | < 1% | Elementwise; fast in practice |
| GeGLU/GELU activation | ~1–3% | Elementwise on MLP intermediate |
| Cross-entropy loss | < 1% (inference) / 1–3% (training) | Vocab projection adds back-pressure on memory |

**Key insight:** At Gemma's typical training sequences (2K–8K for Gemma 2; up to 128K for Gemma 3/4 but with local windows capped at 1024 tokens), matmul dominates overwhelmingly. The lore rule is: *attention FLOPs only equal projection FLOPs when T > 2D*, i.e., at ~10K tokens for a 5K hidden-dim model. Inside 1024-token local windows, attention is small.

**Backward pass note:** Backward pass takes ~2.5× the forward FLOPs (5 matmuls vs. 2 in forward for each linear layer). From [karpathy/llm.c A100 profiling](https://github.com/karpathy/llm.c/discussions/331), layernorm backward is a meaningful optimization target (>2× speedup possible), as is fused classifier (cross-entropy + logit projection, >2× speedup possible).

*Note: No published Gemma-specific profiling traces were found. The above is derived from general transformer analysis, not Gemma tech reports.*

---

## 3. Unusual Architectural Features Requiring New Kernels

This is the most actionable section for Noeris.

### 3.1 Sliding Window (Local) / Global Attention Interleaving

**Gemma 2:** Alternates every other layer. Local layers use a 4096-token sliding window; global layers use full 8192-token attention.

**Gemma 3:** 5:1 local-to-global ratio. Local window = 1024 tokens, global = 128K. First layer is local.

**Gemma 4 (31B):** Same 5:1 ratio. 50 sliding-window layers (window = 1024), 10 global full-attention layers (context = 256K). Global layers additionally use shared KV (unified K and V across heads) and proportional RoPE.

**Noeris gap:** Current Noeris attention covers standard causal attention. Sliding-window / local attention is a distinct kernel variant — causal masking changes shape (only attend to within-window tokens), which affects tiling strategy, SRAM allocation, and the backward pass. This is confirmed to be a meaningful optimization frontier: [vLLM PR #24390](https://github.com/vllm-project/vllm/pull/24390) optimized Triton unified attention for SWA by pruning key/value tiles outside the window.

### 3.2 Logit Softcapping (Gemma 2; returns in Gemma 4)

**Gemma 2:** `logits ← soft_cap × tanh(logits / soft_cap)` applied at two points:
- Attention logits: `soft_cap = 50.0`  
- Final output logits: `soft_cap = 30.0`

**Gemma 3:** Removed softcap, replaced by QK-norm (see 3.3).

**Gemma 4:** Softcap on final logits returns (confirmed by vLLM bug report referencing `30 × tanh(x/30)` for Gemma 4). Attention softcap status in Gemma 4 is **not confirmed from primary sources**.

This is a modification to softmax: the pre-softmax logit normalization is a `tanh` gate, not just a scale. It fuses naturally with the attention kernel or the logit computation. It is distinct from what Noeris currently does in its softmax family.

### 3.3 QK-Norm (Gemma 3+)

**Gemma 3** replaces softcapping with QK-norm: RMSNorm applied to query and key vectors before the attention dot product. This is confirmed in the Gemma 3 tech report and Google Developers blog.

**Implementation challenge:** QK-norm requires materializing Q and K, running normalization, then proceeding to the attention kernel. Naively this breaks FlashAttention's fused tiling. Published work shows fusing QK-norm into the attention kernel yields 1.7–2.5× speedup versus two kernel launches (from [TurboQuant / fused kernel analysis](https://pypi.org/project/fused-turboquant/)). Gemma 4 retains QK-norm per community inference.

**Noeris gap:** QK-norm fused into the flash-attention kernel is a non-trivial extension. It changes both the forward and backward pass of the attention kernel.

### 3.4 Grouped-Query Attention — Specific Ratios

Across the Gemma family:
- Gemma 1 2B: MQA (1 KV head)
- Gemma 1 7B: MHA (16 Q = 16 KV)
- Gemma 2 (all): GQA, ratio Q:KV = 2:1 (e.g., 32 Q heads / 16 KV heads for 27B)
- Gemma 3 27B: GQA, ratio 2:1 (32Q / 16KV)
- Gemma 4 31B: Complex — local layers use standard GQA; **global layers use unified/shared KV** (effectively MQA-style for global layers)

The Gemma 4 global layers with unified KV is a novel variant: K and V are shared across all heads in that layer, reducing KV cache footprint for the long-context layers.

Noeris already handles GQA (it's in the rotary/attention kernel space). The Gemma 4 shared-KV variant for global layers may require a specialized path.

### 3.5 GeGLU Activation Function

**All Gemma generations** use GeGLU (Gaussian Error Gated Linear Unit) in the MLP block:

```
GeGLU(x, W, V, b, c) = GELU(xW + b) ⊙ (xV + c)
```

This differs from SwiGLU (used in Llama) which replaces GELU with SiLU. GeGLU requires two gate projections plus a GELU operation that are then multiplied elementwise. In practice this triples the projection count vs. a plain MLP (up-gate-down instead of up-down).

**Noeris status:** Noeris has GELU and cross-entropy. It does not have a fused GeGLU kernel. The Liger Kernel project and Axolotl both implement fused SwiGLU/GeGLU Triton kernels, showing the optimization is tractable and meaningful.

### 3.6 Per-Layer Embeddings (Gemma 4 E2B/E4B — new)

A novel efficiency feature in Gemma 4's small on-device models: each token gets a small per-layer conditioning vector (token identity + context projection), added to hidden states via a lightweight residual block. This is a new kernel primitive — a fused embedding lookup + projection + residual add per layer.

**Noeris relevance:** This is niche (only the E2B/E4B on-device variants use it) and early-stage. Not a priority for current Noeris scope.

### 3.7 Proportional RoPE (Gemma 4)

Gemma 4 uses two RoPE configurations:
- Local attention layers: standard RoPE (base = 10K)
- Global attention layers: Proportional RoPE (p-RoPE), which scales frequencies proportionally to the extended context length, enabling 256K context

Noeris already has rotary embeddings. p-RoPE is a parametric variant (different theta schedule) that should be expressible in the existing parameterized template with configuration changes rather than a new kernel.

---

## 4. Gemma Training Recipe

### Gemma 3 (Primary Source: [arXiv:2503.19786](https://arxiv.org/abs/2503.19786))

- **Training tokens:** 2T (1B), 4T (4B), 12T (12B), 14T (27B)
- **Context length during training:** Up to 128K; local attention windows capped at 1024 tokens
- **Hardware:** TPUv4, TPUv5e, TPUv5p (512–6144 chips per run)
- **Optimizer sharding:** ZeRO-3-style optimizer state sharding
- **Post-training:** Knowledge distillation from larger models + multi-objective RL (RLHF + ground-truth outcome rewards)
- **Quantization-aware training (QAT):** int4 and FP8 variants trained with QAT
- **Batch size, LR schedule:** Not disclosed in the technical report
- **Kernel-level optimizations mentioned:** None explicitly; Google uses JAX/XLA on TPUs with their own compiler stack

### Gemma 2 (Primary Source: [arXiv:2408.00118](https://arxiv.org/html/2408.00118v1))

- **Training tokens:** 2T (2B), 8T (9B, with distillation from 27B teacher), 13T (27B)
- **Context length:** 8,192 tokens
- **Activation:** GeGLU
- **Norm:** RMSNorm pre- and post-layer

### Gemma 4 (Primary Source: [HuggingFace blog](https://huggingface.co/blog/gemma4), [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/))

- Training details not publicly disclosed as of April 2026
- Released April 2, 2026; architecture is evolution of Gemma 3

*Note: Google has not published kernel-level optimization details for any Gemma generation. Their TPU-centric XLA compilation stack means CUDA/Triton kernel choices are not in their public documentation.*

---

## 5. Public Scripts and Kernel-Level Tuning for Gemma

### Unsloth

[Unsloth](https://unsloth.ai/blog/gemma3) is the most kernel-sophisticated open-source Gemma fine-tuner:
- Custom Triton kernels for RoPE and MLP with padding-free packing (3× faster, 30% less VRAM)
- Precision strategy: matmuls in float16 with tensor cores; layernorm/other ops upcasted to float32; activations in bfloat16
- Gemma 3 training: 1.6× faster, 60% less VRAM vs. HuggingFace + FlashAttention2 baseline
- Gemma 4 training: ~1.5× faster, ~60% less VRAM
- MoE training: 12× faster with specialized Triton sparse permute kernels

### Axolotl

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) integrates:
- FlashAttention 2/3/4
- Liger Kernel (fused RMSNorm, RoPE, SwiGLU/GeGLU)
- Custom Triton kernels for LoRA MLP and attention modules
- [Bounty issue #1038](https://github.com/axolotl-ai-cloud/axolotl/issues/1038) explicitly sought optimized Triton kernels for sliding window attention (Mistral variant)

### Liger Kernel (LinkedIn)

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides production Triton implementations of:
- RMSNorm (fused forward+backward)
- RoPE
- SwiGLU / GeGLU (fused gate projection + activation + elementwise product)
- Cross-entropy (fused with vocab projection)

Claims: 20% throughput improvement multi-GPU, 60% memory reduction. GeGLU is explicitly implemented.

### NVIDIA Transformer Engine

[Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_gemma/tutorial_generation_gemma_with_te.html) supports Gemma inference via FP8, paged attention, and CUDA Graphs but does not publish Gemma-specific kernel profiling.

---

## 6. Recommendations for Noeris

Based on the above research, here are concrete kernel additions and research directions, ordered by impact:

### Recommendation 1: Fused GeGLU Kernel (HIGH PRIORITY)

**What:** A Triton kernel fusing the two gate projections, GELU activation, and elementwise multiplication in the Gemma MLP block.

**Why:** Every Gemma generation uses GeGLU. This is the most impactful missing kernel: MLP matmuls are ~55–65% of training FLOPs, and GeGLU's pointwise portion (after the matmuls) is a prime fusion target since it is register/SRAM-resident. Liger Kernel and Axolotl both have this, but Noeris has neither GeGLU nor SwiGLU. Noeris's parameterized template system is well-suited to expressing both variants under one kernel family (activation function as a template parameter).

**Implementation path:** Fuse: `gate_proj(x)` elementwise-multiply with `GELU(up_proj(x))` as a single kernel launched after the two matmuls complete. The backward kernel fuses the GELU backward + elementwise multiply backward.

### Recommendation 2: Sliding Window Attention Variant (HIGH PRIORITY)

**What:** A FlashAttention-style Triton kernel variant that enforces a local causal window (configurable `window_size`), pruning attention tiles beyond the window.

**Why:** Gemma 2, 3, and 4 all use local/global attention interleaving. At Gemma 3/4's 5:1 ratio, 83% of attention layers are local SWA. Fine-tuning systems (Axolotl bounty, vLLM PR) confirm this is a real bottleneck. The optimization is clean: skip computing tiles where all keys fall outside `[q_pos - window_size, q_pos]`, reducing FLOPs and SRAM pressure. This extends Noeris's existing causal attention kernel parametrically — `window_size=-1` recovers full causal attention.

**Implementation path:** Add `window_size` parameter to the existing causal attention kernel's tile loop; add a skip-tile predicate. The backward pass requires mirroring the mask in the recomputation step.

### Recommendation 3: Fused QK-Norm + Flash Attention (MEDIUM PRIORITY)

**What:** Integrate RMSNorm applied to Q and K vectors inside the FlashAttention forward+backward kernel, eliminating separate kernel launches for the normalization.

**Why:** Gemma 3+ uses QK-norm instead of logit softcapping. Without fusion, QK-norm requires materializing Q and K to global memory, running a separate normalization kernel, then loading them again inside flash attention — 2× the memory bandwidth for those tensors. Research shows 1.7–2.5× speedup when fused. This is directly on Noeris's critical path since Noeris already has both RMSNorm and attention kernels.

**Implementation path:** In the flash attention forward kernel, before computing `S = QKᵀ / sqrt(d)`, apply per-row RMSNorm to the Q and K tiles as they are loaded from HBM. Add the corresponding backward through the RMSNorm inside the attention backward kernel.

### Recommendation 4: Logit Softcap as Softmax Family Variant (MEDIUM PRIORITY)

**What:** A Triton kernel variant implementing `soft_cap × tanh(logits / soft_cap)` applied before the softmax (for attention) or before cross-entropy loss (for output logits).

**Why:** Gemma 2 uses this in attention and on output logits. Gemma 4 uses it on output logits (confirmed). It improves training stability by preventing logit explosion. This is a simple elementwise fused operation but it lives inside the attention kernel (for attention softcap) and inside the fused cross-entropy kernel (for output logit softcap). Noeris already has both — this is an additive configuration parameter.

**Implementation path:** Add `softcap` parameter (default disabled) to the attention kernel before the `S → P` softmax step, and to the cross-entropy kernel before the log-softmax step. When `softcap > 0`, apply the tanh transform. Backward: derivative of `soft_cap × tanh(x / soft_cap)` is `1 - tanh²(x / soft_cap)`, so the chain rule adds one multiplication in the backward pass.

### Recommendation 5: Cross-Run Learning for Local vs. Global Attention Shape Space (RESEARCH DIRECTION)

**What:** Extend Noeris's cross-run shape-indexed learning to explicitly track the local-SWA vs. global-full-attention split. For a Gemma 3 27B workload (62 layers, 10 global, 52 local), Noeris would learn separate frontier models for the two attention types, since their optimal tile sizes, block shapes, and occupancy settings differ significantly.

**Why:** SWA kernels have fundamentally different compute intensity (limited KV range) and SRAM pressure than full-context attention. Treating them as the same shape family underutilizes Noeris's learning capability. In a 62-layer Gemma 3 forward pass, running mistuned attention kernels on 52 local layers (even slightly) compounds across layers.

**Implementation path:** Add `attention_type ∈ {causal_full, causal_local_window, local_window_size=K}` to the Noeris shape index. Collect profiling data separately for each type. The frontier tracker and cost model already handle this level of parametric variation.

---

## Sources

- [Gemma 3 Technical Report — arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- [Gemma 3 Technical Report — HTML](https://arxiv.org/html/2503.19786v1)
- [Gemma 2 Technical Report — arXiv:2408.00118](https://arxiv.org/html/2408.00118v1)
- [Gemma 4 — HuggingFace Blog](https://huggingface.co/blog/gemma4)
- [Gemma 4 — Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma 4 Architecture and Memory Consumption (Kaitchup)](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture)
- [Gemma 4 MoE Architecture — MindStudio](https://www.mindstudio.ai/blog/gemma-4-mixture-of-experts-architecture)
- [Gemma Explained: Architecture Overview — Google Developers](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)
- [Gemma Explained: What's New in Gemma 3 — Google Developers](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)
- [JAX Scaling Book: Transformer Math](https://jax-ml.github.io/scaling-book/transformers/)
- [Transformer Math 101 — EleutherAI](https://blog.eleuther.ai/transformer-math/)
- [karpathy/llm.c A100 Performance Analysis](https://github.com/karpathy/llm.c/discussions/331)
- [Fine-tune Gemma 3 with Unsloth](https://unsloth.ai/blog/gemma3)
- [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/html/2410.10989v2)
- [Axolotl Optimized Triton Kernels Bounty Issue](https://github.com/axolotl-ai-cloud/axolotl/issues/1038)
- [vLLM SWA Triton Optimization PR #24390](https://github.com/vllm-project/vllm/pull/24390)
- [Gemma4 HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Anatomy of a Triton Attention Kernel](https://arxiv.org/html/2511.11581v1)
- [HuggingFace Gemma2 Blog](https://huggingface.co/blog/gemma2)
