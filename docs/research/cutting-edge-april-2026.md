# Cutting-Edge GPU Kernel Optimization: March-April 2026

*Compiled 2026-04-12. Focus: what's new, what it means for Noeris, and what to build next.*

---

## Section 1: What's New (Last 6 Weeks)

### 1.1 FlashAttention-4 (April 8, 2026)
**Paper:** [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://www.together.ai/blog/flashattention-4)  
**Code:** [Dao-AILab/flash-attention (cute/)](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)

Written in CuTeDSL (NVIDIA's Python DSL for CUTLASS), FA4 targets Hopper and Blackwell. On B200 BF16 it hits 1605 TFLOPs/s (71% utilization), 1.3x faster than cuDNN 9.13, 2.7x faster than Triton. Key innovations: fully async MMA pipelining on Blackwell, software-emulated exponential via polynomial approximation on FMA units, 2-CTA MMA mode for backward pass, and a new tile scheduler for causal mask load balancing. PyTorch's FlexAttention now has an FA4 backend with auto-generated CuTeDSL score/mask modifications, yielding 1.2-3.2x over the Triton FlexAttention path.

**Noeris relevance:** FA4 is written in CuTeDSL, not Triton. Noeris's Triton attention kernels remain the best option for the Triton ecosystem, and FA4's existence actually validates the importance of fused attention prologue work (QK-RMSNorm+RoPE) since FA4 doesn't fuse the prologue.

### 1.2 Kernel-Smith (March 28, 2026)
**Paper:** [Kernel-Smith: A Unified Recipe for Evolutionary Kernel Optimization](https://arxiv.org/abs/2603.28342)

Evolutionary kernel optimization with RL post-training. Maintains a population of executable candidates, iteratively improves them using an archive of top-performing + diverse programs with structured execution feedback. The 235B-RL variant achieves SOTA on KernelBench Triton backend, outperforming Gemini-3.0-pro and Claude-4.6-opus. Operators contributed to SGLang and LMDeploy.

**Noeris relevance:** Kernel-Smith's evolutionary loop is the closest analog to Noeris's autonomous search, but it does NOT do cross-run shape-indexed learning or persistent config databases. It optimizes per-problem, not per-(op, shape, hardware) tuple.

### 1.3 KernelFoundry (March 12, 2026)
**Paper:** [KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization](https://arxiv.org/abs/2603.12440)

MAP-Elites quality-diversity search with kernel-specific behavioral dimensions (memory access patterns, parallelism strategies). Meta-prompt evolution co-evolves prompts with kernels. Generates SYCL and CUDA kernels. 2.3x average speedup on KernelBench for SYCL.

**Noeris relevance:** MAP-Elites behavioral dimensions are interesting -- Noeris could index its config database along similar dimensions. But KernelFoundry doesn't persist learnings across runs.

### 1.4 KernelEvolve (Meta, April 2, 2026 blog; ISCA 2026)
**Blog:** [KernelEvolve: How Meta's Ranking Engineer Agent Optimizes AI Infrastructure](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/)  
**Paper:** [arXiv:2512.23236](https://arxiv.org/abs/2512.23236)

Meta's production system for heterogeneous kernel optimization (NVIDIA, AMD, MTIA, CPU). Uses Monte Carlo tree search + evolutionary strategies with selective memory inheritance. 100% pass rate on KernelBench (all 250 problems). Up to 17x over PyTorch baselines. The shared data foundation where successful optimizations become available to future sessions is the closest thing in the literature to Noeris's cross-run learning -- but it's proprietary and focused on Meta's internal fleet.

**Noeris relevance:** KernelEvolve's "shared data foundation" validates Noeris's cross-run config database concept. But KernelEvolve is not open-source and doesn't index by (shape, hardware) tuples.

### 1.5 KernelAgent (Meta/PyTorch, March 2026)
**Blog:** [KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)

Multi-agent architecture (Profiler, Judge, Analyze, Orchestrator, Optimization Manager, Benchmark agents) with NCU hardware profiling feedback (28 metrics). Concurrent optimization strategies with shared memory preventing repeated dead ends.

**Noeris relevance:** The NCU profiling feedback loop is worth adopting. Noeris currently optimizes by wall-clock time; adding hardware counter feedback (compute util, memory bandwidth, cache hits) would enable more targeted optimization.

### 1.6 K-Search (February 2026)
**Paper:** [K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model](https://arxiv.org/abs/2602.19128)  
**Code:** [github.com/caoshiyi/K-Search](https://github.com/caoshiyi/K-Search)

Co-evolving world model -- a structured search tree encoding hypotheses about bottlenecks, design alternatives, and optimization strategies. Decouples high-level algorithmic planning from low-level instantiation. 2.10x average improvement over SOTA evolutionary methods, up to 14.3x on complex MoE kernels. SOTA on GPUMode TriMul task on H100 (1028 us).

**Noeris relevance:** The "co-evolving world model" concept is directly applicable. Noeris's hypothesis engine could maintain an explicit model of *why* certain configs work for certain shapes, not just *which* configs work.

### 1.7 AutoKernel (RightNow AI, April 6, 2026)
**Blog:** [MarkTechPost coverage](https://www.marktechpost.com/2026/04/06/rightnow-ai-releases-autokernel-an-open-source-framework-that-applies-an-autonomous-agent-loop-to-gpu-kernel-optimization-for-arbitrary-pytorch-models/)  
**Code:** [github.com/RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel)

Open-source autonomous loop: profiles PyTorch models, ranks bottlenecks by Amdahl impact, iteratively refines Triton/CUDA kernels through 300-400 experiments overnight. ~90s per iteration (30s correctness, 30s bench, 30s reasoning). On H100: 5.29x RMSNorm, 2.82x softmax, 2.21x cross-entropy over eager.

**Noeris relevance:** AutoKernel is the most direct open-source competitor. Same autonomous loop concept. Key Noeris differentiators: (a) persistent cross-run config DB, (b) shape-indexed learning, (c) multiple operators optimized in concert, (d) Gemma-specific fusions (QK-RMSNorm+RoPE).

### 1.8 Unsloth MoE Triton Kernels (March 2026)
**Blog:** [Unsloth 2026 Update: Faster MoE](https://unslothai.substack.com/p/unsloth-2026-update-faster-moe)

12x faster MoE training with 35% less VRAM via custom Triton grouped-GEMM + LoRA kernels. 2x faster than Transformers v5. Supports gpt-oss, Qwen3, DeepSeek, GLM. gpt-oss-20b fine-tunes in 12.8 GB VRAM.

**Noeris relevance:** Unsloth's MoE kernels are hand-tuned for specific models. Noeris's autonomous approach could potentially match or exceed hand-tuned MoE kernels with less engineering effort, especially for new architectures where hand-tuning hasn't happened yet.

### 1.9 vLLM Model Runner V2 (March 2026)
**Source:** [vLLM GTC 2026 talk](https://x.com/vllm_project/status/2036389182579642544), [v0.19.0](https://github.com/vllm-project/vllm/releases)

GPU-native Triton kernels replacing CPU PyTorch ops. ModularKernel for MoE (mix-and-match GEMM + all-to-all). 56% throughput increase on small models. Supports FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, CuTeDSL attention backends.

**Noeris relevance:** vLLM's ModularKernel architecture means Noeris-optimized kernels could be plugged into vLLM's MoE layer directly. The 4+ separate kernel launches for Gemma 3/4 prologue (confirmed in vLLM source) remain unfused -- Noeris's QK-RMSNorm+RoPE fusion is still a real contribution.

### 1.10 CUTLASS 4.x / CuTeDSL (Ongoing, latest v4.4.2)
**Docs:** [NVIDIA CUTLASS CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)

Python-native CUDA kernel authoring with CUTLASS abstractions. JIT-compiled via MLIR. Performance on par with C++ CUTLASS. 100x faster compile times than NVCC.

**Noeris relevance:** CuTeDSL is emerging as a Triton alternative for Hopper/Blackwell. Worth watching but Noeris should stay on Triton for now -- broader ecosystem, more users, and CuTeDSL is still NVIDIA-only.

### 1.11 DeepGEMM (DeepSeek, Updated for Blackwell)
**Code:** [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)

Clean FP8 GEMM kernels with fine-grained scaling for Hopper and Blackwell. Up to 2.7x over expert-tuned libraries on Hopper, >1350 TFLOPs. Supports normal and MoE grouped GEMMs with TMA.

**Noeris relevance:** DeepGEMM is the FP8 GEMM baseline to beat. Noeris's grouped_gemm operator should benchmark against DeepGEMM directly.

---

## Section 2: Opportunities for Genuinely Novel Contributions

### 2.1 Shape-Indexed Cross-Run Config Transfer
**What:** A persistent database keyed by (operator, tensor shape, hardware) that accumulates optimal configs across runs and transfers knowledge to unseen shapes via interpolation/GP surrogate.  
**Why it's novel:** KernelEvolve has a "shared data foundation" but it's proprietary and doesn't index by shape. Kernel-Smith, AutoKernel, K-Search all start fresh each run. No published open system does shape-indexed transfer.  
**Difficulty:** 3-4 weeks (DB exists; add GP surrogate for shape interpolation).  
**Builds on:** Existing config database, bandit learner, shape buckets.

### 2.2 Co-Evolving World Model for Triton Optimization
**What:** Adapt K-Search's co-evolving world model concept to Triton autotuning -- maintain explicit hypotheses about WHY certain tile sizes / num_warps / num_stages work for certain shapes, and use those hypotheses to guide config proposals.  
**Why it's novel:** K-Search does this for CUDA kernel structure. Nobody has applied it to Triton config space where the search dimensions are well-defined (BLOCK_M/N/K, warps, stages, GROUP_SIZE).  
**Difficulty:** 4-6 weeks.  
**Builds on:** Existing hypothesis engine, LLM proposer.

### 2.3 Hardware-Counter-Guided Optimization Loop
**What:** Integrate NCU/Nsight profiling metrics (compute utilization, memory bandwidth, L2 cache hit rate, occupancy, stall breakdown) into the optimization feedback loop, not just wall-clock time.  
**Why it's novel:** KernelAgent does this for one-shot optimization. No autonomous multi-run system uses hardware counters to guide iterative config search. The counter data could feed into the GP surrogate.  
**Difficulty:** 2-3 weeks (NCU integration, metric extraction, feedback encoding).  
**Builds on:** Existing Modal GPU infrastructure, smoke tests.

### 2.4 Fused Training Prologue Kernels (Beyond Inference)
**What:** Extend QK-RMSNorm+RoPE fusion to the training backward pass. Fuse dRoPE + dRMSNorm + dQ/dK into a single kernel for Gemma 4 fine-tuning.  
**Why it's novel:** vLLM has a forward-only `enable_qk_norm_rope_fusion` pass (disabled by default due to H100 regression). All existing fusions (Liger, Unsloth, vLLM) are either inference-only or fuse individual ops. No framework fuses the full Gemma attention prologue backward pass.  
**Difficulty:** 4-6 weeks (backward pass Triton is harder, need gradient verification).  
**Builds on:** Existing QK-RMSNorm+RoPE forward kernel, verification infrastructure.

### 2.5 Autonomous FP8 Kernel Search
**What:** Add FP8 variants to Noeris's operator suite. Use the autonomous search loop to find optimal FP8 tile configs, which have a different performance landscape than BF16/FP16.  
**Why it's novel:** DeepGEMM hand-tunes FP8 configs. Nobody has applied autonomous config search to FP8 Triton kernels specifically.  
**Difficulty:** 3-4 weeks (need H100/B200 access, FP8 Triton support is maturing).  
**Builds on:** Existing grouped_gemm, matmul operators.

### 2.6 Quality-Diversity Archive for Kernel Configs
**What:** Adopt MAP-Elites from KernelFoundry but apply it to Triton config space. Index configs along behavioral dimensions: memory-bound vs compute-bound, high-occupancy vs high-ILP, small-tile vs large-tile. Maintain a diverse archive that covers the full Pareto frontier.  
**Why it's novel:** KernelFoundry uses MAP-Elites for kernel code structure. Using it for Triton config search with hardware-aware behavioral dimensions is unexplored.  
**Difficulty:** 2-3 weeks.  
**Builds on:** Existing bandit learner, config database.

### 2.7 End-to-End Gemma 4 Inference Pipeline Optimization
**What:** Instead of optimizing operators in isolation, optimize the full Gemma 4 inference pipeline: QK-RMSNorm+RoPE -> Attention -> GQA -> MoE Router -> GroupedGEMM -> GeGLU -> LayerNorm. Co-optimize configs across operators for minimum end-to-end latency.  
**Why it's novel:** All existing systems (AutoKernel, Kernel-Smith, KernelAgent) optimize operators independently. Co-optimization across a full model pipeline is an open problem.  
**Difficulty:** 6-8 weeks.  
**Builds on:** All 13 existing operators, Gemma 4 shape analysis.

---

## Section 3: "Making LLM Training Way Faster" Angle

### Gemma 4 Fine-Tuning (LoRA/QLoRA)

The Gemma 4 fine-tuning stack has known bottlenecks Noeris can target:

| Kernel | Current Baseline | Noeris Measured | Est. E2E Impact |
|--------|-----------------|-----------------|-----------------|
| QK-RMSNorm+RoPE (fwd) | vLLM separated: 4+ launches | 10-13x fusion speedup | 3-5% of total training time |
| QK-RMSNorm+RoPE (bwd) | Unfused in all frameworks | Not yet built | 3-5% additional (backward is ~2x forward) |
| MoE Router (softmax+topk) | vLLM fused_moe | SKIP_TOPK mode exists | 1-2% (router is small) |
| GroupedGEMM (MoE experts) | DeepGEMM/Triton | Shape-indexed autotuning | 5-10% (MoE GEMMs dominate in Gemma 4 MoE variants) |
| Cross-Entropy (fused) | Liger Kernel | Noeris variant exists | 2-3% |
| GeGLU activation | Separate gelu + mul | Fused variant exists | 1-2% |

**Conservative estimate:** 15-25% end-to-end training speedup for Gemma 4 LoRA fine-tuning by combining all fused kernels + autotuned configs. This is on top of what Unsloth/Liger already provide, specifically for Gemma 4's architecture (QK-norm before RoPE is Gemma-specific).

**QLoRA specifically:** The bottleneck shifts to dequantization + GEMM fusion. Noeris's grouped_gemm with INT4 dequantization fusion would be the key kernel. Estimated 10-15% additional speedup over current QLoRA implementations that don't fuse dequant.

### Gemma 4 Inference Throughput

| Optimization | Mechanism | Est. Speedup |
|---|---|---|
| Fused prologue (QK-RMSNorm+RoPE) | Eliminate 3 kernel launches | 8-12% latency reduction per layer |
| Shape-indexed GQA configs | Optimal tile sizes per head_dim=256/512 | 5-8% attention throughput |
| MoE expert GEMM autotuning | Per-shape optimal BLOCK_M/N/K | 5-10% for MoE layers |
| FP8 autotuned configs | Optimal FP8 tile sizes (different landscape than BF16) | 15-20% if FP8 viable |

**Prefill throughput:** 20-35% improvement estimated (long sequences, compute-bound -- fusion helps most).  
**Decode throughput:** 10-15% improvement estimated (memory-bound -- launch overhead reduction helps most).

### Pre-Training Workloads

For pre-training, the impact is smaller because:
1. Pre-training is dominated by large matmuls where cuBLAS/DeepGEMM is already near-optimal
2. Prologue fusion savings are a smaller fraction of total compute
3. The main opportunity is FP8 training recipes with autotuned tile sizes

**Estimated pre-training impact:** 5-10% throughput improvement, primarily from FP8 autotuning and MoE expert GEMM optimization.

---

## Section 4: Recommended Next Moves (Prioritized)

### Priority 1: Shape-Indexed GP Surrogate for Config Transfer (2-3 weeks)
**Novelty: High | Impact: High | Tractability: High**

This is Noeris's clearest differentiator. No published system transfers autotuned configs across shapes. Build a Gaussian process surrogate that predicts performance for unseen (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) configs given a shape, trained on the existing config database. Start with matmul and grouped_gemm where shape variation is highest.

*This is the paper's second headline claim after QK-RMSNorm+RoPE.*

### Priority 2: Hardware Counter Integration (2-3 weeks)
**Novelty: Medium-High | Impact: High | Tractability: High**

Add NCU metric extraction to the Modal benchmark loop. Collect compute utilization, memory bandwidth, L2 hit rate, occupancy for every config tested. Use these as features in the GP surrogate. This turns Noeris from "black-box timing optimization" into "hardware-aware optimization" -- a meaningful upgrade that also generates rich data for the paper.

### Priority 3: Fused Training Backward Pass for QK-RMSNorm+RoPE (4-6 weeks)
**Novelty: High | Impact: Medium-High | Tractability: Medium**

The forward fusion is proven (10-13x). Building the backward pass makes the kernel usable for training, not just inference. No existing framework fuses the Gemma prologue backward pass. This turns the paper from "inference optimization" into "training optimization" -- much higher impact story.

### Priority 4: FP8 Kernel Autotuning (3-4 weeks)
**Novelty: Medium | Impact: High | Tractability: Medium**

FP8 training is becoming standard on Hopper. The tile size landscape for FP8 is different from BF16 (different throughput ratios, different memory access patterns). Running the autonomous search loop on FP8 variants of matmul and grouped_gemm would produce immediately useful results that nobody else has published.

### Priority 5: End-to-End Pipeline Co-Optimization (6-8 weeks)
**Novelty: Very High | Impact: Very High | Tractability: Low**

Optimize multiple operators in a Gemma 4 layer simultaneously, considering inter-kernel data layout and launch scheduling. This is the most ambitious direction and the hardest to execute, but it would be a genuinely new contribution. Defer to after the paper submission unless the first four are done early.

---

## Key Competitive Positioning

The landscape as of April 2026:

| System | Fresh-start? | Cross-run learning? | Shape-indexed? | Hardware-aware search? | Open-source? |
|--------|-------------|---------------------|----------------|----------------------|-------------|
| AutoKernel | Yes | No | No | Via profiling | Yes |
| Kernel-Smith | Yes | No | No | No | Partial |
| KernelFoundry | Yes | No | No | MAP-Elites dimensions | No |
| KernelEvolve | No (shared DB) | Partial | No | Yes (NCU) | No |
| K-Search | Yes | No | No | No | Yes |
| KernelAgent | Yes | No | No | Yes (NCU, 28 metrics) | Yes |
| **Noeris** | **No** | **Yes** | **Yes** | **Planned** | **Yes** |

Noeris's gap: **persistent, shape-indexed, cross-run learning with hardware-counter-guided search.** Nobody else does all four. Make that the paper's thesis.

---

Sources:
- [FlashAttention-4 blog (Together AI)](https://www.together.ai/blog/flashattention-4)
- [Kernel-Smith (arXiv:2603.28342)](https://arxiv.org/abs/2603.28342)
- [KernelFoundry (arXiv:2603.12440)](https://arxiv.org/abs/2603.12440)
- [KernelEvolve (Meta Engineering)](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/)
- [KernelAgent (PyTorch Blog)](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)
- [K-Search (arXiv:2602.19128)](https://arxiv.org/abs/2602.19128)
- [AutoKernel (GitHub)](https://github.com/RightNow-AI/autokernel)
- [CUDA Agent (arXiv:2602.24286)](https://arxiv.org/abs/2602.24286)
- [KernelSkill (arXiv:2603.10085)](https://arxiv.org/abs/2603.10085)
- [Unsloth 2026 MoE Update](https://unslothai.substack.com/p/unsloth-2026-update-faster-moe)
- [vLLM v0.19.0](https://github.com/vllm-project/vllm/releases)
- [DeepGEMM (GitHub)](https://github.com/deepseek-ai/DeepGEMM)
- [CUTLASS CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
- [OpenEvolve GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery)
- [Sakana AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer/)
- [KernelBench Leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)
- [Liger Kernel (GitHub)](https://github.com/linkedin/Liger-Kernel)
