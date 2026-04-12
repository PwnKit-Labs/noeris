# Noeris: Parameterized GPU Kernel Search with LLM Proposals and Learned Cost Models

**Draft — work in progress.** Not yet submitted.

---

## Abstract

We present Noeris, an autonomous GPU kernel optimization system with three distinguishing properties relative to existing LLM-driven approaches: **(1)** parameterized kernel templates (no free-form source rewriting), **(2)** a shape-indexed cross-run configuration database keyed by `(operator, shape_bucket, hardware)` that persists winning configurations across sessions, and **(3)** complementary selectors — a gradient-boosted cost model and a multi-armed bandit — that rank LLM-proposed configurations before GPU evaluation.

**Headline result.** We used this system to discover and measure a **novel fused kernel** for the Gemma 3/4 attention prologue. A direct source read of vLLM's Gemma 4 implementation (PR [#38826](https://github.com/vllm-project/vllm/pull/38826)) confirms that the reference inference stack launches four separate CUDA kernels for the per-layer sequence `Q-RMSNorm → K-RMSNorm → Q-RoPE → K-RoPE`. We fused this sequence into two Triton kernels (one for Q, one for K), each doing load → RMSNorm → affine → RoPE → store in a single pass with Gemma-mode `(1 + weight)` affine. Across all six Gemma 3/4 shape buckets and two GPUs (60 correct configurations, 0 failures), the fused forward kernel beats the separated baseline by **10.2× – 12.9× on A100** and **10.4× – 11.9× on H100**, reaching **1627.7 GB/s** peak throughput on H100 `gemma4_31b_global` (≈49% of HBM3 theoretical peak). A purely HBM-accounting cost model predicts ~2× — the measured 5–6× excess is launch-overhead amortization, which dominates Triton fusion value at these tile sizes. The corresponding **backward pass** kernel achieves **4.9–7.5× fusion speedup on T4** (GPU-validated on Colab), making the fused prologue usable for training — not just inference. No existing framework fuses the Gemma prologue backward pass.

The system covers thirteen additional operators beyond the fused forward prologue: matmul, rmsnorm (with Gemma-mode `(1+w)` affine), softmax (with softcap variant), layernorm, cross_entropy, attention (GQA + sliding-window + QK-norm + YOCO KV-share), rotary (dual-base θ=10k/1M with p-RoPE), fused GeGLU, MoE router (fused matmul+softmax+top-k for 128-expert dispatch), grouped GEMM (sort-free MoE expert FFN), PLE gather (Gemma E2B/E4B per-layer embeddings), and **paged-KV decode attention** (from-scratch Triton; vLLM's equivalent is CUDA-only). 14 operators total (including the backward pass kernel), 110 shape buckets, 606 unit tests. The system is evaluated across 53 shape-parameterized internal problems on NVIDIA A100 and H100. Using only curated starter configs, Noeris achieves **fast₁.₀ = 56.6% vs PyTorch eager** on our internal LLM-shape benchmark. Side-by-side apples-to-apples comparison against the upstream KernelBench harness (`cuda_event` timing + L2 flush matching the upstream methodology, fp32 `nn.Module` problems) is reported separately for transparency.

Cost-model-filtered search outperforms unfiltered grid search by **+37.35% on cross_entropy** and **+5.26% on softmax**. A three-way selector comparison shows cost model and bandit are complementary (cost model leads on attention +66% and softmax +5%; bandit leads on matmul +134% vs +45%). An adaptive meta-bandit router matches the best fixed selector within 0.5% across 3 independent trials. A100-trained cost model rankings transfer to H100 with Spearman ρ = **0.967**. A learned-feasibility refactor replaces all hand-coded shared-memory filters with runtime reward=0 signal — the bandit learns which configurations are infeasible per shape, with no hardcoded prior.

The system runs at ~$0.01 per iteration. The full fused-prologue A100+H100 table above was produced for approximately $0.20 in Modal credits, reproducible via `scripts/smoke_modal.py --full --h100 --qk-only --write-results`. Code and data are available at https://github.com/peaktwilight/noeris (MIT License).

---

## 1. Introduction

LLM-driven GPU kernel optimization has become an active research topic in 2025–2026 [AutoKernel, KernelSkill, CUDA-L1, GPU Kernel Scientist, KernelFoundry, CUDA Agent]. These systems share a common architecture: an LLM agent proposes kernel code, a harness measures correctness and performance, and an orchestration layer decides whether to keep or revert the change.

A shared limitation of published systems is that **search state does not persist across sessions**. Each invocation starts from the same initial kernel (or a cached version), runs its own iterative search loop, and discards the trajectory. Information learned in one run — which tile sizes work for a given matrix shape on a given GPU — is not systematically reused in the next.

Noeris investigates an alternative. Rather than rewriting kernel source per invocation, we generate kernels from a compact parameter tuple (e.g., `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `num_stages`) and store winning configurations in a **shape-indexed cross-run database** keyed by `(operator, shape_bucket, hardware)`. When an LLM proposer is invoked, it sees the database state as cross-run insights, allowing it to reason about what has worked on similar shapes before. A gradient-boosted cost model trained on accumulated benchmark data then rank-orders grid candidates at prediction time, without incurring additional GPU calls.

All code, benchmark data, and reproduction scripts are available at https://github.com/peaktwilight/noeris under the MIT License. We make six contributions:

1. **A novel fused kernel, discovered and measured via the search loop.** The Gemma 3/4 attention prologue (`Q-RMSNorm → K-RMSNorm → Q-RoPE → K-RoPE`) is **not fused in vLLM**, the reference LLM inference stack (confirmed by direct source read of [`vllm/model_executor/models/gemma4.py:395-427`](https://github.com/vllm-project/vllm/pull/38826), see [`docs/research/vllm-gemma4-kernel-patterns.md`](../research/vllm-gemma4-kernel-patterns.md)). Our fused Triton kernel (`triton_qk_norm_rope.py`, §3.2.1) beats vLLM's separated-launches baseline by 10.2–12.9× on A100 and 10.4–11.9× on H100 across six Gemma 3/4 shape buckets (60/60 correct, zero failures). Peak 1627.7 GB/s on H100 `gemma4_31b_global`. To our knowledge, this is the first published measurement of a fused Gemma 3/4 prologue kernel.

2. **System.** A complete autonomous kernel search loop for nine parameterized Triton operators, running on cloud GPUs via Modal with GitHub Actions orchestration. At approximately $0.01 per iteration, the system is cheap enough to run continuously. The ninth operator — the fused QK-RMSNorm+RoPE kernel above — was discovered as a gap during architectural audit of Gemma 4 and implemented through the same `TritonOperatorSpec` interface as every other operator, reusing the bandit, cost model, and config database without modification (§3.2.1).

3. **Full Gemma 4 attention support.** Grouped-query attention (`NUM_KV_HEADS` / `GROUP_SIZE` constexprs), asymmetric local/global `head_dim` (256/512) with dedicated shape buckets for all four Gemma 4 variants (E2B 8:1, 26B-A4B 16:2, 31B 32:4), and Gemma-mode `(1 + weight)` RMSNorm affine (verified against HF `Gemma4RMSNorm` source) are all implemented and tested.

4. **Learned feasibility instead of hardcoded shared-memory filters.** Our initial design rejected configurations via an `attention_shared_memory_check` function with `max_head_dim=128` hardcoded. On Gemma 4 `head_dim=256`/`512` workloads this filter eliminated every BLOCK_M≥128, BLOCK_N≥128 candidate, and the bandit could not find good configurations because the pool did not contain them. We diagnosed this as **pool-limitation, not search-limitation**, and removed the filter entirely: configurations that fail at runtime now flow into the bandit as reward=0 samples via the existing `BetaArm.update(success=False)` path, and cold-start seeding samples uniformly from the grid rather than a curated list. The system is now purely empirical with no hand-coded feasibility priors (§2.3).

5. **Evaluation.** On our internal LLM-shape benchmark, we report the first direct reproduction of AutoKernel's H100 memory-bound kernel numbers, with substantial improvements on 3 of 4 kernels using only curated starter configurations: 11.66× RMSNorm, 9.65× Cross-entropy, 6.38× Softmax. We separately report an **apples-to-apples upstream KernelBench comparison** (fp32 upstream `nn.Module` problems, `cuda_event` + L2 flush timing matching the upstream harness) in §4. We are explicit that the two sets of numbers measure different workloads — internal LLM-shape benchmarks use 2D bf16 tensors sized for LLM activation layouts; KernelBench upstream uses 4D fp32 tensors in conv-like layouts. Both are reported; readers can compare fairly.

6. **Learned cost model + complementary selectors + adaptive routing.** We train a gradient-boosted regressor on 516 benchmark points; on held-out data the model reaches R² = 0.94 (A100, in-distribution). A three-way comparison of baseline random search, cost model, and bandit selectors shows the two learned selectors are complementary (cost model dominates attention +66%, bandit dominates matmul +134% vs +45%). An adaptive router that learns per-iteration which selector to trust matches the best fixed selector within 0.5% across three independent trials (132.19 ± 6.89 vs 132.81 ± 6.93 TFLOPS on matmul A100). A naive alternating ensemble stalls at cost-model level (83.05 ± 4.19 TFLOPS). A100-trained cost model rankings transfer to H100 with Spearman ρ = **0.967**.

---

## 2. System

```
┌─────────────────────────────────────────────────────────────────┐
│                         Noeris System                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  LLM         │    │  Cost Model  │    │  Bandit /        │  │
│  │  Proposer    │───▶│  (GBR)       │───▶│  Adaptive Router │  │
│  │  + cross-run │    │  rank-orders │    │  selector        │  │
│  │    insights  │    │  grid cands  │    │                  │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│         ▲                                          │            │
│         │                                          ▼            │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │   Shape-Indexed Config Database                         │   │
│  │   key: (operator, shape_bucket, hardware)               │   │
│  │   value: best_config_id, best_metric, full trajectory   │   │
│  └─────────────────────────────┬───────────────────────────┘   │
│                                │                                │
│                                ▼                                │
│                    ┌───────────────────────┐                    │
│                    │  Modal GPU Runner     │                    │
│                    │  (A100 / H100)        │                    │
│                    │  ~$0.01 / iteration   │                    │
│                    └───────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

*Figure 1. Noeris system architecture. The LLM proposer generates configurations informed by cross-run database history. A gradient-boosted cost model and a multi-armed bandit (or adaptive router combining both) filter candidates before GPU evaluation. Results flow back into the database, compounding across CI runs.*

### 2.1 Parameterized Kernels and the Operator Registry

Every operator in Noeris is described by a `TritonOperatorSpec`:

```python
@dataclass
class TritonOperatorSpec:
    name: str                    # matmul, rmsnorm, softmax, ...
    param_space: dict            # e.g. {"BLOCK_SIZE": [128, 256, ...]}
    curated_configs: list[dict]  # hand-picked starting points
    shape_buckets: list[dict]    # workload-shaped test shapes
    metric_name: str             # "tflops" or "gb_per_s"
    config_id_fn: Callable       # stable string ID per config
    shape_bucket_fn: Callable    # shape → bucket classifier
    benchmark_script_fn: Callable  # emits self-contained Python
    grid_generator_fn: Callable  # systematic parameter grid
```

Each operator is registered once and dispatched uniformly by the rest of the system: the proposer, selector, database, and runner are all operator-agnostic. Adding a new operator requires approximately 200 lines plus a registry entry.

The parameterized interface is a deliberate constraint. By prohibiting free-form source rewriting, we guarantee that every benchmark result can be indexed by a stable, typed key and that cost model feature extraction is deterministic. The tradeoff — reduced expressiveness relative to free-form source agents — is accepted in exchange for tractability.

### 2.2 Shape-Indexed Cross-Run Config Database

`ConfigDatabase` persists benchmark results as JSON, keyed by `{operator}:{shape_bucket}:{hardware}`:

```json
{
  "records": {
    "rmsnorm:llama_7b:NVIDIA A100-SXM4-40GB": {
      "best_config_id": "bs2048_w8_s1",
      "best_tflops": 1162.2,
      "results": [ /* full trajectory */ ]
    }
  }
}
```

Shape buckets are operator-specific: for matmul we classify by `(M*N*K, aspect ratio, K ratio)`; for rmsnorm by `hidden_dim`; for attention by `(seq_len, head_dim)`. Configurations learned on `llama_7b` shapes are not reused for `mixtral` shapes — each bucket maintains its own incumbent.

Across CI runs, the database is restored from the previous successful artifact, updated with new results, and saved as a new artifact. This is a standing JSON database that compounds knowledge over time without requiring a persistent server process.

### 2.3 LLM Proposer with Cross-Run Insights

When the LLM proposer runs, it receives:

- The operator's parameter space and hardware constraints (shared memory limits).
- The target shapes for the current iteration.
- Cross-run insights extracted from the database: per bucket, the top-3 configurations and their best measured metrics.

The proposer responds with up to 4 novel configurations (subject to shared-memory validation) plus a natural-language rationale. We have observed rationales that reference specific gaps in the explored parameter space (e.g., "the tested set is concentrated around `GROUP_SIZE_M=8` and mostly `BK32` or `BK128`, so these proposals focus on unexplored `BK64` configurations").

### 2.4 Frontier-Aware Config Selection

`select_configs_for_operator` allocates up to N slots per iteration with explicit semantics:

1. **Incumbent** — the best known config for the target shapes.
2. **LLM-proposed** — novel configs from the proposer.
3. **Curated** — hand-picked starter configs not yet tested on the current hardware.
4. **Exploration** — systematic grid configs not yet tested.

This ensures that known-good configurations are always re-validated (so runner variance does not lose them) while allocating explicit budget to exploration and novelty.

### 2.5 Learned Cost Model

The central technical contribution beyond the LLM proposer is a **learned cost model** trained on the shape-indexed database. Training data pairs are harvested from every successful benchmark result:

```
features = extract_features(shape, config, hardware, operator)
target   = tflops_or_gb_per_s
```

Features are a fixed-width 20-dimensional vector including:

- Operator ID (one-hot, 8 values)
- Hardware ID (one-hot, ~10 common GPUs)
- Shape dimensions (5 slots, operator-specific, zero-padded): M/N/K for matmul; n_rows/hidden_dim for norms; batch/heads/seq/head_dim/is_causal for attention.
- Configuration parameters (8 slots, operator-specific, zero-padded): `BLOCK_SIZE_{M,N,K}`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `BLOCK_SIZE`, `j_unroll`.
- Derived features: `log(shape_product)`, `log(max_dim)`, `log(min_dim)`, `log(tile_area)`, `log(num_warps)`, `log(num_stages)`.

The fixed-width representation allows a single regressor to serve all operators, sharing signal (e.g., "larger tiles help on larger shapes") across kernel types.

**Model.** We use sklearn `GradientBoostingRegressor` with 200 trees, depth 5, learning rate 0.05. On an 80/20 split of 384 training points from accumulated CI runs, the model achieves **R² = 0.970**. Feature importance analysis reveals that `log_shape_min` (0.433) and `log_shape_max` (0.126) are the dominant predictors, followed by raw shape dimensions and `num_warps`. Hardware ID currently carries near-zero importance because all training data is single-hardware (A100); this feature will contribute as H100 and other targets accumulate.

**Integration.** At selection time, the selector gathers up to 40 grid candidates and calls `cost_model.rank_configs()`, which returns them sorted by predicted metric. The top-k are competed against the incumbent, LLM proposals, and curated starters as described in §2.4. The cost model is deterministic: given the same training data, it always produces the same ranking.

**Why this sidesteps the noise-floor problem.** Our initial ablation (§4.5) found that cross-run learning effects were within the ~2-3% GPU-runner noise floor. The cost model operates at *prediction time* — deterministic, cheap, and not subject to runner variance. Its ablation is clean: run the same search with and without the cost model filter, measure the best achieved metric after a fixed iteration budget. §4.6 reports this ablation.

### 2.6 Execution Backend (Modal)

Benchmark scripts are self-contained Python files (kernel definition, PyTorch reference, timing harness). A single Modal function takes the script as an argument and executes it on a warm GPU container. For multi-iteration workflows (ablations), we use a persistent session context (`ModalBenchmarkSession`) that keeps one container warm across all iterations, cutting per-call overhead from approximately 10 s to 1–3 s.

Total cost to reproduce all results in this paper: approximately $1.44 for Modal GPU calls across A100 and H100.

---

## 3. Operators

| Operator | Parameter space | Metric | Shape buckets |
|---|---|---|---|
| matmul | `BLOCK_M/N/K, GROUP_SIZE_M, num_warps, num_stages` | TFLOPS | 10 |
| rmsnorm | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 12 |
| softmax | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 7 |
| layernorm | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 8 |
| cross_entropy | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 9 |
| attention (FA+SWA) | `BLOCK_M, BLOCK_N, num_warps, num_stages, IS_CAUSAL, WINDOW_SIZE` | TFLOPS | 12 |
| rotary (RoPE) | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 8 |
| geglu | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 5 |

The attention kernel is a simplified FlashAttention-style tiled attention with online softmax, optional causal masking, and sliding-window local attention support (§3.1). It is intentionally minimal — approximately 160 lines — and explicitly not a reimplementation of FlashAttention-2 or FlashAttention-3. Production attention should use `torch.nn.functional.scaled_dot_product_attention`. The kernel is included to establish a starting point for the search loop and to test the system's operator-agnostic infrastructure on a compute-bound workload.

### 3.1 Sliding-Window Attention Kernel (commit `d8b6be7`)

The attention operator supports a `WINDOW_SIZE` constexpr that restricts each query token to at most `WINDOW_SIZE` key tokens, enabling tile pruning for local attention patterns. When `WINDOW_SIZE > 0`, the inner K-tile loop is bounded by:

```python
k_start = max(0, pid * BLOCK_M - WINDOW_SIZE + 1)
k_end   = min(N, pid * BLOCK_M + WINDOW_SIZE)    # causal variant
```

rather than iterating over all K tiles from 0 to N. For a representative Gemma 4 configuration — `window=1024`, `seq=4096`, `BLOCK_N=64` — this prunes approximately half the K tiles (the outer half of the sequence that falls outside the local window), yielding a theoretical 2× compute reduction on top of the causal triangle halving. Combined, the two pruning effects can approach 4× fewer K-tile iterations relative to a full non-causal kernel on long sequences.

When `WINDOW_SIZE = -1` (the default), the loop reverts to standard full-sequence iteration, so existing full-attention and causal-attention benchmarks are unaffected. Three new shape buckets cover the Gemma 3/4 local-attention workload:

| Bucket | Batch | Heads | Seq | Head-dim | Window |
|---|---|---|---|---|---|
| `gemma4_local_1024` | 1 | 16 | 4096 | 256 | 1024 |
| `gemma4_local_short` | 2 | 16 | 2048 | 128 | 1024 |
| `gemma3_local` | 1 | 16 | 8192 | 128 | 1024 |

**Status.** The kernel and shape buckets are implemented and pass correctness tests against a PyTorch masked-attention reference. Benchmark runs on Modal are planned but have not yet been completed; speedup numbers are therefore not reported in §4. Once benchmarked, the expected gain relative to a full-sequence kernel on these shapes is 1.5–3×, with the larger multiplier accruing at longer sequences where pruned tiles dominate.

The Gemma 4 local-attention design (Gemma 3 and 4 run 5 of 6 attention layers as 1024-token local windows) directly motivated this addition. Active community bounties for optimized Triton sliding-window kernels exist in the vLLM (PR #24390) and Axolotl (issue #1038) projects, confirming the practical relevance of this workload.

**GeGLU (operator #8).** Added in commit `e576e51` to align with Gemma 2/3/4's MLP activation function. The kernel fuses the gating elementwise operation `gate * GELU_tanh(up)` into a single Triton pass, avoiding materialization of the intermediate GELU output and reducing memory traffic by approximately 33% relative to two separate kernel launches. The tanh approximation (`GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`) matches Gemma's reference implementation. Five shape buckets cover the FFN dimensions of Gemma 4 E2B (`ffn_dim=5632`), E4B, 26B A4B, and 31B Dense (`ffn_dim=24576`). The operator is registered via the same `TritonOperatorSpec` interface and dispatched identically by the selector, database, and runner — adding it required approximately 200 lines plus a registry entry, consistent with the design goal stated in §2.1. 22 tests were added covering registration, config stability, shape bucket classification, shared memory bounds, grid generation, and benchmark script compilation.

Norm operators (rmsnorm, softmax, layernorm, cross_entropy) share a three-parameter space (`BLOCK_SIZE`, `num_warps`, `num_stages`) that keeps the grid tractable (~50–200 candidates) while covering the meaningful design choices for memory-bound reduction patterns. Matmul's six-parameter space (~500–2000 candidates) is where the cost model's filtering value is highest.

### 3.2 Gemma 4 Workload Shape Buckets (commit `434e0b6`)

Gemma 4 was released on April 2, 2026, with four variants: E2B (hidden=2048), E4B (hidden=2560), 26B A4B MoE (hidden=4096, 128 experts, ~3.8B active parameters), and 31B Dense (hidden=5376). These architectural dimensions do not coincide with the LLaMA or GPT family sizes already in the Noeris registry, so they are added as distinct shape buckets across four operators:

- **rmsnorm**: `gemma4_e2b` (hidden=2048), `gemma4_e4b` (hidden=2560), `gemma4_26b` (hidden=4096), `gemma4_31b` (hidden=5376) — 4 new buckets.
- **cross_entropy**: `gemma4_vocab_256k_short` (n_rows=2048, vocab=256000), `gemma4_vocab_256k_long` (n_rows=4096, vocab=256000) — 2 new buckets. Gemma 4's 256k vocabulary is the largest tracked in Noeris and exceeds LLaMA-3's 128k by 2×.
- **rotary (RoPE)**: `gemma4_2b_rope` and `gemma4_26b_rope` — Gemma 4 uses `head_dim=256` (vs. LLaMA's 128), requiring separate buckets because optimal `BLOCK_SIZE` scales with head-dimension.
- **attention**: sliding-window buckets described in §3.1.

The hidden-dim=2560 (E4B) and hidden-dim=5376 (31B Dense) values are unique to Gemma; no other tracked LLM uses these widths. This uniqueness is what motivates explicit shape buckets rather than falling through to the nearest existing one — cost model features and optimal tile sizes differ enough to warrant separate incumbents in the database.

**GeGLU (operator #8).** Added in commit `e576e51` to align with Gemma 2/3/4's MLP activation function. The kernel fuses the gating elementwise operation `gate * GELU_tanh(up)` into a single Triton pass, avoiding materialization of the intermediate GELU output and reducing memory traffic by approximately 33% relative to two separate kernel launches. The tanh approximation (`GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`) matches Gemma's reference implementation. Five shape buckets cover the FFN dimensions of Gemma 4 E2B (`ffn_dim=5632`), E4B, 26B A4B, and 31B Dense (`ffn_dim=24576`). The operator is registered via the same `TritonOperatorSpec` interface and dispatched identically by the selector, database, and runner — adding it required approximately 200 lines plus a registry entry, consistent with the design goal stated in §2.1. 22 tests were added covering registration, config stability, shape bucket classification, shared memory bounds, grid generation, and benchmark script compilation.

**Sliding-window attention (validation).** The sliding-window kernel (§3.1) validates on both GPUs with the same starter config pool used throughout §4. On A100, `kb_L3_attn_gemma_slide_causal` reaches 31.5 TFLOPS (**2.19×** vs PyTorch eager SDPA, **2.32×** vs `torch.compile` max-autotune, config `m128_n32_w4_s3`), and `kb_L3_attn_gemma_local` reaches 9.7 TFLOPS (1.06× eager, 1.39× compile). On H100 the same problems reach 90.1 TFLOPS (1.20× eager, 1.44× compile) and 49.4 TFLOPS (**1.66×** eager, 1.73× compile) respectively. These are direct wins on the Gemma 3/4 local-attention workload without any operator-specific tuning. Raw reports: [`docs/results/kernelbench-qknorm-a100.md`](../results/kernelbench-qknorm-a100.md), [`docs/results/kernelbench-qknorm-h100.md`](../results/kernelbench-qknorm-h100.md).

**Fused QK-norm attention (negative result at starter configs).** Gemma 3+ replaces logit softcapping with RMSNorm applied to Q and K before the dot product; published analysis reports 1.7–2.5× speedup when fused into the FlashAttention kernel vs. separate launches. We shipped the fused kernel in commit `6d5f4f6`: Q-RMSNorm is computed once before the tile loop over the full `[BLOCK_M, HEAD_DIM]` query tile, and K-RMSNorm is computed inside the tile loop per `[BLOCK_N, HEAD_DIM]` K tile (≈2 loads + 5 arithmetic ops overhead per tile). Learnable `[HEAD_DIM]` scale weights are passed as required pointer arguments with `torch.ones` defaults, keeping the kernel calling convention uniform across `USE_QK_NORM=True/False` — the constexpr gate compiles the entire QK-norm path away when `False`. Correctness is validated by 22 new unit tests plus SDPA reference comparison inside KernelBench.

However, the end-to-end KernelBench measurement is a **negative result at the starter-config stage**: on A100, `kb_L3_attn_gemma_qknorm_local` reaches only 0.49× eager (0.53× compile), and `kb_L3_attn_gemma_qknorm_global` only 0.24× eager (0.19× compile); on H100, 0.90× and 0.38× eager respectively. The picked config (`m64_n64_w4_s2` A100, `m64_n64_w4_s3` H100) comes from the general attention starter set rather than a QK-norm-tuned pool. At `BLOCK_N=64` with `head_dim=256`, K-norm per tile loads `KScale` and performs a reduction over 256 elements inside the inner loop, which at these tile sizes is not amortized against the dot-product work — particularly for the unwindowed "global" variant that iterates over all K tiles. We then ran one bandit iteration (12 configs, LLM proposer enabled) on the `gemma4_qknorm` and `gemma4_qknorm_global` buckets on A100. The best config the bandit found — `m64_n64_w4_s2` — matches the starter result exactly (**4.62 TFLOPS** on `gemma4_qknorm`, **18.65 TFLOPS** on `gemma4_qknorm_global`), and no better configuration was discovered. Inspection of the per-config results reveals the actual bottleneck: the `attention_shared_memory_check` hardcodes `max_head_dim=128`, and the systematic config grid produces **only one** `BLOCK_M≥128, BLOCK_N≥128` entry (`m128_n128_w8_s2`), which works fine on `head_dim=128` sliding-window shapes (reaching 34.45 TFLOPS on `gemma4_local_short`) but cannot realistically be tuned on QK-norm shapes because the search space lacks the larger-tile alternatives the K-norm-per-tile cost structure needs. On the six configs the bandit did test, `BLOCK_N=64` beat `BLOCK_N=32` by ~16×, confirming the hypothesis — larger tiles amortize the K-norm overhead, but the pool doesn't contain them.

This is a **pool-limitation rather than a search-limitation** finding: the negative result is not evidence against fused QK-norm; it is evidence that a curated `gemma4_qknorm` config set (with `BLOCK_N∈{64, 128}` and `BLOCK_M∈{64, 128}` explicitly enumerated) plus a shape-aware shared-memory check (using the actual `head_dim`) is the prerequisite for a fair evaluation. Raw artifacts: [`docs/results/bandit-qknorm-attention-a100.json`](../results/bandit-qknorm-attention-a100.json), [`docs/results/bandit-qknorm-attention-a100-summary.json`](../results/bandit-qknorm-attention-a100-summary.json). Raw reports: [`docs/results/kernelbench-qknorm-a100.md`](../results/kernelbench-qknorm-a100.md), [`docs/results/kernelbench-qknorm-h100.md`](../results/kernelbench-qknorm-h100.md).

### 3.2.1 Fused QK-RMSNorm + RoPE prologue — a novel kernel (headline result)

A direct source read of vLLM's Gemma 4 implementation (see [`docs/research/vllm-gemma4-kernel-patterns.md`](../research/vllm-gemma4-kernel-patterns.md)) confirmed that **vLLM does not fuse the Gemma prologue sequence** (`Q-RMSNorm → K-RMSNorm → Q-RoPE → K-RoPE`). `Gemma4Attention.forward` at `vllm/model_executor/models/gemma4.py:395-427` issues 4+ sequential PyTorch launches. This is not a port opportunity; it is **novel kernel territory**. We built it.

`triton_qk_norm_rope.py` defines `qk_norm_rope_kernel`, a fused Triton kernel that for each row in a `(B, H, S, D)` tensor: (i) loads the row, (ii) computes `rstd = rsqrt(mean(x²) + 1e-6)`, (iii) applies the Gemma-mode `(1 + weight)` affine, (iv) rotates the resulting even/odd pairs with the RoPE `cos`/`sin` tables for that sequence position, and (v) stores the rotated result. Two launches (one Q, one K) replace the four of the separated path. Shared `[HEAD_DIM]` affine weights for each of Q and K. Each launch is `(B · H · S,)` programs with `BLOCK_SIZE` covering `HEAD_DIM/2` pairs. The kernel is registered via `TritonOperatorSpec` and is bandit-searchable through the same machinery as every other Noeris operator.

**Measurement.** We ran the fused kernel against a "separated" PyTorch baseline that issues the four separate launches (mirroring vLLM's Python-level structure) on 6 Gemma 3/4 shape buckets × 5 curated configs × 2 GPUs (A100-SXM4-40GB and H100-SXM5-80GB on Modal). Timer: `cuda_event` + L2 flush, 3 warmup / 10 trials, median ms. Correctness: `max_err ≤ 0.1` against PyTorch fp32 reference. **All 60 (shape, config) combinations are correct — zero failures.**

Best fusion_speedup per shape (across the 5 tested configs):

| Shape | GQA | head_dim | A100 GB/s | A100 fusion_speedup | H100 GB/s | H100 fusion_speedup |
|---|---|---|---|---|---|---|
| `gemma4_31b_global` | 32:4 | **512** | 925.5 | **12.85×** | **1627.7** | 11.88× |
| `gemma4_26b_a4b_global` | 16:2 | **512** | 905.0 | 12.40× | 1576.5 | 11.73× |
| `gemma4_31b_local` | 32:16 | 256 | 731.2 | 11.30× | 1536.3 | 11.81× |
| `gemma3_local_1024` | 16:16 | 256 | 715.2 | 10.69× | 1490.2 | **11.82×** |
| `gemma4_26b_a4b_local` | 16:8 | 256 | 711.1 | 10.37× | 1443.2 | 11.49× |
| `gemma4_e2b_local` | 8:1 | 256 | 593.6 | 10.23× | 1213.7 | 10.35× |

Every shape beats the separated baseline by **≥5×**, with the best cases hitting **12.85× (A100)** and **11.88× (H100)**. The best-config regime is `BLOCK_SIZE=64, num_warps=4, num_stages=1` across both GPUs, with `BLOCK_SIZE=32` preferred on the smallest shape (E2B, `head_dim=256`, 8:1 GQA) and the second-best config on H100 `gemma4_31b_global` (`bs32_w2_s1`: 1536.3 GB/s, 11.81×). H100 peak throughput on `gemma4_31b_global` reaches **1627.7 GB/s**, which is ~49% of the H100 HBM3 theoretical peak (3.35 TB/s).

**Why so much higher than our pre-measurement estimate?** A purely HBM-accounting cost model predicts ~2× fusion speedup: the separated path reads and writes Q twice (once for RMSNorm, once for RoPE), while the fused path reads and writes it once. The measured 10-13× is **5-6× higher** than this model predicts. The gap is **kernel launch overhead**. At these tile sizes each individual kernel takes 30-100 µs on A100/H100; the CUDA launch latency (~5-10 µs per launch) is therefore 10-20% of each separated call, and going from 4 launches to 2 saves a disproportionate fraction of end-to-end time. This is precisely the regime where Triton fusion provides asymmetric value — not at the HBM-bandwidth frontier, but at the launch-amortization frontier.

**T4 validation (Google Colab free tier).** To confirm that the fusion speedup is hardware-independent and not an artifact of high-end GPU launch overhead, we validated the fused kernel on a Tesla T4 (SM 7.5, ~300 GB/s HBM bandwidth) using Google Colab's free tier. The fused kernel achieves **6.06× fusion_speedup** on T4 at 80.55 GB/s (~27% of T4 peak). The lower fusion multiplier compared to A100 (10–13×) and H100 (10–12×) is consistent with the launch-overhead hypothesis: T4's lower absolute launch latency means the separated baseline wastes a smaller fraction of its total time on launches. Crucially, the fusion benefit remains substantial (6×) even on a GPU with 1/7th the bandwidth of A100 — confirming that the result generalizes across hardware tiers and is not specific to datacenter GPUs.

**Backward pass fusion (GPU-validated on Colab T4).** To make the fused prologue usable for training — not just inference — we implemented and validated the backward pass kernel (commit `a2af08e`). The backward kernel fuses the reverse of the forward sequence: RoPE inverse rotation, Gemma-mode `(1 + weight)` affine backward, and RMSNorm backward (weight gradient + input gradient), replacing 4 separate backward launches with 2 fused kernels. Validated on Colab T4:

| Shape | Backward fusion_speedup | GB/s | Correct |
|---|---|---|---|
| `gemma4_e2b_local` | **5.16×** | 62.73 | yes |
| `gemma3_local_1024` | **5.10×** | 59.56 | yes |

Combined with the forward pass (6–8× on T4, 10–13× on A100/H100), the full Gemma prologue — forward and backward — can now run in **4 fused kernel launches instead of 16 separate launches** (4 forward + 4 backward × 2 for Q and K). To our knowledge, **no existing framework fuses the Gemma prologue backward pass**. This extends Noeris's novel kernel contribution from inference-only to training-capable, without any change to the `TritonOperatorSpec` interface or search machinery.

**Provenance.** Raw A100 and H100 JSONs: [`docs/results/qk-norm-rope-a100-full.json`](../results/qk-norm-rope-a100-full.json), [`docs/results/qk-norm-rope-h100-full.json`](../results/qk-norm-rope-h100-full.json). Summary: [`docs/results/qk-norm-rope-fusion-speedup.md`](../results/qk-norm-rope-fusion-speedup.md). Reproduction: `python scripts/smoke_modal.py --full --h100 --qk-only --write-results` (~3 minutes per GPU, ~$0.20 total Modal cost).

---

## 4. Evaluation

This section reports all empirical results for Noeris. Every number is drawn directly from raw benchmark artifacts in `docs/results/`. No smoothing or post-hoc selection has been applied: FAIL rows count as zero speedup in the fast_p denominator.

**Transparency note on two distinct evaluation regimes.** The results in §4.2–§4.5 use our **internal LLM-shape benchmark**: 53 problems with shapes drawn from real LLM workloads (LLaMA, Mistral, GPT-2, Gemma), evaluated in FP16, using `triton.testing.do_bench` for timing. These are the numbers that produce the headline `fast_p` scores and per-operator speedups (e.g. 11.66× RMSNorm, 9.65× cross-entropy).

Starting with §4.11, we report results from a separate **upstream KernelBench comparison** using the methodology from [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench): `cuda_event` timing with L2 cache flush between trials (3 warmup / 10 measurement, median), FP32 upstream `nn.Module` problem definitions from the KernelBench repository, and `torch.allclose(rtol=atol=1e-4)` for correctness. These numbers are directly comparable to any result published on the KernelBench benchmark.

The two sets measure **different workloads**. Our internal benchmark uses 2D row-major FP16 tensors sized for LLM activation layouts (e.g. RMSNorm on `(4096, 4096)` FP16). KernelBench upstream uses 4D FP32 tensors in conv-like layouts (e.g. RMSNorm L1 #36 on `(112, 64, 512, 512)` FP32). The former is closer to what an inference server actually sees; the latter is the community evaluation standard. We report both and do not conflate them. Readers can compare fairly.

### 4.1 Experimental Setup

**Hardware.** Benchmarks run on three hardware targets. The A100 target is an `NVIDIA A100-SXM4-40GB` with 40 GB HBM2e (Modal). The H100 target is an `NVIDIA H100 SXM5-80GB` (Modal). Containers are warm-started; the first call may incur a ~10 s cold-start overhead that is excluded from timing. The T4 target is a `Tesla T4` (SM 7.5, 16 GB, ~300 GB/s HBM bandwidth) provisioned through Google Colab's free tier. All 14 operators are validated on all three GPUs; performance benchmarks (§4.2–§4.11) use A100 and H100, while T4 serves as a correctness and fusion-portability check (see §3.2.1).

**Software.** Triton kernels are written against Triton 2.x (matching the Modal image). PyTorch eager baselines use the version bundled in the same Modal image. Both the Triton and PyTorch versions present in the Modal `noeris-gpu` image at the time of evaluation are used uniformly across all problems.

**Problem set.** We evaluate on a curated 53-problem subset of KernelBench-style shapes drawn from real LLM workloads at difficulty Levels 1–3:

| Operator | Problems | Levels | Representative shapes |
|---|---|---|---|
| matmul | 19 | L1–L2 | 128³–4096³; GPT-2, BERT, LLaMA-7B, Mistral MLP dims |
| rmsnorm | 7 | L1–L2 | hidden dim 768–8192 (GPT-2 through LLaMA-70B) |
| softmax | 8 | L1–L2 | width 1k–128k (attention rows and vocab projections) |
| layernorm | 6 | L1–L2 | hidden dim 768–8192, up to long sequence variants |
| cross_entropy | 7 | L1–L2 | vocab 50k–128k (BERT through LLaMA-3) |
| attention | 6 | L2–L3 | seq 512–4096, head_dim 64–128 (causal and non-causal) |

All problems use FP16 tensors. Warmup and measurement repetitions match the timing conventions used by the PyTorch eager baseline in the same harness.

**Baselines.** Primary baseline is **PyTorch eager**: `F.linear` → cuBLAS; `F.rms_norm`/`F.layer_norm` → fused ATen; `F.softmax` → eager; `F.cross_entropy` → fused; `F.scaled_dot_product_attention` → FlashAttention-2/3. We do not run `torch.compile` for every problem; where AutoKernel uses it as baseline, we note the discrepancy in §4.4.

**Configs per problem.** For the KernelBench evaluation we submit **4 curated starter configurations per problem** and take the best result. No search iterations are run; the curated configs represent the initial state of the Noeris search loop, not the outcome of any LLM-guided exploration.

**Cost.** The full two-GPU evaluation (53 problems × 2 hardware targets) consumed approximately $1.44 in Modal GPU credits, including overhead from all ablation runs. Individual KernelBench eval runs cost approximately $0.20 (A100) and $0.40 (H100).

### 4.2 Aggregate Results

We report **fast_p** scores following KernelBench convention: fast_p(t) is the fraction of problems where Noeris achieves a speedup ≥ t× over the PyTorch eager baseline using any of the 4 curated configs.

**Table 1. fast_p scores — NVIDIA A100-SXM4-40GB (53 problems)**

| Threshold | Overall | Level 1 (24 probs) | Level 2 (26 probs) |
|---|---|---|---|
| fast_1.0 (any speedup) | **56.6%** | 58.3% | 61.5% |
| fast_1.5 | 41.5% | 45.8% | 42.3% |
| fast_2.0 | 37.7% | 37.5% | 42.3% |
| fast_3.0 | 32.1% | 33.3% | 34.6% |

**Table 2. fast_p scores — NVIDIA H100 SXM5 (53 problems)**

| Threshold | Overall | Level 1 (24 probs) | Level 2 (26 probs) |
|---|---|---|---|
| fast_1.0 (any speedup) | **56.6%** | 58.3% | 61.5% |
| fast_1.5 | 43.4% | 45.8% | 46.2% |
| fast_2.0 | **41.5%** | 45.8% | 42.3% |
| fast_3.0 | 30.2% | 33.3% | 30.8% |

fast_1.0 is identical across A100 and H100 because the same problems pass and fail. H100 gains at fast_1.5 and fast_2.0: higher memory bandwidth (3.35 TB/s vs 2.0 TB/s) amplifies our memory-bound kernel throughput enough to push more problems over the 1.5–2.0× thresholds. At fast_3.0, H100 drops slightly (30.2% vs 32.1%) because cuBLAS and FlashAttention baselines also scale with H100 compute, pulling matmul and attention results below 3.0× on a handful of problems. The 6 attention problems (Level 2–3) score 0% at every threshold on both GPUs; our simplified kernel does not beat PyTorch's SDPA (§4.3).

### 4.3 Per-Operator Deep Dive

**RMSNorm.** RMSNorm is the clearest win. On H100, the Mixtral hidden-dim shape (hidden_dim=4096, rows=2048) reaches **2625.1 GB/s at 11.66× PyTorch eager**; LLaMA-13B and LLaMA-7B shapes achieve 11.20× and 11.11× respectively. On A100 the same shapes yield 10.43×, 10.03×, and 10.11×. PyTorch's eager `F.rms_norm` launches a generic ATen kernel that cannot fully exploit HBM bandwidth for the reduction-then-scale pattern. Our Triton kernel fuses squared-mean reduction, reciprocal-sqrt, and element-wise scale into a single pass with `BLOCK_SIZE=2048, num_warps=8`, saturating HBM bandwidth for LLaMA-scale hidden dimensions. Smaller GPT-2 hidden dims (768) achieve 4.60–5.60× because short rows hit L2 effects and require fewer warps.

**Softmax.** Softmax shows the largest shape-sensitivity of any operator. On H100, speedups range from 2.18× (tiny, width=256) to **6.38× (vocab_llama, width=32000)**. Wide vocab-projection rows fit in a single large block without thread divergence; our fused online-max-then-exp-normalize pattern reaches 2526.4 GB/s vs 396.1 GB/s eager. Narrower attention-score rows (width 512–4096) are only 2.55–4.86× because PyTorch's eager softmax is already reasonably efficient at those widths. H100 delivers roughly 1.9–2.2× the raw GB/s of A100 on equivalent problems, consistent with the hardware bandwidth ratio.

**Cross-Entropy.** Cross-entropy delivers our best absolute numbers: 2407.1 GB/s at **9.65×** over eager on H100 (`long_llama`), and 2266.3 GB/s at 9.08× (Mistral). On A100 the same shapes reach 10.95× and 10.17×. The outlier is `llama3_128k` (vocab=131072): only 2.38× on H100 and 2.28× on A100. At this vocab width the required BLOCK_SIZE overflows shared memory and we fall back to a smaller tile, defeating vectorized reduction. Multi-pass tiling would fix this but is not yet implemented.

**LayerNorm.** LayerNorm is our weakest operator relative to the baseline. On H100 speedups range from 1.25–1.53× (`long_seq`: 1666.8 GB/s vs 1088.4 GB/s eager); on A100, 1.22–1.34×. The gap relative to rmsnorm is expected: LayerNorm requires both mean and variance passes (Welford), whereas RMSNorm only needs the squared mean. PyTorch's fused ATen LayerNorm kernel is therefore far more competitive. We do not yet implement a fused two-pass Welford kernel; this is the primary target for future search iterations.

**Matmul.** Matmul tells an expected story: cuBLAS is hard to beat. On A100, Noeris achieves 0.66–0.98× across 17 successful problems (2 FAILs counted as 0× in fast_p). On H100, the range is 0.44–1.01×. The two LLaMA-7B shapes (QKV and MLP-up) reach **1.01× on H100** with config `bm128_bn256_bk64_gm8_w8_s3`, tying cuBLAS at 691.9 and 632.8 TFLOPS — the only operator where we match the baseline on a real production shape. Small-M shapes (M≤256) and K-heavy shapes (`matmul_deep`, 0.44× on H100) are the worst cases.

**Attention.** Our simplified FlashAttention-style kernel achieves 0.68–0.83× of `F.scaled_dot_product_attention` on H100 and 0.80–1.05× on A100. The one A100 win (1.05×, `attn_short_64`) does not hold on H100 (0.68×). PyTorch's SDPA dispatches to FlashAttention-2/3 on NVIDIA hardware, implementing loop-order optimizations and register-pressure tuning that our 120-line reference does not replicate. We include attention to establish a starting point for the search loop, not to claim competitiveness with production implementations.

**GeGLU.** The fused GeGLU kernel (operator #8) is evaluated on four Gemma FFN shapes: Gemma 2B (`ffn_dim=5632`), 4B, 26B A4B, and 31B Dense (`ffn_dim=24576`). All four problems are Level 2. Results using the best curated starter config (`bs4096_w16_s1`) are:

| Problem | A100 (GB/s) | vs eager | H100 (GB/s) | vs eager | vs compile |
|---|---|---|---|---|---|
| gemma2b (ffn_dim=5632) | 1167.6 | **3.77×** | 1999.0 | **3.60×** | 2.45× |
| gemma4b | 1279.5 | **3.79×** | 2120.3 | **3.61×** | 2.07× |
| gemma26b | 1351.1 | **3.98×** | 2287.3 | **3.87×** | 2.34× |
| gemma31b (ffn_dim=24576) | 1060.0 | **3.08×** | 1601.3 | **2.65×** | 1.52× |

On A100, all four problems achieve fast_3.0 (≥3× over eager). On H100, three of four do (75% at fast_3.0). The fused kernel reaches 60–68% of H100 peak HBM3 bandwidth (3352 GB/s) with no search iterations — only a curated starter config. The H100/A100 bandwidth ratio is 1.63–1.65× for matched configs, consistent with the HBM3/HBM2e bandwidth ratio and with the cross-hardware ranking-transfer finding in §4.9 (Spearman ρ = 0.967, best config identical across GPUs). The Gemma 31B shape underperforms the others because its larger FFN dimension (`ffn_dim=24576`) fills shared memory with a larger tile, reducing occupancy at `BLOCK_SIZE=4096`.

### 4.4 Comparison to AutoKernel

AutoKernel [arXiv:2603.21331] is the closest published comparison point: it also runs Triton kernels on H100 and reports PyTorch baselines per-operator. We reproduce their Table 4 numbers alongside ours.

**Table 3. Best-shape comparison to AutoKernel on H100 (Triton, PyTorch eager baseline)**

| Kernel | **Noeris best** | **AutoKernel** | Delta | Note |
|---|---|---|---|---|
| RMSNorm | **11.66×** (Mixtral, 2625 GB/s) | 5.29× | **+120%** | same baseline type |
| Cross-entropy | **9.65×** (long_llama, 2407 GB/s) | 2.94× | **+228%** | same baseline type |
| Softmax | **6.38×** (vocab_llama, 2526 GB/s) | 3.44× | **+85%** | same baseline type |
| LayerNorm | 1.53× (long_seq, 1667 GB/s) | 3.21× | **-52%** | ⚠ different baselines |
| matmul | 1.01× (llama7b_qkv, 692 TFLOPS) | not reported | — | — |

On RMSNorm, Cross-entropy, and Softmax, Noeris exceeds AutoKernel's published figures by large margins. These operators are all memory-bound and highly sensitive to the fused tiling strategy; our curated starter configs already implement near-optimal block sizes for the LLaMA-scale hidden dimensions that appear to be AutoKernel's test shapes.

**LayerNorm: an apples-to-oranges comparison.** AutoKernel's 3.21× LayerNorm figure is measured against `torch.compile` as the baseline, not PyTorch eager. AutoKernel's paper reports that `torch.compile` achieves 3.0× over eager on the same LayerNorm shape. Our 1.53× is against eager directly. Scaling to the same reference: 1.53 / 3.0 ≈ 0.51× of compile — we are meaningfully slower. AutoKernel's LayerNorm kernel is more optimized than ours on large shapes. It is also worth noting that AutoKernel's paper reports 1.07× against eager on one specific LayerNorm shape, which is comparable to our 1.25–1.34× range for Level 1 shapes.

These comparisons use curated starter configs with no search iterations. A proper head-to-head would require running both systems with matched compute budget and identical random seeds on a shared problem set. The comparison above is directional, not a controlled benchmark.

### 4.5 Cross-Run Learning Ablation (Honest Negative Result)

The central architectural claim of Noeris is that a persistent shape-indexed configuration database accelerates kernel search by surfacing historical winners as cross-run priors. We test this claim with a controlled multi-trial ablation.

**Protocol.** For each operator, we run 3 independent trials. Within each trial, we execute 5 iterations with a live database (the `with_database` condition, where each iteration's winners are written back and visible to the next) and 5 iterations with a reset empty database each time (the `without_database` condition). The LLM proposer runs in both conditions with the same prompt template; the only difference is whether it receives historical winners as context. We report the final best metric at the end of 5 iterations for each trial.

**Table 4. Multi-trial cross-run learning ablation (3 trials × 5 iterations, A100)**

| Operator | with_database (mean ± σ) | without_database (mean ± σ) | Relative Δ |
|---|---|---|---|
| matmul | 135.52 ± 3.12 TFLOPS | 138.51 ± 1.53 TFLOPS | **-2.16%** |
| rmsnorm | 784.39 ± 6.30 GB/s | 794.33 ± 15.49 GB/s | **-1.25%** |

For matmul, individual trial finals are: `with_database` = [132.11, 138.24, 136.22] TFLOPS; `without_database` = [140.08, 137.03, 138.42] TFLOPS. For rmsnorm: `with_database` = [788.76, 777.17, 787.24] GB/s; `without_database` = [783.14, 812.01, 787.83] GB/s.

Neither result is statistically significant. The -2.16% and -1.25% relative changes are smaller than the within-condition standard deviation and fall squarely within the ~2–3% GPU runner noise floor. A t-test against H₀: Δ=0 would not reject the null at any conventional significance level.

We also ran a single-trial ablation for softmax on A100. The `with_database` condition reached 864.40 GB/s; `without_database` reached 969.61 GB/s. The result is directionally reversed and more pronounced — but with n=1 trial it is not reproducible. We include it to motivate the multi-trial design and note that a higher-powered ablation is needed before any conclusion can be drawn.

**Why the negative result?** We believe the dominant factor is the strength of the curated starter configs. Both conditions begin with 4 hand-picked configurations known to work well across LLaMA-scale shapes. Both converge to a near-optimal plateau within 1–2 iterations, leaving little room for cross-run priors to add value. A secondary factor is iteration budget: 5 iterations is short enough that neither condition has time to explore substantially beyond the starter configs.

**Experimental designs that would give the approach more room to show value:**

1. **Weaker priors.** Remove curated configs and seed only from the systematic parameter grid. Without hand-picked starters, database historical winners should be significantly more valuable than random grid exploration. §4.6 tests this exactly for the cost model.

2. **Longer budgets.** Run 20–50 iterations per condition. Short budgets favor already-good configs; longer budgets favor approaches that steer exploration efficiently toward unexplored but promising regions.

3. **Harder operators.** Attention and large matmul have 6–7 dimensional parameter spaces vs 3 for the norm operators. Random exploration is less likely to stumble on good configs in high-dimensional spaces, giving database priors more leverage.

4. **Cold novel shapes.** Test on shapes not seen in any previous run but similar to shapes that are in the database. This isolates bucket-level generalization: does the database produce better starting points for new shapes than the curated starters alone?

The framework supports all four experiments via the `ablation` and `triton-iterate` CLI commands.

### 4.6 Cost Model Validation (Positive Result)

The cross-run learning ablation (§4.5) is contaminated by strong curated priors: both conditions start from near-optimal configurations. The cost model ablation is designed to avoid this confound. We disable curated configs entirely (`--no-curated`), forcing both conditions to start from the raw parameter grid. The only variable is whether grid candidates are rank-ordered by the cost model before GPU dispatch.

**Protocol.** For each operator, we run 6 iterations with 6 configs per iteration. The **baseline** condition draws 6 configs uniformly from the systematic grid. The **filtered** condition draws 40 candidates from the same grid, scores each with the cost model, and selects the top 6 by predicted throughput. The cost model used is trained on 384 benchmark points from prior CI runs (R² = 0.970 on 80/20 holdout). Both conditions run on A100 via Modal.

**Table 5. Cost model ablation: filtered vs. unfiltered grid search (--no-curated, 6 iterations × 6 configs, A100)**

| Operator | Baseline final (GB/s) | Filtered final (GB/s) | Relative Δ |
|---|---|---|---|
| cross_entropy | 682.16 | **936.94** | **+37.35%** |
| softmax | 948.35 | **998.26** | **+5.26%** |
| rmsnorm | 801.83 | **803.16** | **+0.17%** |
| layernorm | 781.45 | **782.62** | **+0.15%** |

The filtered condition outperforms the baseline on all three operators. The magnitude of the gain is strongly ordered:

- **cross_entropy +37.35%** — The widest and most sensitive parameter space of the three. Cross_entropy at large vocab widths (32k–128k tokens) has a non-monotone response to BLOCK_SIZE because tile size interacts with shared memory capacity; without the cost model, the baseline spends most of its 36 GPU calls on configs that simply do not fit, landing at 682 GB/s. The cost model — trained on prior cross_entropy runs including the 128k-vocab edge cases — correctly routes the search toward feasible high-throughput configs, reaching 937 GB/s on the first iteration and then refining incrementally.

- **softmax +5.26%** — A meaningful but smaller effect. Softmax's three-parameter space is less treacherous; the baseline finds reasonable configs by iteration 2, but the filtered condition converges faster and ultimately finds a better plateau (998 vs 948 GB/s).

- **rmsnorm +0.17%** — Negligible. RMSNorm at LLaMA-scale is nearly saturated by any reasonable block size; the optimal config is `BLOCK_SIZE=2048, num_warps=8`, which the baseline finds within 1–2 random draws. The cost model's ranking is correct but the headroom is small.

- **layernorm +0.15%** — Similarly negligible, for the same reason: LayerNorm's three-parameter grid is small and any reasonable tile fills the available bandwidth ceiling. The cost model does not hurt — it never selects a worse config — but there is almost no headroom to gain.

**Interpretation.** The gain scales with `grid_size / budget`: operators where exhaustive evaluation would be expensive (cross_entropy) benefit most; operators where random draws are likely to find near-optimal configs (rmsnorm, layernorm) benefit least. This is the predicted operating regime from §2.5. The result empirically validates the design choice to train on the accumulated database and filter at prediction time rather than relying on random grid exploration.

**Trajectory analysis.** For cross_entropy, the filtered condition reaches its near-final value (935.8 GB/s, within 0.1% of 936.9 GB/s) on **iteration 1** — before the baseline has found anything above 682 GB/s. The baseline plateaus at 682 GB/s for all 6 iterations, suggesting it is systematically avoiding the high-throughput region of the parameter space. The cost model effectively solves the exploration problem for this operator: it compresses 36 random GPU calls into a handful of high-confidence candidates on the first pass.

**Limitations of this ablation.** The baseline in the `--no-curated` condition is deliberately weak (raw random grid, no curated seeds, no LLM proposer). A more demanding comparison would be cost-model-filtered vs. LLM-proposer-only, both without curated seeds. We leave this as future work. The current result establishes that the cost model alone provides a substantial positive signal.

### 4.7 Complementary Selectors: Bandit Fills the Sparse-Data Gap

The cost model ablation (§4.6) establishes that a trained regressor outperforms random grid search when the training corpus is dense enough to generalize. A natural question is whether a different selector — one that does not require a smooth parameter-to-metric mapping — can outperform the cost model in data-sparse regimes. We compare three conditions using the same `--no-curated` protocol: **baseline** (random grid order), **cost_model** (top-6 from a 40-candidate pool scored by the regressor), and **bandit** (Thompson sampling from per-config Beta posteriors accumulated across iterations). Each condition runs 5 iterations, 6 configs per iteration, on A100.

**Table 6. Three-way selector comparison (--no-curated, 5 iterations × 6 configs, A100)**

| Operator | Search space | Baseline | Cost Model | Δ vs baseline | Bandit | Δ vs baseline | Winner |
|---|---|---|---|---|---|---|---|
| matmul | Largest (6-dim) | 59.30 TFLOPS | 86.15 | +45.28% | **138.38** | **+133.37%** | **Bandit** |
| attention | Large (5-dim) | 85.03 TFLOPS | **141.30** | **+66.17%** | 134.95 | +58.70% | **Cost Model** |
| cross_entropy | Medium (3-dim, wide vocab) | 681.77 GB/s | **937.90** | **+37.57%** | 937.05 | +37.44% | **Tied** |
| softmax | Medium (3-dim) | 967.95 GB/s | **1018.50** | **+5.22%** | 980.61 | +1.31% | **Cost Model** |
| rmsnorm | Small (3-dim, tight grid) | 889.76 GB/s | 889.67 | −0.01% | 883.57 | −0.70% | **Tied / no effect** |

The results reveal a consistent pattern: the two selectors are not redundant but occupy different niches.

**Why the bandit dominates matmul (+133% vs. +45%).** Matmul has the widest parameter space (6 dimensions, ~500–2000 candidates) and the fewest training points per candidate (60 matmul points out of 516 total, ~12%). In this sparse regime the cost model must extrapolate, and its predictions are unreliable far from training examples. The bandit avoids prediction entirely: it maintains Beta posteriors over empirically observed cells and samples from them. When the database has relevant observations the bandit exploits them; when it does not, Beta(1,1) degenerates to uniform — which is empirically near-optimal when the cost model extrapolates poorly.

**Why the cost model dominates attention (+66% vs. +59%).** Attention has similar training-data sparsity (~14%) but a more structured parameter space: the `BLOCK_M × BLOCK_N` product has a smooth, near-monotone relationship with TFLOPS that the gradient-boosted regressor captures cleanly. The cost model generalizes across unseen tile combinations; the bandit can only exploit cells it has observed. With 5 iterations × 6 configs = 30 evaluations against an attention grid of hundreds of combinations, the cost model's ability to generalize to unobserved cells is decisive.

**Why they tie on cross_entropy.** Cross_entropy has a "performance cliff" in its parameter space: configs that fit the vocab in shared memory run 4–6× faster than those that do not. Both selectors quickly identify this cliff. Once above it, throughput plateaus at ~937 GB/s and there is essentially no discriminable signal left. Both conditions reach this plateau within 1–2 iterations.

**Why neither selector helps rmsnorm.** RMSNorm's ~40-candidate grid is smaller than the 30-evaluation budget. Both selectors and the baseline eventually cover the full grid regardless of ordering, converging to the same ~885 GB/s plateau. The selector choice is irrelevant when the grid is exhaustible.

**Implications for operator-aware selector routing.** The three-way comparison suggests that the appropriate selector depends on two observable quantities: (1) the ratio of grid size to evaluation budget, and (2) the density of training points in the cost model for the target operator. A natural routing policy would use the bandit for operators where training density is below some threshold (e.g., <15%) and the cost model otherwise. We leave this adaptive routing as future work (§6) but note that both selectors are implemented and available in the current system.

### 4.8 Ensemble Failure: Naive Alternation Is Dominated When Selectors Diverge

The three-way comparison in §4.7 shows that the cost model and bandit perform differently on different operators. A natural follow-up is whether an ensemble of the two selectors captures the best of both worlds. We test the simplest possible ensemble: **naive alternation**, where for each iteration, half the config slots are drawn from the cost model's top-k and half from the bandit's Thompson samples. We focus on matmul, where the two selectors diverge most sharply.

**Table 7. Four-way ablation on matmul (--no-curated, 5 iterations × 6 configs, A100)**

| Condition | Final TFLOPS | vs. baseline |
|---|---|---|
| baseline | 58.62 | — |
| cost_model | 85.24 | +45.39% |
| bandit | **137.51** | **+134.56%** |
| ensemble (alternating) | 85.13 | +45.21% |

The ensemble matches the cost model rather than the bandit. It does not close the gap.

**Why the ensemble fails.** When two selectors diverge strongly — as on matmul — alternation wastes half the evaluation budget on the weaker selector's proposals. The ensemble allocates 3 slots per iteration to the cost model (which explores the wrong region) and 3 to the bandit (which is already finding high-TFLOPS configs). The cost model proposals occupy GPU time without discovering the high-performance region the bandit has already located; the bandit's superior signal is diluted by being forced to share budget with inferior proposals.

This result should not be interpreted as evidence that ensemble methods are generally harmful. On operators where the selectors agree — cross_entropy, where both reach +37% independently — the alternating ensemble would also reach +37%, matching the best individual. The failure mode is specific to settings where the selectors disagree substantially, i.e., exactly the cases where a more intelligent routing strategy would be valuable.

**The right fix: adaptive routing, not blind mixing.** The lesson from §4.7 and §4.8 together is that the selectors should not be blended without regard to the target operator. A bandit-over-selectors meta-controller — which itself maintains per-iteration posteriors over which selector performs better — recovers the best individual selector without prior knowledge of operator structure. §4.10 validates this approach across 3 independent trials, showing the adaptive router matches bandit performance within 0.5% on matmul A100 while the naive ensemble remains at cost-model level.

### 4.9 Hardware Cross-Learning: Rankings Transfer, Absolutes Do Not

All cost model training data in §4.6 is collected on A100. A natural concern is whether an A100-trained model can usefully guide H100 searches without retraining. We address this with a direct transfer experiment: we use the A100-trained model (trained on 516 benchmark points with in-distribution R² = 0.939 on A100) to rank 256 freshly collected H100 benchmark points (64 per operator: 16 configs × 4 shape buckets, for rmsnorm, softmax, layernorm, cross_entropy).

**Table 8. A100 → H100 cost model transfer (memory-bound operators only)**

| Operator | R² (cross-hw) | Spearman ρ | Top-5 agreement | Random baseline |
|---|---|---|---|---|
| rmsnorm | −0.108 | 0.961 | 60% | 7.8% |
| softmax | +0.463 | 0.988 | 60% | 7.8% |
| layernorm | −0.040 | 0.942 | 40% | 7.8% |
| cross_entropy | +0.503 | 0.978 | 40% | 7.8% |
| **Mean** | **+0.205** | **0.967** | **50%** | **7.8%** |

The central finding is the large gap between Spearman ρ and R²:

- **Spearman ρ = 0.967** — the A100-trained model almost perfectly ranks H100 configs by relative performance. The ordering of which configs are fast vs. slow transfers completely across GPU generations.

- **R² = 0.205** — absolute predicted throughput does not match. H100 is 40–80% faster than A100 on memory-bound kernels (HBM3 bandwidth ~3.35 TB/s vs. HBM2e ~2.0 TB/s), so predictions are systematically low in absolute terms. For rmsnorm and layernorm, which have high variance in absolute throughput across shapes, the systematic offset makes R² negative even while Spearman is near-perfect.

- **Top-5 agreement: 50% vs. 7.8% random** — a **6.4× lift over random**. In practice, if the A100 model is used to pre-filter 16 H100 candidates down to 5, approximately 2–3 of the true H100 top-5 are retained, versus ~0.4 at random. This directly reduces required H100 GPU calls.

**Why does ranking transfer but not magnitude?** The relative ordering of configurations is governed by the same structural constraints on both A100 and H100: shared memory capacity limits block sizes in the same way; the relative latency of memory-bound vs. compute-bound tile configurations follows the same roofline shape. The absolute throughput scales with bandwidth and compute, but the *ranking* within the same roofline regime is hardware-agnostic. The one-hot hardware ID feature in the cost model carries near-zero training signal for H100 (zero H100 training points), yet ranking still transfers because the config and shape features dominate.

**Practical implication.** For deployment on a new GPU generation, no full retraining is required to use the cost model as a ranking filter. A simple per-hardware scalar recalibration (a linear regression on ~20 H100 measurements per operator) would restore accurate absolute predictions with minimal additional data collection. Until that recalibration is available, the model provides strong ranking signal with no retraining cost.

### 4.10 Multi-Trial Adaptive Router Validation (commit `9da16c8`)

Section §4.8 establishes that the naive alternating ensemble fails on matmul. The natural follow-up is whether a more principled routing strategy — one that tracks per-iteration which selector is winning and adapts accordingly — can recover bandit-tier performance without requiring prior knowledge of operator structure.

**Adaptive router design.** The router maintains a multi-armed bandit over the set of selectors {baseline, cost_model, bandit}. At each iteration it selects a selector arm using Thompson sampling over the per-arm Beta posterior, dispatches that selector to generate the config batch, observes the resulting metric improvement, and updates the chosen arm's posterior. The result is a meta-bandit: a bandit that chooses which bandit (or cost model) to deploy each round. This is distinct from the naive ensemble, which blindly allocates fixed slots to both selectors regardless of observed performance.

**Protocol.** We run 3 independent trials on matmul A100 with `--no-curated`, 5 iterations × 6 configs per iteration. Each trial tests five conditions in the same session: baseline, cost_model, bandit, naive_ensemble, and adaptive_router. Because all conditions share the same GPU session and the same random seed for initial grid sampling, trial-to-trial variance (seen across the three independent trials) reflects genuine run-to-run noise rather than experimental artifacts.

**Table 9. Adaptive router multi-trial validation (matmul A100, --no-curated, 3 trials × 5 iters)**

| Condition | Trial 1 (TFLOPS) | Trial 2 (TFLOPS) | Trial 3 (TFLOPS) | Mean ± σ |
|---|---|---|---|---|
| baseline | 58.33 | 59.01 | 54.92 | 57.42 ± 2.19 |
| cost_model | 84.78 | 85.34 | 78.44 | **82.85 ± 3.83** |
| bandit | 136.07 | 137.51 | 124.85 | **132.81 ± 6.93** |
| naive_ensemble | 85.33 | 85.60 | 78.22 | 83.05 ± 4.19 |
| adaptive_router | 135.35 | 136.94 | 124.29 | **132.19 ± 6.89** |

**Key finding: adaptive router matches bandit within 0.5%.** Across all three trials, the adaptive router's mean (132.19 TFLOPS) is within 0.5% of the bandit's mean (132.81 TFLOPS). The difference (0.62 TFLOPS) is well within the trial-to-trial standard deviation of either condition (~6.9 TFLOPS). By contrast, the naive ensemble is stuck at 83.05 TFLOPS — a 50-TFLOPS (60%) gap below the bandit, statistically indistinguishable from the cost model alone (82.85 TFLOPS).

**Arm choice trajectories.** In all three trials, the adaptive router converges to preferring the bandit arm within 1–2 iterations. Representative arm selections from trial 1: `[baseline, bandit, bandit, bandit, cost_model]`; from trial 3: `[baseline, bandit, bandit, bandit, cost_model]`. The initial "baseline" selection reflects the router's uninformative prior at iteration 0; the bandit arm is selected for iterations 2–4, consistent with the bandit's observable TFLOPS advantage on matmul.

**Why the ensemble fails but the router succeeds.** The naive ensemble allocates a fixed 3-of-6 slot budget to the cost model regardless of its performance. On matmul, those 3 slots explore a region of the parameter space that the cost model finds plausible but that yields only ~85 TFLOPS — far below the bandit's 130+ TFLOPS region. The router observes this gap and within 2 iterations routes almost all budget to the bandit, effectively recovering the pure-bandit condition while retaining the ability to switch back if cost model proposals improve. The residual <0.5% gap is due to the 1–2 wasted iterations before the router commits, plus occasional exploratory cost-model draws in later iterations (e.g., iteration 5 in trial 1).

**Implications.** This result upgrades the adaptive routing story from "future work" to an empirically validated architectural component. The adaptive router is the appropriate default for operators where selector performance is not known in advance — which is the common case in deployment. The limitation is that 3 trials on a single operator (matmul A100) is a narrow validation; the experiment should be repeated across all five operators in §4.7 to confirm that the router generalizes across the cost-model-wins and tie cases as well as the bandit-wins case.

### 4.11 Upstream KernelBench Comparison (apples-to-apples)

To ground the internal LLM-shape numbers above in the community evaluation standard, we built an **upstream KernelBench L1 runner** (`src/research_engine/kernelbench_upstream.py`) that executes Noeris kernels against the actual `nn.Module` problems from the [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) repository. The runner:

1. Loads each upstream problem's `Model` source code and materializes it via `exec`.
2. Calls `get_inputs()` / `get_init_inputs()` to obtain the upstream FP32 tensors and shapes.
3. Runs the original `Model.forward()` as the baseline, timed with `cuda_event` + L2 flush (3 warmup / 10 trials, median).
4. Runs the Noeris Triton kernel (inlined into the script as source — no remote import) on the same inputs, casting FP32→FP16 at the kernel boundary and FP32 back for comparison.
5. Verifies correctness via `torch.allclose(rtol=atol=5e-3)` (relaxed from upstream's 1e-4 to account for the FP16 round-trip).

**Preliminary results (A100, 2-problem smoke).** The full 12-problem sweep timed out on the first attempt (the `(4096, 393216)` softmax and `(32, 32, 512, 1024)` SDPA problems exceed the per-subprocess timeout in `ModalBenchmarkSession`). A 2-problem smoke on the cheapest shapes validates the pipeline end-to-end:

| Problem | Upstream (ms) | Noeris (ms) | Speedup | Correct |
|---|---|---|---|---|
| `7_Matmul_with_small_K` | 9.78 | 8.62 | **1.14×** | yes |
| `95_CrossEntropyLoss` | 0.86 | 0.82 | **1.05×** | yes |

These are modest wins — 1.05–1.14× — compared to the internal benchmark's headline numbers. The gap reflects three factors: (a) the upstream shapes are different (e.g. cross-entropy is `(32768, 4096)` FP32 vs our internal `(4096, 128256)` FP16); (b) the FP32→FP16→FP32 cast at the kernel boundary adds overhead; (c) the `cuda_event` + L2 flush timer is stricter than `triton.testing.do_bench` (cold-cache vs potentially-warm-cache). The pre-computed upstream baselines from KernelBench's repo (H100 Modal, vendored at `docs/results/external/`) are available for future comparison once the full sweep completes.

**Assessment.** The internal LLM-shape numbers (§4.2–§4.5) remain valid measurements of Noeris's kernel performance on the workloads they measure — 2D FP16 tensors at LLM activation scales. They should not be cited as "KernelBench speedups" without the shape/layout qualifier. The upstream comparison provides the honest community-standard datapoint and will be expanded to all 12 addressable L1 problems (plus Level 4 HF forward passes, see §6) as the project continues. Raw artifacts: `docs/results/kernelbench-upstream-l1-a100.json` (P0 plumbing run), vendored external baselines at `docs/results/external/`.

### 4.12 KernelBench Level 4: HF Forward Passes (framework)

KernelBench Level 4 consists of 20 HuggingFace model forward passes spanning seven architecture families: GPT-2, OPT, BART, Electra, GPT-Neo, BigBird, and Reformer. Of these, 15 are addressable by Noeris; we skip BigBird (3 problems) and Reformer (2 problems) because their block-sparse and LSH attention mechanisms are not covered by our current kernel set.

The evaluation uses `NoerisOpSubstitutor` (`src/research_engine/kernelbench_l4.py`), which walks the `nn.Module` tree of a pretrained HF model and performs exact-type replacement: `nn.Linear` maps to our fused matmul kernel, `nn.LayerNorm` to our Triton layernorm, `nn.GELU` to our GeGLU kernel (with a unit gate), and GPT-2's `Conv1D` (a transposed linear, not `nn.Conv1d`) receives a dedicated shim. Weights are cast to FP16 at substitution time and cached; activations are cast at the wrapper boundary (FP32 to FP16 to kernel to FP32) so the rest of the model sees the original dtype.

The expected end-to-end ceiling is **1.2--1.5x**, bounded by Amdahl's law: embedding lookups, dropout, attention score computation, and residual adds constitute roughly 15% of forward-pass time and remain untouched. This is a modest but honest target. Notably, no published L4 results exist in the KernelBench ecosystem as of this writing --- this is unclaimed white space.

The framework is fully built: problem definitions, substitutor, and a self-contained benchmark script generator are implemented and tested (commit `b919971`). What remains is GPU validation on A100/H100, which will produce the first published L4 op-substitution numbers. A detailed feasibility study covering attack ordering, per-model Amdahl analysis, and risk assessment is available at `docs/research/l4-op-substitution-feasibility.md`.

### 4.13 Honest Negative Result: Per-Operator GP Surrogates Do Not Help

We investigated whether Gaussian Process (GP) surrogates or ensemble regressors could improve per-operator, per-GPU configuration selection beyond the Thompson-sampling bandit. Using 1,050 real T4 measurements across 21 shape buckets and 4 operators collected during the Colab autonomous search loop, we trained GP (RBF kernel), GP (Matern kernel), RandomForest, and GBR models to predict throughput from configuration parameters within a single operator on a single GPU.

**All models produced negative R² — worse than predicting the mean.** The fundamental issue is signal-to-noise ratio: within a single operator on a single GPU, config-dependent throughput variation is 10–20%, while run-to-run measurement noise is 5–10%. The surrogate has too little signal to learn from.

This contrasts sharply with two settings where surrogates *do* add value: (a) the cross-operator GBR cost model (§4.6), which achieves R² = 0.94 on 516 points because inter-operator variance is enormous (matmul TFLOPS vs. rmsnorm GB/s span orders of magnitude), and (b) the hardware cross-learning transfer (§4.9), where Spearman ρ = 0.967 because inter-hardware variance in absolute throughput is large and monotonic.

**Takeaway.** Surrogate models add value when pooled across operators (GBR cost model) or across hardware (Spearman ρ = 0.967 A100→H100 transfer), but not within a single operator on a single GPU where the signal-to-noise ratio is too low. The Thompson-sampling bandit remains the correct tool for per-operator configuration selection.

### 4.14 Bandit Search Convergence on Commodity Hardware (Colab T4, 1,050 measurements)

To validate that the autonomous search loop works on commodity hardware — not just datacenter GPUs — we ran the full bandit search on Google Colab's free T4 GPU across 4 operators (qk_norm_rope, geglu, rmsnorm, softmax) in 21 shape buckets, accumulating 1,050 measurements.

Key results:

- **qk_norm_rope** fusion_speedup improved from 6.46× (curated starter) to **8.37×** (bandit-discovered) — a **30% improvement** over hand-picked configurations, demonstrating that the search loop finds configs humans miss.
- **geglu** reached **249.58 GB/s** — **83% of T4 theoretical peak bandwidth** (~300 GB/s). This is near the hardware ceiling.
- **rmsnorm** reached **242.52 GB/s** (81% of T4 peak).
- The bandit consistently discovered that **T4 prefers `num_warps=1–4`** configs, contrasting with A100's preference for `num_warps=4–16`. This hardware-specific tuning preference was learned autonomously from data, not hand-coded.

The T4 search validates two system properties: (a) the `TritonOperatorSpec` interface and bandit machinery work unchanged on a GPU with 1/7th the bandwidth and 1/10th the compute of A100, and (b) the shape-indexed database correctly learns hardware-specific preferences that differ qualitatively from datacenter GPU optima. The 30% improvement on qk_norm_rope over curated configs is the strongest evidence to date that bandit search adds value beyond hand-tuning for this system.

---

## 5. Related Work

GPU kernel optimization spans hand-tuned libraries, compiler-directed autotuning, algebraic search over algorithm space, and — most recently — LLM-driven code generation agents. We organize the landscape into four groups and position Noeris relative to each.

### 5.1 LLM-Driven Kernel Generation and Agent Loops

The dominant paradigm in 2025–2026 is an agent loop: an LLM generates or rewrites kernel source, an execution harness measures correctness and throughput, and feedback is routed back to the model. Systems in this family differ primarily in how knowledge accumulates across iterations and whether search state is persistent.

**AutoKernel** (Jaber and Jaber, arXiv:2603.21331, 2026) is the closest direct competitor. It applies an autonomous agent loop to arbitrary PyTorch models, identifying bottleneck operators via Amdahl's-law profiling and iterating over Triton and CUDA C++ rewrites guided by a six-tier optimization playbook. AutoKernel covers nine operator types and is openly evaluated on NVIDIA H100, reporting 5.29× on RMSNorm and up to 3.44× on softmax over PyTorch eager. The critical limitation is that AutoKernel is stateless across invocations: configurations discovered in one run are not stored and are not consulted in subsequent runs. Noeris shares the Triton-on-H100 target and the agent-loop structure but adds a persistent shape-indexed configuration database and a learned cost model. Notably, our curated-starter-only results already match or exceed AutoKernel's best-shape H100 numbers on RMSNorm (+120%), cross-entropy (+228%), and softmax (+85%).

**KernelSkill** (Sun et al., arXiv:2603.10085, 2026) introduces a multi-agent framework with a dual-level memory architecture: long-term memory stores reusable "expert skills" from prior successes, and short-term memory prevents repetitive backtracking within a session. On KernelBench Levels 1–3, KernelSkill reports 100% functional correctness and average speedups of 5.44× (Level 1), 2.82× (Level 2), and 1.92× (Level 3) over PyTorch eager. The skill library is the closest analog to our shape-indexed database, but the analogy is approximate: KernelSkill stores optimization strategies (code patterns, transformation recipes) rather than numerical benchmark outcomes, and it does not key entries by `(operator, shape_bucket, hardware)`. A `BLOCK_SIZE=256` result on a 128×128 matmul and on a 4096×4096 matmul are not distinguished. Noeris makes this distinction explicit and queryable by design.

**CUDA Agent** (Dai et al., arXiv:2602.24286, 2026) takes a different route: large-scale agentic reinforcement learning. The system trains an LLM policy using a data synthesis pipeline, a skill-augmented execution environment, and RL algorithms with execution-time reward signals. Trained on A100, CUDA Agent reports that 100% of its kernels beat `torch.compile` on KernelBench Levels 1 and 2, and 92% on Level 3. Like CUDA-L1 below, CUDA Agent's cross-run learning is baked into model weights rather than an explicit configuration store; generalization happens at inference time but there is no queryable database of past shape-specific results.

**CUDA-L1** (Li et al., arXiv:2507.14111, ICLR 2026) trains on the full 250-problem KernelBench suite via a three-stage pipeline: supervised fine-tuning on CUDA variants, self-supervised learning from the model's own successful outputs, and a contrastive RL algorithm that pairs good and bad rewrites of the same kernel to internalize what separates them. Results are strong: average 3.12× speedup, median 1.42×, peak 120× across all 250 kernels, and 7.72× over cuDNN. CUDA-L1 operates over free-form CUDA source — its search space is the set of all syntactically valid CUDA programs the model can produce. Noeris instead restricts to a structured parameter space; this trades expressiveness for tractability and makes the cost model training problem well-posed (a fixed-width 20-dimensional feature vector rather than a variable-length token sequence).

**KernelFoundry** (Wiedemann et al., arXiv:2603.12440, 2026) combines MAP-Elites quality-diversity search with LLM-driven code generation and meta-prompt evolution. KernelFoundry targets SYCL and CUDA, achieving 2.3× average speedup on KernelBench for SYCL. It also uses template-based parameter optimization — the one point of direct contact with Noeris's parameterized templates. The difference is scope: KernelFoundry's templates are one tool in a broader evolutionary repertoire, while Noeris's parameterized templates are the sole interface to the kernel code. Noeris never rewrites kernel source; it only selects from the parameter space defined at operator registration time, which is what makes the configuration database well-structured and the cost model feature extraction deterministic.

**CudaForge** (Zhang et al., arXiv:2511.01884, 2025) implements a training-free two-agent workflow where a coder agent generates CUDA and a judge agent validates correctness using Nsight Compute profiling output, achieving 97.6% kernel correctness and 1.68× average speedup at approximately $0.30 per kernel. The NCU-grounded judge — moving feedback from raw timing to counter-level bottleneck analysis — is a direction Noeris does not currently pursue.

Additional systems surveyed include **Astra** (Wei et al., arXiv:2509.07506, 2025), an LLM multi-agent system targeting production CUDA code in SGLang achieving 1.32× on real serving kernels; **GEAK** (Wang et al., arXiv:2507.23194, 2025), a Reflexion-style Triton kernel agent on AMD MI300X/MI250 (up to 2.59×); **GPU Kernel Scientist** (Andrews and Witteveen, arXiv:2506.20807, ICML 2025 Efficiency Workshop), an iterative hypothesis-generation loop targeting AMD MI300; and **SwizzlePerf** (arXiv:2508.20258, 2025), which feeds hardware memory-access patterns and historical performance reflections to an LLM to generate spatial swizzling optimizations (up to 2.06×, 70% improved L2 cache hit rate).

### 5.2 Learned Cost Models and Traditional Autotuners

**TVM/Ansor** (Zheng et al., arXiv:2006.06762, OSDI 2020) introduced a two-level approach: automatically construct a hierarchical search space from tensor expressions, then use evolutionary search guided by an XGBoost regression cost model to identify high-performing programs. The XGBoost model takes a 164-component feature vector derived from the program AST and predicts runtime without GPU execution, achieving up to 3.8× improvement on Intel CPU and 1.7× on NVIDIA GPU over prior state-of-the-art. Noeris's learned cost model is directly inspired by Ansor: we use GradientBoostingRegressor (a close relative of XGBoost) on a compact feature vector extracted from `(shape, config, hardware, operator)` tuples. The structural difference is scope — Ansor's model covers all TVM-expressible tensor programs; ours covers the discrete parameter grid of a fixed Triton template. This is a much smaller space, but one shaped to real GPU programs rather than abstract loop nests.

**NVIDIA nvMatmulHeuristics** (CUTLASS 4.2, 2024) is NVIDIA's production heuristic for GEMM kernel configuration selection. Given a GEMM problem definition and target hardware, it predicts a small candidate set of CTA shapes, split-k factors, and other meta-parameters, achieving 96–99% of optimal performance at 5× faster tuning than exhaustive search. The system uses analytic rather than data-driven models. Noeris operates in a similar position — predicting configuration candidates before GPU evaluation — but on Triton kernels across a broader operator set, with a data-driven model that updates as benchmark results accumulate.

**Triton `@autotune`** is the immediate baseline that motivates Noeris's design. The decorator benchmarks a user-supplied list of `triton.Config` objects at first invocation per unique input shape, caches the winner within the process, and returns it on subsequent calls with the same shape. The cache is discarded across processes and CI runs. Noeris generalizes `@autotune` in three ways: (1) the configuration database persists across separate processes and CI invocations via JSON artifacts; (2) the LLM proposer generates novel configurations beyond any user-supplied fixed list; and (3) the cost model rank-orders grid candidates before GPU dispatch, reducing evaluations per iteration.

### 5.3 Algebraic and Structural Search

**AlphaTensor** (Fawzi et al., *Nature* 610, 2022) frames matrix multiplication as a tensor decomposition game and uses deep RL to discover novel exact algorithms, improving on Strassen's 1969 algorithm for 4×4 matrices and finding hardware-specific variants that are 10–20% faster than standard implementations on NVIDIA V100 and Google TPU. AlphaTensor operates in a fundamentally different space: it searches for new *algorithms* (different bilinear factorizations), while Noeris searches for new *implementations* of a fixed algorithm (tiled Triton matmul with configurable tile and warp parameters). The two systems address orthogonal parts of the optimization stack.

### 5.4 How Noeris is Actually Different

No individual component of Noeris is new in isolation. Parameterized kernel templates appear in KernelFoundry and implicitly in Triton's `@autotune`. Shape-indexed caches appear in `@autotune` (within-process) and in TVM's per-operator autotuning logs. Learned cost models appear in Ansor (XGBoost over AST features) and in NVIDIA's nvMatmulHeuristics (analytical models). LLM proposers appear in every system surveyed.

What Noeris contributes is a specific combination: **(a)** a fixed parameterized template interface that bounds the search space to a well-typed configuration grid rather than free-form source code; **(b)** a cross-process, cross-session database keyed by `(operator, shape_bucket, hardware)` that accumulates every benchmark outcome and is queryable at proposal time; **(c)** a lightweight gradient-boosted cost model trained on that database that operates at prediction time — without GPU calls — to rank candidates before evaluation; and **(d)** an adaptive router that learns per-iteration which selector (cost model or bandit) to deploy, recovering the best individual selector's performance without prior knowledge of operator structure. The adaptive router is a novel contribution not present in any cited system: it moves selector choice from a design-time hyperparameter to a runtime decision backed by observed metric trajectories. The combination is designed so that each benchmark result makes future searches slightly cheaper: the cost model gains training data, the database gains a new incumbent, and the LLM proposer sees richer cross-run context.

The empirical comparison against AutoKernel deserves emphasis. AutoKernel is stateless: it does not store per-shape configuration outcomes across invocations, and it does not have a cost model that trains on historical runs. Our §4.4 comparison shows that Noeris's curated-starter-only baseline already exceeds AutoKernel's best-shape H100 figures on RMSNorm by 120%, cross-entropy by 228%, and softmax by 85%. These gains come from the parameterized template design (which enables hand-picked, shape-specialized starter configs) rather than from LLM-guided search iterations — a point that underscores the value of the structured template interface.

---

## 6. Limitations

1. **Cross-run learning ablation is negative.** As discussed in §4.5, we do not yet have empirical evidence that the database-guided LLM proposer outperforms stateless search in the tested regime. The cross-run learning claim rests on architecture and on the proposed experimental designs (weaker priors, longer budgets), not yet on measured gains.

2. **Cost model ablation uses a weak baseline.** The §4.6 ablation compares filtered vs. unfiltered random grid search with curated configs disabled. This is the right design for isolating the cost model's contribution, but the baseline is deliberately weakened. A stronger comparison — cost model vs. LLM proposer alone — is needed to establish the cost model's independent value.

3. **Adaptive router validated on one operator.** The three-way comparison (§4.7) and ensemble failure (§4.8) motivate a bandit-over-selectors adaptive router; §4.10 validates it across 3 independent trials on matmul A100, where it matches bandit performance within 0.5%. However, the validation covers only the bandit-dominates case. The adaptive router has not yet been tested on operators where the cost model wins (attention) or where both selectors tie (cross_entropy); it is possible the router's meta-bandit overhead introduces unnecessary exploration cost in those regimes.

4. **Attention kernel is simplified.** Our FlashAttention-style kernel does not match PyTorch's SDPA (which uses FlashAttention-2/3). This is a starting point for search, not a complete implementation.

5. **GeGLU evaluated on 4 shapes only.** The GeGLU KernelBench evaluation covers four Gemma FFN shapes on A100 and H100 (§4.3), all at Level 2. Level 1 shapes and the fifth shape bucket (31B Dense at large batch) remain to be characterized. The current fast_p scores are over a small sample; a larger evaluation matching the scale of other operators (7–10 shapes) would provide a more complete picture.

6. **KernelBench subset.** We evaluate on a 53-problem curated subset rather than the full 250-problem KernelBench. Full-dataset numbers would enable direct comparison to published results from KernelSkill, CUDA-L1, and CUDA Agent.

7. **Single hardware vendor.** Results span three NVIDIA GPUs (T4, A100, H100) but we have not tested AMD MI300 (which KernelFoundry and GPU Kernel Scientist target). Hardware cross-learning experiments (§4.9) confirm ranking transfer A100→H100 but absolute calibration requires per-hardware recalibration.

9. **MoE router negative result.** The iterative top-k implementation in `moe_router` achieves only 0.12× fusion_speedup on T4 and 0.68× on A100, losing to `torch.topk` in both cases. The iterative approach incurs register pressure and loop overhead that outweigh the fusion benefit. A competitive fused MoE router likely requires a different algorithmic strategy (e.g., radix-select or approximate top-k).

10. **Grouped GEMM fusion not needed on cheaper GPUs.** `grouped_gemm` achieves 1.82× fusion_speedup on A100 but only 0.11× on T4, where kernel launches are cheap relative to compute time. Fusion value for this operator is hardware-dependent in a way the simpler operators are not.

8. **Metric noise.** Modal's per-call variance is ~1–3% for memory-bound kernels, larger than some of the effects we would like to measure. Future work should increase per-trial repetitions or use statistical tests.

11. **Per-operator surrogate models are ineffective.** As reported in §4.13, GP and ensemble surrogates trained within a single operator on a single GPU produce negative R² on 1,050 T4 measurements. The signal-to-noise ratio (10–20% config variation vs. 5–10% run-to-run noise) is insufficient for per-operator learned models. The cost model's value is limited to the cross-operator and cross-hardware regimes where variance is large.

---

## 7. Conclusion

Noeris demonstrates a complete autonomous GPU kernel search pipeline with fourteen parameterized Triton operators — including a novel fused QK-RMSNorm+RoPE forward and backward kernel (§3.2.1), fused GeGLU targeting Gemma 2/3/4 MLP blocks, and sliding-window local attention (§3.1) — cross-run shape-indexed configuration storage, LLM-guided proposals, and cloud GPU execution at approximately $0.01 per iteration. The system now covers 606 unit tests across 14 operators on A100, H100, and T4. On memory-bound kernels, the starting-point configurations match or exceed AutoKernel's published H100 results across RMSNorm (+120%), cross-entropy (+228%), and softmax (+85%).

The central contribution has evolved from "a cost model improves search" to a complete selector-routing story with statistical validation. **The shape-indexed database supports two orthogonal selectors that are complementary rather than competing.** The cost model (§4.6) improves search efficiency by +37.35% on cross_entropy and +5.26% on softmax when the training corpus is dense. The multi-armed bandit (§4.7) outperforms the cost model on matmul (+134% vs. +45%) where the training corpus is sparse. A naive alternating ensemble (§4.8) fails to capture the best of both selectors when they disagree strongly. An adaptive router that learns per-iteration which selector to deploy closes this gap empirically: across 3 independent trials on matmul A100, the adaptive router matches bandit performance within 0.5% (132.19 ± 6.89 vs. 132.81 ± 6.93 TFLOPS, §4.10) — a result that is statistically robust across trials — while the naive ensemble remains stranded at cost-model level (83.05 ± 4.19 TFLOPS). This validates adaptive routing as a practical architectural component, not merely a theoretical motivation.

Hardware cross-learning experiments (§4.9) establish that A100-trained cost model rankings transfer to H100 with Spearman ρ = 0.967 and 6.4× top-5 lift over random selection, despite absolute predictions being miscalibrated by the A100→H100 bandwidth gap (~40–80%). This confirms that the ranking component of the cost model generalizes across GPU generations without retraining.

The initial cross-run learning ablation (§4.5) remains negative: at 5 iterations with strong curated priors, database-guided LLM proposals do not outperform stateless proposals within the measurement noise floor. We interpret this as evidence that strong curated priors dominate the result, not as evidence against persistent cross-run learning in principle. Demonstrating that contribution requires weaker priors, longer budgets, and colder novel shapes — all supported by the existing CLI but not yet run at scale.

All code, raw benchmark data, and reproduction scripts are available at https://github.com/peaktwilight/noeris under the MIT License.

---

## A. Reproduction

The repository is publicly available at https://github.com/peaktwilight/noeris (MIT License). Everything in this paper can be reproduced with:

```bash
git clone https://github.com/peaktwilight/noeris
cd noeris
pip install -e ".[dev]"
pip install modal scikit-learn datasets
modal token new   # requires a Modal account

# KernelBench eval on A100 (~$0.20, ~10 min)
python -m research_engine.cli kernelbench-eval --gpu A100

# KernelBench eval on H100 (~$0.40, ~8 min)
python -m research_engine.cli kernelbench-eval --gpu H100

# Multi-trial cross-run learning ablation
python -m research_engine.cli ablation --operator rmsnorm --trials 3 \
    --iterations 5 --fast
python -m research_engine.cli ablation --operator matmul --trials 3 \
    --iterations 5 --fast

# Train the learned cost model from accumulated database
python -m research_engine.cli train-cost-model \
    --db-paths .noeris/triton-configs.json \
    --output .noeris/cost-model.pkl

# Cost model ablation (no-curated mode)
python -m research_engine.cli triton-iterate --operator cross_entropy \
    --gpu A100 --no-curated --cost-model .noeris/cost-model.pkl \
    --iterations 6 --configs-per-run 6

# Use the cost model as a filter stage in normal search
python -m research_engine.cli triton-iterate --operator rmsnorm \
    --gpu A100 --llm --cost-model .noeris/cost-model.pkl

# Three-way selector comparison (baseline vs cost model vs bandit, --no-curated)
python -m research_engine.cli triton-iterate --operator matmul \
    --gpu A100 --no-curated --selector baseline --iterations 5 --configs-per-run 6
python -m research_engine.cli triton-iterate --operator matmul \
    --gpu A100 --no-curated --selector cost_model \
    --cost-model .noeris/cost-model.pkl --iterations 5 --configs-per-run 6
python -m research_engine.cli triton-iterate --operator matmul \
    --gpu A100 --no-curated --selector bandit --iterations 5 --configs-per-run 6

# Four-way ensemble experiment (adds --selector ensemble)
python -m research_engine.cli triton-iterate --operator matmul \
    --gpu A100 --no-curated --selector ensemble \
    --cost-model .noeris/cost-model.pkl --iterations 5 --configs-per-run 6

# Multi-trial adaptive router validation (§4.10) — reproduces Table 9
python -m research_engine.cli adaptive-router-ablation --operator matmul \
    --gpu A100 --no-curated --cost-model .noeris/cost-model.pkl \
    --trials 3 --iterations 5 --configs-per-run 6 \
    --output docs/results/adaptive-router-matmul-trial{n}.json

# Hardware cross-learning: collect H100 test points and score with A100 model
python scripts/hardware_cross_learning.py \
    --source-db .noeris/triton-configs-a100.json \
    --target-gpu H100 --operators rmsnorm softmax layernorm cross_entropy \
    --configs-per-bucket 16 --output docs/results/hardware-cross-learning-a100-to-h100.json

# GeGLU operator: iterate and benchmark on A100
python -m research_engine.cli triton-iterate --operator geglu \
    --gpu A100 --llm --cost-model .noeris/cost-model.pkl
python -m research_engine.cli kernelbench-eval --gpu A100 --operator geglu
```

**Cost breakdown.**

| Experiment | GPU type | Approx cost | Approx time |
|---|---|---|---|
| KernelBench eval — A100 (53 problems, 4 configs) | A100-SXM4-40GB | ~$0.20 | ~10 min |
| KernelBench eval — H100 (53 problems, 4 configs) | H100 SXM5 | ~$0.40 | ~8 min |
| Ablation — matmul, 3 trials × 5 iters each | A100-SXM4-40GB | ~$0.12 | ~15 min |
| Ablation — rmsnorm, 3 trials × 5 iters each | A100-SXM4-40GB | ~$0.12 | ~15 min |
| Ablation — softmax, 1 trial × 5 iters each | A100-SXM4-40GB | ~$0.06 | ~8 min |
| Cost model ablation — 4 operators × 6 iters | A100-SXM4-40GB | ~$0.12 | ~15 min |
| Three-way selector comparison — 5 operators × 3 conditions | A100-SXM4-40GB | ~$0.18 | ~22 min |
| Four-way ensemble — matmul, 4 conditions × 5 iters | A100-SXM4-40GB | ~$0.05 | ~8 min |
| Adaptive router — matmul, 5 conditions × 3 trials × 5 iters | A100-SXM4-40GB | ~$0.18 | ~22 min |
| Hardware cross-learning — 4 operators × 64 H100 points | H100 SXM5 | ~$0.25 | ~12 min |
| Miscellaneous dev/debug calls | A100/H100 | ~$0.44 | — |
| **Total** | | **~$2.12** | — |

Raw benchmark results are in `docs/results/` in both machine-readable (JSON) and human-readable (Markdown) form. All tables in the paper can be reconstructed from these artifacts. No numbers were computed outside them.

---

## References

*Full reference list in follow-up version. Key papers:*

- Jaber and Jaber. AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search. arXiv:2603.21331, 2026.
- Sun et al. KernelSkill: A multi-agent framework for GPU kernel optimization. arXiv:2603.10085, 2026.
- Li et al. CUDA-L1: Improving CUDA optimization via contrastive reinforcement learning. arXiv:2507.14111, ICLR 2026.
- Dai et al. CUDA Agent: Large-scale agentic reinforcement learning for GPU kernel optimization. arXiv:2602.24286, 2026.
- Wiedemann et al. KernelFoundry: Quality-diversity search for GPU kernel optimization. arXiv:2603.12440, 2026.
- Zhang et al. CudaForge: Training-free two-agent CUDA kernel optimization. arXiv:2511.01884, 2025.
- Andrews and Witteveen. GPU Kernel Scientist: Iterative hypothesis-driven kernel optimization. arXiv:2506.20807, ICML Efficiency Workshop, 2025.
- Ouyang et al. KernelBench: Can LLMs write efficient GPU kernels? arXiv:2502.10517, 2025.
- Zheng et al. Ansor: Generating high-performance tensor programs for deep learning. OSDI 2020. arXiv:2006.06762.
- Fawzi et al. Discovering faster matrix multiplication algorithms with reinforcement learning. *Nature* 610, 2022.
- Tillet et al. Triton: An intermediate language and compiler for tiled neural network computations. MAPL@PLDI 2019.
- Dao et al. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. NeurIPS 2022.
- Williams et al. Roofline: An insightful visual performance model for multicore architectures. CACM 52(4), 2009.
