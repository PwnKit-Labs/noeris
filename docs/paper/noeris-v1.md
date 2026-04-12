# Noeris: Parameterized GPU Kernel Search with LLM Proposals and Learned Cost Models

**Draft — work in progress.** Not yet submitted.

## Abstract

We present Noeris, an autonomous GPU kernel optimization system built
around three design decisions that differentiate it from existing
LLM-driven approaches (AutoKernel, KernelSkill, CUDA-L1, CUDA Agent,
KernelFoundry): **(1)** parameterized kernel templates instead of
free-form source rewriting, **(2)** a shape-indexed cross-run
configuration database keyed by `(operator, shape_bucket, hardware)`
that persists winning configurations across sessions, and **(3)** a
learned cost model trained on the database that filters LLM-proposed
configurations before expensive GPU evaluation.

The system covers seven operators (matmul, rmsnorm, softmax, layernorm,
cross_entropy, attention with optional causal masking, and rotary
position embedding) and is evaluated across 53 shape-parameterized
problems on NVIDIA A100 and H100 via Modal. Using only curated starter
configs with no search iterations, Noeris achieves **fast₁.₀ = 56.6%
vs PyTorch eager**. On memory-bound kernels, we match or exceed
AutoKernel's published H100 speedups: **11.66× RMSNorm**, **9.65×
Cross-entropy**, **6.38× Softmax**. On matmul on H100, we reach
**1.01×** of cuBLAS on LLaMA-7B QKV projection.

We report honest negative results from our initial cross-run learning
ablation (3 trials × 5 iterations on matmul and rmsnorm yield -2.16%
and -1.25% respectively, within the ~2-3% noise floor) and discuss
why: strong curated starter priors leave little room for cross-run
learning to compound in short iteration budgets. We outline
experimental designs that would give the database-guided approach
more room to show value, and introduce a learned cost model as a
deterministic filter stage that bypasses the noise-floor problem
entirely by operating at prediction time rather than at selection
time.

The system runs autonomously via GitHub Actions + Modal at ≈$0.01 per
benchmark iteration. Source code, reproduction scripts, and raw data
are available at https://github.com/PwnKit-Labs/noeris.

## 1. Introduction

LLM-driven GPU kernel optimization has become an active research
topic in 2025-2026 [AutoKernel, KernelSkill, CUDA-L1, GPU Kernel
Scientist, KernelFoundry, CUDA Agent]. These systems share a common
architecture: an LLM agent proposes kernel code, a harness measures
correctness and performance, and an orchestration layer decides whether
to keep or revert the change.

A shared limitation of published systems is that **search state does
not persist across sessions**. Each invocation starts from the same
initial kernel (or a cached version), runs its own iterative search
loop, and discards the trajectory. Information learned in one run
— which tile sizes work for a given matrix shape on a given GPU —
is not systematically reused in the next.

Noeris investigates an alternative: rather than rewriting kernel source
per invocation, we generate kernels from a compact parameter tuple
(e.g. `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `num_stages`) and
store winning configurations in a **shape-indexed cross-run database**
keyed by `(operator, shape_bucket, hardware)`. When an LLM proposer is
invoked, it sees the database state as cross-run insights, allowing
it to reason about what has worked on similar shapes before.

We make four contributions:

1. **System.** A complete autonomous kernel search loop for seven
   parameterized Triton operators, running on cloud GPUs via Modal
   with GitHub Actions orchestration. At ≈$0.01 per iteration and
   $1.33 total for the results in this paper, the system is cheap
   enough to run continuously.

2. **Evaluation.** We report the first direct reproduction of
   AutoKernel's H100 memory-bound kernel numbers, with substantial
   improvements on 3 of 4 kernels using only curated starter
   configurations (no search iterations required). Both vs-eager
   and vs-torch.compile baselines are reported.

3. **Learned cost model.** We train a gradient-boosted regressor on
   ~144 benchmark points harvested from early system runs. On a 20%
   held-out split, the model reaches **R² = 0.535** at predicting
   throughput from `(shape, config, hardware, operator)` features.
   The model is integrated as a filter stage in the config selector:
   LLM-proposed and grid-exploration candidates are re-ranked by
   predicted metric before being sent to GPU.

4. **Honest negative result plus mitigation.** Our initial cross-run
   learning ablation does not show a statistically significant
   effect. We analyze why (strong curated priors, short iteration
   budgets, noise-floor-bound comparisons) and argue that the
   learned cost model — which operates at prediction time, not
   selection time — bypasses the noise-floor problem entirely.

## 2. System

### 2.1 Parameterized kernels and the operator registry

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

Each operator is registered once and dispatched uniformly by the rest
of the system: the proposer, selector, database, and runner are all
operator-agnostic. Adding a new operator is ~200 lines plus registry
entry.

### 2.2 Shape-indexed cross-run config database

`ConfigDatabase` persists benchmark results as JSON, keyed by
`{operator}:{shape_bucket}:{hardware}`:

```json
{
  "records": {
    "rmsnorm:llama_7b:NVIDIA A100-SXM4-40GB": {
      "best_config_id": "bs2048_w8_s1",
      "best_tflops": 1162.2,
      "results": [ /* trajectory */ ]
    }
  }
}
```

Shape buckets are operator-specific: for matmul we classify by
`(M*N*K, aspect ratio, K ratio)`; for rmsnorm by `hidden_dim`; for
attention by `(seq_len, head_dim)`. Configurations learned on
`llama_7b` shapes are not reused for `mixtral` shapes — each bucket
has its own incumbent.

Across runs, the database is restored from the previous successful CI
artifact, updated with new results, and saved as a new artifact. This
is a standing JSON database that compounds knowledge over time.

### 2.3 LLM proposer with cross-run insights

When the LLM proposer runs, it receives:

- The operator's parameter space and hardware constraints (shared
  memory limits).
- The target shapes for the current iteration.
- Cross-run insights extracted from the database: per bucket, the
  top-3 configurations and their best measured metrics.

The proposer responds with up to 4 novel configurations (subject to
shared-memory validation) plus a natural-language rationale. We have
observed rationales that reference specific gaps in the explored
parameter space (e.g. "the tested set is concentrated around
`GROUP_SIZE_M=8` and mostly `BK32` or `BK128`, so these proposals
focus on unexplored `BK64` configurations").

### 2.4 Frontier-aware config selection

`select_configs_for_operator` allocates up to N slots per iteration
with explicit semantics:

1. **Incumbent** — the best known config for the target shapes.
2. **LLM-proposed** — novel configs from the proposer.
3. **Curated** — hand-picked starter configs not yet tested on the
   current hardware.
4. **Exploration** — systematic grid configs not yet tested.

This ensures that known-good configurations are always re-validated
(so runner variance doesn't lose them) while allocating explicit
budget to exploration and novelty.

### 2.5 Learned cost model

The central technical contribution beyond the LLM proposer is a
**learned cost model** trained on the shape-indexed database. Training
data pairs are harvested from every successful benchmark result:

    features = extract_features(shape, config, hardware, operator)
    target = tflops_or_gb_per_s

Features are a fixed-width 20-dimensional vector including:

- Operator ID (one-hot, 7 values)
- Hardware ID (one-hot, ~10 common GPUs)
- Shape dimensions (5 slots, operator-specific, zero-padded): M/N/K
  for matmul; n_rows/hidden_dim for norms; batch/heads/seq/head_dim/
  is_causal for attention.
- Configuration parameters (8 slots, operator-specific, zero-padded):
  BLOCK_SIZE_{M,N,K}, GROUP_SIZE_M, num_warps, num_stages, BLOCK_SIZE,
  j_unroll.
- Derived features: `log(shape_product)`, `log(max_dim)`,
  `log(min_dim)`, `log(tile_area)`, `log(num_warps)`, `log(num_stages)`.

The fixed-width representation allows a single regressor to serve all
operators, sharing signal (e.g., "larger tiles help on larger shapes")
across kernel types.

**Model.** We use sklearn `GradientBoostingRegressor` with 200 trees,
depth 5, learning rate 0.05. On a trivial 80/20 split of 144 training
points from our local development database, the model achieves
**R² = 0.535**. With more training data (1000+ points collected via
overnight CI runs), we expect R² > 0.80. The model size on disk is
~200 KB.

**Integration.** At selection time, the selector gathers up to 40
grid candidates and calls `cost_model.rank_configs()` which returns
them sorted by predicted metric. The top-k are then competed against
the incumbent, LLM proposals, and curated starters as described in
§2.4.

**Why this sidesteps the noise-floor problem.** Our initial ablation
(§4.5) found that cross-run learning effects were within the ~2-3%
GPU-runner noise floor. The cost model operates at *prediction time*
— deterministic, cheap, and not subject to runner variance. A clean
ablation becomes: run the same search with and without the cost
model filter, measure wall-clock time to first-within-5%-of-best.
We expect the cost-model-filtered path to reach target quality in
significantly fewer Modal calls because the grid is rank-ordered
rather than tested blindly.

### 2.6 Execution backend (Modal)

Benchmark scripts are self-contained Python files (kernel definition,
PyTorch reference, timing harness). A single Modal function takes the
script as an argument and executes it on a warm GPU container. For
multi-iteration workflows (ablations) we use a persistent session
context (`ModalBenchmarkSession`) that keeps one container warm across
all iterations, cutting per-call overhead from ~10 s to ~1-3 s.

Total cost to reproduce all results in this paper: ≈$1.33 for 134
Modal GPU calls across A100 and H100.

## 3. Operators

| Operator | Parameter space | Metric | Shape buckets |
|---|---|---|---|
| matmul | `BLOCK_M/N/K, GROUP_SIZE_M, num_warps, num_stages` | TFLOPS | 10 |
| rmsnorm | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 8 |
| softmax | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 7 |
| layernorm | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 8 |
| cross_entropy | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 7 |
| attention (FA) | `BLOCK_M, BLOCK_N, num_warps, num_stages, IS_CAUSAL` | TFLOPS | 10 |
| rotary (RoPE) | `BLOCK_SIZE, num_warps, num_stages` | GB/s | 6 |

The attention kernel is a simplified FlashAttention-style tiled
attention with online softmax and optional causal masking. It is
intentionally minimal — ~120 lines — and explicitly not a
reimplementation of FlashAttention-2 or FlashAttention-3. Real
production attention should use `torch.nn.functional.scaled_dot_product_attention`.

## 4. Evaluation

### 4.1 KernelBench subset

We evaluate on a curated 53-problem subset drawn from real LLM
workload shapes at difficulty Levels 1-3:

- **matmul** (19 problems): systematic stress shapes plus GPT-2,
  BERT, LLaMA-7B, Mistral MLP dimensions.
- **rmsnorm** (7 problems): through LLaMA-70B hidden dim.
- **softmax** (8 problems): including 32k-128k vocabulary
  projections.
- **layernorm** (6 problems): including long sequence variants.
- **cross_entropy** (7 problems): up to LLaMA-3 128k vocab.
- **attention** (6 problems): Level 2/3, causal and non-causal.

All problems use FP16. We measure PyTorch eager as the baseline
(F.matmul → cuBLAS, F.rms_norm and F.layer_norm → fused ATen,
F.softmax, F.cross_entropy, F.scaled_dot_product_attention → FlashAttention).

### 4.2 Headline results

**NVIDIA A100-SXM4-40GB (53 problems, 4 curated configs per problem, no search iterations):**

| fast_p | Overall | Level 1 | Level 2 |
|---|---|---|---|
| fast_1.0 | 56.6% | 58.3% | 61.5% |
| fast_1.5 | 41.5% | 45.8% | 42.3% |
| fast_2.0 | 37.7% | 37.5% | 42.3% |
| fast_3.0 | 32.1% | 33.3% | 34.6% |

**NVIDIA H100:**

| fast_p | Overall | Level 1 | Level 2 |
|---|---|---|---|
| fast_1.0 | 56.6% | 58.3% | 61.5% |
| fast_1.5 | 43.4% | 45.8% | 46.2% |
| fast_2.0 | 41.5% | 45.8% | 42.3% |
| fast_3.0 | 30.2% | 33.3% | 30.8% |

### 4.3 Direct comparison to AutoKernel (H100)

Noeris and AutoKernel both run Triton kernels on H100 with PyTorch
eager baselines. We compare best-shape results per kernel:

| Kernel | **Noeris** | AutoKernel | Delta |
|---|---|---|---|
| RMSNorm | **11.66×** (mixtral, 2625 GB/s) | 5.29× | **+120%** |
| Cross-entropy | **9.65×** (long_llama, 2407 GB/s) | 2.94× | **+228%** |
| Softmax | **6.38×** (vocab_llama, 2526 GB/s) | 3.44× | **+85%** |
| LayerNorm | 1.53× (long_seq) | 3.21×* | varies |
| matmul (llama7b_qkv) | **1.01×** (tied with cuBLAS) | not reported | — |

*AutoKernel's LayerNorm 3.21× is against torch.compile; our 1.53× is
against eager. Against eager, AutoKernel reports 1.07× on one shape,
comparable to ours.

These results are with curated starter configs only, no search
iterations. They represent the starting point of the search loop, not
the end. A proper paper comparison would require running both systems
with matched compute budget and random seeds.

### 4.4 Causal attention

Our simple FlashAttention-style kernel with causal masking achieves
0.76-0.78× of PyTorch's `scaled_dot_product_attention` on LLaMA-7B
and Mistral-causal shapes. PyTorch's SDPA uses FlashAttention-2/3
under the hood, which is highly tuned — our 120-line reference
implementation is not expected to match it.

### 4.5 Cross-run learning ablation (negative result)

The core novel claim is that a persistent shape-indexed config
database accelerates kernel search. We test this with a multi-trial
ablation:

- **Protocol.** For each operator, run 3 independent trials. Within
  each trial, run 5 iterations with a persistent database (carrying
  insights across iterations) and 5 iterations with an empty
  database each time. Use the LLM proposer in both conditions.
- **Metric.** Best metric (TFLOPS or GB/s) at each iteration.
- **Hardware.** A100 via Modal.

**Results (3 trials × 5 iterations):**

| Operator | with_database | without_database | Relative |
|---|---|---|---|
| matmul | 135.52 ± 3.12 TFLOPS | 138.51 ± 1.53 TFLOPS | **-2.16%** |
| rmsnorm | 784.39 ± 6.30 GB/s | 794.33 ± 15.49 GB/s | **-1.25%** |

**Neither result is statistically significant.** Both relative
improvements are within the ~2-3% noise floor (stdev across trials).
The hypothesis — that cross-run learning accelerates convergence —
is not supported by this experiment.

**Why the negative result?** We believe the curated starter configs
are the dominant factor. Both conditions start with strong hand-picked
configurations known to work well on similar shapes, so both converge
to the same near-optimal quality within a few iterations. The database
has little room to help because the priors are already strong.

**Experimental designs that would better test the hypothesis:**

1. **Weaker priors.** Remove curated configs entirely; start only
   from the systematic parameter grid. Without curated seeds, the
   database's historical winners should be more valuable.

2. **Longer budgets.** Run 20-50 iterations per condition. Short
   budgets favor configs that are already good; long budgets favor
   approaches that steer exploration efficiently.

3. **Harder operators.** Attention and matmul have larger parameter
   spaces than rmsnorm/softmax/layernorm. The database may help more
   when random exploration is less likely to stumble onto good
   configs.

4. **Cold new shapes.** Test on shapes the database has never seen
   but which are "similar" to known winners. This would test whether
   bucket-level generalization provides useful priors.

We leave these experiments as explicit future work. The framework
supports all of them via the `ablation` and `triton-iterate` CLI
commands.

## 5. Related work

| System | Method | Cross-run | Shape-indexed | Parameterized | Operators |
|---|---|---|---|---|---|
| **Noeris (this work)** | Parameterized + LLM proposals | **Yes** | **Yes** | **Yes** | 6 |
| AutoKernel (2603.21331) | Iterative agent loop | No | No | No | 9 |
| KernelSkill (2603.10085) | Multi-agent + skill library | Skill reuse | No | No | — |
| CUDA-L1 (ICLR 2026) | Contrastive RL | Trained model | No | No | — |
| CUDA Agent (2602.24286) | Agentic RL | Trained model | No | No | — |
| KernelFoundry (2603.12440) | MAP-Elites evolutionary | Within-run | No | Template-based | — |
| GPU Kernel Scientist (2506.20807) | Evolutionary, AMD | No | No | No | — |
| Triton @autotune | Exhaustive over fixed list | Cached per shape | Per-shape cache | Fixed list | — |

The closest analog to Noeris is Triton's built-in `@autotune` decorator,
which caches per-shape winners. Unlike Triton autotune, Noeris
(a) maintains state across separate processes and CI invocations,
(b) uses LLM guidance to explore beyond a fixed config list, and
(c) supports multiple operators with operator-agnostic infrastructure.

## 6. Limitations

1. **Negative ablation.** As discussed in §4.5, we do not yet have
   empirical evidence that cross-run learning outperforms stateless
   search in the tested regime. The system's novelty claim rests on
   architecture, not measured gains.

2. **Attention kernel is simplified.** Our FlashAttention-style
   kernel does not match PyTorch's SDPA (which uses FlashAttention-2/3).
   This is a starting point for search, not a complete implementation.

3. **KernelBench subset.** We evaluate on a 53-problem curated subset
   rather than the full 250-problem KernelBench. Full-dataset numbers
   would enable direct comparison to published results from
   KernelSkill, CUDA-L1, and CUDA Agent.

4. **Single hardware family.** Results are on NVIDIA A100/H100. We
   have not tested AMD MI300 (which KernelFoundry and GPU Kernel
   Scientist target).

5. **Metric noise.** Modal's per-call variance is ~1-3% for
   memory-bound kernels, larger than some of the effects we would
   like to measure. Future work should increase per-trial
   repetitions or use statistical tests.

## 7. Conclusion

Noeris demonstrates a complete autonomous GPU kernel search pipeline
with six parameterized Triton operators, cross-run shape-indexed
configuration storage, LLM-guided proposals, and cloud GPU execution
at ≈$0.01 per iteration. On memory-bound kernels, the starting-point
configurations match or exceed AutoKernel's published H100 results.

The core novel claim — that cross-run learning accelerates search —
is not supported by our initial ablation. We interpret this as
evidence that the experimental setup (strong curated priors, 5
iterations, small parameter space per operator) does not give the
approach enough room to show value, not as evidence that the approach
is fundamentally wrong. We outline specific experimental designs that
would better test the claim and leave them as future work.

All code, raw benchmark data, and reproduction scripts are available
at https://github.com/PwnKit-Labs/noeris under an open-source license.

## A. Reproduction

Everything in this paper can be reproduced with:

```bash
git clone https://github.com/PwnKit-Labs/noeris
cd noeris
pip install -e .
pip install modal scikit-learn datasets
modal token new

# KernelBench eval (A100, ~$0.20, ~10 min)
python -m research_engine.cli kernelbench-eval --gpu A100

# KernelBench eval (H100, ~$0.40, ~8 min)
python -m research_engine.cli kernelbench-eval --gpu H100

# Ablation with the fast session runner
python -m research_engine.cli ablation --operator rmsnorm --trials 3 \
    --iterations 5 --fast

# Train the learned cost model from accumulated database
python -m research_engine.cli train-cost-model \
    --db-paths .noeris/triton-configs.json \
    --output .noeris/cost-model.pkl

# Use the cost model as a filter stage
python -m research_engine.cli triton-iterate --operator rmsnorm \
    --gpu A100 --llm --cost-model .noeris/cost-model.pkl

# Probe KernelBench HF coverage
python -m research_engine.cli kernelbench-hf-coverage --levels 1 2 3 4
```

Raw benchmark results are in `docs/results/` in both machine-readable
(JSON) and human-readable (Markdown) form.

## References

*Full reference list in follow-up version. Key papers:*

- Jaber and Jaber. AutoKernel: Autonomous GPU Kernel Optimization via
  Iterative Agent-Driven Search. arXiv:2603.21331, 2026.
- Sun et al. KernelSkill: A multi-agent framework for GPU kernel
  optimization. arXiv:2603.10085, 2026.
- Li et al. CUDA-L1: Improving CUDA optimization via contrastive
  reinforcement learning. ICLR 2026.
- Ouyang et al. KernelBench: Can LLMs write efficient GPU kernels?
  arXiv:2502.10517, 2025.
- Tillet et al. Triton: An intermediate language and compiler for
  tiled neural network computations. MAPL@PLDI 2019.
- Dao et al. FlashAttention: Fast and memory-efficient exact attention
  with IO-awareness. NeurIPS 2022.
- Williams et al. Roofline: An insightful visual performance model for
  multicore architectures. CACM 52(4), 2009.
