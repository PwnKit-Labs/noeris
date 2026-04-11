# Noeris

Research OS for autonomous empirical discovery, currently led by a flagship GPU kernel optimization track.

**TL;DR:** Noeris is a Research OS with shared ingestion, memory, execution, and verification substrates. Its most mature track today is GPU kernel optimization: parameterized Triton kernels (6 operators), LLM-guided config search, persistent cross-run database, and A100/H100 execution via Modal for ~$0.01/iteration. It already beats AutoKernel's published H100 results on RMSNorm (11.66x vs 5.29x), softmax (6.38x vs 3.44x), and cross-entropy (9.65x vs 2.94x) using curated starter configs.

Noeris is broader than the Triton work, but the Triton/kernel track is the strongest proving ground in the repository today. The core system is meant to generalize across tracks: long-context reasoning, tool-use reliability, evaluation design, and systems optimization all share the same research substrate.

## Results

**KernelBench-style evaluation — 53 problems, 6 operators, 4 curated configs per problem (no search iterations).**

### A100-SXM4-40GB via Modal

| Threshold | Overall | Level 1 | Level 2 |
|---|---|---|---|
| **fast_1.0** (beat PyTorch) | **56.6%** | 58.3% | 61.5% |
| **fast_1.5** | 41.5% | 45.8% | 42.3% |
| **fast_2.0** (2x speedup) | **37.7%** | 37.5% | 42.3% |
| **fast_3.0** (3x speedup) | **32.1%** | 33.3% | 34.6% |

### H100 via Modal

| Threshold | Overall | Level 1 | Level 2 |
|---|---|---|---|
| **fast_1.0** | **56.6%** | 58.3% | 61.5% |
| **fast_1.5** | 43.4% | 45.8% | 46.2% |
| **fast_2.0** | **41.5%** | 45.8% | 42.3% |
| **fast_3.0** | 30.2% | 33.3% | 30.8% |

### Per-operator highlights (H100)

| Operator | Best result | vs PyTorch | Config |
|---|---|---|---|
| **rmsnorm** (mixtral) | 2625.1 GB/s | **11.66x** | `bs2048_w8_s1` |
| **rmsnorm** (llama-13b) | 2418.3 GB/s | **11.20x** | `bs4096_w16_s1` |
| **rmsnorm** (llama-7b) | 2340.2 GB/s | **11.11x** | `bs2048_w8_s1` |
| **cross_entropy** (long_llama) | 2407.1 GB/s | **9.65x** | `bs32768_w16_s1` |
| **cross_entropy** (mistral) | 2266.3 GB/s | **9.08x** | `bs8192_w16_s1` |
| **softmax** (vocab_llama) | 2526.4 GB/s | **6.38x** | `bs4096_w16_s1` |
| **softmax** (large) | 2347.5 GB/s | **5.46x** | `bs2048_w8_s1` |
| **layernorm** (long_seq) | 1666.8 GB/s | **1.53x** | `bs512_w2_s2` |
| **matmul** (llama7b_qkv) | 691.9 TFLOPS | **1.01x** | `bm128_bn256_bk64_gm8_w8_s3` |
| **attention** (llama7b) | 468.4 TFLOPS | 0.78x | `m64_n64_w4_s3` |

Full reports in [`docs/results/`](docs/results/).

Cost: **~$0.01 per iteration**, ~$0.40 for both A100 and H100 full evals (84 problems).

### Direct comparison to AutoKernel (both on H100)

| Kernel | **Noeris** | AutoKernel | Delta |
|---|---|---|---|
| RMSNorm | **11.66x** | 5.29x | **+120%** |
| Cross-entropy | **9.65x** | 2.94x | **+228%** |
| Softmax | **6.38x** | 3.44x | **+85%** |
| LayerNorm | 1.53x | 3.21x | -52% (investigate) |

We beat AutoKernel's published results on 3 of 4 memory-bound kernels **without any search iterations** — just the curated starter configs and the shape-indexed approach. LayerNorm is the one gap and a good target for the full search loop.

### Comparison to published work

| | Noeris | AutoKernel | KernelSkill (ICLR'26) |
|---|---|---|---|
| fast_1.0 (our 53 probs) | **56.6%** | — | 5.44x avg L1 |
| Operators | **6** | 9 | — |
| **Cross-run learning** | **Yes** | No | Skill retrieval |
| **Shape-indexed configs** | **Yes** | No | No |
| Cost/iteration | **$0.01** | dedicated H100 | dedicated GPU |

## What's Novel

Every existing LLM-driven kernel optimization system (AutoKernel, KernelSkill, CUDA-L1, CUDA Agent, KernelFoundry) starts fresh each run and produces one kernel per operator. Noeris is the first to combine:

1. **Parameterized kernel templates** — generate kernels from `(BLOCK_SIZE, num_warps, num_stages)` configs rather than rewriting whole files
2. **Per-shape autotuning** — different configs win different matrix shapes; track which
3. **Cross-run config database** — persist `(operator, shape_bucket, hardware) -> best params` across sessions and feed them back to the LLM proposer
4. **Continuous frontier tracking** — run overnight, chain follow-up runs when the frontier moves, accumulate discoveries

No published system does all four.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  LLM Proposer                                   │
│  (suggests novel kernel configs using           │
│   cross-run insights from config database)      │
└──────────────────────┬──────────────────────────┘
                       │ proposed configs
                       ▼
┌─────────────────────────────────────────────────┐
│  Config Selector                                │
│  incumbent / proposed / curated / exploration   │
│  slots, max N configs per run                   │
└──────────────────────┬──────────────────────────┘
                       │ selected configs
                       ▼
┌─────────────────────────────────────────────────┐
│  Kernel Generator (per operator)                │
│  matmul / rmsnorm / softmax / layernorm /       │
│  cross_entropy                                  │
└──────────────────────┬──────────────────────────┘
                       │ self-contained benchmark scripts
                       ▼
┌─────────────────────────────────────────────────┐
│  Modal GPU Executor (A100/H100)                 │
│  Batched: all configs in one cold start         │
│  5-stage correctness gate + perf benchmark      │
└──────────────────────┬──────────────────────────┘
                       │ results
                       ▼
┌─────────────────────────────────────────────────┐
│  Shape-Indexed Config Database                  │
│  (operator, shape_bucket, hardware) -> configs  │
│  Persisted as JSON across runs                  │
│  Insights extracted for next LLM proposal       │
└─────────────────────────────────────────────────┘
                       │
                       ▼
              auto-chain next run
              (if frontier moved)
```

## Supported Operators

| Operator | Param space | Shape buckets | Metric | Status |
|---|---|---|---|---|
| **matmul** | BLOCK_M/N/K, GROUP_SIZE_M, num_warps, num_stages | 10 | TFLOPS | Working |
| **rmsnorm** | BLOCK_SIZE, num_warps, num_stages | 8 | GB/s | Working |
| **softmax** | BLOCK_SIZE, num_warps, num_stages | 7 | GB/s | Working |
| **layernorm** | BLOCK_SIZE, num_warps, num_stages | 8 | GB/s | Working |
| **cross_entropy** | BLOCK_SIZE, num_warps, num_stages | 7 | GB/s | Working |
| **attention** (FlashAttention) | BLOCK_M, BLOCK_N, num_warps, stages | 7 | TFLOPS | Working |
| **rotary_emb** | BLOCK_SIZE, num_warps | — | GB/s | Planned |

## Repository Layout

```
src/research_engine/
  triton_operators.py      Common operator protocol and registry
  triton_kernels.py        Matmul kernel generator + ConfigDatabase
  triton_rmsnorm.py        RMSNorm operator
  triton_softmax.py        Softmax operator
  triton_layernorm.py      LayerNorm operator
  triton_cross_entropy.py  Cross-entropy operator
  modal_runner.py          Modal-based GPU execution backend
  kernelbench.py           KernelBench-style evaluation with fast_p
  cli.py                   CLI entry point (triton-iterate, kernelbench-eval)
  executors.py             CPU benchmark executors (proving ground)
  pipeline.py              Research cycle orchestration
  llm.py                   LLM-backed proposals and claim extraction
  ingestion.py             Live source discovery (arXiv, GitHub)
  models.py                Data models
  store.py                 Run persistence and history
  export.py                Run export bundles
  benchmarks.py            Benchmark definitions
  components.py            Component protocols
  triton_attention.py      FlashAttention-style kernel with online softmax
  ablation.py              Cross-run learning ablation study
tests/                     Regression coverage (80+ tests)
docs/                      Design docs, thesis, roadmap
.github/workflows/         CI and benchmark automation
```

See [`docs/RESEARCH_OS.md`](docs/RESEARCH_OS.md) for the substrate / track / lane / study framing.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

# Run KernelBench evaluation on an A100 via Modal
python -m research_engine.cli kernelbench-eval --gpu A100

# Search a specific operator
python -m research_engine.cli triton-iterate \
    --operator rmsnorm --gpu A100 --llm --configs-per-run 8

# CPU matmul search (no GPU needed — proving ground)
python -m research_engine.cli iterate matmul-speedup --iterations 3 --llm --live-execution
```

Requires:
- Modal account (`pip install modal && modal token new`)
- Azure OpenAI or OpenAI credentials for the LLM proposer (optional but powerful)

## CI

| Workflow | Trigger | Purpose |
|---|---|---|
| CI | Push + PR | Unit tests (80+ tests), benchmark registry validation |
| Triton Iterate | Twice daily + manual | Autonomous GPU kernel search with auto-chaining |
| Benchmark Iterate | Twice daily + manual | CPU matmul search (proving ground) |
| Benchmark Plans | Weekly + manual | Offline benchmark planning validation |

## Roadmap

### Done

- [x] Autonomous CPU matmul search loop with code generation (proving ground)
- [x] Compiled code generation for parameterized kernels
- [x] LLM-proposed novel configs that actually win
- [x] Parameterized Triton matmul kernel
- [x] Shape-indexed ConfigDatabase with cross-run persistence
- [x] Operator-aware database (multi-operator keying)
- [x] Modal GPU execution backend (A100/H100)
- [x] Six parameterized operators: matmul, rmsnorm, softmax, layernorm, cross_entropy, attention
- [x] FlashAttention kernel with tiled online softmax
- [x] KernelBench-style evaluation with fast_p scoring (53 problems)
- [x] LLM proposer with cross-run insights
- [x] Auto-chaining CI with scheduled runs
- [x] Multi-operator CI matrix (6 operators in parallel)
- [x] Generalized config selection for all operators
- [x] Cross-run learning ablation framework
- [x] H100 evaluation and direct comparison to AutoKernel
- [x] Verified end-to-end on real A100 and H100 — beating published results on memory-bound kernels

### Next

- [ ] Run the ablation study at scale to validate cross-run learning claim
- [ ] Improve LayerNorm kernel (our only operator losing to AutoKernel)
- [ ] Causal masking in attention kernel
- [ ] Rotary embedding kernel
- [ ] Full KernelBench (250 problems) HuggingFace integration
- [ ] Hardware cross-learning (configs learned on A100 applied to H100)
- [ ] Research paper draft (arXiv submission)

## Related Work

| System | Method | Cross-run | Shape-indexed | Parameterized | Operators |
|---|---|---|---|---|---|
| **Noeris** | Parameterized + LLM proposals | **Yes** | **Yes** | **Yes** | 5 |
| AutoKernel (2603.21331) | Iterative agent loop | No | No | No | 9 |
| KernelSkill (2603.10085) | Multi-agent + skill library | Skill reuse | No | No | — |
| CUDA-L1 (ICLR'26) | Contrastive RL | Trained model | No | No | — |
| CUDA Agent (2602.24286) | Agentic RL | Trained model | No | No | — |
| KernelFoundry (2603.12440) | MAP-Elites evolutionary | Within-run | No | Template-based | — |
| Triton autotune | Exhaustive over fixed list | Cached per shape | Per-shape | Fixed list | — |

## Design Principles

- **Evidence-first.** Every claim has an artifact trail. No vibes.
- **Shape-aware.** Different workloads have different optimal configs. Track which.
- **Cross-run.** Discoveries persist and compound. The system gets smarter over time.
- **Reproducible.** Parameterized configs, not monolithic rewrites. Artifact bundles, not summaries.
- **Cheap.** ~$0.01 per iteration on Modal. ~$3/week for scheduled runs.
