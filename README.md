# Noeris

Autonomous GPU kernel optimization engine with cross-run learning and shape-indexed config discovery.

Noeris generates parameterized Triton kernels, benchmarks them across workload shapes on real GPUs via Modal, persists winning configs in a shape-indexed database, and feeds cross-run insights back to an LLM proposer for the next search iteration. It runs continuously via auto-chaining CI.

## Results (A100-SXM4-40GB via Modal)

**KernelBench-style evaluation — 20 problems, 5 operators, only curated starter configs (no search):**

| Threshold | Overall | Level 1 | Level 2 |
|---|---|---|---|
| **fast_1.0** (beat PyTorch) | **60.0%** | 50.0% | 75.0% |
| **fast_2.0** (2x speedup) | **40.0%** | 33.3% | 50.0% |
| **fast_3.0** (3x speedup) | **35.0%** | 33.3% | 37.5% |

### Per-operator highlights

| Operator | Best result | vs PyTorch | Config |
|---|---|---|---|
| **cross_entropy** (llama long) | 1242.9 GB/s | **10.15x** | `bs32768_w16_s1` |
| **rmsnorm** (llama-7b) | 1162.2 GB/s | **10.07x** | `bs2048_w8_s1` |
| **softmax** (vocab) | 1287.3 GB/s | **6.68x** | `bs4096_w16_s1` |
| **layernorm** (gpt) | 927.1 GB/s | **1.32x** | `bs1024_w4_s1` |
| **matmul** (xlarge) | 237.4 TFLOPS | 0.89x | `bm128_bn128_bk32_gm8_w4_s4` |

Cost: **~$0.01 per iteration**, ~$0.15 for the full KernelBench eval.

### Comparison to published work

| | Noeris (A100) | AutoKernel (H100) | KernelSkill (ICLR'26) |
|---|---|---|---|
| RMSNorm speedup | **10.07x** | 5.29x | — |
| Softmax speedup | **6.68x** | 2.82x / 3.44x | — |
| Cross-entropy | **10.15x** | 2.21x / 2.94x | — |
| fast_1.0 | **60.0%** | — | 5.44x avg L1 |
| Operators supported | **5** | 9 | — |
| **Cross-run learning** | **Yes** | No | Skill retrieval |
| **Shape-indexed configs** | **Yes** | No | No |
| Cost/iteration | **$0.01** | dedicated H100 | dedicated GPU |

We match or exceed the published memory-bound kernel speedups **without any search iterations** — these are just the curated starter configs. The full loop with LLM-guided search across the parameter grid improves further.

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
| **attention** (FlashAttention) | BLOCK_M, BLOCK_N, num_warps, stages | — | TFLOPS | Planned |
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
tests/                     Regression coverage (80+ tests)
docs/                      Design docs, thesis, roadmap
.github/workflows/         CI and benchmark automation
```

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
- [x] RMSNorm, Softmax, LayerNorm, Cross-entropy operators
- [x] KernelBench-style evaluation with fast_p scoring
- [x] LLM proposer with cross-run insights
- [x] Auto-chaining CI with scheduled runs
- [x] Verified end-to-end on real A100 — 60% fast_1.0, 10x best speedup

### In progress

- [ ] Generalized config selection across all operators
- [ ] Multi-operator CI workflow (nightly runs over all ops)
- [ ] LLM proposer fine-tuning for memory-bound operators
- [ ] Cross-run learning ablation study

### Next

- [ ] FlashAttention-style attention kernel
- [ ] Rotary embedding kernel
- [ ] H100 evaluation (compare to AutoKernel directly)
- [ ] Full KernelBench (250 problems) integration
- [ ] Hardware cross-learning (configs learned on A100 applied to H100)
- [ ] Research memo publication pipeline

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
