# Noeris

Autonomous kernel optimization engine with cross-run learning and shape-indexed config discovery.

Noeris generates parameterized GPU kernel configurations, benchmarks them across workload shapes, persists winning configs in a shape-indexed database, and feeds cross-run insights back to an LLM proposer for the next search iteration. It runs continuously via auto-chaining CI.

## What's Novel

Every existing LLM-driven kernel optimization system (AutoKernel, KernelSkill, CUDA-L1, CUDA Agent) starts fresh each run and produces one kernel per operator. Noeris is the first to combine:

1. **Parameterized kernel templates** — generate kernels from parameter configs rather than rewriting whole files
2. **Per-shape autotuning** — different configs win different matrix shapes; track which
3. **Cross-run config database** — persist `(op, shape, hardware) -> best params` across sessions and feed them back to the LLM proposer
4. **Continuous frontier tracking** — run overnight, chain follow-up runs when the frontier moves, accumulate discoveries

See [Related Work](#related-work) for how this positions against the field.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  LLM Proposer                                   │
│  (suggests novel kernel configs using            │
│   cross-run insights from config database)       │
└──────────────────────┬──────────────────────────┘
                       │ proposed configs
                       ▼
┌─────────────────────────────────────────────────┐
│  Config Selector                                 │
│  incumbent / proposed / curated / exploration    │
│  slots, max 8 configs per run                    │
└──────────────────────┬──────────────────────────┘
                       │ selected configs
                       ▼
┌─────────────────────────────────────────────────┐
│  Kernel Generator                                │
│  generates Triton kernel source per config       │
│  (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M,      │
│   num_warps, num_stages)                         │
└──────────────────────┬──────────────────────────┘
                       │ benchmark scripts
                       ▼
┌─────────────────────────────────────────────────┐
│  GPU Executor (Modal / self-hosted)              │
│  5-stage correctness gate + perf benchmark       │
│  across shape buckets                            │
└──────────────────────┬──────────────────────────┘
                       │ results
                       ▼
┌─────────────────────────────────────────────────┐
│  Config Database                                 │
│  (shape, hardware) -> ranked configs             │
│  persisted across runs as JSON                   │
│  insights extracted for next LLM proposal        │
└─────────────────────────────────────────────────┘
                       │
                       ▼
              auto-chain next run
              (if frontier moved)
```

## Current Status

### Proven (CPU matmul lane)

The autonomous search loop has been validated on pure-Python matmul as a proving ground:

- Parameterized code generation with compiled specialized kernels
- Grid search discovered `transpose_r2_c4_k1` which **dethroned the hand-written incumbent** on every workload
- LLM proposed a novel `(2,3,1)` config that won workloads — genuine LLM-guided discovery
- Auto-chaining CI ran 18+ iterations autonomously, frontier tracking across runs
- Cross-run learning fed param-to-workload correlations back to the proposer

### In Progress (Triton GPU lane)

Porting the same infrastructure to real GPU kernels via Triton:

- Parameterized Triton matmul kernel generator (BLOCK_M/N/K, GROUP_SIZE_M, num_warps, num_stages)
- Shape-indexed config database with cross-run persistence
- 12 curated configs + systematic grid generation (~200 configs)
- 10 shape buckets covering transformer workloads (QKV, MLP, attention)
- 5-stage correctness harness (smoke, shape sweep, stability, determinism, edge cases)
- Modal-based GPU execution backend (A100, ~$18/night for 300 experiments)

## Repository Layout

```
src/research_engine/
  triton_kernels.py   Triton kernel generator + shape-indexed config database
  executors.py        CPU benchmark executors (matmul, long-context, tool-use)
  cli.py              CLI entry point
  pipeline.py         Core research cycle orchestration
  llm.py              LLM-backed claim extraction and hypothesis generation
  ingestion.py        Live source discovery (arXiv, GitHub)
  models.py           Data models
  store.py            Run persistence and history
  export.py           Run export bundles
  benchmarks.py       Benchmark definitions
  components.py       Component protocol definitions
tests/                Regression coverage (80 tests)
docs/                 Design docs, thesis, roadmap
.github/workflows/    CI and benchmark automation
```

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

# CPU matmul search (no GPU needed)
python -m research_engine.cli iterate matmul-speedup --iterations 3 --llm --live-execution

# Inspect results
python -m research_engine.cli runs
python -m research_engine.cli history --benchmark-id matmul-speedup --limit 5
```

## CI

| Workflow | Trigger | Purpose |
|---|---|---|
| CI | Push + PR | Unit tests (80 tests), benchmark registry validation |
| Benchmark Iterate | Twice daily + manual | Autonomous kernel search with auto-chaining |
| Benchmark Plans | Weekly + manual | Offline benchmark planning validation |
| Codex Research | Weekly + manual | LLM-backed source discovery and research notes |

## Related Work

| System | Method | Cross-run learning | Shape-indexed | Parameterized templates |
|---|---|---|---|---|
| **Noeris** | Parameterized grid + LLM proposals | **Yes** | **Yes** | **Yes** |
| AutoKernel (2603.21331) | Iterative agent loop | No | No | No |
| KernelSkill (2603.10085) | Multi-agent + skill library | Skill reuse | No | No |
| CUDA-L1 (ICLR 2026) | Contrastive RL | Trained model | No | No |
| CUDA Agent (2602.24286) | Agentic RL | Trained model | No | No |
| KernelFoundry (2603.12440) | MAP-Elites evolutionary | Within-run only | No | Template-based |
| Triton autotune | Exhaustive over fixed list | Cached per shape | Per-shape cache | Fixed config list |

The key gap: no published system maintains a persistent database of `(op, shape, hardware) -> (optimal config)` that accumulates across separate search sessions and feeds back to the proposer. Triton's built-in autotune caches per-shape winners but doesn't learn across sessions or use LLM-guided exploration.

## Roadmap

### Done

- [x] Autonomous CPU matmul search loop with code generation
- [x] LLM-proposed novel kernel configurations (verified: LLM-proposed configs won workloads)
- [x] Generated candidate dethroned hand-written incumbent
- [x] Cross-run learning fed back to proposer
- [x] Auto-chaining CI with configurable continuation policies
- [x] Scheduled twice-daily autonomous search sessions

### In Progress

- [ ] Triton matmul kernel generator with parameterized configs
- [ ] Shape-indexed config database with cross-run persistence
- [ ] Modal GPU execution backend for CI
- [ ] 5-stage correctness harness for Triton kernels
- [ ] KernelBench integration for comparable evaluation

### Next

- [ ] Multi-kernel support (softmax, layernorm, RMSNorm, attention)
- [ ] Amdahl's law orchestration across kernel types
- [ ] Hardware-specific config learning (A100 vs H100 vs MI300)
- [ ] Research memo publication with full artifact trails

## Design Principles

- **Evidence-first.** Every claim has an artifact trail. No vibes.
- **Shape-aware.** Different workloads have different optimal configs. Track which.
- **Cross-run.** Discoveries persist and compound. The system gets smarter over time.
- **Reproducible.** Parameterized configs, not monolithic rewrites. Artifact bundles, not summaries.
