# Noeris

Benchmark-first autonomous ML/LLM research engine.

Noeris runs bounded experiments, measures real outcomes, and chains follow-up runs autonomously. It is not a chatbot or a summarizer. It discovers things by running them.

## What It Does

1. Discover fresh sources (arXiv, GitHub) for a standing benchmark.
2. Build structured claims, contradictions, and open questions.
3. Propose bounded hypotheses ranked by evidence strength.
4. Convert hypotheses into benchmark-shaped experiment specs.
5. Run real executors (CPU microbenchmarks, model-backed evals).
6. Persist evidence, artifacts, and frontier deltas.
7. Auto-chain follow-up runs when the frontier moves or the metric improves.

## Current Empirical Lanes

### matmul-speedup

Real CPU microbenchmark lane. Measures workload-share-weighted throughput uplift of generated kernel candidates against a naive baseline.

What works now:

- Parameterized transpose kernel generator over `(row_block, col_block, k_unroll)` grid
- Parameterized ikj kernel generator over `(row_block, j_unroll)` grid
- ~55 generated mutation candidates across two kernel families
- LLM proposer that can suggest novel kernel configurations not in the catalog
- Explicit incumbent / Pareto / challenger / novelty slotting (7 candidates per run)
- Cross-run learning: param-to-workload win correlations fed back to the proposer
- Workload-share-weighted scoring over downscaled training-like projection shapes
- Auto-chaining CI workflow with configurable continuation policies
- Scheduled twice-daily autonomous search sessions (20 chains x 3 iterations)

Current best candidate: `transpose_rowpair_dualcol` (2x2 register-blocked transpose).

**Reliability note:** Metrics vary ~8% CoV across GitHub Actions runs due to shared-tenancy CPU variance. The signal is in which candidates consistently win, not in absolute uplift numbers.

### long-context-reasoning

Model-backed eval lane via the Responses API. Measures accuracy on needle-in-haystack retrieval fixtures with baseline (truncated) vs candidate (full context) conditions.

### tool-use-reliability

Model-backed eval lane comparing terminal-first vs structured execution modes on stateful multi-step tasks.

## Repository Layout

```
src/research_engine/
  cli.py            CLI entry point
  pipeline.py       Core research cycle orchestration
  models.py         Data models (claims, hypotheses, experiment specs)
  ingestion.py      Live source discovery (arXiv, GitHub)
  llm.py            Model-backed claim extraction and hypothesis generation
  executors.py      Benchmark executors (matmul CPU, long-context, tool-use)
  benchmarks.py     Benchmark definitions and scoring
  store.py          Run persistence and history
  export.py         Run export bundles
  agenda.py         Research agenda and standing goals
  defaults.py       Seed components for offline mode
  codex_config.py   Codex CI provider mirroring
  components.py     Component protocol definitions
tests/              Regression coverage (61 tests)
docs/               Design docs, thesis, roadmap
.github/workflows/  CI and benchmark automation
```

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

# Offline (no API keys needed)
python -m research_engine.cli benchmarks
python -m research_engine.cli benchmark-run matmul-speedup

# With LLM provider configured
python -m research_engine.cli iterate matmul-speedup --iterations 3 --llm --live-execution
python -m research_engine.cli cycle --topic "long-context reasoning" --llm --max-results 3

# Inspect results
python -m research_engine.cli runs
python -m research_engine.cli history --benchmark-id matmul-speedup --limit 5
python -m research_engine.cli export-run <run_id>
```

## CI

| Workflow | Trigger | Purpose |
|---|---|---|
| CI | Push + PR | Unit tests, benchmark registry validation |
| Benchmark Iterate | Twice daily + manual | Autonomous kernel search with auto-chaining |
| Benchmark Plans | Weekly + manual | Offline benchmark planning validation |
| Codex Research | Weekly + manual | LLM-backed source discovery and research notes |

## Roadmap

### Done

- [x] Product thesis and research object schema
- [x] Live source discovery (arXiv Atom API, GitHub REST API)
- [x] Model-backed claim extraction and hypothesis generation via Responses API
- [x] Model-backed long-context and tool-use benchmark execution
- [x] Real CPU matmul microbenchmark with workload-weighted scoring
- [x] Parameterized kernel grid search (transpose + ikj families)
- [x] LLM-proposed novel kernel configurations
- [x] Cross-run learning (param-to-workload correlations)
- [x] Persisted research runs with export bundles
- [x] Auto-chaining CI with configurable continuation policies
- [x] Scheduled autonomous search sessions
- [x] Source confidence, contradiction detection, and evidence ranking
- [x] Cross-run history comparison for claims and confidence shifts
- [x] Verification gates around cycle completeness

### Next

- [ ] Reduce metric noise (pin runner type, increase repetitions, or use relative-only scoring)
- [ ] Broader and harder-to-saturate live fixture sets
- [ ] Ranked hypothesis selection beyond single-pass generation
- [ ] Richer benchmark-specific experiment templates
- [ ] Persist structured contradictions across runs
- [ ] Real training / eval runtime orchestration
- [ ] Failure replay UX

### Later

- [ ] GPU kernel benchmarks (CUDA/Triton)
- [ ] Multi-benchmark cross-pollination (insights from one lane inform another)
- [ ] Research memo publication pipeline
- [ ] External eval harness integration (lm-eval, inspect)

## Design Principles

- **Evidence-first, not chat-first.** Every claim has an artifact trail.
- **Benchmark-shaped.** If it can't be turned into a benchmark, a bounded experiment, and a required artifact contract, it's not yet a good focus area.
- **Narrow and high-signal.** Better to deeply explore a small surface than to skim many.
- **Reproducible.** Experiment specs, not just idea generation. Artifact bundles, not just summaries.
