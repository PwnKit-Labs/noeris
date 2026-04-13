# Roadmap

## Phase 0: Thesis and Scaffold (done)

- [x] lock product thesis
- [x] define research object schema
- [x] define cycle contract
- [x] establish repo skeleton

## Phase 1: Source And Planning (done)

- [x] arXiv and repo ingestion
- [x] basic deduplication and normalization
- [x] model-backed claim extraction
- [x] model-backed bounded hypothesis generation
- [x] source confidence and contradiction detection
- [x] evidence-weighted hypothesis ranking

## Phase 2: Benchmark-First Execution (done)

- [x] long-context executor (offline + model-backed)
- [x] tool-use executor (offline + model-backed)
- [x] matmul runtime path (offline + real CPU microbenchmark)
- [x] artifact capture and comparison
- [x] workload-share-weighted scoring
- [x] parameterized kernel grid search (transpose + ikj families)
- [x] LLM-proposed novel kernel configurations
- [x] cross-run learning fed back to proposer

## Phase 3: Continuous Discovery (active)

- [x] auto-chaining CI with configurable continuation policies
- [x] scheduled twice-daily autonomous search sessions
- [x] cross-run history comparison for claims and confidence shifts
- [ ] reduce metric noise (pin runner type, increase repetitions, relative-only scoring)
- [ ] broader and harder-to-saturate live fixture sets
- [ ] persist structured contradictions across runs

## Phase 4: Research Memory (planned)

- [ ] claim graph persistence across sessions
- [ ] method graph
- [ ] topic timelines and source freshness tracking
- [ ] ranked hypothesis selection beyond single-pass generation

## Phase 5: Research Output (planned)

- [ ] evidence-backed research memos with full replay
- [ ] failure replay UX
- [ ] research memo publication pipeline

## Phase 6: Scale (later)

- [ ] GPU kernel benchmarks (CUDA/Triton)
- [ ] real training / eval runtime orchestration
- [ ] multi-benchmark cross-pollination
- [ ] external eval harness integration (lm-eval, inspect)
