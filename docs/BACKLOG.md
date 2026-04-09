# Backlog

This backlog is intentionally benchmark-first.

The goal is not to become a vague autonomous scientist. The goal is to make a few standing research lanes genuinely useful, reproducible, and evidence-backed.

## Current Priority

### P0: Expand Live Benchmark Execution

- Expand the live model-backed long-context and tool-use executors beyond the current small replay harnesses.
- Keep fixture sets deterministic enough to replay while making them harder to saturate.
- Capture:
  - prompt/config manifest
  - baseline outputs
  - candidate outputs
  - scored metrics
  - failure analysis
  - richer cost and latency accounting, including CI summaries and pricing inputs

### P0: Stabilize Repo-Owned LLM CI Lane

- Keep `benchmark-run --llm --live-execution` as the canonical GitHub workflow path for live lanes.
- Upload run JSON, markdown report, and artifact bundle for every benchmark run.
- Track and surface cost / latency in CI summaries.

### P0: Executor Selection Contract

- Add explicit flags for:
  - model-backed planning
  - model-backed execution
  - offline deterministic replay
- Fail fast when the requested executor is not configured.

## Near-Term

### P1: Tool-Use Evaluator Upgrade

- Expand the live tool-use evaluator beyond the current small replay harness.
- Keep terminal-first as the baseline.
- Measure:
  - task success
  - unforced errors
  - recovery rate
  - artifact completeness
  - cost and latency

### P1: Research Memory Upgrade

- Persist contradictions, not just claims.
- Track source freshness and confidence.
- Distinguish:
  - direct evidence
  - weak evidence
  - speculative hypothesis support

### P1: Better Experiment Specs

- Rank hypotheses instead of passing through all generated ones.
- Add per-benchmark experiment templates.
- Attach cost ceilings and runtime assumptions to each spec.

### P1: Benchmark Metadata And Labels

- Keep GitHub labels aligned with benchmark lanes and infra work.
- Add issue triage conventions for benchmark, executor, llm, ci, and research-memory work.

## Later

### P2: Real Matmul Runtime

- Move from synthetic timing fixtures to a genuine measurement harness.
- Keep it separate from default CI.
- Require hardware/profile metadata on every run.

### P2: Review And Replay Surface

- Add run diffing.
- Add artifact comparison.
- Add "why this hypothesis was chosen" replay traces.

### P2: Continuous Discovery

- Daily or weekly benchmark digests.
- Auto-proposed follow-up hypotheses from prior failures.

## Non-Goals For Now

- General-purpose web research assistant
- Multi-domain "research anything" product
- Pretraining infrastructure
- Broad multi-agent orchestration inside the product runtime
