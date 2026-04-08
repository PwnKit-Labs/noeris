# CI

Noeris should use layered CI.

## Layer 1: Cheap Deterministic CI

Always run:

- unit tests
- schema and serialization tests
- benchmark registry validation

This is the `ci.yml` workflow.

## Layer 2: Benchmark Planning Lanes

Run on schedule or manually:

- benchmark-aware planning cycles
- persisted run artifacts
- replayable summaries

This is the `benchmark-plans.yml` workflow.

These runs should stay deterministic and cheap. They prove that Noeris can still:

- build a benchmark-specific topic
- persist the run
- export artifact bundles
- execute the current offline benchmark lanes for long-context and tool-use

## Layer 3: Codex Research Lanes

Run manually or on a separate schedule:

- source discovery
- Codex-written benchmark research note
- artifact upload for review

This is the `codex-benchmark-research.yml` workflow.

It should be treated as optional and secret-gated.

## What Should Not Run In Default CI

- expensive empirical benchmark runs
- GPU-heavy experiments
- long-running search loops
- flaky internet-dependent research execution

Those belong in separate scheduled or manually triggered lanes.

## Practical Rule

Default CI proves the system contract.

Scheduled benchmark lanes prove the planning loop.

Manual or scheduled expensive lanes prove real research progress.
