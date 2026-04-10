# Research Memory

## Goal

Noeris should not treat research as a pile of notes or one giant prompt.

For long-running benchmark work, it needs explicit memory that survives across runs, preserves evidence quality, and makes contradictions visible.

## Recommended Shape

Use a hybrid memory model:

1. Episodic layer
   - each run
   - each benchmark result
   - each failed candidate
   - cost / latency / artifact summaries

2. Semantic layer
   - claims
   - source assessments
   - contradiction records
   - stable benchmark facts
   - intervention patterns

3. Link layer
   - claim -> source
   - hypothesis -> supporting claims
   - experiment -> benchmark
   - result -> candidate / baseline
   - contradiction -> affected claims

## What To Store

Minimum graph entities:

- `Source`
- `Claim`
- `Hypothesis`
- `Experiment`
- `Result`
- `Contradiction`
- `Run`

Minimum edge types:

- `supports`
- `contradicts`
- `derived_from`
- `tested_by`
- `supersedes`
- `same_family_as`

## Why This Shape

This gives Noeris:

- cross-run memory instead of prompt-only memory
- explicit evidence quality instead of flat text
- contradiction-aware ranking instead of novelty theater
- candidate lineage for long searches like matmul

## Matmul Implication

The matmul lane will need many candidate attempts.

That means memory must support:

- candidate family tracking
- baseline lineage
- failure clustering
- shape-specific performance deltas
- superseded candidate records

Without that structure, repeated candidate generation will just rediscover the same weak ideas.

## Near-Term Build Order

1. Keep the current run-history surface.
2. Add stable IDs for claims and hypotheses.
3. Add candidate lineage for matmul search.
4. Add source freshness timestamps and confidence change tracking.
5. Add contradiction-aware reranking across prior runs, not just inside one run.

## Non-Goal

Do not jump straight to a heavyweight graph database.

The first useful version can remain file-backed and benchmark-scoped as long as:

- entities are explicit
- links are explicit
- history is queryable
- contradictions are preserved
