# Architecture

## V1 Principle

Noeris should be built as an evidence pipeline, not a chat agent with tools glued on.

The smallest credible system is:

- deterministic enough to replay
- modular enough to swap components
- strict about verification before publishing memos

## Core Loop

1. Ingest new papers, repos, benchmarks, and evals.
2. Normalize them into structured research objects.
3. Build topic-local research memory with claims, evidence refs, open questions, and contradictions.
4. Rank candidate hypotheses.
5. Generate bounded experiment specs.
6. Execute experiments in reproducible runtimes.
7. Verify whether the cycle produced empirical evidence or only planning artifacts.
8. Publish research memos with artifact references and explicit risks.

## Component Boundaries

- `source_provider`
  collects papers, repos, benchmarks, and other research inputs
- `research_memory`
  derives structured claims and later becomes the claim/method graph layer
- `hypothesis_planner`
  proposes candidate ideas from the topic and current memory
- `experiment_planner`
  converts ranked hypotheses into bounded experiment specs
- `experiment_executor`
  runs or simulates experiments and captures outcomes
- `verifier`
  blocks publication if evidence is missing or the cycle is incomplete
- `memo_writer`
  turns the cycle into a human-readable report with machine-usable structure

## State Model

The core state unit is a `ResearchCycle`.

It should include:

- topic
- research context
- hypotheses
- experiment specs
- experiment results

This is the minimum state needed to:

- replay a cycle
- inspect where a claim came from
- compare planned vs executed work
- decide whether a memo is publishable

## Verification Gates

Every cycle should pass explicit gates before it is treated as evidence-backed.

Current gate shape:

- sources present
- claims present
- hypotheses present
- experiments present
- results recorded
- empirical execution attached

Later gate shape should add:

- source freshness
- citation coverage
- experiment artifact integrity
- baseline comparison completeness
- contradiction and regression checks
- novelty scoring confidence

## Execution Model

V1 should be synchronous and local-first.

That means:

- no distributed orchestration yet
- no queueing system yet
- no long-running multi-agent runtime yet
- no autonomous background loops until the single-cycle contract is solid

## What To Defer

- generalized multi-domain research
- complex multi-agent society simulations
- pre-training infrastructure
- hosted collaboration layer
- heavy graph database choices
- autonomous self-modifying planner loops

## Initial Build Order

1. stable research object schema
2. explicit component interfaces
3. single-topic research cycle with verification gates
4. arXiv and GitHub ingestion
5. artifact-backed experiment result format
6. first claim/method memory layer
7. continuous topic monitoring
