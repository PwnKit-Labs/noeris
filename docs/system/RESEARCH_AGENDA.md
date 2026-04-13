# Research Agenda

Noeris should not chase "improve LLMs" in the abstract.

It should maintain an explicit agenda of problems that are:

- empirically testable
- small-team-tractable
- connected to benchmarkable outcomes
- useful for future products

## Tier 1: Best First Areas

### Long-Context Reasoning

Why:

- clear failure surface
- public evals
- easy fit for post-training and evaluation work

Mode:

- post-training
- retrieval/memory systems
- eval design

### Tool-Use Reliability

Why:

- directly useful for agent products
- measurable through task success and error rates
- benefits from both system design and post-training
- likely benefits from terminal-first baselines and small tool surfaces before adding more structure

Mode:

- post-training
- systems
- evals

### Evaluation Design

Why:

- weak evals create fake progress
- the engine needs this to tell whether any other gain is real

Mode:

- evals
- research infrastructure

### Memory And Retrieval Structures

Why:

- many "reasoning" failures are really context selection failures
- connects naturally to long-context and agent benchmarks

Mode:

- systems
- post-training
- evals

## Tier 2: Strong Secondary Areas

### Post-Training Data Selection

Why:

- high leverage
- tractable without huge compute

Mode:

- post-training

### Inference Efficiency

Why:

- every cost or latency reduction raises the practical ceiling elsewhere

Mode:

- systems

### Kernel And Matmul Speedups

Why:

- concrete proving ground
- strong literature-to-execution benchmark

Mode:

- systems

### Failure Analysis And Error Taxonomy

Why:

- needed to decide what the engine should fix next

Mode:

- evals
- research infrastructure

## Tier 3: Important But Not V1

### Base-Model Pretraining And Scaling

Why not now:

- too capital intensive
- slower loops
- higher chance of infra distraction

Noeris should only move here after it proves value on faster, cheaper loops.

## Product Rule

If a research area cannot be turned into:

- a benchmark
- a bounded experiment
- a required artifact contract

then it is not yet a good Noeris focus area.
