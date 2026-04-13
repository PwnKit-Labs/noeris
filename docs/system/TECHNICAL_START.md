# Technical Start

## First Technical Decision

Start with post-training and evaluation research, not pre-training.

## Why Not Pre-Training First

- too capital intensive
- slower feedback loops
- harder to isolate causality
- easier to drown in infra before proving the research loop

## Why Post-Training First

- cheaper experiments
- faster iteration
- easier benchmark measurement
- easier access to open repos and methods
- better fit for an autonomous experiment loop

## Initial Focus Areas

- reasoning scaffolds
- long-context behavior
- tool use and planning
- preference optimization / data selection
- eval design and failure analysis
- retrieval and memory structures

## First Build Sequence

1. Paper and repo ingestion
2. Topic memory and claim graph
3. Hypothesis generation and ranking
4. Experiment specification
5. Evaluation harness
6. Artifact-backed memo generation

## Suggested Initial Topic Menu

- improve long-context reasoning under fixed compute
- compare synthetic data generation strategies for post-training
- identify underexplored evaluation blind spots
- improve tool-use reliability without larger base models
- find reproducible gains from memory or routing changes

## What "PwnKit LM" Should Mean

Not a pre-training project on day one.

If you use a name like `PwnKit LM`, it should initially mean:

- a research track
- a benchmark pack
- a set of reproducible experiments
- an internal proving ground for the research engine

Only later, if the loop actually yields repeatable wins, should it become:

- a tuned model line
- a dataset program
- or a true foundation-model effort

## Success Criteria For V1

- the system surfaces useful new research directions weekly
- at least some proposed experiments are worth running
- experiments are reproducible and auditable
- memos are materially better than manual literature skims
- the loop helps decide what to do next, not just summarize what happened

