# Company Plan

## Positioning

Build a company around autonomous empirical discovery.

The first product is not "research on anything." It is an autonomous ML/LLM research engine that:

- reads fresh papers, repos, benchmarks, and evals
- maps claims and prior art
- proposes underexplored hypotheses
- runs bounded experiments
- produces evidence-backed research memos and artifacts

## Relationship to PwnKit

Recommended structure:

- `PwnKit`: security product and security research brand
- `PwnKit Labs`: umbrella for internal R&D and advanced agent systems
- `Noeris`: separate private repo and product incubation track inside that umbrella

This is better than putting the work inside core PwnKit because:

- the product thesis is broader than security
- the technical roadmap is different from a security copilot or scanner
- the eventual company/brand may need to stand on its own
- you still keep founder, infrastructure, and distribution leverage

## What To Avoid

- Do not pitch it as a general "research framework" on day one.
- Do not pitch it as a generic deep-research assistant.
- Do not try to cover all sciences at launch.
- Do not start with pre-training infrastructure.

## Wedge

Start with ML/LLMs.

Why this wedge:

- public papers and repos
- fast feedback loops
- benchmarked evaluation
- executable experiments
- direct path to showing useful output

## Initial Customer / User

The first user is you.

The second user is a small, technical research team that wants help answering:

- what changed in the literature this week
- what ideas are underexplored
- what experiments are worth running next
- what claims fail to replicate

## Product Promise

Turn fresh literature and code into:

- ranked hypotheses
- bounded experiments
- artifact-backed research memos
- a living map of what seems true, false, or unclear

## 90-Day Path

### Phase 1

- ingest arXiv papers and relevant GitHub repos
- normalize metadata and basic summaries
- define claim, method, and experiment objects

### Phase 2

- build prior-art and contradiction graph
- generate research questions and candidate hypotheses
- rank by novelty, expected signal, and execution cost

### Phase 3

- scaffold experiment templates
- connect to reproducible runs
- emit research memos with artifacts and replay

## 12-Month Path

- continuous topic monitoring
- richer experiment execution
- self-improving research memory
- selective human review workflows
- first external design partners

## Repo Guidance

- Keep this repo private until the loop is meaningfully differentiated.
- Open-source selected components later, likely around schemas, memory, or replay.
- Keep the company-brand decision separate from the repo codename decision.

