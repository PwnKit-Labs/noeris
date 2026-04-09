# Noeris

Working codename for a benchmark-first ML/LLM research engine.

The name is intentionally provisional and suitable for a private repo. The repository exists to make the product thesis concrete before locking a company or product name.

## Thesis

Most "deep research" products stop at browsing and synthesis. The harder and more valuable loop is:

1. Read fresh papers, repos, benchmarks, and evals.
2. Build a structured view of claims, methods, and open questions.
3. Propose underexplored hypotheses.
4. Run bounded experiments.
5. Publish evidence-backed conclusions with full replay.

This repository is the first scaffold for that loop, starting with ML/LLMs because the data, code, and evals are public and the feedback cycles are faster than most other research domains.

## V1 Scope

- ML/LLM-first, not "research on any subject"
- Evidence-first, not chat-first
- Reproducible experiment specs, not just idea generation
- Prior-art graph and contradiction detection
- Research memos with artifact trails
- Separate from core PwnKit product, but incubatable under a PwnKit Labs umbrella

## Repository Layout

- `src/research_engine/`: minimal Python package for the core loop
- `docs/THESIS.md`: product thesis and wedge
- `docs/ARCHITECTURE.md`: first-pass system design
- `docs/ROADMAP.md`: staged roadmap
- `docs/COMPANY_PLAN.md`: company path, wedge, and org/brand recommendation
- `docs/ORG_AND_REPO.md`: where this should live organizationally
- `docs/TECHNICAL_START.md`: where to start technically and what not to do yet
- `docs/BENCHMARKS.md`: standing research goals and CI benchmark shape
- `docs/CI.md`: layered CI strategy for tests, benchmark planning, and Codex lanes
- `docs/CODEX_CI_ENV.md`: how to mirror local Codex provider config into GitHub Actions
- `docs/RESEARCH_AGENDA.md`: which LLM/ML problems Noeris should pursue first
- `docs/VERIFICATION.md`: evidence rules and publishability gates
- `docs/RESEARCH_LANDSCAPE.md`: external landscape and design implications
- `docs/NAME_SHORTLIST.md`: naming directions under review
- `tests/`: initial regression coverage for the loop scaffold

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m research_engine.cli thesis
python -m research_engine.cli architecture
python -m research_engine.cli agenda
python -m research_engine.cli benchmarks
python -m research_engine.cli ci-env
python -m research_engine.cli sources --topic "long-context reasoning"
python -m research_engine.cli cycle --topic "long-context reasoning" --llm --max-results 1
python -m research_engine.cli benchmark-run long-context-reasoning --llm --live-execution --max-results 1
python -m research_engine.cli benchmark-run matmul-speedup
python -m research_engine.cli runs
python -m research_engine.cli history --benchmark-id tool-use-reliability --limit 3
python -m research_engine.cli export-run <run_id>
python -m unittest discover -s tests
```

## What Noeris Is

Noeris is not a general "deep research" chatbot.

It is a benchmark-first research loop:

1. Discover fresh sources for a standing benchmark.
2. Build structured claims and open questions.
3. Propose bounded hypotheses.
4. Convert them into benchmark-shaped experiment specs.
5. Run an executor.
6. Persist evidence, artifacts, and next actions.

The intended product shape is:

- research planning grounded in fresh sources
- benchmark-specific execution lanes
- artifact-backed verification and replay

## Current Status

The repo has crossed from pure scaffold into an early working loop.

What is real now:

- live source discovery from arXiv and GitHub
- model-backed claim extraction and hypothesis generation via the Responses API
- model-backed long-context benchmark execution via the Responses API
- model-backed tool-use benchmark execution via the Responses API
- token and latency accounting artifacts for live benchmark runs
- real CPU benchmark execution for `matmul-speedup` via `--live-execution`
- source-confidence and contradiction structure in research memory and reports
- cross-run history comparison for claims and confidence shifts
- persisted research runs and export bundles
- verification gates around cycle completeness
- offline benchmark executors for long-context, tool-use, and matmul lanes

What is still incomplete:

- ranked hypothesis selection beyond single-pass generation
- real training / eval runtime orchestration
- stronger failure reporting and replay UX
- broader and harder-to-saturate live fixture sets

Current empirical lanes:

- `matmul-speedup`: deterministic offline systems executor with benchmark artifacts
- `matmul-speedup`: optional real CPU microbenchmark execution with artifact-backed measurements
- `long-context-reasoning`: live-source + model-backed planning, with optional live model-backed eval execution
- `tool-use-reliability`: live-source + model-backed planning, with optional live model-backed terminal-first vs structured evaluation

## Current Recommendation

- Keep this as a private repo for now.
- Incubate it as a separate project under a `PwnKit Labs` umbrella.
- Start with post-training, evaluation, and agent-system research before touching pre-training.

## Immediate Next Steps

- persist structured contradictions and source confidence
- broaden the live fixture sets so benchmark lanes are harder to saturate
- add richer benchmark-specific experiment templates and ranking
- keep the benchmark surface narrow and high-signal instead of expanding scope too early
