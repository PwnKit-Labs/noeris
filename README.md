# Noeris

Working codename for an autonomous ML/LLM research engine.

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
python3 -m pip install -e .
python3 -m research_engine.cli thesis
python3 -m research_engine.cli architecture
python3 -m research_engine.cli agenda
python3 -m research_engine.cli benchmarks
python3 -m research_engine.cli ci-env
python3 -m research_engine.cli sources --topic "long-context reasoning"
python3 -m research_engine.cli run --topic "improve long-context reasoning"
python3 -m research_engine.cli benchmark-run matmul-speedup
python3 -m research_engine.cli runs
python3 -m research_engine.cli export-run <run_id>
python3 -m research_engine.cli cycle --topic "improve long-context reasoning"
python3 -m unittest discover -s tests
```

## Current Status

This is a scaffold, not the finished product. The current package defines explicit seams so later work can plug in:

- literature/repo ingestion
- research memory and claim graph construction
- hypothesis ranking
- experiment generation
- execution backends
- verification and reporting
- verification gates between cycle construction and memo publication
- persisted research runs
- standing benchmark goals

Current empirical lanes:

- `matmul-speedup`: deterministic offline systems executor with benchmark artifacts
- `long-context-reasoning`: deterministic offline executor with eval artifacts
- `tool-use-reliability`: deterministic offline executor comparing terminal-first against structured-tool policy

## Current Recommendation

- Keep this as a private repo for now.
- Incubate it as a separate project under a `PwnKit Labs` umbrella.
- Start with post-training, evaluation, and agent-system research before touching pre-training.
