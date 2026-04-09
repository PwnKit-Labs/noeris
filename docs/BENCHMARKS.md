# Benchmarks

Noeris should have standing research goals, not just ad hoc topic prompts.

The point of these benchmarks is to force the system to:

- read current research
- propose concrete interventions
- define measurable experiments
- produce real evidence or admit it has not yet done so

## Benchmark Policy

Each benchmark should answer three questions:

1. What intervention is being proposed?
2. What baseline is it compared against?
3. What artifact bundle is required before we treat the result as evidence-backed?

If a benchmark cannot answer those questions, it is not ready to be a standing lane.

## Initial Benchmark Tracks

### Matrix Multiplication Speedup

Goal:

- discover and validate techniques that improve matrix multiplication or nearby kernel performance

Why it belongs here:

- it is concrete
- it is measurable
- it forces low-level systems thinking rather than vague idea generation
- it tests whether Noeris can move from literature to execution

Success bar:

- clear hardware and precision assumptions
- explicit baseline
- reproducible measurement
- real speedup or a justified rejection

Required artifacts:

- `hardware-profile.json`
- `benchmark-config.json`
- `raw-timing-results.json`
- `baseline-comparison.md`

CI lane:

- manual expensive lane

Current offline executor:

- deterministic synthetic systems lane with hardware-profile and timing artifacts

### Long-Context Reasoning

Goal:

- improve reasoning quality over long contexts under bounded cost

Current state:

- live-source discovery works
- model-backed planning works
- live model-backed execution now exists for the small replay harness
- the next upgrade is broadening the fixture set and cost accounting

Required artifacts:

- `eval-manifest.json`
- `baseline-metrics.json`
- `candidate-metrics.json`
- `failure-analysis.md`

CI lane:

- scheduled benchmark lane

### Tool-Use Reliability

Goal:

- reduce unforced errors in multi-step tool-using agent tasks

Current state:

- live-source discovery works
- model-backed planning works
- live model-backed execution now exists for the small replay harness
- the next upgrade is richer task fixtures and explicit cost/latency reporting

Tool-use baseline:

- compare terminal-first / bash-first execution against a more structured tool policy

Why:

- `pwnkit`'s own benchmark notes favored shell access over richer structured HTTP tools for security-style tasks
- the model already knows shell workflows, `curl`, pipes, and small scripts
- the more useful additions were memory and targeted playbooks, not a larger structured tool surface

Required artifacts:

- `task-suite.json`
- `terminal-transcript.jsonl`
- `tool-selection-summary.json`
- `success-summary.json`
- `error-taxonomy.md`

CI lane:

- scheduled benchmark lane

## Current Benchmark Surface

Today the intended split is:

- planning can be live and model-backed
- execution can be either offline deterministic or live benchmark-specific

That split is deliberate. It lets Noeris become useful before every executor is production-grade.

## CI Role

These should eventually become part of CI in layers:

1. schema and pipeline tests
2. source ingestion tests
3. benchmark-spec generation tests
4. optional offline replay tests
5. optional expensive empirical benchmark runs outside default CI

Default CI should stay cheap and deterministic.

Empirical benchmark runs should be scheduled separately so they do not turn the repo into a flaky infra mess.
