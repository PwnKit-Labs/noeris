# Benchmarks

Noeris should have standing research goals, not just ad hoc topic prompts.

The point of these benchmarks is to force the system to:

- read current research
- propose concrete interventions
- define measurable experiments
- produce real evidence or admit it has not yet done so

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

### Long-Context Reasoning

Goal:

- improve reasoning quality over long contexts under bounded cost

### Tool-Use Reliability

Goal:

- reduce unforced errors in multi-step tool-using agent tasks

## CI Role

These should eventually become part of CI in layers:

1. schema and pipeline tests
2. source ingestion tests
3. benchmark-spec generation tests
4. optional offline replay tests
5. optional expensive empirical benchmark runs outside default CI

Default CI should stay cheap and deterministic.

Empirical benchmark runs should be scheduled separately so they do not turn the repo into a flaky infra mess.

