You are producing a benchmark-specific research note for Noeris.

Context:
- Benchmark id is in the `NOERIS_BENCHMARK_ID` environment variable.
- `codex-sources.json` contains current source discovery output from arXiv and GitHub.

Task:
1. Summarize the most relevant sources for the benchmark.
2. Propose 3 bounded hypotheses worth testing next.
3. For each hypothesis, define:
   - expected mechanism
   - minimal experiment
   - baseline to compare against
   - artifact(s) required to treat the result as evidence-backed
4. End with a short section called `CI Recommendation` that says whether this benchmark should run in:
   - cheap deterministic CI
   - scheduled benchmark lane
   - manual expensive lane

Constraints:
- Be explicit about uncertainty.
- Do not claim any experiment has been run.
- Prefer concrete testable ideas over broad visions.
