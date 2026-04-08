# Tool-Use Doctrine

Noeris should not assume that more structured tools automatically improve agent performance.

For some domains, especially security-style workflows, the better baseline is:

- terminal-first
- small tool surface
- strong memory
- targeted playbooks after recon

## Why This Exists

The local `pwnkit` findings point in a clear direction:

- shell access beat richer structured HTTP tools on benchmarked security tasks
- external working memory and targeted playbooks mattered more than adding more agent complexity
- adding more sub-agents or heavier prompt machinery did not automatically improve results

## Noeris Rule

For the `tool-use-reliability` benchmark:

- the default comparison should include a terminal-first baseline
- structured tools should be treated as a hypothesis, not as the presumed winner
- the artifact bundle should make the interface choice legible

## What To Measure

- task success
- unforced error rate
- recovery behavior after failure
- number and type of tool transitions
- when memory or playbooks changed the result

## Required Artifacts

- `task-suite.json`
- `terminal-transcript.jsonl`
- `tool-selection-summary.json`
- `success-summary.json`
- `error-taxonomy.md`

## Design Implication

When Noeris researches tool use, it should ask:

- should this stay terminal-first?
- is a structured tool actually reducing cognitive load?
- did memory or playbook injection matter more than interface design?

If the answer is "terminal-first still wins," that is a valid and important finding.

