# Research Memory

Noeris should preserve more than raw source lists.

The minimum useful research-memory unit is:

- source
- assessment
- claim
- contradiction
- hypothesis linkage

## Why This Matters

Without explicit lineage, the system can say:

- "this source mattered"
- "this claim mattered"

but not:

- which source produced which claim
- which claims actually supported a hypothesis
- which claims stayed unsupported across runs

## Current Artifact

Every exported run bundle should include:

- `claim-lineage.json`

That file should make source-to-claim-to-hypothesis linkage visible without requiring a reader to dig through nested run JSON manually.

## Current State

The current seed memory layer now derives simple claims directly from the discovered sources.

That is still shallow, but it is better than a single generic placeholder claim because it gives every track:

- per-source evidence refs
- source confidence notes
- a basic lineage artifact

## Next Step

The next real upgrade is richer live claim extraction:

- better claim synthesis from paper abstracts and repo descriptions
- contradiction detection across multiple sources
- persistent cross-run memory updates

