# Verification

## Principle

Noeris should never treat a planning artifact as empirical evidence.

The system needs an explicit verification layer between:

- cycle construction
- memo publication

That boundary is part of the product, not a cleanup detail.

## Why This Matters

Research agents fail in two common ways:

- they summarize without grounding
- they imply results that were never actually run

Noeris should be opinionated against both failure modes.

## Current V1 Gate Shape

A cycle is only publishable as evidence-backed if it has:

- at least one source
- at least one claim
- at least one hypothesis
- at least one experiment spec
- at least one recorded result
- at least one result that reflects actual execution, not just a placeholder

If those gates fail, the memo must surface the missing evidence directly.

## Verification Levels

### Level 0: Planning Only

- source collection happened
- topic memory exists
- hypotheses and experiments were generated
- nothing empirical was run

Allowed output:

- research plan
- next actions
- explicit blocker report

Not allowed output:

- claims of improvement
- claims of replication
- claims of novelty validation

### Level 1: Single Bounded Run

- one bounded experiment executed
- baseline and comparison path defined
- artifacts stored

Allowed output:

- provisional result summary
- follow-up recommendations

### Level 2: Reproducible Evidence

- execution replay possible
- artifacts intact
- comparison path intact
- result interpretable by a reviewer

Allowed output:

- evidence-backed memo
- recommendation for follow-up or rejection

## Later Gates

These do not need to exist in v1, but the architecture should leave room for them:

- source freshness checks
- citation coverage checks
- artifact integrity checks
- regression and contradiction checks
- novelty scoring confidence
- replication status

## Implementation Rule

Every memo should carry:

- `summary`
- `next_actions`
- `risks`

The `risks` field is not optional in spirit, even if it is empty in some outputs.

