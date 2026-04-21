# Spec Decode Verify+Accept Plan

Issue: `#84` — speculative decoding verify+accept fused kernel.

## Baseline artifact

Use this command to generate the current non-fused baseline on Modal:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/spec_decode_verify_accept_baseline.py
```

Outputs:

- `docs/results/spec-decode-verify-accept-baseline.json`
- `docs/results/spec-decode-verify-accept-baseline.md`

## Current measured path

Baseline operation chain:

1. `argmax(target_logits)`
2. `target_tokens == draft_tokens`
3. first mismatch index / accept length
4. accepted-prefix mask generation

This reflects a realistic verify+accept control path without a fused custom kernel.

## Next kernel step

- Implement a Triton kernel that fuses:
  - token match computation
  - first mismatch detection
  - accepted-prefix mask write
- Keep logits argmax outside the first version unless we can prove register pressure remains manageable.

## Acceptance criteria for first fused iteration

- Correctness parity with baseline for all tested shapes.
- End-to-end verify+accept latency win on at least one Hopper shape bucket.
- Reproducible artifact in `docs/results/` with A100/H100 table.
