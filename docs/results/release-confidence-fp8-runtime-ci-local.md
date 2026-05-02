# Release Confidence: FP8 Runtime + CI Local

Generated: 2026-05-02T20:02:10+00:00

## Commit Span

- `df76e33` Integrate FP8 token-loop runtime cache path and local CI runner
- `5eede0d` Skip GPU integration tests when CUDA stack missing
- `76fefaf` Add noeris patch package and remaining benchmark tooling
- `d881bfc` Add accumulated Gemma and KernelBench result artifacts
- `d4429b4` Gate history regressions with tunable thresholds
- `eaa231a` Export history before regression CI gate

## Validation Results

- Unit tests (via `PYTHON_BIN=python3.11 ./scripts/ci_local.sh`): `853 passed`, `23 skipped`, `23 subtests passed`
- Public artifact refs: pass (`scripts/check_public_claim_artifacts.py`)
- History export files created under `.noeris/history/`: `history-summary.json`, `history-brief.md`, `history-regressions.json`, `history-regressions.md`
- History regression gate: `status=ok`, no blocking regressions

## Outcome

- FP8 runtime policy/cache integration is now covered by local CI parity flow.
- History-based regression guard is active and validated on freshly generated artifacts.
