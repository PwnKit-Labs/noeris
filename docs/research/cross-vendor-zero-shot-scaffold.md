# Cross-Vendor Zero-Shot Scaffold (MI300X from NVIDIA data)

Issue: `#82`.

This scaffold predicts candidate config rankings for an unseen hardware label
(`AMD MI300X`) using only existing NVIDIA benchmark data in the local
ConfigDatabase.

## Run

```bash
PYTHONPATH=src uv run --python 3.11 --no-project python3 scripts/cross_vendor_zero_shot_scaffold.py
```

## Outputs

- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.json`
- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.md`

## What this is (and is not)

- **Is:** a reproducible zero-shot prediction scaffold producing per-operator,
  per-bucket top-k candidate configs for an unseen vendor label.
- **Is not:** a validated cross-vendor transfer result yet (no AMD measurements
  are used in this artifact).

## Next step to convert scaffold into result

Run the predicted candidates on real MI300X/MI250 hardware and compute ranking
transfer metrics (Spearman rho, top-k overlap, best-config hit rate).
