# Cross-Vendor Transfer Evaluation Protocol

This protocol defines how to validate issue #82 once AMD measurements are available.

## Inputs

- Prediction artifact from scaffold lane:
  - `docs/results/cross-vendor-zero-shot-scaffold-mi300x.json`
- Measured AMD artifact (to be produced from MI300X/MI250 runs), format:

```json
{
  "measured": {
    "attention": {
      "bucket_name": [
        {"config_id": "cfg_1", "metric": 123.4, "latency_ms": 1.23}
      ]
    }
  }
}
```

## Command

```bash
PYTHONPATH=src python3 scripts/cross_vendor_transfer_eval.py \
  --prediction-json docs/results/cross-vendor-zero-shot-scaffold-mi300x.json \
  --measured-json docs/results/cross-vendor-measured-mi300x.json \
  --top-k 5
```

## Metrics

- Spearman rank correlation between predicted and measured rankings.
- Top-k hit rate overlap between predicted and measured top-k config IDs.
- Latency regret of predicted-best config vs measured-best config.

## Outputs

- `docs/results/cross-vendor-transfer-eval.json`
- `docs/results/cross-vendor-transfer-eval.md`

These outputs are the paper-facing evidence for cross-vendor ranking transfer quality.
