# QK-Norm Attention Bandit v3 (A100)

Targeted rerun of issue #30 with a larger config budget on the two QK-norm buckets only:

- `gemma4_qknorm`
- `gemma4_qknorm_global`

Command concept: bandit selector, `configs-per-run=30`, A100, no LLM proposer.

## Headline

The larger-budget targeted run strongly improves both QK-norm buckets versus v2 baseline:

| Shape | v2 baseline | v3 best | Delta |
|---|---:|---:|---:|
| `gemma4_qknorm` | 4.62 TFLOPS | **27.69 TFLOPS** (`m32_n32_w4_s2`) | **+499.4%** |
| `gemma4_qknorm_global` | 18.65 TFLOPS | **46.00 TFLOPS** (`m32_n64_w4_s3`) | **+146.6%** |

This confirms the prior "pool-limitation" diagnosis from v2: with a larger budget and direct targeting of qknorm buckets, the bandit finds materially better configs.

## Notes

- Artifact JSON: `docs/results/bandit-qknorm-attention-a100-v3.json`
- Run tested 30 configs × 2 shapes (`60` rows total), with `6` failed rows.
- This is a targeted bucket rerun, not a full-shape attention sweep.
