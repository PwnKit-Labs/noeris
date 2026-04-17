# Results Index

Canonical latest artifacts (current public references):

- Gemma deeper-fusion full-layer main results:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json`
- Gemma deeper-fusion stability reruns:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune-repeat2.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune-repeat3.json`
- Policy-routing sanity checks:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-policy-routing-sanity.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-policy-routing-sanity.json`

Targeted QK-norm attention reruns:

- `docs/results/bandit-qknorm-attention-a100-v3.json`
- `docs/results/bandit-qknorm-attention-a100-v3.md`

Reproducible benchmark-pack command:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/gemma4_layer_benchmark_pack.py
```

This writes canonical pack outputs:

- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json`
- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md`

Current canonical pack artifacts:

- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json`
- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md`

Notes:

- Historical artifacts in this directory are retained for auditability and timeline context.
- README and paper should reference canonical latest artifacts above unless explicitly discussing historical progression.
