# Noeris Research Engine

The autonomous discovery system behind Noeris's kernel optimizations.

## How it works

1. **Ingestion** — monitors ArXiv, GitHub for new model architectures
2. **Hypothesis generation** — identifies fusion opportunities
3. **Experiment execution** — runs benchmarks on Modal (A100/H100) or free Kaggle T4
4. **Cross-run learning** — shape-indexed config database persists across runs
5. **Cost model** — GBR predicts kernel performance (R²=0.94)
6. **Bandit selection** — Thompson sampling picks configs (98% optimal in 1 iteration)
7. **Frontier tracking** — identifies the Pareto-optimal configs per shape per GPU

## Components

- `pipeline.py` — orchestrates the discovery loop
- `ingestion.py` — ArXiv + GitHub source providers
- `bandit_selector.py` — Thompson-sampling config selection
- `cost_model.py` — gradient-boosted performance prediction
- `map_elites.py` — quality-diversity search
- `adaptive_router.py` — meta-bandit selector routing
- `world_model.py` — hypothesis tracking
- `store.py` — persistent config database
