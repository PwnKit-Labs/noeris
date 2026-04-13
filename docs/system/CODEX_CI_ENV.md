# Codex CI Env

Noeris can mirror the provider shape from your local `~/.codex/config.toml` into GitHub Actions.

## Current Local Shape

Your local Codex setup is Azure-backed.

That means the relevant CI inputs are:

- GitHub secret: `AZURE_OPENAI_API_KEY`
- GitHub variable: `AZURE_OPENAI_BASE_URL`
- GitHub variable: `AZURE_OPENAI_MODEL`
- GitHub variable: `AZURE_OPENAI_WIRE_API`

For the optional Codex GitHub Action lane, also set:

- GitHub secret: `OPENAI_API_KEY`

## Why Split Secret vs Variable

- API keys belong in GitHub `Secrets`
- base URLs, model names, and wire API mode can live in GitHub `Variables`

## Local Helper

Run:

```bash
python3 -m research_engine.cli ci-env
```

This reads the local Codex config and prints the GitHub env mapping without printing any secret value.

## Practical Setup

1. Open GitHub repository settings.
2. Add `AZURE_OPENAI_API_KEY` under Secrets.
3. Add `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_MODEL`, and `AZURE_OPENAI_WIRE_API` under Variables.
4. Add `OPENAI_API_KEY` if you want the Codex Action research workflow enabled.

## Current Workflow Usage

- `benchmark-plans.yml`
  consumes the Azure env shape
- `codex-benchmark-research.yml`
  consumes both the Azure env shape and `OPENAI_API_KEY`

