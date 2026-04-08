from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


@dataclass(slots=True)
class CodexProviderConfig:
    provider_name: str
    secret_env_var: str
    base_url: str | None
    model: str | None
    wire_api: str | None


def load_codex_provider_config(
    config_path: str | Path | None = None,
) -> CodexProviderConfig | None:
    path = Path(config_path or Path.home() / ".codex" / "config.toml")
    if not path.exists():
        return None

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    provider_name = payload.get("model_provider")
    if not provider_name:
        return None

    providers = payload.get("model_providers", {})
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        return None

    return CodexProviderConfig(
        provider_name=provider_name,
        secret_env_var=str(provider.get("env_key", "OPENAI_API_KEY")),
        base_url=_string_or_none(provider.get("base_url")),
        model=_string_or_none(payload.get("model")),
        wire_api=_string_or_none(provider.get("wire_api")),
    )


def render_github_env_setup(config: CodexProviderConfig) -> dict:
    return {
        "provider": config.provider_name,
        "github_secrets": [config.secret_env_var],
        "github_vars": {
            "MODEL_PROVIDER": config.provider_name,
            f"{config.provider_name.upper()}_OPENAI_BASE_URL": config.base_url or "",
            f"{config.provider_name.upper()}_OPENAI_MODEL": config.model or "",
            f"{config.provider_name.upper()}_OPENAI_WIRE_API": config.wire_api or "",
        },
        "workflow_env": {
            config.secret_env_var: f"${{{{ secrets.{config.secret_env_var} }}}}",
            f"{config.provider_name.upper()}_OPENAI_BASE_URL": (
                f"${{{{ vars.{config.provider_name.upper()}_OPENAI_BASE_URL }}}}"
            ),
            f"{config.provider_name.upper()}_OPENAI_MODEL": (
                f"${{{{ vars.{config.provider_name.upper()}_OPENAI_MODEL }}}}"
            ),
            f"{config.provider_name.upper()}_OPENAI_WIRE_API": (
                f"${{{{ vars.{config.provider_name.upper()}_OPENAI_WIRE_API }}}}"
            ),
        },
    }


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def has_secret_available(config: CodexProviderConfig) -> bool:
    return bool(os.getenv(config.secret_env_var))
