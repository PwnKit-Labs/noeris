from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_engine.codex_config import load_codex_provider_config, render_github_env_setup


CONFIG_SAMPLE = """
model_provider = "azure"
model = "gpt-5.4"

[model_providers.azure]
base_url = "https://example-resource.openai.azure.com/openai/v1"
env_key = "AZURE_OPENAI_API_KEY"
wire_api = "responses"
"""


class CodexConfigTests(unittest.TestCase):
    def test_load_codex_provider_config_reads_azure_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            path.write_text(CONFIG_SAMPLE, encoding="utf-8")
            config = load_codex_provider_config(path)

        assert config is not None
        self.assertEqual(config.provider_name, "azure")
        self.assertEqual(config.secret_env_var, "AZURE_OPENAI_API_KEY")
        self.assertEqual(
            config.base_url,
            "https://example-resource.openai.azure.com/openai/v1",
        )
        self.assertEqual(config.model, "gpt-5.4")
        self.assertEqual(config.wire_api, "responses")

    def test_render_github_env_setup_builds_expected_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            path.write_text(CONFIG_SAMPLE, encoding="utf-8")
            config = load_codex_provider_config(path)

        assert config is not None
        mapping = render_github_env_setup(config)
        self.assertEqual(mapping["provider"], "azure")
        self.assertEqual(mapping["github_secrets"], ["AZURE_OPENAI_API_KEY"])
        self.assertEqual(
            mapping["github_vars"]["AZURE_OPENAI_MODEL"],
            "gpt-5.4",
        )
        self.assertEqual(
            mapping["workflow_env"]["AZURE_OPENAI_API_KEY"],
            "${{ secrets.AZURE_OPENAI_API_KEY }}",
        )


if __name__ == "__main__":
    unittest.main()
