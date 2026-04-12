"""Tests for the LLM proposer integration in scripts/colab_iterate.py."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Make sure the script is importable — mock torch if it's not available,
# but SAVE and RESTORE the original to avoid poisoning later tests.
REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "src"))

_original_torch = sys.modules.get("torch")
_torch_was_mocked = False

if _original_torch is None:
    _torch_mock = types.ModuleType("torch")
    _torch_mock.cuda = mock.MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch_mock
    _torch_was_mocked = True

import colab_iterate  # noqa: E402

# Restore original torch immediately after import so subsequent test
# modules get the real torch (not our minimal mock).
if _torch_was_mocked:
    del sys.modules["torch"]
elif _original_torch is not None:
    sys.modules["torch"] = _original_torch


# ---------------------------------------------------------------------------
# test_llm_flag_no_crash_without_key
# ---------------------------------------------------------------------------
def test_llm_flag_no_crash_without_key(capsys):
    """--llm without any API key prints a warning and returns empty configs."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")}

    # Minimal mock spec/db
    mock_db = mock.MagicMock()
    mock_db.get_insights.return_value = []
    mock_spec = mock.MagicMock()
    mock_spec.param_space = {"BLOCK_M": [64, 128], "BLOCK_N": [64, 128]}

    with mock.patch.dict(os.environ, env, clear=True):
        configs, source = colab_iterate.get_llm_proposed_configs(
            operator="qk_norm_rope",
            hardware="Tesla T4",
            db=mock_db,
            spec=mock_spec,
            n_configs=4,
        )

    assert configs == []
    assert source == "no_api_key"
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "Falling back to grid-only" in captured.out


# ---------------------------------------------------------------------------
# test_proposer_prompt_contains_insights
# ---------------------------------------------------------------------------
def test_proposer_prompt_contains_insights():
    """Verify the prompt template correctly interpolates insights and params."""
    insights = [{"shape_bucket": "M=2048", "best_tflops": 42.5}]
    best_configs = [{"BLOCK_M": 128, "BLOCK_N": 64}]
    param_space = {"BLOCK_M": [64, 128, 256], "BLOCK_N": [64, 128]}

    prompt = colab_iterate.PROPOSER_PROMPT.format(
        operator="qk_norm_rope",
        hardware="Tesla T4",
        n_configs=4,
        insights_json=json.dumps(insights, indent=2),
        best_configs=json.dumps(best_configs, indent=2),
        param_space=json.dumps(param_space, indent=2),
    )

    assert "qk_norm_rope" in prompt
    assert "Tesla T4" in prompt
    assert "best_tflops" in prompt
    assert "BLOCK_M" in prompt
    assert "BLOCK_N" in prompt
    assert "tile sizes" in prompt


# ---------------------------------------------------------------------------
# test_script_has_llm_flag
# ---------------------------------------------------------------------------
def test_script_has_llm_flag():
    """argparse in main() accepts --llm and --anthropic without error."""
    # Build the parser the same way main() does, but without running the GPU code
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", default="qk_norm_rope")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--configs-per-iter", type=int, default=8)
    parser.add_argument("--db-path", default=".noeris/colab-configs.json")
    parser.add_argument("--shapes", default="standard", choices=["tiny", "standard", "full"])
    parser.add_argument("--no-bandit", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--anthropic", action="store_true")
    parser.add_argument("--all-operators", action="store_true")

    args = parser.parse_args(["--llm", "--anthropic"])
    assert args.llm is True
    assert args.anthropic is True

    # Also verify the actual script source has these flags
    src = (SCRIPTS / "colab_iterate.py").read_text()
    assert '"--llm"' in src
    assert '"--anthropic"' in src


# ---------------------------------------------------------------------------
# test_detect_api_key_priority
# ---------------------------------------------------------------------------
def test_detect_api_key_priority():
    """Anthropic key is preferred when all keys are present."""
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "OPENAI_API_KEY": "sk-oai-test",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        result = colab_iterate._detect_api_key()
    assert result is not None
    provider, key = result
    assert provider == "anthropic"
    assert key == "sk-ant-test"


def test_detect_api_key_openai_fallback():
    """Falls back to OpenAI when only that key is set."""
    env = {"OPENAI_API_KEY": "sk-oai-test"}
    with mock.patch.dict(os.environ, env, clear=True):
        result = colab_iterate._detect_api_key()
    assert result == ("openai", "sk-oai-test")


def test_detect_api_key_none():
    """Returns None when no keys are set."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")}
    with mock.patch.dict(os.environ, env, clear=True):
        assert colab_iterate._detect_api_key() is None


# ---------------------------------------------------------------------------
# test_propose_configs_validates_param_space
# ---------------------------------------------------------------------------
def test_propose_configs_validates_param_space(capsys):
    """Invalid LLM proposals are filtered out by param_space validation."""
    mock_db = mock.MagicMock()
    mock_db.get_insights.return_value = []
    mock_spec = mock.MagicMock()
    mock_spec.param_space = {"BLOCK_M": [64, 128], "BLOCK_N": [64, 128]}

    # Mock the anthropic call to return one valid + one invalid config
    fake_configs = [
        {"BLOCK_M": 64, "BLOCK_N": 128},   # valid
        {"BLOCK_M": 999, "BLOCK_N": 128},   # invalid BLOCK_M
    ]

    env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(colab_iterate, "_propose_configs_anthropic", return_value=fake_configs):
        configs, source = colab_iterate.get_llm_proposed_configs(
            operator="qk_norm_rope",
            hardware="Tesla T4",
            db=mock_db,
            spec=mock_spec,
            n_configs=4,
        )

    assert len(configs) == 1
    assert configs[0] == {"BLOCK_M": 64, "BLOCK_N": 128}
    assert source == "anthropic"
