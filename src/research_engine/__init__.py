"""Core package for the autonomous ML research scaffold."""

from .benchmarks import DEFAULT_BENCHMARKS
from .codex_config import CodexProviderConfig, load_codex_provider_config, render_github_env_setup
from .export import export_run_bundle
from .ingestion import (
    ArxivAtomSourceProvider,
    CompositeSourceProvider,
    GitHubRepositorySourceProvider,
    UrllibHttpClient,
)
from .models import (
    BenchmarkGoal,
    ExperimentResult,
    ExperimentSpec,
    Hypothesis,
    ResearchContext,
    ResearchMemo,
    ResearchRunRecord,
    ResearchSource,
    ResearchTopic,
    VerificationReport,
)
from .pipeline import ResearchPipeline

__all__ = [
    "ArxivAtomSourceProvider",
    "BenchmarkGoal",
    "CodexProviderConfig",
    "CompositeSourceProvider",
    "DEFAULT_BENCHMARKS",
    "ExperimentResult",
    "ExperimentSpec",
    "export_run_bundle",
    "GitHubRepositorySourceProvider",
    "Hypothesis",
    "load_codex_provider_config",
    "ResearchContext",
    "ResearchMemo",
    "ResearchRunRecord",
    "ResearchSource",
    "ResearchPipeline",
    "ResearchTopic",
    "render_github_env_setup",
    "UrllibHttpClient",
    "VerificationReport",
]
