"""Core package for the autonomous ML research scaffold."""

from .agenda import DEFAULT_RESEARCH_AGENDA
from .benchmarks import DEFAULT_BENCHMARKS
from .codex_config import CodexProviderConfig, load_codex_provider_config, render_github_env_setup
from .executors import DefaultExperimentExecutor, LongContextOfflineExecutor
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

# Import operator modules for side-effect registration.
from . import triton_kernels  # noqa: F401  (registers matmul)
from . import triton_rmsnorm  # noqa: F401  (registers rmsnorm)
from . import triton_softmax  # noqa: F401  (registers softmax)
from . import triton_layernorm  # noqa: F401  (registers layernorm)
from . import triton_cross_entropy  # noqa: F401  (registers cross_entropy)
from . import triton_attention  # noqa: F401  (registers attention)
from . import triton_rotary  # noqa: F401  (registers rotary)
from . import triton_geglu  # noqa: F401  (registers geglu)
from . import triton_qk_norm_rope  # noqa: F401  (registers qk_norm_rope)
from . import triton_moe_router  # noqa: F401  (registers moe_router)
from . import triton_grouped_gemm  # noqa: F401  (registers grouped_gemm)
from . import triton_ple_gather  # noqa: F401  (registers ple_gather)
from . import triton_attention_decode  # noqa: F401  (registers attention_decode)
from . import triton_qk_norm_rope_bwd  # noqa: F401  (registers qk_norm_rope_bwd)
from . import triton_matmul_splitk  # noqa: F401  (registers matmul_splitk)
from .triton_operators import REGISTRY as TRITON_OPERATORS  # noqa: F401

__all__ = [
    "ArxivAtomSourceProvider",
    "BenchmarkGoal",
    "CodexProviderConfig",
    "CompositeSourceProvider",
    "DEFAULT_RESEARCH_AGENDA",
    "DEFAULT_BENCHMARKS",
    "DefaultExperimentExecutor",
    "ExperimentResult",
    "ExperimentSpec",
    "export_run_bundle",
    "GitHubRepositorySourceProvider",
    "Hypothesis",
    "load_codex_provider_config",
    "LongContextOfflineExecutor",
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
