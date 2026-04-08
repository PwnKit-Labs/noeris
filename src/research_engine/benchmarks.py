from __future__ import annotations

from .models import BenchmarkGoal


DEFAULT_BENCHMARKS: list[BenchmarkGoal] = [
    BenchmarkGoal(
        benchmark_id="matmul-speedup",
        name="Matrix Multiplication Speedup",
        category="systems",
        goal=(
            "Find and validate techniques that improve matrix multiplication or "
            "closely related kernel performance under realistic constraints."
        ),
        success_metric=(
            "Measured throughput / latency improvement with clear hardware, shape, "
            "precision, and baseline reporting."
        ),
        why_it_matters=(
            "Forces Noeris to bridge literature review, low-level systems "
            "reasoning, experiment design, and empirical validation."
        ),
        baseline_guidance=(
            "Compare against a clearly named baseline kernel or library path for "
            "specific tensor shapes, dtypes, and hardware."
        ),
        required_artifacts=[
            "hardware-profile.json",
            "benchmark-config.json",
            "raw-timing-results.json",
            "baseline-comparison.md",
        ],
        ci_lane="manual-expensive",
        starter_topics=[
            "kernel tiling strategies",
            "sparsity-aware kernels",
            "mixed-precision execution tradeoffs",
            "memory-bandwidth bottlenecks",
        ],
    ),
    BenchmarkGoal(
        benchmark_id="long-context-reasoning",
        name="Long-Context Reasoning",
        category="llm",
        goal=(
            "Propose and evaluate interventions that improve reasoning quality over "
            "long contexts without simply scaling model size."
        ),
        success_metric="Improvement on long-context evals with controlled cost changes.",
        why_it_matters=(
            "Matches a visible LLM pain point and keeps the research loop grounded "
            "in post-training and evaluation work."
        ),
        baseline_guidance=(
            "Compare against a fixed base model or prompt/program baseline on a "
            "known long-context evaluation set."
        ),
        required_artifacts=[
            "eval-manifest.json",
            "baseline-metrics.json",
            "candidate-metrics.json",
            "failure-analysis.md",
        ],
        ci_lane="scheduled-benchmark",
        starter_topics=[
            "memory routing",
            "retrieval structures",
            "attention approximations",
            "context compression",
        ],
    ),
    BenchmarkGoal(
        benchmark_id="tool-use-reliability",
        name="Tool-Use Reliability",
        category="agents",
        goal=(
            "Increase correctness and recovery in multi-step tool-using agent tasks."
        ),
        success_metric="Higher task success and lower unforced-error rate on tool tasks.",
        why_it_matters=(
            "Links directly to the broader agent-system thesis and to products like "
            "PwnKit that depend on reliable tool execution."
        ),
        baseline_guidance=(
            "Compare terminal-first / bash-first execution against a more structured "
            "tool policy on a fixed task suite."
        ),
        required_artifacts=[
            "task-suite.json",
            "terminal-transcript.jsonl",
            "tool-selection-summary.json",
            "success-summary.json",
            "error-taxonomy.md",
        ],
        ci_lane="scheduled-benchmark",
        starter_topics=[
            "bash-first vs structured tools",
            "planner/reviewer separation",
            "error recovery loops",
            "tool choice policies",
            "memory-conditioned tool calling",
        ],
    ),
]


def get_benchmark(benchmark_id: str) -> BenchmarkGoal:
    for benchmark in DEFAULT_BENCHMARKS:
        if benchmark.benchmark_id == benchmark_id:
            return benchmark
    raise KeyError(f"Unknown benchmark_id: {benchmark_id}")


def benchmark_from_topic_constraints(constraints: list[str]) -> BenchmarkGoal | None:
    for constraint in constraints:
        if constraint.startswith("benchmark_id:"):
            return get_benchmark(constraint.split(":", 1)[1])
    return None
