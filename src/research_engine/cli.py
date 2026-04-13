from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path as _Path

from ._legacy.agenda import DEFAULT_RESEARCH_AGENDA
from .benchmarks import DEFAULT_BENCHMARKS, get_benchmark
from .codex_config import load_codex_provider_config, render_github_env_setup
from .export import export_run_bundle
from .executors import (
    DefaultExperimentExecutor,
    LongContextResponsesExecutor,
    MatmulPythonExecutor,
    ToolUseResponsesExecutor,
)
from ._legacy.ingestion import (
    ArxivAtomSourceProvider,
    CompositeSourceProvider,
    GitHubRepositorySourceProvider,
    UrllibHttpClient,
)
from .llm import LlmConfigurationError, LlmHypothesisPlanner, LlmResearchMemory, ResponsesApiClient
from .triton_kernels import (
    MATMUL_SHAPE_BUCKETS,
    ConfigDatabase,
    config_id,
    propose_triton_configs,
    select_configs_for_run,
)
from .models import ResearchTopic
from ._legacy.pipeline import ResearchPipeline
from .store import JsonFileRunStore


THESIS = (
    "An autonomous R&D engine for machine learning that turns fresh "
    "literature and code into ranked hypotheses, reproducible experiments, "
    "and evidence-backed conclusions."
)

ARCHITECTURE = (
    "Stateful research loop with explicit seams for source collection, "
    "research memory, hypothesis planning, experiment planning, execution, "
    "verification, and memo writing."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("thesis", help="print the current product thesis")
    subparsers.add_parser("architecture", help="print the current architecture stance")
    subparsers.add_parser("agenda", help="print the current research agenda")
    subparsers.add_parser("benchmarks", help="print the current standing benchmark goals")
    subparsers.add_parser("ci-env", help="show GitHub env/secret mapping from local Codex config")
    subparsers.add_parser("status", help="show local verification and artifact status")

    cycle_parser = subparsers.add_parser("cycle", help="run a scaffolded research cycle")
    cycle_parser.add_argument("--topic", required=True, help="topic under investigation")
    cycle_parser.add_argument(
        "--objective",
        default="improve measurable ML/LLM performance with a bounded intervention",
        help="goal to optimize for",
    )
    cycle_parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="optional operating constraint; repeatable",
    )
    cycle_parser.add_argument(
        "--llm",
        action="store_true",
        help="use the configured Responses API provider for research memory and hypothesis generation",
    )
    cycle_parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="max live source results per provider when --llm is enabled",
    )
    cycle_parser.add_argument(
        "--live-execution",
        action="store_true",
        help="use a real model-backed executor where available instead of the offline deterministic executor",
    )

    sources_parser = subparsers.add_parser(
        "sources",
        help="discover live sources from arXiv and GitHub",
    )
    sources_parser.add_argument("--topic", required=True, help="topic under investigation")
    sources_parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="max results per provider",
    )

    run_parser = subparsers.add_parser("run", help="run and persist a research cycle")
    run_parser.add_argument("--topic", required=True, help="topic under investigation")
    run_parser.add_argument(
        "--objective",
        default="improve measurable ML/LLM performance with a bounded intervention",
        help="goal to optimize for",
    )
    run_parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="optional operating constraint; repeatable",
    )
    run_parser.add_argument(
        "--llm",
        action="store_true",
        help="use the configured Responses API provider for research memory and hypothesis generation",
    )
    run_parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="max live source results per provider when --llm is enabled",
    )
    run_parser.add_argument(
        "--live-execution",
        action="store_true",
        help="use a real model-backed executor where available instead of the offline deterministic executor",
    )

    subparsers.add_parser("runs", help="list persisted research runs")

    history_parser = subparsers.add_parser(
        "history",
        help="summarize cross-run claim and confidence changes",
    )
    history_parser.add_argument(
        "--benchmark-id",
        default="",
        help="optional benchmark id filter",
    )
    history_parser.add_argument(
        "--topic",
        default="",
        help="optional topic filter",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="max runs to compare",
    )

    history_brief_parser = subparsers.add_parser(
        "history-brief",
        help="render a human-readable history summary across runs",
    )
    history_brief_parser.add_argument(
        "--benchmark-id",
        default="",
        help="optional benchmark id filter",
    )
    history_brief_parser.add_argument(
        "--topic",
        default="",
        help="optional topic filter",
    )
    history_brief_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="max runs to compare",
    )

    export_history_parser = subparsers.add_parser(
        "export-history",
        help="export cross-run history summary as json and markdown",
    )
    export_history_parser.add_argument(
        "--benchmark-id",
        default="",
        help="optional benchmark id filter",
    )
    export_history_parser.add_argument(
        "--topic",
        default="",
        help="optional topic filter",
    )
    export_history_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="max runs to compare",
    )
    export_history_parser.add_argument(
        "--output-dir",
        default=".noeris/history",
        help="directory for exported history artifacts",
    )

    show_run_parser = subparsers.add_parser("show-run", help="show a persisted research run")
    show_run_parser.add_argument("run_id", help="identifier of the run to show")

    export_run_parser = subparsers.add_parser(
        "export-run",
        help="export a persisted run into an artifact bundle",
    )
    export_run_parser.add_argument("run_id", help="identifier of the run to export")
    export_run_parser.add_argument(
        "--output-dir",
        default=".noeris/artifacts",
        help="directory for exported artifact bundles",
    )

    benchmark_run_parser = subparsers.add_parser(
        "benchmark-run",
        help="run and persist a standing benchmark goal",
    )
    benchmark_run_parser.add_argument(
        "benchmark_id",
        help="identifier of the benchmark goal to run",
    )
    benchmark_run_parser.add_argument(
        "--llm",
        action="store_true",
        help="use the configured Responses API provider for research memory and hypothesis generation",
    )
    benchmark_run_parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="max live source results per provider when --llm is enabled",
    )
    benchmark_run_parser.add_argument(
        "--live-execution",
        action="store_true",
        help="use a real model-backed executor where available instead of the offline deterministic executor",
    )

    iterate_parser = subparsers.add_parser(
        "iterate",
        help="run a bounded benchmark iteration loop and report the best result",
    )
    iterate_parser.add_argument(
        "benchmark_id",
        help="identifier of the benchmark goal to iterate on",
    )
    iterate_parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="number of bounded benchmark iterations to run",
    )
    iterate_parser.add_argument(
        "--llm",
        action="store_true",
        help="use the configured Responses API provider for benchmark planning/proposal where supported",
    )
    iterate_parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="max live source results per provider when --llm is enabled",
    )
    iterate_parser.add_argument(
        "--live-execution",
        action="store_true",
        help="use a real model-backed executor where available instead of the offline deterministic executor",
    )

    triton_parser = subparsers.add_parser(
        "triton-iterate",
        help="run Triton kernel search: generate configs, benchmark on GPU, record to config database",
    )
    triton_parser.add_argument(
        "--operator",
        default="matmul",
        choices=["matmul", "rmsnorm", "softmax", "layernorm", "cross_entropy", "attention", "rotary", "geglu"],
        help="which Triton operator to search",
    )
    triton_parser.add_argument(
        "--configs-per-run",
        type=int,
        default=8,
        help="max configs to benchmark per iteration",
    )
    triton_parser.add_argument(
        "--gpu",
        default="A100",
        help="GPU type for Modal execution (A100, H100, T4)",
    )
    triton_parser.add_argument(
        "--llm",
        action="store_true",
        help="use the LLM proposer to suggest novel configs",
    )
    triton_parser.add_argument(
        "--local",
        action="store_true",
        help="run locally instead of via Modal (requires local GPU + triton)",
    )
    triton_parser.add_argument(
        "--shapes",
        default="standard",
        choices=["tiny", "standard", "full"],
        help="shape set to benchmark: tiny (2), standard (6), full (10)",
    )
    triton_parser.add_argument(
        "--db-path",
        default=".noeris/triton-configs.json",
        help="path to the shape-indexed config database",
    )
    triton_parser.add_argument(
        "--cost-model",
        default="",
        help="path to a trained CostModel pickle (enables cost-model-ranked selection)",
    )
    triton_parser.add_argument(
        "--bandit",
        action="store_true",
        help=(
            "use Thompson-sampling bandit selector instead of the standard "
            "frontier-slot selector; exploits empirical reward history in the "
            "config database via Beta posteriors per shape bucket"
        ),
    )

    kb_parser = subparsers.add_parser(
        "kernelbench-eval",
        help="run KernelBench-style evaluation and compute fast_p scores",
    )
    kb_parser.add_argument(
        "--operator",
        default="",
        choices=["", "matmul", "rmsnorm", "softmax", "layernorm", "cross_entropy", "attention", "rotary", "geglu"],
        help="restrict to one operator, or leave empty for all",
    )
    kb_parser.add_argument(
        "--gpu",
        default="A100",
        help="GPU type for Modal execution",
    )
    kb_parser.add_argument(
        "--configs-per-problem",
        type=int,
        default=8,
        help="max configs to test per problem",
    )
    kb_parser.add_argument(
        "--output",
        default=".noeris/kernelbench-report.json",
        help="path to write the JSON report",
    )
    kb_parser.add_argument(
        "--baseline",
        default="measure",
        choices=["measure", "external-h100-modal"],
        help=(
            "baseline source: 'measure' re-runs the PyTorch reference on the "
            "target GPU (default, expensive); 'external-h100-modal' uses the "
            "pre-computed H100 Modal baselines vendored from upstream "
            "KernelBench (drop-in comparable, only valid on H100 Modal)."
        ),
    )
    kb_parser.add_argument(
        "--timer",
        default="cuda_event",
        choices=["cuda_event", "do_bench"],
        help=(
            "timing method for the generated benchmark scripts: 'cuda_event' "
            "(upstream KernelBench default: 3 warmup + 10 trials, L2 flush, "
            "median) or 'do_bench' (legacy Triton adaptive, no L2 flush)."
        ),
    )

    kb_up_parser = subparsers.add_parser(
        "kernelbench-upstream-eval",
        help=(
            "run Noeris kernels against actual upstream KernelBench L1 "
            "problems (fp32, upstream shapes, cuda_event+L2-flush timing) "
            "for honest apples-to-apples comparison"
        ),
    )
    kb_up_parser.add_argument(
        "--gpu", default="A100",
        help="GPU type for Modal execution (A100 or H100).",
    )
    kb_up_parser.add_argument(
        "--timer",
        default="cuda_event",
        choices=["cuda_event", "do_bench"],
        help="timing method (cuda_event matches upstream).",
    )
    kb_up_parser.add_argument(
        "--output",
        default="",
        help="path to write the markdown summary (JSON path is auto-derived). "
             "Defaults to docs/results/kernelbench-upstream-l1-{gpu}.md.",
    )

    ablation_parser = subparsers.add_parser(
        "ablation",
        help="run cross-run learning ablation: with vs without database",
    )
    ablation_parser.add_argument(
        "--operator",
        required=True,
        choices=["matmul", "rmsnorm", "softmax", "layernorm", "cross_entropy", "attention", "rotary", "geglu"],
    )
    ablation_parser.add_argument("--gpu", default="A100")
    ablation_parser.add_argument("--trials", type=int, default=1,
                                  help="Number of independent trials (set >1 for statistical reporting)")
    ablation_parser.add_argument("--iterations", type=int, default=5)
    ablation_parser.add_argument("--configs-per-run", type=int, default=8)
    ablation_parser.add_argument("--shapes", default="standard", choices=["tiny", "standard", "full"])
    ablation_parser.add_argument("--no-llm", action="store_true")
    ablation_parser.add_argument("--warm-up", action="store_true",
                                  help="Warm up database with N iterations before measuring")
    ablation_parser.add_argument("--warm-up-iterations", type=int, default=3)
    ablation_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use persistent Modal session (one warm container across all iterations)",
    )
    ablation_parser.add_argument("--output", default=".noeris/ablation-report.json")

    kb_l4_parser = subparsers.add_parser(
        "kernelbench-l4-eval",
        help="generate (or run) a KernelBench Level 4 op-substitution benchmark script",
    )
    kb_l4_parser.add_argument(
        "--gpu", default="A100",
        help="GPU type for Modal execution (reserved for future use).",
    )
    kb_l4_parser.add_argument(
        "--problems",
        default="",
        help="Comma-separated problem indices to include (default: top-5 attack order).",
    )
    kb_l4_parser.add_argument(
        "--output",
        default="",
        help="Path to write the generated script (default: print to stdout).",
    )
    kb_l4_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Generate and print the benchmark script without executing (default: true).",
    )

    kb_hf_parser = subparsers.add_parser(
        "kernelbench-hf-coverage",
        help="Probe the HuggingFace KernelBench dataset and report operator coverage",
    )
    kb_hf_parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Levels to probe (default all)",
    )
    kb_hf_parser.add_argument(
        "--limit-per-level",
        type=int,
        default=None,
        help="Optional cap on problems per level",
    )

    gemma4_parser = subparsers.add_parser(
        "gemma4-layer-bench",
        help="generate (or run) end-to-end Gemma 4 decoder layer benchmark: Noeris fused vs PyTorch separated",
    )
    gemma4_parser.add_argument(
        "--output",
        default="",
        help="path to write the generated script (default: print to stdout)",
    )
    gemma4_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="generate and print the benchmark script without executing (default: true)",
    )

    train_parser = subparsers.add_parser(
        "train-cost-model",
        help="Train a learned cost model from one or more ConfigDatabase files",
    )
    train_parser.add_argument(
        "--db-paths",
        nargs="+",
        default=[".noeris/triton-configs.json"],
        help="Paths to ConfigDatabase JSON files to use as training data",
    )
    train_parser.add_argument(
        "--output",
        default=".noeris/cost-model.pkl",
        help="Where to save the trained model",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "thesis":
        print(THESIS)
        return 0

    if args.command == "architecture":
        print(ARCHITECTURE)
        return 0
    if args.command == "agenda":
        print(json.dumps([asdict(item) for item in DEFAULT_RESEARCH_AGENDA], indent=2))
        return 0
    if args.command == "benchmarks":
        print(json.dumps([asdict(goal) for goal in DEFAULT_BENCHMARKS], indent=2))
        return 0
    if args.command == "ci-env":
        config = load_codex_provider_config()
        if config is None:
            print(json.dumps({"error": "No local Codex config provider found."}, indent=2))
            return 1
        print(json.dumps(render_github_env_setup(config), indent=2))
        return 0
    if args.command == "status":
        store = JsonFileRunStore()
        runs = store.list_runs()
        latest = runs[0] if runs else None
        artifacts_dir = _Path(".noeris/artifacts")
        history_dir = _Path(".noeris/history")
        try:
            llm_provider_configured = load_codex_provider_config() is not None
        except Exception:
            llm_provider_configured = False
        payload = {
            "capabilities": {
                "gh_cli": shutil.which("gh") is not None,
                "modal_cli": shutil.which("modal") is not None,
                "azure_openai_key": bool(os.getenv("AZURE_OPENAI_API_KEY")),
                "openai_key": bool(os.getenv("OPENAI_API_KEY")),
                "llm_provider_configured": llm_provider_configured,
            },
            "run_count": len(runs),
            "artifact_bundle_count": sum(1 for path in artifacts_dir.iterdir()) if artifacts_dir.exists() else 0,
            "history_artifacts_present": {
                "history_summary_json": (history_dir / "history-summary.json").exists(),
                "history_brief_md": (history_dir / "history-brief.md").exists(),
            },
            "latest_run": latest or {},
            "workflow_summary": _status_workflow_summary(),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "sources":
        topic = ResearchTopic(
            name=args.topic,
            objective="discover relevant sources",
        )
        provider = CompositeSourceProvider(
            providers=[
                ArxivAtomSourceProvider(client=UrllibHttpClient(), max_results=args.max_results),
                GitHubRepositorySourceProvider(
                    client=UrllibHttpClient(),
                    max_results=args.max_results,
                ),
            ]
        )
        print(
            json.dumps(
                [asdict(source) for source in provider.collect(topic)],
                indent=2,
            )
        )
        return 0
    if args.command == "runs":
        store = JsonFileRunStore()
        print(json.dumps(store.list_runs(), indent=2))
        return 0
    if args.command == "history":
        store = JsonFileRunStore()
        print(
            json.dumps(
                store.summarize_history(
                    benchmark_id=args.benchmark_id or None,
                    topic=args.topic or None,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "history-brief":
        store = JsonFileRunStore()
        print(
            store.render_history_brief(
                benchmark_id=args.benchmark_id or None,
                topic=args.topic or None,
                limit=args.limit,
            )
        )
        return 0
    if args.command == "export-history":
        store = JsonFileRunStore()
        summary = store.summarize_history(
            benchmark_id=args.benchmark_id or None,
            topic=args.topic or None,
            limit=args.limit,
        )
        brief = store.render_history_brief(
            benchmark_id=args.benchmark_id or None,
            topic=args.topic or None,
            limit=args.limit,
        )
        output_dir = _Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "history-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "history-brief.md").write_text(
            brief,
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "files": [
                        str(output_dir / "history-summary.json"),
                        str(output_dir / "history-brief.md"),
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "show-run":
        store = JsonFileRunStore()
        print(json.dumps(asdict(store.load(args.run_id)), indent=2))
        return 0
    if args.command == "export-run":
        store = JsonFileRunStore()
        bundle_dir = export_run_bundle(
            store.load(args.run_id),
            output_dir=args.output_dir,
        )
        print(json.dumps({"run_id": args.run_id, "bundle_dir": str(bundle_dir)}, indent=2))
        return 0
    if args.command == "run":
        try:
            pipeline = build_pipeline(
                use_llm=args.llm,
                max_results=args.max_results,
                live_execution=args.live_execution,
                benchmark_id=None,
            )
        except LlmConfigurationError as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1
        topic = ResearchTopic(
            name=args.topic,
            objective=args.objective,
            constraints=args.constraint,
        )
        record = pipeline.run_record_for(topic=topic, benchmark_id=None)
        store = JsonFileRunStore()
        path = store.save(record)
        print(
            json.dumps(
                {
                    "run_id": record.run_id,
                    "path": str(path),
                    "created_at": record.created_at,
                    "verification": asdict(record.verification),
                    "memo": asdict(record.memo),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "benchmark-run":
        benchmark = get_benchmark(args.benchmark_id)
        try:
            pipeline = build_pipeline(
                use_llm=args.llm,
                max_results=args.max_results,
                live_execution=args.live_execution,
                benchmark_id=benchmark.benchmark_id,
            )
        except LlmConfigurationError as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1
        topic = ResearchTopic(
            name=benchmark.name.lower(),
            objective=benchmark.goal,
            benchmark_id=benchmark.benchmark_id,
            constraints=[
                f"benchmark_id:{benchmark.benchmark_id}",
                f"success_metric:{benchmark.success_metric}",
                f"category:{benchmark.category}",
                f"ci_lane:{benchmark.ci_lane}",
            ],
        )
        record = pipeline.run_record_for(
            topic=topic,
            benchmark_id=benchmark.benchmark_id,
        )
        store = JsonFileRunStore()
        path = store.save(record)
        bundle_dir = export_run_bundle(record, output_dir=".noeris/artifacts")
        print(
            json.dumps(
                {
                    "run_id": record.run_id,
                    "benchmark_id": record.benchmark_id,
                    "path": str(path),
                    "bundle_dir": str(bundle_dir),
                    "created_at": record.created_at,
                    "verification": asdict(record.verification),
                    "memo": asdict(record.memo),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "iterate":
        benchmark = get_benchmark(args.benchmark_id)
        store = JsonFileRunStore()
        history_before = store.summarize_history(
            benchmark_id=benchmark.benchmark_id,
            limit=20,
        )
        previous_frontier = _extract_frontier_snapshot(
            store,
            benchmark_id=benchmark.benchmark_id,
            run_id=str(history_before.get("latest_run_id", "")).strip(),
        )
        try:
            pipeline = build_pipeline(
                use_llm=args.llm,
                max_results=args.max_results,
                live_execution=args.live_execution,
                benchmark_id=benchmark.benchmark_id,
            )
        except LlmConfigurationError as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1
        iterations = max(1, args.iterations)
        runs = []
        best_metric = None
        best_run_id = ""
        best_record = None
        for _ in range(iterations):
            topic = ResearchTopic(
                name=benchmark.name.lower(),
                objective=benchmark.goal,
                benchmark_id=benchmark.benchmark_id,
                constraints=[
                    f"benchmark_id:{benchmark.benchmark_id}",
                    f"success_metric:{benchmark.success_metric}",
                    f"category:{benchmark.category}",
                    f"ci_lane:{benchmark.ci_lane}",
                ],
            )
            record = pipeline.run_record_for(
                topic=topic,
                benchmark_id=benchmark.benchmark_id,
            )
            path = store.save(record)
            bundle_dir = export_run_bundle(record, output_dir=".noeris/artifacts")
            metric = _extract_benchmark_metric(record)
            runs.append(
                {
                    "run_id": record.run_id,
                    "path": str(path),
                    "bundle_dir": str(bundle_dir),
                    "metric": metric,
                }
            )
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_run_id = record.run_id
                best_record = record
        previous_best_metric = history_before.get("best_benchmark_metric")
        if previous_best_metric is None:
            outcome = "new_baseline"
        elif best_metric > previous_best_metric:
            outcome = "improved"
        elif best_metric < previous_best_metric:
            outcome = "regressed"
        else:
            outcome = "plateaued"
        best_frontier = _extract_frontier_snapshot_from_record(
            best_record,
            benchmark_id=benchmark.benchmark_id,
        )
        print(
            json.dumps(
                {
                    "benchmark_id": benchmark.benchmark_id,
                    "iterations": iterations,
                    "previous_best_metric": previous_best_metric,
                    "best_run_id": best_run_id,
                    "best_metric": best_metric,
                    "outcome": outcome,
                    "previous_frontier": previous_frontier,
                    "best_frontier": best_frontier,
                    "frontier_delta": _build_frontier_delta(
                        previous_frontier=previous_frontier,
                        best_frontier=best_frontier,
                    ),
                    "runs": runs,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "triton-iterate":
        return _run_triton_iterate(args)

    if args.command == "kernelbench-eval":
        return _run_kernelbench_eval(args)

    if args.command == "kernelbench-upstream-eval":
        return _run_kernelbench_upstream_eval(args)

    if args.command == "ablation":
        return _run_ablation(args)

    if args.command == "train-cost-model":
        return _run_train_cost_model(args)

    if args.command == "kernelbench-l4-eval":
        return _run_kernelbench_l4_eval(args)

    if args.command == "kernelbench-hf-coverage":
        return _run_kernelbench_hf_coverage(args)

    if args.command == "gemma4-layer-bench":
        return _run_gemma4_layer_bench(args)

    try:
        pipeline = build_pipeline(
            use_llm=args.llm,
            max_results=args.max_results,
            live_execution=args.live_execution,
            benchmark_id=None,
        )
    except LlmConfigurationError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    topic = ResearchTopic(
        name=args.topic,
        objective=args.objective,
        constraints=args.constraint,
    )
    print(json.dumps(pipeline.run_cycle_dict(topic), indent=2))
    return 0


def _status_workflow_summary() -> dict[str, object]:
    if shutil.which("gh") is None:
        return {}
    try:
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return {}
    repo = _parse_github_repo_slug(remote_url)
    if not repo:
        return {}
    try:
        payload = json.loads(
            subprocess.check_output(
                [
                    "gh",
                    "run",
                    "list",
                    "--repo",
                    repo,
                    "--limit",
                    "20",
                    "--json",
                    "workflowName,status,conclusion,databaseId,displayTitle",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )
    except Exception:
        return {}

    workflows: dict[str, dict[str, object]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = str(row.get("workflowName", "")).strip()
        if not name:
            continue
        item = workflows.setdefault(
            name,
            {"queued": 0, "in_progress": 0, "completed": 0, "latest_display_title": ""},
        )
        status = str(row.get("status", "")).strip()
        if status in {"queued", "in_progress", "completed"}:
            item[status] = int(item.get(status, 0)) + 1
        if not item["latest_display_title"]:
            item["latest_display_title"] = row.get("displayTitle", "")
        if status == "completed":
            conclusion = str(row.get("conclusion", "")).strip() or "unknown"
            key = f"completed_{conclusion}"
            item[key] = int(item.get(key, 0)) + 1
    return {"repo": repo, "workflows": workflows}


def _parse_github_repo_slug(remote_url: str) -> str:
    text = remote_url.strip()
    if text.startswith("https://github.com/"):
        slug = text.removeprefix("https://github.com/")
    elif text.startswith("git@github.com:"):
        slug = text.removeprefix("git@github.com:")
    else:
        return ""
    if slug.endswith(".git"):
        slug = slug[:-4]
    parts = [part for part in slug.split("/") if part]
    if len(parts) != 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def _run_kernelbench_l4_eval(args) -> int:
    """Generate (and optionally run) a KernelBench Level 4 benchmark script."""
    from .kernelbench_l4 import generate_l4_benchmark_script, get_l4_problems

    all_problems = get_l4_problems()

    # Filter by --problems if provided.
    if args.problems:
        indices = [int(x.strip()) for x in args.problems.split(",")]
        by_id = {p.problem_id: p for p in all_problems}
        problems = []
        for idx in indices:
            if idx not in by_id:
                print(f"Error: problem index {idx} not found in addressable L4 set.")
                return 1
            problems.append(by_id[idx])
    else:
        problems = None  # default: top-5 attack order

    script = generate_l4_benchmark_script(problems)

    if args.output:
        out = _Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script)
        print(f"Benchmark script written to {out}")
    else:
        print(script)

    if args.dry_run:
        print("\n# --dry-run is active; script printed but not executed.", file=sys.stderr)

    return 0


def _run_kernelbench_hf_coverage(args) -> int:
    """Probe the HuggingFace KernelBench dataset and report coverage."""
    from .kernelbench_hf import fetch_and_report_coverage

    report = fetch_and_report_coverage(
        levels=args.levels,
        limit_per_level=args.limit_per_level,
    )
    print(json.dumps(report, indent=2))
    return 0


def _run_gemma4_layer_bench(args) -> int:
    """Generate (and optionally run) the Gemma 4 decoder layer benchmark."""
    from .gemma4_layer_benchmark import generate_gemma4_layer_benchmark_script

    script = generate_gemma4_layer_benchmark_script()

    if args.output:
        out = _Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script)
        print(f"Benchmark script written to {out}")
    else:
        print(script)

    if args.dry_run:
        print("\n# --dry-run is active; script printed but not executed.", file=sys.stderr)

    return 0


def _run_train_cost_model(args) -> int:
    """Train and save the learned cost model."""
    from .cost_model import CostModel

    model = CostModel()
    stats = model.train_from_databases(args.db_paths)
    output = _Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(json.dumps(
        {"output_path": str(output), "stats": stats},
        indent=2,
    ))
    return 0


def _run_ablation(args) -> int:
    """Run cross-run learning ablation: with database vs without."""
    from .ablation import (
        run_ablation,
        run_fast_multi_trial_ablation,
        run_multi_trial_ablation,
    )

    if getattr(args, "trials", 1) > 1:
        if getattr(args, "fast", False):
            report = run_fast_multi_trial_ablation(
                operator=args.operator,
                gpu=args.gpu,
                trials=args.trials,
                iterations=args.iterations,
                configs_per_run=args.configs_per_run,
                use_llm=not args.no_llm,
                shapes_set=args.shapes,
            )
        else:
            report = run_multi_trial_ablation(
                operator=args.operator,
                gpu=args.gpu,
                trials=args.trials,
                iterations=args.iterations,
                configs_per_run=args.configs_per_run,
                use_llm=not args.no_llm,
                shapes_set=args.shapes,
            )
    else:
        report = run_ablation(
            operator=args.operator,
            gpu=args.gpu,
            iterations=args.iterations,
            configs_per_run=args.configs_per_run,
            use_llm=not args.no_llm,
            shapes_set=args.shapes,
            warm_up_database=args.warm_up,
            warm_up_iterations=args.warm_up_iterations,
        )

    output_path = _Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.summary(), indent=2) + "\n")

    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(report.summary_text())

    print(report.summary_text())
    print(f"\nReport written to {output_path}")
    return 0


def _run_kernelbench_eval(args) -> int:
    """Run KernelBench-style evaluation and compute fast_p scores."""
    from .kernelbench import evaluate_kernelbench

    operator = args.operator or None
    report = evaluate_kernelbench(
        operator=operator,
        gpu=args.gpu,
        max_configs_per_problem=args.configs_per_problem,
        baseline_source=getattr(args, "baseline", "measure"),
        timer=getattr(args, "timer", "cuda_event"),
    )

    output_path = _Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(report.summary_text())

    print(report.summary_text())
    print(f"\nFull report written to {output_path}")
    print(f"Summary written to {summary_path}")
    return 0


def _run_kernelbench_upstream_eval(args) -> int:
    """Task 4: run Noeris kernels against actual upstream L1 problems."""
    from .kernelbench_upstream import run_kernelbench_upstream_eval

    report = run_kernelbench_upstream_eval(
        gpu=args.gpu,
        timer=args.timer,
    )
    output_path = args.output or f"docs/results/kernelbench-upstream-l1-{args.gpu.lower()}.md"
    out_md = _Path(output_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report.summary_text())
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(report.summary_text())
    print(f"\nFull report: {out_json}")
    print(f"Summary:     {out_md}")
    return 0


def _run_generic_operator_iterate(*, spec, args, db, shapes) -> int:
    """Run kernel search for any operator via its TritonOperatorSpec."""
    from .modal_runner import run_benchmark_batch_modal_generic
    from .triton_operators import select_configs_for_operator

    operator_name = spec.name
    hardware = args.gpu

    # LLM proposer (if enabled) — runs first so its suggestions feed into selection
    proposed_configs: list[dict] = []
    proposal_result: dict = {"source": "none"}
    if args.llm:
        try:
            client = ResponsesApiClient.from_environment()
            proposal_result = _propose_operator_configs(
                spec=spec,
                proposer=client,
                database=db,
                hardware=hardware,
                target_shapes=shapes,
            )
            proposed_configs = proposal_result.get("configs", [])
        except LlmConfigurationError as exc:
            proposal_result = {"source": "llm_config_error", "error": str(exc)}

    # Load cost model if provided (passes to selector as a filter)
    cost_model = None
    if getattr(args, "cost_model", ""):
        from .cost_model import CostModel
        from pathlib import Path as _P
        cm_path = _P(args.cost_model)
        if cm_path.exists():
            try:
                cost_model = CostModel.load(cm_path)
            except Exception:
                cost_model = None

    # Use the generalized selector with full slotting logic
    if getattr(args, "bandit", False):
        from .bandit_selector import BanditSelector
        _bandit = BanditSelector()
        configs = _bandit.select_configs(
            spec=spec,
            database=db,
            hardware=hardware,
            shapes=shapes,
            max_configs=args.configs_per_run,
            proposed_configs=proposed_configs,
        )
    else:
        configs = select_configs_for_operator(
            spec=spec,
            database=db,
            hardware=hardware,
            shapes=shapes,
            max_configs=args.configs_per_run,
            proposed_configs=proposed_configs,
            cost_model=cost_model,
        )

    # Generate benchmark script
    benchmark_script = spec.benchmark_script_fn(configs, shapes)

    # Run on Modal (single batched call)
    if args.local:
        import subprocess, sys, tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(benchmark_script)
            script_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=300,
            )
            batch_stdout = result.stdout if result.returncode == 0 else ""
            batch_hardware = {}
        finally:
            Path(script_path).unlink(missing_ok=True)
        batch = run_benchmark_batch_modal_generic.__class__  # dummy
        # Parse output
        from .modal_runner import _extract_json_object
        data = _extract_json_object(batch_stdout, "config_results")
        if data is None:
            print(json.dumps({"error": "No JSON in local output", "stdout": batch_stdout[:500]}, indent=2))
            return 1
        config_results = data.get("config_results", [])
        batch_hardware = data.get("hardware", {})
        batch_success = True
    else:
        batch = run_benchmark_batch_modal_generic(
            benchmark_script=benchmark_script,
            gpu=args.gpu,
        )
        if not batch.success:
            print(json.dumps({
                "operator": operator_name,
                "error": batch.error,
                "success": False,
            }, indent=2))
            return 1
        config_results = batch.config_results
        batch_hardware = batch.hardware

    # Record results
    best_metric = 0.0
    best_config_id = ""
    output_results = []
    hw_name = batch_hardware.get("gpu", hardware)

    for config_result in config_results:
        cid = config_result.get("config_id", "")
        config = config_result.get("config", {})
        shape_results = config_result.get("results", [])

        recorded = []
        for shape_result in shape_results:
            if shape_result.get("correct") and shape_result.get("tflops"):
                shape_info = _parse_operator_shape(operator_name, shape_result["shape"])
                bucket = spec.shape_bucket_fn(shape_info)
                metric_value = shape_result["tflops"]  # tflops field = gb_per_s for memory-bound
                is_new_best = db.record_result(
                    shape=shape_info,
                    hardware=hw_name,
                    config=config,
                    tflops=metric_value,
                    ms=shape_result.get("ms", 0),
                    correct=True,
                    run_id=cid,
                    operator=operator_name,
                    bucket=bucket,
                    config_id_str=cid,
                )
                recorded.append({**shape_result, "new_best": is_new_best})

        correct_results = [
            r for r in shape_results if r.get("correct") and r.get("tflops")
        ]
        avg_metric = (
            sum(r["tflops"] for r in correct_results) / len(correct_results)
            if correct_results else 0.0
        )
        if avg_metric > best_metric:
            best_metric = avg_metric
            best_config_id = cid

        output_results.append({
            "config_id": cid,
            "config": config,
            "avg_metric": round(avg_metric, 2),
            "metric_unit": spec.metric_name,
            "shape_results": recorded,
        })

    db.save()

    output = {
        "operator": operator_name,
        "metric_unit": spec.metric_name,
        "hardware": hw_name,
        "configs_tested": len(configs),
        "best_config_id": best_config_id,
        f"best_avg_{spec.metric_name}": round(best_metric, 2),
        "proposal": proposal_result,
        "results": output_results,
        "database_insights": db.get_insights(hardware=hw_name, operator=operator_name),
    }
    print(json.dumps(output, indent=2))
    return 0


def _parse_operator_shape(operator_name: str, shape_str: str) -> dict:
    """Parse shape string like '2048x2048x2048' or '4096x768' into a dict."""
    parts = shape_str.split("x")
    if operator_name == "matmul":
        return {"M": int(parts[0]), "N": int(parts[1]), "K": int(parts[2])}
    elif operator_name in ("rmsnorm", "layernorm"):
        return {"n_rows": int(parts[0]), "hidden_dim": int(parts[1])}
    elif operator_name in ("softmax", "cross_entropy"):
        return {"n_rows": int(parts[0]), "n_cols": int(parts[1])}
    elif operator_name == "attention":
        return {
            "batch": int(parts[0]),
            "heads": int(parts[1]),
            "seq_len": int(parts[2]),
            "head_dim": int(parts[3]),
        }
    elif operator_name == "rotary":
        return {
            "batch": int(parts[0]),
            "seq": int(parts[1]),
            "heads": int(parts[2]),
            "head_dim": int(parts[3]),
        }
    elif operator_name == "geglu":
        return {"n_rows": int(parts[0]), "ffn_dim": int(parts[1])}
    return {"raw": shape_str}


def _propose_operator_configs(
    *, spec, proposer, database, hardware: str, target_shapes: list,
) -> dict:
    """Generic LLM proposer that works for any operator spec."""
    insights = database.get_insights(hardware=hardware, operator=spec.name)

    prompt_data = {
        "operator": spec.name,
        "description": spec.description,
        "metric": spec.metric_name,
        "param_space": spec.param_space,
        "target_shapes": target_shapes[:6],
        "curated_configs_already_tested": [
            spec.config_id_fn(c) for c in spec.curated_configs
        ],
    }
    if insights:
        prompt_data["cross_run_insights"] = insights

    # Build schema dynamically from param space
    param_props = {
        pname: {"type": "integer"} for pname in spec.param_space.keys()
    }
    param_props["rationale"] = {"type": "string"}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "configs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": param_props,
                    "required": list(param_props.keys()),
                },
            },
            "global_rationale": {"type": "string"},
        },
        "required": ["configs", "global_rationale"],
    }

    try:
        payload = proposer.generate_json(
            schema_name=f"{spec.name}_config_proposals",
            schema=schema,
            instructions=(
                f"You are proposing Triton {spec.name} kernel configurations. "
                f"The operator is {spec.description} Different shapes often have different "
                "optimal configs. Use cross_run_insights to see what has worked and propose "
                "configs that explore promising unexplored regions. "
                "Return at most 4 configs with one-sentence rationale each."
            ),
            prompt=json.dumps(prompt_data, indent=2),
            max_output_tokens=1200,
            reasoning_effort="low",
            text_verbosity="low",
        )
    except Exception as exc:
        return {
            "source": "responses_api_error",
            "configs": [],
            "error": type(exc).__name__,
            "detail": str(exc)[:240],
        }

    configs = []
    for item in payload.get("configs", []):
        if not isinstance(item, dict):
            continue
        config = {}
        valid = True
        for pname in spec.param_space.keys():
            val = item.get(pname)
            if not isinstance(val, int) or val not in spec.param_space[pname]:
                valid = False
                break
            config[pname] = val
        # Note: we no longer filter by spec.shared_memory_check_fn here.
        # Feasibility is learned by the bandit from runtime failures (reward=0).
        if valid:
            configs.append(config)

    return {
        "source": "responses_api",
        "configs": configs[:4],
        "global_rationale": " ".join(str(payload.get("global_rationale", "")).split()),
    }


def _run_triton_iterate(args) -> int:
    """Run the Triton kernel search loop for any registered operator."""
    from .modal_runner import (
        run_benchmark_batch_modal,
        run_benchmark_batch_modal_generic,
    )
    from .triton_operators import REGISTRY

    operator_name = getattr(args, "operator", "matmul")
    spec = REGISTRY.get(operator_name)

    db = ConfigDatabase(path=args.db_path)
    hardware = args.gpu

    # Select shape set based on operator's shape_buckets
    if args.shapes == "tiny":
        shapes = spec.shape_buckets[:2]
    elif args.shapes == "full":
        shapes = spec.shape_buckets
    else:
        shapes = spec.shape_buckets[:6]

    # Dispatch to operator-specific flow for non-matmul operators
    if operator_name != "matmul":
        return _run_generic_operator_iterate(
            spec=spec,
            args=args,
            db=db,
            shapes=shapes,
        )

    # Get LLM-proposed configs if enabled
    proposed_configs: list[dict[str, int]] = []
    proposal_result: dict = {"source": "none"}
    if args.llm:
        try:
            client = ResponsesApiClient.from_environment()
            proposal_result = propose_triton_configs(
                proposer=client,
                database=db,
                hardware=hardware,
                target_shapes=shapes,
            )
            proposed_configs = proposal_result.get("configs", [])
        except LlmConfigurationError as exc:
            proposal_result = {"source": "llm_config_error", "error": str(exc)}

    # Load cost model if provided
    cost_model = None
    if getattr(args, "cost_model", ""):
        from .cost_model import CostModel
        from pathlib import Path as _P
        cm_path = _P(args.cost_model)
        if cm_path.exists():
            try:
                cost_model = CostModel.load(cm_path)
            except Exception:
                cost_model = None

    # Select configs for this run
    matmul_shapes = [{"M": s["M"], "N": s["N"], "K": s["K"]} for s in shapes]
    if getattr(args, "bandit", False):
        from .bandit_selector import BanditSelector
        from .triton_operators import REGISTRY as _REGISTRY
        _bandit = BanditSelector()
        configs = _bandit.select_configs(
            spec=_REGISTRY.get("matmul"),
            database=db,
            hardware=hardware,
            shapes=matmul_shapes,
            max_configs=args.configs_per_run,
            proposed_configs=proposed_configs,
        )
    else:
        configs = select_configs_for_run(
            database=db,
            hardware=hardware,
            shapes=matmul_shapes,
            max_configs=args.configs_per_run,
            proposed_configs=proposed_configs,
        )
    # If cost model present, re-rank grid candidates (matmul path uses its own
    # select_configs_for_run; the generic path uses select_configs_for_operator).
    # The cost model is most valuable in the generic path below.

    # Run ALL configs in a single GPU call (one cold start)
    if args.local:
        # Local mode: import and run locally
        from .modal_runner import run_benchmark_local
        results = []
        for config in configs:
            result = run_benchmark_local(config, shapes=shapes)
            results.append({
                "config_id": result.config_id,
                "config": result.config,
                "results": result.results if result.success else [],
                "success": result.success,
                "error": result.error,
            })
        batch_hardware = {}
        batch_success = any(r["success"] for r in results)
    else:
        batch = run_benchmark_batch_modal(configs, shapes=shapes, gpu=args.gpu)
        results = batch.config_results if batch.success else []
        batch_hardware = batch.hardware
        batch_success = batch.success
        if not batch_success:
            print(json.dumps({
                "benchmark": "triton-matmul",
                "error": batch.error,
                "success": False,
            }, indent=2))
            return 1

    # Record results to database
    best_tflops = 0.0
    best_config_id_found = ""
    output_results = []

    for config_result in results:
        cid = config_result.get("config_id", "")
        config = config_result.get("config", {})
        shape_results = config_result.get("results", [])

        recorded = []
        for shape_result in shape_results:
            if shape_result.get("correct") and shape_result.get("tflops"):
                parts = shape_result["shape"].split("x")
                shape_info = {"M": int(parts[0]), "N": int(parts[1]), "K": int(parts[2])}
                hw = batch_hardware.get("gpu", hardware)
                is_new_best = db.record_result(
                    shape=shape_info,
                    hardware=hw,
                    config=config,
                    tflops=shape_result["tflops"],
                    ms=shape_result.get("ms", 0),
                    correct=True,
                    run_id=cid,
                )
                recorded.append({**shape_result, "new_best": is_new_best})

        correct_results = [r for r in shape_results if r.get("correct") and r.get("tflops")]
        avg_tflops = (
            sum(r["tflops"] for r in correct_results) / len(correct_results)
            if correct_results else 0.0
        )
        if avg_tflops > best_tflops:
            best_tflops = avg_tflops
            best_config_id_found = cid

        output_results.append({
            "config_id": cid,
            "config": config,
            "avg_tflops": round(avg_tflops, 2),
            "shape_results": recorded,
        })

    db.save()

    output = {
        "benchmark": "triton-matmul",
        "hardware": batch_hardware.get("gpu", hardware),
        "configs_tested": len(configs),
        "best_config_id": best_config_id_found,
        "best_avg_tflops": round(best_tflops, 2),
        "proposal": proposal_result,
        "results": output_results,
        "database_insights": db.get_insights(hardware=batch_hardware.get("gpu", hardware)),
    }
    print(json.dumps(output, indent=2))
    return 0


def build_pipeline(
    *,
    use_llm: bool,
    max_results: int,
    live_execution: bool,
    benchmark_id: str | None,
) -> ResearchPipeline:
    if not use_llm and not live_execution:
        return ResearchPipeline()

    kwargs = {}
    client = None
    llm_planning_enabled = use_llm and benchmark_id != "matmul-speedup"
    if use_llm or (live_execution and benchmark_id in {"long-context-reasoning", "tool-use-reliability"}):
        client = ResponsesApiClient.from_environment()
    if llm_planning_enabled:
        kwargs["source_provider"] = CompositeSourceProvider(
            providers=[
                ArxivAtomSourceProvider(client=UrllibHttpClient(), max_results=max_results),
                GitHubRepositorySourceProvider(
                    client=UrllibHttpClient(),
                    max_results=max_results,
                ),
            ]
        )
        kwargs["research_memory"] = LlmResearchMemory(client=client, max_sources=max_results * 2)
        kwargs["hypothesis_planner"] = LlmHypothesisPlanner(client=client)
    if live_execution:
        matmul_history = None
        if benchmark_id == "matmul-speedup":
            matmul_history = JsonFileRunStore().summarize_history(
                benchmark_id="matmul-speedup",
                limit=20,
            )
        kwargs["experiment_executor"] = DefaultExperimentExecutor(
            long_context_executor=(
                LongContextResponsesExecutor(client=client)
                if benchmark_id == "long-context-reasoning" and client is not None
                else DefaultExperimentExecutor().long_context_executor
            ),
            tool_use_executor=(
                ToolUseResponsesExecutor(client=client)
                if benchmark_id == "tool-use-reliability" and client is not None
                else DefaultExperimentExecutor().tool_use_executor
            ),
            matmul_executor=(
                MatmulPythonExecutor(
                    history_summary=matmul_history,
                    proposer=client if use_llm and client is not None else None,
                )
                if benchmark_id == "matmul-speedup"
                else DefaultExperimentExecutor().matmul_executor
            ),
        )
    return ResearchPipeline(**kwargs)


def _extract_benchmark_metric(record) -> float:
    result = record.memo.results[0]
    payloads = result.artifact_payloads
    if record.benchmark_id == "matmul-speedup":
        return float(payloads.get("raw-timing-results.json", {}).get("mean_uplift_pct", 0.0))
    if record.benchmark_id == "long-context-reasoning":
        return float(payloads.get("candidate-metrics.json", {}).get("accuracy", 0.0))
    if record.benchmark_id == "tool-use-reliability":
        return float(payloads.get("tool-selection-summary.json", {}).get("terminal_first_success_rate", 0.0))
    return 0.0


def _extract_frontier_snapshot(
    store: JsonFileRunStore,
    *,
    benchmark_id: str,
    run_id: str,
) -> dict[str, object]:
    if not run_id:
        return {}
    try:
        record = store.load(run_id)
    except FileNotFoundError:
        return {}
    return _extract_frontier_snapshot_from_record(record, benchmark_id=benchmark_id)


def _extract_frontier_snapshot_from_record(
    record,
    *,
    benchmark_id: str,
) -> dict[str, object]:
    if record is None or not record.memo.results:
        return {}
    payloads = record.memo.results[0].artifact_payloads
    snapshot: dict[str, object] = {
        "run_id": record.run_id,
        "metric": _extract_benchmark_metric(record),
    }
    if benchmark_id == "matmul-speedup":
        snapshot.update(
            {
                "best_candidate_id": payloads.get("best-candidate-summary.json", {}).get(
                    "best_overall_candidate_id",
                    "",
                ),
                "selected_candidate_ids": [
                    candidate.get("id", "")
                    for candidate in payloads.get("candidate-catalog.json", {}).get(
                        "selected_candidates",
                        [],
                    )
                    if isinstance(candidate, dict) and candidate.get("id")
                ],
                "proposal_source": payloads.get("candidate-proposals.json", {}).get("source", ""),
                "frontier_archive": payloads.get("frontier-archive.json", {}).get(
                    "workload_winners",
                    [],
                ),
                "pareto_candidate_ids": payloads.get("pareto-frontier.json", {}).get(
                    "candidate_ids",
                    [],
                ),
            }
        )
    return snapshot


def _build_frontier_delta(
    *,
    previous_frontier: dict[str, object],
    best_frontier: dict[str, object],
) -> dict[str, object]:
    previous_candidates = {
        candidate_id
        for candidate_id in previous_frontier.get("selected_candidate_ids", [])
        if isinstance(candidate_id, str) and candidate_id
    }
    best_candidates = {
        candidate_id
        for candidate_id in best_frontier.get("selected_candidate_ids", [])
        if isinstance(candidate_id, str) and candidate_id
    }
    previous_pareto = {
        candidate_id
        for candidate_id in previous_frontier.get("pareto_candidate_ids", [])
        if isinstance(candidate_id, str) and candidate_id
    }
    best_pareto = {
        candidate_id
        for candidate_id in best_frontier.get("pareto_candidate_ids", [])
        if isinstance(candidate_id, str) and candidate_id
    }
    previous_best_candidate_id = str(previous_frontier.get("best_candidate_id", "")).strip()
    best_candidate_id = str(best_frontier.get("best_candidate_id", "")).strip()
    previous_archive = {
        str(item.get("workload_tag", "")).strip(): str(item.get("best_candidate_id", "")).strip()
        for item in previous_frontier.get("frontier_archive", [])
        if isinstance(item, dict)
        and str(item.get("workload_tag", "")).strip()
        and str(item.get("best_candidate_id", "")).strip()
    }
    best_archive = {
        str(item.get("workload_tag", "")).strip(): str(item.get("best_candidate_id", "")).strip()
        for item in best_frontier.get("frontier_archive", [])
        if isinstance(item, dict)
        and str(item.get("workload_tag", "")).strip()
        and str(item.get("best_candidate_id", "")).strip()
    }
    workload_changes = [
        {
            "workload_tag": workload_tag,
            "previous_best_candidate_id": previous_archive.get(workload_tag, ""),
            "best_candidate_id": best_archive.get(workload_tag, ""),
        }
        for workload_tag in sorted(set(previous_archive) | set(best_archive))
        if previous_archive.get(workload_tag, "") != best_archive.get(workload_tag, "")
    ]
    return {
        "best_candidate_changed": (
            previous_best_candidate_id != best_candidate_id
            if previous_best_candidate_id or best_candidate_id
            else False
        ),
        "previous_best_candidate_id": previous_best_candidate_id,
        "best_candidate_id": best_candidate_id,
        "candidate_set_changed": previous_candidates != best_candidates,
        "added_candidates": sorted(best_candidates - previous_candidates),
        "dropped_candidates": sorted(previous_candidates - best_candidates),
        "pareto_frontier_changed": previous_pareto != best_pareto,
        "added_pareto_candidates": sorted(best_pareto - previous_pareto),
        "dropped_pareto_candidates": sorted(previous_pareto - best_pareto),
        "workload_frontier_changed": bool(workload_changes),
        "workload_changes": workload_changes,
        "proposal_source": best_frontier.get("proposal_source", ""),
    }


if __name__ == "__main__":
    sys.exit(main())
