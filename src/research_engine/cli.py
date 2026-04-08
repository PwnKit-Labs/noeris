from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from .agenda import DEFAULT_RESEARCH_AGENDA
from .benchmarks import DEFAULT_BENCHMARKS, get_benchmark
from .codex_config import load_codex_provider_config, render_github_env_setup
from .export import export_run_bundle
from .ingestion import (
    ArxivAtomSourceProvider,
    CompositeSourceProvider,
    GitHubRepositorySourceProvider,
    UrllibHttpClient,
)
from .models import ResearchTopic
from .pipeline import ResearchPipeline
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

    subparsers.add_parser("runs", help="list persisted research runs")

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
        pipeline = ResearchPipeline()
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
        pipeline = ResearchPipeline()
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

    pipeline = ResearchPipeline()
    topic = ResearchTopic(
        name=args.topic,
        objective=args.objective,
        constraints=args.constraint,
    )
    print(json.dumps(pipeline.run_cycle_dict(topic), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
