from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import (
    Claim,
    Contradiction,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    Hypothesis,
    ResearchContext,
    ResearchCycle,
    ResearchMemo,
    ResearchRunRecord,
    ResearchSource,
    ResearchTopic,
    SourceAssessment,
    VerificationReport,
)


def _json_default(value: object) -> object:
    if isinstance(value, ExperimentStatus):
        return value.value
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def serialize_run_record(record: ResearchRunRecord) -> dict:
    return json.loads(json.dumps(asdict(record), default=_json_default))


def deserialize_run_record(payload: dict) -> ResearchRunRecord:
    cycle_payload = payload["cycle"]
    context_payload = cycle_payload["context"]
    memo_payload = payload["memo"]

    topic = ResearchTopic(**cycle_payload["topic"])
    context_sources = [_deserialize_source(item) for item in context_payload["sources"]]
    context_claims = [_deserialize_claim(item) for item in context_payload["claims"]]
    context = ResearchContext(
        topic=context_payload["topic"],
        sources=context_sources,
        claims=context_claims,
        open_questions=context_payload.get("open_questions", []),
        contradictions=[
            _deserialize_contradiction(item)
            for item in context_payload.get("contradictions", [])
        ],
        source_assessments=[
            _deserialize_source_assessment(item)
            for item in context_payload.get("source_assessments", [])
        ],
    )
    cycle = ResearchCycle(
        topic=topic,
        context=context,
        hypotheses=[Hypothesis(**item) for item in cycle_payload["hypotheses"]],
        experiments=[_deserialize_experiment_spec(item) for item in cycle_payload["experiments"]],
        results=[_deserialize_result(item) for item in cycle_payload["results"]],
    )
    verification = VerificationReport(**payload["verification"])
    memo = ResearchMemo(
        topic=memo_payload["topic"],
        summary=memo_payload["summary"],
        sources=[_deserialize_source(item) for item in memo_payload["sources"]],
        source_assessments=[
            _deserialize_source_assessment(item)
            for item in memo_payload.get("source_assessments", [])
        ],
        claims=[_deserialize_claim(item) for item in memo_payload["claims"]],
        contradictions=[
            _deserialize_contradiction(item)
            for item in memo_payload.get("contradictions", [])
        ],
        hypotheses=[Hypothesis(**item) for item in memo_payload["hypotheses"]],
        experiments=[_deserialize_experiment_spec(item) for item in memo_payload["experiments"]],
        results=[_deserialize_result(item) for item in memo_payload["results"]],
        next_actions=memo_payload["next_actions"],
        risks=memo_payload.get("risks", []),
    )
    return ResearchRunRecord(
        run_id=payload["run_id"],
        created_at=payload["created_at"],
        benchmark_id=payload.get("benchmark_id"),
        cycle=cycle,
        verification=verification,
        memo=memo,
    )


def _deserialize_source(payload: dict) -> ResearchSource:
    return ResearchSource(**payload)


def _deserialize_claim(payload: dict) -> Claim:
    return Claim(**payload)


def _deserialize_source_assessment(payload: dict) -> SourceAssessment:
    return SourceAssessment(**payload)


def _deserialize_contradiction(payload: dict) -> Contradiction:
    if isinstance(payload, str):
        return Contradiction(
            title=payload,
            summary=payload,
            claim_titles=[],
            severity="medium",
        )
    return Contradiction(**payload)


def _deserialize_experiment_spec(payload: dict) -> ExperimentSpec:
    return ExperimentSpec(
        name=payload["name"],
        benchmark_id=payload.get("benchmark_id"),
        hypothesis_title=payload["hypothesis_title"],
        success_metric=payload["success_metric"],
        budget=payload["budget"],
        baseline=payload.get("baseline", ""),
        protocol=payload["protocol"],
        required_artifacts=payload.get("required_artifacts", []),
        evaluation_notes=payload.get("evaluation_notes", []),
    )


def _deserialize_result(payload: dict) -> ExperimentResult:
    return ExperimentResult(
        spec_name=payload["spec_name"],
        status=ExperimentStatus(payload["status"]),
        outcome_summary=payload["outcome_summary"],
        artifact_refs=payload.get("artifact_refs", []),
        artifact_payloads=payload.get("artifact_payloads", {}),
    )


class JsonFileRunStore:
    def __init__(self, base_dir: str | Path = ".noeris/runs") -> None:
        self.base_dir = Path(base_dir)

    def save(self, record: ResearchRunRecord) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{record.run_id}.json"
        path.write_text(
            json.dumps(serialize_run_record(record), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, run_id: str) -> ResearchRunRecord:
        path = self.base_dir / f"{run_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_run_record(payload)

    def list_runs(self) -> list[dict[str, str]]:
        if not self.base_dir.exists():
            return []

        summaries: list[dict[str, str]] = []
        for path in sorted(self.base_dir.glob("*.json"), reverse=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_id": payload["run_id"],
                    "created_at": payload["created_at"],
                    "benchmark_id": payload.get("benchmark_id") or "",
                    "topic": payload["cycle"]["topic"]["name"],
                    "path": str(path),
                }
            )
        return summaries

    def summarize_history(
        self,
        *,
        benchmark_id: str | None = None,
        topic: str | None = None,
        limit: int = 5,
    ) -> dict[str, object]:
        records = self._load_matching_runs(benchmark_id=benchmark_id, topic=topic)[:limit]
        latest = records[0] if records else None
        previous = records[1] if len(records) > 1 else None

        latest_claim_titles = {claim.title for claim in latest.memo.claims} if latest else set()
        previous_claim_titles = {claim.title for claim in previous.memo.claims} if previous else set()
        latest_source_ids = {source.identifier for source in latest.memo.sources} if latest else set()
        previous_source_ids = {source.identifier for source in previous.memo.sources} if previous else set()
        latest_assessments = {
            assessment.source_id: assessment.confidence
            for assessment in (latest.memo.source_assessments if latest else [])
        }
        previous_assessments = {
            assessment.source_id: assessment.confidence
            for assessment in (previous.memo.source_assessments if previous else [])
        }
        latest_evidence_kinds = {
            claim.title: claim.evidence_kind
            for claim in (latest.memo.claims if latest else [])
        }
        previous_evidence_kinds = {
            claim.title: claim.evidence_kind
            for claim in (previous.memo.claims if previous else [])
        }
        latest_contradiction_titles = {
            contradiction.title for contradiction in (latest.memo.contradictions if latest else [])
        }
        previous_contradiction_titles = {
            contradiction.title for contradiction in (previous.memo.contradictions if previous else [])
        }
        latest_source_updates = {
            source.identifier: source.updated_at
            for source in (latest.memo.sources if latest else [])
            if source.updated_at
        }
        latest_source_titles = {
            source.identifier: source.title
            for source in (latest.memo.sources if latest else [])
        }
        latest_fp8_layout_summary = _extract_fp8_layout_summary_from_record(latest)
        previous_fp8_layout_summary = _extract_fp8_layout_summary_from_record(previous)

        confidence_changes = []
        for source_id in sorted(set(latest_assessments) & set(previous_assessments)):
            if latest_assessments[source_id] == previous_assessments[source_id]:
                continue
            confidence_changes.append(
                {
                    "source_id": source_id,
                    "previous_confidence": previous_assessments[source_id],
                    "latest_confidence": latest_assessments[source_id],
                }
            )

        evidence_kind_changes = []
        for claim_title in sorted(set(latest_evidence_kinds) & set(previous_evidence_kinds)):
            if latest_evidence_kinds[claim_title] == previous_evidence_kinds[claim_title]:
                continue
            evidence_kind_changes.append(
                {
                    "claim_title": claim_title,
                    "previous_evidence_kind": previous_evidence_kinds[claim_title],
                    "latest_evidence_kind": latest_evidence_kinds[claim_title],
                }
            )

        matmul_candidate_wins: dict[str, int] = {}
        matmul_family_wins: dict[str, int] = {}
        matmul_shape_winners: dict[str, dict[str, object]] = {}
        matmul_shape_challengers: dict[str, dict[str, object]] = {}
        matmul_workload_winners: dict[str, dict[str, object]] = {}
        matmul_workload_challengers: dict[str, dict[str, object]] = {}
        matmul_frontier_archive: list[dict[str, object]] = []
        matmul_pareto_candidate_ids: list[str] = []
        fp8_layout_counts: dict[str, int] = {}
        fp8_weighted_share_by_layout: dict[str, float] = {}
        fp8_total_fixture_count = 0
        fp8_reuse_bucket_totals: dict[str, int] = {}
        fp8_reuse_bucket_layout_wins: dict[str, dict[str, int]] = {}
        if benchmark_id == "matmul-speedup":
            for record in records:
                for result in record.memo.results:
                    payload = result.artifact_payloads.get("best-candidate-summary.json")
                    if not isinstance(payload, dict):
                        continue
                    if not matmul_pareto_candidate_ids:
                        pareto_ids = payload.get("pareto_candidate_ids", [])
                        if isinstance(pareto_ids, list):
                            matmul_pareto_candidate_ids = [
                                item for item in pareto_ids if isinstance(item, str) and item
                            ]
                    winner_counts = payload.get("winner_counts", {})
                    if not isinstance(winner_counts, dict):
                        continue
                    for candidate_id, count in winner_counts.items():
                        try:
                            matmul_candidate_wins[str(candidate_id)] = (
                                matmul_candidate_wins.get(str(candidate_id), 0) + int(count)
                            )
                        except (TypeError, ValueError):
                            continue
                    raw_rows = result.artifact_payloads.get("raw-timing-results.json", {}).get("rows", [])
                    if not matmul_frontier_archive:
                        archive_payload = result.artifact_payloads.get("frontier-archive.json", {})
                        if isinstance(archive_payload, dict):
                            workload_winners = archive_payload.get("workload_winners", [])
                            if isinstance(workload_winners, list):
                                matmul_frontier_archive = [
                                    item for item in workload_winners if isinstance(item, dict)
                                ]
                    if isinstance(raw_rows, list):
                        for row in raw_rows:
                            if not isinstance(row, dict):
                                continue
                            shape = str(row.get("shape", "")).strip()
                            workload_tag = str(row.get("workload_tag", "")).strip()
                            workload_share = row.get("workload_share")
                            winner = str(row.get("best_candidate_id", "")).strip()
                            uplift = row.get("uplift_pct")
                            if not shape or not winner:
                                continue
                            candidate_result_rows = row.get("candidate_results", [])
                            if not isinstance(candidate_result_rows, list):
                                candidate_result_rows = []
                            family_by_candidate = {
                                str(candidate_row.get("candidate_id", "")).strip(): str(
                                    candidate_row.get("candidate_family", "")
                                ).strip()
                                for candidate_row in candidate_result_rows
                                if isinstance(candidate_row, dict)
                            }
                            winner_family = family_by_candidate.get(winner, "")
                            if winner_family:
                                matmul_family_wins[winner_family] = (
                                    matmul_family_wins.get(winner_family, 0) + 1
                                )
                            entry = matmul_shape_winners.setdefault(
                                shape,
                                {"winner_counts": {}, "latest_winner": winner, "latest_uplift_pct": uplift},
                            )
                            entry["winner_counts"][winner] = entry["winner_counts"].get(winner, 0) + 1
                            if workload_tag:
                                workload_entry = matmul_workload_winners.setdefault(
                                    workload_tag,
                                    {
                                        "winner_counts": {},
                                        "latest_winner": winner,
                                        "latest_uplift_pct": uplift,
                                        "latest_workload_share": workload_share,
                                    },
                                )
                                workload_entry["winner_counts"][winner] = (
                                    workload_entry["winner_counts"].get(winner, 0) + 1
                                )
                            runner_up = str(row.get("runner_up_candidate_id", "")).strip()
                            runner_up_gap_pct = row.get("runner_up_gap_pct")
                            if runner_up:
                                challenger_entry = matmul_shape_challengers.setdefault(
                                    shape,
                                    {
                                        "runner_up_counts": {},
                                        "latest_runner_up": runner_up,
                                        "latest_runner_up_gap_pct": runner_up_gap_pct,
                                    },
                                )
                                challenger_entry["runner_up_counts"][runner_up] = (
                                    challenger_entry["runner_up_counts"].get(runner_up, 0) + 1
                                )
                                if workload_tag:
                                    workload_challenger_entry = matmul_workload_challengers.setdefault(
                                        workload_tag,
                                        {
                                            "runner_up_counts": {},
                                            "latest_runner_up": runner_up,
                                            "latest_runner_up_gap_pct": runner_up_gap_pct,
                                            "latest_workload_share": workload_share,
                                        },
                                    )
                                    workload_challenger_entry["runner_up_counts"][runner_up] = (
                                        workload_challenger_entry["runner_up_counts"].get(runner_up, 0)
                                        + 1
                                    )

                    fp8_layout_payload = result.artifact_payloads.get("fp8-runtime-layout-summary.json")
                    if isinstance(fp8_layout_payload, dict):
                        fp8_total_fixture_count += int(fp8_layout_payload.get("fp8_fixture_count", 0) or 0)
                        layout_counts = fp8_layout_payload.get("layout_counts", {})
                        if isinstance(layout_counts, dict):
                            for layout, count in layout_counts.items():
                                layout_key = str(layout)
                                try:
                                    fp8_layout_counts[layout_key] = fp8_layout_counts.get(layout_key, 0) + int(count)
                                except (TypeError, ValueError):
                                    continue
                        weighted_share = fp8_layout_payload.get("weighted_share_by_layout", {})
                        if isinstance(weighted_share, dict):
                            for layout, share in weighted_share.items():
                                layout_key = str(layout)
                                try:
                                    fp8_weighted_share_by_layout[layout_key] = (
                                        fp8_weighted_share_by_layout.get(layout_key, 0.0) + float(share)
                                    )
                                except (TypeError, ValueError):
                                    continue
                        fixtures = fp8_layout_payload.get("fixtures", [])
                        if isinstance(fixtures, list):
                            for fixture in fixtures:
                                if not isinstance(fixture, dict):
                                    continue
                                layout = str(fixture.get("layout", "")).strip()
                                if not layout:
                                    continue
                                try:
                                    reuse = int(fixture.get("expected_weight_reuse", 1))
                                except (TypeError, ValueError):
                                    reuse = 1
                                if reuse <= 1:
                                    bucket = "reuse_1"
                                elif reuse <= 4:
                                    bucket = "reuse_2_4"
                                else:
                                    bucket = "reuse_5_plus"
                                fp8_reuse_bucket_totals[bucket] = fp8_reuse_bucket_totals.get(bucket, 0) + 1
                                bucket_layouts = fp8_reuse_bucket_layout_wins.setdefault(bucket, {})
                                bucket_layouts[layout] = bucket_layouts.get(layout, 0) + 1

        source_freshness = _summarize_source_freshness(latest.memo.sources if latest else [])
        ranked_sources = _rank_sources_for_history(
            sources=latest.memo.sources if latest else [],
            assessments=latest.memo.source_assessments if latest else [],
            claims=latest.memo.claims if latest else [],
            contradictions=latest.memo.contradictions if latest else [],
        )
        fp8_policy_alignment = _summarize_fp8_policy_alignment(
            reuse_bucket_totals=fp8_reuse_bucket_totals,
            reuse_bucket_layout_wins=fp8_reuse_bucket_layout_wins,
            layout_counts=fp8_layout_counts,
        )
        fp8_latest_alignment = _alignment_from_layout_summary(latest_fp8_layout_summary)
        fp8_previous_alignment = _alignment_from_layout_summary(previous_fp8_layout_summary)
        fp8_policy_regressions = _detect_fp8_policy_regressions(
            latest_alignment=fp8_latest_alignment,
            previous_alignment=fp8_previous_alignment,
        )

        return {
            "benchmark_id": benchmark_id or (latest.benchmark_id if latest else ""),
            "topic": topic or (latest.memo.topic if latest else ""),
            "run_count": len(records),
            "latest_run_id": latest.run_id if latest else "",
            "previous_run_id": previous.run_id if previous else "",
            "latest_source_updates": latest_source_updates,
            "latest_source_titles": latest_source_titles,
            "source_freshness": source_freshness,
            "ranked_sources": ranked_sources,
            "new_source_ids": sorted(latest_source_ids - previous_source_ids),
            "dropped_source_ids": sorted(previous_source_ids - latest_source_ids),
            "shared_source_ids": sorted(latest_source_ids & previous_source_ids),
            "new_claim_titles": sorted(latest_claim_titles - previous_claim_titles),
            "dropped_claim_titles": sorted(previous_claim_titles - latest_claim_titles),
            "shared_claim_titles": sorted(latest_claim_titles & previous_claim_titles),
            "confidence_changes": confidence_changes,
            "evidence_kind_changes": evidence_kind_changes,
            "new_contradiction_titles": sorted(
                latest_contradiction_titles - previous_contradiction_titles
            ),
            "dropped_contradiction_titles": sorted(
                previous_contradiction_titles - latest_contradiction_titles
            ),
            "latest_contradictions": [
                {
                    "title": contradiction.title,
                    "severity": contradiction.severity,
                    "summary": contradiction.summary,
                }
                for contradiction in (latest.memo.contradictions if latest else [])
            ],
            "matmul_candidate_wins": matmul_candidate_wins,
            "matmul_family_wins": matmul_family_wins,
            "best_matmul_candidate_id": (
                max(matmul_candidate_wins, key=matmul_candidate_wins.get)
                if matmul_candidate_wins
                else ""
            ),
            "best_benchmark_metric": _best_metric(records, benchmark_id),
            "matmul_shape_winners": matmul_shape_winners,
            "matmul_shape_challengers": matmul_shape_challengers,
            "matmul_workload_winners": matmul_workload_winners,
            "matmul_workload_challengers": matmul_workload_challengers,
            "matmul_frontier_archive": matmul_frontier_archive,
            "matmul_pareto_candidate_ids": matmul_pareto_candidate_ids,
            "fp8_layout_counts": fp8_layout_counts,
            "fp8_weighted_share_by_layout": {
                layout: round(value, 4)
                for layout, value in fp8_weighted_share_by_layout.items()
            },
            "fp8_total_fixture_count": fp8_total_fixture_count,
            "fp8_reuse_bucket_totals": fp8_reuse_bucket_totals,
            "fp8_reuse_bucket_layout_wins": fp8_reuse_bucket_layout_wins,
            "fp8_policy_alignment": fp8_policy_alignment,
            "fp8_latest_alignment": fp8_latest_alignment,
            "fp8_previous_alignment": fp8_previous_alignment,
            "fp8_policy_regressions": fp8_policy_regressions,
            "weakest_matmul_shapes": sorted(
                [
                    {
                        "shape": shape,
                        "runner_up_candidate_id": entry.get("latest_runner_up", ""),
                        "runner_up_gap_pct": entry.get("latest_runner_up_gap_pct", 0),
                    }
                    for shape, entry in matmul_shape_challengers.items()
                ],
                key=lambda item: item.get("runner_up_gap_pct", 10**9),
            ),
            "weakest_matmul_workloads": sorted(
                [
                    {
                        "workload_tag": workload_tag,
                        "runner_up_candidate_id": entry.get("latest_runner_up", ""),
                        "runner_up_gap_pct": entry.get("latest_runner_up_gap_pct", 0),
                        "workload_share": entry.get("latest_workload_share"),
                    }
                    for workload_tag, entry in matmul_workload_challengers.items()
                ],
                key=lambda item: item.get("runner_up_gap_pct", 10**9),
            ),
        }

    def render_history_brief(
        self,
        *,
        benchmark_id: str | None = None,
        topic: str | None = None,
        limit: int = 5,
    ) -> str:
        summary = self.summarize_history(
            benchmark_id=benchmark_id,
            topic=topic,
            limit=limit,
        )
        lines = [
            "# History Brief",
            "",
            f"- Benchmark: `{summary.get('benchmark_id') or 'none'}`",
            f"- Topic: `{summary.get('topic') or 'none'}`",
            f"- Runs compared: `{summary.get('run_count', 0)}`",
            "",
        ]
        if summary.get("new_claim_titles") or summary.get("dropped_claim_titles"):
            lines.extend(["## Claim Changes", ""])
            lines.extend(f"- New claim: `{item}`" for item in summary.get("new_claim_titles", []))
            lines.extend(f"- Dropped claim: `{item}`" for item in summary.get("dropped_claim_titles", []))
            lines.append("")
        if summary.get("new_source_ids") or summary.get("dropped_source_ids"):
            lines.extend(["## Source Changes", ""])
            lines.extend(f"- New source: `{item}`" for item in summary.get("new_source_ids", []))
            lines.extend(f"- Dropped source: `{item}`" for item in summary.get("dropped_source_ids", []))
            lines.append("")
        freshness = summary.get("source_freshness") or {}
        if freshness.get("source_count_with_timestamps", 0):
            lines.extend(["## Source Freshness", ""])
            lines.append(
                f"- Sources with timestamps: `{freshness.get('source_count_with_timestamps', 0)}`"
            )
            if freshness.get("bucket_counts"):
                bucket_bits = ", ".join(
                    f"{key}={value}" for key, value in freshness["bucket_counts"].items()
                )
                lines.append(f"- Buckets: `{bucket_bits}`")
            if freshness.get("newest_source_id"):
                lines.append(
                    f"- Newest source: `{freshness['newest_source_id']}` at `{freshness['newest_updated_at']}`"
                )
            if freshness.get("oldest_source_id"):
                lines.append(
                    f"- Oldest source: `{freshness['oldest_source_id']}` at `{freshness['oldest_updated_at']}`"
                )
            lines.append("")
        if summary.get("ranked_sources"):
            lines.extend(["## Ranked Sources", ""])
            for item in summary["ranked_sources"]:
                lines.append(
                    f"- `{item['source_id']}` | score=`{item['score']:.3f}` | "
                    f"confidence=`{item['confidence']}` | contradictions=`{item['contradiction_count']}` | "
                    f"staleness=`{item['staleness_label']}` | updated_at=`{item['updated_at'] or 'unknown'}` | "
                    f"title={item['title']}"
                )
            lines.append("")
        if summary.get("confidence_changes"):
            lines.extend(["## Confidence Changes", ""])
            for item in summary["confidence_changes"]:
                lines.append(
                    f"- `{item['source_id']}`: `{item['previous_confidence']}` -> `{item['latest_confidence']}`"
                )
            lines.append("")
        if summary.get("evidence_kind_changes"):
            lines.extend(["## Evidence Kind Changes", ""])
            for item in summary["evidence_kind_changes"]:
                lines.append(
                    f"- `{item['claim_title']}`: `{item['previous_evidence_kind']}` -> `{item['latest_evidence_kind']}`"
                )
            lines.append("")
        if summary.get("new_contradiction_titles") or summary.get("dropped_contradiction_titles"):
            lines.extend(["## Contradiction Changes", ""])
            lines.extend(
                f"- New contradiction: `{item}`"
                for item in summary.get("new_contradiction_titles", [])
            )
            lines.extend(
                f"- Dropped contradiction: `{item}`"
                for item in summary.get("dropped_contradiction_titles", [])
            )
            lines.append("")
        if summary.get("best_matmul_candidate_id"):
            lines.extend(
                [
                    "## Matmul Frontier",
                    "",
                    f"- Best candidate: `{summary['best_matmul_candidate_id']}`",
                    "",
                ]
            )
        if summary.get("fp8_total_fixture_count", 0):
            lines.extend(["## FP8 Layout Trends", ""])
            lines.append(f"- FP8 fixtures observed: `{summary.get('fp8_total_fixture_count', 0)}`")
            layout_counts = summary.get("fp8_layout_counts", {}) or {}
            if layout_counts:
                lines.append(
                    "- Layout counts: `"
                    + ", ".join(f"{layout}={count}" for layout, count in layout_counts.items())
                    + "`"
                )
            share_by_layout = summary.get("fp8_weighted_share_by_layout", {}) or {}
            if share_by_layout:
                lines.append(
                    "- Weighted share by layout: `"
                    + ", ".join(f"{layout}={share}" for layout, share in share_by_layout.items())
                    + "`"
                )
            reuse_wins = summary.get("fp8_reuse_bucket_layout_wins", {}) or {}
            for bucket, wins in reuse_wins.items():
                if not wins:
                    continue
                lines.append(
                    f"- {bucket}: `"
                    + ", ".join(f"{layout}={count}" for layout, count in wins.items())
                    + "`"
                )
            alignment = summary.get("fp8_policy_alignment", {}) or {}
            if alignment.get("overall_nk_rate") is not None:
                lines.append(f"- overall_nk_rate: `{alignment['overall_nk_rate']}`")
            if alignment.get("reuse_1_kn_rate") is not None:
                lines.append(f"- reuse_1_kn_rate: `{alignment['reuse_1_kn_rate']}`")
            if alignment.get("reuse_2_4_nk_rate") is not None:
                lines.append(f"- reuse_2_4_nk_rate: `{alignment['reuse_2_4_nk_rate']}`")
            if alignment.get("reuse_5_plus_nk_rate") is not None:
                lines.append(f"- reuse_5_plus_nk_rate: `{alignment['reuse_5_plus_nk_rate']}`")
            regressions = summary.get("fp8_policy_regressions", []) or []
            if regressions:
                lines.append("- Policy regression warnings:")
                lines.extend(f"  - {warning}" for warning in regressions)
            lines.append("")
        return "\n".join(lines)

    def _load_matching_runs(
        self,
        *,
        benchmark_id: str | None,
        topic: str | None,
    ) -> list[ResearchRunRecord]:
        if not self.base_dir.exists():
            return []

        records: list[ResearchRunRecord] = []
        for path in self.base_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = deserialize_run_record(payload)
            if benchmark_id and record.benchmark_id != benchmark_id:
                continue
            if topic and record.memo.topic != topic:
                continue
            records.append(record)
        return sorted(records, key=lambda record: record.created_at, reverse=True)


def _best_metric(records: list[ResearchRunRecord], benchmark_id: str | None) -> float | None:
    best = None
    for record in records:
        metric = _extract_metric_from_record(record, benchmark_id)
        if metric is None:
            continue
        best = metric if best is None or metric > best else best
    return best


def _extract_metric_from_record(
    record: ResearchRunRecord,
    benchmark_id: str | None,
) -> float | None:
    if not record.memo.results:
        return None
    payloads = record.memo.results[0].artifact_payloads
    if benchmark_id == "matmul-speedup":
        value = payloads.get("raw-timing-results.json", {}).get("mean_uplift_pct")
    elif benchmark_id == "long-context-reasoning":
        value = payloads.get("candidate-metrics.json", {}).get("accuracy")
    elif benchmark_id == "tool-use-reliability":
        value = payloads.get("tool-selection-summary.json", {}).get("terminal_first_success_rate")
    else:
        value = None
    return float(value) if isinstance(value, (int, float)) else None


def _summarize_source_freshness(sources: list[ResearchSource]) -> dict[str, object]:
    dated = [
        (source.identifier, source.updated_at)
        for source in sources
        if source.updated_at
    ]
    if not dated:
        return {
            "source_count_with_timestamps": 0,
            "newest_source_id": "",
            "newest_updated_at": "",
            "oldest_source_id": "",
            "oldest_updated_at": "",
        }
    newest_source_id, newest_updated_at = max(dated, key=lambda item: item[1] or "")
    oldest_source_id, oldest_updated_at = min(dated, key=lambda item: item[1] or "")
    bucket_counts = {"fresh": 0, "recent": 0, "aging": 0, "stale": 0}
    for _, updated_at in dated:
        bucket = _staleness_label(updated_at=updated_at, newest_updated_at=newest_updated_at)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    return {
        "source_count_with_timestamps": len(dated),
        "newest_source_id": newest_source_id,
        "newest_updated_at": newest_updated_at,
        "oldest_source_id": oldest_source_id,
        "oldest_updated_at": oldest_updated_at,
        "bucket_counts": bucket_counts,
    }


def _rank_sources_for_history(
    *,
    sources: list[ResearchSource],
    assessments: list[SourceAssessment],
    claims: list[Claim],
    contradictions: list[Contradiction],
) -> list[dict[str, object]]:
    assessment_map = {
        assessment.source_id: assessment
        for assessment in assessments
    }
    claim_to_source = {claim.title: claim.source for claim in claims}
    contradiction_counts: dict[str, int] = {}
    for contradiction in contradictions:
        for claim_title in contradiction.claim_titles:
            source_id = claim_to_source.get(claim_title)
            if not source_id:
                continue
            contradiction_counts[source_id] = contradiction_counts.get(source_id, 0) + 1
    scores = []
    newest_ts = max((source.updated_at for source in sources if source.updated_at), default=None)
    for source in sources:
        assessment = assessment_map.get(source.identifier)
        confidence = assessment.confidence if assessment else "medium"
        confidence_weight = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
        }.get(confidence, 0.4)
        freshness_bonus = 0.0
        if source.updated_at and newest_ts:
            freshness_bonus = 0.2 if source.updated_at == newest_ts else 0.1
        contradiction_penalty = min(0.15 * contradiction_counts.get(source.identifier, 0), 0.3)
        staleness_label = _staleness_label(
            updated_at=source.updated_at,
            newest_updated_at=newest_ts,
        )
        scores.append(
            {
                "source_id": source.identifier,
                "title": source.title,
                "confidence": confidence,
                "updated_at": source.updated_at,
                "contradiction_count": contradiction_counts.get(source.identifier, 0),
                "staleness_label": staleness_label,
                "score": round(confidence_weight + freshness_bonus - contradiction_penalty, 3),
            }
        )
    return sorted(scores, key=lambda item: (-item["score"], item["source_id"]))


def _staleness_label(*, updated_at: str | None, newest_updated_at: str | None) -> str:
    if not updated_at or not newest_updated_at:
        return "unknown"
    if updated_at == newest_updated_at:
        return "fresh"
    try:
        updated_dt = _parse_iso_datetime(updated_at)
        newest_dt = _parse_iso_datetime(newest_updated_at)
    except ValueError:
        return "recent"
    delta_days = max((newest_dt - updated_dt).total_seconds() / 86400.0, 0.0)
    if delta_days <= 1:
        return "fresh"
    if delta_days <= 7:
        return "recent"
    if delta_days <= 30:
        return "aging"
    return "stale"


def _parse_iso_datetime(value: str):
    from datetime import datetime

    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _summarize_fp8_policy_alignment(
    *,
    reuse_bucket_totals: dict[str, int],
    reuse_bucket_layout_wins: dict[str, dict[str, int]],
    layout_counts: dict[str, int],
) -> dict[str, float | None]:
    def _rate(numerator: int, denominator: int) -> float | None:
        if denominator <= 0:
            return None
        return round(float(numerator) / float(denominator), 4)

    overall_total = sum(int(v) for v in layout_counts.values())
    overall_nk = int(layout_counts.get("nk", 0))

    reuse_1_total = int(reuse_bucket_totals.get("reuse_1", 0))
    reuse_1_kn = int((reuse_bucket_layout_wins.get("reuse_1", {}) or {}).get("kn", 0))

    reuse_2_4_total = int(reuse_bucket_totals.get("reuse_2_4", 0))
    reuse_2_4_nk = int((reuse_bucket_layout_wins.get("reuse_2_4", {}) or {}).get("nk", 0))

    reuse_5_plus_total = int(reuse_bucket_totals.get("reuse_5_plus", 0))
    reuse_5_plus_nk = int((reuse_bucket_layout_wins.get("reuse_5_plus", {}) or {}).get("nk", 0))

    return {
        "overall_nk_rate": _rate(overall_nk, overall_total),
        "reuse_1_kn_rate": _rate(reuse_1_kn, reuse_1_total),
        "reuse_2_4_nk_rate": _rate(reuse_2_4_nk, reuse_2_4_total),
        "reuse_5_plus_nk_rate": _rate(reuse_5_plus_nk, reuse_5_plus_total),
    }


def _extract_fp8_layout_summary_from_record(record: ResearchRunRecord | None) -> dict | None:
    if record is None:
        return None
    for result in record.memo.results:
        payload = result.artifact_payloads.get("fp8-runtime-layout-summary.json")
        if isinstance(payload, dict):
            return payload
    return None


def _alignment_from_layout_summary(payload: dict | None) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {
            "overall_nk_rate": None,
            "reuse_1_kn_rate": None,
            "reuse_2_4_nk_rate": None,
            "reuse_5_plus_nk_rate": None,
        }

    layout_counts = payload.get("layout_counts", {})
    if not isinstance(layout_counts, dict):
        layout_counts = {}

    bucket_totals = {"reuse_1": 0, "reuse_2_4": 0, "reuse_5_plus": 0}
    bucket_wins = {
        "reuse_1": {},
        "reuse_2_4": {},
        "reuse_5_plus": {},
    }
    fixtures = payload.get("fixtures", [])
    if isinstance(fixtures, list):
        for row in fixtures:
            if not isinstance(row, dict):
                continue
            layout = str(row.get("layout", "")).strip()
            if not layout:
                continue
            try:
                reuse = int(row.get("expected_weight_reuse", 1))
            except (TypeError, ValueError):
                reuse = 1
            if reuse <= 1:
                bucket = "reuse_1"
            elif reuse <= 4:
                bucket = "reuse_2_4"
            else:
                bucket = "reuse_5_plus"
            bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
            wins = bucket_wins.setdefault(bucket, {})
            wins[layout] = wins.get(layout, 0) + 1

    return _summarize_fp8_policy_alignment(
        reuse_bucket_totals=bucket_totals,
        reuse_bucket_layout_wins=bucket_wins,
        layout_counts={
            str(k): int(v)
            for k, v in layout_counts.items()
            if isinstance(k, str) and isinstance(v, int)
        },
    )


def _detect_fp8_policy_regressions(
    *,
    latest_alignment: dict[str, float | None],
    previous_alignment: dict[str, float | None],
    threshold_drop: float = 0.20,
) -> list[str]:
    regressions: list[str] = []

    def _check_drop(metric_key: str, label: str) -> None:
        latest = latest_alignment.get(metric_key)
        previous = previous_alignment.get(metric_key)
        if latest is None or previous is None:
            return
        if float(previous) - float(latest) >= threshold_drop:
            regressions.append(
                f"{label} dropped from {float(previous):.4f} to {float(latest):.4f}"
            )

    _check_drop("reuse_1_kn_rate", "reuse_1_kn_rate")
    _check_drop("reuse_2_4_nk_rate", "reuse_2_4_nk_rate")
    _check_drop("reuse_5_plus_nk_rate", "reuse_5_plus_nk_rate")
    return regressions
