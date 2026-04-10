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
        latest_assessments = {
            assessment.source_id: assessment.confidence
            for assessment in (latest.memo.source_assessments if latest else [])
        }
        previous_assessments = {
            assessment.source_id: assessment.confidence
            for assessment in (previous.memo.source_assessments if previous else [])
        }

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

        matmul_candidate_wins: dict[str, int] = {}
        matmul_shape_winners: dict[str, dict[str, object]] = {}
        matmul_shape_challengers: dict[str, dict[str, object]] = {}
        if benchmark_id == "matmul-speedup":
            for record in records:
                for result in record.memo.results:
                    payload = result.artifact_payloads.get("best-candidate-summary.json")
                    if not isinstance(payload, dict):
                        continue
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
                    if isinstance(raw_rows, list):
                        for row in raw_rows:
                            if not isinstance(row, dict):
                                continue
                            shape = str(row.get("shape", "")).strip()
                            winner = str(row.get("best_candidate_id", "")).strip()
                            uplift = row.get("uplift_pct")
                            if not shape or not winner:
                                continue
                            entry = matmul_shape_winners.setdefault(
                                shape,
                                {"winner_counts": {}, "latest_winner": winner, "latest_uplift_pct": uplift},
                            )
                            entry["winner_counts"][winner] = entry["winner_counts"].get(winner, 0) + 1
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

        return {
            "benchmark_id": benchmark_id or (latest.benchmark_id if latest else ""),
            "topic": topic or (latest.memo.topic if latest else ""),
            "run_count": len(records),
            "latest_run_id": latest.run_id if latest else "",
            "previous_run_id": previous.run_id if previous else "",
            "new_claim_titles": sorted(latest_claim_titles - previous_claim_titles),
            "dropped_claim_titles": sorted(previous_claim_titles - latest_claim_titles),
            "shared_claim_titles": sorted(latest_claim_titles & previous_claim_titles),
            "confidence_changes": confidence_changes,
            "latest_contradictions": [
                {
                    "title": contradiction.title,
                    "severity": contradiction.severity,
                    "summary": contradiction.summary,
                }
                for contradiction in (latest.memo.contradictions if latest else [])
            ],
            "matmul_candidate_wins": matmul_candidate_wins,
            "best_matmul_candidate_id": (
                max(matmul_candidate_wins, key=matmul_candidate_wins.get)
                if matmul_candidate_wins
                else ""
            ),
            "best_benchmark_metric": _best_metric(records, benchmark_id),
            "matmul_shape_winners": matmul_shape_winners,
            "matmul_shape_challengers": matmul_shape_challengers,
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
        }

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
