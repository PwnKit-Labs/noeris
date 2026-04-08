from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import (
    Claim,
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
        contradictions=context_payload.get("contradictions", []),
    )
    cycle = ResearchCycle(
        topic=topic,
        context=context,
        hypotheses=[Hypothesis(**item) for item in cycle_payload["hypotheses"]],
        experiments=[ExperimentSpec(**item) for item in cycle_payload["experiments"]],
        results=[_deserialize_result(item) for item in cycle_payload["results"]],
    )
    verification = VerificationReport(**payload["verification"])
    memo = ResearchMemo(
        topic=memo_payload["topic"],
        summary=memo_payload["summary"],
        sources=[_deserialize_source(item) for item in memo_payload["sources"]],
        claims=[_deserialize_claim(item) for item in memo_payload["claims"]],
        hypotheses=[Hypothesis(**item) for item in memo_payload["hypotheses"]],
        experiments=[ExperimentSpec(**item) for item in memo_payload["experiments"]],
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


def _deserialize_result(payload: dict) -> ExperimentResult:
    return ExperimentResult(
        spec_name=payload["spec_name"],
        status=ExperimentStatus(payload["status"]),
        outcome_summary=payload["outcome_summary"],
        artifact_refs=payload.get("artifact_refs", []),
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
