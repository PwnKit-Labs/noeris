from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import ResearchRunRecord
from .store import serialize_run_record


def export_run_bundle(record: ResearchRunRecord, output_dir: str | Path) -> Path:
    base_dir = Path(output_dir) / record.run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    (base_dir / "run.json").write_text(
        json.dumps(serialize_run_record(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (base_dir / "memo.json").write_text(
        json.dumps(asdict(record.memo), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (base_dir / "verification.json").write_text(
        json.dumps(asdict(record.verification), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (base_dir / "summary.md").write_text(_render_summary(record), encoding="utf-8")
    (base_dir / "claim-lineage.json").write_text(
        json.dumps(_build_claim_lineage(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_result_artifacts(record, base_dir)
    return base_dir


def _render_summary(record: ResearchRunRecord) -> str:
    benchmark_line = (
        f"- Benchmark: `{record.benchmark_id}`\n" if record.benchmark_id else ""
    )
    cost_lines = []
    for result in record.cycle.results:
        payload = result.artifact_payloads.get("cost-summary.json")
        if not isinstance(payload, dict):
            continue
        cost_lines.append(f"- Requests: `{payload.get('request_count', 0)}`")
        cost_lines.append(f"- Tokens: `{payload.get('total_tokens', 0)}`")
        cost_lines.append(f"- Elapsed ms: `{payload.get('elapsed_ms', 0)}`")
        if payload.get("estimated_cost_usd") is not None:
            cost_lines.append(f"- Estimated cost USD: `{payload['estimated_cost_usd']}`")
        warnings = payload.get("warnings") or []
        if warnings:
            cost_lines.append("- Warnings:")
            cost_lines.extend(f"  - {warning}" for warning in warnings)
    sections = [
        f"# Noeris Run {record.run_id}",
        "",
        f"- Created: `{record.created_at}`",
        benchmark_line.rstrip() if benchmark_line else None,
        f"- Topic: `{record.cycle.topic.name}`",
        f"- Verification passed: `{record.verification.passed}`",
        "",
    ]
    if cost_lines:
        sections.extend(["## Cost Summary", "", *cost_lines, ""])
    if record.memo.source_assessments:
        sections.extend(["## Source Confidence", ""])
        for assessment in record.memo.source_assessments:
            sections.append(
                f"- `{assessment.source_id}` | confidence=`{assessment.confidence}` | {assessment.rationale}"
            )
        sections.append("")
    if record.memo.contradictions:
        sections.extend(["## Contradictions", ""])
        for contradiction in record.memo.contradictions:
            sections.append(
                f"- **{contradiction.title}** (`{contradiction.severity}`): {contradiction.summary}"
            )
        sections.append("")
    sections.extend(
        [
            "## Checks",
            "",
            *[f"- `{check}`" for check in record.verification.checks],
            "",
            "## Blockers",
            "",
            *(
                [f"- `{blocker}`" for blocker in record.verification.blockers]
                if record.verification.blockers
                else ["- none"]
            ),
            "",
            "## Next Actions",
            "",
            *[f"- {item}" for item in record.memo.next_actions],
            "",
        ]
    )
    return "\n".join(line for line in sections if line is not None)


def _write_result_artifacts(record: ResearchRunRecord, base_dir: Path) -> None:
    for result in record.cycle.results:
        for artifact_name, payload in result.artifact_payloads.items():
            path = base_dir / artifact_name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, str) and artifact_name.endswith(".md"):
                path.write_text(payload, encoding="utf-8")
            elif artifact_name.endswith(".jsonl") and isinstance(payload, list):
                path.write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in payload),
                    encoding="utf-8",
                )
            else:
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


def _build_claim_lineage(record: ResearchRunRecord) -> dict[str, object]:
    source_map = {
        source.identifier: {
            "kind": source.kind,
            "title": source.title,
            "locator": source.locator,
        }
        for source in record.memo.sources
    }
    source_assessment_map = {
        assessment.source_id: {
            "confidence": assessment.confidence,
            "rationale": assessment.rationale,
        }
        for assessment in record.memo.source_assessments
    }
    contradiction_map: dict[str, list[dict[str, object]]] = {}
    for contradiction in record.memo.contradictions:
        for claim_title in contradiction.claim_titles:
            contradiction_map.setdefault(claim_title, []).append(
                {
                    "title": contradiction.title,
                    "summary": contradiction.summary,
                    "severity": contradiction.severity,
                }
            )
    claim_entries = []
    for claim in record.memo.claims:
        linked_sources = []
        for ref in claim.evidence_refs:
            if ref not in source_map:
                continue
            linked_sources.append(
                {
                    **source_map[ref],
                    "source_id": ref,
                    "assessment": source_assessment_map.get(ref),
                }
            )
        supported_by = [
            hypothesis.title
            for hypothesis in record.memo.hypotheses
            if claim.title in hypothesis.supporting_claims
        ]
        claim_entries.append(
            {
                "claim_title": claim.title,
                "claim_summary": claim.summary,
                "source": claim.source,
                "evidence_kind": claim.evidence_kind,
                "evidence_refs": claim.evidence_refs,
                "linked_sources": linked_sources,
                "contradictions": contradiction_map.get(claim.title, []),
                "supporting_hypotheses": supported_by,
            }
        )
    return {
        "topic": record.memo.topic,
        "benchmark_id": record.benchmark_id,
        "source_count": len(record.memo.sources),
        "claim_count": len(record.memo.claims),
        "claims": claim_entries,
    }
