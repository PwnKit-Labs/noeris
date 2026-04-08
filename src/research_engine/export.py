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
    return base_dir


def _render_summary(record: ResearchRunRecord) -> str:
    benchmark_line = (
        f"- Benchmark: `{record.benchmark_id}`\n" if record.benchmark_id else ""
    )
    return (
        f"# Noeris Run {record.run_id}\n\n"
        f"- Created: `{record.created_at}`\n"
        f"{benchmark_line}"
        f"- Topic: `{record.cycle.topic.name}`\n"
        f"- Verification passed: `{record.verification.passed}`\n\n"
        "## Checks\n\n"
        + "\n".join(f"- `{check}`" for check in record.verification.checks)
        + "\n\n## Blockers\n\n"
        + (
            "\n".join(f"- `{blocker}`" for blocker in record.verification.blockers)
            if record.verification.blockers
            else "- none"
        )
        + "\n\n## Next Actions\n\n"
        + "\n".join(f"- {item}" for item in record.memo.next_actions)
        + "\n"
    )
