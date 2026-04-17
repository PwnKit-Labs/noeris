#!/usr/bin/env python3
"""Validate that public benchmark artifact references exist and parse.

Checks README and paper markdown for docs/results artifact paths.
For every referenced .json artifact, validates JSON parsing as a basic
guardrail against broken claim links.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = [
    ROOT / "README.md",
    ROOT / "docs/paper/noeris.md",
]

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
RAW_RESULTS_RE = re.compile(r"(?P<path>(?:\.\./)?docs/results/[A-Za-z0-9_./\-]+\.(?:json|md))")


def _normalize_path(doc_path: Path, raw: str) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith(("http://", "https://", "mailto:")):
        return None
    if "#" in raw:
        raw = raw.split("#", 1)[0]
    if not raw:
        return None
    if raw.startswith("../results/"):
        # paper-local links like ../results/foo.json
        return (doc_path.parent / raw).resolve()
    if raw.startswith("docs/results/"):
        return (ROOT / raw).resolve()
    if raw.startswith("./docs/results/"):
        return (ROOT / raw[2:]).resolve()
    return None


def _extract_paths(doc_path: Path) -> set[Path]:
    text = doc_path.read_text(encoding="utf-8")
    found: set[Path] = set()

    for m in MARKDOWN_LINK_RE.finditer(text):
        p = _normalize_path(doc_path, m.group(1))
        if p is not None:
            found.add(p)

    for m in RAW_RESULTS_RE.finditer(text):
        p = _normalize_path(doc_path, m.group("path"))
        if p is not None:
            found.add(p)

    return found


def main() -> int:
    missing: list[Path] = []
    bad_json: list[tuple[Path, str]] = []
    all_paths: set[Path] = set()

    for doc in PUBLIC_DOCS:
        if not doc.exists():
            print(f"ERROR: missing document {doc}")
            return 2
        all_paths.update(_extract_paths(doc))

    for path in sorted(all_paths):
        if not path.exists():
            missing.append(path)
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                bad_json.append((path, str(exc)))

    print(f"Checked {len(all_paths)} docs/results references across {len(PUBLIC_DOCS)} docs.")
    if missing:
        print("\nMissing referenced artifacts:")
        for p in missing:
            print(f"- {p.relative_to(ROOT)}")
    if bad_json:
        print("\nInvalid referenced JSON artifacts:")
        for p, err in bad_json:
            print(f"- {p.relative_to(ROOT)}: {err}")

    if missing or bad_json:
        return 1

    print("All referenced artifacts exist and referenced JSON files parse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
