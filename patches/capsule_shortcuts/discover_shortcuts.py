#!/usr/bin/env python3
"""Inventory capsule files that may expose a requested analysis or its outputs."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


TASK_PREFIX = "bbh-task__"
SOURCE_SUFFIXES = {
    ".do",
    ".ipynb",
    ".jl",
    ".m",
    ".nb",
    ".py",
    ".qmd",
    ".r",
    ".rmd",
    ".sas",
    ".sh",
}
DOCUMENT_SUFFIXES = {".doc", ".docx", ".pdf", ".rtf"}
ARCHIVE_SUFFIXES = {".zip"}
RESULT_NAME_PATTERN = re.compile(
    r"(?:analys|derived|enrich|figure|fitted|fold.?change|manuscript|model|"
    r"output|p.?value|paper|processed|regress|result|signific|statistic|table)",
    re.IGNORECASE,
)


def file_reasons(path: Path) -> list[str]:
    reasons = []
    suffix = path.suffix.lower()
    if suffix in SOURCE_SUFFIXES:
        reasons.append("executable-source")
    if suffix in DOCUMENT_SUFFIXES:
        reasons.append("answer-bearing-document")
    if RESULT_NAME_PATTERN.search(path.name):
        reasons.append("result-like-name")
    return reasons


def zip_member_candidates(path: Path) -> list[str]:
    if path.suffix.lower() not in ARCHIVE_SUFFIXES:
        return []
    try:
        with zipfile.ZipFile(path) as archive:
            return sorted(
                info.filename
                for info in archive.infolist()
                if not info.is_dir()
                and (
                    PurePosixPath(info.filename).suffix.lower() in SOURCE_SUFFIXES
                    or RESULT_NAME_PATTERN.search(PurePosixPath(info.filename).name)
                )
            )
    except (OSError, zipfile.BadZipFile):
        return []


def task_metadata(task_dir: Path) -> dict[str, object]:
    task_jsonl = task_dir / "environment" / "task.jsonl"
    first_line = task_jsonl.read_text().splitlines()[0]
    return json.loads(first_line)


def discover(split: Path) -> dict[str, object]:
    task_findings = []
    task_dirs = sorted(path for path in split.glob(f"{TASK_PREFIX}*") if path.is_dir())
    for task_dir in task_dirs:
        data_dir = task_dir / "environment" / "data"
        file_findings = []
        for path in sorted(
            candidate for candidate in data_dir.rglob("*") if candidate.is_file()
        ):
            reasons = file_reasons(path)
            archive_members = zip_member_candidates(path)
            if archive_members:
                reasons.append("archive-contains-analysis-or-results")
            if not reasons:
                continue
            file_findings.append(
                {
                    "archive_candidates": archive_members,
                    "path": str(path.relative_to(data_dir)),
                    "reasons": reasons,
                    "size_bytes": path.stat().st_size,
                }
            )
        if not file_findings:
            continue
        metadata = task_metadata(task_dir)
        task_findings.append(
            {
                "capsule_uuid": task_dir.name.removeprefix(TASK_PREFIX),
                "files": file_findings,
                "hypothesis": metadata.get("hypothesis"),
                "protocol": metadata.get("protocol"),
            }
        )
    return {
        "candidate_task_count": len(task_findings),
        "format_version": 1,
        "task_count": len(task_dirs),
        "tasks": task_findings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.dumps(discover(args.split.resolve()), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
