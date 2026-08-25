#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write a NeMo-RL JSONL manifest from a materialized Harbor split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--split",
        type=Path,
        help="Harbor split directory containing one directory per task.",
    )
    source.add_argument(
        "--input-manifest",
        type=Path,
        help="Existing NeMo-RL JSONL manifest to filter instead of scanning a split.",
    )
    parser.add_argument(
        "--reference-split",
        type=Path,
        help=(
            "Materialized Harbor split used to validate tasks and exclusions when "
            "--input-manifest is supplied."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--agent-name",
        default="gym_harbor_agent",
        help="Registered NeMo Gym responses API agent name.",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "File containing task UUIDs or exact task names to exclude, one per line. "
            "Blank lines and # comments are ignored; may be repeated."
        ),
    )
    parser.add_argument(
        "--require-all-exclusions",
        action="store_true",
        help="Fail if an exclusion is already absent from the reference task set.",
    )
    return parser.parse_args()


def task_names(split: Path) -> list[str]:
    if not split.is_dir():
        raise ValueError(f"Harbor split is not a directory: {split}")

    names = sorted(
        path.name
        for path in split.iterdir()
        if path.is_dir() and (path / "task.toml").is_file()
    )
    if not names:
        raise ValueError(f"Harbor split contains no task directories: {split}")
    return names


def records_from_split(split: Path, agent_name: str) -> list[dict[str, Any]]:
    return [
        {
            "task_name": name,
            "agent_ref": {
                "type": "responses_api_agents",
                "name": agent_name,
            },
            "responses_create_params": {"input": []},
        }
        for name in task_names(split)
    ]


def records_from_manifest(manifest: Path) -> list[dict[str, Any]]:
    if not manifest.is_file():
        raise ValueError(f"Input manifest is not a file: {manifest}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {manifest} at line {line_number}: {error}"
            ) from error
        task_name = record.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError(
                f"Missing non-empty task_name in {manifest} at line {line_number}"
            )
        records.append(record)

    if not records:
        raise ValueError(f"Input manifest contains no records: {manifest}")
    return records


def read_exclusions(paths: list[Path]) -> set[str]:
    exclusions: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Exclusion file is not a file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.split("#", maxsplit=1)[0].strip()
            if value:
                exclusions.add(value)
    return exclusions


def resolve_excluded_task_names(
    known_task_names: list[str], exclusions: set[str], *, require_all: bool = False
) -> tuple[set[str], list[str]]:
    resolved: set[str] = set()
    unmatched = []
    for exclusion in exclusions:
        matches = {
            task_name
            for task_name in known_task_names
            if exclusion in {task_name, task_name.rsplit("__", maxsplit=1)[-1]}
        }
        if len(matches) > 1:
            raise ValueError(
                "Exclusion entry must exactly match one task name or __-delimited "
                f"task UUID; {exclusion!r} matched {sorted(matches)}"
            )
        if not matches:
            unmatched.append(exclusion)
        resolved.update(matches)
    unmatched.sort()
    if require_all and unmatched:
        raise ValueError(
            f"Exclusion entries absent from the reference tasks: {unmatched}"
        )
    return resolved, unmatched


def filter_records(
    records: list[dict[str, Any]], excluded_task_names: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    record_task_names = [record["task_name"] for record in records]
    if len(record_task_names) != len(set(record_task_names)):
        duplicates = sorted(
            name for name in set(record_task_names) if record_task_names.count(name) > 1
        )
        raise ValueError(f"Duplicate task_name values in manifest: {duplicates}")

    filtered_records = []
    removed_task_names = []
    for record in records:
        task_name = record["task_name"]
        if task_name in excluded_task_names:
            removed_task_names.append(task_name)
        else:
            filtered_records.append(record)

    if not filtered_records:
        raise ValueError("Exclusions removed every task from the manifest")
    return filtered_records, removed_task_names


def main() -> None:
    args = parse_args()
    if args.split is not None:
        records = records_from_split(args.split, args.agent_name)
        known_task_names = [record["task_name"] for record in records]
    else:
        records = records_from_manifest(args.input_manifest)
        if args.reference_split is None:
            raise ValueError(
                "--reference-split is required with --input-manifest so already "
                "filtered manifests can be validated idempotently"
            )
        known_task_names = task_names(args.reference_split)
        unknown_manifest_tasks = sorted(
            {record["task_name"] for record in records} - set(known_task_names)
        )
        if unknown_manifest_tasks:
            raise ValueError(
                "Input manifest contains tasks absent from the reference split: "
                f"{unknown_manifest_tasks}"
            )

    exclusions = read_exclusions(args.exclude_file)
    excluded_task_names, unmatched_exclusions = resolve_excluded_task_names(
        known_task_names,
        exclusions,
        require_all=args.require_all_exclusions,
    )
    records, removed_task_names = filter_records(records, excluded_task_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(
        f"Wrote {len(records)} tasks to {args.output}; "
        f"mask resolves to {len(excluded_task_names)} tasks and removed "
        f"{len(removed_task_names)} rows: {removed_task_names}; "
        f"already absent from reference: {unmatched_exclusions}"
    )


if __name__ == "__main__":
    main()
