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

"""Build an uploadable OpenCode Harbor dataset from Hypotest JSONL and capsules."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patches.capsule_shortcuts.prepare_clean_split import (
    capsule_data_dir,
    collect_patch_findings,
    read_mask,
)
from patches.harbor_opencode.convert_bbh_to_opencode import (
    OUTPUT_PREFIX,
    TEMPLATE_DIR,
    calibrate_rubric,
    render_instruction,
    render_judge_instruction,
    task_toml,
    validate_task,
)


DEFAULT_OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "default_agent": "analysis",
    "subagent_depth": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--capsules-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument(
        "--capsule-inventory-jsonl",
        type=Path,
        help="Optional expected path/size/SHA-256 inventory for the capsule root.",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        type=Path,
        help="Capsule UUIDs to omit; may be repeated.",
    )
    parser.add_argument(
        "--capsule-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="How to materialize the deduplicated capsules tree (default: hardlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_sha256(path: Path) -> str:
    return file_sha256(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def read_exclusions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        excluded.update(read_mask(path.resolve()))
    return excluded


def validate_task_records(
    records: list[dict[str, Any]], capsules_dir: Path, excluded: set[str]
) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    required = {
        "answer",
        "hypothesis",
        "id",
        "input_data_path",
        "max_points",
        "nb_primary_language",
        "orig_rubric",
        "protocol",
        "rubric",
    }
    kept: list[tuple[int, dict[str, Any]]] = []
    dropped: list[str] = []
    content_signatures: set[tuple[str, str, str, str]] = set()
    for index, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"task row {index} is missing fields: {missing}")
        capsule_uuid = record["id"]
        if not isinstance(capsule_uuid, str) or not capsule_uuid:
            raise ValueError(f"task row {index} has an invalid id")
        expected_input_path = f"capsule_{capsule_uuid}"
        if record["input_data_path"] != expected_input_path:
            raise ValueError(
                f"task row {index} input_data_path is {record['input_data_path']!r}, "
                f"expected {expected_input_path!r}"
            )
        if capsule_uuid in excluded:
            dropped.append(capsule_uuid)
            continue
        if not (capsules_dir / expected_input_path).is_dir():
            raise FileNotFoundError(
                f"task row {index} references missing capsule {expected_input_path}"
            )
        signature = (
            capsule_uuid,
            str(record["hypothesis"]),
            str(record["protocol"]),
            str(record["rubric"]),
        )
        if signature in content_signatures:
            raise ValueError(
                f"duplicate task content at row {index} for {capsule_uuid}"
            )
        content_signatures.add(signature)
        kept.append((index, record))
    return kept, sorted(set(dropped))


def normalized_task(record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    task = copy.deepcopy(record)
    try:
        calibrate_rubric(task)
        return task, None
    except ValueError as paraphrased_error:
        task = copy.deepcopy(record)
        task["rubric"] = task["orig_rubric"]
        try:
            calibrate_rubric(task)
        except ValueError as original_error:
            raise ValueError(
                f"neither paraphrased nor original rubric is valid for {task['id']}: "
                f"paraphrased={paraphrased_error}; original={original_error}"
            ) from original_error
        return task, str(paraphrased_error)


def expected_inventory(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    records = read_jsonl(path.resolve())
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"invalid inventory path in {path}: {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate inventory path in {path}: {relative}")
        expected[relative] = record
    return expected


def inventory_capsules(
    capsules_dir: Path,
    capsule_uuids: set[str],
    expected: dict[str, dict[str, Any]] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    inventories: dict[str, list[dict[str, Any]]] = {}
    observed_paths: set[str] = set()
    total_bytes = 0
    total_files = 0
    for ordinal, capsule_uuid in enumerate(sorted(capsule_uuids), start=1):
        capsule = capsules_dir / f"capsule_{capsule_uuid}"
        entries = []
        for path in sorted(item for item in capsule.rglob("*") if item.is_file()):
            if path.is_symlink():
                raise ValueError(f"capsule input symlinks are unsupported: {path}")
            relative_to_capsules = path.relative_to(capsules_dir).as_posix()
            relative_to_capsule = path.relative_to(capsule).as_posix()
            stat = path.stat()
            digest = file_sha256(path)
            observed_paths.add(relative_to_capsules)
            total_bytes += stat.st_size
            total_files += 1
            if expected is not None:
                expected_entry = expected.get(relative_to_capsules)
                if expected_entry is None:
                    raise ValueError(
                        f"capsule file absent from expected inventory: {relative_to_capsules}"
                    )
                if expected_entry.get("size_bytes") != stat.st_size:
                    raise ValueError(f"capsule size mismatch: {relative_to_capsules}")
                if expected_entry.get("sha256") != digest:
                    raise ValueError(
                        f"capsule SHA-256 mismatch: {relative_to_capsules}"
                    )
            entries.append(
                {
                    "path": (Path("data") / relative_to_capsule).as_posix(),
                    "size": stat.st_size,
                    "sha256": digest,
                }
            )
        inventories[capsule_uuid] = entries
        print(
            f"[{ordinal}/{len(capsule_uuids)}] inventoried capsule_{capsule_uuid} "
            f"({len(entries)} files)",
            flush=True,
        )
    if expected is not None:
        relevant_expected = {
            path
            for path in expected
            if path.partition("/")[0].removeprefix("capsule_") in capsule_uuids
        }
        missing = sorted(relevant_expected - observed_paths)
        if missing:
            raise ValueError(
                f"expected capsule inventory paths are missing: {missing[:20]}"
            )
    return inventories, {
        "capsules": len(inventories),
        "files": total_files,
        "bytes": total_bytes,
        "expected_inventory_verified": expected is not None,
    }


def copy_tree(source: Path, destination: Path, mode: str) -> None:
    copy_function = os.link if mode == "hardlink" else shutil.copy2
    shutil.copytree(source, destination, copy_function=copy_function)


def install_static_templates(root: Path) -> dict[str, Path]:
    template_root = root / "_templates"
    template_root.mkdir(parents=True)
    sources = {
        "analysis.md": TEMPLATE_DIR / "analysis_agent.md",
        "score_solution.py": TEMPLATE_DIR / "score_solution.py",
        "test.sh": TEMPLATE_DIR / "test.sh",
        "setup.sh": TEMPLATE_DIR / "setup.sh",
    }
    installed: dict[str, Path] = {}
    for name, source in sources.items():
        destination = template_root / name
        shutil.copy2(source, destination)
        installed[name] = destination
    opencode_path = template_root / "opencode.json"
    opencode_path.write_text(
        json.dumps(DEFAULT_OPENCODE_CONFIG, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    installed["opencode.json"] = opencode_path
    return installed


def link_static(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def write_harbor_task(
    *,
    destination: Path,
    task: dict[str, Any],
    task_index: int,
    input_inventory: list[dict[str, Any]],
    docker_image: str,
    templates: dict[str, Path],
    rubric_fallback_reason: str | None,
) -> str:
    capsule_uuid = task["id"]
    task_suffix = f"{task_index:08d}__{capsule_uuid}"
    task_name = f"{OUTPUT_PREFIX}{task_suffix}"
    tests_dir = destination / "steps" / "rollout" / "tests"
    workdir_dir = destination / "steps" / "rollout" / "workdir"
    agent_dir = destination / "environment" / ".opencode" / "agents"
    tests_dir.mkdir(parents=True)
    workdir_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)

    link_static(templates["analysis.md"], agent_dir / "analysis.md")
    link_static(templates["score_solution.py"], tests_dir / "score_solution.py")
    link_static(templates["test.sh"], tests_dir / "test.sh")
    link_static(templates["setup.sh"], workdir_dir / "setup.sh")
    link_static(
        templates["opencode.json"], destination / "environment" / "opencode.json"
    )

    hidden = copy.deepcopy(task)
    hidden["source_task_name"] = f"hypotest/{task_index}"
    hidden["input_file_manifest"] = input_inventory
    hidden["harbor_conversion"] = {
        "source_row_index": task_index,
        "rubric_source": "orig_rubric" if rubric_fallback_reason else "rubric",
        "rubric_fallback_reason": rubric_fallback_reason,
    }
    (destination / "steps" / "rollout" / "instruction.md").write_text(
        render_instruction(hidden), encoding="utf-8"
    )
    (tests_dir / "judge_instruction.md").write_text(
        render_judge_instruction(hidden), encoding="utf-8"
    )
    (tests_dir / "task.json").write_text(
        json.dumps(hidden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    source_config = {
        "environment": {"docker_image": docker_image, "workdir": "/app"},
        "agent": {"timeout_sec": 3600},
    }
    (destination / "task.toml").write_text(
        task_toml(source_config, task_id=task_suffix), encoding="utf-8"
    )
    validate_task(destination)
    return task_name


def write_manifest(path: Path, task_names: list[str]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        for task_name in task_names:
            record = {
                "task_name": task_name,
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "gym_harbor_agent",
                },
                "responses_create_params": {"input": []},
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    tasks_jsonl = args.tasks_jsonl.resolve()
    capsules_dir = args.capsules_dir.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {output_root}")
        shutil.rmtree(output_root)
    if not capsules_dir.is_dir():
        raise ValueError(f"capsules directory does not exist: {capsules_dir}")

    records = read_jsonl(tasks_jsonl)
    source_row_count = len(records)
    excluded = read_exclusions(args.exclude_file)
    indexed_records, dropped_capsules = validate_task_records(
        records, capsules_dir, excluded
    )
    capsule_uuids = {record["id"] for _, record in indexed_records}
    unexpected_capsules = sorted(
        path.name.removeprefix("capsule_")
        for path in capsules_dir.glob("capsule_*")
        if path.is_dir() and path.name.removeprefix("capsule_") not in capsule_uuids
    )
    if unexpected_capsules and not set(unexpected_capsules).issubset(excluded):
        raise ValueError(
            "capsule root contains unreferenced, non-excluded capsules: "
            f"{unexpected_capsules[:20]}"
        )

    def selected_capsule_data_dir(root: Path, task_id: str) -> Path:
        if task_id not in capsule_uuids:
            return root / ".excluded-capsule"
        return capsule_data_dir(root, task_id)

    shortcut_findings = collect_patch_findings(
        capsules_dir, data_dir_resolver=selected_capsule_data_dir
    )
    if shortcut_findings:
        raise ValueError(
            "known shortcut audit failed:\n" + "\n".join(shortcut_findings)
        )

    expected = expected_inventory(
        args.capsule_inventory_jsonl.resolve()
        if args.capsule_inventory_jsonl is not None
        else None
    )
    inventories, inventory_summary = inventory_capsules(
        capsules_dir, capsule_uuids, expected
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        templates = install_static_templates(temporary_root)
        selected_capsules_root = temporary_root / "capsules"
        selected_capsules_root.mkdir()
        for capsule_uuid in sorted(capsule_uuids):
            copy_tree(
                capsules_dir / f"capsule_{capsule_uuid}",
                selected_capsules_root / f"capsule_{capsule_uuid}",
                args.capsule_mode,
            )

        split = temporary_root / "datasets" / "train"
        split.mkdir(parents=True)
        task_names = []
        rubric_fallbacks = []
        for output_ordinal, (task_index, record) in enumerate(indexed_records, start=1):
            task, fallback_reason = normalized_task(record)
            task_name = write_harbor_task(
                destination=split / f"{OUTPUT_PREFIX}{task_index:08d}__{task['id']}",
                task=task,
                task_index=task_index,
                input_inventory=inventories[task["id"]],
                docker_image=args.docker_image,
                templates=templates,
                rubric_fallback_reason=fallback_reason,
            )
            task_names.append(task_name)
            if fallback_reason is not None:
                rubric_fallbacks.append(
                    {
                        "task_index": task_index,
                        "capsule_uuid": task["id"],
                        "reason": fallback_reason,
                    }
                )
            if output_ordinal % 250 == 0 or output_ordinal == len(indexed_records):
                print(
                    f"[{output_ordinal}/{len(indexed_records)}] materialized Harbor tasks",
                    flush=True,
                )

        if len(task_names) != len(set(task_names)):
            raise ValueError("generated duplicate Harbor task names")
        manifest_path = temporary_root / "manifests" / "train.jsonl"
        write_manifest(manifest_path, task_names)
        fallback_capsules = Counter(item["capsule_uuid"] for item in rubric_fallbacks)
        report = {
            "format_version": 1,
            "status": "clean",
            "source": {
                "tasks_file": tasks_jsonl.name,
                "tasks_sha256": jsonl_sha256(tasks_jsonl),
                "rows_before_exclusions": source_row_count,
            },
            "output": {
                "tasks": len(task_names),
                "capsules": len(capsule_uuids),
                "manifest": "manifests/train.jsonl",
                "manifest_sha256": file_sha256(manifest_path),
                "split": "datasets/train",
                "capsules_root": "capsules",
                "task_name_format": "bbh-task__{zero_padded_row_index}__{capsule_uuid}",
                "runtime_task_id": "capsule UUID (final __-delimited task-name field)",
            },
            "capsule_inventory": inventory_summary,
            "shortcut_audit": {
                "known_findings": 0,
                "excluded_capsules": dropped_capsules,
                "unexpected_capsules": unexpected_capsules,
            },
            "rubrics": {
                "fallback_rows": len(rubric_fallbacks),
                "fallback_capsules": dict(sorted(fallback_capsules.items())),
                "fallbacks": rubric_fallbacks,
            },
        }
        audit_dir = temporary_root / "audit"
        audit_dir.mkdir()
        if args.capsule_inventory_jsonl is not None:
            inventory_path = args.capsule_inventory_jsonl.resolve()
            shutil.copy2(inventory_path, audit_dir / "capsule_inventory.jsonl")
            report["capsule_inventory"].update(
                {
                    "file": "audit/capsule_inventory.jsonl",
                    "sha256": file_sha256(inventory_path),
                }
            )
        (audit_dir / "conversion_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_root, output_root)
        return report
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    report = build_bundle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
