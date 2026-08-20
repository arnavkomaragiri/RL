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

"""Convert BBH MCP Harbor tasks into artifact-based OpenCode tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SOURCE_PREFIX = "bbh-task__"
OUTPUT_PREFIX = SOURCE_PREFIX
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

RUBRIC_ADDITIONS = {
    "0923d260-fe1b-4fb4-4398-79edf546e584": (
        "* 1 point: Requested visualizations (for example pathway dot plots, "
        "heatmaps, or volcano plots) are generated and support the comparison."
    ),
    "1d54e4a7-8b0f-4224-bd31-efcfded0d46c": (
        "* 1 point: Overlap among the three mutant DE gene sets is visualized with "
        "a Venn diagram, UpSet plot, or equivalent summary."
    ),
    "975f3e91-53b0-44b1-ac9f-20023d9c8cd0": (
        "* 1 point: Fungal and animal tree-length distributions are visualized and "
        "annotated with the key summary statistics."
    ),
    "d59734d2-a3e0-462a-a5fd-c8ddc11392b8": (
        "* 1 point: Circularity trajectories over time are plotted for wild-type and "
        "the QS mutants."
    ),
}

RUBRIC_TEXT_CORRECTIONS = {
    "03db3687-1fcf-4c5c-8a2e-f779d7eb3cf0": {
        "1 ponit: builds the weighted graph network object with residues as nodes "
        "and contact frequencies as edge weights.": (
            "1 point: Builds the weighted graph network object with residues as nodes "
            "and contact frequencies as edge weights."
        ),
    },
}

# Source metadata corrections for rubrics whose declared maximum does not match
# the itemized allocations after known text corrections and additions.
RUBRIC_MAX_POINT_CORRECTIONS = {
    "03db3687-1fcf-4c5c-8a2e-f779d7eb3cf0": 12,
    "75adb693-e3d5-4f23-bd1d-d2341e209d66": 17,
}

RUBRIC_POINT_PATTERNS = (
    re.compile(
        r"^\s*(?:[*\-\u2022]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
        r"(\d+)\s*(?:points?|pts?)(?:\*\*)?\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:[*\-\u2022]\s*)?(?:\d+[.)]\s*)?.*?"
        r"(?:\(\s*|[\-\u2013\u2014]\s*)(\d+)\s*(?:points?|pts?)\s*\)?\s*(?::|$)",
        re.IGNORECASE,
    ),
)


def rubric_item_points(rubric: Any) -> list[int]:
    if not isinstance(rubric, str) or not rubric.strip():
        raise ValueError("task has no rubric")
    point_groups: list[list[int]] = [[] for _ in RUBRIC_POINT_PATTERNS]
    for line in rubric.splitlines():
        for index, pattern in enumerate(RUBRIC_POINT_PATTERNS):
            if match := pattern.match(line):
                point_groups[index].append(int(match.group(1)))
                break
    populated_groups = [points for points in point_groups if points]
    if not populated_groups:
        raise ValueError("rubric has no itemized point values")
    totals = {sum(points) for points in populated_groups}
    if len(totals) != 1:
        raise ValueError(
            "rubric mixes incompatible prefix and heading point allocations: "
            f"{sorted(totals)}"
        )
    # Some source rubrics repeat each allocation in a section heading and its
    # nested bullet items. Prefer the finer prefix allocations when present;
    # equal group totals describe the same score once.
    return point_groups[0] or populated_groups[0]


def rubric_point_total(rubric: Any) -> int:
    return sum(rubric_item_points(rubric))


def calibrate_rubric(task: dict[str, Any]) -> None:
    task_id = str(task.get("id", ""))
    rubric = str(task.get("rubric", ""))
    for original, replacement in RUBRIC_TEXT_CORRECTIONS.get(task_id, {}).items():
        rubric = rubric.replace(original, replacement)
    task["rubric"] = rubric
    addition = RUBRIC_ADDITIONS.get(task_id)
    if addition:
        rubric = rubric.rstrip()
        if addition not in rubric:
            task["rubric"] = f"{rubric}\n{addition}"
    if task_id in RUBRIC_MAX_POINT_CORRECTIONS:
        task["max_points"] = RUBRIC_MAX_POINT_CORRECTIONS[task_id]
    maximum = task.get("max_points")
    item_points = rubric_item_points(task.get("rubric"))
    task["rubric_item_points"] = item_points
    total = sum(item_points)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum != total:
        raise ValueError(
            f"rubric points do not match max_points for {task_id}: "
            f"rubric={total}, max_points={maximum!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path)
    parser.add_argument("--output-split", type=Path)
    parser.add_argument(
        "--refresh-split",
        type=Path,
        help="Refresh policy instructions and hidden verifier files in place.",
    )
    parser.add_argument(
        "--data-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="How to materialize environment/data (default: hardlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output task directories that already exist.",
    )
    return parser.parse_args()


def read_jsonl_record(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"expected exactly one JSON object in {path}")
    return records[0]


def copy_data(source: Path, destination: Path, mode: str) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing task data directory: {source}")
    copy_function = os.link if mode == "hardlink" else shutil.copy2
    shutil.copytree(source, destination, copy_function=copy_function)


def inventory_input_files(data_root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in data_root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError(f"input data symlinks are not supported: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append(
            {
                "path": (Path("data") / path.relative_to(data_root)).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return inventory


def render_instruction(task: dict[str, Any]) -> str:
    hypothesis = str(task.get("hypothesis", "")).strip()
    protocol = str(task.get("protocol", "")).strip()
    language = str(task.get("nb_primary_language", "unspecified")).strip()
    return f"""Analyze the provided dataset and rigorously evaluate the hypothesis below.

## Hypothesis

{hypothesis}

## Analysis protocol

{protocol}

## Deliverables

Work directly in `/app`. The input dataset is available under `/app/data`; treat it as read-only source data. The source task's primary analysis language is `{language}`, but you may use any installed language or command-line tool appropriate to the data.

Create a reproducible analysis workspace with:

1. `REPORT.md`, containing the objective, data inventory, methods and analytical decisions, quantitative results, conclusion about the hypothesis, limitations, and exact reproduction commands.
2. Executable reproduction material. Reusable source files and static workflow configuration are recommended for nontrivial analyses, but self-contained commands or code in `REPORT.md` are also accepted.
3. Any additional tables, machine-readable results, logs, or figures needed by the requested protocol or rubric. Findings may be reported directly in `REPORT.md` when the reproduction commands recompute them; a separate generated artifact is not otherwise required.

Supplementary documentation such as `README.md` is allowed. No machine-readable submission manifest is required: the verifier discovers original capsule inputs, analysis source, documentation, and generated outputs from the workspace. Reproduction commands must run from `/app`, be stated exactly in `REPORT.md`, and recompute every reported finding from the original inputs and submitted reproduction material.

Run the submitted analysis yourself before finishing. Inspect binary inputs with an appropriate executable program, such as `Rscript` plus `readRDS()` for RDS files; do not infer their contents from filenames or treat a text-reader failure as evidence that they are unusable. If the inputs cannot support the full requested protocol, run a diagnostic program or self-contained command that establishes what they contain and report the resulting evidence in `REPORT.md`.

Before your final response, use the shell to complete this checklist:

1. Run every reproduction command documented in `REPORT.md` from `/app` and require a zero exit status.
2. Confirm that every analysis source and every additional supporting output named in `REPORT.md` exists.
3. Confirm that each additional supporting output is a nonempty file recreated by the documented commands.

Every quantitative claim and conclusion in `REPORT.md` must be derived by the submitted reproduction material from the provided data. Do not hardcode claimed results, fabricate evidence, create placeholder outputs, or use commands or scripts that merely print conclusions. The verifier will inspect the submitted workspace, validate the original data against its hidden inventory, and ask an independent judge to run the documented reproduction commands in a clean audit directory containing the original `data/`, `REPORT.md`, any separate analysis source, and no generated outputs.
"""


def render_judge_instruction(task: dict[str, Any]) -> str:
    template = (TEMPLATE_DIR / "judge_instruction.md").read_text()
    replacements = {
        "{{HYPOTHESIS}}": str(task.get("hypothesis", "")).strip(),
        "{{PROTOCOL}}": str(task.get("protocol", "")).strip(),
        "{{RUBRIC}}": str(task.get("rubric", "")).strip(),
        "{{MAX_POINTS}}": str(task.get("max_points", "")),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def task_toml(
    source_config: dict[str, Any],
    *,
    task_id: str,
) -> str:
    environment = source_config.get("environment", {})
    agent = source_config.get("agent", {})
    image = environment.get("docker_image")
    workdir = environment.get("workdir", "/app")
    if not isinstance(image, str) or not image:
        raise ValueError("source task has no environment.docker_image")
    timeout = agent.get("timeout_sec", 3600)
    return f"""schema_version = "1.0"

[task]
name = {json.dumps(f"edison/{OUTPUT_PREFIX}{task_id}")}
version = "1.0"

[environment]
docker_image = {json.dumps(image)}
workdir = {json.dumps(workdir)}

[agent]
timeout_sec = {int(timeout)}

[verifier]
timeout_sec = 2400.0
environment_mode = "separate"

[verifier.env]
RUBRIC_MODEL = "${{RUBRIC_MODEL}}"
RUBRIC_MODEL_API_BASE = "${{RUBRIC_MODEL_API_BASE}}"
RUBRIC_MODEL_API_KEY = "${{RUBRIC_MODEL_API_KEY}}"

[[steps]]
name = "rollout"
"""


def write_task(
    source: Path,
    destination: Path,
    data_mode: str,
) -> None:
    task_id = source.name.removeprefix(SOURCE_PREFIX)
    hidden = read_jsonl_record(source / "environment" / "task.jsonl")
    if str(hidden.get("id", "")) != task_id:
        raise ValueError(f"task id mismatch in {source}")
    calibrate_rubric(hidden)

    with (source / "task.toml").open("rb") as handle:
        source_config = tomllib.load(handle)
    with (source / "environment" / "opencode.json").open() as handle:
        opencode = json.load(handle)
    opencode["default_agent"] = "analysis"
    opencode.pop("mcp", None)

    agent_dir = destination / "environment" / ".opencode" / "agents"
    tests_dir = destination / "steps" / "rollout" / "tests"
    workdir_dir = destination / "steps" / "rollout" / "workdir"
    agent_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    workdir_dir.mkdir(parents=True)

    copy_data(
        source / "environment" / "data", destination / "environment" / "data", data_mode
    )
    shutil.copy2(TEMPLATE_DIR / "analysis_agent.md", agent_dir / "analysis.md")
    shutil.copy2(TEMPLATE_DIR / "score_solution.py", tests_dir / "score_solution.py")
    shutil.copy2(TEMPLATE_DIR / "test.sh", tests_dir / "test.sh")
    shutil.copy2(TEMPLATE_DIR / "setup.sh", workdir_dir / "setup.sh")

    (destination / "environment" / "opencode.json").write_text(
        json.dumps(opencode, indent=2, ensure_ascii=False) + "\n"
    )
    (destination / "steps" / "rollout" / "instruction.md").write_text(
        render_instruction(hidden)
    )
    (tests_dir / "judge_instruction.md").write_text(render_judge_instruction(hidden))
    hidden["source_task_name"] = source_config.get("task", {}).get("name", source.name)
    hidden["input_file_manifest"] = inventory_input_files(
        source / "environment" / "data"
    )
    (tests_dir / "task.json").write_text(
        json.dumps(hidden, indent=2, ensure_ascii=False) + "\n"
    )
    (destination / "task.toml").write_text(
        task_toml(
            source_config,
            task_id=task_id,
        )
    )


def set_separate_verifier_environment(path: Path) -> None:
    task_toml_path = path / "task.toml"
    updated, replacements = re.subn(
        r'(?m)^environment_mode = "(?:same|separate)"$',
        'environment_mode = "separate"',
        task_toml_path.read_text(),
    )
    if replacements != 1:
        raise ValueError(
            f"expected exactly one verifier environment_mode in {task_toml_path}, "
            f"found {replacements}"
        )
    task_toml_path.write_text(updated)


def refresh_task(path: Path) -> None:
    tests_dir = path / "steps" / "rollout" / "tests"
    task_path = tests_dir / "task.json"
    task = json.loads(task_path.read_text())
    calibrate_rubric(task)
    shutil.copy2(
        TEMPLATE_DIR / "analysis_agent.md",
        path / "environment" / ".opencode" / "agents" / "analysis.md",
    )
    shutil.copy2(TEMPLATE_DIR / "score_solution.py", tests_dir / "score_solution.py")
    (path / "steps" / "rollout" / "instruction.md").write_text(render_instruction(task))
    (tests_dir / "judge_instruction.md").write_text(render_judge_instruction(task))
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
    set_separate_verifier_environment(path)
    validate_task(path)


def validate_task(path: Path) -> None:
    required = (
        path / "task.toml",
        path / "environment" / "opencode.json",
        path / "environment" / ".opencode" / "agents" / "analysis.md",
        path / "steps" / "rollout" / "instruction.md",
        path / "steps" / "rollout" / "tests" / "judge_instruction.md",
        path / "steps" / "rollout" / "tests" / "task.json",
        path / "steps" / "rollout" / "tests" / "score_solution.py",
        path / "steps" / "rollout" / "tests" / "test.sh",
        path / "steps" / "rollout" / "workdir" / "setup.sh",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"missing generated files: {missing}")
    if (path / "environment" / "task.jsonl").exists():
        raise ValueError(f"hidden task metadata leaked into policy environment: {path}")
    with (path / "task.toml").open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("environment", {}).get("mcp_servers"):
        raise ValueError(f"MCP server remains configured in {path}")
    if "env" in config.get("environment", {}):
        raise ValueError(f"verifier credentials remain in policy environment: {path}")
    if config.get("verifier", {}).get("environment_mode") != "separate":
        raise ValueError(f"agentic verifier must use a separate environment in {path}")
    task = json.loads((path / "steps" / "rollout" / "tests" / "task.json").read_text())
    item_points = rubric_item_points(task.get("rubric"))
    if task.get("rubric_item_points") != item_points:
        raise ValueError(
            f"generated rubric_item_points do not match rubric in {path}: "
            f"parsed={item_points}, stored={task.get('rubric_item_points')!r}"
        )
    total = sum(item_points)
    if total != task.get("max_points"):
        raise ValueError(
            f"generated rubric points do not match max_points in {path}: "
            f"rubric={total}, max_points={task.get('max_points')!r}"
        )
    policy_text = "\n".join(
        (
            (path / "steps" / "rollout" / "instruction.md").read_text(),
            (path / "environment" / ".opencode" / "agents" / "analysis.md").read_text(),
        )
    )
    if "submission.json" in policy_text:
        raise ValueError(f"legacy submission manifest remains policy-facing in {path}")
    text = "\n".join(item.read_text(errors="replace") for item in required)
    if "bbh_mcp" in text or "submit_answer" in text:
        raise ValueError(f"legacy BBH MCP reference remains in {path}")


def main() -> int:
    args = parse_args()
    if args.refresh_split is not None:
        if args.source_split is not None or args.output_split is not None:
            raise ValueError(
                "--refresh-split cannot be combined with --source-split or --output-split"
            )
        split = args.refresh_split.resolve()
        tasks = sorted(
            path for path in split.glob(f"{OUTPUT_PREFIX}*") if path.is_dir()
        )
        if not tasks:
            raise ValueError(f"no {OUTPUT_PREFIX}* tasks found under {split}")
        for index, task in enumerate(tasks, start=1):
            refresh_task(task)
            print(f"[{index}/{len(tasks)}] refreshed {task.name}")
        print(f"refreshed {len(tasks)} tasks under {split}")
        return 0

    if args.source_split is None or args.output_split is None:
        raise ValueError(
            "--source-split and --output-split are required unless --refresh-split is used"
        )
    source_split = args.source_split.resolve()
    output_split = args.output_split.resolve()
    tasks = sorted(
        path for path in source_split.glob(f"{SOURCE_PREFIX}*") if path.is_dir()
    )
    if not tasks:
        raise ValueError(f"no {SOURCE_PREFIX}* tasks found under {source_split}")
    output_split.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(tasks, start=1):
        task_id = source.name.removeprefix(SOURCE_PREFIX)
        destination = output_split / f"{OUTPUT_PREFIX}{task_id}"
        if destination.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"output exists (pass --overwrite): {destination}"
                )
            shutil.rmtree(destination)
        write_task(
            source,
            destination,
            args.data_mode,
        )
        validate_task(destination)
        print(f"[{index}/{len(tasks)}] {source.name} -> {destination.name}")

    print(f"converted {len(tasks)} tasks into {output_split}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
