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

"""Minimal MCP score sink for the artifact-analysis judge agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


APP_ROOT = Path("/app")
TASK_PATH = Path("/tests/task.json")
VERIFIER_LOGS = Path("/logs/verifier")
REWARD_PATH = VERIFIER_LOGS / "reward.json"
EVALUATION_PATH = VERIFIER_LOGS / "evaluation.json"
CODE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".ipynb",
    ".java",
    ".jl",
    ".lua",
    ".m",
    ".py",
    ".qmd",
    ".r",
    ".rmd",
    ".sas",
    ".scala",
    ".sh",
    ".sql",
    ".stan",
}
CONFIG_SUFFIXES = {".json", ".lock", ".toml", ".yaml", ".yml"}
SOURCE_NAMES = {"justfile", "makefile", "requirements.txt", "snakefile"}
DOCUMENT_SUFFIXES = {".md", ".rst"}
BASELINE_PATHS = {Path("opencode.json"), Path("setup.sh")}
EXCLUDED_PARTS = {
    ".git",
    ".harbor",
    ".opencode",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
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


class InvalidSubmissionError(ValueError):
    """A policy-caused integrity failure that must receive zero reward."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_files() -> list[Path]:
    files = []
    for path in APP_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(APP_ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append(relative)
    return sorted(files, key=lambda item: item.as_posix())


def is_analysis_source(path: Path) -> bool:
    if path.suffix.lower() in CODE_SUFFIXES or path.name.lower() in SOURCE_NAMES:
        return True
    if path.suffix.lower() in CONFIG_SUFFIXES:
        stem = path.stem.lower()
        return any(
            marker in stem
            for marker in (
                "config",
                "environment",
                "params",
                "requirements",
                "workflow",
            )
        )
    return False


def is_document(path: Path) -> bool:
    return (
        path.suffix.lower() in DOCUMENT_SUFFIXES
        or path.name.lower().startswith("readme")
        or path.name.lower() in {"license", "license.txt", "notice", "submission.json"}
    )


def rubric_item_points(rubric: Any) -> list[int]:
    if not isinstance(rubric, str) or not rubric.strip():
        raise ValueError("hidden task metadata has no rubric")
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
    return point_groups[0] or populated_groups[0]


def task_rubric_item_points(task: dict[str, Any]) -> list[int]:
    item_points = task.get("rubric_item_points")
    if item_points is None:
        return rubric_item_points(task.get("rubric"))
    if (
        not isinstance(item_points, list)
        or not item_points
        or any(
            isinstance(points, bool) or not isinstance(points, int) or points <= 0
            for points in item_points
        )
    ):
        raise ValueError(
            "hidden rubric_item_points must be a non-empty list of positive integers"
        )
    return item_points


def validate_input_inventory(task: dict[str, Any]) -> list[str]:
    records = task.get("input_file_manifest")
    if not isinstance(records, list) or not records:
        raise ValueError("hidden task metadata has no input_file_manifest")
    expected: dict[Path, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid hidden input inventory record")
        relative = Path(str(record.get("path", "")))
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "data"
            or ".." in relative.parts
        ):
            raise ValueError(f"invalid hidden input path: {relative}")
        expected[relative] = record

    actual = {
        path.relative_to(APP_ROOT)
        for path in (APP_ROOT / "data").rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise InvalidSubmissionError(
            "capsule input inventory changed during the policy rollout"
        )
    for relative, record in expected.items():
        path = APP_ROOT / relative
        if path.stat().st_size != record.get("size") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise InvalidSubmissionError(
                f"capsule input was modified during rollout: {relative}"
            )
    return [
        path.as_posix() for path in sorted(expected, key=lambda item: item.as_posix())
    ]


def validate_submission(task: dict[str, Any]) -> dict[str, Any]:
    report = APP_ROOT / "REPORT.md"
    if not report.is_file() or not report.read_text(errors="replace").strip():
        raise InvalidSubmissionError("REPORT.md is missing or empty")

    inputs = validate_input_inventory(task)
    source: list[str] = []
    artifacts: list[str] = []
    documents: list[str] = []
    for relative in workspace_files():
        absolute = APP_ROOT / relative
        if absolute.is_symlink():
            raise InvalidSubmissionError(
                f"policy-created symlinks are not supported: {relative}"
            )
        if relative.parts[0] == "data" or relative in BASELINE_PATHS:
            continue
        if relative == Path("REPORT.md") or is_document(relative):
            documents.append(relative.as_posix())
        elif is_analysis_source(relative):
            source.append(relative.as_posix())
        else:
            artifacts.append(relative.as_posix())

    return {
        "valid": True,
        "validation_errors": [],
        "analysis_source": source,
        "capsule_inputs": inputs,
        "generated_artifacts": artifacts,
        "documentation": documents,
    }


def validate_assessment(
    arguments: dict[str, Any], item_maxima: list[int]
) -> tuple[int, list[dict[str, Any]]]:
    score = arguments.get("score")
    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError("score must be an integer")
    maximum = sum(item_maxima)
    if not 0 <= score <= maximum:
        raise ValueError(f"score must be between 0 and {maximum}")

    assessment = arguments.get("rubric_assessment")
    if not isinstance(assessment, list) or len(assessment) != len(item_maxima):
        raise ValueError(
            f"rubric_assessment must contain exactly {len(item_maxima)} entries"
        )
    awarded = []
    for index, (item, item_maximum) in enumerate(zip(assessment, item_maxima), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"rubric_assessment entry {index} must be an object")
        points = item.get("points_awarded")
        if (
            isinstance(points, bool)
            or not isinstance(points, int)
            or not 0 <= points <= item_maximum
        ):
            raise ValueError(
                f"rubric_assessment entry {index} points_awarded must be between 0 and {item_maximum}"
            )
        if (
            not isinstance(item.get("justification"), str)
            or not item["justification"].strip()
        ):
            raise ValueError(f"rubric_assessment entry {index} needs a justification")
        awarded.append(points)
    if sum(awarded) != score:
        raise ValueError("rubric points awarded must sum to score")
    return score, assessment


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def score_solution(arguments: dict[str, Any]) -> dict[str, Any]:
    if REWARD_PATH.exists():
        raise ValueError("score_solution has already been called successfully")
    task = json.loads(TASK_PATH.read_text())
    item_maxima = task_rubric_item_points(task)
    if sum(item_maxima) != task.get("max_points"):
        raise ValueError("hidden rubric points do not match max_points")
    score, assessment = validate_assessment(arguments, item_maxima)
    try:
        submission = validate_submission(task)
    except InvalidSubmissionError as exc:
        submission = {
            "valid": False,
            "validation_errors": [str(exc)],
            "analysis_source": [],
            "capsule_inputs": [],
            "generated_artifacts": [],
            "documentation": [],
        }
    maximum = sum(item_maxima)
    effective_score = score if submission["valid"] else 0
    normalized_reward = effective_score / maximum
    evaluation = {
        "score": effective_score,
        "submitted_score": score,
        "max_points": maximum,
        "normalized_reward": normalized_reward,
        "summary": arguments.get("summary", ""),
        "rubric_assessment": assessment,
        "submission": submission,
    }
    write_json_atomic(EVALUATION_PATH, evaluation)
    write_json_atomic(REWARD_PATH, {"reward": normalized_reward})
    return {
        "accepted": True,
        "reward": normalized_reward,
        "submission_valid": submission["valid"],
    }


TOOL = {
    "name": "score_solution",
    "description": (
        "Submit the final rubric score after inspecting all analysis source, "
        "reproducing the analysis, and comparing the generated evidence."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer"},
            "summary": {"type": "string"},
            "rubric_assessment": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "points_awarded": {"type": "integer"},
                        "justification": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "criterion",
                        "points_awarded",
                        "justification",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["score", "summary", "rubric_assessment"],
        "additionalProperties": False,
    },
}


def response(
    request_id: Any, result: Any = None, error: dict[str, Any] | None = None
) -> None:
    payload = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        payload["result"] = result
    else:
        payload["error"] = error
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def serve_mcp() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if method == "initialize":
                requested = request.get("params", {}).get("protocolVersion")
                response(
                    request_id,
                    {
                        "protocolVersion": requested or "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "harbor-score", "version": "1.0.0"},
                    },
                )
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                response(request_id, {"tools": [TOOL]})
            elif method == "tools/call":
                params = request.get("params", {})
                if params.get("name") != "score_solution":
                    raise ValueError(f"unknown tool: {params.get('name')}")
                result = score_solution(params.get("arguments", {}))
                response(
                    request_id,
                    {"content": [{"type": "text", "text": json.dumps(result)}]},
                )
            elif request_id is not None:
                response(
                    request_id, error={"code": -32601, "message": "method not found"}
                )
        except Exception as exc:
            if "request_id" in locals() and request_id is not None:
                response(
                    request_id,
                    error={"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
                )
            else:
                print(
                    f"score server error: {type(exc).__name__}: {exc}", file=sys.stderr
                )


def main() -> None:
    serve_mcp()


if __name__ == "__main__":
    main()
