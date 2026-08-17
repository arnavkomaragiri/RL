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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="Harbor split directory containing one directory per task.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--agent-name",
        default="gym_harbor_agent",
        help="Registered NeMo Gym responses API agent name.",
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


def main() -> None:
    args = parse_args()
    names = task_names(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as manifest:
        for name in names:
            record = {
                "task_name": name,
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": args.agent_name,
                },
                "responses_create_params": {"input": []},
            }
            manifest.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Wrote {len(names)} tasks to {args.output}")


if __name__ == "__main__":
    main()
