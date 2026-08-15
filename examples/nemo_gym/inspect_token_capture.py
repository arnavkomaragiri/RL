#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect retained NeMo Gym token-capture records without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _read_entries(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            entries.append(entry)
    return sorted(
        entries,
        key=lambda entry: (
            float(entry.get("created_at") or 0),
            str(entry["model_call_id"]),
        ),
    )


def _read_model_calls(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    calls = {}
    for entry in _read_entries(path):
        model_call_id = entry.get("model_call_id")
        if isinstance(model_call_id, str):
            calls[model_call_id] = entry
    return calls


def _request_summary(exchange: dict[str, Any] | None) -> str:
    if not exchange:
        return "request=unavailable"
    request = exchange.get("request")
    if not isinstance(request, dict):
        return f"dialect={exchange.get('dialect', 'unknown')} request=unparsed"
    input_items = request.get("input")
    if not isinstance(input_items, list):
        input_items = request.get("messages")
    items = input_items if isinstance(input_items, list) else []
    item_kinds = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_kinds.append(str(item.get("type") or item.get("role") or "unknown"))
    encoded = json.dumps(request, sort_keys=True).lower()
    markers = [
        marker
        for marker in ("title", "summary", "summarize", "compact")
        if marker in encoded
    ]
    return (
        f"dialect={exchange.get('dialect', 'unknown')} input_items={len(items)} "
        f"item_kinds={','.join(item_kinds) or '-'} tools={len(request.get('tools') or [])} "
        f"markers={','.join(markers) or '-'}"
    )


def _inspect(path: Path, model_call_dir: Path | None) -> None:
    entries = _read_entries(path)
    rollout_id = path.name.removesuffix(".tokens.jsonl")
    model_calls = _read_model_calls(
        model_call_dir / f"{rollout_id}.capture.jsonl"
        if model_call_dir is not None
        else None
    )
    print(f"\n{path}: {len(entries)} call(s)")
    print(
        "idx prompt generation previous_cumulative prompt_delta lcp closest_cumulative status call_id"
    )
    previous: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        prompt = list(entry.get("prompt_token_ids") or [])
        generation = list(entry.get("generation_token_ids") or [])
        if not previous:
            print(
                f"{index:>3} {len(prompt):>6} {len(generation):>10} {'-':>19} {'-':>12} "
                f"{'-':>5} {'-':>18} root {entry['model_call_id']}"
            )
        else:
            immediate = previous[-1]
            immediate_prompt = list(immediate.get("prompt_token_ids") or [])
            immediate_generation = list(immediate.get("generation_token_ids") or [])
            immediate_cumulative = immediate_prompt + immediate_generation
            closest = max(
                previous,
                key=lambda candidate: _common_prefix_length(
                    list(candidate.get("prompt_token_ids") or [])
                    + list(candidate.get("generation_token_ids") or []),
                    prompt,
                ),
            )
            closest_cumulative = list(closest.get("prompt_token_ids") or []) + list(
                closest.get("generation_token_ids") or []
            )
            common = _common_prefix_length(closest_cumulative, prompt)
            exact = common == len(closest_cumulative)
            if exact:
                status = "extends"
            elif len(prompt) < len(immediate_cumulative):
                status = "prior-output-not-refed"
            else:
                status = "rewritten"
            print(
                f"{index:>3} {len(prompt):>6} {len(generation):>10} {len(immediate_cumulative):>19} "
                f"{len(prompt) - len(immediate_prompt):>12} {common:>5} {len(closest_cumulative):>18} "
                f"{status} {entry['model_call_id']}"
            )
        print(f"    {_request_summary(model_calls.get(str(entry['model_call_id'])))}")
        previous.append(entry)

    captured = sum(len(entry.get("generation_token_ids") or []) for entry in entries)
    print(f"generated_tokens_captured={captured}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_dir",
        type=Path,
        help="Directory containing retained *.tokens.jsonl files",
    )
    parser.add_argument(
        "--model-call-dir",
        type=Path,
        help="Optional directory containing matching *.capture.jsonl request records",
    )
    args = parser.parse_args()

    paths = sorted(args.capture_dir.glob("*.tokens.jsonl"))
    if not paths:
        raise SystemExit(f"No *.tokens.jsonl files found under {args.capture_dir}")
    for path in paths:
        _inspect(path, args.model_call_dir)


if __name__ == "__main__":
    main()
