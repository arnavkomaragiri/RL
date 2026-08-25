#!/usr/bin/env python3

"""Audit exact Gym token captures against NeMo-RL logprob diagnostics.

The audit follows the production data path without loading a model:

1. Read the exact prompt, generation, and logprob arrays captured per model call.
2. Run Gym's production prefix builder to exclude only proven-unused retries.
3. Run Gym's independent-call projection and prove that it changes no token array.
4. Reconstruct NeMo-RL's concatenated rollout row and independent attention segments.
5. Match diagnostic rows by total/trainable length and verify every reported token,
   attention boundary, loss offset, context window, and captured generation logprob.

This script intentionally imports the production Gym builder instead of carrying a
second implementation whose agreement would not establish what training consumed.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_ROOT = REPO_ROOT / "3rdparty" / "Gym-workspace" / "Gym"
sys.path.insert(0, str(GYM_ROOT))


@dataclass
class TokenEntry:
    """Dependency-light adapter for the fields consumed by Gym's builder."""

    schema_version: int
    rollout_id: str
    model_call_id: str
    model: str
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    routed_experts: Any | None
    output_items: list[dict[str, Any]]
    token_item_index: int | None
    created_at: float

    @classmethod
    def model_validate(cls, record: dict[str, Any]) -> "TokenEntry":
        return cls(
            schema_version=int(record.get("schema_version", 1)),
            rollout_id=str(record["rollout_id"]),
            model_call_id=str(record["model_call_id"]),
            model=str(record.get("model", "")),
            prompt_token_ids=list(record["prompt_token_ids"]),
            generation_token_ids=list(record["generation_token_ids"]),
            generation_log_probs=list(record["generation_log_probs"]),
            routed_experts=record.get("routed_experts"),
            output_items=list(record.get("output_items") or []),
            token_item_index=record.get("token_item_index"),
            created_at=float(record.get("created_at", 0.0)),
        )


def _install_builder_import_adapter() -> None:
    """Load production builder code without importing Gym's full runtime package.

    Retained Gym venvs point to the Python installation inside the training
    container. The builder itself is pure Python and only needs a record with the
    fields above, so a tiny package adapter lets the login-node Python execute the
    exact production builder and projection implementation.
    """

    nemo_gym_package = types.ModuleType("nemo_gym")
    nemo_gym_package.__path__ = [str(GYM_ROOT / "nemo_gym")]
    capture_package = types.ModuleType("nemo_gym.token_id_capture")
    capture_package.__path__ = [str(GYM_ROOT / "nemo_gym" / "token_id_capture")]
    records_module = types.ModuleType("nemo_gym.token_id_capture.records")
    records_module.TokenEntry = TokenEntry
    sys.modules.setdefault("nemo_gym", nemo_gym_package)
    sys.modules.setdefault("nemo_gym.token_id_capture", capture_package)
    sys.modules.setdefault("nemo_gym.token_id_capture.records", records_module)


_install_builder_import_adapter()

from nemo_gym.token_id_capture.builder import (  # noqa: E402
    project_independent_call_responses,
    run_builder,
)


@dataclass(frozen=True)
class Segment:
    entry: TokenEntry
    start: int
    end: int

    @property
    def prompt_length(self) -> int:
        return len(self.entry.prompt_token_ids)

    @property
    def generation_length(self) -> int:
        return len(self.entry.generation_token_ids)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def _load_diagnostics(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [record for file_path in files for record in _read_jsonl(file_path)]


def _rollout_task_indices(harbor_jobs_dir: Path | None) -> dict[str, int]:
    if harbor_jobs_dir is None or not harbor_jobs_dir.is_dir():
        return {}
    result: dict[str, int] = {}
    pattern = re.compile(r"^t(?P<task>\d+)-r\d+-(?P<rollout>.+)$")
    for path in harbor_jobs_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            result[match.group("rollout")] = int(match.group("task"))
    return result


def _projected_token_item(response: dict[str, Any]) -> dict[str, Any]:
    generated = [
        item
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("generation_token_ids") is not None
    ]
    if len(generated) != 1:
        raise ValueError(
            f"independent response has {len(generated)} token-bearing output items"
        )
    return generated[0]


def _segments_for_entries(entries: list[TokenEntry]) -> tuple[list[Segment], dict[str, Any]]:
    build = run_builder(entries, "prefix_merging")
    unused_retries = set(build.notes.unused_retry_calls)
    usable = sorted(
        (
            entry
            for entry in entries
            if entry.generation_token_ids and entry.model_call_id not in unused_retries
        ),
        key=lambda entry: (entry.created_at, entry.model_call_id),
    )
    projected = project_independent_call_responses("audit", entries, build)
    if len(projected) != len(usable):
        raise ValueError(
            "independent-call projection count differs from usable capture count: "
            f"{len(projected)} != {len(usable)}"
        )

    offset = 0
    segments: list[Segment] = []
    for entry, response in zip(usable, projected):
        item = _projected_token_item(response)
        if item["prompt_token_ids"] != entry.prompt_token_ids:
            raise ValueError(f"projection changed prompt tokens for {entry.model_call_id}")
        if item["generation_token_ids"] != entry.generation_token_ids:
            raise ValueError(
                f"projection changed generation tokens for {entry.model_call_id}"
            )
        if item["generation_log_probs"] != entry.generation_log_probs:
            raise ValueError(
                f"projection changed generation logprobs for {entry.model_call_id}"
            )
        length = len(entry.prompt_token_ids) + len(entry.generation_token_ids)
        segments.append(Segment(entry=entry, start=offset, end=offset + length))
        offset += length

    return segments, {
        "chains": build.notes.chains,
        "roots": build.notes.roots,
        "quarantined_calls": len(build.quarantined),
        "unused_retry_calls": len(build.notes.unused_retry_calls),
        "unresolved_retries": len(build.notes.unresolved_retries),
        "empty_generation_calls": len(build.notes.empty_generation_calls),
        "retokenized_boundaries": build.notes.retokenized_boundaries,
    }


def _token_at(segment: Segment, local_position: int) -> int:
    if local_position < segment.prompt_length:
        return segment.entry.prompt_token_ids[local_position]
    return segment.entry.generation_token_ids[local_position - segment.prompt_length]


def _segment_token_slice(
    segment: Segment, absolute_start: int, absolute_end: int
) -> list[int]:
    local_start = absolute_start - segment.start
    local_end = absolute_end - segment.start
    sequence = segment.entry.prompt_token_ids + segment.entry.generation_token_ids
    return sequence[local_start:local_end]


def _verify_diagnostic(
    record: dict[str, Any], rollout_id: str, segments: list[Segment]
) -> list[str]:
    errors: list[str] = []
    starts = [segment.start for segment in segments]
    full_length = segments[-1].end if segments else 0
    trainable_tokens = sum(segment.generation_length for segment in segments)
    if record.get("full_length") != full_length:
        errors.append(
            f"full_length {record.get('full_length')} != reconstructed {full_length}"
        )
    if record.get("trainable_token_count") != trainable_tokens:
        errors.append(
            "trainable_token_count "
            f"{record.get('trainable_token_count')} != reconstructed {trainable_tokens}"
        )

    for rank, token in enumerate(record.get("top_tokens", [])):
        position = int(token["token_position"])
        segment_index = bisect.bisect_right(starts, position) - 1
        prefix = f"top_token[{rank}] position={position}"
        if segment_index < 0 or segment_index >= len(segments):
            errors.append(f"{prefix}: outside reconstructed segments")
            continue
        segment = segments[segment_index]
        local_position = position - segment.start
        if position >= segment.end:
            errors.append(f"{prefix}: falls in no reconstructed segment")
            continue
        if token.get("attention_segment_index") != segment_index:
            errors.append(
                f"{prefix}: attention segment index {token.get('attention_segment_index')} "
                f"!= {segment_index}"
            )
        if token.get("attention_segment_start") != segment.start:
            errors.append(
                f"{prefix}: attention start {token.get('attention_segment_start')} "
                f"!= {segment.start}"
            )
        if token.get("attention_segment_end") != segment.end:
            errors.append(
                f"{prefix}: attention end {token.get('attention_segment_end')} "
                f"!= {segment.end}"
            )
        captured_token = _token_at(segment, local_position)
        if token.get("token_id") != captured_token:
            errors.append(
                f"{prefix}: token id {token.get('token_id')} != captured {captured_token}"
            )

        generation_offset = local_position - segment.prompt_length
        if generation_offset < 0:
            errors.append(f"{prefix}: diagnostic selected a prompt token for loss")
            continue
        if generation_offset >= segment.generation_length:
            errors.append(f"{prefix}: generation offset is outside captured generation")
            continue
        captured_logprob = segment.entry.generation_log_probs[generation_offset]
        diagnostic_logprob = token.get("generation_logprob")
        if diagnostic_logprob is None or not math.isclose(
            float(diagnostic_logprob),
            float(captured_logprob),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            errors.append(
                f"{prefix}: generation logprob {diagnostic_logprob} "
                f"!= captured {captured_logprob}"
            )
        if token.get("loss_span_index") != segment_index:
            errors.append(
                f"{prefix}: loss span index {token.get('loss_span_index')} != {segment_index}"
            )
        if token.get("distance_from_loss_span_start") != generation_offset:
            errors.append(
                f"{prefix}: loss-start distance "
                f"{token.get('distance_from_loss_span_start')} != {generation_offset}"
            )
        expected_to_end = segment.generation_length - generation_offset - 1
        if token.get("distance_to_loss_span_end") != expected_to_end:
            errors.append(
                f"{prefix}: loss-end distance {token.get('distance_to_loss_span_end')} "
                f"!= {expected_to_end}"
            )
        context_start = int(token["context_start"])
        context_end = context_start + len(token.get("context_token_ids", []))
        expected_context = _segment_token_slice(segment, context_start, context_end)
        if token.get("context_token_ids") != expected_context:
            errors.append(f"{prefix}: context tokens differ from captured sequence")

    return errors


def _audit_capture(path: Path) -> tuple[list[Segment], dict[str, Any], list[str]]:
    entries = [TokenEntry.model_validate(record) for record in _read_jsonl(path)]
    errors: list[str] = []
    call_ids = [entry.model_call_id for entry in entries]
    if len(call_ids) != len(set(call_ids)):
        errors.append("duplicate model_call_id in capture")
    for entry in entries:
        if len(entry.generation_token_ids) != len(entry.generation_log_probs):
            errors.append(
                f"{entry.model_call_id}: generation token/logprob length mismatch"
            )
        if not all(math.isfinite(value) for value in entry.generation_log_probs):
            errors.append(f"{entry.model_call_id}: non-finite generation logprob")
    try:
        segments, build_metrics = _segments_for_entries(entries)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        errors.append(f"production projection failed: {type(error).__name__}: {error}")
        segments = []
        build_metrics = {}
    metrics = {
        "captured_calls": len(entries),
        "training_segments": len(segments),
        "full_length": segments[-1].end if segments else 0,
        "trainable_token_count": sum(
            segment.generation_length for segment in segments
        ),
        "max_segment_length": max(
            (segment.end - segment.start for segment in segments), default=0
        ),
        "calls_with_routes": sum(
            entry.routed_experts is not None for entry in entries
        ),
        **build_metrics,
    }
    return segments, metrics, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="A token_logprob_outliers JSONL file or directory containing them.",
    )
    parser.add_argument(
        "--harbor-jobs-dir",
        type=Path,
        help="Optional directory used to attach dataset task indices to rollout IDs.",
    )
    parser.add_argument("--output", type=Path, help="Write the full JSON report here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = _load_diagnostics(args.diagnostics)
    diagnostics_by_lengths: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in diagnostics:
        key = (int(record["full_length"]), int(record["trainable_token_count"]))
        diagnostics_by_lengths.setdefault(key, []).append(record)

    task_indices = _rollout_task_indices(args.harbor_jobs_dir)
    capture_reports: list[dict[str, Any]] = []
    matches_by_diagnostic: dict[int, list[str]] = {
        id(record): [] for record in diagnostics
    }
    total_errors = 0
    total_calls = 0
    total_segments = 0
    calls_with_routes = 0

    for path in sorted(args.capture_dir.glob("*.tokens.jsonl")):
        rollout_id = path.name.removesuffix(".tokens.jsonl")
        segments, metrics, errors = _audit_capture(path)
        total_calls += metrics["captured_calls"]
        total_segments += metrics["training_segments"]
        calls_with_routes += metrics["calls_with_routes"]
        key = (metrics["full_length"], metrics["trainable_token_count"])
        matched_records = diagnostics_by_lengths.get(key, [])
        diagnostic_checks = []
        for record in matched_records:
            matches_by_diagnostic[id(record)].append(rollout_id)
            diagnostic_errors = _verify_diagnostic(record, rollout_id, segments)
            errors.extend(diagnostic_errors)
            diagnostic_checks.append(
                {
                    "step": record.get("step"),
                    "sample_index": record.get("sample_index"),
                    "top_tokens_checked": len(record.get("top_tokens", [])),
                    "errors": diagnostic_errors,
                }
            )
        total_errors += len(errors)
        capture_reports.append(
            {
                "rollout_id": rollout_id,
                "task_index": task_indices.get(rollout_id),
                **metrics,
                "diagnostic_checks": diagnostic_checks,
                "errors": errors,
            }
        )

    unmatched = []
    ambiguous = []
    for record in diagnostics:
        matches = matches_by_diagnostic[id(record)]
        identity = {
            "step": record.get("step"),
            "sample_index": record.get("sample_index"),
            "full_length": record.get("full_length"),
            "trainable_token_count": record.get("trainable_token_count"),
        }
        if not matches:
            unmatched.append(identity)
        elif len(matches) > 1:
            ambiguous.append({**identity, "rollout_ids": matches})

    report = {
        "capture_dir": str(args.capture_dir),
        "diagnostics": str(args.diagnostics) if args.diagnostics else None,
        "summary": {
            "captures_scanned": len(capture_reports),
            "captured_calls": total_calls,
            "training_segments": total_segments,
            "calls_with_routes": calls_with_routes,
            "diagnostic_records": len(diagnostics),
            "diagnostic_records_matched_once": len(diagnostics)
            - len(unmatched)
            - len(ambiguous),
            "diagnostic_records_unmatched": len(unmatched),
            "diagnostic_records_ambiguous": len(ambiguous),
            "invariant_errors": total_errors,
        },
        "unmatched_diagnostics": unmatched,
        "ambiguous_diagnostics": ambiguous,
        "captures": capture_reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    matched = [item for item in capture_reports if item["diagnostic_checks"]]
    for item in matched:
        print(
            f"matched rollout={item['rollout_id']} task_index={item['task_index']} "
            f"segments={item['training_segments']} full_length={item['full_length']} "
            f"trainable={item['trainable_token_count']} errors={len(item['errors'])}"
        )
    if total_errors or unmatched or ambiguous:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
