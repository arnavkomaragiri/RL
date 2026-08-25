import json
from pathlib import Path

import pytest

from patches.harbor_opencode.write_nemo_rl_manifest import (
    filter_records,
    read_exclusions,
    records_from_manifest,
    resolve_excluded_task_names,
    task_names,
)


def make_task(split: Path, name: str) -> None:
    task_dir = split / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("version = '1.0'\n", encoding="utf-8")


def test_exclusions_match_exact_uuid_and_ignore_inline_comments(tmp_path: Path) -> None:
    split = tmp_path / "train"
    first = "bbh-task__11111111-1111-1111-1111-111111111111"
    second = "bbh-task__22222222-2222-2222-2222-222222222222"
    make_task(split, first)
    make_task(split, second)
    mask = tmp_path / "mask.txt"
    mask.write_text(
        "# blocked\n11111111-1111-1111-1111-111111111111 # reason\n",
        encoding="utf-8",
    )

    exclusions = read_exclusions([mask])
    excluded_tasks, unmatched = resolve_excluded_task_names(
        task_names(split), exclusions
    )
    records = [{"task_name": first}, {"task_name": second}]
    filtered, removed = filter_records(records, excluded_tasks)

    assert filtered == [{"task_name": second}]
    assert removed == [first]
    assert unmatched == []


def test_already_filtered_manifest_is_validated_against_reference_split(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train"
    excluded = "bbh-task__11111111-1111-1111-1111-111111111111"
    included = "bbh-task__22222222-2222-2222-2222-222222222222"
    make_task(split, excluded)
    make_task(split, included)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"task_name": included}) + "\n", encoding="utf-8")

    records = records_from_manifest(manifest)
    excluded_tasks, unmatched = resolve_excluded_task_names(
        task_names(split), {excluded.rsplit("__", maxsplit=1)[-1]}
    )
    filtered, removed = filter_records(records, excluded_tasks)

    assert filtered == records
    assert removed == []
    assert unmatched == []


def test_stale_exclusion_fails_loudly(tmp_path: Path) -> None:
    split = tmp_path / "train"
    make_task(split, "bbh-task__11111111-1111-1111-1111-111111111111")

    with pytest.raises(ValueError, match="absent from the reference"):
        resolve_excluded_task_names(
            task_names(split),
            {"22222222-2222-2222-2222-222222222222"},
            require_all=True,
        )
