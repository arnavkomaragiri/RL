import argparse
import json
from pathlib import Path

from patches.harbor_opencode.build_hypotest_harbor_bundle import (
    build_bundle,
    validate_task_records,
)


CAPSULE_ID = "11111111-1111-1111-1111-111111111111"


def task(rubric: str, *, hypothesis: str) -> dict[str, object]:
    return {
        "answer": "Supported.",
        "hypothesis": hypothesis,
        "id": CAPSULE_ID,
        "input_data_path": f"capsule_{CAPSULE_ID}",
        "max_points": 2,
        "nb_primary_language": "Python",
        "orig_rubric": "* 1 point: Analyze.\n* 1 point: Conclude.",
        "protocol": "Analyze input.csv and report the conclusion.",
        "rubric": rubric,
    }


def test_build_preserves_paraphrases_and_deduplicates_capsules(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    records = [
        task("Unparseable rubric text.", hypothesis="First paraphrase."),
        task(
            "* 1 point: Analyze.\n* 1 point: Conclude.", hypothesis="Second paraphrase."
        ),
    ]
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    capsules = tmp_path / "source-capsules"
    capsule = capsules / f"capsule_{CAPSULE_ID}"
    capsule.mkdir(parents=True)
    source_data = capsule / "input.csv"
    source_data.write_text("value\n1\n")
    output = tmp_path / "output"
    args = argparse.Namespace(
        tasks_jsonl=tasks_path,
        capsules_dir=capsules,
        output_root=output,
        docker_image="example.invalid/bixbench:test",
        capsule_inventory_jsonl=None,
        exclude_file=[],
        capsule_mode="hardlink",
        overwrite=False,
    )

    report = build_bundle(args)

    manifest = [
        json.loads(line)
        for line in (output / "manifests" / "train.jsonl").read_text().splitlines()
    ]
    assert [row["task_name"] for row in manifest] == [
        f"bbh-task__00000000__{CAPSULE_ID}",
        f"bbh-task__00000001__{CAPSULE_ID}",
    ]
    assert report["output"]["tasks"] == 2
    assert report["output"]["capsules"] == 1
    assert len(report["output"]["manifest_sha256"]) == 64
    assert report["rubrics"]["fallback_rows"] == 1
    first_task = output / "datasets" / "train" / manifest[0]["task_name"]
    hidden = json.loads(
        (first_task / "steps" / "rollout" / "tests" / "task.json").read_text()
    )
    assert hidden["rubric"] == hidden["orig_rubric"]
    assert hidden["harbor_conversion"]["rubric_source"] == "orig_rubric"
    assert not (first_task / "environment" / "data").exists()
    output_data = output / "capsules" / f"capsule_{CAPSULE_ID}" / "input.csv"
    assert output_data.stat().st_ino == source_data.stat().st_ino


def test_filtering_preserves_source_row_index(tmp_path: Path) -> None:
    second_id = "22222222-2222-2222-2222-222222222222"
    first = task("* 1 point: Analyze.\n* 1 point: Conclude.", hypothesis="First.")
    second = task("* 1 point: Analyze.\n* 1 point: Conclude.", hypothesis="Second.")
    second["id"] = second_id
    second["input_data_path"] = f"capsule_{second_id}"
    capsules = tmp_path / "capsules"
    (capsules / f"capsule_{CAPSULE_ID}").mkdir(parents=True)
    (capsules / f"capsule_{second_id}").mkdir()

    kept, dropped = validate_task_records([first, second], capsules, {CAPSULE_ID})

    assert kept == [(1, second)]
    assert dropped == [CAPSULE_ID]
