from pathlib import Path

import pytest

from patches.harbor_opencode.convert_bbh_to_opencode import (
    calibrate_rubric,
    rubric_item_points,
    rubric_point_total,
    set_separate_verifier_environment,
    task_toml,
)


@pytest.mark.parametrize(
    ("rubric", "expected"),
    [
        ("* 1 point: Load data.\n* 4 points: Evaluate the hypothesis.", [1, 4]),
        ("- **2 points**: Load data.\n- **3 pts**: Analyze it.", [2, 3]),
        ("1.\t1 point: Load data.\n2. 2 points: Analyze it.", [1, 2]),
        ("\u2022\t1 point: Load data.\n\u2022 4 points: Analyze it.", [1, 4]),
        (
            "1. Sample filtering (2 pts): Filter (1 pt) and align (1 pt).\n"
            "Data analysis - 3 pts",
            [2, 3],
        ),
        ("Data preprocessing (2 pts)\nConclusion \u2013 3 points", [2, 3]),
    ],
)
def test_rubric_item_points_formats(rubric: str, expected: list[int]) -> None:
    assert rubric_item_points(rubric) == expected
    assert rubric_point_total(rubric) == sum(expected)


def test_rubric_point_total_rejects_unitemized_rubric() -> None:
    with pytest.raises(ValueError, match="no itemized point values"):
        rubric_point_total("Analyze the data thoroughly.")


def test_rubric_point_total_does_not_double_count_nested_allocations() -> None:
    rubric = (
        "Data preprocessing (2 pts)\n"
        "- 1 pt: Load the data.\n"
        "- 1 pt: Filter invalid rows.\n"
        "Conclusion (3 pts)\n"
        "- 3 pts: State whether the hypothesis is supported."
    )
    assert rubric_item_points(rubric) == [1, 1, 3]
    assert rubric_point_total(rubric) == 5


def test_rubric_point_total_rejects_incompatible_mixed_allocations() -> None:
    rubric = "Data preprocessing (3 pts)\n- 1 pt: Load the data."
    with pytest.raises(ValueError, match="incompatible prefix and heading"):
        rubric_point_total(rubric)


def test_calibrate_rubric_corrects_known_bad_maximum() -> None:
    task = {
        "id": "75adb693-e3d5-4f23-bd1d-d2341e209d66",
        "rubric": "Filtering (2 pts): Correctly filter.\nConclusion (15 pts): Conclude.",
        "max_points": 29,
    }

    calibrate_rubric(task)

    assert task["max_points"] == 17
    assert task["rubric_item_points"] == [2, 15]


def test_calibrate_rubric_corrects_htt_graph_item_cardinality() -> None:
    task = {
        "id": "03db3687-1fcf-4c5c-8a2e-f779d7eb3cf0",
        "rubric": (
            "1 point: Extract ensembles.\n"
            "1 point: Isolate Exon 1.\n"
            "1 point: Calculate radius of gyration.\n"
            "1 point: Construct contact frequencies.\n"
            "1 ponit: builds the weighted graph network object with residues as "
            "nodes and contact frequencies as edge weights.\n"
            "1 point: Calculate closeness centrality.\n"
            "1 point: Compare centrality profiles.\n"
            "5 points: Approve or reject the hypothesis."
        ),
        "max_points": 11,
    }

    calibrate_rubric(task)

    assert task["max_points"] == 12
    assert task["rubric_item_points"] == [1, 1, 1, 1, 1, 1, 1, 5]
    assert rubric_point_total(task["rubric"]) == 12
    assert len(task["rubric"].splitlines()) == 8
    assert "ponit" not in task["rubric"]


def test_task_toml_uses_separate_verifier_environment() -> None:
    rendered = task_toml(
        {
            "environment": {"docker_image": "example/image:latest", "workdir": "/app"},
            "agent": {"timeout_sec": 3600},
        },
        task_id="task-id",
    )

    assert 'environment_mode = "separate"' in rendered


def test_refresh_restores_separate_verifier_environment(tmp_path: Path) -> None:
    task_path = tmp_path / "task"
    task_path.mkdir()
    config_path = task_path / "task.toml"
    config_path.write_text('[verifier]\nenvironment_mode = "same"\n')

    set_separate_verifier_environment(task_path)

    assert config_path.read_text() == '[verifier]\nenvironment_mode = "separate"\n'
