from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCORER_PATH = Path(__file__).parents[1] / "templates" / "score_solution.py"


def load_scorer():
    spec = importlib.util.spec_from_file_location(
        "test_score_solution_module", SCORER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer at {SCORER_PATH}")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


class ScoreSolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.app = self.root / "app"
        self.tests = self.root / "tests"
        self.logs = self.root / "logs" / "verifier"
        (self.app / "data").mkdir(parents=True)
        self.tests.mkdir()
        contents = b"input\n"
        (self.app / "data" / "input.csv").write_bytes(contents)
        self.task = {
            "rubric": "* 1 point: Reproducible result.",
            "rubric_item_points": [1],
            "max_points": 1,
            "input_file_manifest": [
                {
                    "path": "data/input.csv",
                    "size": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
            ],
        }
        (self.tests / "task.json").write_text(json.dumps(self.task))
        self.scorer = load_scorer()
        self.scorer.APP_ROOT = self.app
        self.scorer.TASK_PATH = self.tests / "task.json"
        self.scorer.VERIFIER_LOGS = self.logs
        self.scorer.REWARD_PATH = self.logs / "reward.json"
        self.scorer.EVALUATION_PATH = self.logs / "evaluation.json"

    def tearDown(self) -> None:
        self.temporary_dir.cleanup()

    def arguments(self, score: int = 1) -> dict:
        return {
            "score": score,
            "summary": "audited",
            "rubric_assessment": [
                {
                    "criterion": "Reproducible result.",
                    "points_awarded": score,
                    "justification": "The documented command reproduced the result.",
                    "evidence": ["command exited zero"],
                }
            ],
        }

    def test_accepts_report_only_reproducible_submission(self) -> None:
        (self.app / "REPORT.md").write_text(
            "# Result\n\nReproduce with `wc -l data/input.csv`.\n"
        )

        result = self.scorer.score_solution(self.arguments())

        self.assertEqual(result["reward"], 1.0)
        evaluation = json.loads(self.scorer.EVALUATION_PATH.read_text())
        self.assertTrue(evaluation["submission"]["valid"])
        self.assertEqual(evaluation["submission"]["analysis_source"], [])
        self.assertEqual(evaluation["submission"]["generated_artifacts"], [])

    def test_structured_rubric_points_do_not_parse_rubric_prose(self) -> None:
        (self.app / "REPORT.md").write_text("# Result\n")
        self.task["rubric"] = "Evaluate the result thoroughly."
        (self.tests / "task.json").write_text(json.dumps(self.task))

        result = self.scorer.score_solution(self.arguments())

        self.assertEqual(result["reward"], 1.0)

    def test_legacy_rubric_parser_accepts_converter_formats(self) -> None:
        cases = (
            ("* 1 point: Result.", [1]),
            ("- **2 points**: Result.\n- **3 pts**: Conclusion.", [2, 3]),
            ("1.\t1 point: Result.\n2. 2 points: Conclusion.", [1, 2]),
            ("\u2022\t1 point: Result.\n\u2022 4 points: Conclusion.", [1, 4]),
            ("Data preprocessing (2 pts)\nConclusion \u2013 3 points", [2, 3]),
        )
        for rubric, expected in cases:
            with self.subTest(rubric=rubric):
                self.assertEqual(self.scorer.rubric_item_points(rubric), expected)

    def test_legacy_task_uses_rubric_parser_fallback(self) -> None:
        (self.app / "REPORT.md").write_text("# Result\n")
        del self.task["rubric_item_points"]
        self.task["rubric"] = "- **1 point**: Reproducible result."
        (self.tests / "task.json").write_text(json.dumps(self.task))

        result = self.scorer.score_solution(self.arguments())

        self.assertEqual(result["reward"], 1.0)

    def test_policy_integrity_failure_is_terminal_zero(self) -> None:
        result = self.scorer.score_solution(self.arguments())

        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["submission_valid"])
        evaluation = json.loads(self.scorer.EVALUATION_PATH.read_text())
        self.assertEqual(evaluation["submitted_score"], 1)
        self.assertEqual(evaluation["score"], 0)
        self.assertEqual(
            evaluation["submission"]["validation_errors"],
            ["REPORT.md is missing or empty"],
        )
        self.assertEqual(
            evaluation["submission"]["capsule_inputs"], ["data/input.csv"]
        )
        self.assertEqual(
            json.loads(self.scorer.REWARD_PATH.read_text()), {"reward": 0.0}
        )

    def test_modified_capsule_input_is_terminal_zero(self) -> None:
        (self.app / "REPORT.md").write_text("# Result\n")
        (self.app / "data" / "input.csv").write_text("modified\n")

        result = self.scorer.score_solution(self.arguments())

        self.assertEqual(result["reward"], 0.0)
        evaluation = json.loads(self.scorer.EVALUATION_PATH.read_text())
        self.assertEqual(evaluation["score"], 0)
        self.assertIn(
            "capsule input was modified during rollout",
            evaluation["submission"]["validation_errors"][0],
        )

    def test_accepts_declared_empty_input_inventory(self) -> None:
        (self.app / "REPORT.md").write_text("# Result\n")
        (self.app / "data" / "input.csv").unlink()
        self.task["input_file_manifest"] = []
        (self.tests / "task.json").write_text(json.dumps(self.task))

        result = self.scorer.score_solution(self.arguments())

        self.assertEqual(result["reward"], 1.0)
        evaluation = json.loads(self.scorer.EVALUATION_PATH.read_text())
        self.assertEqual(evaluation["submission"]["capsule_inputs"], [])

    def test_hidden_task_error_still_raises(self) -> None:
        (self.app / "REPORT.md").write_text("# Result\n")
        del self.task["input_file_manifest"]
        (self.tests / "task.json").write_text(json.dumps(self.task))

        with self.assertRaisesRegex(ValueError, "hidden task metadata"):
            self.scorer.score_solution(self.arguments())

        self.assertFalse(self.scorer.REWARD_PATH.exists())


if __name__ == "__main__":
    unittest.main()
