from __future__ import annotations

import unittest
from unittest.mock import patch

from chapter06.build_optimizer_notebooks import CHAPTER_DIR, NOTEBOOKS
from chapter06.notebook_support import optimizer_row, verify_prompt_artifact
from chapter06.optimizer_runtime import AIDetector, _compile_prompt_optimizer
from chapter06.validate_notebooks import validate_notebook


class PublishedNotebookTest(unittest.TestCase):
    def test_every_optimizer_notebook_is_executed_and_educational(self) -> None:
        for filename, spec in NOTEBOOKS.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    validate_notebook(
                        CHAPTER_DIR / filename,
                        spec["optimizer"],
                        require_executed=True,
                    ),
                    [],
                )

    def test_frozen_program_prompt_state_matches_extracted_prompt(self) -> None:
        for spec in NOTEBOOKS.values():
            optimizer = spec["optimizer"]
            row = optimizer_row(optimizer)
            check = verify_prompt_artifact(optimizer)
            with self.subTest(optimizer=optimizer):
                self.assertEqual(row["status"], "completed")
                self.assertTrue(check["checked"])
                self.assertTrue(check["prompt_state_equal"])

    def test_every_notebook_loads_the_shared_split_and_can_run_live(self) -> None:
        for filename in NOTEBOOKS:
            source = (CHAPTER_DIR / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn("load_frozen_examples", source)
                self.assertIn("run_optimizer", source)
                self.assertIn("CHAPTER06_RUN_LIVE", source)

    def test_gepa_uses_the_standard_light_configuration_and_protocol(self) -> None:
        spec = NOTEBOOKS["gepa.ipynb"]
        self.assertIn("auto='light'", spec["compile"])
        self.assertIn("candidate_selection_strategy='pareto'", spec["compile"])
        self.assertIn("use_merge=False", spec["compile"])

        row = optimizer_row("gepa")
        self.assertEqual(
            row["evaluation_protocol"],
            "one fresh uncached validation pass, then one locked-test pass",
        )
        self.assertEqual(row["baseline_accuracy_pct"], 53.75)
        self.assertEqual(row["locked_test_accuracy_pct"], 80.0)

    def test_live_gepa_runner_matches_the_published_configuration(self) -> None:
        class FakeGEPA:
            kwargs: dict[str, object] = {}

            def __init__(self, **kwargs: object) -> None:
                type(self).kwargs = kwargs

            def compile(self, student: AIDetector, **kwargs: object) -> AIDetector:
                return student

        with patch("chapter06.optimizer_runtime.dspy.GEPA", FakeGEPA):
            _compile_prompt_optimizer(
                "gepa",
                AIDetector(),
                [],
                [],
                object(),
                object(),
                smoke=False,
            )

        self.assertEqual(FakeGEPA.kwargs["auto"], "light")
        self.assertNotIn("max_full_evals", FakeGEPA.kwargs)
        self.assertEqual(FakeGEPA.kwargs["candidate_selection_strategy"], "pareto")
        self.assertIs(FakeGEPA.kwargs["use_merge"], False)


if __name__ == "__main__":
    unittest.main()
