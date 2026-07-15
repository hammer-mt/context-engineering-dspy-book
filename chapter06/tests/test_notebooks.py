from __future__ import annotations

import unittest

from chapter06.build_optimizer_notebooks import CHAPTER_DIR, NOTEBOOKS
from chapter06.notebook_support import optimizer_row, verify_prompt_artifact
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
                if row["status"] == "completed":
                    self.assertTrue(check["checked"])
                    self.assertTrue(check["prompt_state_equal"])
                else:
                    self.assertEqual(row["status"], "hardware_blocked")
                    self.assertFalse(check["checked"])


if __name__ == "__main__":
    unittest.main()
