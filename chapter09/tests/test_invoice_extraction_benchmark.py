from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = CHAPTER_DIR / "invoice-extraction.ipynb"
RESULT_PATH = CHAPTER_DIR / "results" / "invoice_extraction_benchmark.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def python_source(cell: dict) -> str:
    lines = []
    for line in "".join(cell.get("source", [])).splitlines():
        lines.append("pass" if line.lstrip().startswith(("%", "!")) else line)
    return "\n".join(lines)


class InvoiceExtractionBenchmarkTest(unittest.TestCase):
    def test_notebook_code_is_valid_and_outputs_are_cleared(self) -> None:
        notebook = load_json(NOTEBOOK_PATH)

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            with self.subTest(cell=index):
                ast.parse(python_source(cell) or "pass")
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_result_uses_only_the_declared_holdout(self) -> None:
        result = load_json(RESULT_PATH)

        self.assertEqual(result["holdout_size"], 2)
        self.assertEqual(result["inference_repeats"], 3)
        self.assertEqual(
            result["holdout_invoice_ids"],
            ["missing-bank-details", "non-standard-labels"],
        )
        self.assertTrue(
            set(result["train_invoice_ids"]).isdisjoint(result["holdout_invoice_ids"])
        )

        for approach in result["approaches"].values():
            self.assertEqual(len(approach["predictions"]), 6)
            self.assertEqual(
                {prediction["invoice_id"] for prediction in approach["predictions"]},
                set(result["holdout_invoice_ids"]),
            )

    def test_recorded_scores_and_gepa_diagnosis_are_internally_consistent(self) -> None:
        result = load_json(RESULT_PATH)
        approaches = result["approaches"]

        self.assertEqual(approaches["baseline"]["exact_field_accuracy_pct"], 81.25)
        self.assertEqual(approaches["few_shot"]["exact_field_accuracy_pct"], 87.5)
        self.assertEqual(
            approaches["majority_voting"]["exact_field_accuracy_pct"], 81.25
        )
        self.assertEqual(approaches["gepa"]["exact_field_accuracy_pct"], 81.25)

        optimization = result["gepa_optimization_usage"]
        self.assertGreater(optimization["task_lm"]["requests"], 0)
        self.assertEqual(optimization["reflection_lm"]["requests"], 0)
        self.assertEqual(
            result["gepa_learned_instructions"],
            ["Extract structured invoice fields from free-form invoice text."],
        )


if __name__ == "__main__":
    unittest.main()
