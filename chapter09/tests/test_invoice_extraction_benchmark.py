"""Offline checks for the Chapter 9 invoice-extraction notebook and its recorded run.

The notebook accompanies "Invoice Entity Extraction" in Chapter 9. Its
held-out benchmark is supporting material: the checked-in JSON file is the
record of one run, and the notebook shows that record in a table. These tests
keep the notebook, the table, and the record consistent. They make no model
calls.
"""

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

    def test_recorded_run_table_matches_the_result_file(self) -> None:
        result = load_json(RESULT_PATH)
        notebook = load_json(NOTEBOOK_PATH)
        recorded = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and "".join(cell["source"]).startswith("### Recorded run")
        )

        # The table names the model that produced the record, which is not
        # the notebook's default model.
        self.assertIn(f"`{result['task_model']}` as the task model", recorded)
        self.assertIn("`openai/gpt-5.6-luna`", recorded)

        rows = {
            "Baseline": "baseline",
            "Labeled few-shot": "few_shot",
            "Majority voting (three samples)": "majority_voting",
            "GEPA": "gepa",
        }
        self.assertEqual(set(rows.values()), set(result["approaches"]))
        for label, key in rows.items():
            approach = result["approaches"][key]
            usage = approach["inference_usage"]
            with self.subTest(approach=key):
                self.assertIn(
                    f"| {label} "
                    f"| {approach['exact_field_accuracy_pct']:.2f}% "
                    f"| {approach['partial_credit_score_pct']:.2f}% "
                    f"| {usage['input_tokens']:,} "
                    f"| {usage['output_tokens']:,} |",
                    recorded,
                )

    def test_reader_runs_do_not_overwrite_the_recorded_run(self) -> None:
        notebook = load_json(NOTEBOOK_PATH)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("'invoice_extraction_benchmark.local.json'", code)
        self.assertNotIn("'invoice_extraction_benchmark.json'", code)
        ignored = (CHAPTER_DIR / "results" / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.local.json", ignored.splitlines())

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
