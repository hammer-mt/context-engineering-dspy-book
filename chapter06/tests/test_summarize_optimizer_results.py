from __future__ import annotations

import csv
import json
import unittest
from chapter06.summarize_optimizer_results import (
    FROZEN_PROGRAMS_DIR,
    FROZEN_PROMPTS_DIR,
    REPO_ROOT,
    SUMMARY_CSV,
    SUMMARY_JSON,
    SUMMARY_MARKDOWN,
    _seconds,
)


class FormattingTest(unittest.TestCase):
    def test_formats_seconds_and_minutes(self) -> None:
        self.assertEqual(_seconds(None), "—")
        self.assertEqual(_seconds(12.34), "12.3s")
        self.assertEqual(_seconds(90), "1.5min")


class FrozenSummaryTest(unittest.TestCase):
    def test_checked_in_summary_and_canonical_artifacts_agree(self) -> None:
        if not SUMMARY_JSON.exists():
            self.skipTest("benchmark summary is generated after full optimizer runs")

        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        completed = [row for row in summary["rows"] if row["status"] == "completed"]

        self.assertTrue(completed)
        gepa = next(row for row in completed if row["optimizer"] == "gepa")
        self.assertEqual(gepa["relative_uplift_vs_reference_percent"], 125.0)
        for optimizer in ("bootstrap-finetune", "better-together"):
            row = next(item for item in completed if item["optimizer"] == optimizer)
            self.assertEqual(row["task_model"], "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(row["finetune_teacher_mode"], "local")
            self.assertFalse(row["reference_comparable"])
            self.assertIsNone(row["uplift_vs_reference_points"])
            self.assertEqual(row["run_uplift_points"], 0.0)
            run_dir = REPO_ROOT / row["run_dir"]
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["reload_prompt_parity"])
            self.assertTrue(metrics["reload_model_parity"])
            self.assertTrue(metrics["reload_prediction_parity"])
            training_summary_path = next(
                (run_dir / "training").glob("*/training_summary.json")
            )
            training_summary = json.loads(
                training_summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(training_summary["requested_device"], "mps")
            self.assertEqual(training_summary["trainer_device"], "mps")
            if optimizer == "better-together":
                strategies = {
                    candidate["strategy"]
                    for candidate in metrics["optimizer_state"]["candidate_programs"]
                }
                self.assertIn("p -> w", strategies)
                self.assertFalse(
                    metrics["optimizer_state"]["flag_compilation_error_occurred"]
                )
        self.assertTrue(SUMMARY_MARKDOWN.exists())
        self.assertTrue(SUMMARY_CSV.exists())
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), len(summary["rows"]))

        for row in completed:
            optimizer = row["optimizer"]
            self.assertTrue((FROZEN_PROGRAMS_DIR / f"{optimizer}.json").exists())
            self.assertTrue((FROZEN_PROMPTS_DIR / f"{optimizer}.json").exists())
            run_dir = REPO_ROOT / row["run_dir"]
            for artifact_path in row["artifacts"].values():
                self.assertTrue((REPO_ROOT / artifact_path).exists())
            predictions = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            optimized = [item for item in predictions if item["phase"] == "optimized"]
            scored = optimized or [
                item for item in predictions if item["phase"] == "baseline"
            ]
            score = 100 * sum(item["correct"] for item in scored) / len(scored)
            self.assertAlmostEqual(score, row["accuracy"])


if __name__ == "__main__":
    unittest.main()
