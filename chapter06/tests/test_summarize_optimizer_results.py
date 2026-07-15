from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chapter06.summarize_optimizer_results as summarize_optimizer_results

from chapter06.summarize_optimizer_results import (
    FROZEN_PROGRAMS_DIR,
    FROZEN_PROMPTS_DIR,
    REPO_ROOT,
    SUMMARY_CSV,
    SUMMARY_JSON,
    SUMMARY_MARKDOWN,
    _latest_matching_run,
    _seconds,
)


class FormattingTest(unittest.TestCase):
    def test_formats_seconds_and_minutes(self) -> None:
        self.assertEqual(_seconds(None), "—")
        self.assertEqual(_seconds(12.34), "12.3s")
        self.assertEqual(_seconds(90), "1.5min")

    def test_latest_run_requires_completed_bounded_reload_prediction_check(
        self,
    ) -> None:
        dataset_manifest = {"dataset_sha256": "frozen"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer_root = root / "full" / "quickstart"
            for run_id, all_equal in (("001", True), ("002", False)):
                run_dir = optimizer_root / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "dataset_manifest": dataset_manifest,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "reload_prompt_parity": True,
                            "reload_prediction_parity": {
                                "checked": 3,
                                "completed": 3,
                                "matching": 3 if all_equal else 2,
                                "all_equal": all_equal,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.object(summarize_optimizer_results, "RESULTS_ROOT", root):
                selected = _latest_matching_run("quickstart", dataset_manifest)

        self.assertEqual(selected.name, "002")


class FrozenSummaryTest(unittest.TestCase):
    def test_checked_in_summary_and_canonical_artifacts_agree(self) -> None:
        if not SUMMARY_JSON.exists():
            self.skipTest("benchmark summary is generated after full optimizer runs")

        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        completed = [row for row in summary["rows"] if row["status"] == "completed"]

        self.assertTrue(completed)
        self.assertEqual(summary["schema_version"], 2)
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
            self.assertTrue(row["reload_prompt_parity"])
            parity = row["reload_prediction_parity"]
            self.assertGreaterEqual(parity["checked"], 3)
            self.assertEqual(parity["completed"], parity["checked"])
            self.assertGreaterEqual(parity["matching"], 0)
            self.assertLessEqual(parity["matching"], parity["checked"])
            self.assertEqual(
                parity["all_equal"], parity["matching"] == parity["checked"]
            )


if __name__ == "__main__":
    unittest.main()
