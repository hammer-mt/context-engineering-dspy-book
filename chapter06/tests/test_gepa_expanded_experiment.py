from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dspy

from chapter06.experiments.gepa_expanded.analysis import (
    majority_vote,
    mcnemar_exact,
    paired_pair_bootstrap,
)
from chapter06.experiments.gepa_expanded.guardrails import atomic_write_json, write_jsonl

from chapter06.experiments.gepa_expanded.dataset import (
    DATA_PATH,
    LOCK_PATH,
    ORIGINAL_DATA_PATH,
    RESULTS_DIR,
    SPLIT_PATH,
    assert_lock_unchanged,
    assert_original_rows_preserved,
    read_csv,
    source_family,
    validate_pairs,
)
from chapter06.experiments.gepa_expanded.run_experiment import (
    _save_program_artifacts,
    evaluate_repeated,
    finalize_result,
    materialize_gepa_candidates,
    run_gepa,
)
from chapter06.experiments.gepa_expanded.runtime import AIDetector


class ExpandedDatasetExperimentTest(unittest.TestCase):
    def test_expanded_dataset_preserves_original_and_complete_pairs(self) -> None:
        rows = read_csv(DATA_PATH)
        summary = validate_pairs(rows)
        self.assertEqual(len(rows), 300)
        self.assertEqual(summary["pair_count"], 150)
        self.assertEqual(summary["complete_pair_fraction"], 1)
        self.assertEqual(summary["provenance_complete_fraction"], 1)
        self.assertEqual(assert_original_rows_preserved(rows), 74)

    def test_expanded_split_is_grouped_disjoint_and_locked(self) -> None:
        rows = read_csv(DATA_PATH)
        manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        split_sets = {
            name: set(pair_ids) for name, pair_ids in manifest["splits"].items()
        }
        self.assertEqual(
            {name: len(ids) for name, ids in split_sets.items()},
            {"train": 80, "validation": 30, "test": 40},
        )
        self.assertFalse(split_sets["train"] & split_sets["validation"])
        self.assertFalse(split_sets["train"] & split_sets["test"])
        self.assertFalse(split_sets["validation"] & split_sets["test"])
        self.assertEqual(
            set().union(*split_sets.values()), {row["pair_id"] for row in rows}
        )
        lock = assert_lock_unchanged()
        self.assertTrue(lock["locked_before_gepa"])
        self.assertEqual(lock["gepa_test_leakage_count"], 0)

    def test_sources_and_baseline_gate(self) -> None:
        rows = read_csv(DATA_PATH)
        self.assertEqual(len({row["source_id"] for row in rows}), 21)
        self.assertEqual(
            {source_family(row) for row in rows},
            {"technical_documentation", "wikipedia"},
        )
        baseline = json.loads(
            (RESULTS_DIR / "baseline_confirmation" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(baseline["repeat_count"], 3)
        self.assertLessEqual(baseline["baseline_test_accuracy_pct"], 50)
        self.assertLessEqual(baseline["majority_correct"], 40)
        prediction_count = len(
            (RESULTS_DIR / "baseline_confirmation" / "majority_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(prediction_count, 80)

    def test_budget_guard_preserves_safety_reserve(self) -> None:
        ledger = json.loads((RESULTS_DIR / "budget_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["ceiling_usd"], 95)
        self.assertLess(ledger["total_cost_usd"], ledger["ceiling_usd"])
        self.assertGreater(ledger["remaining_usd"], 5)

    def test_majority_and_paired_statistics_are_pair_aware(self) -> None:
        expected = {
            "a-human": ("a", False),
            "a-ai": ("a", True),
            "b-human": ("b", False),
            "b-ai": ("b", True),
        }

        def run(predictions: dict[str, bool]) -> list[dict[str, object]]:
            return [
                {
                    "pair_id": expected[example_id][0],
                    "example_id": example_id,
                    "expected_is_ai": expected[example_id][1],
                    "predicted_is_ai": prediction,
                    "correct": int(prediction == expected[example_id][1]),
                }
                for example_id, prediction in predictions.items()
            ]

        optimized = majority_vote(
            [
                run({"a-human": False, "a-ai": True, "b-human": False, "b-ai": True}),
                run({"a-human": False, "a-ai": True, "b-human": True, "b-ai": True}),
                run({"a-human": True, "a-ai": True, "b-human": False, "b-ai": True}),
            ]
        )
        self.assertTrue(all(row["correct"] for row in optimized))
        baseline = run(
            {"a-human": True, "a-ai": True, "b-human": True, "b-ai": True}
        )
        mcnemar = mcnemar_exact(baseline, optimized)
        self.assertEqual(mcnemar["optimized_only_correct"], 2)
        bootstrap = paired_pair_bootstrap(baseline, optimized, samples=100, seed=42)
        self.assertEqual(bootstrap["pair_count"], 2)
        self.assertEqual(bootstrap["observed_uplift_pct_points"], 50)

    def test_locked_test_evaluation_requires_validation_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "validation summary"):
                evaluate_repeated(
                    program_path=root / "program.json",
                    split_name="test",
                    output_dir=root / "test",
                    validation_summary_path=None,
                )

    def test_saved_gepa_candidates_can_be_materialized_without_api_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            optimizer_results = root / "optimizer_results.json"
            atomic_write_json(
                optimizer_results,
                {
                    "candidates": [
                        {"detect.predict": "Original instruction"},
                        {"detect.predict": "Learned candidate instruction"},
                    ],
                    "val_aggregate_scores": [0.5, 0.8],
                },
            )
            summary = materialize_gepa_candidates(
                optimizer_results_path=optimizer_results,
                output_dir=root / "pool",
                candidate_indexes=[1],
            )
            candidate = summary["candidates"][0]
            self.assertEqual(candidate["candidate_index"], 1)
            self.assertEqual(candidate["gepa_internal_validation_accuracy_pct"], 80)
            prompt = json.loads(
                (root / "pool" / "candidate-1" / "learned_prompt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                prompt["predictors"]["detect.predict"]["instructions"],
                "Learned candidate instruction",
            )

    def test_gepa_smoke_runner_is_test_blind_and_persists_program(self) -> None:
        class FakeLedger:
            def assert_can_spend(self, projected_cost_usd: float) -> None:
                self.projected = projected_cost_usd

            def record(self, **kwargs: object) -> None:
                self.recorded = kwargs

        class FakeGEPA:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def compile(
                self,
                student: dspy.Module,
                *,
                trainset: list[dspy.Example],
                valset: list[dspy.Example],
            ) -> dspy.Module:
                self.test_ids = {
                    str(example.example_id) for example in load_test_examples()
                }
                compiled_ids = {
                    str(example.example_id) for example in trainset + valset
                }
                if self.test_ids & compiled_ids:
                    raise AssertionError("locked test leaked into fake GEPA compile")
                for _ in valset:
                    dspy.settings.lm.history.append(
                        {
                            "model": "openai/gpt-5.6-luna",
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                            "cost": 0,
                        }
                    )
                return student

        def load_test_examples() -> list[dspy.Example]:
            from chapter06.experiments.gepa_expanded.run_experiment import load_examples

            return load_examples()["test"]

        def fake_lm(model: str, **kwargs: object) -> dspy.LM:
            return dspy.LM(model, cache=False, max_tokens=500)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "smoke"
            with (
                patch(
                    "chapter06.experiments.gepa_expanded.run_experiment.load_project_env"
                ),
                patch(
                    "chapter06.experiments.gepa_expanded.run_experiment.BudgetLedger",
                    FakeLedger,
                ),
                patch(
                    "chapter06.experiments.gepa_expanded.run_experiment.make_lm",
                    side_effect=fake_lm,
                ),
                patch(
                    "chapter06.experiments.gepa_expanded.run_experiment.dspy.GEPA",
                    FakeGEPA,
                ),
            ):
                summary = run_gepa(
                    mode="smoke",
                    output_dir=output_dir,
                    max_full_evals=1,
                    stage_cap_usd=2,
                    threads=1,
                )
            self.assertEqual(summary["test_rows_seen"], 0)
            self.assertTrue((output_dir / "optimized_program.json").exists())
            self.assertTrue((output_dir / "learned_prompt.json").exists())

    def test_finalizer_emits_copy_ready_cost_latency_table(self) -> None:
        baseline_rows = [
            json.loads(line)
            for line in (RESULTS_DIR / "baseline_confirmation" / "majority_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        optimized_rows = [
            {
                **row,
                "predicted_is_ai": row["expected_is_ai"],
                "correct": 1,
                "positive_votes": 3 if row["expected_is_ai"] else 0,
            }
            for row in baseline_rows
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            test_dir = root / "test"
            final_dir = root / "final"
            _save_program_artifacts(AIDetector(), candidate)
            atomic_write_json(
                candidate / "summary.json",
                {
                    "cost": {
                        "cost_usd": 1.25,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    "optimize_time_seconds": 90,
                },
            )
            test_dir.mkdir()
            write_jsonl(test_dir / "majority_predictions.jsonl", optimized_rows)
            atomic_write_json(
                test_dir / "summary.json",
                {
                    "majority_accuracy_pct": 100,
                    "majority_correct": 80,
                    "mean_latency_seconds": 1.5,
                    "p95_latency_seconds": 2.5,
                    "cost_usd": 0.3,
                    "reference_strategy": "test fixture",
                },
            )
            summary = finalize_result(
                candidate_dir=candidate,
                test_evaluation_dir=test_dir,
                final_dir=final_dir,
            )
            self.assertEqual(summary["optimized_test_accuracy_pct"], 100)
            self.assertEqual(summary["optimize_total_tokens"], 150)
            table = (final_dir / "chapter_table.csv").read_text(encoding="utf-8")
            self.assertIn("optimization_cost_usd", table)
            self.assertIn("mean_inference_latency_seconds", table)


if __name__ == "__main__":
    unittest.main()
