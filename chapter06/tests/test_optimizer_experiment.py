from __future__ import annotations

import json
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chapter06.optimizer_experiment as optimizer_experiment
from chapter06.experiment_runtime import BudgetExceeded, BudgetLedger

from chapter06.optimizer_experiment import (
    RunProfile,
    LMHistoryTracker,
    feedback_metric,
    boolean_majority,
    extract_program_prompts,
    hashed_ngram_embeddings,
    load_frozen_examples,
    profile_for,
)


class _Signature:
    instructions = "Use concrete stylistic evidence."


class _Predictor:
    signature = _Signature()
    demos = [{"text": "A sample", "is_ai": False}]


class _Program:
    def named_predictors(self):
        return [("detect", _Predictor())]


class _FakeLM:
    def __init__(self):
        self.history = []


class OptimizerExperimentTest(unittest.TestCase):
    def test_extracts_final_instruction_and_demos(self) -> None:
        prompts = extract_program_prompts(_Program())

        self.assertEqual(prompts["detect"]["instructions"], "Use concrete stylistic evidence.")
        self.assertEqual(prompts["detect"]["demos"][0]["is_ai"], False)

    def test_profiles_keep_smoke_small_and_full_unbounded(self) -> None:
        smoke = profile_for("smoke")
        full = profile_for("full")

        self.assertIsInstance(smoke, RunProfile)
        self.assertLess(smoke.train_limit, 10)
        self.assertEqual(full.train_limit, 0)
        self.assertGreater(full.gepa_max_full_evals, smoke.gepa_max_full_evals)

    def test_hashed_ngram_embeddings_are_deterministic_and_normalized(self) -> None:
        first = hashed_ngram_embeddings(["alpha beta beta", "gamma delta"], dimensions=32)
        second = hashed_ngram_embeddings(["alpha beta beta", "gamma delta"], dimensions=32)

        self.assertEqual(first.shape, (2, 32))
        self.assertTrue((first == second).all())
        self.assertAlmostEqual(float((first[0] ** 2).sum()), 1.0, places=6)

    def test_tracker_collects_history_from_copied_lm_instances(self) -> None:
        tracker = LMHistoryTracker(reflection_model="openai/reflection")
        copied_lm = _FakeLM()
        tracker.on_lm_start("call-1", copied_lm, {})
        copied_lm.history.append({"model": "openai/task", "cost": 0.01, "usage": {}})
        tracker.on_lm_end("call-1", ["done"], None)

        self.assertEqual(len(tracker.history), 1)
        self.assertEqual(tracker.history[0]["experiment_model_role"], "task")
        self.assertEqual(tracker.history[0]["experiment_call_id"], "call-1")

    def test_tracker_refuses_a_new_call_that_would_cross_the_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", ceiling_usd=0.05)
            tracker = LMHistoryTracker(reflection_model="openai/reflection", ledger=ledger)

            with self.assertRaises(BudgetExceeded):
                tracker.on_lm_start("call-1", _FakeLM(), {})

            self.assertEqual(tracker.snapshot(), 0)

    def test_gepa_feedback_metric_accepts_required_five_arguments(self) -> None:
        inspect.signature(feedback_metric).bind(None, None, None, None, None)

    def test_boolean_majority_handles_typed_boolean_outputs(self) -> None:
        import dspy

        result = boolean_majority(
            [dspy.Prediction(is_ai=True), dspy.Prediction(is_ai=False), dspy.Prediction(is_ai=True)]
        )

        self.assertIs(result.is_ai, True)

    def test_loads_pair_grouped_frozen_examples(self) -> None:
        fields = (
            "pair_id,example_id,text,is_ai,notes,source_id,source_url,source_title,"
            "source_author,license,generation_model,parent_example_id\n"
        )
        rows = [
            "p1,p1-human,Human one,False,n,s,u,t,a,l,,\n",
            "p1,p1-ai,AI one,True,n,s,u,t,a,l,m,p1-human\n",
            "p2,p2-human,Human two,False,n,s,u,t,a,l,,\n",
            "p2,p2-ai,AI two,True,n,s,u,t,a,l,m,p2-human\n",
            "p3,p3-human,Human three,False,n,s,u,t,a,l,,\n",
            "p3,p3-ai,AI three,True,n,s,u,t,a,l,m,p3-human\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            split_path = root / "splits.json"
            data_path.write_text(fields + "".join(rows), encoding="utf-8")
            split_path.write_text(
                json.dumps({"splits": {"train": ["p1"], "validation": ["p2"], "test": ["p3"]}}),
                encoding="utf-8",
            )

            loaded = load_frozen_examples(data_path=data_path, split_path=split_path)

        self.assertEqual([example.pair_id for example in loaded["train"]], ["p1", "p1"])
        self.assertEqual([example.pair_id for example in loaded["test"]], ["p3", "p3"])
        self.assertIs(loaded["test"][1].is_ai, True)

    def test_missing_api_key_leaves_a_terminal_artifact_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = BudgetLedger(root / "ledger.json")
            with patch.object(optimizer_experiment, "RESULTS_ROOT", root / "runs"), patch.object(
                optimizer_experiment, "BudgetLedger", return_value=ledger
            ), patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                with self.assertRaisesRegex(EnvironmentError, "OPENAI_API_KEY"):
                    optimizer_experiment.run_experiment("quickstart", profile_name="smoke")

            manifests = list((root / "runs").glob("smoke/quickstart/*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("complete_output", manifest["artifacts"])
            self.assertNotIn("program", manifest["artifacts"])
            self.assertNotIn("prompts", manifest["artifacts"])

    def test_budget_preflight_uses_the_full_profile_projection(self) -> None:
        class RejectingLedger:
            projected: float | None = None

            def assert_can_spend(self, projected_cost_usd: float) -> None:
                self.projected = projected_cost_usd
                raise BudgetExceeded("guarded ceiling")

            def record(self, **_kwargs) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = RejectingLedger()
            with patch.object(optimizer_experiment, "RESULTS_ROOT", root / "runs"), patch.object(
                optimizer_experiment, "BudgetLedger", return_value=ledger
            ), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
                with self.assertRaises(BudgetExceeded):
                    optimizer_experiment.run_experiment("gepa", profile_name="full")

            self.assertEqual(ledger.projected, optimizer_experiment.profile_for("full").projected_stage_cost_usd)
            manifests = list((root / "runs").glob("full/gepa/*/manifest.json"))
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "budget_stopped")


if __name__ == "__main__":
    unittest.main()
