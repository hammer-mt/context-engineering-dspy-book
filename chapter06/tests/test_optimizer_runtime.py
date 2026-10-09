"""Offline checks for the shared code behind the Chapter 6 notebooks.

The notebooks import their dataset, metric, and scoring helpers from
``chapter06/optimizer_runtime.py``. These tests cover the frozen split, the
KNNFewShot listing on page 140, the balanced fine-tuning optimizer on
page 151, and the scoring helpers. They make no model calls.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import dspy
from dspy.clients.lm_local import LocalProvider

from chapter06.apple_finetune import (
    MacLocalProvider,
    make_model_spec,
    parse_model_spec,
)
from chapter06.optimizer_runtime import (
    AIDetector,
    BalancedBootstrapFinetune,
    SharedHistoryLM,
    accepted_trace_label_counts,
    evaluate,
    exact_match,
    finetune_training_config,
    hashed_ngram_embeddings,
    load_frozen_examples,
    published_result,
    score_summary,
)


def scoring_example(index: int, is_ai: bool) -> dspy.Example:
    return dspy.Example(
        text=f"passage {index}",
        is_ai=is_ai,
        pair_id=f"pair-{index}",
        example_id=f"example-{index}",
    ).with_inputs("text")


def trace(label: bool, *, score: float = 1.0) -> dict:
    return {"prediction": dspy.Prediction(is_ai=label), "score": score, "trace": []}


class FrozenDatasetTest(unittest.TestCase):
    def test_shared_split_is_balanced_and_pair_grouped(self) -> None:
        splits = load_frozen_examples()
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 160,
            "validation": 60,
            "test": 80,
        })
        for examples in splits.values():
            self.assertEqual(
                sum(bool(example.is_ai) for example in examples), len(examples) // 2
            )
            pairs: dict[str, int] = {}
            for example in examples:
                pairs[example.pair_id] = pairs.get(example.pair_id, 0) + 1
            self.assertEqual(set(pairs.values()), {2})

    def test_examples_carry_the_notes_used_by_the_gepa_listing(self) -> None:
        # Chapter 6, "GEPA" (page 145): the metric returns ``example.notes``.
        splits = load_frozen_examples()
        for examples in splits.values():
            for example in examples:
                self.assertTrue(example.notes.strip())
        self.assertEqual(splits["train"][0].inputs().keys(), ["text"])


class KNNFewShotListingTest(unittest.TestCase):
    def test_numpy_is_loaded_when_the_runtime_is_imported(self) -> None:
        # DSPy 3.3.0 registers NumPy lazily; cached embeddings can only be
        # read back once the real module is loaded.
        self.assertIs(type(sys.modules["numpy"]), types.ModuleType)

    def test_listing_builds_and_compiles_without_a_model_call(self) -> None:
        trainset = load_frozen_examples()["train"]
        # Run twice so the second pass reads the embeddings back from DSPy's cache.
        for _ in range(2):
            optimizer = dspy.teleprompt.KNNFewShot(
                k=4,
                metric=exact_match,
                trainset=trainset,
                vectorizer=dspy.Embedder(hashed_ngram_embeddings),
            )
            optimized_detector = optimizer.compile(AIDetector())
            self.assertIsInstance(optimized_detector, AIDetector)


class BalancedBootstrapFinetuneTest(unittest.TestCase):
    def test_published_finetune_results_keep_the_printed_trace_counts(self) -> None:
        # Chapter 6 prints "Accepted Traces: 77 human / 63 AI" for
        # BootstrapFinetune (page 152) and "77 human / 55 AI" for
        # BetterTogether (page 153).
        for optimizer, human, ai in (
            ("bootstrap-finetune", 77, 63),
            ("better-together", 77, 55),
        ):
            result = published_result(optimizer)
            with self.subTest(optimizer=optimizer):
                self.assertEqual(result["status"], "completed")
                self.assertEqual(
                    result["accepted_trace_labels"],
                    {"human": human, "ai": ai, "total": human + ai},
                )
                # Both classes clear the minimum the balanced fine-tuner requires.
                self.assertGreaterEqual(min(human, ai), 2)

    def test_counts_only_metric_accepted_traces(self) -> None:
        self.assertEqual(
            accepted_trace_label_counts(
                [trace(False), trace(False, score=0), trace(True), trace(True)]
            ),
            {"human": 1, "ai": 2, "total": 3},
        )

    def test_rejects_one_class_trace_set(self) -> None:
        finetuner = BalancedBootstrapFinetune(
            metric=lambda example, prediction: True,
            min_examples_per_class=2,
        )
        with self.assertRaisesRegex(
            ValueError, "human=17, ai=0; require at least 2 of each class"
        ):
            finetuner._prepare_finetune_data(
                [trace(False) for _ in range(17)],
                lm=object(),  # The balance check runs before DSPy formats data.
            )
        self.assertEqual(
            finetuner.accepted_label_counts,
            {"human": 17, "ai": 0, "total": 17},
        )


class NativeLocalProviderTest(unittest.TestCase):
    def test_mac_provider_is_a_thin_dspy_local_provider_subclass(self) -> None:
        self.assertIs(MacLocalProvider.__mro__[1], LocalProvider)
        model = make_model_spec("Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(model, "openai/local:Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(
            parse_model_spec(model), ("Qwen/Qwen2.5-0.5B-Instruct", None)
        )

    def test_training_kwargs_match_dspy_local_provider(self) -> None:
        kwargs = finetune_training_config(Path("chapter06-test-output"))
        self.assertTrue(kwargs["use_peft"])
        self.assertFalse(kwargs["bf16"])
        self.assertEqual(kwargs["num_train_epochs"], 10)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 4)
        self.assertEqual(kwargs["learning_rate"], 2e-4)
        self.assertEqual(kwargs["max_seq_length"], 768)
        self.assertNotIn("max_steps", kwargs)
        self.assertNotIn("lora_rank", kwargs)


class EvaluationIntegrityTest(unittest.TestCase):
    def test_parse_errors_are_retained_as_incorrect_predictions(self) -> None:
        class MalformedProgram(dspy.Module):
            def forward(self, **kwargs):
                raise ValueError("not a boolean")

        example = dspy.Example(
            text="example",
            is_ai=True,
            pair_id="pair-1",
            example_id="example-1",
        ).with_inputs("text")
        result = evaluate(MalformedProgram(), [example])

        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["predictions"][0]["status"], "parse_error")
        self.assertIsNone(result["predictions"][0]["predicted_is_ai"])

    def test_score_summary_reports_validation_then_locked_test(self) -> None:
        class AlwaysAI(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(is_ai=True)

        def example(index: int, is_ai: bool) -> dspy.Example:
            return dspy.Example(
                text="example",
                is_ai=is_ai,
                pair_id=f"pair-{index}",
                example_id=f"example-{index}",
            ).with_inputs("text")

        valset = [example(1, True), example(2, False)]
        testset = [example(3, True), example(4, True), example(5, True), example(6, False)]
        lines = score_summary(AlwaysAI(), valset, testset).splitlines()

        self.assertEqual(lines[0], "Validation Accuracy:   50.00% (1/2)")
        self.assertEqual(lines[1], "Locked Test Accuracy:  75.00% (3/4)")
        self.assertTrue(lines[2].startswith("Mean / p95 Latency:    "))

    def test_score_summary_stops_when_every_model_call_fails(self) -> None:
        calls: list[str] = []

        class Unreachable(dspy.Module):
            def forward(self, **kwargs):
                calls.append(kwargs["text"])
                raise ConnectionError("no API key")

        def example(index: int) -> dspy.Example:
            return dspy.Example(
                text=f"passage {index}",
                is_ai=True,
                pair_id=f"pair-{index}",
                example_id=f"example-{index}",
            ).with_inputs("text")

        valset = [example(1), example(2)]
        testset = [example(3)]
        with self.assertRaisesRegex(RuntimeError, "ConnectionError: no API key"):
            score_summary(Unreachable(), valset, testset)
        # The locked test is not touched after a failed validation pass.
        self.assertEqual(calls, ["passage 1", "passage 2"])

    def test_score_summary_stops_after_three_failed_calls(self) -> None:
        # A rejected key or a missing connection fails every call the same
        # way, so scoring stops at once instead of retrying the whole split.
        calls: list[str] = []

        class Unreachable(dspy.Module):
            def forward(self, **kwargs):
                calls.append(kwargs["text"])
                raise ConnectionError("connection refused")

        valset = [scoring_example(index, True) for index in range(1, 7)]
        testset = [scoring_example(7, True)]
        with self.assertRaisesRegex(
            RuntimeError,
            "The first 3 model calls failed.*ConnectionError: connection refused",
        ):
            score_summary(Unreachable(), valset, testset)
        self.assertEqual(calls, ["passage 1", "passage 2", "passage 3"])

    def test_score_summary_counts_failed_calls_as_wrong_and_reports_them(self) -> None:
        # An answer that cannot be read as a boolean is a wrong prediction; it
        # never stops the run, even when the first answers are all unreadable.
        class UnreadableAtFirst(dspy.Module):
            def forward(self, **kwargs):
                index = int(kwargs["text"].split()[-1])
                return dspy.Prediction(is_ai="maybe" if index <= 4 else True)

        valset = [scoring_example(index, True) for index in range(1, 7)]
        testset = [scoring_example(index, True) for index in range(7, 9)]
        lines = score_summary(UnreadableAtFirst(), valset, testset).splitlines()

        self.assertEqual(lines[0], "Validation Accuracy:   33.33% (2/6)")
        self.assertEqual(lines[1], "Locked Test Accuracy:  100.00% (2/2)")
        self.assertEqual(lines[3], "Failed Calls:          4 of 8 (counted as wrong)")
        self.assertEqual(len(lines), 4)

    def test_score_summary_stops_when_no_validation_prediction_can_be_scored(
        self,
    ) -> None:
        class NeverBoolean(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(is_ai="AI-generated")

        valset = [scoring_example(index, True) for index in range(1, 6)]
        with self.assertRaisesRegex(
            RuntimeError,
            "No validation prediction could be scored.*cannot interpret",
        ):
            score_summary(NeverBoolean(), valset, [scoring_example(6, True)])

    def test_score_summary_has_no_failed_calls_line_when_every_call_succeeds(
        self,
    ) -> None:
        class AlwaysAI(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(is_ai=True)

        lines = score_summary(
            AlwaysAI(), [scoring_example(1, True)], [scoring_example(2, False)]
        ).splitlines()
        self.assertEqual(len(lines), 3)
        self.assertNotIn("Failed Calls", "\n".join(lines))


class UsageAccountingTest(unittest.TestCase):
    def test_copies_of_the_accounting_lm_share_one_history(self) -> None:
        # MIPROv2's instruction proposer and SIMBA call ``LM.copy()`` to vary
        # the temperature; their calls must stay in the recorded usage.
        lm = SharedHistoryLM("openai/gpt-5.6-luna", cache=False)
        clone = lm.copy(rollout_id=1, temperature=1.0)

        self.assertIsNot(clone, lm)
        self.assertIs(clone.history, lm.history)
        self.assertEqual(clone.kwargs["temperature"], 1.0)
        self.assertNotEqual(lm.kwargs.get("temperature"), 1.0)
        self.assertEqual(clone.model, "openai/gpt-5.6-luna")

        clone.history.append({"cost": 0.25})
        self.assertEqual(lm.history, [{"cost": 0.25}])


if __name__ == "__main__":
    unittest.main()
