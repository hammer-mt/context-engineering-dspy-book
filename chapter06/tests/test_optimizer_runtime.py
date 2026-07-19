from __future__ import annotations

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
    BalancedBootstrapFinetune,
    _finetune_kwargs,
    accepted_trace_label_counts,
    evaluate,
    load_frozen_examples,
    published_result,
)


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


class BalancedBootstrapFinetuneTest(unittest.TestCase):
    def test_completed_expanded_finetune_keeps_balanced_trace_evidence(self) -> None:
        result = published_result("bootstrap-finetune")
        if result.get("status") != "completed":
            self.assertIn(result["status"], {"pending", "blocked", "failed"})
            return
        counts = result["accepted_trace_labels"]
        self.assertGreaterEqual(counts["human"], 2)
        self.assertGreaterEqual(counts["ai"], 2)
        self.assertEqual(counts["total"], counts["human"] + counts["ai"])

    def test_counts_only_metric_accepted_traces(self) -> None:
        self.assertEqual(
            accepted_trace_label_counts(
                [trace(False), trace(False, score=0), trace(True), trace(True)]
            ),
            {"human": 1, "ai": 2, "total": 3},
        )

    def test_rejects_the_previous_negative_only_trace_set(self) -> None:
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
        kwargs = _finetune_kwargs(Path("/tmp/chapter06-test"))
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


if __name__ == "__main__":
    unittest.main()
