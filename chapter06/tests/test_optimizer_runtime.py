from __future__ import annotations

import unittest

import dspy

from chapter06.optimizer_runtime import (
    BalancedBootstrapFinetune,
    accepted_trace_label_counts,
    load_frozen_examples,
    published_result,
)


def trace(label: bool, *, score: float = 1.0) -> dict:
    return {"prediction": dspy.Prediction(is_ai=label), "score": score, "trace": []}


class FrozenDatasetTest(unittest.TestCase):
    def test_shared_split_is_balanced_and_pair_grouped(self) -> None:
        splits = load_frozen_examples()
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 36,
            "validation": 18,
            "test": 20,
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
    def test_published_rerun_keeps_balanced_trace_evidence(self) -> None:
        result = published_result("bootstrap-finetune")
        self.assertEqual(result["final_accuracy"], 35.0)
        self.assertEqual(result["correct"], 7)
        self.assertEqual(
            result["accepted_trace_labels"],
            {"human": 17, "ai": 16, "total": 33},
        )

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


if __name__ == "__main__":
    unittest.main()
