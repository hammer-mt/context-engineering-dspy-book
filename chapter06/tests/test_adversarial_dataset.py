from __future__ import annotations

import copy
import unittest

from chapter06.build_adversarial_dataset import (
    _rewrite_instruction,
    baseline_accuracy_by_split,
    dataset_digest,
    extract_documentation_passages,
    extract_html_passages,
    extract_passages,
    extract_short_passages,
    quality_rejection_reasons,
    split_pair_ids,
    stable_adversarial_splits,
    validate_pairs,
)


class ExtractPassagesTest(unittest.TestCase):
    def test_extracts_body_paragraphs_within_word_limits(self) -> None:
        body = " ".join(f"word{index}" for index in range(80))
        raw = (
            "Project header\n\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n\n"
            f"{body}\n\n"
            "Short paragraph.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n\n"
            "License footer"
        )

        self.assertEqual(extract_passages(raw, min_words=75, max_words=90), [body])

    def test_joins_wrapped_lines_but_not_separate_paragraphs(self) -> None:
        first = " ".join(["A deliberately formal sentence"] * 20)
        second = " ".join(["Another polished human paragraph"] * 20)
        raw = f"*** START OF TEST ***\n\n{first}\n\n{second}\n\n*** END OF TEST ***"

        passages = extract_passages(raw, min_words=60, max_words=100)

        self.assertEqual(passages, [first, second])

    def test_extracts_clean_wikipedia_paragraphs_without_citations(self) -> None:
        prose = " ".join(["A polished human-authored explanation connects several ideas."] * 8)
        html = (
            "<div><table><tr><td>navigation noise</td></tr></table>"
            f"<p>{prose}<sup class='reference'>[12]</sup></p>"
            "<p>Too short.</p></div>"
        )

        passages = extract_html_passages(html, min_words=40, max_words=100)

        self.assertEqual(passages, [prose])

    def test_extracts_complete_short_sentence_windows(self) -> None:
        first = " ".join(["Formal human prose"] * 10) + "."
        second = " ".join(["Another connected observation"] * 10) + "."
        raw = f"*** START OF TEST ***\n\n{first} {second}\n\n*** END OF TEST ***"

        passages = extract_short_passages(raw, min_words=25, max_words=40)

        self.assertEqual(passages, [first, second])

    def test_extracts_documentation_prose_and_drops_code_blocks(self) -> None:
        prose = " ".join(["This comprehensive framework supports several practical workflows."] * 8)
        markdown = (
            "# Heading\n\n"
            f"{prose} See [the guide](https://example.test/guide).\n\n"
            "```python\nprint('not prose')\n```\n\n"
            "- short list item\n"
        )

        passages = extract_documentation_passages(markdown, min_words=40, max_words=100)

        self.assertEqual(passages, [f"{prose} See the guide."])

    def test_rewrite_instruction_preserves_source_scale(self) -> None:
        source = " ".join(f"word{index}" for index in range(40))

        instruction = _rewrite_instruction(source, attempt=0)

        self.assertIn("34-46 words", instruction)
        self.assertNotIn("80-180 words", instruction)

    def test_quality_filter_rejects_fragments_and_residual_markup(self) -> None:
        fragment = "As mentioned above, this works in the usual way."
        markup = " ".join(["A complete technical explanation"] * 8) + " <api-reference>."
        complete = " ".join(["A complete technical explanation"] * 8) + "."

        self.assertIn("too_short", quality_rejection_reasons(fragment))
        self.assertIn("residual_markup", quality_rejection_reasons(markup))
        self.assertEqual(quality_rejection_reasons(complete), [])


class PairValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "pair_id": "pair-001",
                "example_id": "pair-001-human",
                "text": "A human passage with a distinctive and verified source.",
                "is_ai": False,
                "notes": "Exact public-domain excerpt.",
                "source_id": "source-1",
                "source_url": "https://example.test/source-1",
                "source_title": "Source One",
                "source_author": "Human Author",
                "license": "Public domain in the USA",
                "generation_model": "",
                "parent_example_id": "",
            },
            {
                "pair_id": "pair-001",
                "example_id": "pair-001-ai",
                "text": "An AI rewrite with substantially different wording and provenance.",
                "is_ai": True,
                "notes": "Generated rewrite.",
                "source_id": "source-1",
                "source_url": "https://example.test/source-1",
                "source_title": "Source One",
                "source_author": "Human Author",
                "license": "Public domain in the USA",
                "generation_model": "openai/test-model",
                "parent_example_id": "pair-001-human",
            },
        ]

    def test_accepts_one_human_and_one_ai_row_with_provenance(self) -> None:
        validate_pairs(self.rows)

    def test_rejects_orphaned_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two"):
            validate_pairs(self.rows[:1])

    def test_rejects_duplicate_text_hashes(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[1]["text"] = rows[0]["text"]

        with self.assertRaisesRegex(ValueError, "duplicate text"):
            validate_pairs(rows)

    def test_rejects_ai_row_without_generation_provenance(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[1]["generation_model"] = ""

        with self.assertRaisesRegex(ValueError, "generation_model"):
            validate_pairs(rows)

    def test_rejects_human_row_without_source_license(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["license"] = ""

        with self.assertRaisesRegex(ValueError, "license"):
            validate_pairs(rows)

    def test_digest_is_stable_across_row_order_and_changes_with_text(self) -> None:
        forward = dataset_digest(self.rows)
        reverse = dataset_digest(list(reversed(self.rows)))
        changed_rows = copy.deepcopy(self.rows)
        changed_rows[0]["text"] += " Changed."

        self.assertEqual(forward, reverse)
        self.assertNotEqual(forward, dataset_digest(changed_rows))


class GroupSplitTest(unittest.TestCase):
    def test_split_is_deterministic_complete_and_disjoint(self) -> None:
        pair_ids = [f"pair-{index:03d}" for index in range(100)]

        split = split_pair_ids(pair_ids, seed=42)

        self.assertEqual(split, split_pair_ids(list(reversed(pair_ids)), seed=42))
        self.assertEqual(len(split["train"]), 50)
        self.assertEqual(len(split["validation"]), 25)
        self.assertEqual(len(split["test"]), 25)
        self.assertEqual(set().union(*map(set, split.values())), set(pair_ids))
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["validation"]) & set(split["test"]))

    def test_stable_split_uses_worst_case_repeat_accuracy_for_test(self) -> None:
        rows = []
        predictions = []
        for index in range(8):
            pair_id = f"pair-{index}"
            for label in ("human", "ai"):
                example_id = f"{pair_id}-{label}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "example_id": example_id,
                        "source_id": f"source-{index % 4}",
                    }
                )
                for repeat in range(2):
                    predictions.append(
                        {
                            "repeat": repeat,
                            "pair_id": pair_id,
                            "example_id": example_id,
                            "correct": index >= 2,
                            "status": "completed",
                        }
                    )

        splits, metadata = stable_adversarial_splits(rows, predictions, repeats=2, seed=42)

        self.assertEqual(set(splits["test"]), {"pair-0", "pair-1"})
        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertTrue(all(item["worst_repeat_correct"] == 0 for item in metadata))

    def test_rejects_duplicate_pair_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            split_pair_ids(["pair-001", "pair-001"], seed=42)

    def test_reports_baseline_accuracy_for_each_pair_group(self) -> None:
        selected = [
            {"pair_id": "pair-001", "pair_correct": 0},
            {"pair_id": "pair-002", "pair_correct": 1},
            {"pair_id": "pair-003", "pair_correct": 2},
        ]
        splits = {"train": ["pair-001"], "validation": ["pair-002"], "test": ["pair-003"]}

        metrics = baseline_accuracy_by_split(selected, splits)

        self.assertEqual(metrics["train"]["accuracy"], 0.0)
        self.assertEqual(metrics["validation"]["accuracy"], 50.0)
        self.assertEqual(metrics["test"]["accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
