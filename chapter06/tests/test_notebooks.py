"""Offline checks for the Chapter 6 optimizer notebooks and their saved results.

Chapter 6, "Deep Dive into Prompt Optimizers", prints one listing and one
result block per optimizer. These tests check that the notebooks contain the
listings, that the saved results reproduce the printed numbers, and that the
notebooks run in their default mode without calling a model.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from decimal import ROUND_HALF_UP, Decimal
from unittest.mock import patch

import dspy

from chapter06.build_expanded_comparison import _build_run_ledger, _run_row
from chapter06.build_optimizer_notebooks import (
    CHAPTER_DIR,
    LISTINGS,
    NOTEBOOKS,
    PRINTED_OUTPUT,
    make_notebook,
)
from chapter06.notebook_support import optimizer_row, verify_prompt_artifact
from chapter06.optimizer_runtime import (
    AIDetector,
    _compile_prompt_optimizer,
    format_result,
    published_result,
)
from chapter06.tests.book_result_blocks import PRINTED_RESULT_BLOCKS
from chapter06.validate_notebooks import (
    _mode_source,
    contains_listing,
    undefined_names,
    validate_notebook,
)


# Listings that make no model call and therefore run in the default mode.
FREE_LISTINGS = {"load-dataset", "labeled-few-shot", "knn-few-shot"}


def code_cells(filename: str) -> list[str]:
    notebook = json.loads((CHAPTER_DIR / filename).read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def listing_items(spec: dict) -> list[dict]:
    return [*spec["listings"], *spec.get("extras", [])]


def all_cells(filename: str) -> list[tuple[str, str]]:
    notebook = json.loads((CHAPTER_DIR / filename).read_text(encoding="utf-8"))
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]


# Output printed in the chapter that each notebook quotes, in notebook order.
QUOTED_OUTPUT = {
    "quickstart-ai-detector.ipynb": ["split-summary", "baseline-validation", "baseline-test"],
    "labeled-few-shot.ipynb": ["labeled-few-shot"],
    "bootstrap-few-shot.ipynb": ["bootstrap-few-shot"],
    "bootstrap-random-search.ipynb": ["bootstrap-random-search"],
    "knn-few-shot.ipynb": ["knn-few-shot"],
    "copro.ipynb": ["copro", "copro-instruction"],
    "miprov2.ipynb": ["miprov2"],
    "gepa.ipynb": ["gepa", "gepa-instruction"],
    "simba.ipynb": ["simba"],
    "ensemble.ipynb": ["ensemble"],
    "bootstrap-finetune.ipynb": ["bootstrap-finetune"],
    "better-together.ipynb": ["better-together"],
}


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def squeeze(line: str) -> str:
    return " ".join(line.split())


def matches_printed_line(printed: str, shown: str) -> bool:
    """True if a result line shown by the notebooks agrees with the printed line.

    The notebooks may add a detail in parentheses after the printed text, such
    as the "(38/60)" count after a validation percentage. They may also show a
    figure with more decimals than the book, which rounds the COPRO and
    MIPROv2 percentages to one decimal.
    """
    printed, shown = squeeze(printed), squeeze(shown)
    if shown == printed or shown.startswith(printed + " ("):
        return True
    if NUMBER.sub("#", printed) != NUMBER.sub("#", shown):
        return False
    for book_number, repo_number in zip(NUMBER.findall(printed), NUMBER.findall(shown)):
        decimals = len(book_number.partition(".")[2])
        rounded = Decimal(repo_number.replace(",", "")).quantize(
            Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP
        )
        if rounded != Decimal(book_number.replace(",", "")):
            return False
    return True


class PublishedResultTest(unittest.TestCase):
    def test_copro_miprov2_and_simba_results_match_the_book(self) -> None:
        copro = optimizer_row("copro")
        self.assertEqual(copro["optimized_validation_accuracy_pct"], 55.0)
        self.assertEqual(copro["locked_test_accuracy_pct"], 56.25)
        self.assertEqual(copro["optimization_cost_usd"], 0.81558)
        self.assertAlmostEqual(
            copro["optimization_time_seconds"], 1155.7725224589813
        )
        self.assertEqual(
            copro["result_artifact"],
            "chapter06/results/expanded_notebooks/copro/full/result.json",
        )

        # The instruction printed in Chapter 6, "COPRO" (page 143).
        expected_copro_instruction = (
            "Analyze the supplied passage for linguistic, stylistic, and structural "
            "signals associated with AI-generated versus human-written text. Consider "
            "factors such as formulaic phrasing, repetition, coherence patterns, "
            "specificity, natural variation, and errors, while avoiding reliance on "
            "topic or unsupported assumptions. Make a forced-choice classification "
            "based only on the passage. Output exactly one label: `AI-generated` or "
            "`Human-written`."
        )
        copro_prompt = json.loads(
            (CHAPTER_DIR.parent / copro["prompt_artifact"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            copro_prompt["predictors"]["detect.predict"]["instructions"],
            expected_copro_instruction,
        )

        miprov2 = optimizer_row("miprov2")
        self.assertEqual(miprov2["optimized_validation_accuracy_pct"], 80.0)
        self.assertEqual(miprov2["locked_test_accuracy_pct"], 66.25)
        self.assertEqual(miprov2["optimization_cost_usd"], 0.854762)
        self.assertAlmostEqual(miprov2["evaluation_cost_usd"], 0.213767)
        self.assertEqual(
            miprov2["result_artifact"],
            "chapter06/results/expanded_notebooks/miprov2/full/result.json",
        )

        simba = optimizer_row("simba")
        simba_prompt = json.loads(
            (CHAPTER_DIR.parent / simba["prompt_artifact"]).read_text(encoding="utf-8")
        )["predictors"]["detect.predict"]
        self.assertEqual(
            simba_prompt["instructions"],
            "Decide whether the supplied passage was generated by an AI.",
        )
        self.assertEqual(simba_prompt["demos"], [])

        ledger = json.loads(
            (
                CHAPTER_DIR
                / "results"
                / "expanded_notebooks"
                / "run_ledger.json"
            ).read_text(encoding="utf-8")
        )
        for optimizer, row in (("copro", copro), ("miprov2", miprov2)):
            published_runs = [
                run
                for run in ledger["runs"]
                if run.get("optimizer") == optimizer
                and run.get("status") == "completed"
                and run.get("published_in_chapter")
            ]
            with self.subTest(optimizer=optimizer):
                self.assertEqual(len(published_runs), 1)
                self.assertEqual(published_runs[0]["run_name"], "full")
                self.assertEqual(
                    published_runs[0]["result_artifact"], row["result_artifact"]
                )
                self.assertEqual(
                    published_runs[0]["optimization_cost_status"], "recorded"
                )
                self.assertEqual(
                    published_runs[0]["evaluation_cost_status"], "recorded"
                )

        self.assertFalse((CHAPTER_DIR / "results" / "final_prompts").exists())
        self.assertFalse(
            (CHAPTER_DIR / "results" / "benchmark_summary.json").exists()
        )

    def test_comparison_builder_publishes_the_full_copro_run(self) -> None:
        copro = _run_row(
            "copro",
            canonical_baseline=53.75,
            baseline_predictions=None,
        )
        self.assertEqual(copro["run_name"], "full")
        self.assertEqual(copro["locked_test_accuracy_pct"], 56.25)
        self.assertEqual(copro["optimization_cost_status"], "recorded")
        self.assertEqual(copro["evaluation_cost_status"], "recorded")

        copro_runs = [
            run
            for run in _build_run_ledger()["runs"]
            if run.get("optimizer") == "copro" and run.get("status") == "completed"
        ]
        published_runs = [run for run in copro_runs if run["published_in_chapter"]]
        self.assertEqual([run["run_name"] for run in published_runs], ["full"])
        self.assertEqual(published_runs[0]["optimization_cost_status"], "recorded")

    def test_frozen_program_prompt_state_matches_extracted_prompt(self) -> None:
        for spec in NOTEBOOKS.values():
            optimizer = spec["optimizer"]
            row = optimizer_row(optimizer)
            check = verify_prompt_artifact(optimizer)
            with self.subTest(optimizer=optimizer):
                self.assertEqual(row["status"], "completed")
                self.assertTrue(check["checked"])
                self.assertTrue(check["prompt_state_equal"])

    def test_result_blocks_use_the_layout_printed_in_the_book(self) -> None:
        # Chapter 6, "LabeledFewShot" (page 137).
        labeled = format_result(published_result("labeled-few-shot")).splitlines()
        self.assertEqual(labeled[0], "=" * 60)
        self.assertEqual(labeled[1], "OPTIMIZER: LabeledFewShot")
        for line in (
            "Validation Accuracy:   63.33% (38/60)",
            "Locked Test Accuracy:  67.50% (54/80)",
            "Same-Model Baseline:   53.75% (43/80)",
            "Accuracy Uplift:       +13.75 percentage points",
            "-" * 60,
            "Optimization Cost:     $0.0000",
            "Evaluation Cost:       $0.2263",
            "Optimization Time:     0.0s",
            "Mean / p95 Latency:    1.621s / 2.482s",
        ):
            self.assertIn(line, labeled)

        # Table 6-1 (page 157) reports these two rows to two decimals.
        miprov2 = format_result(published_result("miprov2"))
        self.assertIn("Locked Test Accuracy:  66.25% (53/80)", miprov2)
        self.assertIn("Total Tokens:          565,949", miprov2)
        self.assertIn("Total Recorded Cost:   $1.0685", miprov2)
        self.assertIn("Optimization Time:     285.2s (4.8 minutes)", miprov2)
        self.assertIn(
            "Accuracy Uplift:       -6.25 percentage points",
            format_result(published_result("simba")),
        )

        ensemble = format_result(published_result("ensemble")).splitlines()
        self.assertEqual(ensemble[1], "TRANSFORMATION: Ensemble")

        finetune = format_result(published_result("bootstrap-finetune"))
        self.assertIn("Student Model:         Qwen/Qwen2.5-0.5B-Instruct", finetune)
        self.assertIn("Teacher Model:         openai/gpt-5.6-sol", finetune)
        self.assertIn("Same-Model Baseline:   51.25% (41/80)", finetune)
        self.assertIn("Accepted Traces:       77 human / 63 AI", finetune)

    def test_published_results_reproduce_every_printed_result_block(self) -> None:
        self.assertEqual(
            set(PRINTED_RESULT_BLOCKS),
            {spec["optimizer"] for spec in NOTEBOOKS.values()} - {"quickstart"},
        )
        for optimizer, (page, printed) in PRINTED_RESULT_BLOCKS.items():
            shown = format_result(published_result(optimizer)).splitlines()
            for line in printed.strip().splitlines():
                with self.subTest(optimizer=optimizer, page=page, line=line):
                    self.assertTrue(
                        any(matches_printed_line(line, candidate) for candidate in shown)
                    )

        # The comparison accepts added detail and extra decimals, nothing else.
        self.assertTrue(
            matches_printed_line(
                "Validation Accuracy:   63.33%", "Validation Accuracy:   63.33% (38/60)"
            )
        )
        self.assertTrue(
            matches_printed_line(
                "Locked Test Accuracy: 56.3% (45/80)",
                "Locked Test Accuracy:  56.25% (45/80)",
            )
        )
        self.assertFalse(
            matches_printed_line(
                "Locked Test Accuracy: 56.3% (45/80)",
                "Locked Test Accuracy:  50.00% (40/80)",
            )
        )
        self.assertFalse(
            matches_printed_line(
                "Optimization Cost:     $0.8156", "Optimization Cost:     $0.0732"
            )
        )

    def test_notebooks_quote_the_output_printed_in_the_book(self) -> None:
        # The result blocks quoted in the notebooks are the printed ones.
        for optimizer, (page, printed) in PRINTED_RESULT_BLOCKS.items():
            with self.subTest(optimizer=optimizer):
                self.assertEqual(PRINTED_OUTPUT[optimizer], (page, printed.strip("\n")))
        self.assertEqual(
            PRINTED_OUTPUT["baseline-validation"],
            (135, "Average Metric: 33.00 / 60 (55.0%)"),
        )
        self.assertEqual(
            PRINTED_OUTPUT["baseline-test"],
            (136, "Average Metric: 43.00 / 80 (53.8%)"),
        )

        self.assertEqual(set(QUOTED_OUTPUT), set(NOTEBOOKS))
        self.assertEqual(
            sorted(key for keys in QUOTED_OUTPUT.values() for key in keys),
            sorted(PRINTED_OUTPUT),
        )
        heading = "**Output printed in the book (page"
        for filename, keys in QUOTED_OUTPUT.items():
            cells = all_cells(filename)
            quoted = [
                index
                for index, (cell_type, source) in enumerate(cells)
                if cell_type == "markdown" and source.startswith(heading)
            ]
            with self.subTest(filename=filename):
                self.assertEqual(len(quoted), len(keys))
            for index, key in zip(quoted, keys):
                page, printed = PRINTED_OUTPUT[key]
                # A block that continues on the next page names both pages.
                where = (
                    f"page {page}"
                    if isinstance(page, int)
                    else f"pages {page[0]}–{page[1]}"
                )
                source = cells[index][1]
                with self.subTest(filename=filename, output=key):
                    # Quoted in markdown, directly under the code cell it belongs to.
                    self.assertEqual(cells[index - 1][0], "code")
                    self.assertTrue(
                        source.startswith(
                            f"**Output printed in the book ({where}):**\n\n"
                            f"```text\n{printed}\n```\n\n"
                        )
                    )

        # The BootstrapRS block runs from page 139 to page 140, and the
        # BetterTogether block from page 153 to page 154.
        for filename, where in (
            ("bootstrap-random-search.ipynb", "pages 139–140"),
            ("better-together.ipynb", "pages 153–154"),
        ):
            with self.subTest(filename=filename):
                self.assertTrue(
                    any(
                        cell_type == "markdown"
                        and source.startswith(f"**Output printed in the book ({where}):**")
                        for cell_type, source in all_cells(filename)
                    )
                )

        # The printed instructions are the saved ones: the page drops the
        # backticks, list markers, and blank lines of the saved text.
        def flatten(text: str) -> str:
            lines = [line.removeprefix("- ") for line in text.replace("`", "").splitlines()]
            return " ".join(" ".join(lines).split())

        for optimizer, key in (("copro", "copro-instruction"), ("gepa", "gepa-instruction")):
            prompt = json.loads(
                (CHAPTER_DIR.parent / optimizer_row(optimizer)["prompt_artifact"]).read_text(
                    encoding="utf-8"
                )
            )["predictors"]["detect.predict"]["instructions"]
            saved, printed = flatten(prompt), flatten(PRINTED_OUTPUT[key][1])
            with self.subTest(optimizer=optimizer):
                if optimizer == "copro":
                    # Page 143 prints the whole COPRO instruction.
                    self.assertEqual(saved, printed)
                else:
                    # Page 147 prints the beginning of the GEPA instruction.
                    self.assertTrue(saved.startswith(printed))
                    self.assertGreater(len(saved), len(printed))

    def test_baseline_notebook_shows_the_printed_average_metric_lines(self) -> None:
        # "Setting Up the AI Detection Module" (pages 135-136) prints
        # "Average Metric: 33.00 / 60 (55.0%)" and "43.00 / 80 (53.8%)".
        baseline = published_result("quickstart")
        self.assertEqual(
            (baseline["validation_correct"], baseline["validation_examples"]), (33, 60)
        )
        self.assertEqual((baseline["correct"], baseline["test_examples"]), (43, 80))
        self.assertEqual(baseline["validation_accuracy"], 55.0)
        self.assertEqual(baseline["final_accuracy"], 53.75)

    def test_result_block_accepts_a_live_run_summary(self) -> None:
        block = format_result(
            {
                "optimizer": "copro",
                "status": "completed",
                "task_model": "openai/gpt-5.6-luna",
                "final_accuracy": 50.0,
                "correct": 2,
                "test_examples": 4,
                "validation_accuracy": 75.0,
                "validation_correct": 3,
                "validation_examples": 4,
                "optimization_seconds": 12.5,
                "optimization_cost_usd": 0.01,
                "evaluation_cost_usd": 0.002,
                "mean_latency_seconds": 1.0,
                "p95_latency_seconds": 2.0,
                "accepted_trace_labels": None,
                "output_dir": None,
            }
        )
        self.assertIn("Validation Accuracy:   75.00% (3/4)", block)
        self.assertIn("Locked Test Accuracy:  50.00% (2/4)", block)
        self.assertNotIn("Same-Model Baseline", block)
        self.assertNotIn("Total Tokens", block)

    def test_result_block_reports_no_test_score_for_a_smoke_run(self) -> None:
        # A smoke run stops after validation, so its summary repeats the
        # validation score in the ``final_accuracy`` field.
        block = format_result(
            {
                "optimizer": "labeled-few-shot",
                "status": "completed",
                "mode": "smoke",
                "task_model": "openai/gpt-5.6-luna",
                "final_accuracy": 75.0,
                "correct": 3,
                "test_examples": 4,
                "validation_accuracy": 75.0,
                "validation_correct": 3,
                "validation_examples": 4,
                "optimization_seconds": 0.0,
                "optimization_cost_usd": 0.0,
                "evaluation_cost_usd": 0.002,
                "mean_latency_seconds": 1.0,
                "p95_latency_seconds": 2.0,
            }
        )
        self.assertIn("Validation Accuracy:   75.00% (3/4)", block)
        self.assertIn("Locked Test Accuracy:  not scored (smoke mode)", block)


class GeneratedNotebookTest(unittest.TestCase):
    def test_every_optimizer_notebook_is_generated_and_executed(self) -> None:
        for filename, spec in NOTEBOOKS.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    validate_notebook(
                        CHAPTER_DIR / filename,
                        spec["optimizer"],
                        require_executed=True,
                    ),
                    [],
                )

    def test_every_notebook_loads_the_shared_split_and_shows_the_saved_result(
        self,
    ) -> None:
        for filename in NOTEBOOKS:
            source = "\n".join(code_cells(filename))
            with self.subTest(filename=filename):
                self.assertTrue(contains_listing(LISTINGS["load-dataset"], source))
                self.assertIn("print(format_result(published_result(OPTIMIZER)))", source)
                self.assertIn("CHAPTER06_RUN_LIVE", source)
                self.assertNotIn("run_optimizer(", source)

    def test_every_chapter_listing_appears_verbatim_in_a_code_cell(self) -> None:
        shown: set[str] = {"load-dataset"}
        for filename, spec in NOTEBOOKS.items():
            cells = code_cells(filename)
            for item in listing_items(spec):
                shown.add(item["key"])
                with self.subTest(filename=filename, listing=item["key"]):
                    self.assertTrue(
                        any(contains_listing(LISTINGS[item["key"]], cell) for cell in cells)
                    )
        self.assertEqual(shown, set(LISTINGS))

    def test_listings_that_call_a_model_sit_under_the_live_guard(self) -> None:
        for filename, spec in NOTEBOOKS.items():
            cells = code_cells(filename)
            for item in listing_items(spec):
                listing = LISTINGS[item["key"]]
                cell = next(cell for cell in cells if contains_listing(listing, cell))
                with self.subTest(filename=filename, listing=item["key"]):
                    if item["key"] in FREE_LISTINGS:
                        self.assertIsNone(item["guard"])
                        self.assertEqual(cell, listing)
                    else:
                        self.assertTrue(cell.startswith("if RUN_LIVE:\n"))
                        self.assertIn("\nelse:\n    print(", cell)

    def test_live_mode_scores_the_program_the_listing_compiled(self) -> None:
        for filename, spec in NOTEBOOKS.items():
            source = "\n".join(code_cells(filename))
            with self.subTest(filename=filename):
                if spec["optimizer"] == "quickstart":
                    # The baseline listings score the program with dspy.Evaluate.
                    self.assertIn("validation_result = validation_evaluator(detector)", source)
                    self.assertIn("test_result = test_evaluator(detector)", source)
                    continue
                program = spec["score"]["program"]
                self.assertIn(f"{program} = ", source)
                self.assertIn(
                    f"    print(score_summary({program}, valset, testset))", source
                )

    def test_instruction_optimizer_notebooks_show_the_live_program(self) -> None:
        # In live mode the scoring cell of the four instruction optimizers
        # first prints the instruction and demonstration count of the program
        # the listing compiled. Run those lines on an unoptimized detector.
        for filename in ("copro.ipynb", "miprov2.ipynb", "gepa.ipynb", "simba.ipynb"):
            cell = next(
                cell for cell in code_cells(filename) if "score_summary(" in cell
            )
            body = [
                line
                for line in _mode_source(cell, live=True).split("\n")
                if "score_summary(" not in line
            ]
            self.assertEqual(body[0], "predictor = optimized_detector.detect.predict")
            printed: list[str] = []
            namespace = {
                "optimized_detector": AIDetector(),
                "print": lambda *args: printed.append(" ".join(map(str, args))),
            }
            with self.subTest(filename=filename):
                exec("\n".join(body), namespace)
                self.assertEqual(
                    printed,
                    [
                        "Instruction: Decide whether the supplied passage was "
                        "generated by an AI.",
                        "Demonstrations: 0\n",
                    ],
                )

    def test_names_resolve_in_default_and_live_mode(self) -> None:
        for filename in NOTEBOOKS:
            cells = code_cells(filename)
            for live in (False, True):
                with self.subTest(filename=filename, live=live):
                    self.assertEqual(undefined_names(cells, live=live), [])

        # A listing that uses a name from a live-only cell is reported.
        self.assertEqual(
            undefined_names(
                ["RUN_LIVE = False", "if RUN_LIVE:\n    program = 1", "print(program)"],
                live=False,
            ),
            [(2, "program")],
        )

    def test_default_mode_runs_offline_without_model_calls(self) -> None:
        def refuse_model_call(*args: object, **kwargs: object) -> None:
            raise AssertionError("a notebook cell called a model in the default mode")

        # The notebooks call dspy.configure; restore the setting afterwards.
        self.addCleanup(dspy.configure, lm=dspy.settings.lm)
        for filename, spec in NOTEBOOKS.items():
            namespace: dict[str, object] = {"__name__": "__main__"}
            with (
                self.subTest(filename=filename),
                patch.dict(os.environ, {"CHAPTER06_RUN_LIVE": "0"}),
                patch("pathlib.Path.cwd", return_value=CHAPTER_DIR.parent),
                patch("dotenv.load_dotenv"),
                patch.object(dspy.LM, "__call__", refuse_model_call),
                patch("builtins.print"),
            ):
                for cell in code_cells(filename):
                    exec(compile(cell, filename, "exec"), namespace)
                self.assertIs(namespace["RUN_LIVE"], False)
                self.assertEqual(namespace["OPTIMIZER"], spec["optimizer"])
                self.assertEqual(len(namespace["testset"]), 80)

    def test_live_mode_stops_in_the_setup_cell_without_an_api_key(self) -> None:
        # Every live run calls OpenAI (the task model, or the GPT-5.6-sol
        # teacher of the fine-tuning notebooks), so a missing key is reported
        # by the first code cell instead of after minutes of failed calls.
        for filename in NOTEBOOKS:
            setup_cell = code_cells(filename)[0]
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "OPENAI_API_KEY"
            }
            environment["CHAPTER06_RUN_LIVE"] = "1"
            with (
                self.subTest(filename=filename),
                patch.dict(os.environ, environment, clear=True),
                patch("pathlib.Path.cwd", return_value=CHAPTER_DIR.parent),
                patch("dotenv.load_dotenv"),
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "CHAPTER06_RUN_LIVE=1 needs OPENAI_API_KEY"
                ):
                    exec(compile(setup_cell, filename, "exec"), {"__name__": "__main__"})

                # With a key the setup cell only reports the mode; it calls no model.
                os.environ["OPENAI_API_KEY"] = "placeholder-key-for-this-test"
                namespace: dict[str, object] = {"__name__": "__main__"}
                exec(compile(setup_cell, filename, "exec"), namespace)
                self.assertIs(namespace["RUN_LIVE"], True)

    def test_generator_output_matches_the_checked_in_notebooks(self) -> None:
        for filename, spec in NOTEBOOKS.items():
            generated = [
                "".join(cell["source"])
                for cell in make_notebook(spec)["cells"]
                if cell["cell_type"] == "code"
            ]
            with self.subTest(filename=filename):
                self.assertEqual(code_cells(filename), generated)

    def test_gepa_listing_uses_the_standard_light_configuration_and_protocol(
        self,
    ) -> None:
        listing = LISTINGS["gepa"]
        self.assertEqual(NOTEBOOKS["gepa.ipynb"]["compile"], listing)
        self.assertIn("def exact_match_with_feedback(", listing)
        self.assertIn("feedback=example.notes,", listing)
        self.assertIn('auto="light",', listing)
        self.assertIn('candidate_selection_strategy="pareto",', listing)
        self.assertIn("use_merge=False,", listing)
        self.assertIn("seed=42,", listing)

        row = optimizer_row("gepa")
        self.assertEqual(
            row["evaluation_protocol"],
            "one fresh uncached validation pass, then one locked-test pass",
        )
        self.assertEqual(row["baseline_accuracy_pct"], 53.75)
        self.assertEqual(row["locked_test_accuracy_pct"], 80.0)

    def test_gepa_runner_matches_the_published_configuration(self) -> None:
        class FakeGEPA:
            kwargs: dict[str, object] = {}

            def __init__(self, **kwargs: object) -> None:
                type(self).kwargs = kwargs

            def compile(self, student: AIDetector, **kwargs: object) -> AIDetector:
                return student

        with patch("chapter06.optimizer_runtime.dspy.GEPA", FakeGEPA):
            _compile_prompt_optimizer(
                "gepa",
                AIDetector(),
                [],
                [],
                object(),
                object(),
                smoke=False,
            )

        self.assertEqual(FakeGEPA.kwargs["auto"], "light")
        self.assertNotIn("max_full_evals", FakeGEPA.kwargs)
        self.assertEqual(FakeGEPA.kwargs["candidate_selection_strategy"], "pareto")
        self.assertIs(FakeGEPA.kwargs["use_merge"], False)


if __name__ == "__main__":
    unittest.main()
