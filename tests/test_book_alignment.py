"""Checks that the notebooks match the listings printed in the book.

Each test names the chapter and page it protects. The tests read the notebooks
as files and, where it is cheap, check the printed code against the installed
DSPy. They make no model calls and need no API keys.
"""

from __future__ import annotations

import ast
import asyncio
import csv
import inspect
import json
import re
import sys
import unittest
from functools import cache
from importlib import import_module
from pathlib import Path

import dspy
from dspy.utils import DummyLM
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chapter06 import build_optimizer_notebooks


REPO_ROOT = Path(__file__).resolve().parents[1]

# Every model identifier the book prints, in the provider/model form that
# DSPy passes to LiteLLM.
BOOK_MODELS = {
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "openai/gpt-image-1.5",
    "openai/gpt-image-2",
    "openai/text-embedding-3-small",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-5",
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemini-3.5-flash",
    "openrouter/anthropic/claude-opus-4.7",
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter/moonshotai/kimi-k2.6",
}
# Identifiers that appear in one notebook for a stated reason.
NOTEBOOK_SPECIFIC_MODELS = {
    # DSPy's local-provider form of the student model on page 152,
    # Qwen/Qwen2.5-0.5B-Instruct.
    "chapter06/bootstrap-finetune.ipynb": {"openai/local:Qwen/Qwen2.5-0.5B-Instruct"},
    # The notebook names the model that produced its recorded benchmark run.
    "chapter09/invoice-extraction.ipynb": {"openai/gpt-4o-mini"},
}
MODEL_PATTERN = re.compile(
    r"(?<![\w/.\-])((?:openai|anthropic|gemini|openrouter)/[\w.\-:/]+)"
)

# The printed book has no section numbers, so a reference such as
# "Chapter 7.4.5" or a heading such as "## 10.3 Optimization" cannot be
# looked up by a reader.
NUMBERED_SECTION_PATTERNS = (
    re.compile(r"§"),
    re.compile(r"^#{1,6}\s+\d+\.\d+", re.MULTILINE),
    re.compile(r"\b(?:Chapter|Ch\.?|[Ss]ections?)\s+\d+\.\d+"),
)

# GEPA metrics that a notebook builds inside a factory function. The value
# lists the inner functions that the factory returns.
FACTORY_BUILT_GEPA_METRICS = {
    ("chapter03/dspy-in-8-steps.ipynb", "optimized_llm_judge"): ("_metric",),
    ("chapter10/dspyui-gradio.ipynb", "metric"): (
        "exact_match",
        "cosine_metric",
        "judge_metric",
    ),
}


def notebook_paths() -> list[str]:
    return sorted(
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.glob("chapter[0-9][0-9]/*.ipynb")
    )


@cache
def load_notebook(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def cell_sources(relative_path: str, cell_type: str | None = None) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in load_notebook(relative_path).get("cells", [])
        if cell_type is None or cell.get("cell_type") == cell_type
    ]


def notebook_source(relative_path: str, *, code_only: bool = False) -> str:
    return "\n".join(cell_sources(relative_path, "code" if code_only else None))


def python_source(source: str) -> str:
    """Replace IPython ``%`` and ``!`` lines with ``pass`` so ``ast`` can parse the cell."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append(line[: len(line) - len(stripped)] + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


@cache
def notebook_tree(relative_path: str) -> ast.Module:
    """All code cells of a notebook parsed as one module."""
    return ast.parse(
        "\n".join(
            python_source(source) for source in cell_sources(relative_path, "code")
        )
    )


def contains_listing(cell: str, listing: str) -> bool:
    """True if the printed lines appear together, in order, inside ``cell``.

    The lines may be indented as a block, as they are under an ``if`` guard.
    """
    wanted = listing.strip("\n").splitlines()
    present = cell.splitlines()
    for start in range(len(present) - len(wanted) + 1):
        first = present[start]
        if not first.endswith(wanted[0]):
            continue
        indent = first[: len(first) - len(wanted[0])]
        if indent.strip():
            continue
        if all(
            (not want.strip() and not have.strip()) or have == indent + want
            for want, have in zip(wanted, present[start:])
        ):
            return True
    return False


def calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    """Calls to ``name(...)`` or ``<anything>.name(...)``."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = (
            function.attr
            if isinstance(function, ast.Attribute)
            else getattr(function, "id", None)
        )
        if called == name:
            calls.append(node)
    return calls


def keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def accepts_five_positional_arguments(function: ast.FunctionDef) -> bool:
    arguments = function.args
    return (
        arguments.vararg is not None
        or len(arguments.posonlyargs) + len(arguments.args) >= 5
    )


class BookListingTestCase(unittest.TestCase):
    def assertListing(
        self, relative_path: str, listing: str, *, cell_type: str = "code"
    ) -> None:
        """Assert that a printed listing is in one cell of the notebook, line for line."""
        if not any(
            contains_listing(cell, listing)
            for cell in cell_sources(relative_path, cell_type)
        ):
            self.fail(
                f"{relative_path} has no {cell_type} cell with these printed "
                f"lines:\n{listing}"
            )


class PrefaceAndDataTest(BookListingTestCase):
    def test_chapter8_mcp_server_starts_and_lists_its_tools(self) -> None:
        # Preface, "Software Requirements for This Book": the server is run
        # with `uv run python chapter08/mcp_server.py`. The Chapter 8 agent on
        # pages 206-207 connects to it over stdio.
        async def list_tool_names() -> list[str]:
            server = StdioServerParameters(
                command=sys.executable,
                args=[str(REPO_ROOT / "chapter08/mcp_server.py")],
                env={"PYTHONWARNINGS": "ignore"},
            )
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return [tool.name for tool in result.tools]

        tool_names = asyncio.run(asyncio.wait_for(list_tool_names(), timeout=10))
        self.assertEqual(sorted(tool_names), ["book_flight", "search_flights"])

        self.assertListing(
            "chapter08/mcp-integration.ipynb",
            """
    server_params = StdioServerParameters(
        command="python",
        args=["chapter08/mcp_server.py"]
    )
""",
            cell_type="markdown",
        )
        self.assertListing(
            "chapter08/mcp-integration.ipynb",
            """
            mcp_tools = await session.list_tools()
            tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]
""",
        )

    def test_chapter3_dataset_has_ten_ai_and_ten_human_texts(self) -> None:
        # Chapter 3, "Example Dataset" (page 62): "a dataset of 20 texts,
        # 10 AI-generated and 10 human-written".
        with (REPO_ROOT / "data/ai_vs_human.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["is_ai"] == "True" for row in rows), 10)
        self.assertEqual(sum(row["is_ai"] == "False" for row in rows), 10)
        for row in rows:
            self.assertTrue(row["text"].strip())
            self.assertTrue(row["notes"].strip())


class ModelAndReferenceTest(unittest.TestCase):
    def test_notebooks_use_only_model_identifiers_printed_in_the_book(self) -> None:
        used_anywhere: set[str] = set()
        for relative_path in notebook_paths():
            used = {
                match.rstrip(".,:")
                for match in MODEL_PATTERN.findall(notebook_source(relative_path))
            }
            # "openrouter/..." in a comment names the provider, not a model.
            used = {name for name in used if not name.endswith("/")}
            used_anywhere |= used
            allowed = BOOK_MODELS | NOTEBOOK_SPECIFIC_MODELS.get(relative_path, set())
            with self.subTest(notebook=relative_path):
                self.assertEqual(sorted(used - allowed), [])

        # The book's default model (page xvii) and its stronger model are used.
        self.assertIn("openai/gpt-5.6-luna", used_anywhere)
        self.assertIn("openai/gpt-5.6-sol", used_anywhere)
        for relative_path, extra in NOTEBOOK_SPECIFIC_MODELS.items():
            with self.subTest(notebook=relative_path):
                self.assertLessEqual(
                    extra,
                    set(MODEL_PATTERN.findall(notebook_source(relative_path))),
                )

    def test_notebooks_and_guides_refer_to_sections_by_title(self) -> None:
        texts = {
            relative_path: notebook_source(relative_path)
            for relative_path in notebook_paths()
        }
        for guide in (
            "README.md",
            "BOOK_NOTES.md",
            "data/README.md",
            "chapter06/README.md",
            "chapter06/CHAPTER_RESULTS.md",
            "chapter10/README.md",
        ):
            texts[guide] = (REPO_ROOT / guide).read_text(encoding="utf-8")

        for name, text in texts.items():
            found = [
                match.group(0)
                for pattern in NUMBERED_SECTION_PATTERNS
                for match in pattern.finditer(text)
            ]
            with self.subTest(file=name):
                self.assertEqual(found, [])

        # Chapter 10 reuses the adapter from Chapter 7 and names that section
        # by its printed title (pages 191-193).
        self.assertIn(
            'Chapter 7, "Building Custom Adapters"',
            texts["chapter10/fastapi-invoice-api.ipynb"],
        )


class GepaMetricTest(BookListingTestCase):
    def test_gepa_rejects_a_metric_with_fewer_than_five_arguments(self) -> None:
        def three_arguments(example, prediction, trace=None):
            return 1.0

        def five_arguments(
            example, prediction, trace=None, pred_name=None, pred_trace=None
        ):
            return 1.0

        reflection_lm = DummyLM([{"answer": "unused"}])
        with self.assertRaises(TypeError):
            dspy.GEPA(metric=three_arguments, auto="light", reflection_lm=reflection_lm)
        self.assertIsInstance(
            dspy.GEPA(metric=five_arguments, auto="light", reflection_lm=reflection_lm),
            dspy.GEPA,
        )

    def test_every_gepa_metric_in_the_notebooks_takes_five_arguments(self) -> None:
        checked: set[tuple[str, str]] = set()
        for relative_path in notebook_paths():
            tree = notebook_tree(relative_path)
            functions: dict[str, list[ast.FunctionDef]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.setdefault(node.name, []).append(node)

            for call in calls_named(tree, "GEPA"):
                metric = keyword_value(call, "metric")
                if not isinstance(metric, ast.Name):
                    continue
                names = FACTORY_BUILT_GEPA_METRICS.get(
                    (relative_path, metric.id), (metric.id,)
                )
                for name in names:
                    with self.subTest(notebook=relative_path, metric=name):
                        self.assertIn(name, functions)
                        for function in functions[name]:
                            self.assertTrue(
                                accepts_five_positional_arguments(function)
                            )
                    checked.add((relative_path, name))

        # Metrics that the book's GEPA listings use, with the page that
        # prints the metric or the GEPA call that takes it.
        for expected in (
            ("chapter02/dspy-tour.ipynb", "decision_match"),  # page 43
            ("chapter03/dspy-in-8-steps.ipynb", "exact_match"),  # page 64
            ("chapter05/human-then-llm-judge.ipynb", "llm_judge_metric"),  # page 123
            ("chapter06/gepa.ipynb", "exact_match_with_feedback"),  # page 145
            ("chapter09/invoice-extraction.ipynb", "field_accuracy_metric"),  # page 227
            ("chapter10/mlflow-tracking.ipynb", "accuracy_with_feedback"),  # page 256
            ("chapter11/image-cli-optimizer.ipynb", "cli_metric"),  # page 298
        ):
            self.assertIn(expected, checked)

    def test_chapter2_shows_the_printed_metric_and_a_runnable_one(self) -> None:
        # Page 43 prints the signature across two lines. The notebook shows
        # it as printed, then defines the metric on one line with a comma
        # between `pred_name` and `pred_trace`, the form dspy.GEPA calls.
        notebook = "chapter02/dspy-tour.ipynb"
        printed = """
def decision_match(example, prediction, trace=None, pred_name=None and
    pred_trace=None):
    \"\"\"Check whether the predicted decision matches the expected decision.\"\"\"
    return example.decision == prediction.decision
"""
        runnable = """
def decision_match(example, prediction, trace=None, pred_name=None, pred_trace=None):
    \"\"\"Check whether the predicted decision matches the expected decision.\"\"\"
    return example.decision == prediction.decision
"""
        self.assertListing(notebook, printed, cell_type="markdown")
        self.assertListing(notebook, runnable)

        namespace: dict[str, object] = {}
        exec(runnable, namespace)
        decision_match = namespace["decision_match"]
        example = dspy.Example(decision="approve")
        self.assertTrue(decision_match(example, dspy.Prediction(decision="approve")))
        self.assertFalse(decision_match(example, dspy.Prediction(decision="reject")))


class ChapterListingTest(BookListingTestCase):
    def test_saved_programs_are_loaded_with_allow_pickle(self) -> None:
        # dspy.load raises ValueError for a program saved with
        # save_program=True unless the caller passes allow_pickle=True.
        self.assertIs(
            inspect.signature(dspy.load).parameters["allow_pickle"].default, False
        )

        loads: list[tuple[str, ast.Call]] = []
        for relative_path in notebook_paths():
            for call in calls_named(notebook_tree(relative_path), "load"):
                function = call.func
                if (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "dspy"
                ):
                    loads.append((relative_path, call))

        self.assertEqual(
            sorted({relative_path for relative_path, _ in loads}),
            ["chapter03/dspy-in-8-steps.ipynb", "chapter10/fastapi-invoice-api.ipynb"],
        )
        self.assertGreaterEqual(len(loads), 3)
        for relative_path, call in loads:
            with self.subTest(notebook=relative_path, call=ast.unparse(call)):
                allow_pickle = keyword_value(call, "allow_pickle")
                self.assertIsInstance(allow_pickle, ast.Constant)
                self.assertIs(allow_pickle.value, True)

        # Chapter 3, "Running the GEPA Optimizer" (page 71).
        self.assertListing(
            "chapter03/dspy-in-8-steps.ipynb",
            """
# Save the program state and architecture
optimized_judge.save("./ai_detector/", save_program=True)

# Load the judge again from the saved state
loaded_optimized_judge = dspy.load("./ai_detector/", allow_pickle=True)
""",
        )
        # Chapter 10, "Saving and Loading Programs" (page 263).
        self.assertListing(
            "chapter10/fastapi-invoice-api.ipynb",
            """
extractor = dspy.load(
    "./invoice_extractor/",
    allow_pickle=True,
)
""",
        )

    def test_chapter4_dataset_listings_match_pages_86_to_97(self) -> None:
        router = "chapter04/error-analysis-router.ipynb"
        datasets = "chapter04/hf-datasets.ipynb"

        # "Define a Dataset" (page 86).
        self.assertListing(
            router,
            """
# Combine all your examples
all_examples = list(dataset)

# Shuffle with seed for reproducibility
random.Random(2024).shuffle(all_examples)
""",
        )

        # "Built-In DSPy Datasets" (page 95). DSPy 3.3.0 provides GSM8K in the
        # dspy.datasets.gsm8k module, which is the import the book prints.
        self.assertTrue(hasattr(import_module("dspy.datasets.gsm8k"), "GSM8K"))
        self.assertNotIn(
            "from dspy.datasets import GSM8K", notebook_source(datasets)
        )
        self.assertListing(
            datasets,
            """
from dspy.datasets.gsm8k import GSM8K
dataset = GSM8K()
train_set = (
    [dspy.Example(question=x.question, answer=x.answer).with_inputs('question')
            for x in dataset.train[:100]]
)

print(train_set[0].question)
# "James writes a 3-page letter to 2 different friends twice a week. How many
# pages does he write a year?"
print(train_set[0].answer)
# "624"
""",
        )
        # The comments above show the example printed in the book. The reader
        # notes name the example that the pinned DSPy and datasets versions
        # return first, so a reader is not surprised by different output.
        notes = "\n".join(cell_sources(datasets, "markdown"))
        self.assertIn("At My Window", notes)
        self.assertIn("Statistics exam Marion and Ella", notes)

        # "Hugging Face Datasets" (page 97). The listing stores the evaluation
        # result in `score` and prints `score.score`.
        self.assertListing(
            datasets,
            """
# Evaluate on the full dataset
evaluator = dspy.Evaluate(
    devset=train_set_full,
    metric=exact_match,
    display_progress=True,
)
score = evaluator(optimized_program)
print(f"Full dataset accuracy: {score.score:.1f}%")
""",
        )
        self.assertNotIn("result.score", notebook_source(datasets, code_only=True))

    def test_chapter6_listings_match_pages_135_to_149(self) -> None:
        listings = build_optimizer_notebooks.LISTINGS
        notebooks = build_optimizer_notebooks.NOTEBOOKS

        def assert_printed(key: str, notebook: str, printed: str) -> None:
            with self.subTest(listing=key):
                self.assertTrue(contains_listing(listings[key], printed))
                self.assertListing(f"chapter06/{notebook}", printed)

        # "Setting Up the AI Detection Module" (page 135). Every optimizer
        # notebook starts from this listing.
        load_dataset = """
# Load the frozen Chapter 6 dataset and pair-grouped split
from chapter06.optimizer_runtime import (
    AIDetector,
    exact_match,
    hashed_ngram_embeddings,
    load_frozen_examples,
    split_summary,
)

NUM_THREADS = 1
splits = load_frozen_examples()
trainset = splits["train"]
valset = splits["validation"]
testset = splits["test"]

print(split_summary(splits))
"""
        self.assertEqual(listings["load-dataset"].strip("\n"), load_dataset.strip("\n"))
        for filename, spec in notebooks.items():
            generated = [
                "".join(cell["source"])
                for cell in build_optimizer_notebooks.make_notebook(spec)["cells"]
                if cell["cell_type"] == "code"
            ]
            with self.subTest(notebook=filename):
                self.assertTrue(
                    any(contains_listing(cell, load_dataset) for cell in generated)
                )
                self.assertListing(f"chapter06/{filename}", load_dataset)

        # "BootstrapFewShot" (page 138).
        assert_printed(
            "bootstrap-few-shot",
            "bootstrap-few-shot.ipynb",
            """
optimizer = dspy.teleprompt.BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=1,
    )
""",
        )

        # "KNNFewShot" (page 140). The training set and the vectorizer go to
        # the constructor, and compile() takes only the program.
        assert_printed(
            "knn-few-shot",
            "knn-few-shot.ipynb",
            """
# Set up the optimizer and training data
optimizer = dspy.teleprompt.KNNFewShot(
        k=4,
        metric=exact_match,
        trainset=trainset,
        vectorizer=dspy.Embedder(hashed_ngram_embeddings),

    )


optimized_detector = optimizer.compile(
    AIDetector()
)
""",
        )
        self.assertNotIn("max_bootstrapped_demos", listings["knn-few-shot"])
        self.assertNotIn("max_labeled_demos", listings["knn-few-shot"])

        # "COPRO" (page 142). The listing compiles on the training split.
        assert_printed(
            "copro",
            "copro.ipynb",
            """
# Compile with training data
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    eval_kwargs={
        "num_threads": NUM_THREADS,
        "display_progress": True,
        "display_table": False,
    },
)
""",
        )
        self.assertNotIn("trainset=valset", listings["copro"])

        # "SIMBA" (page 149).
        assert_printed(
            "simba",
            "simba.ipynb",
            """
optimizer = dspy.teleprompt.SIMBA(
        metric=exact_match,
        bsize=8,  # Mini-batch size
        num_candidates=4,  # Number of candidates per iteration
        max_steps=6,  # Number of optimization steps
        max_demos=2,
        prompt_model=teacher_lm,
        num_threads=NUM_THREADS,
    )
# Compile with training data
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    seed=42,
)
""",
        )
        # The command-line runner uses the printed six steps outside smoke mode.
        runtime = (REPO_ROOT / "chapter06/optimizer_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_steps=1 if smoke else 6", runtime)

    def test_chapter6_sources_name_the_pinned_dspy_version(self) -> None:
        project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        pinned = re.search(r'"dspy==(\d+\.\d+\.\d+)"', project)
        self.assertIsNotNone(pinned)
        self.assertEqual(dspy.__version__, pinned.group(1))

        mentioned: set[str] = set()
        for source_file in ("build_optimizer_notebooks.py", "apple_finetune.py"):
            source = (REPO_ROOT / "chapter06" / source_file).read_text(
                encoding="utf-8"
            )
            mentioned |= set(re.findall(r"DSPy (\d+\.\d+\.\d+)", source))
        self.assertEqual(mentioned, {pinned.group(1)})

    def test_chapter7_listings_match_pages_170_to_180(self) -> None:
        # "Pattern 5: Self-refinement loop" (page 174). The revise step reads
        # the earlier text through its own `previous_draft` input field.
        patterns = "chapter07/multi-stage-patterns.ipynb"
        self.assertListing(
            patterns,
            """
        self.revise = dspy.ChainOfThought(
            (
                "brief: str, previous_draft: str, "
                "feedback: str -> draft_text: str"
            )
        )
""",
        )
        self.assertListing(
            patterns,
            """
            result = self.revise(
                brief=brief,
                previous_draft=result.draft_text,
                feedback=review.feedback,
            )
""",
        )

        # "Images" (page 180). The listing uses placeholder files, so the
        # notebook shows it as printed.
        self.assertTrue(callable(dspy.Image.from_path))
        self.assertListing(
            "chapter07/multimodal.ipynb",
            """
# All of these work
img1 = dspy.Image("https://example.com/product.jpg")
img2 = dspy.Image.from_path("/path/to/local/photo.png")
img3 = dspy.Image(pil_image)  # PIL.Image object
""",
            cell_type="markdown",
        )

        # "Flex" (page 170). GEPA takes exactly one budget; the book uses the
        # light preset.
        self.assertListing(
            "chapter07/flex.ipynb",
            """
optimized = dspy.GEPA(
    metric=metric,
    auto="light",
    reflection_lm=reflection_lm
).compile(
    program,
    trainset=trainset,
    valset=valset
)
""",
        )
        for notebook in ("chapter07/flex.ipynb", "chapter07/modules-tour.ipynb"):
            calls = calls_named(notebook_tree(notebook), "GEPA")
            self.assertTrue(calls)
            for call in calls:
                with self.subTest(notebook=notebook, call=ast.unparse(call)):
                    budgets = [
                        keyword.arg
                        for keyword in call.keywords
                        if keyword.arg in {"auto", "max_full_evals", "max_metric_calls"}
                    ]
                    self.assertEqual(budgets, ["auto"])
                    self.assertEqual(keyword_value(call, "auto").value, "light")

    def test_chapter8_memory_listings_match_pages_218_to_220(self) -> None:
        memory = "chapter08/history-mem0-rlm.ipynb"

        # "Memory with Mem0" (page 218). The configuration has an LLM and an
        # embedder and no graph store.
        self.assertListing(
            memory,
            """
config = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-5.6-luna", "temperature": 0.1}
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
}
memory = Memory.from_config(config)
""",
        )
        self.assertNotIn("graph_store", notebook_source(memory, code_only=True))
        # The tip on page 219 mentions graph memory; the notebook tells the
        # reader where that feature is available.
        self.assertIn("Mem0 Platform", notebook_source(memory))

        # "Context Management with RLMs" (page 220). `max_iters` is the
        # iteration limit that dspy.RLM accepts.
        self.assertIn("max_iters", inspect.signature(dspy.RLM.__init__).parameters)
        self.assertListing(
            memory,
            """
rlm = dspy.RLM(
    "documents, question -> answer",
    max_iters=10,
    max_llm_calls=50,
    sub_lm=dspy.LM("openai/gpt-5.6-luna")
)

result = rlm(
    documents=very_long_text,  # Could be 100K+ characters
    question="What were the key findings about customer retention?"
)
""",
        )

    def test_chapter10_listings_match_pages_255_to_270(self) -> None:
        fastapi = "chapter10/fastapi-invoice-api.ipynb"
        mlflow = "chapter10/mlflow-tracking.ipynb"

        # "Viewing Logs" (page 255). The notebook defines the classifier
        # itself, so it does not depend on another chapter's notebook.
        self.assertListing(
            mlflow,
            """
from typing import Literal
class SentimentClassifier(dspy.Signature):
    \"\"\"Classify sentiment of a given text.\"\"\"
    text: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()

classifier = dspy.ChainOfThought(SentimentClassifier)
""",
        )

        # "Avoiding DSPy-Flavored Markers in Exported Prompts" (page 265).
        self.assertIn(
            "# Later, export -- the messages now use Markdown headings, "
            "no `[[ ## ]]` markers",
            notebook_source(fastapi),
        )

        # "Streaming" (page 269). The listing wraps the stream in an adapter
        # named listener_events and hands it to DSPy's streaming_response.
        self.assertTrue(callable(dspy.streaming.streaming_response))
        self.assertListing(
            fastapi,
            """
import json
from dspy.streaming import streaming_response
from fastapi.responses import StreamingResponse
""",
        )
        self.assertListing(
            fastapi,
            """
async def listener_events(stream):
    async for value in stream:
        if isinstance(value, dspy.streaming.StreamResponse):
            data = {
                "field": value.signature_field_name,
                "chunk": value.chunk,
            }
            yield f"data: {json.dumps(data)}\\n\\n"
        elif isinstance(value, dspy.streaming.StatusMessage):
            data = {"status": value.message}
            yield f"data: {json.dumps(data)}\\n\\n"
        else:
            yield value

@app.post("/extract/stream")
async def stream_extract(request: InvoiceRequest):
    stream = streaming_extractor(text=request.text)
    return StreamingResponse(
        streaming_response(listener_events(stream)),
        media_type="text/event-stream"
    )
""",
        )

        # "Error Handling and Rate Limits" (page 270). The fallback catches
        # DSPy's own LMError, so the notebook needs no LiteLLM exception import.
        self.assertTrue(issubclass(dspy.LMError, Exception))
        self.assertListing(
            fastapi,
            """
# Bump DSPy's built-in retries from 3 to 5 for the primary provider
lm = dspy.LM("openai/gpt-5.6-sol", num_retries=5)
fallback_lm = dspy.LM("anthropic/claude-opus-4.8", num_retries=3)
dspy.configure(lm=lm)

@app.post("/extract")
async def extract_invoice(request: InvoiceRequest):
    try:
        result = await async_extractor(text=request.text)
        return result.toDict()
    except dspy.LMError:
        # Primary provider is down even after retries. Fall over.
        with dspy.context(lm=fallback_lm):
            result = await async_extractor(text=request.text)
            return result.toDict()
""",
        )
        self.assertNotIn(
            "from litellm.exceptions import", notebook_source(fastapi, code_only=True)
        )


if __name__ == "__main__":
    unittest.main()
