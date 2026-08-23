from __future__ import annotations

import ast
import asyncio
import csv
import json
import sys
import unittest
from functools import cache
from pathlib import Path

from chapter06 import build_optimizer_notebooks
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parents[1]


@cache
def load_notebook(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def notebook_source(relative_path: str, *, code_only: bool = False) -> str:
    notebook = load_notebook(relative_path)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if not code_only or cell.get("cell_type") == "code"
    )


class FinalManuscriptAlignmentTest(unittest.TestCase):
    def test_bundled_mcp_server_starts_from_locked_environment(self) -> None:
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
        self.assertIn("search_flights", tool_names)

    def test_chapter3_dataset_is_balanced_ten_and_ten(self) -> None:
        with (REPO_ROOT / "data/ai_vs_human.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["is_ai"] == "True" for row in rows), 10)
        self.assertEqual(sum(row["is_ai"] == "False" for row in rows), 10)

    def test_gepa_metrics_expose_the_five_argument_callback_shape(self) -> None:
        chapter2 = notebook_source("chapter02/dspy-tour.ipynb")
        self.assertIn(
            "def decision_match(example, prediction, trace=None, "
            "pred_name=None, pred_trace=None):",
            chapter2,
        )

    def test_saved_program_loads_explicitly_allow_pickle(self) -> None:
        chapter3 = notebook_source("chapter03/dspy-in-8-steps.ipynb", code_only=True)
        chapter3 = "\n".join(
            "pass" if line.lstrip().startswith(("%", "!")) else line
            for line in chapter3.splitlines()
        )
        tree = ast.parse(chapter3)
        loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dspy"
            and node.func.attr == "load"
        ]
        self.assertGreaterEqual(len(loads), 2)
        for call in loads:
            with self.subTest(line=call.lineno):
                allow_pickle = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "allow_pickle"),
                    None,
                )
                self.assertIsInstance(allow_pickle, ast.Constant)
                self.assertIs(allow_pickle.value, True)

    def test_dataset_examples_use_current_imports_and_result_access(self) -> None:
        router = notebook_source("chapter04/error-analysis-router.ipynb")
        datasets = notebook_source("chapter04/hf-datasets.ipynb")
        self.assertIn("all_examples = list(dataset)", router)
        self.assertIn("from dspy.datasets.gsm8k import GSM8K", datasets)
        self.assertIn("result.score", datasets)
        self.assertIn("At My Window", datasets)
        self.assertIn("Statistics exam Marion and Ella", datasets)
        self.assertNotIn("from dspy.datasets import GSM8K", datasets)

    def test_chapter6_current_surfaces_match_the_published_configs(self) -> None:
        notebooks = build_optimizer_notebooks.NOTEBOOKS
        runtime = (REPO_ROOT / "chapter06/optimizer_runtime.py").read_text(
            encoding="utf-8"
        )
        apple = (REPO_ROOT / "chapter06/apple_finetune.py").read_text(
            encoding="utf-8"
        )
        generated = build_optimizer_notebooks.make_notebook(
            notebooks["bootstrap-random-search.ipynb"]
        )
        setup = "\n".join(
            "".join(cell.get("source", [])) for cell in generated["cells"]
        )
        self.assertIn("AIDetector", setup)
        self.assertIn("hashed_ngram_embeddings", setup)
        self.assertIn("NUM_THREADS = 1", setup)

        bootstrap = notebooks["bootstrap-few-shot.ipynb"]["compile"]
        self.assertIn("max_bootstrapped_demos=2", bootstrap)
        self.assertIn("max_labeled_demos=2", bootstrap)
        self.assertIn(
            "max_bootstrapped_demos=0, max_labeled_demos=4",
            notebooks["knn-few-shot.ipynb"]["compile"],
        )
        self.assertIn("detector, trainset=valset", notebooks["copro.ipynb"]["compile"])
        self.assertIn("max_demos=2", notebooks["simba.ipynb"]["compile"])
        self.assertIn("max_steps=1 if smoke else 6", runtime)
        builder_source = Path(build_optimizer_notebooks.__file__).read_text(encoding="utf-8")
        self.assertNotIn("DSPy 3.2.1", builder_source)
        self.assertNotIn("DSPy 3.2.1", apple)

    def test_chapter7_examples_use_valid_fields_images_and_gepa_budget(self) -> None:
        patterns = notebook_source("chapter07/multi-stage-patterns.ipynb")
        multimodal = notebook_source("chapter07/multimodal.ipynb")
        modules = notebook_source("chapter07/modules-tour.ipynb")
        self.assertIn("previous_draft: str", patterns)
        self.assertIn("previous_draft=result.draft_text", patterns)
        self.assertIn("dspy.Image.from_path", multimodal)
        self.assertIn('auto="light"', modules)

    def test_mem0_and_rlm_examples_match_current_platform_behavior(self) -> None:
        memory = notebook_source("chapter08/history-mem0-rlm.ipynb")
        memory_code = notebook_source(
            "chapter08/history-mem0-rlm.ipynb", code_only=True
        )
        self.assertNotIn("graph_store", memory_code)
        self.assertIn("Mem0 Platform", memory)
        self.assertIn("max_iters=10", memory)

    def test_chapter10_uses_correct_streaming_errors_and_cross_references(self) -> None:
        fastapi = notebook_source("chapter10/fastapi-invoice-api.ipynb")
        mlflow = notebook_source("chapter10/mlflow-tracking.ipynb")
        self.assertIn("async def sse_events", fastapi)
        self.assertIn("json.dumps(data)", fastapi)
        self.assertNotIn("from dspy.streaming import streaming_response", fastapi)
        self.assertIn("except dspy.LMError:", fastapi)
        self.assertNotIn("from litellm.exceptions import", fastapi)
        self.assertIn("output fields now use Markdown headings", fastapi)
        self.assertIn("Building Custom Adapters", fastapi)
        self.assertNotIn("Chapter 7.4.5", fastapi)
        self.assertIn("Define a `SentimentClassifier`", mlflow)
        self.assertNotIn("SentimentClassifier` from Chapter 9", mlflow)

    def test_chapter11_companion_metric_is_present(self) -> None:
        image_cli = notebook_source("chapter11/image-cli-optimizer.ipynb")
        self.assertIn("def cli_metric(", image_cli)


if __name__ == "__main__":
    unittest.main()
