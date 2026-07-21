from __future__ import annotations

import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import dspy
import pandas as pd
from litellm import ModelResponse


CHAPTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CHAPTER_DIR.parent
PRODUCTION_NOTEBOOKS = (
    "mlflow-tracking.ipynb",
    "fastapi-invoice-api.ipynb",
    "dspyui-gradio.ipynb",
)
CODING_AGENT_NOTEBOOKS = (
    "landing-page-skill-optimizer.ipynb",
    "image-cli-optimizer.ipynb",
    "skill-discovery-rlm.ipynb",
    "test-agents-md.ipynb",
    "clawsona-dspy.ipynb",
)


def load_notebook(filename: str) -> dict:
    return json.loads((CHAPTER_DIR / filename).read_text(encoding="utf-8"))


def notebook_source(filename: str) -> str:
    notebook = load_notebook(filename)
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def notebook_code_source(filename: str) -> str:
    notebook = load_notebook(filename)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def python_source(cell: dict) -> str:
    lines = []
    for line in "".join(cell.get("source", [])).splitlines():
        if line.lstrip().startswith(("%", "!")):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


class ProductionNotebookTest(unittest.TestCase):
    def test_chapter_directories_match_the_reordered_book(self) -> None:
        self.assertEqual(
            sorted(path.name for path in CHAPTER_DIR.glob("*.ipynb")),
            sorted(PRODUCTION_NOTEBOOKS),
        )
        chapter11 = REPO_ROOT / "chapter11"
        for filename in CODING_AGENT_NOTEBOOKS:
            self.assertTrue((chapter11 / filename).is_file(), filename)

        production = "\n".join(notebook_source(name) for name in PRODUCTION_NOTEBOOKS)
        coding_agents = "\n".join(
            (chapter11 / name).read_text(encoding="utf-8")
            for name in CODING_AGENT_NOTEBOOKS
        )
        self.assertNotIn("Chapter 11 §11.", production)
        self.assertNotIn("Chapter 10 §10.", coding_agents)
        self.assertNotIn("§10.", coding_agents)
        self.assertNotIn("–10.", coding_agents)

    def test_every_code_cell_is_valid_python_after_magics_are_removed(self) -> None:
        for filename in PRODUCTION_NOTEBOOKS + CODING_AGENT_NOTEBOOKS:
            directory = CHAPTER_DIR if filename in PRODUCTION_NOTEBOOKS else REPO_ROOT / "chapter11"
            notebook = json.loads((directory / filename).read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                with self.subTest(filename=filename, cell=index):
                    ast.parse(python_source(cell) or "pass")

    def test_documented_model_identifiers_are_preserved(self) -> None:
        source = "\n".join(notebook_source(name) for name in PRODUCTION_NOTEBOOKS)
        for model in (
            "openai/gpt-5.6-luna",
            "openai/gpt-5.6-sol",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-sonnet-5",
            "gemini/gemini-3.5-flash",
        ):
            self.assertIn(model, source)
        for stale_model in (
            "openai/gpt-5-mini",
            "openai/gpt-5.5",
            "openai/gpt-4o-mini",
            "anthropic/claude-haiku-4-5-20251001",
        ):
            self.assertNotIn(stale_model, source)

    def test_pinned_dspy_apis_are_used_in_executable_forms(self) -> None:
        mlflow = notebook_code_source("mlflow-tracking.ipynb")
        fastapi = notebook_source("fastapi-invoice-api.ipynb")
        ui = notebook_source("dspyui-gradio.ipynb")

        self.assertIn('auto="light"', mlflow)
        self.assertIn("reflection_lm=reflection_lm", mlflow)
        self.assertNotIn("max_rounds=3", mlflow)

        self.assertIn("from dspy.streaming import streaming_response", fastapi)
        self.assertNotIn("from dspy.utils.streaming", fastapi)
        self.assertIn("streaming_extractor = dspy.streamify(\n    extractor,", fastapi)
        self.assertNotIn("dspy.streamify(\n    async_extractor,", fastapi)
        self.assertIn("allow_pickle=True", fastapi)

        self.assertIn("dspy.BootstrapFewShotWithRandomSearch", ui)
        self.assertIn("dspy.MIPROv2", ui)
        self.assertIn("dspy.GEPA", ui)
        self.assertIn("save_path.parent.mkdir", ui)

    def test_signature_module_and_dataset_helpers_run_locally(self) -> None:
        notebook = load_notebook("dspyui-gradio.ipynb")
        namespace = {"dspy": dspy}
        for index in (5, 8, 11):
            exec(python_source(notebook["cells"][index]), namespace)

        signature = namespace["create_custom_signature"](
            ["question"], ["answer"], "Answer accurately", ["Input"], ["Output"]
        )
        self.assertIsInstance(namespace["create_module"]("Predict", signature), dspy.Predict)
        with self.assertRaises(ValueError):
            namespace["create_module"]("Unknown", signature)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "examples.csv"
            pd.DataFrame(
                [
                    {"question": "one", "answer": "1"},
                    {"question": "two", "answer": "2"},
                    {"question": "three", "answer": "3"},
                ]
            ).to_csv(csv_path, index=False)
            trainset, devset = namespace["load_and_split"](
                csv_path, ["question"], ["answer"]
            )
            self.assertEqual((len(trainset), len(devset)), (2, 1))

    def test_state_round_trip_and_stream_listener_setup_run_without_paid_calls(self) -> None:
        class InvoiceExtraction(dspy.Signature):
            text: str = dspy.InputField()
            rationale: str = dspy.OutputField()

        extractor = dspy.ChainOfThought(InvoiceExtraction)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "program.json"
            extractor.save(state_path)
            restored = dspy.ChainOfThought(InvoiceExtraction)
            restored.load(state_path)

        stream = dspy.streamify(
            restored,
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="rationale")
            ],
        )
        self.assertTrue(callable(stream))

    def test_redis_cache_example_preserves_litellm_response_type(self) -> None:
        class FakeRedis:
            def __init__(self, *args, **kwargs):
                self.values = {}

            def get(self, key):
                return self.values.get(key)

            def setex(self, key, ttl, value):
                self.values[key] = value

        redis_module = types.ModuleType("redis")
        redis_module.Redis = FakeRedis
        previous_module = sys.modules.get("redis")
        previous_cache = dspy.cache
        sys.modules["redis"] = redis_module
        try:
            notebook = load_notebook("fastapi-invoice-api.ipynb")
            namespace = {"dspy": dspy}
            exec(python_source(notebook["cells"][41]), namespace)
            cache = dspy.cache
            request = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
            response = ModelResponse(
                model="test-model",
                choices=[
                    {"message": {"role": "assistant", "content": "world"}}
                ],
            )
            cache.put(request, response)
            restored = cache.get(request)
            self.assertIsInstance(restored, ModelResponse)
            self.assertEqual(restored.choices[0].message.content, "world")
        finally:
            dspy.cache = previous_cache
            if previous_module is None:
                sys.modules.pop("redis", None)
            else:
                sys.modules["redis"] = previous_module


if __name__ == "__main__":
    unittest.main()
