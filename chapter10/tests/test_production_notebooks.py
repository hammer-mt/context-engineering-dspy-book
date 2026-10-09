"""Offline checks for the Chapter 10 notebooks.

The tests read the notebooks as files and run their self-contained cells with
stand-in models and clients. They make no model calls and need no API keys,
MLflow server, Redis, or Gradio.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import re
import sys
import tempfile
import threading
import types
import unittest
import warnings
from functools import cache
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pandas as pd
from dspy.utils import DummyLM
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
# Model identifiers that Chapter 10 prints, plus the embedding model from
# Chapter 5 that the DSPyUI cosine-similarity metric uses.
BOOK_MODELS = {
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-5",
    "gemini/gemini-3.5-flash",
    "openai/text-embedding-3-small",
}
MODEL_PATTERN = re.compile(r"""["'`]((?:openai|anthropic|gemini)/[\w.\-]+)["'`]""")


@cache
def load_notebook(filename: str, directory: Path = CHAPTER_DIR) -> dict:
    return json.loads((directory / filename).read_text(encoding="utf-8"))


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


def code_cell(filename: str, marker: str, directory: Path = CHAPTER_DIR) -> str:
    """Return the source of the one code cell that contains ``marker``."""
    matches = [
        python_source(cell)
        for cell in load_notebook(filename, directory).get("cells", [])
        if cell.get("cell_type") == "code"
        and marker in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one code cell containing {marker!r} in {filename}, "
            f"found {len(matches)}"
        )
    return matches[0]


def class_source(cell_source: str, class_name: str) -> str:
    for node in ast.parse(cell_source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(cell_source, node)
    raise AssertionError(f"class {class_name} not found")


def collect(stream) -> list:
    async def gather() -> list:
        return [value async for value in stream]

    return asyncio.run(gather())


def run_in_worker_thread(function, *args):
    """Call ``function`` off the main thread, the way Gradio calls a click handler."""
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["value"] = function(*args)
        except BaseException as error:  # Re-raised in the calling thread below.
            outcome["error"] = error

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=120)
    if thread.is_alive():
        raise AssertionError("the handler did not finish")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def stand_in_gradio() -> types.ModuleType:
    """A stand-in for Gradio: components do nothing and ``gr.Error`` is an exception."""

    class Error(Exception):
        pass

    module = types.ModuleType("gradio")
    module.Error = Error
    module.__getattr__ = lambda name: MagicMock(name=f"gradio.{name}")
    return module


class DspyWithStandInModels:
    """The ``dspy`` module, except that ``dspy.LM(...)`` returns an offline model."""

    def __init__(self, lm) -> None:
        self._lm = lm
        self.requested_models: list[str] = []

    def LM(self, model, **kwargs):
        self.requested_models.append(model)
        return self._lm

    def __getattr__(self, name):
        return getattr(dspy, name)


def dspyui_app_namespace(lm) -> dict[str, object]:
    """Run the DSPyUI cells that define the app, with stand-ins for Gradio and the models."""
    namespace: dict[str, object] = {"dspy": DspyWithStandInModels(lm)}
    previous_gradio = sys.modules.get("gradio")
    sys.modules["gradio"] = stand_in_gradio()
    try:
        for marker in (
            "def create_custom_signature",
            "def create_app_signature",
            "MODEL_OPTIONS = [",
            "def create_module",
            "def load_and_split",
            "def load_manual_and_split",
            "def compile_program",
            "def compile_with_any_optimizer",
            "def run_inference",
            "def end_to_end",
        ):
            exec(code_cell("dspyui-gradio.ipynb", marker), namespace)
    finally:
        if previous_gradio is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = previous_gradio
    return namespace


class ProductionNotebookTest(unittest.TestCase):
    def test_expected_notebooks_are_present(self) -> None:
        self.assertEqual(
            sorted(path.name for path in CHAPTER_DIR.glob("*.ipynb")),
            sorted(PRODUCTION_NOTEBOOKS),
        )
        chapter11 = REPO_ROOT / "chapter11"
        for filename in CODING_AGENT_NOTEBOOKS:
            self.assertTrue((chapter11 / filename).is_file(), filename)

    def test_every_code_cell_is_valid_python_after_magics_are_removed(self) -> None:
        for filename in PRODUCTION_NOTEBOOKS + CODING_AGENT_NOTEBOOKS:
            directory = CHAPTER_DIR if filename in PRODUCTION_NOTEBOOKS else REPO_ROOT / "chapter11"
            notebook = json.loads((directory / filename).read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                with self.subTest(filename=filename, cell=index):
                    ast.parse(python_source(cell) or "pass")

    def test_notebooks_use_the_model_identifiers_printed_in_the_book(self) -> None:
        source = "\n".join(notebook_source(name) for name in PRODUCTION_NOTEBOOKS)
        for model in (
            "openai/gpt-5.6-luna",
            "openai/gpt-5.6-sol",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-sonnet-5",
            "gemini/gemini-3.5-flash",
        ):
            self.assertIn(model, source)
        for filename in PRODUCTION_NOTEBOOKS:
            used = set(MODEL_PATTERN.findall(notebook_source(filename)))
            with self.subTest(filename=filename):
                self.assertTrue(used)
                self.assertLessEqual(used, BOOK_MODELS)

    def test_notebooks_use_the_apis_shown_in_the_chapter(self) -> None:
        project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        mlflow = notebook_code_source("mlflow-tracking.ipynb")
        mlflow_all = notebook_source("mlflow-tracking.ipynb")
        fastapi = notebook_source("fastapi-invoice-api.ipynb")
        ui = notebook_source("dspyui-gradio.ipynb")

        self.assertIn('"mlflow>=3.14"', project)
        self.assertIn("pip install 'mlflow>=3.5.1'", mlflow_all)
        self.assertIn("metric=accuracy_with_feedback", mlflow)
        self.assertIn('auto="light"', mlflow)
        self.assertIn("reflection_lm=reflection_lm", mlflow)
        self.assertIn('mlflow.log_metric("accuracy", result.score)', mlflow)
        self.assertIn("pip install 'mlflow[mcp]>=3.5.1'", mlflow_all)
        self.assertIn('--with "mlflow[mcp]>=3.5.1" mlflow mcp run', mlflow_all)
        self.assertIn("log_trace_feedback", mlflow_all)
        self.assertIn("log_trace_expectation", mlflow_all)
        self.assertIn("evaluate_traces", mlflow_all)

        self.assertIn("uvicorn api_server:app --host 127.0.0.1 --port 8000", fastapi)
        self.assertIn("allow_pickle=True", fastapi)
        self.assertIn("from dspy.streaming import streaming_response", fastapi)
        self.assertIn("async def listener_events", fastapi)
        self.assertIn("json.dumps(data)", fastapi)
        self.assertIn("streaming_response(listener_events(stream))", fastapi)
        self.assertIn("streaming_extractor = dspy.streamify(\n    extractor,", fastapi)
        self.assertIn("Start with four for local testing", fastapi)
        self.assertIn("except dspy.LMError:", fastapi)
        self.assertIn("with dspy.context(lm=fallback_lm):", fastapi)
        self.assertIn("cache_control_injection_points", fastapi)

        self.assertIn("dspy.BootstrapFewShot(metric=metric)", ui)
        self.assertIn("dspy.BootstrapFewShotWithRandomSearch", ui)
        self.assertIn("dspy.MIPROv2", ui)
        self.assertIn("dspy.GEPA", ui)
        self.assertIn('program_path = Path("programs/latest.json")', ui)
        self.assertIn("program_path.parent.mkdir", ui)
        self.assertIn("METRIC_OPTIONS", ui)
        self.assertIn("Cosine Similarity", ui)
        self.assertIn("LLM-as-a-Judge", ui)
        self.assertIn("load_manual_and_split", ui)
        self.assertIn('gr.Tab("Inference")', ui)
        self.assertIn("run_saved_program", ui)
        self.assertIn("with dspy.context(lm=dspy.LM(student_model)):", ui)
        for heading in (
            "## Signatures",
            "## Modules",
            "## Datasets",
            "## Optimization",
            "## Inference",
        ):
            self.assertIn(heading, ui)

    def test_mlflow_mcp_config_block_is_valid_json(self) -> None:
        mlflow_all = notebook_source("mlflow-tracking.ipynb")
        block = re.search(r"```json\n(.*?)```", mlflow_all, re.DOTALL)
        self.assertIsNotNone(block)
        config = json.loads(block.group(1))
        server = config["mcpServers"]["mlflow-mcp"]
        self.assertEqual(server["command"], "uv")
        self.assertEqual(
            server["args"],
            ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
        )
        self.assertEqual(
            server["env"], {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5000"}
        )

    def test_mlflow_supporting_metrics_work_with_gepa(self) -> None:
        namespace = {"dspy": dspy}
        exec(
            code_cell("mlflow-tracking.ipynb", "def accuracy_with_feedback"),
            namespace,
        )
        self.assertEqual(len(namespace["train_examples"]), 3)
        self.assertEqual(len(namespace["valset"]), 2)

        example = namespace["train_examples"][0]
        right = dspy.Prediction(sentiment=example.sentiment.upper())
        wrong = dspy.Prediction(sentiment="neutral")
        self.assertEqual(namespace["accuracy"](example, right), 1.0)
        self.assertEqual(namespace["accuracy"](example, wrong), 0.0)

        graded = namespace["accuracy_with_feedback"](example, wrong)
        self.assertEqual(graded.score, 0.0)
        self.assertIn(example.sentiment, graded.feedback)
        self.assertEqual(
            namespace["accuracy_with_feedback"](example, right).score, 1.0
        )

        # GEPA checks the metric's five-argument signature when it is created.
        optimizer = dspy.GEPA(
            metric=namespace["accuracy_with_feedback"],
            auto="light",
            reflection_lm=namespace["reflection_lm"],
        )
        self.assertIsInstance(optimizer, dspy.GEPA)

    def test_signature_module_and_dataset_helpers_run_locally(self) -> None:
        namespace = {"dspy": dspy}
        for marker in (
            "def create_custom_signature",
            "def create_module",
            "def load_and_split",
            "def load_manual_and_split",
        ):
            exec(code_cell("dspyui-gradio.ipynb", marker), namespace)

        signature = namespace["create_custom_signature"](
            ["question"], ["answer"], "Answer accurately", ["Input"], ["Output"]
        )
        self.assertEqual(list(signature.input_fields), ["question"])
        self.assertEqual(list(signature.output_fields), ["answer"])
        self.assertEqual(signature.instructions, "Answer accurately")
        self.assertIsInstance(namespace["create_module"]("Predict", signature), dspy.Predict)
        self.assertIsInstance(
            namespace["create_module"]("ChainOfThought", signature),
            dspy.ChainOfThought,
        )
        # The factory in the book handles the two module names the UI offers.
        self.assertIsNone(namespace["create_module"]("Unknown", signature))

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
            self.assertEqual(trainset[0].inputs().keys(), ["question"])

            manual_rows = pd.DataFrame([["four", "4"], ["five", "5"]])
            trainset, devset = namespace["load_manual_and_split"](
                manual_rows, ["question"], ["answer"]
            )
            self.assertEqual((len(trainset), len(devset)), (1, 1))
            with self.assertRaises(ValueError):
                namespace["load_manual_and_split"](
                    pd.DataFrame([["only", "1"]]), ["question"], ["answer"]
                )

    def test_compile_and_inference_helpers_round_trip_a_saved_program(self) -> None:
        namespace = {"dspy": dspy}
        for marker in (
            "def create_custom_signature",
            "def create_module",
            "def compile_program",
            "def compile_with_any_optimizer",
            "def run_inference",
        ):
            exec(code_cell("dspyui-gradio.ipynb", marker), namespace)

        signature = namespace["create_custom_signature"](
            ["question"], ["answer"], "Answer with a digit", [], []
        )
        module = namespace["create_module"]("Predict", signature)
        trainset = [
            dspy.Example(question="one", answer="1").with_inputs("question"),
            dspy.Example(question="two", answer="1").with_inputs("question"),
        ]
        devset = [dspy.Example(question="three", answer="1").with_inputs("question")]

        def exact_match(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return float(pred.answer == gold.answer)

        lm = DummyLM([{"answer": "1"}] * 20)
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                with dspy.context(lm=lm):
                    baseline, final, compiled = namespace["compile_with_any_optimizer"](
                        module, trainset, devset, "BootstrapFewShot", exact_match, lm
                    )
                    self.assertTrue(Path("programs/latest.json").is_file())
                    result, messages = namespace["run_inference"](
                        "programs/latest.json", signature, "Predict", {"question": "four"}
                    )
                    with self.assertRaises(ValueError):
                        namespace["compile_with_any_optimizer"](
                            module, trainset, devset, "Unknown", exact_match, lm
                        )
            finally:
                os.chdir(previous_directory)

        # dspy.Evaluate reports scores on a 0-100 scale.
        self.assertEqual((baseline, final), (100.0, 100.0))
        self.assertIsInstance(compiled, dspy.Predict)
        self.assertEqual(result.answer, "1")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("four", messages[-1]["content"])

    def test_blank_field_descriptions_stay_out_of_the_app_prompt(self) -> None:
        namespace = {"dspy": dspy}
        for marker in ("def create_custom_signature", "def create_app_signature"):
            exec(code_cell("dspyui-gradio.ipynb", marker), namespace)

        def field_lines(build, descriptions: list[str]) -> list[str]:
            signature = build(
                ["joke_topic"],
                ["funny_rating"],
                "Rate how funny this joke is on a scale of 1 to 10",
                descriptions,
                descriptions,
            )
            prompt = dspy.ChatAdapter().format(
                signature, demos=[], inputs={"joke_topic": "cats"}
            )[0]["content"]
            return [line for line in prompt.splitlines() if "(str)" in line]

        # The listing on pages 275-276 passes desc=None for a blank
        # description, and DSPy 3.3.0 then writes the word None into the prompt.
        printed = field_lines(namespace["create_custom_signature"], [""])
        self.assertEqual(
            printed, ["1. `joke_topic` (str): None", "1. `funny_rating` (str): None"]
        )
        # The app has no description boxes, so it builds signatures with the
        # variant that leaves `desc` out when a description is blank.
        app = field_lines(namespace["create_app_signature"], [""])
        self.assertEqual(len(app), 2)
        for line in app:
            self.assertNotIn("None", line)
        described = field_lines(namespace["create_app_signature"], ["Be specific"])
        self.assertEqual(
            described,
            [
                "1. `joke_topic` (str): Be specific",
                "1. `funny_rating` (str): Be specific",
            ],
        )

        app_cell = ast.parse(code_cell("dspyui-gradio.ipynb", "def end_to_end"))
        for handler in ("end_to_end", "run_saved_program"):
            function = next(
                node
                for node in app_cell.body
                if isinstance(node, ast.FunctionDef) and node.name == handler
            )
            called = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            with self.subTest(handler=handler):
                self.assertIn("create_app_signature", called)
                self.assertNotIn("create_custom_signature", called)

    def test_dspyui_configures_dspy_only_in_the_setup_cell(self) -> None:
        # DSPy lets only the thread that first called dspy.configure call it
        # again, and Gradio runs click handlers in worker threads. A handler
        # that called dspy.configure would fail when the reader clicks Compile.
        def configure_calls(tree: ast.AST) -> int:
            return sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "configure"
                for node in ast.walk(tree)
            )

        notebook = load_notebook("dspyui-gradio.ipynb")
        total = 0
        functions_that_configure = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            tree = ast.parse(python_source(cell) or "pass")
            total += configure_calls(tree)
            functions_that_configure += [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and configure_calls(node)
            ]
        self.assertEqual(functions_that_configure, [])
        self.assertEqual(total, 1)

    def test_dspyui_handlers_run_in_a_worker_thread(self) -> None:
        lm = DummyLM([{"answer": "1"}] * 40)
        namespace = dspyui_app_namespace(lm)
        manual_rows = pd.DataFrame(
            [
                ["one", "1", ""],
                ["two", "1", ""],
                ["three", "1", ""],
                ["four", "1", ""],
                ["five", "1", ""],
                ["", "", ""],  # A row left blank in the table is ignored.
            ]
        )

        # Make the main thread the one that configured DSPy, as the notebook's
        # setup cell does, and confirm the rule the handlers have to respect.
        dspy.configure(lm=dspy.settings.lm)
        with self.assertRaisesRegex(RuntimeError, "thread that initially configured"):
            run_in_worker_thread(lambda: dspy.configure(lm=lm))

        previous_directory = Path.cwd()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder-for-offline-test"}),
        ):
            os.chdir(tmpdir)
            try:
                status, state = run_in_worker_thread(
                    namespace["end_to_end"],
                    "Answer with a digit",
                    "question",
                    "answer",
                    "Predict",
                    "openai/gpt-5.6-luna",
                    "openai/gpt-5.6-sol",
                    "BootstrapFewShot",
                    "Exact Match",
                    None,
                    manual_rows,
                )
                self.assertTrue(Path("programs/latest.json").is_file())
                prediction, messages = run_in_worker_thread(
                    namespace["run_saved_program"], state, '{"question": "six"}'
                )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(status, "Baseline: 100.00%, Optimized: 100.00%")
        self.assertEqual(state["student_model"], "openai/gpt-5.6-luna")
        self.assertEqual(state["program_path"], "programs/latest.json")
        self.assertEqual(prediction, {"answer": "1"})
        self.assertEqual(messages[0]["role"], "system")
        self.assertNotIn("(str): None", messages[0]["content"])
        self.assertIn("six", messages[-1]["content"])
        # The student and teacher models are the ones chosen in the form.
        self.assertEqual(
            set(namespace["dspy"].requested_models),
            {"openai/gpt-5.6-luna", "openai/gpt-5.6-sol"},
        )

    def test_dspyui_reports_a_missing_api_key_before_any_model_call(self) -> None:
        lm = DummyLM([{"answer": "1"}] * 4)
        namespace = dspyui_app_namespace(lm)
        manual_rows = pd.DataFrame([["one", "1"], ["two", "1"]])
        arguments = [
            "Answer with a digit",
            "question",
            "answer",
            "Predict",
            "anthropic/claude-opus-4.8",
            "openai/gpt-5.6-sol",
            "BootstrapFewShot",
            "Exact Match",
            None,
            manual_rows,
        ]
        environment = {
            name: value
            for name, value in os.environ.items()
            if name not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
        }
        # The app raises gr.Error, which Gradio shows to the reader as a message.
        # The call runs in a temporary folder so that nothing is written into
        # the repository if the check ever lets the compile step start.
        previous_directory = Path.cwd()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, environment, clear=True),
        ):
            os.chdir(tmpdir)
            try:
                with self.assertRaises(namespace["gr"].Error) as raised:
                    namespace["end_to_end"](*arguments)
            finally:
                os.chdir(previous_directory)
        self.assertIn("anthropic/claude-opus-4.8", str(raised.exception))
        self.assertIn("ANTHROPIC_API_KEY", str(raised.exception))
        self.assertEqual(namespace["dspy"].requested_models, [])
        self.assertEqual(lm.history, [])

        # Each model in the app's list is satisfied by the variable the
        # Preface names for its provider.
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        for model in namespace["MODEL_OPTIONS"]:
            key = provider_keys[model.split("/")[0]]
            with (
                self.subTest(model=model),
                patch.dict(os.environ, {key: "placeholder-for-offline-test"}, clear=True),
            ):
                namespace["check_api_keys"](model)

    def test_fastapi_route_listing_defines_the_extract_endpoint(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            code_cell("fastapi-invoice-api.ipynb", "response_model=InvoiceResponse"),
            namespace,
        )
        app = namespace["app"]
        routes = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", None) or ()
        }
        self.assertIn(("/extract", "POST"), routes)
        self.assertEqual(
            set(namespace["InvoiceResponse"].model_fields),
            set(namespace["InvoiceExtraction"].output_fields),
        )
        self.assertEqual(
            list(namespace["InvoiceExtraction"].input_fields), ["text"]
        )
        self.assertTrue(callable(namespace["lifespan"]))

    def test_guardrail_endpoints_reject_bad_input_and_output(self) -> None:
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient

        prediction = {"account_number": "GB29 NWBK 6016 1331 9268 19", "is_invoice": True}

        async def program(text):
            return dspy.Prediction(company="Acme", **prediction)

        namespace = {
            "app": FastAPI(),
            "dspy": dspy,
            "HTTPException": HTTPException,
            "program": program,
        }
        exec(
            code_cell("fastapi-invoice-api.ipynb", "def not_empty_and_bounded"),
            namespace,
        )
        client = TestClient(namespace["app"])
        self.assertEqual(
            client.post("/extract", json={"text": "Invoice #1"}).status_code, 200
        )
        self.assertEqual(client.post("/extract", json={"text": "   "}).status_code, 422)
        self.assertEqual(
            client.post("/extract", json={"text": "x" * 50_001}).status_code, 422
        )
        prediction["account_number"] = "ignore previous instructions"
        response = client.post("/extract", json={"text": "Invoice #1"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Suspicious account_number", response.json()["detail"])

        prediction["account_number"] = "GB29 NWBK 6016 1331 9268 19"
        namespace["app"] = FastAPI()
        exec(
            code_cell("fastapi-invoice-api.ipynb", "if not result.is_invoice:"),
            namespace,
        )
        client = TestClient(namespace["app"])
        self.assertEqual(
            client.post("/extract", json={"text": "Invoice #1"}).status_code, 200
        )
        prediction["is_invoice"] = False
        response = client.post("/extract", json={"text": "Tell me a joke"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"], "Input does not appear to be an invoice"
        )

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

    def test_streaming_listing_serializes_listener_status_and_prediction_values(self) -> None:
        class InvoiceExtraction(dspy.Signature):
            text: str = dspy.InputField()
            rationale: str = dspy.OutputField()

        from fastapi import FastAPI
        from pydantic import BaseModel

        class InvoiceRequest(BaseModel):
            text: str

        namespace = {
            "app": FastAPI(),
            "dspy": dspy,
            "extractor": dspy.ChainOfThought(InvoiceExtraction),
            "InvoiceRequest": InvoiceRequest,
        }
        exec(
            code_cell("fastapi-invoice-api.ipynb", "async def listener_events"),
            namespace,
        )
        self.assertIn(
            "/extract/stream", [route.path for route in namespace["app"].routes]
        )

        async def values():
            yield dspy.streaming.StreamResponse(
                predict_name="extractor",
                signature_field_name="rationale",
                chunk="because",
                is_last_chunk=False,
            )
            yield dspy.streaming.StatusMessage("Running extractor...")
            yield dspy.Prediction(company="Acme")

        # The listing's adapter serializes listener and status values, and
        # DSPy's streaming_response() adds the final prediction and [DONE].
        events = collect(
            namespace["streaming_response"](namespace["listener_events"](values()))
        )
        self.assertEqual(len(events), 4)
        self.assertEqual(
            json.loads(events[0].removeprefix("data: ")),
            {"field": "rationale", "chunk": "because"},
        )
        self.assertEqual(
            json.loads(events[1].removeprefix("data: ")),
            {"status": "Running extractor..."},
        )
        self.assertEqual(
            json.loads(events[2].removeprefix("data: ")),
            {"prediction": {"company": "Acme"}},
        )
        self.assertEqual(events[-1], "data: [DONE]\n\n")

        exec(code_cell("fastapi-invoice-api.ipynb", "class ExtractionStatus"), namespace)
        provider = namespace["ExtractionStatus"]()
        self.assertEqual(
            provider.module_start_status_message(namespace["extractor"], {}),
            "Running ChainOfThought...",
        )
        self.assertTrue(callable(namespace["streaming_extractor"]))

    def test_fallback_endpoint_switches_provider_only_for_lm_errors(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from pydantic import BaseModel

        class InvoiceRequest(BaseModel):
            text: str

        calls: list[str] = []
        failure: dict[str, Exception] = {}

        async def async_extractor(text):
            active = dspy.settings.lm
            calls.append(active.model)
            if active is namespace["lm"]:
                raise failure["error"]
            return dspy.Prediction(company="Acme")

        namespace = {
            "app": FastAPI(),
            "dspy": dspy,
            "InvoiceRequest": InvoiceRequest,
            "async_extractor": async_extractor,
        }
        previous_lm = dspy.settings.lm
        try:
            exec(
                code_cell("fastapi-invoice-api.ipynb", "except dspy.LMError:"),
                namespace,
            )
            client = TestClient(namespace["app"])

            failure["error"] = dspy.LMError("primary provider unavailable")
            response = client.post("/extract", json={"text": "Invoice #1"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"company": "Acme"})
            self.assertEqual(
                calls, ["openai/gpt-5.6-sol", "anthropic/claude-opus-4.8"]
            )

            # A bug in application code must not be retried on the fallback.
            calls.clear()
            failure["error"] = KeyError("unrelated application failure")
            with self.assertRaises(KeyError):
                client.post("/extract", json={"text": "Invoice #1"})
            self.assertEqual(calls, ["openai/gpt-5.6-sol"])
        finally:
            dspy.configure(lm=previous_lm)

    def test_markdown_adapter_formats_and_parses_markdown_headings(self) -> None:
        cell = code_cell("fastapi-invoice-api.ipynb", "class MarkdownAdapter")
        namespace: dict[str, object] = {}
        exec(cell, namespace)
        adapter = namespace["MarkdownAdapter"]()

        class Example(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        messages = adapter.format(Example, demos=[], inputs={"question": "Why?"})
        system = messages[0]["content"]
        user = messages[-1]["content"]
        assistant = adapter.format_assistant_message_content(
            Example, {"answer": "Because."}
        )

        self.assertIn("## answer", system)
        self.assertNotIn("[[ ## answer ## ]]", system)
        self.assertIn("## answer\nBecause.", assistant)
        self.assertNotIn("[[ ## answer ## ]]", assistant)
        self.assertIn(
            "Respond using Markdown headings for: `## answer`, "
            "then end with `## completed`.",
            user,
        )
        # With the pinned DSPy, the input field of a user message and the
        # completion marker keep DSPy's bracket form.
        self.assertIn("[[ ## question ## ]]", user)
        self.assertIn("[[ ## completed ## ]]", assistant)

        class TwoOutputs(dspy.Signature):
            answer: str = dspy.OutputField()
            confidence: float = dspy.OutputField()

        parsed = adapter.parse(
            TwoOutputs,
            "## answer\nBecause.\n\n## confidence\n0.85\n\n## completed",
        )
        self.assertEqual(parsed, {"answer": "Because.", "confidence": 0.85})
        self.assertIsInstance(parsed["confidence"], float)

        # The class is the one built in Chapter 7, "Building Custom Adapters".
        chapter7_cell = code_cell(
            "adapters.ipynb", "class MarkdownAdapter", REPO_ROOT / "chapter07"
        )
        self.assertEqual(
            class_source(cell, "MarkdownAdapter"),
            class_source(chapter7_cell, "MarkdownAdapter"),
        )

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
            namespace = {"dspy": dspy}
            exec(code_cell("fastapi-invoice-api.ipynb", "class RedisCache"), namespace)
            cache = dspy.cache
            self.assertIsInstance(cache, namespace["RedisCache"])
            self.assertEqual(cache.ttl, 24 * 3600)
            request = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
            self.assertIsNone(cache.get(request))
            response = ModelResponse(
                model="test-model",
                choices=[
                    {"message": {"role": "assistant", "content": "world"}}
                ],
            )
            with warnings.catch_warnings():
                # model_dump() on a LiteLLM response can emit a Pydantic
                # serializer warning; the cached value is unaffected.
                warnings.simplefilter("ignore", UserWarning)
                cache.put(request, response)
            restored = cache.get(request)
            self.assertIsInstance(restored, ModelResponse)
            self.assertEqual(restored.choices[0].message.content, "world")
            self.assertEqual(restored.usage, {})
            self.assertTrue(restored.cache_hit)
            with self.assertRaises(TypeError):
                cache.put(request, {"not": "a ModelResponse"})
        finally:
            dspy.cache = previous_cache
            if previous_module is None:
                sys.modules.pop("redis", None)
            else:
                sys.modules["redis"] = previous_module


class RelatedChapterNotebookTest(unittest.TestCase):
    """Checks on RLM examples in other chapters that Chapter 10 refers to."""

    def test_rlm_examples_use_the_iteration_limit_dspy_accepts(self) -> None:
        self.assertIn("max_iters", inspect.signature(dspy.RLM.__init__).parameters)
        for path in (
            REPO_ROOT / "chapter07" / "codeact-and-rlm.ipynb",
            REPO_ROOT / "chapter09" / "financial-analyst.ipynb",
            REPO_ROOT / "chapter11" / "skill-discovery-rlm.ipynb",
        ):
            with self.subTest(notebook=path.name):
                self.assertIn("max_iters", path.read_text(encoding="utf-8"))

    def test_skill_discovery_rlm_passes_session_texts_directly(self) -> None:
        discovery = (REPO_ROOT / "chapter11" / "skill-discovery-rlm.ipynb").read_text(
            encoding="utf-8"
        )
        self.assertIn("session_texts: list[str]", discovery)
        self.assertIn("result = rlm(session_texts=session_texts)", discovery)
        self.assertIn('sub_lm = dspy.LM(\\"openai/gpt-5.6-luna\\")', discovery)


if __name__ == "__main__":
    unittest.main()
