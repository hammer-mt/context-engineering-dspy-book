"""Static compatibility checks for the generated Chapter 6 notebooks."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import dspy


CHAPTER_DIR = Path(__file__).resolve().parent


def python_source(cell: dict) -> str:
    raw = "".join(cell.get("source", []))
    # IPython install/shell lines are valid notebook syntax but not Python AST syntax.
    return "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in raw.splitlines()
    )


def validate_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: invalid JSON: {exc}"]

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        errors.append(f"{path.name}: expected notebook format 4 with a cells list")
        return errors

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            errors.append(f"{path.name}: cell {index} contains execution state or output")
        try:
            ast.parse(python_source(cell) or "pass", filename=f"{path.name}:cell{index}")
        except SyntaxError as exc:
            errors.append(f"{path.name}: cell {index}: {exc.msg} (line {exc.lineno})")
    return errors


def validate_dspy_api() -> list[str]:
    errors: list[str] = []
    major, minor = map(int, dspy.__version__.split(".")[:2])
    if (major, minor) < (3, 2):
        errors.append(f"DSPy {dspy.__version__} is too old; Chapter 6 targets DSPy 3.2+")

    knn_init = inspect.signature(dspy.KNNFewShot)
    for required in ("k", "trainset", "vectorizer"):
        if required not in knn_init.parameters:
            errors.append(f"KNNFewShot constructor is missing expected argument {required!r}")
    if "trainset" in inspect.signature(dspy.KNNFewShot.compile).parameters:
        errors.append("KNNFewShot.compile unexpectedly accepts trainset; review the notebook API example")
    if "num_threads" in inspect.signature(dspy.BootstrapFewShot).parameters:
        errors.append("BootstrapFewShot now accepts num_threads; review the KNN warning and example")
    if "size" in inspect.signature(dspy.Ensemble.compile).parameters:
        errors.append("Ensemble.compile now accepts size; review where the notebook supplies it")

    knn_notebook = json.loads((CHAPTER_DIR / "knn-few-shot.ipynb").read_text(encoding="utf-8"))
    optimizer_tree = None
    for cell in knn_notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse(python_source(cell) or "pass")
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "KNNFewShot"
            for node in ast.walk(tree)
        ):
            optimizer_tree = tree
            break
    if optimizer_tree is None:
        errors.append("KNN notebook does not construct KNNFewShot")
        return errors

    knn_calls = [
        node
        for node in ast.walk(optimizer_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "KNNFewShot"
    ]
    if any(keyword.arg == "num_threads" for keyword in knn_calls[0].keywords):
        errors.append("KNN notebook passes forbidden num_threads to KNNFewShot")

    compile_calls = [
        node
        for node in ast.walk(optimizer_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "optimizer"
        and node.func.attr == "compile"
    ]
    if len(compile_calls) != 1 or len(compile_calls[0].args) != 1 or compile_calls[0].keywords:
        errors.append("KNN notebook must compile with only the student")
    return errors


def main() -> None:
    notebooks = sorted(CHAPTER_DIR.glob("*.ipynb"))
    errors = [error for path in notebooks for error in validate_notebook(path)]
    errors.extend(validate_dspy_api())
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(notebooks)} Chapter 6 notebooks against DSPy {dspy.__version__}.")


if __name__ == "__main__":
    main()
