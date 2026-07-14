"""Validate Chapter 6 notebook syntax, cleanliness, and optimizer coverage."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from chapter06.build_optimizer_notebooks import NOTEBOOKS


CHAPTER_DIR = Path(__file__).resolve().parent


def _python_source(cell: dict) -> str:
    return "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in "".join(cell.get("source", [])).splitlines()
    )


def validate_notebook(path: Path, expected_optimizer: str) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        return [f"{path.name}: expected notebook format 4"]
    source = "".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    if repr(expected_optimizer) not in source:
        errors.append(f"{path.name}: missing optimizer identifier {expected_optimizer!r}")
    if "run_experiment" not in source:
        errors.append(f"{path.name}: missing shared experiment runner")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            errors.append(f"{path.name}: cell {index} contains execution state")
        try:
            ast.parse(_python_source(cell) or "pass", filename=f"{path.name}:cell{index}")
        except SyntaxError as exc:
            errors.append(f"{path.name}: cell {index}: {exc.msg}")
    return errors


def main() -> None:
    errors: list[str] = []
    for filename, (_, _, optimizer) in NOTEBOOKS.items():
        errors.extend(validate_notebook(CHAPTER_DIR / filename, optimizer))
    extra = sorted(path.name for path in CHAPTER_DIR.glob("*.ipynb") if path.name not in NOTEBOOKS)
    if extra:
        errors.append(f"unexpected Chapter 6 notebooks: {extra}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(NOTEBOOKS)} output-free Chapter 6 notebooks.")


if __name__ == "__main__":
    main()
