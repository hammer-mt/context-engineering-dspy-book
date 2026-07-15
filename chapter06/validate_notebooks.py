"""Validate Chapter 6 notebook education, safe defaults, outputs, and coverage."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Sequence

from chapter06.build_optimizer_notebooks import NOTEBOOKS


CHAPTER_DIR = Path(__file__).resolve().parent


def _python_source(cell: dict) -> str:
    return "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in "".join(cell.get("source", [])).splitlines()
    )


def validate_notebook(
    path: Path, expected_optimizer: str, *, require_executed: bool = True
) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        return [f"{path.name}: expected notebook format 4"]
    cells = notebook.get("cells", [])
    source = "".join("".join(cell.get("source", [])) for cell in cells)
    required_lessons = (
        "Use it when",
        "What compilation changes",
        "Compile shape",
        "Read the result",
        "Run it yourself",
        "CHAPTER06_RUN=full",
        "benchmark_snapshot",
        "learned_program_preview",
        "verify_prompt_artifact",
        "run_experiment",
    )
    if repr(expected_optimizer) not in source:
        errors.append(
            f"{path.name}: missing optimizer identifier {expected_optimizer!r}"
        )
    for lesson in required_lessons:
        if lesson not in source:
            errors.append(f"{path.name}: missing educational/run content {lesson!r}")
    if 'os.getenv("CHAPTER06_RUN", "inspect")' not in source:
        errors.append(f"{path.name}: safe inspect mode is not the default")
    if len(cells) < 8:
        errors.append(f"{path.name}: expected at least eight concise teaching cells")
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if len(code_cells) < 4:
        errors.append(
            f"{path.name}: expected setup, inspection, prompt, and opt-in run cells"
        )
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if require_executed:
            if not isinstance(cell.get("execution_count"), int):
                errors.append(f"{path.name}: cell {index} was not executed")
            if not cell.get("outputs"):
                errors.append(f"{path.name}: cell {index} has no saved output")
        try:
            ast.parse(
                _python_source(cell) or "pass", filename=f"{path.name}:cell{index}"
            )
        except SyntaxError as exc:
            errors.append(f"{path.name}: cell {index}: {exc.msg}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unexecuted",
        action="store_true",
        help="check generated source before execution; publishing validation requires outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    errors: list[str] = []
    for filename, spec in NOTEBOOKS.items():
        errors.extend(
            validate_notebook(
                CHAPTER_DIR / filename,
                spec["optimizer"],
                require_executed=not args.allow_unexecuted,
            )
        )
    extra = sorted(
        path.name for path in CHAPTER_DIR.glob("*.ipynb") if path.name not in NOTEBOOKS
    )
    if extra:
        errors.append(f"unexpected Chapter 6 notebooks: {extra}")
    if errors:
        raise SystemExit("\n".join(errors))
    state = "executed" if not args.allow_unexecuted else "generated"
    print(f"Validated {len(NOTEBOOKS)} {state} educational Chapter 6 notebooks.")


if __name__ == "__main__":
    main()
