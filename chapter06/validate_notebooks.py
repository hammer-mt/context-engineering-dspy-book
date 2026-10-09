"""Check the generated Chapter 6 notebooks: content, book listings, and saved outputs."""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import symtable
from pathlib import Path
from typing import Sequence

from chapter06.build_optimizer_notebooks import LISTINGS, NOTEBOOKS, make_notebook


CHAPTER_DIR = Path(__file__).resolve().parent
SUPPORT_NOTEBOOKS = {"gepa-expanded-dataset-experiment.ipynb"}
REQUIRED_CONTENT = (
    "Use it when",
    "What compilation changes",
    "Saved result",
    "Read the result",
    "Apply the pattern",
    "load_frozen_examples",
    "split_summary",
    "published_result",
    "CHAPTER06_RUN_LIVE",
    "learned_program_preview",
    "verify_prompt_artifact",
)
# The notebooks run the printed listings; they never call the batch runner.
FORBIDDEN_CODE = ("run_optimizer(",)
LOCAL_PATH_PATTERN = re.compile(r"(/Users/|/home/|/private/|/var/folders/|[A-Za-z]:\\Users\\)")
MINIMUM_PYTHON = (3, 12)


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _python_source(cell: dict) -> str:
    return "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in _source(cell).splitlines()
    )


def contains_listing(listing: str, source: str) -> bool:
    """True when ``listing`` appears line for line in ``source``.

    A uniform extra indentation is allowed, which is how a listing sits under
    the ``if RUN_LIVE:`` guard.
    """

    wanted = listing.strip("\n").split("\n")
    lines = source.split("\n")
    stripped = [line.strip() for line in lines]
    target = [line.strip() for line in wanted]
    for start in range(len(lines) - len(wanted) + 1):
        if stripped[start : start + len(wanted)] != target:
            continue
        offsets = {
            (len(lines[start + index]) - len(lines[start + index].lstrip()))
            - (len(wanted[index]) - len(wanted[index].lstrip()))
            for index in range(len(wanted))
            if target[index]
        }
        if len(offsets) <= 1:
            return True
    return False


def _mode_source(cell_source: str, *, live: bool) -> str:
    """Return the code a cell runs in one mode.

    Top-level ``if RUN_LIVE:`` statements are replaced by the branch that
    executes: the body in live mode, the ``else`` branch in the default mode.
    """

    statements: list[ast.stmt] = []
    for node in ast.parse(cell_source).body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "RUN_LIVE"
        ):
            statements.extend(node.body if live else node.orelse)
        else:
            statements.append(node)
    return ast.unparse(ast.Module(body=statements, type_ignores=[]))


def _bound_and_used(source: str) -> tuple[set[str], set[str]]:
    """Names a cell binds at module level, and global names it reads."""

    bound: set[str] = set()
    used: set[str] = set()

    def visit(table: symtable.SymbolTable, *, top: bool) -> None:
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if top:
                if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace():
                    bound.add(name)
                if symbol.is_referenced():
                    used.add(name)
            elif symbol.is_global() and symbol.is_referenced():
                used.add(name)
        for child in table.get_children():
            visit(child, top=False)

    visit(symtable.symtable(source, "<cell>", "exec"), top=True)
    return bound, used


def undefined_names(code_sources: Sequence[str], *, live: bool) -> list[tuple[int, str]]:
    """Find names a notebook reads before any earlier cell has defined them.

    Returns ``(code cell number, name)`` pairs for the given mode. This is a
    static check, so it needs no kernel and makes no model call.
    """

    defined = set(dir(builtins))
    problems: list[tuple[int, str]] = []
    for number, cell_source in enumerate(code_sources):
        bound, used = _bound_and_used(_mode_source(cell_source, live=live))
        problems.extend(
            (number, name) for name in sorted(used - bound - defined)
        )
        defined |= bound
    return problems


def _listing_keys(spec: dict) -> list[str]:
    keys = ["load-dataset"]
    keys.extend(item["key"] for item in spec["listings"])
    keys.extend(item["key"] for item in spec.get("extras", []))
    return keys


def _output_text(output: dict) -> str:
    parts: list[str] = []
    for key in ("text", "traceback", "evalue"):
        value = output.get(key, "")
        parts.append("".join(value) if isinstance(value, list) else str(value))
    for value in output.get("data", {}).values():
        parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def validate_notebook(
    path: Path, expected_optimizer: str, *, require_executed: bool = True
) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        return [f"{path.name}: expected notebook format 4"]
    cells = notebook.get("cells", [])
    source = "".join(_source(cell) for cell in cells)
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    code_sources = [_source(cell) for cell in code_cells]

    if repr(expected_optimizer) not in source:
        errors.append(
            f"{path.name}: missing optimizer identifier {expected_optimizer!r}"
        )
    for content in REQUIRED_CONTENT:
        if content not in source:
            errors.append(f"{path.name}: missing content {content!r}")
    for forbidden in FORBIDDEN_CODE:
        if any(forbidden in code for code in code_sources):
            errors.append(f"{path.name}: code cells must not call {forbidden!r}")

    spec = NOTEBOOKS.get(path.name)
    if spec is not None:
        generated = make_notebook(spec)["cells"]
        observed = [(cell.get("cell_type"), _source(cell)) for cell in cells]
        expected = [(cell["cell_type"], _source(cell)) for cell in generated]
        if observed != expected:
            errors.append(
                f"{path.name}: differs from the generator; regenerate it with "
                "`python -m chapter06.build_optimizer_notebooks`"
            )
        for key in _listing_keys(spec):
            if not any(contains_listing(LISTINGS[key], code) for code in code_sources):
                errors.append(
                    f"{path.name}: book listing {key!r} is not present verbatim in a code cell"
                )

    try:
        python_sources = [_python_source(cell) for cell in code_cells]
        for live in (False, True):
            mode = "live" if live else "default"
            for number, name in undefined_names(python_sources, live=live):
                errors.append(
                    f"{path.name}: code cell {number} uses {name!r} before it is "
                    f"defined in {mode} mode"
                )
    except SyntaxError:
        pass  # reported per cell below

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        cell_source = _source(cell)
        if require_executed:
            if not isinstance(cell.get("execution_count"), int):
                errors.append(f"{path.name}: cell {index} was not executed")
            # Cells that print must keep their saved output; a listing that only
            # compiles (for example LabeledFewShot) legitimately prints nothing.
            if "print(" in cell_source and not cell.get("outputs"):
                errors.append(f"{path.name}: cell {index} has no saved output")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"{path.name}: cell {index} saved an error output")
            if LOCAL_PATH_PATTERN.search(_output_text(output)):
                errors.append(
                    f"{path.name}: cell {index} output contains a local filesystem path"
                )
        try:
            ast.parse(
                _python_source(cell) or "pass", filename=f"{path.name}:cell{index}"
            )
        except SyntaxError as exc:
            errors.append(f"{path.name}: cell {index}: {exc.msg}")

    version = notebook.get("metadata", {}).get("language_info", {}).get("version")
    if version:
        parts = tuple(int(part) for part in re.findall(r"\d+", version)[:2])
        if parts < MINIMUM_PYTHON:
            errors.append(
                f"{path.name}: saved with Python {version}; execute it with Python "
                f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer"
            )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unexecuted",
        action="store_true",
        help="check generated source without requiring saved outputs",
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
        path.name
        for path in CHAPTER_DIR.glob("*.ipynb")
        if path.name not in NOTEBOOKS and path.name not in SUPPORT_NOTEBOOKS
    )
    if extra:
        errors.append(f"unexpected Chapter 6 notebooks: {extra}")
    if errors:
        raise SystemExit("\n".join(errors))
    state = "executed" if not args.allow_unexecuted else "generated"
    print(f"Validated {len(NOTEBOOKS)} {state} Chapter 6 notebooks.")


if __name__ == "__main__":
    main()
