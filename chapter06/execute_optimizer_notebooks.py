"""Execute the Chapter 6 notebooks in their default mode and save the outputs.

The default mode makes no model or API call: paid listings are skipped and each
notebook displays its saved result. Run from the repository root with the
project's Python environment::

    python -m chapter06.execute_optimizer_notebooks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient

from chapter06.build_optimizer_notebooks import CHAPTER_DIR, NOTEBOOKS


REPO_ROOT = CHAPTER_DIR.parent


def _output_text(output: dict) -> str:
    parts: list[str] = []
    for key in ("text", "traceback", "evalue"):
        value = output.get(key, "")
        parts.append("".join(value) if isinstance(value, list) else str(value))
    for value in output.get("data", {}).values():
        parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def _check_outputs_are_portable(notebook: nbformat.NotebookNode, path: Path) -> None:
    """Refuse to save outputs that mention this machine's directories."""

    local_paths = {str(REPO_ROOT), str(Path.home())}
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            text = _output_text(output)
            if any(local_path in text for local_path in local_paths):
                raise RuntimeError(
                    f"{path.name}: cell {index} output contains a local filesystem path"
                )


def execute_notebook(path: Path, *, timeout: int = 120) -> None:
    notebook = nbformat.read(path, as_version=4)
    previous_mode = os.environ.get("CHAPTER06_RUN_LIVE")
    # The kernel inherits this environment. Pin the default mode so that no
    # listing can call a model while outputs are being refreshed.
    os.environ["CHAPTER06_RUN_LIVE"] = "0"
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute(cwd=str(REPO_ROOT))
    finally:
        if previous_mode is None:
            os.environ.pop("CHAPTER06_RUN_LIVE", None)
        else:
            os.environ["CHAPTER06_RUN_LIVE"] = previous_mode
    _check_outputs_are_portable(notebook, path)
    nbformat.write(notebook, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="append", choices=tuple(NOTEBOOKS))
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    filenames = args.notebook or list(NOTEBOOKS)
    for filename in filenames:
        path = CHAPTER_DIR / filename
        execute_notebook(path, timeout=args.timeout)
        print(f"Executed {path.relative_to(REPO_ROOT)} in the default mode (no model calls)")


if __name__ == "__main__":
    main()
