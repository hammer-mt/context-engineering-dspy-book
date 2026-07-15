"""Execute all Chapter 6 notebooks in safe inspect mode and save their outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient

from chapter06.build_optimizer_notebooks import CHAPTER_DIR, NOTEBOOKS


REPO_ROOT = CHAPTER_DIR.parent


def execute_notebook(path: Path, *, timeout: int = 120) -> None:
    notebook = nbformat.read(path, as_version=4)
    previous_mode = os.environ.pop("CHAPTER06_RUN", None)
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute(cwd=str(REPO_ROOT))
    finally:
        if previous_mode is not None:
            os.environ["CHAPTER06_RUN"] = previous_mode
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
        print(f"Executed {path.relative_to(REPO_ROOT)} in safe inspect mode")


if __name__ == "__main__":
    main()
