"""Checks that every notebook in the repository opens and reads cleanly.

These tests use only the Python standard library, so they run without the
book's dependencies, API keys, or network access. They protect what a reader
sees on opening a notebook: a valid file, no output left over from someone
else's run, code that Python can parse, and no path that exists only on
another person's machine.
"""

from __future__ import annotations

import ast
import json
import re
import unittest
from collections.abc import Iterator
from functools import cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_FOLDERS = tuple(f"chapter{number:02d}" for number in range(1, 12))
CHAPTER6_GENERATOR = REPO_ROOT / "chapter06" / "build_optimizer_notebooks.py"

# Home directories and per-user temporary folders. A path like these in a
# notebook would point at one person's machine. Placeholder paths such as
# "/path/to/local/photo.png" (Chapter 7, page 180) are not matched.
LOCAL_PATH = re.compile(
    r"(?<![\w.~])/Users/[^/\s\"'`]+/"
    r"|(?<![\w.~])/home/[^/\s\"'`]+/"
    r"|\b[A-Za-z]:\\+Users\\"
    r"|(?<![\w.~])/private/(?:tmp|var)/"
    r"|(?<![\w.~])/var/folders/"
)


def notebook_paths() -> list[Path]:
    """Every notebook in the chapter folders, in a stable order."""
    paths = []
    for folder in CHAPTER_FOLDERS:
        for path in sorted((REPO_ROOT / folder).rglob("*.ipynb")):
            relative = path.relative_to(REPO_ROOT)
            if not any(part.startswith(".") for part in relative.parts):
                paths.append(path)
    return paths


@cache
def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def code_cells(notebook: dict) -> Iterator[tuple[int, dict]]:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            yield index, cell


def python_source(source: str) -> str:
    """Replace IPython ``%`` and ``!`` lines with ``pass`` so Python can parse the cell."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append(line[: len(line) - len(stripped)] + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def strings_in(value: object) -> Iterator[str]:
    """Every string stored anywhere in a notebook: sources, outputs, and metadata."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


@cache
def generated_chapter6_notebooks() -> frozenset[str]:
    """Names of the notebooks that the Chapter 6 generator writes.

    The names are read from the generator's ``NOTEBOOKS`` table without
    importing it, so this module stays free of third-party imports.
    """
    tree = ast.parse(CHAPTER6_GENERATOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "NOTEBOOKS":
            return frozenset(key.value for key in node.value.keys)
    raise AssertionError(f"NOTEBOOKS table not found in {CHAPTER6_GENERATOR.name}")


def keeps_outputs(path: Path) -> bool:
    """The generated Chapter 6 notebooks show the published results as saved output."""
    return (
        path.parent == REPO_ROOT / "chapter06"
        and path.name in generated_chapter6_notebooks()
    )


class NotebookHygieneTest(unittest.TestCase):
    def test_every_chapter_folder_has_a_notebook(self) -> None:
        folders = {path.relative_to(REPO_ROOT).parts[0] for path in notebook_paths()}
        self.assertEqual(folders, set(CHAPTER_FOLDERS))

        generated = generated_chapter6_notebooks()
        self.assertEqual(len(generated), 12)
        for name in sorted(generated):
            with self.subTest(notebook=name):
                self.assertTrue((REPO_ROOT / "chapter06" / name).is_file())

    def test_every_notebook_is_valid_notebook_json(self) -> None:
        for path in notebook_paths():
            with self.subTest(notebook=str(path.relative_to(REPO_ROOT))):
                notebook = load_notebook(path)
                self.assertEqual(notebook["nbformat"], 4)
                self.assertIsInstance(notebook["metadata"], dict)
                self.assertTrue(notebook["cells"])
                for cell in notebook["cells"]:
                    self.assertIn(cell["cell_type"], {"markdown", "code", "raw"})
                    self.assertIsInstance(cell["source"], (str, list))
                # A notebook opens with a markdown cell that names the chapter.
                self.assertEqual(notebook["cells"][0]["cell_type"], "markdown")
                self.assertIn("Chapter", cell_source(notebook["cells"][0]))

    def test_outputs_are_cleared(self) -> None:
        for path in notebook_paths():
            if keeps_outputs(path):
                continue
            notebook = load_notebook(path)
            for index, cell in code_cells(notebook):
                with self.subTest(
                    notebook=str(path.relative_to(REPO_ROOT)), cell=index
                ):
                    self.assertEqual(cell.get("outputs", []), [])
                    self.assertIsNone(cell.get("execution_count"))

    def test_generated_chapter6_notebooks_keep_only_text_output(self) -> None:
        # These notebooks are written and run by the generator in their default
        # mode, which makes no model call, and keep the printed result blocks.
        for path in notebook_paths():
            if not keeps_outputs(path):
                continue
            notebook = load_notebook(path)
            saved_text = []
            for index, cell in code_cells(notebook):
                for output in cell.get("outputs", []):
                    with self.subTest(notebook=path.name, cell=index):
                        self.assertEqual(output.get("output_type"), "stream")
                        self.assertEqual(output.get("name"), "stdout")
                    saved_text.append("".join(output.get("text", [])))
            with self.subTest(notebook=path.name):
                self.assertTrue("".join(saved_text).strip())

    def test_every_code_cell_compiles(self) -> None:
        for path in notebook_paths():
            name = str(path.relative_to(REPO_ROOT))
            for index, cell in code_cells(load_notebook(path)):
                source = cell_source(cell)
                if source.lstrip().startswith("%%"):
                    continue  # A cell magic hands the whole cell to another program.
                with self.subTest(notebook=name, cell=index):
                    try:
                        # Jupyter allows ``await`` at the top level of a cell.
                        compile(
                            python_source(source) or "pass",
                            f"{name} [cell {index}]",
                            "exec",
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                            dont_inherit=True,
                        )
                    except SyntaxError as error:
                        self.fail(f"cell does not compile: {error}")

    def test_notebooks_contain_no_local_absolute_paths(self) -> None:
        for path in notebook_paths():
            found = sorted(
                {
                    match.group(0)
                    for text in strings_in(load_notebook(path))
                    for match in LOCAL_PATH.finditer(text)
                }
            )
            with self.subTest(notebook=str(path.relative_to(REPO_ROOT))):
                self.assertEqual(found, [])

    def test_local_path_pattern_recognizes_machine_specific_paths(self) -> None:
        for text in (
            "/Users/someone/project/data.csv",
            "open('/home/someone/notes.txt')",
            "C:\\Users\\someone\\data.csv",
            "/private/tmp/session/output.json",
            "/var/folders/ab/cd/T/tmp123",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(LOCAL_PATH.search(text))
        for text in (
            'dspy.Image.from_path("/path/to/local/photo.png")',
            "~/.claude/skills/landing-page/SKILL.md",
            "https://example.com/home/page/index.html",
            "the vector store in `/tmp/qdrant`",
        ):
            with self.subTest(text=text):
                self.assertIsNone(LOCAL_PATH.search(text))


if __name__ == "__main__":
    unittest.main()
