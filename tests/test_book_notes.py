"""Checks that the collected notes and shared data files stay in step with their sources.

BOOK_NOTES.md quotes the "Book vs. notebook" notes from the notebooks, and the
Chapter 3 folder holds a copy of the dataset so that the printed file name
works from that folder. These tests fail when either copy drifts from its
source. They use only the Python standard library.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_book_notes import OUTPUT_PATH, build


REPO_ROOT = Path(__file__).resolve().parents[1]


class BookNotesTest(unittest.TestCase):
    def test_book_notes_page_matches_the_notebooks(self) -> None:
        self.assertTrue(OUTPUT_PATH.exists(), "BOOK_NOTES.md is missing")
        self.assertEqual(
            OUTPUT_PATH.read_text(encoding="utf-8"),
            build(),
            "BOOK_NOTES.md is out of date. Run: uv run python scripts/build_book_notes.py",
        )

    def test_every_linked_notebook_exists(self) -> None:
        for line in build().splitlines():
            if line.startswith("### [`"):
                relative_path = line.split("](", 1)[1].rstrip(")")
                self.assertTrue((REPO_ROOT / relative_path).is_file(), relative_path)


class SharedDataTest(unittest.TestCase):
    def test_chapter3_dataset_copy_matches_the_shared_dataset(self) -> None:
        shared = (REPO_ROOT / "data" / "ai_vs_human.csv").read_bytes()
        chapter_copy = (REPO_ROOT / "chapter03" / "ai_vs_human.csv").read_bytes()
        self.assertEqual(shared, chapter_copy)


if __name__ == "__main__":
    unittest.main()
