from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class SetupDocumentationTests(unittest.TestCase):
    def test_readme_documents_a_complete_quick_start(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text()
        for expected in (
            "uv sync --frozen",
            "cp .env.example .env",
            "scripts/check_setup.py --require-openai-key",
            "uv run jupyter lab",
            "chapter01/hello-dspy.ipynb",
        ):
            self.assertIn(expected, readme)

    def test_environment_template_matches_notebook_variable_names(self) -> None:
        template = (REPO_ROOT / ".env.example").read_text()
        self.assertIn("GEMINI_API_KEY=", template)
        self.assertNotIn("GOOGLE_API_KEY=", template)

    def test_stale_preface_setup_instructions_are_not_in_readme(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text()
        for stale_text in ("requirements.in", "dspy==3.0.3", "content/chapter_x"):
            self.assertNotIn(stale_text, readme)

    def test_setup_checker_passes_without_optional_api_keys(self) -> None:
        environment = os.environ.copy()
        for name in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "FAL_KEY",
            "SERPER_API_KEY",
            "TAVILY_API_KEY",
            "QDRANT_API_KEY",
        ):
            environment.pop(name, None)

        result = subprocess.run(
            [sys.executable, "scripts/check_setup.py", "--no-dotenv"],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("api_key_goes_here", result.stdout)

    def test_setup_checker_can_require_the_introductory_key(self) -> None:
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/check_setup.py",
                "--no-dotenv",
                "--require-openai-key",
            ],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("OPENAI_API_KEY is missing", result.stdout)


if __name__ == "__main__":
    unittest.main()
