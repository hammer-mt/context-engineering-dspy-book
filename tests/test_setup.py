"""Checks that the setup instructions match the Preface of the book.

The Preface ("Software Requirements for This Book") tells readers how to
install the project, which environment variables to set, and how to check the
installation. These tests keep the README, the environment template, and the
setup checker in step with those instructions. They make no network calls.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

# Commands printed in the Preface, by the section that prints them.
PREFACE_COMMANDS = {
    "Installing the Project with uv": (
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "irm https://astral.sh/uv/install.ps1 | iex",
        "git clone https://github.com/hammer-mt/context-engineering-dspy-book.git",
        "cd context-engineering-dspy-book",
        "uv sync --frozen",
        "cp .env.example .env",
        "Copy-Item .env.example .env",
        "uv run python scripts/check_setup.py --require-openai-key",
        "uv run jupyter lab",
        "uv run python chapter08/mcp_server.py",
    ),
    "Installing Without uv": (
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "py -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "python -m pip install -r requirements.txt",
        "python scripts/check_setup.py --require-openai-key",
        "python -m jupyter lab",
    ),
    "Configuring Environment Variables": (
        'export OPENAI_API_KEY="your-api-key-here"',
        "set OPENAI_API_KEY=your-api-key-here",
        '$env:OPENAI_API_KEY="your-api-key-here"',
    ),
    "Deno": (
        "curl -fsSL https://deno.land/install.sh | sh",
        "irm https://deno.land/install.ps1 | iex",
        "deno --version",
    ),
    "Docker and Qdrant": (
        "docker --version",
        "docker run --rm --name qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant",
        "QDRANT_URL=http://127.0.0.1:6333",
        "uv pip install dspy-qdrant qdrant-client fastembed",
    ),
    "Other optional local tools": ("uv run playwright install chromium",),
}
# Variables in the Preface table, "Configuring Environment Variables".
PREFACE_VARIABLES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "FAL_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY",
    "QDRANT_API_KEY",
)


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

    def test_environment_template_lists_the_preface_variables(self) -> None:
        template = (REPO_ROOT / ".env.example").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        for name in (*PREFACE_VARIABLES, "OPENROUTER_API_KEY", "QDRANT_URL"):
            with self.subTest(variable=name):
                self.assertIn(f"{name}=", template)
                self.assertIn(f"`{name}`", readme)

        # Copying the template must not switch on placeholder values for keys
        # a reader has not filled in: only the OpenAI key line is active.
        active = [
            line.split("=", 1)[0]
            for line in template.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(active, ["OPENAI_API_KEY"])

    def test_readme_shows_the_preface_commands(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text()
        code_lines = {
            line.strip()
            for block in re.findall(r"```[a-z]*\n(.*?)```", readme, re.DOTALL)
            for line in block.splitlines()
        }
        for section, commands in PREFACE_COMMANDS.items():
            for command in commands:
                with self.subTest(section=section, command=command):
                    self.assertIn(command, code_lines)

    def test_dependency_files_pin_the_dspy_version_the_book_uses(self) -> None:
        project = (REPO_ROOT / "pyproject.toml").read_text()
        requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
        self.assertIn('"dspy==3.3.0"', project)
        self.assertIn("dspy==3.3.0", requirements)
        self.assertIn('requires-python = ">=3.12,<3.15"', project)

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
