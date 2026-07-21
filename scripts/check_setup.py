#!/usr/bin/env python3
"""Validate the local environment without exposing API-key values."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON = ((3, 12), (3, 15))
REQUIRED_FILES = ("pyproject.toml", "uv.lock", "requirements.txt", ".env.example")
REQUIRED_PACKAGES = (
    ("dspy", "dspy", "3.2.1"),
    ("jupyterlab", "jupyterlab", None),
    ("ipykernel", "ipykernel", None),
    ("python-dotenv", "dotenv", None),
)
API_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "FAL_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY",
    "QDRANT_API_KEY",
)
PLACEHOLDER_VALUES = {
    "api_key_goes_here",
    "your-api-key-here",
    "replace_me",
    "changeme",
}


def configured(name: str) -> bool:
    """Return whether an environment variable contains a non-placeholder value."""
    value = os.getenv(name, "").strip()
    return bool(value) and value.lower() not in PLACEHOLDER_VALUES


def check_python() -> list[str]:
    current = sys.version_info[:2]
    minimum, maximum = SUPPORTED_PYTHON
    if minimum <= current < maximum:
        print(f"[ok] Python {current[0]}.{current[1]} is supported")
        return []
    message = (
        f"Python {current[0]}.{current[1]} is unsupported; "
        "use Python 3.12, 3.13, or 3.14"
    )
    print(f"[error] {message}")
    return [message]


def check_files() -> list[str]:
    errors: list[str] = []
    for relative_path in REQUIRED_FILES:
        if (REPO_ROOT / relative_path).is_file():
            print(f"[ok] Found {relative_path}")
        else:
            message = f"Missing required repository file: {relative_path}"
            print(f"[error] {message}")
            errors.append(message)
    return errors


def check_packages() -> list[str]:
    errors: list[str] = []
    for distribution, module, expected_version in REQUIRED_PACKAGES:
        try:
            module_spec = importlib.util.find_spec(module)
            installed_version = importlib.metadata.version(distribution)
        except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
            module_spec = None

        if module_spec is None:
            message = f"Missing package: {distribution}; run `uv sync --frozen`"
            print(f"[error] {message}")
            errors.append(message)
            continue

        if expected_version and installed_version != expected_version:
            message = (
                f"{distribution} {installed_version} is installed; "
                f"this repository requires {expected_version}"
            )
            print(f"[error] {message}")
            errors.append(message)
        else:
            print(f"[ok] {distribution} {installed_version}")
    return errors


def load_environment(skip_dotenv: bool) -> list[str]:
    if skip_dotenv:
        return []

    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        print("[warning] .env is missing; copy .env.example to .env")
        return []

    try:
        from dotenv import load_dotenv
    except ImportError:
        # The package check reports the actionable installation error.
        return []

    load_dotenv(dotenv_path=dotenv_path, override=False)
    print("[ok] Loaded .env")
    return []


def check_api_keys(require_openai_key: bool) -> list[str]:
    errors: list[str] = []
    for name in API_KEYS:
        if configured(name):
            print(f"[ok] {name} is configured")
        elif name == "OPENAI_API_KEY" and require_openai_key:
            message = f"{name} is missing or still uses a placeholder"
            print(f"[error] {message}")
            errors.append(message)
        elif name == "OPENAI_API_KEY":
            print(
                f"[warning] {name} is not configured; "
                "live introductory examples will fail"
            )
        else:
            print(f"[optional] {name} is not configured")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-openai-key",
        action="store_true",
        help="Fail when OPENAI_API_KEY is missing or still a placeholder.",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not load .env (useful for CI and deterministic diagnostics).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors: list[str] = []
    errors.extend(check_python())
    errors.extend(check_files())
    errors.extend(check_packages())
    errors.extend(load_environment(args.no_dotenv))
    errors.extend(check_api_keys(args.require_openai_key))

    if errors:
        print(f"\nSetup check failed with {len(errors)} error(s).")
        return 1

    print("\nSetup check passed. Start Jupyter with `uv run jupyter lab`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
